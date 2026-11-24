from collections import defaultdict

import cotengra as ctg
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _mps_tensors(state, n):
    bonds = [state.bond(i, i + 1) for i in range(n - 1)]
    tensors = []
    for i in range(n):
        inds = ([bonds[i - 1]] if i else []) + [state.site_ind(i)]
        if i < n - 1:
            inds.append(bonds[i])
        tensor = np.asarray(state[i].transpose(*inds).data)
        left = tensor.shape[0] if i else 1
        right = tensor.shape[-1] if i < n - 1 else 1
        tensors.append(tensor.reshape(left, 2, right))
    return tensors


def _light_cones(n, layers):
    # Keep precisely the bricks that overlap the backward support at each layer.
    # Padding beyond the chain gives boundary terms the same batched geometry.
    groups = defaultdict(list)
    for width in (1, 2):
        for site in range(n + 1 - width):
            left, right = site, site + width
            bonds = []
            for layer in reversed(range(layers)):
                left -= (left - layer) % 2
                right += (right - layer) % 2
                bonds.append(range(left, right, 2))
            schedule = tuple(tuple(i - left for i in row) for row in reversed(bonds))
            groups[right - left, site - left, width, schedule].append(left)
    return groups


def _cone_inputs(state, n, groups):
    tensors = _mps_tensors(state, n)
    chi = max(t.shape[-1] for t in tensors)
    with tc.runtime_backend("numpy"):
        mps = tc.MPSCircuit(n, tensors=tensors)
        gauges = {}
        centers = sorted({max(0, i) for starts in groups.values() for i in starts})
        for first in centers:
            # The exterior of a window in this gauge traces to identity on its bonds.
            mps.position(first)
            gauges[first] = [
                np.pad(t, ((0, chi - t.shape[0]), (0, 0), (0, chi - t.shape[2])))
                for t in mps.get_tensors()
            ]
    dummy = np.zeros((chi, 2, chi), dtype=np.complex128)
    dummy[0, 0, 0] = 1
    inputs = {}
    for key, starts in groups.items():
        k = key[0]
        windows = [
            [gauges[max(0, a)][i] if 0 <= i < n else dummy for i in range(a, a + k)]
            for a in starts
        ]
        inputs[key] = tuple(
            jnp.asarray(np.stack(ts), dtype=jnp.complex128) for ts in zip(*windows)
        )
    return inputs, chi


def _brick(left, right, angles):
    c = tc.Circuit(2)
    for i, (z1, y, z2) in enumerate((left, right)):
        c.rz(i, theta=z1)
        c.ry(i, theta=y)
        c.rz(i, theta=z2)
    # The three Pauli products commute. TensorCircuit defines RPP as exp(-i θ PP/2).
    c.rxx(0, 1, theta=2 * angles[0])
    c.ryy(0, 1, theta=2 * angles[1])
    c.rzz(0, 1, theta=2 * angles[2])
    return c.matrix()


def _expectation(tensors, bricks, key, left_edge, right_edge):
    n, site, width, schedule = key
    # The two untouched boundary legs purify the input window in its MPS gauge.
    c = tc.Circuit(n + 2, tensors=(left_edge, *tensors, right_edge))
    for row, gates in zip(schedule, bricks):
        for i, gate in zip(row, gates):
            c.any(i + 1, i + 2, unitary=gate)
    ops = ((tc.gates.x(), [site + 1]),) if width == 1 else (
        (tc.gates.z(), [site + 1]), (tc.gates.z(), [site + 2])
    )
    return c.expectation(*ops, reuse=False).real


def _optimize(config):
    n = config["n_qubits"]
    layers = config["n_layers"]
    groups = _light_cones(n, layers)
    inputs, chi = _cone_inputs(config["dmrg_state"], n, groups)
    eye = jnp.eye(chi, dtype=jnp.complex128)
    optimizer = optax.adam(config["learning_rate"])
    rng = np.random.default_rng(7)
    pair_slots = [np.arange(layer % 2 + 1, n, 2) for layer in range(layers)]
    params = (
        jnp.asarray(rng.normal(0, 0.001, (layers, n, 3))),
        tuple(jnp.asarray(rng.normal(0, 0.001, (len(sites), 3))) for sites in pair_slots),
    )
    gate_indices = {}
    for key, starts in groups.items():
        for layer, row in enumerate(key[3]):
            sites = np.asarray(starts)[:, None] + np.asarray(row)[None, :]
            gate_indices[key, layer] = np.where(
                (sites >= -1) & (sites <= n - 1), sites + 1, n + 1
            )

    def energy(params):
        local, pair_angles = params
        local = jnp.pad(local, ((0, 0), (1, 1), (0, 0)))
        # A brick crossing an open boundary contains only the real site's rotation.
        pair = jnp.zeros((layers, n + 1, 3), dtype=local.dtype)
        for layer, (slots, angles) in enumerate(zip(pair_slots, pair_angles)):
            pair = pair.at[layer, slots].set(angles)
        bricks = jax.vmap(jax.vmap(_brick))(local[:, :-1], local[:, 1:], pair)
        idle = jnp.broadcast_to(jnp.eye(4, dtype=bricks.dtype), (layers, 1, 4, 4))
        bricks = jnp.concatenate((bricks, idle), axis=1)
        total = 0.0
        for key in groups:
            relevant = tuple(bricks[l, gate_indices[key, l]] for l in range(layers))

            def measure(tensors, gates):
                return _expectation(tensors, gates, key, eye[None, :, :], eye[:, :, None])

            with tc.runtime_contractor(
                "custom",
                optimizer=ctg.HyperOptimizer(
                    methods=["greedy", "random-greedy"], max_repeats=64, parallel=False
                ),
                use_primitives=True,
            ):
                values = jax.vmap(measure)(inputs[key], relevant)
            total -= (config["field"] if key[2] == 1 else 1.0) * jnp.sum(values)
        return total

    def step(carry, _):
        params, opt_state = carry
        value, grad = jax.value_and_grad(energy)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), value

    _, history = jax.lax.scan(
        step, (params, optimizer.init(params)), None, length=config["max_steps"]
    )
    return np.asarray(history)


def run_solution(config):
    """Refine the supplied DMRG state and record the loss before each Adam update."""
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        return {"energy_history": _optimize(config)}
