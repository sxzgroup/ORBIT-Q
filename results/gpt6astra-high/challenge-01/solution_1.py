"""Exact TensorCircuit light-cone contractions of a supplied MPS and circuit."""
import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
import cotengra


def run_solution(config):
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor(
        "custom",
        optimizer=cotengra.ReusableHyperOptimizer(
            methods=["greedy"], max_repeats=8, parallel=False, progbar=False
        ),
        preprocessing=True,
    )
    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    steps = int(config["max_steps"])
    state = config["dmrg_state"]
    raw = []
    # Interoperability only: order input MPS axes as left, physical, right.
    for i in range(n):
        inds = ([state.bond(i - 1, i)] if i else [])
        inds += [state.site_ind(i)]
        inds += [state.bond(i, i + 1)] if i < n - 1 else []
        left = state.bond_size(i - 1, i) if i else 1
        right = state.bond_size(i, i + 1) if i < n - 1 else 1
        raw.append(np.asarray(state[i].transpose(*inds).data).reshape(left, 2, right))
    chi = max(max(a.shape[0], a.shape[2]) for a in raw)
    raw = [np.pad(a, ((0, chi-a.shape[0]), (0, 0), (0, chi-a.shape[2]))) for a in raw]

    # Translate each observable to site n, with |0> padding around the entire
    # unchanged input MPS. Equal shapes allow batched exact contractions.
    width = 2 * n
    groups = []
    x = np.asarray(tc.gates.x().tensor)
    z = np.asarray(tc.gates.z().tensor)
    identity = np.eye(2)
    for parity in range(2):
        states, maps, operators = [], [], []
        for i in range(parity, n, 2):
            for zz in (False, True):
                if zz and i == n - 1:
                    continue
                start = n - i
                a = np.zeros((width, chi, 2, chi), dtype=np.complex128)
                a[:, 0, 0, 0] = 1
                a[start:start+n] = raw
                states.append(a)
                sites = np.arange(width) - start
                maps.append(np.where((sites >= 0) & (sites < n), sites, n))
                operators.append(-np.kron(z, z) if zz else -config["field"] * np.kron(x, identity))
        stacked = np.stack(states)
        tensors = tuple(
            jnp.asarray(stacked[:, j, :1 if j == 0 else chi, :, :1 if j == width-1 else chi])
            for j in range(width)
        )
        groups.append((tensors, jnp.asarray(maps), jnp.asarray(operators)))

    def rotation(p):
        c = tc.Circuit(1)
        c.rz(0, theta=p[0])
        c.ry(0, theta=p[1])
        c.rz(0, theta=p[2])
        return c.matrix()

    def interaction(p):
        c = tc.Circuit(2)
        # The commuting Pauli products give the requested exponential exactly.
        c.rxx(0, 1, theta=2*p[0])
        c.ryy(0, 1, theta=2*p[1])
        c.rzz(0, 1, theta=2*p[2])
        return c.matrix()

    def local_expectation(u, v, tensors, indices, operator, parity):
        c = tc.Circuit(width, tensors=tensors)
        for layer in range(layers):
            for i in range(width):
                c.any(i, unitary=u[layer, indices[i]])
            for i in range((layer+n-parity) % 2, width-1, 2):
                c.any(i, i+1, unitary=v[layer, indices[i]])
        return jnp.real(c.expectation(
            (tc.gates.any(operator), [n, n+1]), enable_lightcone=True
        ))

    def energy(p):
        u = jax.vmap(jax.vmap(rotation))(p[:, :, :3])
        v = jax.vmap(jax.vmap(interaction))(p[:, :, 3:])
        # Padding sites and the nonexistent bond beyond site n-1 are identity.
        v = v.at[:, -1].set(jnp.eye(4))
        u = jnp.concatenate((u, jnp.broadcast_to(jnp.eye(2), (layers, 1, 2, 2))), axis=1)
        v = jnp.concatenate((v, jnp.broadcast_to(jnp.eye(4), (layers, 1, 4, 4))), axis=1)
        result = 0.0
        for parity, (tensors, indices, operators) in enumerate(groups):
            evaluate = lambda t, ix, op: local_expectation(u, v, t, ix, op, parity)
            result += jnp.sum(jax.vmap(evaluate)(tensors, indices, operators))
        return result

    value_and_grad = jax.jit(jax.value_and_grad(energy))
    adam = optax.adam(float(config["learning_rate"]))
    parameters = jnp.asarray(np.random.default_rng(0).normal(size=(layers, n, 6)) * 1e-3)

    def update(carry, _):
        p, optimizer_state = carry
        value, gradient = value_and_grad(p)
        updates, optimizer_state = adam.update(gradient, optimizer_state, p)
        return (optax.apply_updates(p, updates), optimizer_state), value

    @jax.jit
    def optimize(p):
        return jax.lax.scan(update, (p, adam.init(p)), None, length=steps)[1]

    return {"energy_history": np.asarray(optimize(parameters), dtype=np.float64)}
