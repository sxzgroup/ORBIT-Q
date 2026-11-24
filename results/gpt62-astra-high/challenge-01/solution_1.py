import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _input_mps(state, n):
    tensors = []
    for i in range(n):
        left = [state.bond(i - 1, i)] if i else []
        right = [state.bond(i, i + 1)] if i + 1 < n else []
        data = state[i].transpose(*left, state.site_ind(i), *right).data
        tensors.append(np.asarray(data, dtype=np.complex128).reshape(
            state.bond_size(i - 1, i) if left else 1, 2, -1))
    return tc.MPSCircuit(n, tensors=tensors)


def _pair_gate(a, b, coupling):
    c = tc.Circuit(2)
    for site, angles in enumerate((a, b)):
        c.rz(site, theta=angles[0])
        c.ry(site, theta=angles[1])
        c.rz(site, theta=angles[2])
    c.rxx(0, 1, theta=2 * coupling[0])
    c.ryy(0, 1, theta=2 * coupling[1])
    c.rzz(0, 1, theta=2 * coupling[2])
    return c.matrix()


def _cone(mps, n, layers, parity, operator):
    bonds = np.arange(parity, n - 1, 2)
    lo, hi = parity, parity + 1
    rows = []
    for layer in reversed(range(layers)):
        lo -= (lo - layer) % 2
        hi += (hi - layer) % 2 == 0
        rows.append((layer, np.arange(lo, hi, 2)))
    width = hi - lo + 1
    origins = bonds - parity + lo
    chi = max(t.shape[2] for t in mps.get_tensors())
    vacuum = np.array([1, 0]).reshape(1, 2, 1)
    blocks = []
    for origin in origins:
        # In mixed canonical form the exterior environments are identities.
        mps.position(max(0, origin))
        tensors = mps.get_tensors()
        block = []
        for site in range(origin, origin + width):
            t = tensors[site] if 0 <= site < n else vacuum
            block.append(np.pad(t, ((0, chi-t.shape[0]), (0, 0),
                                    (0, chi-t.shape[2]))))
        blocks.append(block)
    blocks = jnp.asarray(np.array(blocks))
    boundary = jnp.eye(chi)
    locations = [jnp.asarray(bonds[:, None] - parity + row)
                 for _, row in reversed(rows)]

    def measure(matrices, block):
        # The two extra sites purify the virtual boundary indices.
        inputs = [boundary.reshape(1, chi, chi), *block,
                  boundary.reshape(chi, chi, 1)]
        c = tc.Circuit(width + 2, tensors=inputs)
        for (layer, row), gates in zip(reversed(rows), matrices):
            for start, gate in zip(row, gates):
                i = int(start - lo + 1)
                c.any(i, i + 1, unitary=gate)
        i = parity - lo + 1
        return c.expectation((operator, [i, i + 1]), reuse=False).real

    def sum_terms(gates):
        matrices = [gates[layer, index + layers + 1]
                    for layer, index in enumerate(locations)]
        return jax.vmap(measure)(matrices, blocks).sum()
    return sum_terms


def run_solution(config):
    n, layers = config['n_qubits'], config['n_layers']
    tc.set_backend('numpy')
    tc.set_dtype('complex128')
    mps = _input_mps(config['dmrg_state'], n)
    tc.set_backend('jax')
    tc.set_dtype('complex128')
    tc.set_contractor('omeco-8-24')
    x, z = tc.gates.x().tensor, tc.gates.z().tensor
    zz = -jnp.kron(z, z)
    field = -config['field'] * (jnp.kron(x, jnp.eye(2)) + jnp.kron(jnp.eye(2), x))
    # On an even chain the even bonds partition all single-site field terms.
    sums = [_cone(mps, n, layers, p, zz + field if p == 0 else zz)
            for p in range(2)]
    margin = layers + 1

    def energy(params):
        angles, couplings = params
        angles = jnp.pad(angles, ((0, 0), (margin, margin), (0, 0)))
        couplings = jnp.pad(couplings, ((0, 0), (margin, margin), (0, 0)))
        gates = jax.vmap(jax.vmap(_pair_gate))(
            angles[:, :-1], angles[:, 1:], couplings)
        return sum(f(gates) for f in sums)

    rng = np.random.default_rng(123)
    params = tuple(jnp.asarray(rng.normal(0, 0.001, (layers, sites, 3)))
                   for sites in (n, n - 1))
    adam = optax.adam(config['learning_rate'])

    def update(carry, _):
        p, state = carry
        value, grad = jax.value_and_grad(energy)(p)
        delta, state = adam.update(grad, state)
        return (optax.apply_updates(p, delta), state), value

    _, history = jax.lax.scan(update, (params, adam.init(params)), None,
                              length=config['max_steps'])
    return {'energy_history': np.asarray(history)}
