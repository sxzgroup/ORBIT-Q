import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _hamiltonian(n, anisotropy, field):
    strings, weights = [], []
    for q in range(n):
        term = [0] * n
        term[q] = 3
        strings.append(term)
        weights.append(field * (-1) ** q)
        if q + 1 < n:
            for pauli, coupling in enumerate((1.0, 1.0, anisotropy), start=1):
                term = [0] * n
                term[q:q + 2] = [pauli, pauli]
                strings.append(term)
                weights.append(coupling)
    matrix = tc.quantum.PauliStringSum2COO(strings, weight=weights, numpy=True)
    # All terms are real, including Y tensor Y.
    matrix = matrix.real.tocoo()
    matrix.eliminate_zeros()
    return tc.backend.coo_sparse_matrix_from_numpy(matrix.astype(np.complex64))


def _rotate(circuit, q, angles):
    circuit.ry(q, theta=angles[0])
    circuit.rz(q, theta=angles[1])


def _bond_matrix(rotations, interaction):
    local = tc.Circuit(2)
    _rotate(local, 0, rotations[0])
    _rotate(local, 1, rotations[1])
    # The three Pauli products commute. TensorCircuit rotations use half angles.
    local.rxx(0, 1, theta=2 * interaction[0])
    local.ryy(0, 1, theta=2 * interaction[1])
    local.rzz(0, 1, theta=2 * interaction[2])
    return local.matrix()


def _sublayer(state, rotations, interactions, parity):
    n = rotations.shape[0]
    bonds = range(parity, n - 1, 2)
    stop = parity + 2 * len(bonds)
    pairs = rotations[parity:stop].reshape(len(bonds), 2, 2)
    matrices = jax.vmap(_bond_matrix)(pairs, interactions)
    circuit = tc.Circuit(n, inputs=state)
    for i, q in enumerate(bonds):
        circuit.unitary(q, q + 1, unitary=matrices[i])
    for q in (*range(parity), *range(stop, n)):
        _rotate(circuit, q, rotations[q])
    return circuit.state()


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    n = config["n_qubits"]
    blocks = config["n_layers"] // 2
    left = list(range(config["subsystem_size"]))
    targets = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    h = _hamiltonian(n, config["zz_anisotropy"], config["staggered_field"])

    preparation = tc.Circuit(n)
    for q in range(1, n, 2):
        preparation.x(q)
    initial_state = preparation.state()

    def block(state, params):
        rotations, even, odd = params
        state = _sublayer(state, rotations[0], even, 0)
        state = _sublayer(state, rotations[1], odd, 1)
        rho = tc.quantum.reduced_density_matrix(state, subsystem_to_keep=left)
        return state, tc.quantum.renyi_entropy(rho, k=2)

    def objective(params):
        state, entropies = jax.lax.scan(block, initial_state, params)
        final = tc.Circuit(n, inputs=state)
        energy = tc.templates.measurements.sparse_expectation(final, h) / n
        mse = jnp.mean(jnp.square(entropies - targets))
        loss = energy + config["entropy_weight"] * mse
        return loss, (energy, mse, entropies)

    adam = optax.adam(config["learning_rate"])

    def update(carry, _):
        params, moments = carry
        (loss, (energy, mse, entropies)), grad = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        updates, moments = adam.update(grad, moments, params)
        return (optax.apply_updates(params, updates), moments), (energy, loss, mse, entropies)

    rng = np.random.default_rng(0)
    shapes = [(blocks, 2, n, 2), (blocks, n // 2, 3), (blocks, (n - 1) // 2, 3)]
    params = tuple(jnp.asarray(rng.normal(0, 0.02, s), dtype=jnp.float32) for s in shapes)
    _, records = jax.lax.scan(update, (params, adam.init(params)), None, config["max_steps"])
    keys = ("energy_density_history", "loss_history", "entropy_mse_history", "entropy_history")
    return dict(zip(keys, map(np.asarray, records)))
