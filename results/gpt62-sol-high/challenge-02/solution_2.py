import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _xxz_hamiltonian(n_qubits, anisotropy, field):
    terms, weights = [], []
    for i in range(n_qubits - 1):
        for pauli, coupling in enumerate((1.0, 1.0, anisotropy), start=1):
            term = [0] * n_qubits
            term[i] = term[i + 1] = pauli
            terms.append(term)
            weights.append(coupling)
    for i in range(n_qubits):
        term = [0] * n_qubits
        term[i] = 3
        terms.append(term)
        weights.append(field * (-1) ** i)
    with tc.runtime_backend("numpy"):
        return tc.quantum.PauliStringSum2COO(terms, weights, numpy=True)


def run_solution(config):
    n_qubits = config["n_qubits"]
    n_blocks = config["n_layers"] // 2
    left_half = tuple(range(config["subsystem_size"]))
    target = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    entropy_weight = config["entropy_weight"]
    bonds = tuple(tuple(range(parity, n_qubits - 1, 2)) for parity in (0, 1))
    n_rotations = 2 * n_qubits
    even_size = n_rotations + 3 * len(bonds[0])
    block_size = even_size + n_rotations + 3 * len(bonds[1])

    with tc.runtime_dtype("complex64"):
        sparse_h = _xxz_hamiltonian(
            n_qubits, config["zz_anisotropy"], config["staggered_field"]
        )
        with tc.runtime_backend("jax") as backend, tc.runtime_contractor("greedy"):
            hamiltonian = backend.coo_sparse_matrix_from_numpy(sparse_h)
            initial_circuit = tc.Circuit(n_qubits)
            for q in range(1, n_qubits, 2):
                initial_circuit.x(q)
            initial_state = initial_circuit.state()

            def sublayer(circuit, params, bond_starts):
                rotations = params[:n_rotations].reshape(n_qubits, 2)
                interactions = params[n_rotations:].reshape(-1, 3)
                for q, (y, z) in enumerate(rotations):
                    circuit.ry(q, theta=y)
                    circuit.rz(q, theta=z)
                # XX, YY and ZZ commute on each bond. The native RPP gates use half angles.
                for q, (xx, yy, zz) in zip(bond_starts, interactions):
                    circuit.rxx(q, q + 1, theta=2 * xx)
                    circuit.ryy(q, q + 1, theta=2 * yy)
                    circuit.rzz(q, q + 1, theta=2 * zz)

            def block(state, params):
                circuit = tc.Circuit(n_qubits, inputs=state)
                sublayer(circuit, params[:even_size], bonds[0])
                sublayer(circuit, params[even_size:], bonds[1])
                state = circuit.state()
                rho = tc.quantum.reduced_density_matrix(state, subsystem_to_keep=left_half)
                return state, tc.quantum.renyi_entropy(rho, k=2)

            def objective(params):
                state, entropies = jax.lax.scan(block, initial_state, params)
                final_circuit = tc.Circuit(n_qubits, inputs=state)
                energy = tc.templates.measurements.sparse_expectation(final_circuit, hamiltonian)
                energy_density = energy / n_qubits
                mse = jnp.mean(jnp.square(entropies - target))
                return energy_density + entropy_weight * mse, (energy_density, mse, entropies)

            adam = optax.adam(config["learning_rate"])
            value_and_grad = jax.value_and_grad(objective, has_aux=True)

            def step(carry, _):
                params, opt_state = carry
                (loss, (energy, mse, entropies)), grads = value_and_grad(params)
                updates, opt_state = adam.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (energy, loss, mse, entropies)

            params = np.random.default_rng(0).normal(0, 0.02, (n_blocks, block_size))
            params = jnp.asarray(params, dtype=jnp.float32)
            _, histories = jax.lax.scan(
                step, (params, adam.init(params)), xs=None, length=config["max_steps"]
            )
            keys = ("energy_density_history", "loss_history", "entropy_mse_history", "entropy_history")
            return dict(zip(keys, map(np.asarray, histories)))
