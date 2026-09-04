"""TensorCircuit VQE with three prescribed Renyi-2 checkpoints."""

import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    n = int(config["n_qubits"])
    blocks = int(config["n_layers"]) // 2
    if int(config["n_layers"]) % 2:
        raise ValueError("Each block requires an even and an odd sublayer.")
    targets = np.asarray(config["target_entropies"], dtype=np.float32)
    if targets.shape != (blocks,):
        raise ValueError("One entropy target is required per block.")

    # Construct the open-chain operator using TensorCircuit's Pauli API.
    tc.set_backend("numpy")
    tc.set_dtype("complex64")
    strings, weights = [], []
    for i in range(n - 1):
        for pauli, weight in [(1, 1.0), (2, 1.0),
                              (3, float(config["zz_anisotropy"]))]:
            term = [0] * n
            term[i] = term[i + 1] = pauli
            strings.append(term)
            weights.append(weight)
    for i in range(n):
        term = [0] * n
        term[i] = 3
        strings.append(term)
        weights.append(float(config["staggered_field"]) * (-1) ** i)
    sparse_h = tc.quantum.PauliStringSum2COO(strings, weights, numpy=True)
    backend = tc.set_backend("jax")
    tc.set_contractor("greedy")
    hamiltonian = backend.coo_sparse_matrix_from_numpy(sparse_h)
    target = jnp.asarray(targets)
    initial = tc.Circuit(n)
    for i in range(1, n, 2):
        initial.x(i)
    initial_state = initial.state()
    block_size = 4 * n + 3 * (n - 1)
    parameters = jnp.asarray(np.random.default_rng(42).normal(
        0.0, 0.02, (blocks, block_size)).astype(np.float32))

    def block(state, angles):
        offset = 0
        for parity in (0, 1):
            circuit = tc.Circuit(n, inputs=state)
            for q in range(n):
                circuit.ry(q, theta=angles[offset])
                circuit.rz(q, theta=angles[offset + 1])
                offset += 2
            for q in range(parity, n - 1, 2):
                # XX, YY and ZZ commute. TC rotations include a factor 1/2.
                circuit.rxx(q, q + 1, theta=2 * angles[offset])
                circuit.ryy(q, q + 1, theta=2 * angles[offset + 1])
                circuit.rzz(q, q + 1, theta=2 * angles[offset + 2])
                offset += 3
            state = circuit.state()
        rho = tc.quantum.reduced_density_matrix(
            state, subsystem_to_keep=list(range(int(config["subsystem_size"]))))
        entropy = tc.quantum.renyi_entropy(rho, k=2)
        return state, entropy

    def objective(angles):
        state, entropies = jax.lax.scan(block, initial_state, angles)
        circuit = tc.Circuit(n, inputs=state)
        energy = tc.templates.measurements.sparse_expectation(circuit, hamiltonian)
        density = energy / n
        mse = jnp.mean(jnp.square(entropies - target))
        loss = density + float(config["entropy_weight"]) * mse
        return loss, (density, mse, entropies)

    optimizer = optax.adam(float(config["learning_rate"]))
    value_and_grad = jax.value_and_grad(objective, has_aux=True)

    def update(carry, unused):
        angles, opt_state = carry
        (loss, (density, mse, entropies)), gradient = value_and_grad(angles)
        updates, opt_state = optimizer.update(gradient, opt_state, angles)
        angles = optax.apply_updates(angles, updates)
        return (angles, opt_state), (density, loss, mse, entropies)

    @jax.jit
    def train(angles):
        return jax.lax.scan(update, (angles, optimizer.init(angles)), None,
                            length=int(config["max_steps"]))[1]

    history = train(parameters)
    return dict(zip(("energy_density_history", "loss_history",
                     "entropy_mse_history", "entropy_history"),
                    (np.asarray(values) for values in history)))
