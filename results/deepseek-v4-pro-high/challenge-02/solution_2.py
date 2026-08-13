import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")

    n = config["n_qubits"]
    sub = config["subsystem_size"]
    jz = config["zz_anisotropy"]
    field = config["staggered_field"]
    steps = config["max_steps"]
    lr = config["learning_rate"]
    weight = config["entropy_weight"]
    target = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    target_np = np.asarray(config["target_entropies"], dtype=np.float32)
    block_params = 81

    pauli_strings = []
    pauli_weights = []
    for i in range(n - 1):
        for pauli, coeff in ((1, 1.0), (2, 1.0), (3, jz)):
            string = [0] * n
            string[i] = pauli
            string[i + 1] = pauli
            pauli_strings.append(string)
            pauli_weights.append(coeff)
    for i in range(n):
        string = [0] * n
        string[i] = 3
        pauli_strings.append(string)
        pauli_weights.append(field * ((-1) ** i))

    hamiltonian = tc.quantum.PauliStringSum2COO(pauli_strings, pauli_weights)
    even_bonds = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    odd_bonds = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    eye4 = jnp.eye(4, dtype=jnp.complex64)
    xx = jnp.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
        dtype=jnp.complex64,
    )
    yy = jnp.array(
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
        dtype=jnp.complex64,
    )
    zz = jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.complex64))

    def rotation(theta_y, theta_z):
        cy = jnp.cos(theta_y / 2)
        sy = jnp.sin(theta_y / 2)
        ry = jnp.array([[cy, -sy], [sy, cy]], dtype=jnp.complex64)
        phase = jnp.exp(-0.5j * theta_z)
        rz = jnp.diag(jnp.array([phase, jnp.conj(phase)], dtype=jnp.complex64))
        return rz @ ry

    def interaction(theta_x, theta_y, theta_z):
        ex = jnp.cos(theta_x) * eye4 - 1j * jnp.sin(theta_x) * xx
        ey = jnp.cos(theta_y) * eye4 - 1j * jnp.sin(theta_y) * yy
        ez = jnp.cos(theta_z) * eye4 - 1j * jnp.sin(theta_z) * zz
        return ex @ ey @ ez

    def apply_block(circuit, params):
        for q in range(n):
            circuit.any(q, unitary=rotation(params[2 * q], params[2 * q + 1]))
        for k, (i, j) in enumerate(even_bonds):
            circuit.any(
                i,
                j,
                unitary=interaction(
                    params[24 + 3 * k],
                    params[24 + 3 * k + 1],
                    params[24 + 3 * k + 2],
                ),
            )
        for q in range(n):
            circuit.any(
                q, unitary=rotation(params[42 + 2 * q], params[42 + 2 * q + 1])
            )
        for k, (i, j) in enumerate(odd_bonds):
            circuit.any(
                i,
                j,
                unitary=interaction(
                    params[66 + 3 * k],
                    params[66 + 3 * k + 1],
                    params[66 + 3 * k + 2],
                ),
            )

    def entropy_of(state):
        rho = tc.quantum.reduced_density_matrix(
            state, subsystem_to_keep=list(range(sub))
        )
        return tc.quantum.renyi_entropy(rho, 2)

    def objective(params):
        circuit = tc.Circuit(n)
        for q in range(1, n, 2):
            circuit.x(q)

        apply_block(circuit, params[0:block_params])
        state1 = circuit.state()
        apply_block(circuit, params[block_params : 2 * block_params])
        state2 = circuit.state()
        apply_block(circuit, params[2 * block_params : 3 * block_params])
        state3 = circuit.state()

        entropies = jnp.stack(
            [entropy_of(state1), entropy_of(state2), entropy_of(state3)]
        )
        mse = jnp.mean((entropies - target) ** 2)
        energy_density = jnp.real(jnp.vdot(state3, hamiltonian @ state3)) / n
        return energy_density + weight * mse, (energy_density, mse, entropies)

    params = jnp.asarray(
        np.random.default_rng(12345).normal(0.0, 0.02, size=3 * block_params).astype(
            np.float32
        )
    )

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    energy_history = []
    loss_history = []
    mse_history = []
    entropy_history = []

    for _ in range(steps):
        ((_, (energy_density, _, entropies)), grads) = value_and_grad(params)
        entropies_np = np.asarray(entropies)
        energy_density_np = float(energy_density)
        mse_np = float(np.mean((entropies_np - target_np) ** 2))
        energy_history.append(energy_density_np)
        loss_history.append(energy_density_np + weight * mse_np)
        mse_history.append(mse_np)
        entropy_history.append(entropies_np)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return {
        "energy_density_history": np.asarray(energy_history),
        "loss_history": np.asarray(loss_history),
        "entropy_mse_history": np.asarray(mse_history),
        "entropy_history": np.stack(entropy_history),
    }
