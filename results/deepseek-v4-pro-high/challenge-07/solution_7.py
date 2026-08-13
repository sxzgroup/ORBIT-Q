import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    backend = tc.backend

    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    seed = int(config["seed"])
    field = float(config["transverse_field"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    steps = int(config["max_steps"])

    zz_structures = [
        [3 if (j == i or j == i + 1) else 0 for j in range(n_data)]
        for i in range(n_data - 1)
    ]
    xx_structures = [[1 if j == i else 0 for j in range(n_data)] for i in range(n_data)]
    mvp = tc.quantum.PauliStringSum2MVP(
        zz_structures + xx_structures,
        [-1.0] * (n_data - 1) + [-field] * n_data,
    )

    def one_energy(params, status):
        c = tc.Circuit(n_qubits)
        last_bits = []
        for layer in range(n_layers):
            bits = []
            for i in range(n_data):
                c.ry(i, theta=params[layer, 0, i])
            for i in range(n_data):
                c.ry(n_data + i, theta=params[layer, 1, i])
            for i in range(n_data):
                c.rzz(n_data + i, i, theta=params[layer, 2, i])
            for i in range(n_anc - 1):
                c.cnot(n_data + i, n_data + i + 1)
            for i in range(n_data):
                r = c.cond_measurement(n_data + i, status=status[layer, i])
                bits.append(r)
                c.conditional_gate(
                    r,
                    [
                        tc.gates.rzz_gate(theta=params[layer, 3, i]),
                        tc.gates.rzz_gate(theta=params[layer, 4, i]),
                    ],
                    n_data + i,
                    i,
                )
            for i in range(n_data - 1):
                c.cnot(i, i + 1)
            for i in range(n_data):
                c.rz(i, theta=params[layer, 5, i])
            last_bits = bits

        state = backend.reshape(c.state(), [2] * n_qubits)
        data_state = tc.quantum.reduced_wavefunction(
            state, subsystem_to_keep=list(range(n_data)), measure=last_bits
        )
        data_state = backend.reshape(data_state, [-1])
        data_state = data_state / backend.norm(data_state)
        hpsi = mvp(data_state)
        return backend.real(backend.sum(backend.conj(data_state) * hpsi))

    def objective(params, statuses):
        return jnp.mean(jax.vmap(one_energy, in_axes=(None, 0))(params, statuses))

    rng = np.random.default_rng(seed)
    params = jnp.asarray(
        rng.normal(0.0, scale, size=(n_layers, 6, n_data)), dtype=jnp.float32
    )
    statuses = jnp.asarray(
        rng.uniform(size=(n_traj, n_layers, n_data)), dtype=jnp.float32
    )

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    energy_history = []
    for _ in range(steps):
        value, grads = value_and_grad(params, statuses)
        energy_history.append(np.asarray(value, dtype=np.float64).item())
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    batch_energy = jax.jit(jax.vmap(one_energy, in_axes=(None, 0)))
    final_trajectory_energies = np.asarray(batch_energy(params, statuses), dtype=np.float64)

    return {
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "final_trajectory_energies": final_trajectory_energies,
    }
