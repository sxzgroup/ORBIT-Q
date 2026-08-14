import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _pauli_terms(n_qubits, kind):
    structs, weights = [], []
    if kind in ("xy", "target"):
        for i in range(n_qubits - 1):
            xx = [0] * n_qubits
            yy = [0] * n_qubits
            xx[i] = xx[i + 1] = 1
            yy[i] = yy[i + 1] = 2
            structs.extend([xx, yy])
            coeff = 0.7 if kind == "target" else 1.0
            weights.extend([coeff, coeff])
    if kind == "target":
        for i in range(n_qubits - 1):
            zz = [0] * n_qubits
            zz[i] = zz[i + 1] = 3
            structs.append(zz)
            weights.append(1.1)
    if kind in ("z", "target"):
        for i in range(n_qubits):
            z = [0] * n_qubits
            z[i] = 3
            structs.append(z)
            weights.append((0.25 if kind == "target" else 1.0) * ((-1.0) ** i))
    return structs, weights


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    backend = tc.backend

    n_qubits = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    h_xy = tc.quantum.PauliStringSum2COO(*_pauli_terms(n_qubits, "xy"), numpy=True).tocsr()
    h_z = tc.quantum.PauliStringSum2COO(*_pauli_terms(n_qubits, "z"), numpy=True).tocsr()
    h_target = tc.quantum.PauliStringSum2COO(*_pauli_terms(n_qubits, "target"), numpy=True).tocoo()
    h_xy = backend.coo_sparse_matrix_from_numpy(h_xy.tocoo())
    h_z_diag = backend.convert_to_tensor(np.asarray(h_z.diagonal(), dtype=np.complex64))
    h_target = backend.coo_sparse_matrix_from_numpy(h_target)

    init = tc.Circuit(n_qubits)
    for q in range(1, n_qubits, 2):
        init.x(q)
    psi0 = init.state()

    params = (
        jnp.zeros((n_blocks,), dtype=jnp.float32),
        jnp.full((n_blocks,), 0.1, dtype=jnp.float32),
        jnp.full((n_blocks,), 0.1, dtype=jnp.float32),
        jnp.asarray(
            np.random.default_rng(0).normal(0.0, 0.1, size=(n_blocks, n_qubits, 3)),
            dtype=jnp.float32,
        ),
    )

    def energy_density(params):
        raw_times, raw_couplings, raw_detunings, angles = params
        times = t_min + (t_max - t_min) * jax.nn.sigmoid(raw_times)
        couplings = jnp.tanh(raw_couplings)
        detunings = jnp.tanh(raw_detunings)

        def analog_digital_block(psi, block_params):
            time, coupling, detuning, block_angles = block_params
            circuit = tc.AnalogCircuit(n_qubits, inputs=psi)

            def analog_hamiltonian(state, _t, coupling=coupling, detuning=detuning):
                xy = backend.sparse_dense_matmul(h_xy, state)
                return -1.0j * (coupling * xy + detuning * (h_z_diag * state))

            circuit.add_analog_block(
                analog_hamiltonian,
                time,
                rtol=ode_rtol,
                atol=ode_atol,
                max_steps=ode_max_steps,
                ode_backend="jaxode",
                mode="raw",
            )
            for qubit in range(n_qubits):
                circuit.rz(qubit, theta=block_angles[qubit, 0])
                circuit.ry(qubit, theta=block_angles[qubit, 1])
                circuit.rz(qubit, theta=block_angles[qubit, 2])
            return circuit.state(), None

        psi, _ = jax.lax.scan(
            analog_digital_block, psi0, (times, couplings, detunings, angles)
        )
        final_circuit = tc.Circuit(n_qubits, inputs=psi)
        energy = tc.templates.measurements.sparse_expectation(final_circuit, h_target)
        return backend.real(energy) / n_qubits

    value_and_grad = backend.jit(backend.value_and_grad(energy_density))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    history = []
    for _ in range(max_steps):
        value, grads = value_and_grad(params)
        history.append(value)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    raw_times, raw_couplings, raw_detunings, _ = params
    return {
        "final_analog_times": np.asarray(
            t_min + (t_max - t_min) * jax.nn.sigmoid(raw_times)
        ),
        "final_analog_couplings": np.asarray(jnp.tanh(raw_couplings)),
        "final_analog_detunings": np.asarray(jnp.tanh(raw_detunings)),
        "energy_density_history": np.asarray(history, dtype=np.float64),
    }
