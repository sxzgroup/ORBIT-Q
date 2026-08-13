import numpy as np
import optax
import tensorcircuit as tc
from tensorcircuit.templates.measurements import operator_expectation


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    K = tc.backend

    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    scale = float(config["initial_parameter_scale"])
    hfield = float(config["transverse_field"])

    paulis, weights = [], []
    for i in range(n_data - 1):
        s = [0] * n_data
        s[i] = 3
        s[i + 1] = 3
        paulis.append(s)
        weights.append(-1.0)
    for i in range(n_data):
        s = [0] * n_data
        s[i] = 1
        paulis.append(s)
        weights.append(-hfield)
    ham = tc.quantum.PauliStringSum2Dense(paulis, weight=weights)
    left_data = tuple(range(n_data))

    def one_energy(params, status):
        data = None
        anc_bits = None
        last_data = None
        for layer in range(n_layers):
            cd = tc.Circuit(n_data, inputs=data) if data is not None else tc.Circuit(n_data)
            ca = tc.Circuit(n_anc)
            if anc_bits is not None:
                for i in range(n_anc):
                    ca.conditional_gate(
                        anc_bits[i], [tc.gates.i(), tc.gates.x()], i
                    )
            for i in range(n_data):
                cd.ry(i, theta=params[layer, 0, i])
            for i in range(n_anc):
                ca.ry(i, theta=params[layer, 1, i])
            c = tc.Circuit(n_qubits, mps_inputs=(cd.quvector() | ca.quvector()))
            for i in range(n_data):
                c.rzz(n_data + i, i, theta=params[layer, 2, i])
            for i in range(n_anc - 1):
                c.cnot(n_data + i, n_data + i + 1)
            ms = [
                c.cond_measure(n_data + i, status=status[layer, i])
                for i in range(n_anc)
            ]
            for i in range(n_anc):
                c.conditional_gate(
                    ms[i],
                    [
                        tc.gates.rzz(theta=params[layer, 3, i]),
                        tc.gates.rzz(theta=params[layer, 4, i]),
                    ],
                    n_data + i,
                    i,
                )
            for i in range(n_data - 1):
                c.cnot(i, i + 1)
            for i in range(n_data):
                c.rz(i, theta=params[layer, 5, i])
            bits = K.stack(ms)
            zeros = K.zeros((n_data,), dtype="float32")
            traceout = K.concat([zeros, K.cast(bits, "float32")])
            data = c.projected_subsystem(traceout, left=left_data)
            anc_bits = bits
            last_data = data
        return operator_expectation(tc.Circuit(n_data, inputs=last_data), ham)

    def batch_energies(params, statuses):
        return K.vmap(one_energy, vectorized_argnums=1)(params, statuses)

    def batch_energy(params, statuses):
        return K.mean(batch_energies(params, statuses))

    rng = np.random.default_rng(seed)
    params = K.convert_to_tensor(
        rng.normal(0.0, scale, size=(n_layers, 6, n_data)).astype(np.float32)
    )
    statuses = K.convert_to_tensor(
        rng.random((n_traj, n_layers, n_anc)).astype(np.float32)
    )

    vg = K.jit(K.value_and_grad(lambda p: batch_energy(p, statuses)))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    energy_history = np.empty((max_steps,), dtype=np.float64)

    for t in range(max_steps):
        energy, grad = vg(params)
        energy_history[t] = np.asarray(energy)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    final_es = K.jit(lambda p: batch_energies(p, statuses))(params)
    return {
        "energy_history": energy_history,
        "final_trajectory_energies": np.asarray(final_es).astype(np.float64).reshape(
            (n_traj,)
        ),
    }
