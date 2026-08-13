import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
import jax


def _lightcone_data(paulis, tape):
    active = {q for _, q in paulis}
    selected = []
    for idx in range(len(tape) - 1, -1, -1):
        gate = tape[idx]
        qubits = gate[1:3] if len(gate) == 4 else gate[1:2]
        if active.intersection(qubits):
            selected.append((idx, gate))
            active.update(qubits)
    selected.reverse()
    mapping = {q: i for i, q in enumerate(sorted(active))}
    gates = []
    param_indices = []
    pos = 0
    for _, gate in selected:
        if len(gate) == 3:
            name, a, p = gate
            qubits = (mapping[a],)
        else:
            name, a, b, p = gate
            qubits = (mapping[a], mapping[b])
        gates.append((name, qubits, pos))
        param_indices.append(p)
        pos += 1
    ops = tuple((p, mapping[q]) for p, q in paulis)
    return {
        "n": len(mapping),
        "gates": tuple(gates),
        "ops": ops,
        "indices": np.asarray(param_indices, dtype=np.int64),
    }


def _expectation(params, data):
    c = tc.Circuit(data["n"])
    for name, qubits, pos in data["gates"]:
        getattr(c, name)(*qubits, theta=params[pos])
    ops = tuple((getattr(tc.gates, p)(), [q]) for p, q in data["ops"])
    return tc.backend.real(c.expectation(*ops))


def _compile_term(data):
    return jax.jit(jax.value_and_grad(lambda p: _expectation(p, data)))


def run_solution(config):
    tape = config["gate_tape"]
    parameter_count = config["parameter_count"]
    n_restarts = config["n_restarts"]
    max_steps = config["max_steps"]
    learning_rate = config["learning_rate"]
    scale = config["initial_parameter_scale"]
    base_seed = config["seed"] + 100000

    terms = []
    for coefficient, paulis in config["pauli_terms"]:
        data = _lightcone_data(paulis, tape)
        data["value_and_grad"] = _compile_term(data)
        terms.append((coefficient, data))

    history = np.empty((n_restarts, max_steps), dtype=np.float32)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for restart in range(n_restarts):
        rng = np.random.default_rng(base_seed + restart)
        theta = rng.normal(0.0, scale, size=parameter_count).astype(np.float32)
        m = np.zeros(parameter_count, dtype=np.float32)
        v = np.zeros(parameter_count, dtype=np.float32)

        for step in range(max_steps):
            objective = 0.0
            grad = np.zeros(parameter_count, dtype=np.float32)
            for coefficient, data in terms:
                value, term_grad = data["value_and_grad"](theta[data["indices"]])
                objective += coefficient * float(value)
                np.add.at(
                    grad,
                    data["indices"],
                    -coefficient * np.asarray(term_grad, dtype=np.float32),
                )

            history[restart, step] = objective
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1 ** (step + 1))
            v_hat = v / (1.0 - beta2 ** (step + 1))
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    return {"observable_history": history}
