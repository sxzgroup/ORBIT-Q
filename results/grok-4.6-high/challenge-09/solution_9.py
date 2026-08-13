import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")


def _extract_cone(gate_tape, measured_qubits):
    cone = set(measured_qubits)
    relevant = []
    for gate in reversed(gate_tape):
        if len(gate) == 3:
            name, qubit, parameter_index = gate
            if qubit in cone:
                relevant.append(gate)
        else:
            name, qubit_a, qubit_b, parameter_index = gate
            if qubit_a in cone or qubit_b in cone:
                cone.add(qubit_a)
                cone.add(qubit_b)
                relevant.append(gate)
    relevant.reverse()
    cone_qubits = sorted(cone)
    qmap = {qubit: i for i, qubit in enumerate(cone_qubits)}
    local_gates = []
    full_indices = []
    for gate in relevant:
        if len(gate) == 3:
            name, qubit, parameter_index = gate
            local_gates.append((name, qmap[qubit]))
            full_indices.append(parameter_index)
        else:
            name, qubit_a, qubit_b, parameter_index = gate
            local_gates.append((name, qmap[qubit_a], qmap[qubit_b]))
            full_indices.append(parameter_index)
    return len(cone_qubits), local_gates, np.asarray(full_indices, dtype=np.int32), qmap


def _expectation_fn(n_qubits, local_gates, xyz):
    def expectation(params):
        circuit = tc.Circuit(n_qubits)
        for qubit in range(n_qubits):
            circuit.h(qubit)
        for index, gate in enumerate(local_gates):
            theta = params[index]
            if len(gate) == 2:
                getattr(circuit, gate[0])(gate[1], theta=theta)
            else:
                getattr(circuit, gate[0])(gate[1], gate[2], theta=theta)
        return tc.backend.real(circuit.expectation_ps(**xyz))

    return expectation


def run_solution(config):
    gate_tape = config["gate_tape"]
    pauli_terms = config["pauli_terms"]
    n_restarts = int(config["n_restarts"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    parameter_count = int(config.get("parameter_count", max(gate[-1] for gate in gate_tape) + 1))

    term_data = []
    for weight, pauli_ops in pauli_terms:
        measured = [qubit for _, qubit in pauli_ops]
        n_qubits, local_gates, full_indices, qmap = _extract_cone(gate_tape, measured)
        xyz = {}
        for axis, qubit in pauli_ops:
            xyz.setdefault(axis, []).append(int(qmap[qubit]))
        term_data.append(
            {
                "weight": float(weight),
                "n_qubits": n_qubits,
                "local_gates": local_gates,
                "full_indices": full_indices,
                "xyz": xyz,
            }
        )

    energy_fns = [
        _expectation_fn(term["n_qubits"], term["local_gates"], term["xyz"])
        for term in term_data
    ]
    weights = tuple(term["weight"] for term in term_data)

    def objective(param_tree):
        value = 0.0
        for energy, params, weight in zip(energy_fns, param_tree, weights):
            value = value + weight * energy(params)
        return value

    optimizer = optax.adam(learning_rate)

    def optimize_one(param_tree):
        def body(carry, _):
            params, opt_state = carry
            value, grads = jax.value_and_grad(lambda tree: -objective(tree))(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), -value

        opt_state = optimizer.init(param_tree)
        _, history = jax.lax.scan(body, (param_tree, opt_state), None, length=max_steps)
        return history

    optimize_one = jax.jit(optimize_one)

    init_trees = []
    for restart_index in range(n_restarts):
        rng = np.random.default_rng(seed + 100000 + restart_index)
        full_params = rng.normal(0.0, scale, size=parameter_count).astype(np.float32)
        init_trees.append(
            tuple(jnp.asarray(full_params[term["full_indices"]]) for term in term_data)
        )

    histories = [optimize_one(tree) for tree in init_trees]
    observable_history = np.asarray(jnp.stack(histories), dtype=np.float64)
    return {"observable_history": observable_history}
