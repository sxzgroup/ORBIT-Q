"""
Task Suite Problem 9: random local light-cone optimization.

The evaluator passes an explicit framework-neutral gate tape and Pauli-term
list. This solution constructs the full 512-qubit circuit from that tape and
uses TensorCircuit's automatic light-cone contraction for the two local terms.
"""

import numpy as np

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def parameter_index(gate):
    return gate[2] if len(gate) == 3 else gate[3]


def pauli_indices(term, qubit_map):
    xs, ys, zs = [], [], []
    for pauli, qubit in term:
        mapped = qubit_map[qubit]
        if pauli == "x":
            xs.append(mapped)
        elif pauli == "y":
            ys.append(mapped)
        elif pauli == "z":
            zs.append(mapped)
        else:
            raise ValueError(f"Unknown Pauli axis: {pauli}")
    return xs, ys, zs


def extract_cone(gate_tape, pauli_item):
    coeff, term = pauli_item
    support = {qubit for _, qubit in term}
    retained = []
    for gate in reversed(gate_tape):
        if len(gate) == 3:
            relevant = gate[1] in support
        else:
            relevant = gate[1] in support or gate[2] in support
            if relevant:
                support.update((gate[1], gate[2]))
        if relevant:
            retained.append(gate)
    qubit_map = {qubit: index for index, qubit in enumerate(sorted(support))}
    compact_gates = []
    for gate in reversed(retained):
        if len(gate) == 3:
            compact_gates.append((gate[0], qubit_map[gate[1]], gate[2]))
        else:
            compact_gates.append(
                (gate[0], qubit_map[gate[1]], qubit_map[gate[2]], gate[3])
            )
    return {
        "coeff": coeff,
        "n_qubits": len(support),
        "gates": tuple(compact_gates),
        "paulis": pauli_indices(term, qubit_map),
        "parameters": frozenset(parameter_index(gate) for gate in retained),
    }


def parameter_groups(cones):
    remaining = set(range(len(cones)))
    groups = []
    while remaining:
        component = {min(remaining)}
        active = set(cones[next(iter(component))]["parameters"])
        changed = True
        while changed:
            changed = False
            for index in sorted(remaining - component):
                if active.intersection(cones[index]["parameters"]):
                    component.add(index)
                    active.update(cones[index]["parameters"])
                    changed = True
        remaining.difference_update(component)
        groups.append(tuple(cones[index] for index in sorted(component)))
    return tuple(groups)


def initial_active_parameters(config, active_indices):
    active_indices = np.asarray(active_indices, dtype=np.int64)
    params = np.empty((config["n_restarts"], len(active_indices)), dtype=np.float32)
    for restart_index in range(config["n_restarts"]):
        rng = np.random.default_rng(config["seed"] + 100000 + restart_index)
        full_row = rng.normal(
            scale=config["initial_parameter_scale"],
            size=(config["parameter_count"],),
        ).astype(np.float32)
        params[restart_index] = full_row[active_indices]
    return K.convert_to_tensor(params)


def train_group(config, cones, initial_params, active_indices):
    positions = {parameter: index for index, parameter in enumerate(active_indices)}

    def loss_fn(params):
        total = 0.0
        for cone in cones:
            circuit = tc.Circuit(cone["n_qubits"])
            for qubit in range(cone["n_qubits"]):
                circuit.h(qubit)
            for gate in cone["gates"]:
                if len(gate) == 3:
                    getattr(circuit, gate[0])(
                        gate[1], theta=params[positions[gate[2]]]
                    )
                else:
                    getattr(circuit, gate[0])(
                        gate[1], gate[2], theta=params[positions[gate[3]]]
                    )
            xs, ys, zs = cone["paulis"]
            total += cone["coeff"] * K.real(
                circuit.expectation_ps(x=xs, y=ys, z=zs, enable_lightcone=True)
            )
        return -total

    value_and_grad = K.vmap(K.value_and_grad(loss_fn), vectorized_argnums=0)

    def train_step(state, _):
        params, moment1, moment2, step_count = state
        values, grads = value_and_grad(params)
        step_count = step_count + 1.0
        moment1 = 0.9 * moment1 + 0.1 * grads
        moment2 = 0.999 * moment2 + 0.001 * grads * grads
        moment1_hat = moment1 / (1.0 - 0.9**step_count)
        moment2_hat = moment2 / (1.0 - 0.999**step_count)
        params = params - config["learning_rate"] * moment1_hat / (
            K.sqrt(moment2_hat) + 1.0e-8
        )
        return (params, moment1, moment2, step_count), -values

    def optimize(params):
        init = (
            params,
            K.zeros_like(params),
            K.zeros_like(params),
            K.convert_to_tensor(np.array(0.0, dtype=np.float32)),
        )
        _, history = K.jaxy_scan(train_step, init, K.zeros([config["max_steps"]]))
        return history

    return K.jit(optimize)(initial_params)


def run_solution(config):
    gate_tape = tuple(config["gate_tape"])
    cones = tuple(extract_cone(gate_tape, term) for term in config["pauli_terms"])
    groups = parameter_groups(cones)
    all_active = tuple(sorted(set().union(*(cone["parameters"] for cone in cones))))
    initial_params = initial_active_parameters(config, all_active)
    all_positions = {parameter: index for index, parameter in enumerate(all_active)}

    histories = []
    for group in groups:
        group_active = tuple(sorted(set().union(*(cone["parameters"] for cone in group))))
        columns = tuple(all_positions[parameter] for parameter in group_active)
        histories.append(
            train_group(config, group, initial_params[:, columns], group_active)
        )

    history = histories[0]
    for group_history in histories[1:]:
        history = history + group_history
    return {
        "observable_history": K.numpy(history).T.astype(np.float64),
    }
