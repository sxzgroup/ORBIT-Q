"""
Task Suite Problem 7: 16-qubit measurement-feedback VQE.

The TensorCircuit-NG baseline uses cond_measure for ancilla measurements and
batches fixed trajectories with vmap for deterministic trajectory-averaged
energy optimization.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("plain-experimental")

PARAMS_PER_LAYER = 48


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    return K.convert_to_tensor(
        rng.normal(
            scale=config["initial_parameter_scale"],
            size=(config["n_layers"] * PARAMS_PER_LAYER,),
        ).astype(np.float32)
    )


def trajectory_status(config):
    rng = np.random.default_rng(config["seed"] + 1)
    return K.convert_to_tensor(
        rng.random(
            (config["n_trajectories"], config["n_layers"] * config["n_ancilla_qubits"]),
            dtype=np.float32,
        )
    )


def trajectory_patterns(config, params, status):
    # RZZ is diagonal, and the ancilla CNOT ladder is a computational-basis
    # permutation. Therefore the measured bits can be sampled analytically
    # from the independent pre-ladder ancilla bits with the exact same fixed
    # uniforms. The objective is piecewise constant in the ancilla RY angles,
    # so their pathwise gradients are zero and these patterns remain fixed.
    p = np.asarray(K.numpy(params))
    uniforms = np.asarray(K.numpy(status))
    measured = np.zeros(
        (config["n_trajectories"], config["n_layers"], config["n_ancilla_qubits"]),
        dtype=np.int32,
    )
    source = np.zeros_like(measured)
    for trajectory in range(config["n_trajectories"]):
        previous_layer = np.zeros(config["n_ancilla_qubits"], dtype=np.int32)
        for layer in range(config["n_layers"]):
            offset = layer * PARAMS_PER_LAYER
            base = np.sin(p[offset + 8 : offset + 16] / 2.0) ** 2
            probability_one = base + previous_layer * (1.0 - 2.0 * base)
            previous_output = 0
            for a in range(config["n_ancilla_qubits"]):
                q = probability_one[a]
                if previous_output:
                    q = 1.0 - q
                bit = int(
                    uniforms[trajectory, layer * config["n_ancilla_qubits"] + a]
                    > 1.0 - q
                )
                measured[trajectory, layer, a] = bit
                source[trajectory, layer, a] = bit ^ previous_output
                previous_output = bit
            previous_layer = measured[trajectory, layer]
    patterns = np.stack([measured, source], axis=1)
    unique, inverse, counts = np.unique(
        patterns.reshape(config["n_trajectories"], -1),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    unique = unique.reshape(
        -1, 2, config["n_layers"], config["n_ancilla_qubits"]
    )
    return (
        K.convert_to_tensor(unique),
        K.convert_to_tensor(inverse, dtype="int32"),
        K.convert_to_tensor(counts.astype(np.float32) / config["n_trajectories"]),
    )


def make_one_pattern(config):
    n_data = config["n_data_qubits"]
    n_layers = config["n_layers"]
    transverse_field = config["transverse_field"]

    pauli_strings = []
    weights = []
    for i in range(n_data - 1):
        term = [0] * n_data
        term[i] = 3
        term[i + 1] = 3
        pauli_strings.append(term)
        weights.append(-1.0)
    for i in range(n_data):
        term = [0] * n_data
        term[i] = 1
        pauli_strings.append(term)
        weights.append(-transverse_field)
    hamiltonian = tc.quantum.PauliStringSum2COO(pauli_strings, weights)

    def one_pattern(params, pattern):
        measured, source = pattern
        c = tc.Circuit(n_data)
        for layer in range(n_layers):
            offset = layer * PARAMS_PER_LAYER
            for q in range(n_data):
                c.ry(q, theta=params[offset + q])
            theta0 = params[offset + 24 : offset + 32]
            theta1 = params[offset + 32 : offset + 40]
            for q in range(n_data):
                bitf = K.cast(measured[layer, q], "float32")
                sourcef = K.cast(source[layer, q], "float32")
                feedback_theta = theta0[q] + bitf * (theta1[q] - theta0[q])
                c.rz(
                    q,
                    theta=(1.0 - 2.0 * sourcef) * params[offset + 16 + q]
                    + (1.0 - 2.0 * bitf) * feedback_theta,
                )
            for q in range(n_data - 1):
                c.cnot(q, q + 1)
            for q in range(n_data):
                c.rz(q, theta=params[offset + 40 + q])
        return tc.templates.measurements.operator_expectation(c, hamiltonian)

    return one_pattern


def run_solution(config):
    params = initial_parameters(config)
    status = trajectory_status(config)
    patterns, inverse, weights = trajectory_patterns(config, params, status)
    one_pattern = make_one_pattern(config)
    batched_patterns = K.jit(K.vmap(one_pattern, vectorized_argnums=1))
    optimizer = optax.adam(config["learning_rate"])

    def loss_fn(p):
        return K.sum(batched_patterns(p, patterns) * weights)

    def train_step(p, state):
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, value

    train_step = K.jit(train_step)
    opt_state = optimizer.init(params)
    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, value = train_step(params, opt_state)
        energy_history.append(value)

    final_pattern_energies = batched_patterns(params, patterns)
    final_trajectory_energies = final_pattern_energies[inverse]
    return {
        "energy_history": K.numpy(K.stack(energy_history)),
        "final_trajectory_energies": K.numpy(final_trajectory_energies),
    }
