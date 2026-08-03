"""
Challenge Suite Problem 5: custom non-unitary gate cooling.

The cooling filters are implemented as ordinary RX/RZZ gates with imaginary
angles. The solution returns only NumPy values consumed by evaluate_5.py.
"""

import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.templates.measurements import mpo_expectation


K = tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("omeco-4-4")


def initial_parameters(config):
    value = config["initial_filter_strength"]
    initial = np.full((config["n_layers"] // 2, 2), value, dtype=np.float64)
    return {
        "a": K.convert_to_tensor(initial),
        "b": K.convert_to_tensor(initial),
    }


def _tfim_mpo(config):
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex128)
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    bulk[0, 0] = identity
    bulk[1, 0] = pauli_z
    bulk[2, 0] = -config["transverse_field"] * pauli_x
    bulk[2, 1] = -pauli_z
    bulk[2, 2] = identity
    arrays = [
        bulk[2:3],
        *([bulk] * (config["n_qubits"] - 2)),
        bulk[:, 0:1],
    ]
    nodes = [tc.quantum.Node(K.convert_to_tensor(array)) for array in arrays]
    for i in range(config["n_qubits"] - 1):
        nodes[i][1] ^ nodes[i + 1][0]
    return tc.quantum.QuOperator(
        out_edges=[node[2] for node in nodes],
        in_edges=[node[3] for node in nodes],
        ref_nodes=nodes,
        ignore_edges=[nodes[0][0], nodes[-1][1]],
    )


def _cooling_circuit(params, config):
    n_qubits = config["n_qubits"]
    circuit = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        circuit.h(i)
    identity = tc.gates.i()
    scale_identity = K.eye(2, dtype=tc.dtypestr)
    for block in range(config["n_layers"] // 2):
        for parity in range(2):
            rx = tc.gates.rx(
                theta=2.0j * params["a"][block, parity]
            ).tensor
            rx_pair = K.kron(rx, rx)
            rzz = K.reshape(
                tc.gates.rzz(
                    theta=2.0j * params["b"][block, parity]
                ).tensor,
                (4, 4),
            )
            fused = K.matmul(rzz, rx_pair)
            for i in range(parity, n_qubits - 1, 2):
                circuit.unitary(
                    i,
                    i + 1,
                    unitary=tc.gates.Gate(K.reshape(fused, (2, 2, 2, 2))),
                    name="fused_filter",
                )
            if parity:
                circuit.unitary(
                    0,
                    unitary=tc.gates.Gate(rx),
                    name="endpoint_filter",
                )
                circuit.unitary(
                    n_qubits - 1,
                    unitary=tc.gates.Gate(rx),
                    name="endpoint_filter",
                )
            norm_squared = K.real(
                circuit.expectation((identity, [0]), reuse=False)
            )
            scale = 1.0 / K.sqrt(norm_squared)
            circuit.unitary(
                0,
                unitary=tc.gates.Gate(scale_identity * scale),
                name="layer_rescaling",
            )
    return circuit


def energy_density(params, mpo, config):
    circuit = _cooling_circuit(params, config)
    return mpo_expectation(circuit, mpo) / config["n_qubits"]


def run_solution(config):
    params = initial_parameters(config)
    mpo = _tfim_mpo(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        current, state = carry
        value, grads = K.value_and_grad(energy_density)(current, mpo, config)
        updates, state = optimizer.update(grads, state, current)
        return (optax.apply_updates(current, updates), state), value

    @K.jit
    def optimize(current, state):
        return K.jaxy_scan(
            train_step,
            (current, state),
            K.arange(config["max_steps"]),
        )

    (params, _), energy_history = optimize(params, opt_state)
    return {
        "final_a": K.numpy(params["a"]),
        "final_b": K.numpy(params["b"]),
        "energy_density_history": K.numpy(energy_history),
    }
