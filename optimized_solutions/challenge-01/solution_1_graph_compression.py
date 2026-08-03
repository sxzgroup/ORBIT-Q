"""
Task Suite Problem 1: DMRG-MPS input with variational circuit refinement.

The DMRG state is injected into a regular TensorCircuit Circuit. The solution
returns NumPy values only; external validation lives in evaluate_1.py.
"""

import numpy as np
import optax

import tensorcircuit as tc
from tensorcircuit.templates.measurements import mpo_expectation

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco-1-1")

PAULIS = [
    K.reshape(gate().tensor, [2, 2])
    for gate in (tc.gates.i, tc.gates.x, tc.gates.y, tc.gates.z)
]
PAIR_PAULIS = K.stack([K.kron(gate, gate) for gate in PAULIS])


def parameter_count(config):
    count = 0
    for layer in range(config["n_layers"]):
        count += 3 * config["n_qubits"]
        count += 3 * len(range(layer % 2, config["n_qubits"] - 1, 2))
    return count


def initial_parameters(config):
    rng = np.random.default_rng(1234)
    params = rng.normal(scale=1e-4, size=parameter_count(config)).astype(np.float32)
    return K.convert_to_tensor(params)


def tfim_mpo(config):
    eye = np.eye(2, dtype=np.complex64)
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    z_gate = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex64)
    bulk[0, 0] = eye
    bulk[1, 0] = z_gate
    bulk[2, 0] = -config["field"] * x_gate
    bulk[2, 1] = -z_gate
    bulk[2, 2] = eye

    tensors = [bulk[2:3]] + [bulk] * (config["n_qubits"] - 2) + [bulk[:, 0:1]]
    nodes = [tc.quantum.Node(K.convert_to_tensor(tensor)) for tensor in tensors]
    for i in range(config["n_qubits"] - 1):
        nodes[i][1] ^ nodes[i + 1][0]

    return tc.quantum.QuOperator(
        out_edges=[node[2] for node in nodes],
        in_edges=[node[3] for node in nodes],
        ref_nodes=nodes,
        ignore_edges=[nodes[0][0], nodes[-1][1]],
    )


def one_qubit_gates(theta):
    first, middle, last = theta[:, 0], theta[:, 1], theta[:, 2]
    cosine, sine = K.cos(middle / 2.0), K.sin(middle / 2.0)
    total = (last + first) / 2.0
    difference = (last - first) / 2.0
    row0 = K.stack(
        [
            cosine * K.exp(-1.0j * total),
            -sine * K.exp(-1.0j * difference),
        ],
        axis=-1,
    )
    row1 = K.stack(
        [
            sine * K.exp(1.0j * difference),
            cosine * K.exp(1.0j * total),
        ],
        axis=-1,
    )
    return K.stack([row0, row1], axis=-2)


def two_qubit_gates(theta):
    sine = K.sin(theta / 2.0)
    cosine = K.cos(theta / 2.0)
    sx, sy, sz = sine[:, 0], sine[:, 1], sine[:, 2]
    cx, cy, cz = cosine[:, 0], cosine[:, 1], cosine[:, 2]
    coefficients = K.stack(
        [
            cz * cy * cx - 1.0j * sz * sy * sx,
            sz * sy * cx - 1.0j * cz * cy * sx,
            sz * cy * sx - 1.0j * cz * sy * cx,
            cz * sy * sx - 1.0j * sz * cy * cx,
        ],
        axis=-1,
    )
    return K.einsum("ba,aij->bij", coefficients, PAIR_PAULIS)


def apply_variational_layers(circuit, params, config):
    n = config["n_qubits"]
    offset = 0
    single_blocks = []
    pair_blocks = []
    for layer in range(config["n_layers"]):
        single_blocks.append(params[offset : offset + 3 * n])
        offset += 3 * n
        count = len(range(layer % 2, n - 1, 2))
        pair_blocks.append(params[offset : offset + 3 * count])
        offset += 3 * count
    singles = K.reshape(
        one_qubit_gates(K.reshape(K.concat(single_blocks), [-1, 3])),
        [config["n_layers"], n, 2, 2],
    )
    pairs = two_qubit_gates(K.reshape(K.concat(pair_blocks), [-1, 3]))
    pair_offset = 0
    for layer in range(config["n_layers"]):
        bonds = tuple(range(layer % 2, n - 1, 2))
        count = len(bonds)
        layer_pairs = pairs[pair_offset : pair_offset + count]
        pair_offset += count
        products = K.reshape(
            K.einsum(
                "kab,kcd->kacbd",
                singles[layer, layer % 2 : n - 1 : 2],
                singles[layer, layer % 2 + 1 : n : 2],
            ),
            [count, 4, 4],
        )
        unitaries = K.einsum("kij,kjl->kil", layer_pairs, products)
        for j, i in enumerate(bonds):
            circuit.unitary(
                i,
                i + 1,
                unitary=tc.gates.Gate(
                    K.reshape(unitaries[j], [2, 2, 2, 2])
                ),
                name="fused_layer_gate",
            )
        if layer % 2:
            for i in (0, n - 1):
                circuit.unitary(
                    i,
                    unitary=tc.gates.Gate(singles[layer, i]),
                    name="fused_site_gate",
                )


def circuit_energy(params, mps_input, config, mpo):
    circuit = tc.Circuit(config["n_qubits"], mps_inputs=mps_input)
    apply_variational_layers(circuit, params, config)
    return mpo_expectation(circuit, mpo)


def run_solution(config):
    mps_input = tc.quantum.quimb2qop(config["dmrg_state"])
    params = initial_parameters(config)
    mpo = tfim_mpo(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    energy_fn = lambda p: circuit_energy(p, mps_input, config, mpo)

    def train_step(p, state):
        energy, grads = K.value_and_grad(energy_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, energy

    train_step = K.jit(train_step)

    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, energy = train_step(params, opt_state)
        energy_history.append(energy)

    return {
        "energy_history": K.numpy(K.stack(energy_history)),
    }
