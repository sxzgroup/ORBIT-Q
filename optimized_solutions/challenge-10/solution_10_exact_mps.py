"""
Task Suite Problem 10: VQE with an 18-qubit CMZ hyperedge.

The ansatz uses TensorCircuit-NG's built-in cmz gate on non-adjacent selected
qubits. The Hamiltonian is evaluated as an MPO expectation so the optimization
step can be JIT-compiled into a highly efficient fixed graph.
"""

import numpy as np
import optax

import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    shape = (config["n_layers"], config["n_qubits"], 3)
    return K.convert_to_tensor(
        rng.normal(
            scale=config["initial_parameter_scale"],
            size=shape,
        ).astype(np.float32)
    )


def _cmz_mpo(n_qubits, selected):
    """I - 2 prod_q |1><1|_q, including identity sites, at bond dimension 2."""
    identity = np.eye(2, dtype=np.complex64)
    projector = np.diag([0.0, 1.0]).astype(np.complex64)
    selected = set(selected)
    tensors = []
    for q in range(n_qubits):
        local = projector if q in selected else identity
        if q == 0:
            tensor = np.zeros((1, 2, 2, 2), dtype=np.complex64)
            tensor[0, :, :, 0] = identity
            tensor[0, :, :, 1] = -2.0 * local
        elif q == n_qubits - 1:
            tensor = np.zeros((2, 2, 2, 1), dtype=np.complex64)
            tensor[0, :, :, 0] = identity
            tensor[1, :, :, 0] = local
        else:
            tensor = np.zeros((2, 2, 2, 2), dtype=np.complex64)
            tensor[0, :, :, 0] = identity
            tensor[1, :, :, 1] = local
        tensors.append(K.convert_to_tensor(tensor))
    return tensors


def _tfim_mpo(n_qubits, zz, xs):
    """-zz sum ZZ - xs sum X as a left-to-right bond-3 MPO."""
    identity = np.eye(2, dtype=np.complex64)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex64)
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex64)
    bulk[0, 0] = identity
    bulk[1, 0] = pauli_z
    bulk[2, 0] = -xs * pauli_x
    bulk[2, 1] = -zz * pauli_z
    bulk[2, 2] = identity
    arrays = [bulk[2:3], *([bulk] * (n_qubits - 2)), bulk[:, 0:1]]
    return [
        K.convert_to_tensor(np.transpose(array, (0, 2, 3, 1)))
        for array in arrays
    ]


def _initial_mps(n_qubits, initial_ones):
    initial_ones = set(initial_ones)
    tensors = []
    for q in range(n_qubits):
        tensor = np.zeros((1, 2, 1), dtype=np.complex64)
        tensor[0, int(q in initial_ones), 0] = 1.0
        tensors.append(K.convert_to_tensor(tensor))
    return tensors


def _rotation(theta):
    return (
        tc.gates.ry(theta=theta[2]).tensor
        @ tc.gates.rz(theta=theta[1]).tensor
        @ tc.gates.rx(theta=theta[0]).tensor
    )


def _apply_one_site(tensor, gate):
    return K.einsum("ab,ibj->iaj", gate, tensor)


def _apply_mpo(tensors, mpo):
    """The local contraction kernel from MPSCircuit.apply_MPO, without QR."""
    result = []
    for tensor, operator in zip(tensors, mpo):
        contracted = K.einsum("iabj,kbl->ikajl", operator, tensor)
        ni, nk, d, nj, nl = K.shape_tuple(contracted)
        result.append(K.reshape(contracted, (ni * nk, d, nj * nl)))
    return result


def _mpo_expectation(tensors, mpo):
    environment = K.ones((1, 1, 1), dtype=tc.dtypestr)
    for tensor, operator in zip(tensors, mpo):
        environment = K.einsum(
            "xik,xay,iabj,kbl->yjl",
            environment,
            K.conj(tensor),
            operator,
            tensor,
        )
    return K.real(environment[0, 0, 0])


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    initial_mps = _initial_mps(n_qubits, config["initial_ones"])
    cmz_mpo = _cmz_mpo(n_qubits, config["selected_qubits"])
    tfim_mpo = _tfim_mpo(
        n_qubits,
        float(config["zz_strength"]),
        float(config["x_strength"]),
    )

    def energy_density(params):
        circuit = tc.MPSCircuit(
            n_qubits,
            tensors=initial_mps,
            center_position=0,
        )
        tensors = circuit.get_tensors()
        for layer in range(n_layers):
            tensors = [
                _apply_one_site(tensor, _rotation(params[layer, q]))
                for q, tensor in enumerate(tensors)
            ]
            tensors = _apply_mpo(tensors, cmz_mpo)
        circuit = tc.MPSCircuit(
            n_qubits,
            tensors=tensors,
            center_position=0,
        )
        return _mpo_expectation(circuit.get_tensors(), tfim_mpo) / n_qubits

    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train(carry, _):
        current, state = carry
        value, grads = K.value_and_grad(energy_density)(current)
        updates, state = optimizer.update(grads, state, current)
        return (optax.apply_updates(current, updates), state), value

    @K.jit
    def optimize(current, state):
        return K.jaxy_scan(train, (current, state), K.arange(max_steps))

    (params, _), energy_history = optimize(params, opt_state)
    return {
        "energy_history": K.numpy(energy_history),
        "final_parameters": K.numpy(params),
    }
