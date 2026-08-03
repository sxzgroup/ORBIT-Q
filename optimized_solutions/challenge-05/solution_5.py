"""
Task Suite Problem 5: custom non-unitary gate cooling.

The cooling filters are implemented as ordinary RX/RZZ gates with imaginary
angles. The solution returns only NumPy values consumed by evaluate_5.py.
"""

import numpy as np
import optax

import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def initial_parameters(config):
    value = config["initial_filter_strength"]
    initial = np.full((config["n_layers"] // 2, 2), value, dtype=np.float32)
    return {
        "a": K.convert_to_tensor(initial),
        "b": K.convert_to_tensor(initial),
    }


def _initial_mps(n_qubits):
    plus = np.full((1, 2, 1), 1.0 / np.sqrt(2.0), dtype=np.complex64)
    tensors = [K.convert_to_tensor(plus) for _ in range(n_qubits)]
    return tc.MPSCircuit(
        n_qubits,
        tensors=tensors,
        center_position=0,
    ).get_tensors()


def _apply_one_site(tensor, gate):
    return K.einsum("ab,ibj->iaj", gate, tensor)


def _apply_local_mpo(tensor, operator):
    """Local contraction kernel from MPSCircuit.apply_MPO, without QR/RQ."""
    contracted = K.einsum("iabj,kbl->ikajl", operator, tensor)
    ni, nk, d, nj, nl = K.shape_tuple(contracted)
    return K.reshape(contracted, (ni * nk, d, nj * nl))


def _apply_rzz(tensors, site, strength):
    """Apply exp(strength Z.Z) exactly as a two-site bond-2 MPO."""
    identity = K.eye(2, dtype=tc.dtypestr)
    pauli_z = K.convert_to_tensor(
        np.diag([1.0, -1.0]).astype(np.complex64)
    )
    positive = K.exp(strength)
    negative = K.exp(-strength)
    cosh = 0.5 * (positive + negative)
    sinh = 0.5 * (positive - negative)
    left = K.reshape(K.stack([identity, pauli_z], axis=-1), (1, 2, 2, 2))
    right = K.reshape(
        K.stack([cosh * identity, sinh * pauli_z], axis=0),
        (2, 2, 2, 1),
    )
    tensors = list(tensors)
    tensors[site] = _apply_local_mpo(tensors[site], left)
    tensors[site + 1] = _apply_local_mpo(tensors[site + 1], right)
    return tensors


def _mps_norm(tensors):
    environment = K.ones((1, 1), dtype=tc.dtypestr)
    for tensor in tensors:
        environment = K.einsum(
            "ij,iar,jas->rs",
            environment,
            K.conj(tensor),
            tensor,
        )
    return K.sqrt(K.real(environment[0, 0]))


def _apply_filter_layer(tensors, a, b, bonds):
    rx = tc.gates.rx(theta=2.0j * a).tensor
    tensors = [_apply_one_site(tensor, rx) for tensor in tensors]
    for site in bonds:
        tensors = _apply_rzz(tensors, site, b)
    norm = _mps_norm(tensors)
    tensors = list(tensors)
    tensors[0] = tensors[0] / norm
    return tensors


def cooling_tensors(params, config):
    n_qubits = int(config["n_qubits"])
    even = range(0, n_qubits - 1, 2)
    odd = range(1, n_qubits - 1, 2)
    tensors = _initial_mps(n_qubits)
    for block in range(config["n_layers"] // 2):
        tensors = _apply_filter_layer(
            tensors,
            params["a"][block, 0],
            params["b"][block, 0],
            even,
        )
        tensors = _apply_filter_layer(
            tensors,
            params["a"][block, 1],
            params["b"][block, 1],
            odd,
        )
    return tensors


def _tfim_mpo(n_qubits, transverse_field):
    identity = np.eye(2, dtype=np.complex64)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    pauli_z = np.diag([1.0, -1.0]).astype(np.complex64)
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex64)
    bulk[0, 0] = identity
    bulk[1, 0] = pauli_z
    bulk[2, 0] = -transverse_field * pauli_x
    bulk[2, 1] = -pauli_z
    bulk[2, 2] = identity
    arrays = [bulk[2:3], *([bulk] * (n_qubits - 2)), bulk[:, 0:1]]
    return [
        K.convert_to_tensor(np.transpose(array, (0, 2, 3, 1)))
        for array in arrays
    ]


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


def energy_density(params, tfim_mpo, config):
    tensors = cooling_tensors(params, config)
    return _mpo_expectation(tensors, tfim_mpo) / config["n_qubits"]


def run_solution(config):
    params = initial_parameters(config)
    tfim_mpo = _tfim_mpo(
        int(config["n_qubits"]),
        float(config["transverse_field"]),
    )
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        current, state = carry
        value, grads = K.value_and_grad(energy_density)(
            current,
            tfim_mpo,
            config,
        )
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
