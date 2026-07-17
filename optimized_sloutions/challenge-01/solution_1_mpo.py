"""
Challenge Suite Problem 1: DMRG-MPS input with variational circuit refinement.

This reference-derived variant keeps the published gate-level ansatz intact but
evaluates the TFIM energy as a TensorCircuit MPO expectation instead of summing
individual Pauli-pattern measurements.
"""

import numpy as np
import optax
import quimb.tensor as qtn

import tensorcircuit as tc
from tensorcircuit.templates.measurements import mpo_expectation

K = tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")


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
    hamiltonian = qtn.SpinHam1D(S=0.5)
    hamiltonian += -4.0, "Z", "Z"
    hamiltonian += -2.0 * config["field"], "X"
    return tc.quantum.quimb2qop(hamiltonian.build_mpo(config["n_qubits"]))


def apply_variational_layers(circuit, params, config):
    offset = 0
    for layer in range(config["n_layers"]):
        for i in range(config["n_qubits"]):
            circuit.rz(i, theta=params[offset])
            circuit.ry(i, theta=params[offset + 1])
            circuit.rz(i, theta=params[offset + 2])
            offset += 3

        for i in range(layer % 2, config["n_qubits"] - 1, 2):
            circuit.rxx(i, i + 1, theta=params[offset])
            circuit.ryy(i, i + 1, theta=params[offset + 1])
            circuit.rzz(i, i + 1, theta=params[offset + 2])
            offset += 3


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
    energy_fn = lambda p, m: circuit_energy(p, m, config, mpo)

    def train_step(p, state, m):
        energy, grads = K.value_and_grad(energy_fn)(p, m)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, energy

    train_step = K.jit(train_step, static_argnums=(2,))

    energy_history = []
    for _ in range(config["max_steps"]):
        params, opt_state, energy = train_step(params, opt_state, mps_input)
        energy_history.append(energy)

    return {
        "energy_history": K.numpy(K.stack(energy_history)),
    }
