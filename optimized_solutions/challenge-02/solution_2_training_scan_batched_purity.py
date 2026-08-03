"""
Task Suite Problem 2: entanglement-profile-constrained VQE.

The ansatz is scanned as three even+odd brickwork blocks. Each block output is
materialized so the half-chain Renyi-2 profile is part of the differentiable
loss. The solution returns only NumPy values consumed by evaluate_2.py.
"""

import numpy as np
import optax

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def block_count(config):
    return config["n_layers"] // 2


def even_bonds(config):
    return list(range(0, config["n_qubits"] - 1, 2))


def odd_bonds(config):
    return list(range(1, config["n_qubits"] - 1, 2))


def initial_parameters(config):
    rng = np.random.default_rng(2026)
    n_repeats = block_count(config)
    n_qubits = config["n_qubits"]
    n_even = len(even_bonds(config))
    n_odd = len(odd_bonds(config))

    def normal(shape):
        return K.convert_to_tensor(
            rng.normal(scale=0.02, size=shape).astype(np.float32)
        )

    return {
        "even": {
            "ry": normal((n_repeats, n_qubits)),
            "rz": normal((n_repeats, n_qubits)),
            "xx": normal((n_repeats, n_even)),
            "yy": normal((n_repeats, n_even)),
            "zz": normal((n_repeats, n_even)),
        },
        "odd": {
            "ry": normal((n_repeats, n_qubits)),
            "rz": normal((n_repeats, n_qubits)),
            "xx": normal((n_repeats, n_odd)),
            "yy": normal((n_repeats, n_odd)),
            "zz": normal((n_repeats, n_odd)),
        },
    }


def apply_layer(circuit, layer_params, bonds, config):
    for i in range(config["n_qubits"]):
        circuit.ry(i, theta=layer_params["ry"][i])
        circuit.rz(i, theta=layer_params["rz"][i])

    for bond_index, i in enumerate(bonds):
        circuit.rxx(i, i + 1, theta=layer_params["xx"][bond_index])
        circuit.ryy(i, i + 1, theta=layer_params["yy"][bond_index])
        circuit.rzz(i, i + 1, theta=layer_params["zz"][bond_index])


def renyi2_entropy(state, config):
    traceout = list(range(config["subsystem_size"], config["n_qubits"]))
    rho = tc.quantum.reduced_density_matrix(state, cut=traceout)
    purity = K.real(K.sum(rho * K.conj(rho)))
    return -K.real(K.log(purity))


def block_states(params, input_state, config):
    even = even_bonds(config)
    odd = odd_bonds(config)

    def block_step(state, block_params):
        circuit = tc.Circuit(config["n_qubits"], inputs=state)
        apply_layer(circuit, block_params["even"], even, config)
        apply_layer(circuit, block_params["odd"], odd, config)
        state = circuit.state()

        return state, state

    final_state, states = K.jaxy_scan(block_step, input_state, params)
    entropies = K.vmap(lambda state: renyi2_entropy(state, config))(states)
    return final_state, entropies


def build_xxz_mvp(config):
    structures = []
    weights = []

    for i in range(config["n_qubits"] - 1):
        for pauli, weight in ((1, 1.0), (2, 1.0), (3, config["zz_anisotropy"])):
            term = [0] * config["n_qubits"]
            term[i] = pauli
            term[i + 1] = pauli
            structures.append(term)
            weights.append(weight)

    for i in range(config["n_qubits"]):
        term = [0] * config["n_qubits"]
        term[i] = 3
        structures.append(term)
        weights.append(config["staggered_field"] * ((-1.0) ** i))

    return tc.quantum.PauliStringSum2MVP(structures, weights)


def xxz_energy(state, hamiltonian_mvp):
    h_state = hamiltonian_mvp(state)
    return K.real(K.tensordot(K.conj(state), h_state, 1))


def observables(params, input_state, hamiltonian_mvp, config, target_entropies):
    final_state, entropies = block_states(params, input_state, config)
    energy_density = xxz_energy(final_state, hamiltonian_mvp) / config["n_qubits"]
    entropy_mse = K.mean((entropies - target_entropies) ** 2)
    loss = energy_density + config["entropy_weight"] * entropy_mse
    return loss, (energy_density, entropies, entropy_mse)


def run_solution(config):
    params = initial_parameters(config)
    circuit = tc.Circuit(config["n_qubits"])
    for i in range(1, config["n_qubits"], 2):
        circuit.x(i)
    input_state = circuit.state()
    hamiltonian_mvp = build_xxz_mvp(config)
    target_entropies = K.convert_to_tensor(
        np.asarray(config["target_entropies"], dtype=np.float32)
    )
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return observables(p, input_state, hamiltonian_mvp, config, target_entropies)

    def train_step(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(loss_fn, has_aux=True)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        energy_density, entropies, entropy_mse = aux
        return (p, state), (loss, energy_density, entropies, entropy_mse)

    def train(carry):
        return K.jaxy_scan(
            train_step, carry, K.arange(config["max_steps"])
        )

    (_, _), history = K.jit(train)((params, opt_state))
    loss_history, energy_density_history, entropy_history, entropy_mse_history = (
        history
    )

    return {
        "energy_density_history": K.numpy(energy_density_history),
        "loss_history": K.numpy(loss_history),
        "entropy_mse_history": K.numpy(entropy_mse_history),
        "entropy_history": K.numpy(entropy_history),
    }
