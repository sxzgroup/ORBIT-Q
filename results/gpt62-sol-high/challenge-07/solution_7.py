import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
from tensorcircuit.templates.measurements import operator_expectation


def run_solution(config):
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        return _train(config)


def _train(config):
    n = config["n_data_qubits"]
    size = config["n_qubits"]
    layers = config["n_layers"]
    data = tuple(range(n))
    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(
        rng.normal(0, config["initial_parameter_scale"], (layers, 6, n))
    )
    uniforms = jnp.asarray(rng.random((config["n_trajectories"], layers, 1)))

    terms = np.zeros((2 * n - 1, n), dtype=int)
    for i in data:
        terms[i, i] = 1
    for i in range(n - 1):
        terms[n + i, i : i + 2] = 3
    weights = [-config["transverse_field"]] * n + [-1.0] * (n - 1)
    hamiltonian = tc.quantum.PauliStringSum2Dense(terms, weights)

    def ladder(circuit, start):
        for i in range(start, start + n - 1):
            circuit.cnot(i, i + 1)

    def in_basis(bits):
        circuit = tc.Circuit(n)
        for i in data:
            circuit.conditional_gate(bits[i], (tc.gates.i(), tc.gates.x()), i)
        return circuit

    def joint(state, ancillas):
        data_state = tc.Circuit(n, inputs=state).quvector()
        return tc.Circuit(size, mps_inputs=data_state | ancillas)

    def layer(carry, inputs):
        state, previous_bits = carry
        angles, uniform = inputs
        data_ry, ancilla_ry, entangle, feedback0, feedback1, post = angles
        ancilla = in_basis(previous_bits)
        circuit = joint(state, ancilla.quvector())
        for i in data:
            circuit.ry(i, theta=data_ry[i])
        for i in data:
            circuit.ry(n + i, theta=ancilla_ry[i])
        for i in data:
            circuit.rzz(n + i, i, theta=entangle[i])
        ladder(circuit, n)

        # RZZ is diagonal on the ancillas, and the ladder permutes their
        # basis strings. The ancilla-only circuit gives the same Z marginal.
        for i in data:
            ancilla.ry(i, theta=ancilla_ry[i])
        ladder(ancilla, 0)
        sample = tc.backend.probability_sample(1, ancilla.probability(), status=uniform)
        bits = tc.quantum.sample_int2bin(sample, n)[0].astype(jnp.int32)
        basis = jnp.concatenate((jnp.zeros_like(bits), bits))
        collapsed = circuit.projected_subsystem(basis, data)

        # Reattach the measured register in its observed computational state.
        circuit = joint(collapsed, in_basis(bits).quvector())
        for i in data:
            theta = jnp.where(bits[i] == 0, feedback0[i], feedback1[i])
            circuit.rzz(n + i, i, theta=theta)
        ladder(circuit, 0)
        for i in data:
            circuit.rz(i, theta=post[i])
        return (circuit.projected_subsystem(basis, data), bits), None

    initial_state = tc.Circuit(n).state(), jnp.zeros(n, dtype=jnp.int32)

    def trajectory(params, uniform):
        (state, _), _ = jax.lax.scan(layer, initial_state, (params, uniform))
        return operator_expectation(tc.Circuit(n, inputs=state), hamiltonian)

    energies = jax.vmap(trajectory, in_axes=(None, 0))

    def objective(params):
        return jnp.mean(energies(params, uniforms))

    # With uniforms held fixed, branch choices are locally constant; autodiff
    # differentiates the normalized conditional states on the selected branches.
    value_and_grad = jax.value_and_grad(objective)
    adam = optax.adam(config["learning_rate"])

    def update(carry, _):
        params, optimizer_state = carry
        energy, gradient = value_and_grad(params)
        updates, optimizer_state = adam.update(gradient, optimizer_state, params)
        return (optax.apply_updates(params, updates), optimizer_state), energy

    @jax.jit
    def fit(params):
        (params, _), history = jax.lax.scan(
            update, (params, adam.init(params)), None, length=config["max_steps"]
        )
        return history, energies(params, uniforms)

    history, final = fit(params)
    return {
        "energy_history": np.asarray(history),
        "final_trajectory_energies": np.asarray(final),
    }
