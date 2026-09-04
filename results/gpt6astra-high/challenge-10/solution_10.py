"""TensorCircuit VQE with a native diagonal CMZ tensor network."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    n = config["n_qubits"]
    layers = config["n_layers"]
    selected = tuple(config["selected_qubits"])
    strings, weights = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = s[i + 1] = 3
        strings.append(s)
        weights.append(-config["zz_strength"])
    for i in range(n):
        s = [0] * n
        s[i] = 1
        strings.append(s)
        weights.append(-config["x_strength"])
    h_matvec = tc.quantum.PauliStringSum2MVP(strings, weights)

    def energy(parameters):
        c = tc.Circuit(n)
        for i in config["initial_ones"]:
            c.x(i)
        for layer in range(layers):
            for i in range(n):
                c.rx(i, theta=parameters[layer, i, 0])
                c.rz(i, theta=parameters[layer, i, 1])
                c.ry(i, theta=parameters[layer, i, 2])
            c.cmz(*selected)
        psi = c.state()
        return jnp.real(jnp.vdot(psi, h_matvec(psi))) / n

    parameters = jnp.asarray(
        np.random.default_rng(config["seed"]).normal(
            0, config["initial_parameter_scale"], (layers, n, 3)
        ), dtype=jnp.float32
    )
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(parameters)

    @jax.jit
    def step(parameters, state):
        value, gradient = jax.value_and_grad(energy)(parameters)
        updates, state = optimizer.update(gradient, state, parameters)
        return optax.apply_updates(parameters, updates), state, value

    history = []
    for _ in range(config["max_steps"]):
        parameters, state, value = step(parameters, state)
        history.append(float(value))
    return {"energy_history": np.asarray(history),
            "final_parameters": np.asarray(parameters)}
