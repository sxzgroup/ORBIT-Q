"""
Task Suite Problem 3: probability-aware post-selected cooling.

The selected measurement branch is represented with Circuit.post_select. The
solution returns only NumPy values consumed by evaluate_3.py.
"""

import numpy as np
import optax
import tensorcircuit as tc


K = tc.set_backend("jax")
tc.set_dtype("complex64")


def initial_parameters(config):
    rng = np.random.default_rng(2027)
    repeats = config["n_steps"] // 2
    n = config["n_qubits"]

    def normal(shape):
        return K.convert_to_tensor(
            rng.normal(scale=0.02, size=shape).astype(np.float32)
        )

    return {
        "even": {
            "rx": normal((repeats, n)),
            "xx": normal((repeats, n // 2)),
            "zz": normal((repeats, n // 2)),
        },
        "odd": {
            "rx": normal((repeats, n)),
            "xx": normal((repeats, n // 2 - 1)),
            "zz": normal((repeats, n // 2 - 1)),
        },
    }


def pair_map(odd, even, xx, zz, odd_rx, even_rx, even_left):
    rxx = K.reshape(tc.gates.rxx_gate(theta=xx).tensor, [4, 4])
    rzz = K.reshape(tc.gates.rzz_gate(theta=zz).tensor, [4, 4])
    ro = tc.gates.rx_gate(theta=odd_rx).tensor
    re = tc.gates.rx_gate(theta=even_rx).tensor
    if even_left:
        state = K.kron(even, odd)
        rotations = K.kron(re, ro)
    else:
        state = K.kron(odd, even)
        rotations = K.kron(ro, re)
    state = K.tensordot(
        rotations,
        K.tensordot(rzz, K.tensordot(rxx, state, 1), 1),
        1,
    )
    state = K.reshape(state, [2, 2])
    return state[0, :] if even_left else state[:, 0]


def cooling_product(params, config):
    n_survivors = config["n_qubits"] // 2
    zero = K.convert_to_tensor(np.array([1.0, 0.0], dtype=np.complex64))
    plus = tc.gates.h().tensor[:, 0]
    states = K.stack([plus for _ in range(n_survivors)])
    even_inputs = K.stack(
        [plus] + [zero for _ in range(config["n_steps"] // 2 - 1)]
    )

    def block_step(carry, inputs):
        states, boundary_amplitude = carry
        even_params, odd_params, even_input = inputs

        def even_map(odd, xx, zz, odd_rx, even_rx):
            return pair_map(
                odd,
                even_input,
                xx,
                zz,
                odd_rx,
                even_rx,
                True,
            )

        states = K.vmap(
            even_map, vectorized_argnums=(0, 1, 2, 3, 4)
        )(
            states,
            even_params["xx"],
            even_params["zz"],
            even_params["rx"][1::2],
            even_params["rx"][::2],
        )

        even_only = K.tensordot(
            tc.gates.rx_gate(theta=odd_params["rx"][0]).tensor, zero, 1
        )
        boundary_amplitude = boundary_amplitude * even_only[0]

        def odd_map(odd, xx, zz, odd_rx, even_rx):
            return pair_map(
                odd,
                zero,
                xx,
                zz,
                odd_rx,
                even_rx,
                False,
            )

        paired_states = K.vmap(
            odd_map, vectorized_argnums=(0, 1, 2, 3, 4)
        )(
            states[:-1],
            odd_params["xx"],
            odd_params["zz"],
            odd_params["rx"][1:-1:2],
            odd_params["rx"][2:-1:2],
        )
        last_state = K.tensordot(
            tc.gates.rx_gate(theta=odd_params["rx"][-1]).tensor,
            states[-1],
            1,
        )
        states = K.concat(
            [paired_states, K.reshape(last_state, [1, 2])]
        )
        return (states, boundary_amplitude), K.zeros([])

    (states, boundary_amplitude), _ = K.jaxy_scan(
        block_step,
        (states, K.convert_to_tensor(np.complex64(1.0))),
        (params["even"], params["odd"], even_inputs),
    )
    probabilities = K.real(K.sum(K.conj(states) * states, axis=1))
    states = states / K.reshape(K.sqrt(probabilities + 1e-12), [-1, 1])
    boundary_probability = K.real(
        K.conj(boundary_amplitude) * boundary_amplitude
    )
    log_probability_sum = (
        K.sum(K.log(probabilities + 1e-12))
        + K.log(boundary_probability + 1e-12)
    )
    return states, log_probability_sum


def one_qubit_expectation(state, operator):
    return K.real(
        K.sum(K.conj(state) * K.tensordot(operator.tensor, state, 1))
    )


def observables(params, config):
    states, log_probability_sum = cooling_product(params, config)
    x_gate = tc.gates.x()
    z_gate = tc.gates.z()
    xs = K.vmap(lambda state: one_qubit_expectation(state, x_gate))(states)
    zs = K.vmap(lambda state: one_qubit_expectation(state, z_gate))(states)
    zz_sum = 2.0 * K.sum(zs[:-1]) + zs[-1]
    energy = -(zz_sum + config["transverse_field"] * K.sum(xs))
    energy_density = energy / config["n_qubits"]
    n_events = config["n_steps"] * (config["n_qubits"] // 2)
    mean_log_probability = log_probability_sum / n_events
    success_probability = K.exp(log_probability_sum)
    loss = (
        energy_density
        - config["log_probability_weight"] * mean_log_probability
    )
    return loss, (energy_density, success_probability, mean_log_probability)


def run_solution(config):
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def train_step(carry, _):
        p, state = carry
        (loss, aux), grads = K.value_and_grad(observables, has_aux=True)(p, config)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        energy, success, mean_log = aux
        return (p, state), K.stack([energy, success, mean_log, loss])

    def train(p, state):
        return K.jaxy_scan(
            train_step, (p, state), K.arange(config["max_steps"])
        )

    (_, _), history = K.jit(train)(params, opt_state)
    history = K.numpy(history)
    return {
        "energy_density_history": history[:, 0],
        "success_probability_history": history[:, 1],
        "mean_log_probability_history": history[:, 2],
        "loss_history": history[:, 3],
    }
