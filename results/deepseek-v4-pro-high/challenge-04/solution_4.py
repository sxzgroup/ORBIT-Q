import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


tc.set_backend("jax")


def _make_inputs(n):
    dtype = tc.dtypestr

    zero = jnp.zeros((2,) * n, dtype=dtype).at[(0,) * n].set(1.0)
    plus = jnp.ones((2,) * n, dtype=dtype) / jnp.sqrt(2.0**n)

    ghz = (
        jnp.zeros((2,) * n, dtype=dtype)
        .at[(0,) * n]
        .set(1.0)
        .at[(1,) * n]
        .set(1.0)
    )
    ghz = ghz / jnp.sqrt(2.0)

    pair = (
        jnp.zeros((2, 2), dtype=dtype)
        .at[(0, 1)]
        .set(1.0)
        .at[(1, 0)]
        .set(1.0)
    )
    pair = pair / jnp.sqrt(2.0)
    bell = pair
    for _ in range(n // 2 - 1):
        bell = jnp.kron(bell, pair)
    bell = bell.reshape((2,) * n)

    return jnp.stack([ghz, bell, zero, plus])


def _channel(p01, p10):
    dtype = tc.dtypestr
    k0 = jnp.diag(
        jnp.array(
            [jnp.sqrt(1.0 - p01), jnp.sqrt(1.0 - p10)],
            dtype=dtype,
        )
    )
    k1 = jnp.zeros((2, 2), dtype=dtype).at[(0, 1)].set(jnp.sqrt(p10))
    k2 = jnp.zeros((2, 2), dtype=dtype).at[(1, 0)].set(jnp.sqrt(p01))
    return [tc.Gate(k0), tc.Gate(k1), tc.Gate(k2)]


def _one_circuit_expectations(state, p01, p10, n, theta):
    c = tc.DMCircuit2(n, inputs=state)

    for i in range(0, n, 2):
        c.rxx(i, i + 1, theta=theta)
        for q in (i, i + 1):
            c.general_kraus(_channel(p01, p10), q)

    for i in range(1, n - 1, 2):
        c.rxx(i, i + 1, theta=theta)
        for q in (i, i + 1):
            c.general_kraus(_channel(p01, p10), q)

    values = []
    for i in range(n):
        values.append(jnp.real(c.expectation((tc.gates.z(), [i]), reuse=False)))

    parity_ops = [(tc.gates.z(), [i]) for i in range(n)]
    values.append(jnp.real(c.expectation(*parity_ops, reuse=False)))
    return jnp.stack(values)


def _all_expectations(r, inputs, n, theta):
    p01 = jax.nn.sigmoid(r[0])
    p10 = jax.nn.sigmoid(r[1])
    rows = [
        _one_circuit_expectations(inputs[k], p01, p10, n, theta)
        for k in range(inputs.shape[0])
    ]
    return jnp.stack(rows)


def run_solution(config):
    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    true_p01 = float(config["true_p01"])
    true_p10 = float(config["true_p10"])
    initial_p01 = float(config["initial_p01"])
    initial_p10 = float(config["initial_p10"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    inputs = _make_inputs(n)

    r_true = jnp.array(
        [
            jnp.log(true_p01 / (1.0 - true_p01)),
            jnp.log(true_p10 / (1.0 - true_p10)),
        ]
    )
    r_init = jnp.array(
        [
            jnp.log(initial_p01 / (1.0 - initial_p01)),
            jnp.log(initial_p10 / (1.0 - initial_p10)),
        ]
    )

    table_fn = jax.jit(lambda r: _all_expectations(r, inputs, n, theta))
    target = table_fn(r_true)

    def loss_fn(r):
        return jnp.mean((table_fn(r) - target) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    params = r_init
    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)

    loss_history = np.empty(max_steps, dtype=np.float64)

    for step in range(max_steps):
        loss_value, grads = value_and_grad(params)
        loss_history[step] = float(np.asarray(loss_value))

        first_moment = beta1 * first_moment + (1.0 - beta1) * grads
        second_moment = beta2 * second_moment + (1.0 - beta2) * grads * grads
        first_hat = first_moment / (1.0 - beta1 ** (step + 1))
        second_hat = second_moment / (1.0 - beta2 ** (step + 1))
        params = params - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + epsilon
        )

    final_probabilities = np.asarray(jax.nn.sigmoid(params), dtype=np.float64)
    fitted_expectations = np.asarray(table_fn(params), dtype=np.float64)

    return {
        "loss_history": loss_history,
        "final_probabilities": final_probabilities,
        "fitted_expectations": fitted_expectations,
    }
