"""Calibrate a custom Kraus channel using exact tensor-network expectations."""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy", preprocessing=True)
    n = int(config["n_qubits"])
    angle = float(config["entangler_angle"])
    steps = int(config["max_steps"])

    # Each row specifies either one Z or the full-chain Z parity.
    masks = np.concatenate([np.eye(n), np.ones((1, n))], axis=0)
    operators = jnp.asarray(
        (1 - masks[..., None, None]) * np.eye(2)
        + masks[..., None, None] * np.diag([1., -1.]),
        dtype=jnp.complex128,
    )

    def expectations(probabilities):
        p01, p10 = probabilities
        kraus = [
            jnp.diag(jnp.sqrt(1 - probabilities)),
            jnp.sqrt(p10) * jnp.array([[0., 1.], [0., 0.]]),
            jnp.sqrt(p01) * jnp.array([[0., 0.], [1., 0.]]),
        ]
        rows = []
        for probe in range(4):
            c = tc.DMCircuit2(n)
            if probe == 0:
                c.h(0)
                for q in range(n - 1):
                    c.cnot(q, q + 1)
            elif probe == 1:
                for q in range(0, n, 2):
                    c.h(q)
                    c.x(q + 1)
                    c.cnot(q, q + 1)
            elif probe == 3:
                for q in range(n):
                    c.h(q)
            for offset in (0, 1):
                for q in range(offset, n - 1, 2):
                    c.rxx(q, q + 1, theta=angle)
                    c.general_kraus(kraus, q)
                    c.general_kraus(kraus, q + 1)

            def measure(ops):
                return jnp.real(c.expectation(
                    *[(ops[q], [q]) for q in range(n)], reuse=False
                ))

            rows.append(jax.vmap(measure)(operators))
        return jnp.stack(rows)

    predict = jax.jit(expectations)
    target = predict(jnp.array([config["[REDACTED]_p01"], config["[REDACTED]_p10"]]))
    initial = jnp.array([config["initial_p01"], config["initial_p10"]])
    logits = jnp.log(initial) - jnp.log1p(-initial)
    optimizer = optax.adam(float(config["learning_rate"]))

    def loss(r):
        return jnp.mean((predict(jax.nn.sigmoid(r)) - target) ** 2)

    def update(carry, _):
        r, state = carry
        value, gradient = jax.value_and_grad(loss)(r)
        updates, state = optimizer.update(gradient, state, r)
        return (optax.apply_updates(r, updates), state), value

    @jax.jit
    def train(r):
        return jax.lax.scan(update, (r, optimizer.init(r)), None, length=steps)

    (logits, _), history = train(logits)
    final = jax.nn.sigmoid(logits)
    fitted = predict(final)
    return {
        "loss_history": np.asarray(history),
        "final_probabilities": np.asarray(final),
        "fitted_expectations": np.asarray(fitted),
    }
