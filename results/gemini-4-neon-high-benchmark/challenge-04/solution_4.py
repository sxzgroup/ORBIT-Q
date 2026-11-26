import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _asymmetric_bit_flip_kraus(p01, p10):
    dtype = tc.cons.dtypestr
    s1_01 = jnp.sqrt(1.0 - p01).astype(dtype)
    s1_10 = jnp.sqrt(1.0 - p10).astype(dtype)
    s_10 = jnp.sqrt(p10).astype(dtype)
    s_01 = jnp.sqrt(p01).astype(dtype)
    zero = jnp.zeros((), dtype=dtype)
    k0 = jnp.array([[s1_01, zero], [zero, s1_10]], dtype=dtype)
    k1 = jnp.array([[zero, s_10], [zero, zero]], dtype=dtype)
    k2 = jnp.array([[zero, zero], [s_01, zero]], dtype=dtype)
    return [tc.Gate(k0), tc.Gate(k1), tc.Gate(k2)]


def _build_probe_circuit(probe_idx, n_qubits, entangler_angle, kraus):
    c = tc.DMCircuit2(n_qubits)
    if probe_idx == 0:
        c.apply_general_gate(tc.gates.h_gate(), 0)
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)
    elif probe_idx == 1:
        for i in range(0, n_qubits - 1, 2):
            c.apply_general_gate(tc.gates.h_gate(), i)
            c.x(i + 1)
            c.cnot(i, i + 1)
    elif probe_idx == 3:
        for i in range(n_qubits):
            c.apply_general_gate(tc.gates.h_gate(), i)

    for i in range(0, n_qubits - 1, 2):
        c.rxx(i, i + 1, theta=entangler_angle)
        c.apply_general_kraus(kraus, i)
        c.apply_general_kraus(kraus, i + 1)

    for i in range(1, n_qubits - 1, 2):
        c.rxx(i, i + 1, theta=entangler_angle)
        c.apply_general_kraus(kraus, i)
        c.apply_general_kraus(kraus, i + 1)

    return c


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")

    n_qubits = int(config["n_qubits"])
    entangler_angle = float(config["entangler_angle"])
    true_probs = jnp.asarray(
        [float(config["true_p01"]), float(config["true_p10"])], dtype=jnp.float64
    )
    init_probs = jnp.asarray(
        [float(config["initial_p01"]), float(config["initial_p10"])], dtype=jnp.float64
    )
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    all_qubits = list(range(n_qubits))

    def probe_expectations(probs):
        kraus = _asymmetric_bit_flip_kraus(probs[0], probs[1])
        rows = []
        for probe_idx in range(4):
            c = _build_probe_circuit(probe_idx, n_qubits, entangler_angle, kraus)
            exps = [
                jnp.real(c.expectation_ps(z=[i], reuse=False))
                for i in range(n_qubits)
            ]
            exps.append(jnp.real(c.expectation_ps(z=all_qubits, reuse=False)))
            rows.append(jnp.stack(exps))
        return jnp.stack(rows)

    eye2 = jnp.eye(2, dtype=jnp.float64)

    @jax.jit
    def exps_and_jac(probs):
        def single_jvp(tangent):
            return jax.jvp(probe_expectations, (probs,), (tangent,))

        exps_batch, jac = jax.vmap(single_jvp)(eye2)
        return exps_batch[0], jac

    target_expectations, _ = exps_and_jac(true_probs)

    optimizer = optax.adam(learning_rate=learning_rate)
    params = jnp.log(init_probs / (1.0 - init_probs))
    opt_state = optimizer.init(params)

    @jax.jit
    def adam_step(params, opt_state, probs, preds, jac, target):
        diff = preds - target
        loss = jnp.mean(diff * diff)
        grad_probs = jnp.mean(2.0 * diff[None, :, :] * jac, axis=(1, 2))
        grads = grad_probs * probs * (1.0 - probs)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for _ in range(max_steps):
        probs = jax.nn.sigmoid(params)
        preds, jac = exps_and_jac(probs)
        params, opt_state, loss = adam_step(
            params, opt_state, probs, preds, jac, target_expectations
        )
        losses.append(loss)

    final_probs = jax.nn.sigmoid(params)
    fitted_expectations, _ = exps_and_jac(final_probs)

    return {
        "loss_history": np.asarray(jnp.stack(losses), dtype=np.float64),
        "final_probabilities": np.asarray(final_probs, dtype=np.float64),
        "fitted_expectations": np.asarray(fitted_expectations, dtype=np.float64),
    }
