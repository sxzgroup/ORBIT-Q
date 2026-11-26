"""Trainable asymmetric bit-flip Kraus channel calibration via TensorCircuit."""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def _asymmetric_bitflip_kraus(p01: Any, p10: Any) -> List[tc.gates.Gate]:
    """Construct Kraus operators K0, K1, K2 for asymmetric bit-flip noise."""
    zero = jnp.asarray(0.0, dtype=jnp.complex128)
    s00 = jnp.sqrt(jnp.asarray(1.0 - p01, dtype=jnp.complex128))
    s11 = jnp.sqrt(jnp.asarray(1.0 - p10, dtype=jnp.complex128))
    s01 = jnp.sqrt(jnp.asarray(p10, dtype=jnp.complex128))
    s10 = jnp.sqrt(jnp.asarray(p01, dtype=jnp.complex128))
    k0 = jnp.array([[s00, zero], [zero, s11]], dtype=jnp.complex128)
    k1 = jnp.array([[zero, s01], [zero, zero]], dtype=jnp.complex128)
    k2 = jnp.array([[zero, zero], [s10, zero]], dtype=jnp.complex128)
    return [tc.gates.Gate(k0), tc.gates.Gate(k1), tc.gates.Gate(k2)]


def _prepare_probe_circuit(n_qubits: int, probe_idx: int) -> tc.DMCircuit2:
    """Create a 12-qubit (or n-qubit) DMCircuit2 initialized in one of the 4 probe states."""
    c = tc.DMCircuit2(n_qubits)
    if probe_idx == 0:
        c.apply_general_gate(tc.gates.h(), 0)
        for i in range(n_qubits - 1):
            c.apply_general_gate(tc.gates.cnot(), i, i + 1)
    elif probe_idx == 1:
        for i in range(0, n_qubits - 1, 2):
            c.apply_general_gate(tc.gates.x(), i + 1)
            c.apply_general_gate(tc.gates.h(), i)
            c.apply_general_gate(tc.gates.cnot(), i, i + 1)
    elif probe_idx == 3:
        for i in range(n_qubits):
            c.apply_general_gate(tc.gates.h(), i)
    return c


def _apply_noisy_brickwork(
    c: tc.DMCircuit2,
    n_qubits: int,
    angle: Any,
    kraus: List[tc.gates.Gate],
) -> None:
    """Apply even-bond then odd-bond RXX entanglers followed by per-qubit Kraus noise."""
    for start in (0, 1):
        for i in range(start, n_qubits - 1, 2):
            c.apply_general_gate(tc.gates.rxx(theta=angle), i, i + 1)
            c.apply_general_kraus(kraus, i)
            c.apply_general_kraus(kraus, i + 1)


def _evaluate_expectations(
    n_qubits: int, angle: Any, p01: Any, p10: Any
) -> jax.Array:
    """Compute shape (4, n_qubits + 1) table of single-site Z_i and full parity expectations."""
    kraus = _asymmetric_bitflip_kraus(p01, p10)
    rows = []
    for probe_idx in range(4):
        c = _prepare_probe_circuit(n_qubits, probe_idx)
        _apply_noisy_brickwork(c, n_qubits, angle, kraus)
        exps = [
            jnp.real(c.expectation((tc.gates.z(), [q]), reuse=False))
            for q in range(n_qubits)
        ]
        parity_ops = [(tc.gates.z(), [q]) for q in range(n_qubits)]
        exps.append(jnp.real(c.expectation(*parity_ops, reuse=False)))
        rows.append(jnp.stack(exps))
    return jnp.stack(rows)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Generate synthetic noisy observables and fit channel probabilities via Adam."""
    n_qubits = int(config["n_qubits"])
    angle = jnp.asarray(config["entangler_angle"], dtype=jnp.float64)
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    eval_fn = jax.jit(
        lambda p01, p10: _evaluate_expectations(n_qubits, angle, p01, p10)
    )
    target_table = eval_fn(
        jnp.asarray(config["true_p01"], dtype=jnp.float64),
        jnp.asarray(config["true_p10"], dtype=jnp.float64),
    )

    def loss_fn(logits: jax.Array) -> jax.Array:
        probs = jax.nn.sigmoid(logits)
        preds = _evaluate_expectations(n_qubits, angle, probs[0], probs[1])
        return jnp.mean((preds - target_table) ** 2)

    optimizer = optax.adam(learning_rate=lr)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def adam_step(
        logits: jax.Array, opt_state: optax.OptState
    ) -> Tuple[jax.Array, jax.Array, optax.OptState]:
        loss, grads = loss_and_grad(logits)
        updates, new_opt_state = optimizer.update(grads, opt_state, logits)
        new_logits = optax.apply_updates(logits, updates)
        return loss, new_logits, new_opt_state

    logits = jnp.array(
        [_logit(float(config["initial_p01"])), _logit(float(config["initial_p10"]))],
        dtype=jnp.float64,
    )
    opt_state = optimizer.init(logits)

    loss_history = np.empty(max_steps, dtype=np.float64)
    for step in range(max_steps):
        loss, logits, opt_state = adam_step(logits, opt_state)
        loss_history[step] = float(loss)

    final_probs_jax = jax.nn.sigmoid(logits)
    fitted_expectations = np.asarray(
        eval_fn(final_probs_jax[0], final_probs_jax[1]), dtype=np.float64
    )
    final_probabilities = np.asarray(final_probs_jax, dtype=np.float64)

    return {
        "loss_history": loss_history,
        "final_probabilities": final_probabilities,
        "fitted_expectations": fitted_expectations,
    }
