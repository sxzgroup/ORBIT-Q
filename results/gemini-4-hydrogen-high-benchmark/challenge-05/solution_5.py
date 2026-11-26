"""Variational non-unitary cooling circuit optimization using TensorCircuit."""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize a non-unitary cooling circuit for the 1D TFIM using TensorCircuit."""
    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_strength = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    if n_layers % 2 != 0:
        raise ValueError("n_layers must be even to form (n_layers // 2, 2) blocks.")
    n_blocks = n_layers // 2

    tc.set_backend("jax")
    tc.set_dtype("complex64")

    # Prepare initial |+>^n state through TensorCircuit.
    init_circ = tc.Circuit(n_qubits)
    init_circ.h(range(n_qubits))
    psi0 = init_circ.state()

    x_mat = tc.gates.x().tensor
    zz_mat = tc.gates._zz_matrix

    def apply_layer(psi: Any, a_val: Any, b_val: Any, bond_start: int) -> Any:
        circ = tc.Circuit(n_qubits, inputs=psi)
        g1 = tc.gates.exp1_gate(x_mat, 1j * a_val).tensor
        for i in range(n_qubits):
            circ.any(i, unitary=g1)
        g2 = tc.gates.exp1_gate(zz_mat, 1j * b_val).tensor
        for i in range(bond_start, n_qubits - 1, 2):
            circ.any(i, i + 1, unitary=g2)
        state = circ.state()
        return state / tc.backend.norm(state)

    def apply_block(psi: Any, block_params: Any) -> Any:
        a_even, a_odd, b_even, b_odd = block_params
        psi = apply_layer(psi, a_even, b_even, 0)
        psi = apply_layer(psi, a_odd, b_odd, 1)
        return psi, None

    def energy_density_loss(params: Any) -> Any:
        a, b = params
        psi, _ = jax.lax.scan(apply_block, psi0, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
        final_circ = tc.Circuit(n_qubits, inputs=psi)
        energy = 0.0
        for i in range(n_qubits - 1):
            energy = energy - final_circ.expectation_ps(z=[i, i + 1], reuse=True)
        for i in range(n_qubits):
            energy = energy - transverse_field * final_circ.expectation_ps(
                x=[i], reuse=True
            )
        return tc.backend.real(energy) / n_qubits

    init_a = jnp.full((n_blocks, 2), init_strength, dtype=jnp.float32)
    init_b = jnp.full((n_blocks, 2), init_strength, dtype=jnp.float32)
    init_params = (init_a, init_b)

    optimizer = optax.adam(learning_rate)
    init_opt_state = optimizer.init(init_params)
    value_and_grad_fn = jax.value_and_grad(energy_density_loss)

    def update_step(carry: Any, _: Any) -> Any:
        params, opt_state = carry
        loss_val, grads = value_and_grad_fn(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return (next_params, next_opt_state), loss_val

    @jax.jit
    def optimize(params: Any, opt_state: Any) -> Any:
        (final_params, _), history = jax.lax.scan(
            update_step, (params, opt_state), None, length=max_steps
        )
        return final_params, history

    (final_a, final_b), history = optimize(init_params, init_opt_state)

    return {
        "final_a": np.asarray(final_a, dtype=np.float64),
        "final_b": np.asarray(final_b, dtype=np.float64),
        "energy_density_history": np.asarray(history, dtype=np.float64),
    }
