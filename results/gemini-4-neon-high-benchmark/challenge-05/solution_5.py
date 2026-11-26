"""Problem 5: Custom Non-Unitary Gate Cooling using TensorCircuit-NG."""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize a variational non-unitary cooling circuit for the 1D open TFIM."""
    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    initial_filter_strength = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    if n_layers % 2 != 0:
        raise ValueError("n_layers must be even for even+odd brickwork blocks.")
    n_blocks = n_layers // 2

    with (
        tc.runtime_backend("jax") as K,
        tc.runtime_dtype("complex128"),
        tc.runtime_contractor("plain-experimental", local_steps=0),
    ):
        init_circuit = tc.Circuit(n_qubits)
        for i in range(n_qubits):
            init_circuit.h(i)
        psi0 = init_circuit.state()

        pauli_strings = []
        weights = []
        for i in range(n_qubits - 1):
            term = [0] * n_qubits
            term[i] = 3
            term[i + 1] = 3
            pauli_strings.append(term)
            weights.append(-1.0)
        for i in range(n_qubits):
            term = [0] * n_qubits
            term[i] = 1
            pauli_strings.append(term)
            weights.append(-transverse_field)
        hamiltonian_mvp = tc.quantum.PauliStringSum2MVP(pauli_strings, weights)

        i1 = tc.array_to_tensor(tc.gates._i_matrix)
        x1 = tc.array_to_tensor(tc.gates._x_matrix)
        i2 = tc.array_to_tensor(tc.gates._ii_matrix)
        zz2 = tc.array_to_tensor(tc.gates._zz_matrix)

        def _apply_sublayer(state: Any, a_val: Any, b_val: Any, start_bond: int) -> Any:
            c = tc.Circuit(n_qubits, inputs=state)
            g1 = K.cosh(a_val) * i1 + K.sinh(a_val) * x1
            g2 = K.cosh(b_val) * i2 + K.sinh(b_val) * zz2
            for i in range(n_qubits):
                c.any(i, unitary=g1)
            for i in range(start_bond, n_qubits - 1, 2):
                c.any(i, i + 1, unitary=g2)
            out_state = c.state()
            return out_state / K.norm(out_state)

        apply_sublayer = jax.checkpoint(_apply_sublayer, static_argnums=(3,))

        def energy_density(params: Any) -> Any:
            a_arr, b_arr = params

            def block_step(state: Any, block_params: Any) -> Any:
                a_blk, b_blk = block_params
                state = apply_sublayer(state, a_blk[0], b_blk[0], 0)
                state = apply_sublayer(state, a_blk[1], b_blk[1], 1)
                return state, None

            final_state, _ = jax.lax.scan(block_step, psi0, (a_arr, b_arr))
            energy = K.real(K.sum(K.conj(final_state) * hamiltonian_mvp(final_state)))
            return energy / n_qubits

        optimizer = optax.adam(learning_rate)
        value_and_grad = jax.value_and_grad(energy_density)

        @jax.jit
        def train(a_init: Any, b_init: Any) -> Any:
            params = (a_init, b_init)
            opt_state = optimizer.init(params)

            def update_step(carry: Any, _: Any) -> Any:
                cur_params, cur_opt_state = carry
                loss, grads = value_and_grad(cur_params)
                updates, next_opt_state = optimizer.update(
                    grads, cur_opt_state, cur_params
                )
                next_params = optax.apply_updates(cur_params, updates)
                return (next_params, next_opt_state), loss

            (final_params, _), history = jax.lax.scan(
                update_step, (params, opt_state), None, length=max_steps
            )
            return final_params[0], final_params[1], history

        a0 = jnp.full((n_blocks, 2), initial_filter_strength, dtype=jnp.float64)
        b0 = jnp.full((n_blocks, 2), initial_filter_strength, dtype=jnp.float64)
        final_a, final_b, history = train(a0, b0)
        history.block_until_ready()

    return {
        "final_a": np.asarray(final_a, dtype=np.float64),
        "final_b": np.asarray(final_b, dtype=np.float64),
        "energy_density_history": np.asarray(history, dtype=np.float64),
    }
