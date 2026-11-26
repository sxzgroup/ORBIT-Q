"""Variational post-selected many-body cooling using TensorCircuit-NG (JAX)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _build_tfim_hamiltonian(n_qubits: int, transverse_field: float):
    """Build sparse COO TFIM Hamiltonian via TensorCircuit PauliStringSum2COO."""
    pauli_strings = []
    weights = []
    for i in range(n_qubits - 1):
        ps = [0] * n_qubits
        ps[i] = 3
        ps[i + 1] = 3
        pauli_strings.append(ps)
        weights.append(-1.0)
    for i in range(n_qubits):
        ps = [0] * n_qubits
        ps[i] = 1
        pauli_strings.append(ps)
        weights.append(-float(transverse_field))
    return tc.quantum.PauliStringSum2COO(pauli_strings, weight=weights, numpy=False)


def _apply_cooling_step(state, xx_params, zz_params, rx_params, bonds, even_qubits, n_qubits):
    """Apply one brickwork cooling step and sequential Z=+1 post-selections."""
    c = tc.Circuit(n_qubits, inputs=state)
    for idx, (i, j) in enumerate(bonds):
        c.rxx(i, j, theta=2.0 * xx_params[idx])
        c.rzz(i, j, theta=2.0 * zz_params[idx])
    for q in range(n_qubits):
        c.rx(q, theta=rx_params[q])
    state = c.state()

    event_probs = []
    for q in even_qubits:
        c_meas = tc.Circuit(n_qubits, inputs=state)
        c_meas.mid_measurement(q, keep=0)
        unnorm = c_meas.state()
        p_event = jnp.real(jnp.vdot(unnorm, unnorm))
        event_probs.append(p_event)
        state = unnorm / jnp.sqrt(p_event + 1e-30)
    return state, jnp.stack(event_probs)


def run_solution(config):
    """Optimize the post-selected variational cooling circuit and return histories."""
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    log_prob_weight = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    h_sparse = _build_tfim_hamiltonian(n_qubits, transverse_field)

    c_init = tc.Circuit(n_qubits)
    for q in range(n_qubits):
        c_init.h(q)
    init_state = c_init.state()

    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    even_qubits = list(range(0, n_qubits, 2))
    n_blocks = n_steps // 2
    has_tail = (n_steps % 2) == 1

    def forward(params):
        even_xx, even_zz, even_rx, odd_xx, odd_zz, odd_rx, tail_xx, tail_zz, tail_rx = params
        state = init_state
        prob_chunks = []

        if n_blocks > 0:
            def block_fn(carry_state, block_params):
                exx, ezz, erx, oxx, ozz, orx = block_params
                next_state, pe = _apply_cooling_step(
                    carry_state, exx, ezz, erx, even_bonds, even_qubits, n_qubits
                )
                next_state, po = _apply_cooling_step(
                    next_state, oxx, ozz, orx, odd_bonds, even_qubits, n_qubits
                )
                return next_state, jnp.concatenate([pe, po])

            state, block_probs = jax.lax.scan(
                block_fn, state, (even_xx, even_zz, even_rx, odd_xx, odd_zz, odd_rx)
            )
            prob_chunks.append(block_probs.reshape(-1))

        if has_tail:
            state, pt = _apply_cooling_step(
                state, tail_xx, tail_zz, tail_rx, even_bonds, even_qubits, n_qubits
            )
            prob_chunks.append(pt)

        all_probs = jnp.concatenate(prob_chunks)
        c_final = tc.Circuit(n_qubits, inputs=state)
        energy = tc.templates.measurements.sparse_expectation(c_final, h_sparse)
        energy_density = energy / n_qubits
        mean_log_prob = jnp.mean(jnp.log(all_probs + 1e-12))
        success_prob = jnp.prod(all_probs)
        loss = energy_density - log_prob_weight * mean_log_prob
        return loss, (energy_density, success_prob, mean_log_prob)

    rng = np.random.default_rng(42)
    params = (
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, len(even_bonds))), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, len(even_bonds))), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, n_qubits)), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, len(odd_bonds))), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, len(odd_bonds))), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_blocks, n_qubits)), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (len(even_bonds),)), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (len(even_bonds),)), dtype=jnp.float64),
        jnp.asarray(rng.normal(0.0, 0.1, (n_qubits,)), dtype=jnp.float64),
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    value_and_grad_fn = jax.value_and_grad(forward, has_aux=True)

    @jax.jit
    def train(init_params, init_opt_state):
        def update_step(carry, _):
            cur_params, cur_opt_state = carry
            (loss_val, (ed_val, sp_val, mlp_val)), grads = value_and_grad_fn(cur_params)
            updates, next_opt_state = optimizer.update(grads, cur_opt_state, cur_params)
            next_params = optax.apply_updates(cur_params, updates)
            return (next_params, next_opt_state), (ed_val, sp_val, mlp_val, loss_val)

        _, histories = jax.lax.scan(
            update_step, (init_params, init_opt_state), None, length=max_steps
        )
        return histories

    ed_hist, sp_hist, mlp_hist, loss_hist = train(params, opt_state)
    return {
        "energy_density_history": np.asarray(ed_hist, dtype=np.float64),
        "success_probability_history": np.asarray(sp_hist, dtype=np.float64),
        "mean_log_probability_history": np.asarray(mlp_hist, dtype=np.float64),
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
    }
