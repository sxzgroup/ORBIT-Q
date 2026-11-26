import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    log_prob_weight = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    meas_qubits = list(range(0, n_qubits, 2))

    # Prepare initial state |+>^{\otimes n_qubits} via TensorCircuit.
    init_circuit = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        init_circuit.h(i)
    init_state = init_circuit.state()
    init_state = init_state / tc.backend.norm(init_state)

    def apply_cooling_step(state, theta_xx, theta_zz, theta_rx, bonds):
        c = tc.Circuit(n_qubits, inputs=state)
        for k, (i, j) in enumerate(bonds):
            # [XX, ZZ] = 0 => exp[-i(theta_xx XX + theta_zz ZZ)] = RXX(2*theta_xx) RZZ(2*theta_zz)
            c.rxx(i, j, theta=2.0 * theta_xx[k])
            c.rzz(i, j, theta=2.0 * theta_zz[k])
        for i in range(n_qubits):
            c.rx(i, theta=theta_rx[i])
        psi = c.state()

        event_probs = []
        event_log_probs = []
        for q in meas_qubits:
            cm = tc.Circuit(n_qubits, inputs=psi)
            cm.post_select(q, keep=0)
            psi_proj = cm.state()
            p_event = jnp.maximum(jnp.real(jnp.vdot(psi_proj, psi_proj)), 0.0)
            event_probs.append(p_event)
            event_log_probs.append(jnp.log(p_event + 1e-12))
            psi = psi_proj / jnp.sqrt(jnp.maximum(p_event, 1e-30))
        return psi, jnp.stack(event_probs), jnp.stack(event_log_probs)

    def paired_block(state, block_params):
        e_xx, e_zz, e_rx, o_xx, o_zz, o_rx = block_params
        state, p_e, lp_e = apply_cooling_step(state, e_xx, e_zz, e_rx, even_bonds)
        state, p_o, lp_o = apply_cooling_step(state, o_xx, o_zz, o_rx, odd_bonds)
        return state, (jnp.concatenate([p_e, p_o]), jnp.concatenate([lp_e, lp_o]))

    n_pairs = n_steps // 2
    has_tail = (n_steps % 2) == 1

    def evaluate_objective(params):
        pair_params, tail_params = params
        state = init_state
        prob_chunks = []
        log_prob_chunks = []

        if n_pairs > 0:
            state, (pair_p, pair_lp) = jax.lax.scan(paired_block, state, pair_params)
            prob_chunks.append(pair_p.reshape(-1))
            log_prob_chunks.append(pair_lp.reshape(-1))

        if has_tail:
            t_xx, t_zz, t_rx = tail_params
            state, p_t, lp_t = apply_cooling_step(state, t_xx, t_zz, t_rx, even_bonds)
            prob_chunks.append(p_t)
            log_prob_chunks.append(lp_t)

        all_probs = jnp.concatenate(prob_chunks)
        all_log_probs = jnp.concatenate(log_prob_chunks)

        cf = tc.Circuit(n_qubits, inputs=state)
        energy = 0.0
        for i in range(n_qubits - 1):
            energy = energy - cf.expectation(
                (tc.gates.z(), [i]), (tc.gates.z(), [i + 1]), reuse=True
            )
        for i in range(n_qubits):
            energy = energy - transverse_field * cf.expectation(
                (tc.gates.x(), [i]), reuse=True
            )
        energy_density = jnp.real(energy) / n_qubits
        mean_log_prob = jnp.mean(all_log_probs)
        success_prob = jnp.prod(all_probs)
        loss = energy_density - log_prob_weight * mean_log_prob
        return loss, (energy_density, success_prob, mean_log_prob)

    rng = np.random.default_rng(42)
    pair_params = (
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, len(even_bonds))), dtype=jnp.float64),
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, len(even_bonds))), dtype=jnp.float64),
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, n_qubits)), dtype=jnp.float64),
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, len(odd_bonds))), dtype=jnp.float64),
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, len(odd_bonds))), dtype=jnp.float64),
        jnp.array(rng.normal(0.0, 0.1, (n_pairs, n_qubits)), dtype=jnp.float64),
    )
    if has_tail:
        tail_params = (
            jnp.array(rng.normal(0.0, 0.1, (len(even_bonds),)), dtype=jnp.float64),
            jnp.array(rng.normal(0.0, 0.1, (len(even_bonds),)), dtype=jnp.float64),
            jnp.array(rng.normal(0.0, 0.1, (n_qubits,)), dtype=jnp.float64),
        )
    else:
        tail_params = ()
    params = (pair_params, tail_params)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    value_and_grad_fn = jax.value_and_grad(evaluate_objective, has_aux=True)

    def train_step(carry, _):
        current_params, current_opt_state = carry
        (loss, (energy_density, success_prob, mean_log_prob)), grads = (
            value_and_grad_fn(current_params)
        )
        updates, next_opt_state = optimizer.update(
            grads, current_opt_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return (next_params, next_opt_state), (
            energy_density,
            success_prob,
            mean_log_prob,
            loss,
        )

    @jax.jit
    def optimize(init_params, init_opt_state):
        _, history = jax.lax.scan(
            train_step, (init_params, init_opt_state), None, length=max_steps
        )
        return history

    ed_hist, sp_hist, mlp_hist, loss_hist = optimize(params, opt_state)

    return {
        "energy_density_history": np.asarray(ed_hist, dtype=np.float64),
        "success_probability_history": np.asarray(sp_hist, dtype=np.float64),
        "mean_log_probability_history": np.asarray(mlp_hist, dtype=np.float64),
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
    }
