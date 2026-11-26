import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")
K = tc.backend


def _build_tfim_hamiltonian(n_qubits: int, transverse_field: float):
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        row = [0] * n_qubits
        row[i] = 3
        row[i + 1] = 3
        structures.append(row)
        weights.append(-1.0)
    for i in range(n_qubits):
        row = [0] * n_qubits
        row[i] = 1
        structures.append(row)
        weights.append(-transverse_field)
    return tc.quantum.PauliStringSum2COO(structures, weights)


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    log_prob_weight = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    if max_steps <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "energy_density_history": empty.copy(),
            "success_probability_history": empty.copy(),
            "mean_log_probability_history": empty.copy(),
            "loss_history": empty.copy(),
        }

    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    meas_qubits = list(range(0, n_qubits, 2))
    ham_sparse = _build_tfim_hamiltonian(n_qubits, transverse_field)

    c_init = tc.Circuit(n_qubits)
    for q in range(n_qubits):
        c_init.apply(tc.gates.h(), q)
    psi0 = c_init.state()

    def apply_step(psi, bond_params, rx_params, bonds):
        c = tc.Circuit(n_qubits, inputs=psi)
        for b_idx, (i, j) in enumerate(bonds):
            c.rxx(i, j, theta=2.0 * bond_params[b_idx, 0])
            c.rzz(i, j, theta=2.0 * bond_params[b_idx, 1])
        for q in range(n_qubits):
            c.rx(q, theta=rx_params[q])
        psi = c.state()

        p_list = []
        log_p_list = []
        for q in meas_qubits:
            cm = tc.Circuit(n_qubits, inputs=psi)
            cm.post_select(q, keep=0)
            psi_proj = cm.state()
            p_ev = K.real(K.sum(K.conj(psi_proj) * psi_proj))
            p_list.append(p_ev)
            log_p_list.append(jnp.log(p_ev + 1e-12))
            norm = K.cast(jnp.sqrt(jnp.maximum(p_ev, 1e-30)), psi_proj.dtype)
            psi = psi_proj / norm
        return psi, jnp.stack(p_list), jnp.stack(log_p_list)

    n_pairs = n_steps // 2
    has_rem = (n_steps % 2) == 1

    def loss_and_metrics(params):
        def pair_body(psi, pair_params):
            e_bond, e_rx, o_bond, o_rx = pair_params
            psi, p_e, lp_e = apply_step(psi, e_bond, e_rx, even_bonds)
            psi, p_o, lp_o = apply_step(psi, o_bond, o_rx, odd_bonds)
            return psi, (jnp.concatenate([p_e, p_o]), jnp.concatenate([lp_e, lp_o]))

        psi = psi0
        p_parts = []
        lp_parts = []
        if n_pairs > 0:
            psi, (p_scan, lp_scan) = jax.lax.scan(pair_body, psi, params["pairs"])
            p_parts.append(p_scan.reshape(-1))
            lp_parts.append(lp_scan.reshape(-1))
        if has_rem:
            e_bond, e_rx = params["rem"]
            psi, p_rem, lp_rem = apply_step(psi, e_bond, e_rx, even_bonds)
            p_parts.append(p_rem)
            lp_parts.append(lp_rem)

        if p_parts:
            p_events = jnp.concatenate(p_parts)
            log_p_events = jnp.concatenate(lp_parts)
            mean_log_prob = jnp.mean(log_p_events)
            success_prob = jnp.prod(p_events)
        else:
            mean_log_prob = jnp.asarray(0.0, dtype=jnp.float64)
            success_prob = jnp.asarray(1.0, dtype=jnp.float64)

        c_final = tc.Circuit(n_qubits, inputs=psi)
        energy = tc.templates.measurements.operator_expectation(c_final, ham_sparse)
        energy_density = energy / n_qubits
        loss = energy_density - log_prob_weight * mean_log_prob
        return loss, (energy_density, success_prob, mean_log_prob)

    rng = np.random.default_rng(42)
    init_scale = 0.05
    params = {}
    if n_pairs > 0:
        params["pairs"] = (
            jnp.asarray(rng.normal(0.0, init_scale, (n_pairs, len(even_bonds), 2))),
            jnp.asarray(rng.normal(0.0, init_scale, (n_pairs, n_qubits))),
            jnp.asarray(rng.normal(0.0, init_scale, (n_pairs, len(odd_bonds), 2))),
            jnp.asarray(rng.normal(0.0, init_scale, (n_pairs, n_qubits))),
        )
    if has_rem:
        params["rem"] = (
            jnp.asarray(rng.normal(0.0, init_scale, (len(even_bonds), 2))),
            jnp.asarray(rng.normal(0.0, init_scale, (n_qubits,))),
        )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    val_and_grad = jax.value_and_grad(loss_and_metrics, has_aux=True)

    def train_step(carry, _):
        curr_params, curr_opt_state = carry
        (loss, (ed, sp, mlp)), grads = val_and_grad(curr_params)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
        next_params = optax.apply_updates(curr_params, updates)
        return (next_params, next_opt_state), (ed, sp, mlp, loss)

    _, (ed_hist, sp_hist, mlp_hist, loss_hist) = jax.lax.scan(
        train_step, (params, opt_state), None, length=max_steps
    )

    return {
        "energy_density_history": np.asarray(ed_hist, dtype=np.float64),
        "success_probability_history": np.asarray(sp_hist, dtype=np.float64),
        "mean_log_probability_history": np.asarray(mlp_hist, dtype=np.float64),
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
    }
