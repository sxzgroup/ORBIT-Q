"""Problem 7: 16-Qubit Measurement-Feedback VQE using TensorCircuit-NG."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _pick_chunk_size(n_traj: int, max_chunk: int = 8) -> int:
    for c in range(min(max_chunk, n_traj), 0, -1):
        if n_traj % c == 0:
            return c
    return 1


def _projective_measure_z(c: tc.Circuit, qubit: int, uniform: jax.Array, p0_proj: jax.Array, p1_proj: jax.Array):
    """Projective computational-basis measurement using TensorCircuit Circuit APIs."""
    z_exp = tc.backend.real(c.expectation_ps(z=[qubit], reuse=True))
    p0 = jnp.clip(0.5 * (1.0 + z_exp), 0.0, 1.0)
    outcome = jnp.where(uniform < p0, jnp.int32(0), jnp.int32(1))
    prob_m = jnp.where(outcome == 0, p0, 1.0 - p0)
    inv_norm = jax.lax.rsqrt(jnp.maximum(prob_m, jnp.float32(1e-12))).astype(jnp.complex64)
    c.conditional_gate(outcome, [p0_proj * inv_norm, p1_proj * inv_norm], qubit)
    return outcome


def run_solution(config):
    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config.get("n_qubits", n_data + n_anc))
    n_pairs = min(n_data, n_anc)
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    init_scale = float(config["initial_parameter_scale"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    h_field = float(config["transverse_field"])

    chunk_size = _pick_chunk_size(n_traj, max_chunk=8)
    n_chunks = n_traj // chunk_size

    rng = np.random.default_rng(seed)
    params_init = (
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_data)), dtype=jnp.float32),
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_anc)), dtype=jnp.float32),
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_pairs)), dtype=jnp.float32),
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_pairs)), dtype=jnp.float32),
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_pairs)), dtype=jnp.float32),
        jnp.array(rng.normal(0.0, init_scale, size=(n_layers, n_data)), dtype=jnp.float32),
    )
    uniforms = jnp.array(rng.uniform(0.0, 1.0, size=(n_traj, n_layers, n_anc)), dtype=jnp.float32)

    p0_proj = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)
    p1_proj = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
    data_indices = tuple(range(n_data))
    zero_data_bits = jnp.zeros((n_data,), dtype=jnp.int32)

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex64"),
        tc.runtime_contractor("plain-experimental", local_steps=2),
    ):
        def pre_measure_block(state, layer_params):
            theta_data, theta_anc, theta_ent, _, _, _ = layer_params
            c = tc.Circuit(n_qubits, inputs=state)
            for i in range(n_data):
                c.ry(i, theta=theta_data[i])
            for i in range(n_anc):
                c.ry(n_data + i, theta=theta_anc[i])
            for i in range(n_pairs):
                c.rzz(n_data + i, i, theta=theta_ent[i])
            for i in range(n_anc - 1):
                c.cnot(n_data + i, n_data + i + 1)
            return c.state()

        def measure_and_post_block(state, layer_params, layer_u):
            _, _, _, theta_fb0, theta_fb1, theta_post = layer_params
            outcomes = []
            for i in range(n_anc):
                c = tc.Circuit(n_qubits, inputs=state)
                m = _projective_measure_z(c, n_data + i, layer_u[i], p0_proj, p1_proj)
                outcomes.append(m)
                state = c.state()

            c = tc.Circuit(n_qubits, inputs=state)
            for i in range(n_pairs):
                c.conditional_gate(
                    outcomes[i],
                    [tc.gates.rzz_gate(theta=theta_fb0[i]), tc.gates.rzz_gate(theta=theta_fb1[i])],
                    n_data + i,
                    i,
                )
            for i in range(n_data - 1):
                c.cnot(i, i + 1)
            for i in range(n_data):
                c.rz(i, theta=theta_post[i])
            return c.state(), jnp.stack(outcomes)

        def full_layer(state, layer_params, layer_u):
            state = pre_measure_block(state, layer_params)
            return measure_and_post_block(state, layer_params, layer_u)

        def data_hamiltonian_energy(state, last_outcomes):
            c = tc.Circuit(n_qubits, inputs=state)
            traceout = jnp.concatenate([zero_data_bits, last_outcomes])
            data_state = c.projected_subsystem(traceout, left=data_indices)
            c_data = tc.Circuit(n_data, inputs=data_state)
            energy = jnp.float32(0.0)
            for i in range(n_data - 1):
                energy = energy - tc.backend.real(c_data.expectation_ps(z=[i, i + 1], reuse=True))
            for i in range(n_data):
                energy = energy - h_field * tc.backend.real(c_data.expectation_ps(x=[i], reuse=True))
            return energy

        def trajectory_energy(params, state0_pre, traj_u):
            lp0 = tuple(x[0] for x in params)
            state, last_outcomes = measure_and_post_block(state0_pre, lp0, traj_u[0])
            if n_layers > 1:
                rest_params = tuple(x[1:] for x in params)
                rest_u = traj_u[1:]

                def layer_scan(carry, xs):
                    st, _ = carry
                    st, ms = full_layer(st, xs[0], xs[1])
                    return (st, ms), None

                (state, last_outcomes), _ = jax.lax.scan(
                    layer_scan, (state, last_outcomes), (rest_params, rest_u)
                )
            return data_hamiltonian_energy(state, last_outcomes)

        def chunk_loss(params, state0_pre, u_chunk):
            energies = jax.vmap(trajectory_energy, in_axes=(None, None, 0))(params, state0_pre, u_chunk)
            return jnp.mean(energies), energies

        chunk_vg = jax.value_and_grad(chunk_loss, argnums=(0, 1), has_aux=True)

        def eval_all_energies(params, u_all):
            init_state = tc.Circuit(n_qubits).state()
            state0_pre = pre_measure_block(init_state, tuple(x[0] for x in params))
            u_chunks = u_all.reshape(n_chunks, chunk_size, n_layers, n_anc)
            _, e_chunks = jax.lax.scan(
                lambda _, uc: (None, jax.vmap(trajectory_energy, in_axes=(None, None, 0))(params, state0_pre, uc)),
                None,
                u_chunks,
            )
            return e_chunks.reshape(n_traj)

        def eval_value_and_grad(params, u_all):
            init_state = tc.Circuit(n_qubits).state()
            state0_pre, vjp_s0 = jax.vjp(
                lambda p: pre_measure_block(init_state, tuple(x[0] for x in p)), params
            )
            u_chunks = u_all.reshape(n_chunks, chunk_size, n_layers, n_anc)
            gp_init = jax.tree_util.tree_map(jnp.zeros_like, params)
            gs0_init = jnp.zeros_like(state0_pre)

            def scan_chunk(carry, uc):
                gp_acc, gs0_acc = carry
                (_, es), (gp, gs0) = chunk_vg(params, state0_pre, uc)
                gp_acc = jax.tree_util.tree_map(lambda a, b: a + b, gp_acc, gp)
                return (gp_acc, gs0_acc + gs0), es

            (gp_sum, gs0_sum), e_chunks = jax.lax.scan(scan_chunk, (gp_init, gs0_init), u_chunks)
            gp_mean = jax.tree_util.tree_map(lambda a: a / n_chunks, gp_sum)
            (gp_from_s0,) = vjp_s0(gs0_sum / n_chunks)
            grads = jax.tree_util.tree_map(lambda a, b: a + b, gp_mean, gp_from_s0)
            energies = e_chunks.reshape(n_traj)
            return jnp.mean(energies), grads

        optimizer = optax.adam(lr)

        @jax.jit
        def optimize(p0, u_all):
            opt_state = optimizer.init(p0)

            def step(carry, _):
                p, os = carry
                e_avg, grads = eval_value_and_grad(p, u_all)
                updates, os = optimizer.update(grads, os, p)
                p = optax.apply_updates(p, updates)
                return (p, os), e_avg

            (final_p, _), history = jax.lax.scan(step, (p0, opt_state), None, length=max_steps)
            final_energies = eval_all_energies(final_p, u_all)
            return history, final_energies

        history, final_energies = optimize(params_init, uniforms)
        history_np = np.asarray(jax.device_get(history), dtype=np.float64)
        final_energies_np = np.asarray(jax.device_get(final_energies), dtype=np.float64)

    return {
        "energy_history": history_np,
        "final_trajectory_energies": final_energies_np,
    }
