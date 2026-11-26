import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _build_pauli_terms(n_qubits):
    xy_ps, xy_w = [], []
    stag_ps, stag_w = [], []
    tgt_ps, tgt_w = [], []

    for i in range(n_qubits - 1):
        for pauli in (1, 2):
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            xy_ps.append(term)
            xy_w.append(1.0)
        for pauli, weight in ((1, 0.7), (2, 0.7), (3, 1.1)):
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            tgt_ps.append(term)
            tgt_w.append(weight)

    for i in range(n_qubits):
        sign = 1.0 if i % 2 == 0 else -1.0
        term = [0] * n_qubits
        term[i] = 3
        stag_ps.append(term)
        stag_w.append(sign)
        tgt_ps.append(term)
        tgt_w.append(0.25 * sign)

    return (xy_ps, xy_w), (stag_ps, stag_w), (tgt_ps, tgt_w)


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    seed = int(config.get("seed", 42))

    tc.set_backend("jax")
    tc.set_dtype("complex128")

    (xy_ps, xy_w), (stag_ps, stag_w), (tgt_ps, tgt_w) = _build_pauli_terms(n_qubits)
    mvp_xy = tc.quantum.PauliStringSum2MVP(xy_ps, xy_w)
    mvp_stag = tc.quantum.PauliStringSum2MVP(stag_ps, stag_w)
    h_tgt_coo = tc.quantum.PauliStringSum2COO(tgt_ps, tgt_w, numpy=True)
    h_tgt_sparse = tc.backend.sparse_csr_from_coo(
        tc.backend.coo_sparse_matrix_from_numpy(h_tgt_coo)
    )

    init_circuit = tc.Circuit(n_qubits)
    for k in range(1, n_qubits, 2):
        init_circuit.x(k)
    neel_state = init_circuit.state()

    dim = 1 << n_qubits
    zero_t = jnp.array(0.0, dtype=jnp.float64)

    def energy_density(params):
        t_vals = t_min + (t_max - t_min) * jax.nn.sigmoid(params["s"])
        j_vals = jnp.tanh(params["j"])
        d_vals = jnp.tanh(params["d"])

        def block_step(psi, block_inputs):
            t_l, j_l, d_l, angles_l = block_inputs

            def h_analog(_t):
                return tc.LinearOperator(
                    (dim, dim),
                    lambda v: j_l * mvp_xy(v) + d_l * mvp_stag(v),
                )

            times = jnp.stack([zero_t, t_l])
            psi = tc.timeevol.ode_evol_global(
                h_analog,
                psi,
                times,
                ode_backend="jaxode",
                rtol=ode_rtol,
                atol=ode_atol,
                max_steps=ode_max_steps,
            )[-1]

            digital_circuit = tc.Circuit(n_qubits, inputs=psi)
            for k in range(n_qubits):
                digital_circuit.rz(k, theta=angles_l[k, 2])
                digital_circuit.ry(k, theta=angles_l[k, 1])
                digital_circuit.rz(k, theta=angles_l[k, 0])
            return digital_circuit.state(), None

        final_state, _ = jax.lax.scan(
            block_step,
            neel_state,
            (t_vals, j_vals, d_vals, params["angles"]),
        )
        final_circuit = tc.Circuit(n_qubits, inputs=final_state)
        energy = tc.templates.measurements.sparse_expectation(
            final_circuit, h_tgt_sparse
        )
        return energy / n_qubits

    rng = np.random.default_rng(seed)
    init_params = {
        "s": jnp.zeros((n_blocks,), dtype=jnp.float64),
        "j": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "d": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "angles": jnp.asarray(
            rng.normal(0.0, 0.1, size=(n_blocks, n_qubits, 3)), dtype=jnp.float64
        ),
    }

    optimizer = optax.adam(learning_rate)
    value_and_grad_fn = jax.value_and_grad(energy_density)

    @jax.jit
    def optimize(params):
        opt_state = optimizer.init(params)

        def adam_step(carry, _):
            cur_params, cur_opt_state = carry
            val, grads = value_and_grad_fn(cur_params)
            updates, next_opt_state = optimizer.update(grads, cur_opt_state, cur_params)
            next_params = optax.apply_updates(cur_params, updates)
            return (next_params, next_opt_state), val

        (learned_params, _), history = jax.lax.scan(
            adam_step, (params, opt_state), None, length=max_steps
        )
        return learned_params, history

    final_params, energy_history = optimize(init_params)

    final_analog_times = np.asarray(
        t_min + (t_max - t_min) * jax.nn.sigmoid(final_params["s"]),
        dtype=np.float64,
    )
    final_analog_couplings = np.asarray(
        jnp.tanh(final_params["j"]),
        dtype=np.float64,
    )
    final_analog_detunings = np.asarray(
        jnp.tanh(final_params["d"]),
        dtype=np.float64,
    )
    energy_density_history = np.asarray(energy_history, dtype=np.float64)

    return {
        "final_analog_times": final_analog_times,
        "final_analog_couplings": final_analog_couplings,
        "final_analog_detunings": final_analog_detunings,
        "energy_density_history": energy_density_history,
    }
