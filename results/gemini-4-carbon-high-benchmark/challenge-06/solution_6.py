import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def _build_operators(n_qubits: int):
    ls_xy, ws_xy = [], []
    ls_zz, ws_zz = [], []
    for i in range(n_qubits - 1):
        for p in (1, 2):
            row = [0] * n_qubits
            row[i] = p
            row[i + 1] = p
            ls_xy.append(row)
            ws_xy.append(1.0)
        row_zz = [0] * n_qubits
        row_zz[i] = 3
        row_zz[i + 1] = 3
        ls_zz.append(row_zz)
        ws_zz.append(1.1)

    ls_z, ws_z = [], []
    for i in range(n_qubits):
        row_z = [0] * n_qubits
        row_z[i] = 3
        ls_z.append(row_z)
        ws_z.append(float((-1) ** i))

    coo_xy = tc.quantum.PauliStringSum2COO_numpy(ls_xy, ws_xy)
    coo_z = tc.quantum.PauliStringSum2COO_numpy(ls_z, ws_z)
    coo_zz = tc.quantum.PauliStringSum2COO_numpy(ls_zz, ws_zz)
    coo_target = (0.7 * coo_xy + coo_zz + 0.25 * coo_z).tocoo()

    to_bcsr = lambda m: tc.backend.sparse_csr_from_coo(
        tc.backend.coo_sparse_matrix_from_numpy(m)
    )
    return to_bcsr(coo_xy), to_bcsr(coo_z), to_bcsr(coo_target)


def run_solution(config):
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    csr_xy, csr_z, csr_target = _build_operators(n_qubits)
    dim = 1 << n_qubits

    neel_circuit = tc.Circuit(n_qubits)
    for q in range(1, n_qubits, 2):
        neel_circuit.x(q)
    psi_init = neel_circuit.state()

    def make_analog_hamiltonian(J_val, D_val):
        return tc.quantum.LinearOperator(
            (dim, dim),
            lambda v: (
                J_val * tc.backend.sparse_dense_matmul(csr_xy, v)
                + D_val * tc.backend.sparse_dense_matmul(csr_z, v)
            ),
            dtype="complex128",
        )

    def hybrid_block(psi, block_params):
        s_l, j_l, d_l, rot_l = block_params
        t_l = t_min + (t_max - t_min) * jax.nn.sigmoid(s_l)
        J_l = jnp.tanh(j_l)
        D_l = jnp.tanh(d_l)

        ac = tc.AnalogCircuit(n_qubits, inputs=psi)
        ac.set_solver_options(
            ode_backend="jaxode",
            rtol=ode_rtol,
            atol=ode_atol,
            max_steps=ode_max_steps,
        )
        ac.add_analog_block(lambda _t: make_analog_hamiltonian(J_l, D_l), t_l)
        for k in range(n_qubits):
            ac.rz(k, theta=rot_l[k, 2])
            ac.ry(k, theta=rot_l[k, 1])
            ac.rz(k, theta=rot_l[k, 0])
        psi_next = ac.state()
        psi_next = psi_next / tc.backend.norm(psi_next)
        return psi_next, None

    def loss_fn(params):
        psi_final, _ = jax.lax.scan(
            hybrid_block,
            psi_init,
            (params["s"], params["j"], params["d"], params["rot"]),
        )
        eval_circuit = tc.Circuit(n_qubits, inputs=psi_final)
        energy = tc.templates.measurements.operator_expectation(
            eval_circuit, csr_target
        )
        return energy / n_qubits

    rng = np.random.default_rng(int(config.get("seed", 3)))
    init_params = {
        "s": jnp.zeros((n_blocks,), dtype=jnp.float64),
        "j": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "d": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "rot": jnp.asarray(
            rng.normal(0.0, 0.1, size=(n_blocks, n_qubits, 3)),
            dtype=jnp.float64,
        ),
    }

    optimizer = optax.adam(learning_rate)
    init_opt_state = optimizer.init(init_params)
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    def opt_step(carry, _):
        params, opt_state = carry
        loss, grads = value_and_grad_fn(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return (next_params, next_opt_state), loss

    @jax.jit
    def run_optimization(params, opt_state):
        return jax.lax.scan(opt_step, (params, opt_state), None, length=max_steps)

    (final_params, _), energy_history = run_optimization(init_params, init_opt_state)

    final_times = t_min + (t_max - t_min) * jax.nn.sigmoid(final_params["s"])
    final_couplings = jnp.tanh(final_params["j"])
    final_detunings = jnp.tanh(final_params["d"])

    return {
        "final_analog_times": np.asarray(final_times, dtype=np.float64),
        "final_analog_couplings": np.asarray(final_couplings, dtype=np.float64),
        "final_analog_detunings": np.asarray(final_detunings, dtype=np.float64),
        "energy_density_history": np.asarray(energy_history, dtype=np.float64),
    }
