import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

with tc.runtime_backend("numpy"):
    tc.set_dtype("complex128")
    tc.Circuit._meta_apply()
tc.set_backend("jax")
tc.set_contractor("omeco-8-16", use_primitives=True)


def _apply_cooling_layer(psi, a_l, b_l, n_qubits, parity):
    c = tc.Circuit(n_qubits, inputs=psi)
    theta_x = 1.0j * tc.backend.cast(a_l, "complex128")
    theta_zz = 1.0j * tc.backend.cast(b_l, "complex128")
    for i in range(n_qubits):
        c.exp1(i, unitary=tc.gates._x_matrix, theta=theta_x)
    for i in range(parity, n_qubits - 1, 2):
        c.exp1(i, i + 1, unitary=tc.gates._zz_matrix, theta=theta_zz)
    psi_next = c.state()
    return psi_next / tc.backend.norm(psi_next)


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    transverse_field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_strength = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    c_init = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c_init.h(i)
    psi0 = c_init.state()
    psi0 = psi0 / tc.backend.norm(psi0)

    lattice_graph = tc.templates.graphs.Line1D(n_qubits, pbc=False)
    h_np_coo = tc.quantum.heisenberg_hamiltonian(
        lattice_graph,
        hzz=-1.0,
        hxx=0.0,
        hyy=0.0,
        hx=-transverse_field,
        numpy=True,
    )
    hamiltonian = tc.backend.sparse_csr_from_coo(
        tc.backend.coo_sparse_matrix_from_numpy(h_np_coo)
    )

    n_blocks = n_layers // 2
    has_rem = (n_layers % 2) == 1

    def loss_fn(params):
        a_blocks, b_blocks, a_rem, b_rem = params

        def block_fn(psi, ab_pair):
            a_pair, b_pair = ab_pair
            psi = _apply_cooling_layer(psi, a_pair[0], b_pair[0], n_qubits, 0)
            psi = _apply_cooling_layer(psi, a_pair[1], b_pair[1], n_qubits, 1)
            return psi, None

        psi_final, _ = jax.lax.scan(block_fn, psi0, (a_blocks, b_blocks))
        if has_rem:
            psi_final = _apply_cooling_layer(psi_final, a_rem[0], b_rem[0], n_qubits, 0)

        c_final = tc.Circuit(n_qubits, inputs=psi_final)
        energy = tc.templates.measurements.operator_expectation(c_final, hamiltonian)
        return tc.backend.real(energy) / n_qubits

    optimizer = optax.adam(learning_rate)
    a0 = jnp.full((n_blocks, 2), init_strength, dtype=jnp.float64)
    b0 = jnp.full((n_blocks, 2), init_strength, dtype=jnp.float64)
    rem_len = 1 if has_rem else 0
    a_rem0 = jnp.full((rem_len,), init_strength, dtype=jnp.float64)
    b_rem0 = jnp.full((rem_len,), init_strength, dtype=jnp.float64)
    init_params = (a0, b0, a_rem0, b_rem0)
    init_opt_state = optimizer.init(init_params)

    val_and_grad = jax.value_and_grad(loss_fn)

    def step_fn(carry, _):
        params, opt_state = carry
        loss, grads = val_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def optimize(params, opt_state):
        (final_params, _), history = jax.lax.scan(
            step_fn, (params, opt_state), None, length=max_steps
        )
        return final_params, history

    (final_a, final_b, _, _), energy_history = optimize(init_params, init_opt_state)

    return {
        "final_a": np.asarray(final_a, dtype=np.float64),
        "final_b": np.asarray(final_b, dtype=np.float64),
        "energy_density_history": np.asarray(energy_history, dtype=np.float64),
    }
