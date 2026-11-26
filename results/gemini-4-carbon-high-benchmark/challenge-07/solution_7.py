import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
import tensornetwork as tn

tc.set_backend("jax")
tc.set_dtype("complex128")
tc.set_contractor("greedy", preprocessing=True)


def _build_data_hamiltonian(n_data: int, transverse_field: float) -> jnp.ndarray:
    pauli_strings = []
    weights = []
    for i in range(n_data - 1):
        ps = [0] * n_data
        ps[i] = 3
        ps[i + 1] = 3
        pauli_strings.append(ps)
        weights.append(-1.0)
    for i in range(n_data):
        ps = [0] * n_data
        ps[i] = 1
        pauli_strings.append(ps)
        weights.append(-float(transverse_field))
    h_dense = tc.quantum.PauliStringSum2Dense(pauli_strings, weights, numpy=True)
    return jnp.asarray(h_dense, dtype=jnp.complex128)


def _make_quvector(psi_data: jnp.ndarray, anc_bits: jnp.ndarray, n_data: int, n_anc: int):
    data_node = tn.Node(jnp.reshape(psi_data, [2] * n_data))
    anc_vecs = tc.quantum.onehot_d_tensor(anc_bits, d=2)
    anc_nodes = [tn.Node(anc_vecs[i]) for i in range(n_anc)]
    edges = [data_node[i] for i in range(n_data)] + [anc_nodes[i][0] for i in range(n_anc)]
    return tc.quantum.QuVector(edges)


def run_solution(config: dict) -> dict:
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy", preprocessing=True)

    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_trajectories = int(config["n_trajectories"])
    init_scale = float(config["initial_parameter_scale"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    seed = int(config["seed"])
    transverse_field = float(config["transverse_field"])
    n_pairs = min(n_data, n_anc)

    rng = np.random.default_rng(seed)
    theta_data = rng.normal(0.0, init_scale, size=(n_layers, n_data))
    theta_anc = rng.normal(0.0, init_scale, size=(n_layers, n_anc))
    theta_ent = rng.normal(0.0, init_scale, size=(n_layers, n_pairs))
    theta_fb0 = rng.normal(0.0, init_scale, size=(n_layers, n_pairs))
    theta_fb1 = rng.normal(0.0, init_scale, size=(n_layers, n_pairs))
    theta_post = rng.normal(0.0, init_scale, size=(n_layers, n_data))
    params = {
        "data": jnp.asarray(theta_data, dtype=jnp.float64),
        "anc": jnp.asarray(theta_anc, dtype=jnp.float64),
        "ent": jnp.asarray(theta_ent, dtype=jnp.float64),
        "fb0": jnp.asarray(theta_fb0, dtype=jnp.float64),
        "fb1": jnp.asarray(theta_fb1, dtype=jnp.float64),
        "post": jnp.asarray(theta_post, dtype=jnp.float64),
    }
    uniforms = jnp.asarray(
        rng.uniform(0.0, 1.0, size=(n_trajectories, n_layers, n_anc)),
        dtype=jnp.float64,
    )

    hamiltonian = _build_data_hamiltonian(n_data, transverse_field)
    init_data_state = tc.Circuit(n_data).state()
    init_anc_bits = jnp.zeros(n_anc, dtype=jnp.int32)
    zero_data_bits = jnp.zeros(n_data, dtype=jnp.int32)
    left_data = tuple(range(n_data))

    def trajectory_energy(p: dict, u_traj: jnp.ndarray) -> jnp.ndarray:
        def layer_step(carry, layer_inputs):
            psi_d, a_bits = carry
            p_l, u_l = layer_inputs
            c = tc.Circuit(n_qubits, mps_inputs=_make_quvector(psi_d, a_bits, n_data, n_anc))

            for i in range(n_data):
                c.ry(i, theta=p_l["data"][i])
            for i in range(n_anc):
                c.ry(n_data + i, theta=p_l["anc"][i])
            for i in range(n_pairs):
                c.rzz(n_data + i, i, theta=p_l["ent"][i])
            for i in range(n_anc - 1):
                c.cnot(n_data + i, n_data + i + 1)

            m_bits = jnp.stack(
                [c.cond_measurement(n_data + i, status=u_l[i]) for i in range(n_anc)]
            )

            for i in range(n_pairs):
                c.conditional_gate(
                    m_bits[i],
                    [
                        tc.gates.rzz(theta=p_l["fb0"][i]),
                        tc.gates.rzz(theta=p_l["fb1"][i]),
                    ],
                    n_data + i,
                    i,
                )
            for i in range(n_data - 1):
                c.cnot(i, i + 1)
            for i in range(n_data):
                c.rz(i, theta=p_l["post"][i])

            traceout = jnp.concatenate([zero_data_bits, m_bits])
            psi_next = c.projected_subsystem(traceout, left=left_data)
            return (psi_next, m_bits), None

        (psi_final, _), _ = jax.lax.scan(
            layer_step, (init_data_state, init_anc_bits), (p, u_traj)
        )
        c_obs = tc.Circuit(n_data, inputs=psi_final)
        return tc.templates.measurements.operator_expectation(c_obs, hamiltonian)

    batched_trajectory_energies = jax.jit(jax.vmap(trajectory_energy, in_axes=(None, 0)))

    def objective(p: dict) -> jnp.ndarray:
        return jnp.mean(batched_trajectory_energies(p, uniforms))

    if max_steps > 0:
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        value_and_grad_fn = jax.value_and_grad(objective)

        def opt_step(carry, _):
            p_curr, state_curr = carry
            loss, grads = value_and_grad_fn(p_curr)
            updates, state_next = optimizer.update(grads, state_curr, p_curr)
            p_next = optax.apply_updates(p_curr, updates)
            return (p_next, state_next), loss

        (final_params, _), energy_history = jax.lax.scan(
            opt_step, (params, opt_state), None, length=max_steps
        )
        energy_history_np = np.asarray(energy_history, dtype=np.float64)
    else:
        final_params = params
        energy_history_np = np.empty((0,), dtype=np.float64)

    final_traj_np = np.asarray(
        batched_trajectory_energies(final_params, uniforms), dtype=np.float64
    )

    return {
        "energy_history": energy_history_np,
        "final_trajectory_energies": final_traj_np,
    }
