"""16-Qubit Measurement-Feedback VQE implemented with TensorCircuit-NG (JAX)."""

import math
import cotengra as ctg
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
import tensornetwork as tn


def _build_tfim_mpo(n_data: int, n_anc: int, h: float) -> tc.quantum.QuOperator:
    """Construct the open-boundary 1D TFIM Hamiltonian as a QuOperator MPO."""
    i2 = np.eye(2, dtype=np.complex64)
    x2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
    z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)

    if n_data == 1:
        tensors = [(-h * x2)[None, None, :, :]]
    else:
        w0 = np.stack([i2, -z2, -h * x2], axis=0)[None, ...]
        wb = np.zeros((3, 3, 2, 2), dtype=np.complex64)
        wb[0, 0] = i2
        wb[0, 1] = -z2
        wb[0, 2] = -h * x2
        wb[1, 2] = z2
        wb[2, 2] = i2
        w_last = np.stack([-h * x2, z2, i2], axis=0)[:, None, ...]
        tensors = [w0] + [wb] * (n_data - 2) + [w_last]

    tensors.extend([i2[None, None, :, :]] * n_anc)
    nodes = [tn.Node(jnp.asarray(w)) for w in tensors]
    for k in range(len(nodes) - 1):
        nodes[k][1] ^ nodes[k + 1][0]
    b_left = tn.Node(jnp.array([1.0], dtype=np.complex64))
    b_right = tn.Node(jnp.array([1.0], dtype=np.complex64))
    b_left[0] ^ nodes[0][0]
    nodes[-1][1] ^ b_right[0]
    return tc.quantum.QuOperator(
        out_edges=[n[2] for n in nodes],
        in_edges=[n[3] for n in nodes],
    )


def _apply_pre_measurement(
    c: tc.Circuit, params: dict, layer: int, n_data: int, n_anc: int
) -> None:
    """Apply steps 1-3 (data RY, ancilla RY, pair RZZ, ancilla CNOT ladder)."""
    for i in range(n_data):
        c.ry(i, theta=params["data"][layer, i])
    for i in range(n_anc):
        c.ry(n_data + i, theta=params["anc"][layer, i])
    for i in range(min(n_data, n_anc)):
        c.rzz(n_data + i, i, theta=params["ent"][layer, i])
    for i in range(n_anc - 1):
        c.cnot(n_data + i, n_data + i + 1)


def _apply_post_measurement(
    c: tc.Circuit,
    ms: list,
    params: dict,
    layer: int,
    n_data: int,
    n_anc: int,
) -> None:
    """Apply steps 5-7 (conditional RZZ feedback, data CNOT ladder, data RZ)."""
    for i in range(min(n_data, n_anc)):
        g0 = tc.gates.rzz(theta=params["fb0"][layer, i])
        g1 = tc.gates.rzz(theta=params["fb1"][layer, i])
        c.conditional_gate(ms[i], [g0, g1], n_data + i, i)
    for i in range(n_data - 1):
        c.cnot(i, i + 1)
    for i in range(n_data):
        c.rz(i, theta=params["post"][layer, i])


def run_solution(config: dict) -> dict:
    """Optimize the measurement-feedback VQE protocol via Adam."""
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    opt_ctg = ctg.ReusableHyperOptimizer(
        methods=["greedy", "random-greedy"],
        max_repeats=32,
        progbar=False,
        parallel=False,
    )
    tc.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)

    n_data = int(config["n_data_qubits"])
    n_anc = int(config["n_ancilla_qubits"])
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    init_scale = float(config["initial_parameter_scale"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    h_field = float(config["transverse_field"])

    rng = np.random.default_rng(seed)
    n_pairs = min(n_data, n_anc)
    shapes = {
        "data": (n_layers, n_data),
        "anc": (n_layers, n_anc),
        "ent": (n_layers, n_pairs),
        "fb0": (n_layers, n_pairs),
        "fb1": (n_layers, n_pairs),
        "post": (n_layers, n_data),
    }
    params = {
        k: jnp.array(rng.normal(0.0, init_scale, size=s), dtype=jnp.float32)
        for k, s in shapes.items()
    }
    us_batch = jnp.array(
        rng.uniform(0.0, 1.0, size=(n_traj, n_layers, n_anc)),
        dtype=jnp.float32,
    )

    h_mpo = _build_tfim_mpo(n_data, n_anc, h_field)
    chunk = math.gcd(n_traj, 8)
    n_chunks = n_traj // chunk

    def traj_energy(p: dict, us: jax.Array, state_l0: jax.Array) -> jax.Array:
        state = state_l0
        for layer in range(n_layers):
            ms = []
            for i in range(n_anc):
                cm = tc.Circuit(n_qubits, inputs=state)
                ms.append(cm.cond_measurement(n_data + i, status=us[layer, i]))
                state = cm.state()
            cp = tc.Circuit(n_qubits, inputs=state)
            _apply_post_measurement(cp, ms, p, layer, n_data, n_anc)
            if layer + 1 < n_layers:
                _apply_pre_measurement(cp, p, layer + 1, n_data, n_anc)
            state = cp.state()
        cf = tc.Circuit(n_qubits, inputs=state)
        return tc.templates.measurements.mpo_expectation(cf, h_mpo)

    def eval_trajectories(p: dict, us: jax.Array) -> jax.Array:
        c0 = tc.Circuit(n_qubits)
        _apply_pre_measurement(c0, p, 0, n_data, n_anc)
        state_l0 = c0.state()
        us_c = us.reshape(n_chunks, chunk, n_layers, n_anc)

        def scan_chunk(_, u_c):
            return None, jax.vmap(traj_energy, in_axes=(None, 0, None))(
                p, u_c, state_l0
            )

        _, es_c = jax.lax.scan(scan_chunk, None, us_c)
        return es_c.reshape(n_traj)

    def batch_objective(p: dict, us: jax.Array) -> jax.Array:
        return jnp.mean(eval_trajectories(p, us))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    val_and_grad = jax.value_and_grad(batch_objective)

    def train_step(carry, _):
        p, o_state = carry
        loss, grads = val_and_grad(p, us_batch)
        updates, new_o_state = optimizer.update(grads, o_state, p)
        new_p = optax.apply_updates(p, updates)
        return (new_p, new_o_state), loss

    @jax.jit
    def run_training(init_p, init_o):
        (final_p, _), history = jax.lax.scan(
            train_step, (init_p, init_o), None, length=max_steps
        )
        final_es = eval_trajectories(final_p, us_batch)
        return history, final_es

    history, final_es = run_training(params, opt_state)
    return {
        "energy_history": np.asarray(history, dtype=np.float64),
        "final_trajectory_energies": np.asarray(final_es, dtype=np.float64),
    }
