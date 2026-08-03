"""Challenge 7: 16-qubit measurement-feedback VQE (trajectory-averaged).

TensorCircuit-NG dense simulation (JAX backend). Each trajectory runs the
full hybrid protocol on one circuit: trainable RY data/ancilla layers, RZZ
data-ancilla entanglers, fixed ancilla CNOT ladder, projective mid-circuit
ancilla measurements via the framework's jittable ``cond_measure`` (driven by
a fixed batch of external uniforms so the 64-trajectory objective is
reproducible across optimizer updates), measurement-conditioned RZZ feedback
gates, fixed data CNOT ladder, and trainable RZ post-rotations. The
trajectory-averaged data-Hamiltonian energy is minimized with Adam for
exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax
from tensorcircuit.templates.measurements import operator_expectation


def _data_tfim(n_total, n_data, field):
    # H = -sum Z_i Z_{i+1} - field * sum X_i acting on the data qubits only
    ls, ws = [], []
    for i in range(n_data - 1):
        s = [0] * n_total
        s[i] = 3
        s[i + 1] = 3
        ls.append(s)
        ws.append(-1.0)
    for i in range(n_data):
        s = [0] * n_total
        s[i] = 1
        ls.append(s)
        ws.append(-field)
    return tc.quantum.PauliStringSum2COO(ls, ws)


def run_solution(config):
    nd = int(config["n_data_qubits"])
    na = int(config["n_ancilla_qubits"])
    n = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    n_traj = int(config["n_trajectories"])
    scale = float(config["initial_parameter_scale"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    field = float(config["transverse_field"])

    ham = _data_tfim(n, nd, field)

    def trajectory_energy(params, u):
        # u: (n_layers, na) fixed uniforms driving the projective measurements
        c = tc.Circuit(n)
        for l in range(n_layers):
            for i in range(nd):
                c.ry(i, theta=params["data"][l, i])
            for i in range(na):
                c.ry(nd + i, theta=params["anc"][l, i])
            for i in range(nd):
                c.rzz(nd + i, i, theta=params["ent"][l, i])
            for i in range(na - 1):
                c.cnot(nd + i, nd + i + 1)
            bits = [c.cond_measure(nd + i, status=u[l, i]) for i in range(na)]
            for i in range(nd):
                angle = jnp.where(bits[i], params["fb1"][l, i], params["fb0"][l, i])
                c.rzz(nd + i, i, theta=angle)
            for i in range(nd - 1):
                c.cnot(i, i + 1)
            for i in range(nd):
                c.rz(i, theta=params["post"][l, i])
        return tc.backend.real(operator_expectation(c, ham))

    traj_fn = trajectory_energy

    key = jax.random.PRNGKey(seed)
    k_init, k_traj = jax.random.split(key)
    names = ("data", "anc", "ent", "fb0", "fb1", "post")
    init = scale * jax.random.normal(k_init, (len(names), n_layers, nd), dtype=jnp.float32)
    params = {name: init[k] for k, name in enumerate(names)}
    # fixed per-trajectory measurement uniforms, drawn once and reused
    uniforms = jax.random.uniform(k_traj, (n_traj, n_layers, na), dtype=jnp.float32)

    def loss_fn(params):
        return jnp.mean(jax.vmap(lambda u: traj_fn(params, u))(uniforms))

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    opt_state = opt.init(params)
    es = []
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), dtype=np.float64)

    final_traj = jax.jit(
        lambda p: jax.vmap(lambda u: trajectory_energy(p, u))(uniforms)
    )(params)
    return {
        "energy_history": e_hist,
        "final_trajectory_energies": np.asarray(final_traj, dtype=np.float64),
    }
