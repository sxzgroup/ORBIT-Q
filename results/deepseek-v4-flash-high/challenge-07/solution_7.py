import numpy as np
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    import jax
    import optax

    nq = config["n_qubits"]
    nd = config["n_data_qubits"]
    nl = config["n_layers"]
    nt = config["n_trajectories"]
    tf = config["transverse_field"]
    steps = config["max_steps"]
    lr = config["learning_rate"]
    scale = config["initial_parameter_scale"]
    seed = config["seed"]
    npq = 6 * nl * nd

    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64)

    def unpack(p):
        return (
            p[0 : nl * nd].reshape(nl, nd),
            p[nl * nd : 2 * nl * nd].reshape(nl, nd),
            p[2 * nl * nd : 3 * nl * nd].reshape(nl, nd),
            p[3 * nl * nd : 4 * nl * nd].reshape(nl, nd),
            p[4 * nl * nd : 5 * nl * nd].reshape(nl, nd),
            p[5 * nl * nd : 6 * nl * nd].reshape(nl, nd),
        )

    def energy(c):
        e = c.expectation_ps(z=[0, 1])
        for i in range(1, nd - 1):
            e = e + c.expectation_ps(z=[i, i + 1])
        ex = c.expectation_ps(x=[0])
        for i in range(1, nd):
            ex = ex + c.expectation_ps(x=[i])
        return -(tc.backend.real(e) + tf * tc.backend.real(ex))

    def sample(params, status):
        c = tc.Circuit(nq)
        th_d, th_a, th_e, th_f0, th_f1, th_p = unpack(params)
        bits = []
        probs = []
        for l in range(nl):
            for i in range(nd):
                c.ry(i, theta=th_d[l, i])
            for i in range(nd):
                c.ry(nd + i, theta=th_a[l, i])
            for i in range(nd):
                c.rzz(nd + i, i, theta=th_e[l, i])
            for i in range(nd - 1):
                c.cnot(nd + i, nd + i + 1)
            for i in range(nd):
                pick, pr = c.general_kraus(
                    [P0, P1], nd + i, status=status[l, i], with_prob=True
                )
                bits.append(pick)
                probs.append(tc.backend.where(pick == 0, pr[0], pr[1]))
            for i in range(nd):
                sel = tc.backend.where(bits[-nd + i] == 0, th_f0[l, i], th_f1[l, i])
                c.rzz(nd + i, i, theta=sel)
            for i in range(nd - 1):
                c.cnot(i, i + 1)
            for i in range(nd):
                c.rz(i, theta=th_p[l, i])
        return tc.backend.stack(bits), tc.backend.stack(probs)

    def traj_energy(params, bits, probs):
        c = tc.Circuit(nq)
        th_d, th_a, th_e, th_f0, th_f1, th_p = unpack(params)
        for l in range(nl):
            for i in range(nd):
                c.ry(i, theta=th_d[l, i])
            for i in range(nd):
                c.ry(nd + i, theta=th_a[l, i])
            for i in range(nd):
                c.rzz(nd + i, i, theta=th_e[l, i])
            for i in range(nd - 1):
                c.cnot(nd + i, nd + i + 1)
            for i in range(nd):
                b = bits[l * nd + i]
                proj = tc.backend.where(b == 0, P0, P1)
                c.any(nd + i, unitary=proj / tc.backend.sqrt(probs[l * nd + i]))
            for i in range(nd):
                sel = tc.backend.where(bits[l * nd + i] == 0, th_f0[l, i], th_f1[l, i])
                c.rzz(nd + i, i, theta=sel)
            for i in range(nd - 1):
                c.cnot(i, i + 1)
            for i in range(nd):
                c.rz(i, theta=th_p[l, i])
        return energy(c)

    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, scale, size=(npq,)).astype(np.float32)
    status = np.random.default_rng(seed + 1).random((nt, nl, nd)).astype(np.float32)

    b_samp = tc.backend.vmap(sample, vectorized_argnums=(1,))
    b_traj = tc.backend.vmap(traj_energy, vectorized_argnums=(1, 2))
    sample_j = jax.jit(b_samp)
    obj = lambda p, b, pr: tc.backend.mean(b_traj(p, b, pr))
    vg = jax.jit(jax.value_and_grad(obj))

    opt = optax.adam(lr)
    opt_state = opt.init(params)
    history = np.zeros(steps)
    for t in range(steps):
        bits, probs = sample_j(params, status)
        val, grads = vg(params, bits, probs)
        history[t] = float(np.asarray(val))
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    bits, probs = sample_j(params, status)
    final_traj = np.asarray(b_traj(params, bits, probs))
    return {
        "energy_history": history,
        "final_trajectory_energies": final_traj,
    }
