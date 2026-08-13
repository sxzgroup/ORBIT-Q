import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO

tc.set_backend("jax")


def _build_operators(n):
    xx = []
    yy = []
    zz = []
    zst = []
    for i in range(n - 1):
        row = [0] * n
        row[i] = 1
        row[i + 1] = 1
        xx.append(row)
        row = [0] * n
        row[i] = 2
        row[i + 1] = 2
        yy.append(row)
        row = [0] * n
        row[i] = 3
        row[i + 1] = 3
        zz.append(row)
    for i in range(n):
        row = [0] * n
        row[i] = 3
        zst.append(row)

    ones_m = jnp.ones((n - 1,), dtype=jnp.float32)
    zweights = jnp.array([(-1.0) ** i for i in range(n)], dtype=jnp.float32)
    hxx = PauliStringSum2COO(xx, ones_m)
    hyy = PauliStringSum2COO(yy, ones_m)
    hzz = PauliStringSum2COO(zz, ones_m)
    hzst = PauliStringSum2COO(zst, zweights)
    analog_xy = hxx + hyy
    target = 0.7 * hxx + 0.7 * hyy + 1.1 * hzz + 0.25 * hzst
    return analog_xy, hzst, target


def run_solution(config):
    n = int(config["n_qubits"])
    nb = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    lr = float(config["learning_rate"])
    max_steps = int(config["max_steps"])
    rtol = float(config["ode_rtol"])
    atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])

    analog_xy, hzst, target = _build_operators(n)

    init_circuit = tc.Circuit(n)
    for i in range(1, n, 2):
        init_circuit.x(i)
    neel = init_circuit.wavefunction()

    block_dim = 3 + 3 * n
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(nb):
        rows.append([0.0, 0.1, 0.1] + list(rng.normal(0.0, 0.1, 3 * n)))
    params = jnp.array(rows, dtype=jnp.float32)

    def sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    def analog_hamiltonian(t, jj, dd):
        return jj * analog_xy + dd * hzst

    def block_step(psi, lp):
        time = t_min + (t_max - t_min) * sigmoid(lp[0])
        jj = jnp.tanh(lp[1])
        dd = jnp.tanh(lp[2])
        psi = tc.timeevol.ode_evol_global(
            analog_hamiltonian,
            psi,
            jnp.array([0.0, time], dtype=jnp.float32),
            None,
            jj,
            dd,
            rtol=rtol,
            atol=atol,
            max_steps=ode_max_steps,
        )[-1]

        c = tc.Circuit(n, inputs=psi)
        idx = 3
        for _ in range(n):
            c.rz(_, theta=lp[idx])
            c.ry(_, theta=lp[idx + 1])
            c.rz(_, theta=lp[idx + 2])
            idx += 3
        return c.wavefunction(), None

    def energy(params_blocks):
        psi, _ = jax.lax.scan(block_step, neel, params_blocks)
        hpsi = target @ psi
        return tc.backend.real(tc.backend.conj(psi) @ hpsi) / n

    value_and_grad = jax.jit(jax.value_and_grad(energy))

    @jax.jit
    def adam_step(p, g, m, v, step):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * jnp.square(g)
        mhat = m / (1.0 - jnp.power(0.9, step))
        vhat = v / (1.0 - jnp.power(0.999, step))
        p = p - lr * mhat / (jnp.sqrt(vhat) + 1e-8)
        return p, m, v

    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    history = np.empty(max_steps, dtype=np.float64)

    for step in range(max_steps):
        val, grads = value_and_grad(params)
        val.block_until_ready()
        grads.block_until_ready()
        history[step] = float(val)
        params, m, v = adam_step(
            params,
            grads,
            m,
            v,
            jnp.array(step + 1, dtype=jnp.float32),
        )

    params.block_until_ready()
    params_np = np.asarray(params)
    times = []
    couplings = []
    detunings = []
    for l in range(nb):
        s, jj, dd = params_np[l, 0], params_np[l, 1], params_np[l, 2]
        times.append(t_min + (t_max - t_min) * (1.0 / (1.0 + np.exp(-s))))
        couplings.append(np.tanh(jj))
        detunings.append(np.tanh(dd))

    return {
        "final_analog_times": np.asarray(times, dtype=np.float64),
        "final_analog_couplings": np.asarray(couplings, dtype=np.float64),
        "final_analog_detunings": np.asarray(detunings, dtype=np.float64),
        "energy_density_history": history,
    }
