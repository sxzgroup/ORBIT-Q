import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    n = config["n_qubits"]
    nb = config["n_blocks"]
    tmin = config["t_min"]
    tmax = config["t_max"]
    rtol = config["ode_rtol"]
    atol = config["ode_atol"]
    mx = config["ode_max_steps"]
    steps = config["max_steps"]
    lr = config["learning_rate"]

    I2 = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    def kron_seq(mats):
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format="csr")
        return out.tocoo()

    A = None
    for i in range(n - 1):
        mats = [I2] * n
        mats[i] = X
        mats[i + 1] = X
        term = kron_seq(mats)
        mats[i] = Y
        mats[i + 1] = Y
        term = term + kron_seq(mats)
        A = term if A is None else A + term
    A = A.tocoo()

    B = None
    for i in range(n):
        mats = [I2] * n
        mats[i] = Z
        term = kron_seq(mats).tocoo()
        B = term if B is None else B + term
    B = B.tocoo()

    def to_bcoo(m):
        data = jnp.array(m.data, dtype=jnp.complex64)
        idx = jnp.array(np.stack([m.row, m.col], axis=1), dtype=jnp.int32)
        return jsp.BCOO((data, idx), shape=m.shape)

    A_j = to_bcoo(A)
    B_j = to_bcoo(B)

    gx = tc.gates.x()
    gy = tc.gates.y()
    gz = tc.gates.z()
    terms = []
    for i in range(n - 1):
        terms.append(((gx * 0.7, [i]), (gx, [i + 1])))
        terms.append(((gy * 0.7, [i]), (gy, [i + 1])))
        terms.append(((gz * 1.1, [i]), (gz, [i + 1])))
    for i in range(n):
        terms.append(((gz * (0.25 * (-1.0) ** i), [i]),))

    def target_energy(state):
        e = tc.backend.cast(0.0, tc.rdtypestr)
        for ops in terms:
            e = e + tc.expectation(*ops, ket=state)
        return e

    def u2x2(a, b, g):
        ca = jnp.cos(a / 2)
        sa = jnp.sin(a / 2)
        e1 = ca - 1j * sa
        e2 = ca + 1j * sa
        cb = jnp.cos(b / 2)
        sb = jnp.sin(b / 2)
        cg = jnp.cos(g / 2)
        sg = jnp.sin(g / 2)
        f1 = cg - 1j * sg
        f2 = cg + 1j * sg
        m1 = jnp.array([[cb * f1, -sb * f2], [sb * f1, cb * f2]], dtype=jnp.complex64)
        row0 = jnp.stack([e1 * m1[0, 0], e1 * m1[0, 1]])
        row1 = jnp.stack([e2 * m1[1, 0], e2 * m1[1, 1]])
        return jnp.stack([row0, row1])

    def loss(params):
        s = params[:nb]
        j = params[nb : 2 * nb]
        d = params[2 * nb : 3 * nb]
        t = tmin + (tmax - tmin) * jax.nn.sigmoid(s)
        J = jnp.tanh(j)
        D = jnp.tanh(d)
        rot = params[3 * nb :].reshape(nb, n, 3)

        def ham(tt, J, D):
            return J * A_j + D * B_j

        c0 = tc.Circuit(n)
        for i in range(1, n, 2):
            c0.x(i)
        state = c0.state()
        for l in range(nb):
            ts = tc.backend.stack([tc.backend.cast(0.0, tc.rdtypestr), t[l]])
            s1 = tc.timeevol.ode_evol_global(
                ham,
                state,
                ts,
                None,
                J[l],
                D[l],
                mode="hamiltonian",
                rtol=rtol,
                atol=atol,
                max_steps=mx,
            )
            c = tc.Circuit(n, inputs=s1[-1])
            for k in range(n):
                c.apply_general_gate(
                    tc.gates.Gate(u2x2(rot[l, k, 2], rot[l, k, 1], rot[l, k, 0])), k
                )
            state = c.state()
        return jnp.real(target_energy(state)) / n

    vg = jax.jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(0)
    params = np.concatenate(
        [
            np.zeros(nb),
            0.1 * np.ones(nb),
            0.1 * np.ones(nb),
            rng.normal(0.0, 0.1, nb * n * 3),
        ]
    ).astype(np.float32)

    m = np.zeros_like(params)
    vv = np.zeros_like(params)
    hist = []
    for it in range(1, steps + 1):
        v, g = vg(jnp.array(params))
        hist.append(float(v))
        g = np.asarray(g)
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g * g
        mh = m / (1.0 - 0.9 ** it)
        vh = vv / (1.0 - 0.999 ** it)
        params = params - lr * mh / (np.sqrt(vh) + 1e-8)

    s = params[:nb]
    j = params[nb : 2 * nb]
    d = params[2 * nb : 3 * nb]
    return {
        "final_analog_times": tmin + (tmax - tmin) / (1.0 + np.exp(-s)),
        "final_analog_couplings": np.tanh(j),
        "final_analog_detunings": np.tanh(d),
        "energy_density_history": np.asarray(hist),
    }
