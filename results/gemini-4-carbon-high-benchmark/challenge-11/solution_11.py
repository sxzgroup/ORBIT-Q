from functools import lru_cache
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import optax
import tensorcircuit as tc


def _spin1_constants():
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    sx = inv_sqrt2 * np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128
    )
    sy = inv_sqrt2 * np.array(
        [[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype=np.complex128
    )
    sz = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.complex128
    )
    xx_yy = (np.kron(sx, sx) + np.kron(sy, sy)).real
    zz = np.kron(sz, sz).real
    s_dot_s = xx_yy + zz
    s_dot_s_sq = s_dot_s @ s_dot_s
    sz2 = (sz @ sz).real
    return sx, sy, sz, xx_yy, zz, s_dot_s, s_dot_s_sq, sz2


def _build_hamiltonian_bcsr(n_sites: int, beta: float, anisotropy: float):
    _, _, _, _, _, s_dot_s, s_dot_s_sq, sz2 = _spin1_constants()
    hb_sp = sp.csr_matrix((s_dot_s + beta * s_dot_s_sq).astype(np.float32))
    sz2_sp = sp.csr_matrix(sz2.astype(np.float32))
    dim_tot = 3**n_sites
    ham_csr = sp.csr_matrix((dim_tot, dim_tot), dtype=np.float32)
    for i in range(n_sites - 1):
        left = sp.eye(3**i, dtype=np.float32, format="csr")
        right = sp.eye(3 ** (n_sites - 2 - i), dtype=np.float32, format="csr")
        ham_csr = ham_csr + sp.kron(sp.kron(left, hb_sp, format="csr"), right, format="csr")
    if anisotropy != 0.0:
        d_val = np.float32(anisotropy)
        for i in range(n_sites):
            left = sp.eye(3**i, dtype=np.float32, format="csr")
            right = sp.eye(3 ** (n_sites - 1 - i), dtype=np.float32, format="csr")
            ham_csr = ham_csr + d_val * sp.kron(
                sp.kron(left, sz2_sp, format="csr"), right, format="csr"
            )
    ham_csr.eliminate_zeros()
    return jsparse.BCSR(
        (
            jnp.asarray(ham_csr.data, dtype=jnp.complex64),
            jnp.asarray(ham_csr.indices, dtype=jnp.int32),
            jnp.asarray(ham_csr.indptr, dtype=jnp.int32),
        ),
        shape=ham_csr.shape,
    )


@lru_cache(maxsize=8)
def _get_compiled_runner(n_sites: int, n_layers: int, max_steps: int):
    _, _, sz_np, xx_yy_np, zz_np, _, s_dot_s_sq_np, _ = _spin1_constants()
    xx_yy = jnp.asarray(xx_yy_np, dtype=jnp.complex64)
    zz = jnp.asarray(zz_np, dtype=jnp.complex64)
    s_dot_s_sq = jnp.asarray(s_dot_s_sq_np, dtype=jnp.complex64)
    sz_gate = jnp.asarray(sz_np, dtype=jnp.complex64)
    str_gate = jnp.diag(jnp.array([-1.0, 1.0, -1.0], dtype=jnp.complex64))
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    n_even = n_sites // 2
    n_odd = (n_sites - 1) // 2
    string_pairs = [(r, n_sites - 1 - r) for r in range(3)]

    def single_rot(abc):
        alpha, beta_rot, gamma = abc[0], abc[1], abc[2]
        cb = jnp.cos(beta_rot)
        sb = jnp.sin(beta_rot)
        ry = jnp.array(
            [
                [0.5 * (1.0 + cb), -inv_sqrt2 * sb, 0.5 * (1.0 - cb)],
                [inv_sqrt2 * sb, cb, -inv_sqrt2 * sb],
                [0.5 * (1.0 - cb), inv_sqrt2 * sb, 0.5 * (1.0 + cb)],
            ],
            dtype=jnp.complex64,
        )
        pa = jnp.array([jnp.exp(-1j * alpha), 1.0 + 0.0j, jnp.exp(1j * alpha)], dtype=jnp.complex64)
        pg = jnp.array([jnp.exp(-1j * gamma), 1.0 + 0.0j, jnp.exp(1j * gamma)], dtype=jnp.complex64)
        return pa[:, None] * ry * pg[None, :]

    single_rot_vmap = jax.vmap(single_rot)

    def build_circuit(params, beta_param):
        biquad = beta_param * s_dot_s_sq

        def bond_unitary(tp):
            gen = tp[0] * xx_yy + tp[1] * zz + biquad
            return tc.backend.expm(-1j * gen)

        bond_vmap = jax.vmap(bond_unitary)
        c = tc.QuditCircuit(n_sites, dim=3)
        for i in range(1, n_sites, 2):
            c.x(i)
            c.x(i)
        for l in range(n_layers):
            lp = params[l]
            u1s = single_rot_vmap(lp[: 3 * n_sites].reshape(n_sites, 3))
            u_even = bond_vmap(lp[3 * n_sites : 3 * n_sites + 2 * n_even].reshape(n_even, 2))
            u_odd = bond_vmap(lp[3 * n_sites + 2 * n_even :].reshape(n_odd, 2))
            for i in range(n_sites):
                c.unitary(i, unitary=u1s[i])
            for b, i in enumerate(range(0, n_sites - 1, 2)):
                c.unitary(i, i + 1, unitary=u_even[b])
            for b, i in enumerate(range(1, n_sites - 1, 2)):
                c.unitary(i, i + 1, unitary=u_odd[b])
        return c

    def energy_density(params, ham, beta_param):
        c = build_circuit(params, beta_param)
        return tc.templates.measurements.operator_expectation(c, ham) / n_sites

    vag_fn = jax.value_and_grad(energy_density, argnums=0)

    @jax.jit
    def execute(init_params, ham, beta_param, lr):
        opt = optax.adam(lr)
        opt_state0 = opt.init(init_params)

        def step_body(carry, _):
            p, opt_s = carry
            val, grads = vag_fn(p, ham, beta_param)
            updates, opt_s = opt.update(grads, opt_s, p)
            p = optax.apply_updates(p, updates)
            return (p, opt_s), val

        (final_params, _), history = jax.lax.scan(
            step_body, (init_params, opt_state0), None, length=max_steps
        )
        c_fin = build_circuit(final_params, beta_param)
        final_e = tc.templates.measurements.operator_expectation(c_fin, ham) / n_sites
        so_vals = [
            jnp.real(
                c_fin.expectation(
                    *([(sz_gate, [i])] + [(str_gate, [k]) for k in range(i + 1, j)] + [(sz_gate, [j])]),
                    reuse=True,
                )
            )
            for (i, j) in string_pairs
        ]
        return history, final_e, jnp.stack(so_vals)

    return execute


def run_solution(config):
    n_sites = int(config["n_sites"])
    n_layers = int(config["n_layers"])
    beta = float(config["beta"])
    anisotropy = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    n_params_per_layer = 3 * n_sites + 2 * (n_sites - 1)
    rng = np.random.default_rng(seed)
    init_params_np = rng.normal(0.0, init_scale, size=(n_layers, n_params_per_layer)).astype(
        np.float32
    )

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex64"),
        tc.runtime_contractor("omeco-4-8"),
    ):
        ham_bcsr = _build_hamiltonian_bcsr(n_sites, beta, anisotropy)
        runner = _get_compiled_runner(n_sites, n_layers, max_steps)
        history, final_e, final_so = runner(
            jnp.asarray(init_params_np),
            ham_bcsr,
            jnp.asarray(beta, dtype=jnp.complex64),
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        history_np = np.asarray(history, dtype=np.float64)
        final_e_np = np.float64(final_e)
        final_so_np = np.asarray(final_so, dtype=np.float64)

    return {
        "energy_density_history": history_np,
        "final_energy_density": final_e_np,
        "final_string_orders": final_so_np,
    }
