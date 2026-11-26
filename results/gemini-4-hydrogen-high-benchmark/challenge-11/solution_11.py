"""Variational Quantum Eigensolver for an open spin-1 Haldane chain."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _spin1_operators():
    """Return spin-1 operators (Sx, Sy, Sz) in the {|+1>, |0>, |-1>} basis."""
    s2 = np.sqrt(2.0)
    sx = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex64) / s2
    sy = np.array([[0.0, -1j, 0.0], [1j, 0.0, -1j], [0.0, 1j, 0.0]], dtype=np.complex64) / s2
    sz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.complex64)
    return sx, sy, sz


def _single_site_gate(alpha, beta_ang, gamma, sy_j, sy2_j):
    """Build Rz(alpha) @ Ry(beta_ang) @ Rz(gamma) for spin-1."""
    ea = jnp.array([jnp.exp(-1j * alpha), 1.0, jnp.exp(1j * alpha)], dtype=jnp.complex64)
    eg = jnp.array([jnp.exp(-1j * gamma), 1.0, jnp.exp(1j * gamma)], dtype=jnp.complex64)
    ry = (
        jnp.eye(3, dtype=jnp.complex64)
        - 1j * jnp.sin(beta_ang) * sy_j
        + (jnp.cos(beta_ang) - 1.0) * sy2_j
    )
    return (ea[:, None] * ry) * eg[None, :]


def _two_site_gate(theta, phi, beta):
    """Build exp(-i [theta (Sx⊗Sx + Sy⊗Sy) + phi (Sz⊗Sz) + beta (S·S)^2]) via Sz-sectors."""
    e_pm2 = jnp.exp(-1j * (phi + beta))
    eb = jnp.exp(-1j * beta)
    ct = jnp.cos(theta) * eb
    st = -1j * jnp.sin(theta) * eb

    # Sz = 0 sector splits into |-> singlet-type eigenstate and 2x2 {|+>, |0,0>} block
    h0 = -0.5 * phi + 2.5 * beta
    hz = -0.5 * phi + 0.5 * beta
    hx = jnp.sqrt(2.0) * (theta - beta)
    r2 = hx * hx + hz * hz
    small = r2 < 1e-10
    r_safe = jnp.sqrt(jnp.where(small, 1.0, r2))
    cr = jnp.where(small, 1.0 - 0.5 * r2 + r2 * r2 / 24.0, jnp.cos(r_safe))
    sr = jnp.where(small, 1.0 - r2 / 6.0 + r2 * r2 / 120.0, jnp.sin(r_safe) / r_safe)

    e0 = jnp.exp(-1j * h0)
    u00 = e0 * (cr - 1j * sr * hz)
    u11 = e0 * (cr + 1j * sr * hz)
    u01 = e0 * (-1j * sr * hx)
    em = jnp.exp(-1j * (-phi + beta))

    a22 = 0.5 * (u00 + em)
    a26 = 0.5 * (u00 - em)
    a24 = u01 / jnp.sqrt(2.0)

    u = jnp.zeros((9, 9), dtype=jnp.complex64)
    u = u.at[0, 0].set(e_pm2).at[8, 8].set(e_pm2)
    u = u.at[1, 1].set(ct).at[1, 3].set(st).at[3, 1].set(st).at[3, 3].set(ct)
    u = u.at[5, 5].set(ct).at[5, 7].set(st).at[7, 5].set(st).at[7, 7].set(ct)
    u = u.at[2, 2].set(a22).at[2, 6].set(a26).at[6, 2].set(a26).at[6, 6].set(a22)
    u = u.at[2, 4].set(a24).at[4, 2].set(a24).at[6, 4].set(a24).at[4, 6].set(a24)
    return u.at[4, 4].set(u11)


def _build_bond_hamiltonians(n_sites, beta, ani, sx, sy, sz):
    """Build two-site bond Hamiltonians with single-ion anisotropy folded in."""
    i3 = np.eye(3, dtype=np.complex64)
    sz2 = sz @ sz
    s_dot_s = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    h2_base = s_dot_s + beta * (s_dot_s @ s_dot_s)
    bonds = []
    for i in range(n_sites - 1):
        w_l = 1.0 if i == 0 else 0.5
        w_r = 1.0 if i == n_sites - 2 else 0.5
        hb = h2_base + ani * (w_l * np.kron(sz2, i3) + w_r * np.kron(i3, sz2))
        bonds.append(jnp.asarray(hb.reshape(3, 3, 3, 3), dtype=jnp.complex64))
    return bonds


def run_solution(config):
    """Run brickwork spin-1 VQE with Adam and return energy history and string orders."""
    n_sites = int(config["n_sites"])
    n_layers = int(config["n_layers"])
    beta = float(config["beta"])
    ani = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    with (
        tc.cons.runtime_backend("jax"),
        tc.cons.runtime_dtype("complex64"),
        tc.cons.runtime_contractor("greedy", preprocessing=True),
    ):
        sx, sy, sz = _spin1_operators()
        sy_j = jnp.asarray(sy)
        sy2_j = sy_j @ sy_j
        sz_j = jnp.asarray(sz)
        p_mid_j = jnp.diag(jnp.array([-1.0, 1.0, -1.0], dtype=jnp.complex64))
        h_bonds = _build_bond_hamiltonians(n_sites, beta, ani, sx, sy, sz)

        c_init = tc.QuditCircuit(n_sites, dim=3)
        flip = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.complex64)
        for i in range(1, n_sites, 2):
            c_init.any(i, unitary=flip)
        psi0 = c_init.wavefunction()

        n_even = n_sites // 2
        n_odd = (n_sites - 1) // 2
        n_params_layer = 3 * n_sites + 2 * (n_sites - 1)

        def apply_layer(psi, layer_params):
            c = tc.QuditCircuit(n_sites, dim=3, inputs=psi)
            p1 = layer_params[: 3 * n_sites].reshape(n_sites, 3)
            p2_even = layer_params[3 * n_sites : 3 * n_sites + 2 * n_even].reshape(n_even, 2)
            p2_odd = layer_params[3 * n_sites + 2 * n_even :].reshape(n_odd, 2)

            for i in range(n_sites):
                c.any(i, unitary=_single_site_gate(p1[i, 0], p1[i, 1], p1[i, 2], sy_j, sy2_j))
            for k, i in enumerate(range(0, n_sites - 1, 2)):
                c.any(i, i + 1, unitary=_two_site_gate(p2_even[k, 0], p2_even[k, 1], beta))
            for k, i in enumerate(range(1, n_sites - 1, 2)):
                c.any(i, i + 1, unitary=_two_site_gate(p2_odd[k, 0], p2_odd[k, 1], beta))
            return c.wavefunction(), None

        def energy_density_fn(params):
            psi, _ = jax.lax.scan(apply_layer, psi0, params)
            c_out = tc.QuditCircuit(n_sites, dim=3, inputs=psi)
            total_e = 0.0
            for i in range(n_sites - 1):
                total_e = total_e + c_out.expectation((h_bonds[i], [i, i + 1]), reuse=True)
            return jnp.real(total_e) / n_sites

        optimizer = optax.adam(learning_rate=lr)
        vg_fn = jax.value_and_grad(energy_density_fn)
        string_pairs = [(0, n_sites - 1), (1, n_sites - 2), (2, n_sites - 3)]

        @jax.jit
        def solve(p_init):
            opt_state = optimizer.init(p_init)

            def step(carry, _):
                params, state = carry
                e_val, grads = vg_fn(params)
                updates, state = optimizer.update(grads, state, params)
                params = optax.apply_updates(params, updates)
                return (params, state), e_val

            (final_params, _), history = jax.lax.scan(
                step, (p_init, opt_state), None, length=max_steps
            )
            final_e = energy_density_fn(final_params)
            psi_final, _ = jax.lax.scan(apply_layer, psi0, final_params)
            c_final = tc.QuditCircuit(n_sites, dim=3, inputs=psi_final)
            strings = []
            for i, j in string_pairs:
                ops = [(sz_j, [i])] + [(p_mid_j, [k]) for k in range(i + 1, j)] + [(sz_j, [j])]
                strings.append(jnp.real(c_final.expectation(*ops, reuse=True)))
            return history, final_e, jnp.stack(strings)

        rng = np.random.default_rng(seed)
        p_init = jnp.asarray(
            rng.normal(0.0, init_scale, size=(n_layers, n_params_layer)).astype(np.float32)
        )
        history, final_e, strings = solve(p_init)
        history.block_until_ready()

        return {
            "energy_density_history": np.asarray(history, dtype=np.float64),
            "final_energy_density": float(final_e),
            "final_string_orders": np.asarray(strings, dtype=np.float64),
        }
