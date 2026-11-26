"""Problem 11: Spin-1 Haldane-Chain VQE With String-Order Verification."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _build_spin1_operators(n_sites, beta, anisotropy, cdtype):
    s2 = 1.0 / np.sqrt(2.0)
    sx = np.array([[0.0, s2, 0.0], [s2, 0.0, s2], [0.0, s2, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j * s2, 0.0], [1j * s2, 0.0, -1j * s2], [0.0, 1j * s2, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.complex128)
    sz2, eye3 = sz @ sz, np.eye(3, dtype=np.complex128)
    s_dot_s = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    h_bond_base = s_dot_s + beta * (s_dot_s @ s_dot_s)

    h_bonds = []
    for i in range(n_sites - 1):
        w_l = 1.0 if i == 0 else 0.5
        w_r = 1.0 if i == n_sites - 2 else 0.5
        hb = h_bond_base + anisotropy * (w_l * np.kron(sz2, eye3) + w_r * np.kron(eye3, sz2))
        h_bonds.append(jnp.asarray(hb.reshape(3, 3, 3, 3), dtype=cdtype))

    sz_j = jnp.asarray(sz, dtype=cdtype)
    exp_ipi_sz = jnp.asarray(np.diag([-1.0, 1.0, -1.0]), dtype=cdtype)
    swap02 = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=cdtype)
    return h_bonds, sz_j, exp_ipi_sz, swap02


def _single_site_unitaries(alpha, beta_ang, gamma, cdtype, rdtype):
    sqrt2 = jnp.asarray(np.sqrt(2.0), dtype=rdtype)
    cb, sb = jnp.cos(beta_ang), jnp.sin(beta_ang)
    c2, s2, m = 0.5 * (1.0 + cb), 0.5 * (1.0 - cb), sb / sqrt2
    ry = jnp.stack([
        jnp.stack([c2, -m, s2], axis=-1),
        jnp.stack([m, cb, -m], axis=-1),
        jnp.stack([s2, m, c2], axis=-1),
    ], axis=-2).astype(cdtype)
    ones = jnp.ones_like(alpha, dtype=cdtype)
    left = jnp.stack([jnp.exp(-1j * alpha), ones, jnp.exp(1j * alpha)], axis=-1)
    right = jnp.stack([jnp.exp(-1j * gamma), ones, jnp.exp(1j * gamma)], axis=-1)
    return (left[..., :, None] * ry) * right[..., None, :]


def _two_site_unitaries(theta, phi, beta, cdtype, rdtype):
    sqrt2 = jnp.asarray(np.sqrt(2.0), dtype=rdtype)
    beta_t = jnp.asarray(beta, dtype=rdtype)
    d0 = jnp.exp(-1j * (phi + beta_t))
    d2 = jnp.exp(-1j * (theta + beta_t))
    d3 = jnp.exp(-1j * (-theta + beta_t))
    d6 = jnp.exp(-1j * (-phi + beta_t))
    h00, h11, h01 = -phi + 3.0 * beta_t, 2.0 * beta_t, sqrt2 * (theta - beta_t)
    avg, diff = 0.5 * (h00 + h11), 0.5 * (h00 - h11)
    r2 = diff * diff + h01 * h01
    r = jnp.sqrt(jnp.where(r2 > 0.0, r2, 1.0))
    c = jnp.where(r2 > 0.0, jnp.cos(r), 1.0)
    s_over_r = jnp.where(r2 > 1e-12, jnp.sin(r) / r, 1.0 - r2 / 6.0)
    phase = jnp.exp(-1j * avg)
    u77 = phase * (c - 1j * s_over_r * diff)
    u88 = phase * (c + 1j * s_over_r * diff)
    u78 = phase * (-1j * s_over_r * h01)

    p13, m13 = 0.5 * (d2 + d3), 0.5 * (d2 - d3)
    p26, m26 = 0.5 * (u77 + d6), 0.5 * (u77 - d6)
    o4, z = u78 / sqrt2, jnp.zeros_like(d0)
    rows = [
        jnp.stack([d0, z, z, z, z, z, z, z, z], axis=-1),
        jnp.stack([z, p13, z, m13, z, z, z, z, z], axis=-1),
        jnp.stack([z, z, p26, z, o4, z, m26, z, z], axis=-1),
        jnp.stack([z, m13, z, p13, z, z, z, z, z], axis=-1),
        jnp.stack([z, z, o4, z, u88, z, o4, z, z], axis=-1),
        jnp.stack([z, z, z, z, z, p13, z, m13, z], axis=-1),
        jnp.stack([z, z, m26, z, o4, z, p26, z, z], axis=-1),
        jnp.stack([z, z, z, z, z, m13, z, p13, z], axis=-1),
        jnp.stack([z, z, z, z, z, z, z, z, d0], axis=-1),
    ]
    return jnp.stack(rows, axis=-2).reshape(*theta.shape, 3, 3, 3, 3)


def run_solution(config):
    n_sites = int(config["n_sites"])
    n_layers = int(config["n_layers"])
    beta = float(config["beta"])
    anisotropy = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    n_even, n_odd = n_sites // 2, (n_sites - 1) // 2
    n_layer_params = 3 * n_sites + 2 * (n_sites - 1)
    string_pairs = [(0, n_sites - 1), (1, n_sites - 2), (2, n_sites - 3)]

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex64"),
        tc.runtime_contractor("greedy", preprocessing=True, use_primitives=True),
    ):
        cdtype, rdtype = jnp.complex64, jnp.float32
        h_bonds, sz_j, exp_ipi_sz, swap02 = _build_spin1_operators(n_sites, beta, anisotropy, cdtype)

        def build_gates(params):
            off = 3 * n_sites
            off2 = off + 2 * n_even
            u1 = _single_site_unitaries(
                params[:, 0:n_sites], params[:, n_sites : 2 * n_sites], params[:, 2 * n_sites : off], cdtype, rdtype
            )
            u2e = _two_site_unitaries(params[:, off : off + n_even], params[:, off + n_even : off2], beta, cdtype, rdtype)
            u2o = _two_site_unitaries(params[:, off2 : off2 + n_odd], params[:, off2 + n_odd :], beta, cdtype, rdtype)
            return u1, u2e, u2o

        def apply_gates(c, u1_l, u2e_l, u2o_l):
            for i in range(n_sites):
                c.unitary(i, unitary=u1_l[i])
            for b in range(n_even):
                c.unitary(2 * b, 2 * b + 1, unitary=u2e_l[b])
            for b in range(n_odd):
                c.unitary(2 * b + 1, 2 * b + 2, unitary=u2o_l[b])
            return c.state()

        def apply_layer(psi, layer_gates):
            c = tc.QuditCircuit(n_sites, dim=3, inputs=psi)
            return apply_gates(c, *layer_gates), None

        def state_from_params(params):
            c0 = tc.QuditCircuit(n_sites, dim=3)
            for i in range(1, n_sites, 2):
                c0.unitary(i, unitary=swap02)
            if n_layers == 0:
                return c0.state()
            u1, u2e, u2o = build_gates(params)
            psi = apply_gates(c0, u1[0], u2e[0], u2o[0])
            if n_layers > 1:
                psi, _ = jax.lax.scan(apply_layer, psi, (u1[1:], u2e[1:], u2o[1:]))
            return psi

        def energy_from_state(psi):
            c_eval = tc.QuditCircuit(n_sites, dim=3, inputs=psi)
            total_e = jnp.asarray(0.0, dtype=rdtype)
            for i in range(n_sites - 1):
                total_e = total_e + jnp.real(c_eval.expectation((h_bonds[i], [i, i + 1]), reuse=True))
            return total_e / n_sites

        val_and_grad = jax.value_and_grad(lambda p: energy_from_state(state_from_params(p)))
        optimizer = optax.adam(learning_rate)

        @jax.jit
        def optimize(params0):
            opt_state0 = optimizer.init(params0)

            def step(carry, _):
                params, opt_state = carry
                energy_density, grads = val_and_grad(params)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), energy_density

            (final_params, _), history = jax.lax.scan(step, (params0, opt_state0), None, length=max_steps)
            psi_final = state_from_params(final_params)
            final_energy = energy_from_state(psi_final)
            c_final = tc.QuditCircuit(n_sites, dim=3, inputs=psi_final)
            string_orders = [
                jnp.real(c_final.expectation(*([(sz_j, [i])] + [(exp_ipi_sz, [k]) for k in range(i + 1, j)] + [(sz_j, [j])]), reuse=True))
                for i, j in string_pairs
            ]
            return history, final_energy, jnp.stack(string_orders)

        rng = np.random.default_rng(seed)
        params_init = jnp.asarray(rng.normal(0.0, init_scale, size=(n_layers, n_layer_params)), dtype=rdtype)
        history, final_energy, string_orders = optimize(params_init)
        history_np = np.asarray(history, dtype=np.float64)
        final_energy_val = float(final_energy)
        string_orders_np = np.asarray(string_orders, dtype=np.float64)

    return {
        "energy_density_history": history_np,
        "final_energy_density": final_energy_val,
        "final_string_orders": string_orders_np,
    }
