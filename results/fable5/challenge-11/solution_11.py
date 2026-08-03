"""Challenge 11: spin-1 Haldane-chain VQE with string-order verification.

Native spin-1 simulation on TensorCircuit-NG's QuditCircuit (dim=3, 12
sites). Single-site rotations use the closed spin-1 forms exp(-i phi Sz)
(diagonal) and exp(-i theta Sy) = I - i sin(theta) Sy + (cos(theta)-1) Sy^2
(valid because Sy^3 = Sy for spin 1); each bond gate is the matrix
exponential of the task-defined 9x9 generator theta (SxSx + SySy) +
phi SzSz + beta (S.S)^2. Energy and the nonlocal string correlators are
framework expectation values; Adam runs for exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax
from jax.scipy.linalg import expm

_SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
_SY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
_SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
_I3 = np.eye(3, dtype=complex)
_SDOTS = sum(np.kron(a, a) for a in (_SX, _SY, _SZ))
_A_XY = np.kron(_SX, _SX) + np.kron(_SY, _SY)
_B_ZZ = np.kron(_SZ, _SZ)
_C_BIQ = _SDOTS @ _SDOTS
_STRING_MID = np.diag([-1.0, 1.0, -1.0]).astype(complex)  # exp(i pi Sz)


def _rz(phi):
    return jnp.diag(jnp.stack([jnp.exp(-1j * phi), jnp.ones((), jnp.complex64), jnp.exp(1j * phi)]))


def _ry(theta):
    sy = jnp.asarray(_SY, dtype=jnp.complex64)
    sy2 = jnp.asarray(_SY @ _SY, dtype=jnp.complex64)
    return (jnp.eye(3, dtype=jnp.complex64)
            - 1j * jnp.sin(theta) * sy
            + (jnp.cos(theta) - 1.0) * sy2)


def run_solution(config):
    n = int(config["n_sites"])
    n_layers = int(config["n_layers"])
    beta = float(config["beta"])
    dcoef = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    even_bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]
    a_xy = jnp.asarray(_A_XY, dtype=jnp.complex64)
    b_zz = jnp.asarray(_B_ZZ, dtype=jnp.complex64)
    c_biq = jnp.asarray(_C_BIQ, dtype=jnp.complex64)
    h_bond_m = np.asarray(_SDOTS + beta * _C_BIQ, dtype=np.complex64).reshape(3, 3, 3, 3)
    sz2_m = np.asarray(_SZ @ _SZ, dtype=np.complex64)
    sz_m = np.asarray(_SZ, dtype=np.complex64)
    mid_m = np.asarray(_STRING_MID, dtype=np.complex64)
    gate = tc.gates.Gate  # fresh node per operator occurrence
    swap02 = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.complex64)

    def bond_gate(theta, phi):
        gen = theta * a_xy + phi * b_zz + beta * c_biq
        return expm(-1j * gen)

    def build_state_circuit(params):
        c = tc.QuditCircuit(n, dim=3)
        for q in range(1, n, 2):
            c.any(q, unitary=swap02)  # Neel |+1,-1,...>: odd sites to |-1>
        for l in range(n_layers):
            for q in range(n):
                m = _rz(params["rot"][l, q, 0]) @ _ry(params["rot"][l, q, 1]) @ _rz(params["rot"][l, q, 2])
                c.any(q, unitary=m)
            for k, (a, b) in enumerate(even_bonds):
                c.any(a, b, unitary=bond_gate(params["even"][l, k, 0], params["even"][l, k, 1]))
            for k, (a, b) in enumerate(odd_bonds):
                c.any(a, b, unitary=bond_gate(params["odd"][l, k, 0], params["odd"][l, k, 1]))
        return c

    def energy_density(params):
        c = build_state_circuit(params)
        e = 0.0
        for i in range(n - 1):
            e = e + tc.backend.real(c.expectation((gate(h_bond_m), [i, i + 1])))
        for i in range(n):
            e = e + dcoef * tc.backend.real(c.expectation((gate(sz2_m), [i])))
        return e / n

    def string_orders(params):
        c = build_state_circuit(params)
        vals = []
        for (i, j) in ((0, 11), (1, 10), (2, 9)):
            ops = [(gate(sz_m), [i])] + [(gate(mid_m), [k]) for k in range(i + 1, j)] + [(gate(sz_m), [j])]
            vals.append(tc.backend.real(c.expectation(*ops)))
        return jnp.stack(vals)

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(energy_density)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "rot": scale * jax.random.normal(k1, (n_layers, n, 3), dtype=jnp.float32),
        "even": scale * jax.random.normal(k2, (n_layers, len(even_bonds), 2), dtype=jnp.float32),
        "odd": scale * jax.random.normal(k3, (n_layers, len(odd_bonds), 2), dtype=jnp.float32),
    }
    opt_state = opt.init(params)

    es = []
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), dtype=np.float64)

    e_fin = float(jax.jit(energy_density)(params))
    s_fin = np.asarray(jax.device_get(jax.jit(string_orders)(params)), dtype=np.float64)
    return {
        "energy_density_history": e_hist,
        "final_energy_density": np.float64(e_fin),
        "final_string_orders": s_fin,
    }
