"""Challenge 2: entanglement-profile-constrained VQE on a 12-qubit XXZ chain.

Dense statevector simulation with TensorCircuit-NG (JAX backend). The ansatz
is three even+odd brickwork blocks (RY-RZ rotations + exp[-i(txx XX + tyy YY
+ tzz ZZ)] bond interactions). After each block the half-chain Renyi-2
entropy is computed from the framework's reduced-density-matrix utilities,
and the loss E/n + w * mean((S2 - target)^2) is minimized with Adam for
exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax
from tensorcircuit import gates as g
from tensorcircuit.templates.measurements import operator_expectation


def _xxz_hamiltonian(n, delta, h_stag):
    # H = sum_i (X X + Y Y + delta Z Z) + h_stag * sum_i (-1)^i Z_i
    ls, ws = [], []
    for i in range(n - 1):
        for code, wt in ((1, 1.0), (2, 1.0), (3, delta)):
            s = [0] * n
            s[i] = code
            s[i + 1] = code
            ls.append(s)
            ws.append(wt)
    for i in range(n):
        s = [0] * n
        s[i] = 3
        ls.append(s)
        ws.append(h_stag * ((-1.0) ** i))
    return tc.quantum.PauliStringSum2COO(ls, ws)


def _two_q(txx, tyy, tzz):
    # exp[-i (txx XX + tyy YY + tzz ZZ)]; commuting terms, and tc's
    # rxx(theta) = exp(-i theta/2 XX), hence the factor 2.
    return (
        g.rxx(theta=2.0 * txx).tensor.reshape(4, 4)
        @ g.ryy(theta=2.0 * tyy).tensor.reshape(4, 4)
        @ g.rzz(theta=2.0 * tzz).tensor.reshape(4, 4)
    )


def run_solution(config):
    n = int(config["n_qubits"])
    delta = float(config["zz_anisotropy"])
    h_stag = float(config["staggered_field"])
    n_blocks = int(config["n_layers"]) // 2  # one block = even + odd sublayer
    n_a = int(config["subsystem_size"])
    targets = jnp.asarray(np.asarray(config["target_entropies"], dtype=np.float64))
    w_ent = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    ham = _xxz_hamiltonian(n, delta, h_stag)
    even_bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]
    traced = list(range(n_a, n))  # keep left half A = qubits 0..n_a-1

    def initial_state():
        # |010101...>, qubit 0 (leftmost bit) in |0>
        c = tc.Circuit(n)
        for i in range(1, n, 2):
            c.x(i)
        return c.state()

    psi0 = initial_state()

    def apply_block(state, rot, even, odd):
        # rot: (2, n, 2) [sublayer, qubit, (theta_y, theta_z)]
        c = tc.Circuit(n, inputs=state)
        for i in range(n):
            c.ry(i, theta=rot[0, i, 0])
            c.rz(i, theta=rot[0, i, 1])
        for k, (a, b) in enumerate(even_bonds):
            c.any(a, b, unitary=_two_q(even[k, 0], even[k, 1], even[k, 2]))
        for i in range(n):
            c.ry(i, theta=rot[1, i, 0])
            c.rz(i, theta=rot[1, i, 1])
        for k, (a, b) in enumerate(odd_bonds):
            c.any(a, b, unitary=_two_q(odd[k, 0], odd[k, 1], odd[k, 2]))
        return c.state()

    def metrics(params):
        state = psi0
        ents = []
        for b in range(n_blocks):
            state = apply_block(state, params["rot"][b], params["even"][b], params["odd"][b])
            rho = tc.quantum.reduced_density_matrix(state, cut=traced)
            ents.append(tc.backend.real(tc.quantum.renyi_entropy(rho, 2)))
        ents = jnp.stack(ents)
        c = tc.Circuit(n, inputs=state)
        e_density = tc.backend.real(operator_expectation(c, ham)) / n
        mse = jnp.mean((ents - targets) ** 2)
        loss = e_density + w_ent * mse
        return loss, (e_density, mse, ents)

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        (loss, aux), grad = jax.value_and_grad(metrics, has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    # Gaussian initialization with standard deviation 0.02, as specified.
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    params = {
        "rot": 0.02 * jax.random.normal(k1, (n_blocks, 2, n, 2), dtype=jnp.float64),
        "even": 0.02 * jax.random.normal(k2, (n_blocks, len(even_bonds), 3), dtype=jnp.float64),
        "odd": 0.02 * jax.random.normal(k3, (n_blocks, len(odd_bonds), 3), dtype=jnp.float64),
    }
    opt_state = opt.init(params)

    loss_hist = np.empty(max_steps, dtype=np.float64)
    e_hist = np.empty(max_steps, dtype=np.float64)
    mse_hist = np.empty(max_steps, dtype=np.float64)
    ent_hist = np.empty((max_steps, n_blocks), dtype=np.float64)
    for k in range(max_steps):
        params, opt_state, loss, (e_density, mse, ents) = step(params, opt_state)
        loss_hist[k] = float(loss)
        e_hist[k] = float(e_density)
        mse_hist[k] = float(mse)
        ent_hist[k] = np.asarray(ents)

    return {
        "energy_density_history": e_hist,
        "loss_history": loss_hist,
        "entropy_mse_history": mse_hist,
        "entropy_history": ent_hist,
    }
