import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _local_ops(beta, aniso):
    s2 = np.sqrt(2.0)
    sx = (1.0 / s2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.complex64)
    sy = (1.0 / s2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], np.complex64)
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], np.complex64)
    ss = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
    sz2 = sz @ sz
    eye = np.eye(3, dtype=np.complex64)
    h2 = ss + beta * (ss @ ss) + aniso * np.kron(sz2, eye)
    return {
        "sy": jnp.asarray(sy),
        "sz": jnp.asarray(sz),
        "sz2": jnp.asarray(sz2),
        "xy": jnp.asarray(np.kron(sx, sx) + np.kron(sy, sy)),
        "zz": jnp.asarray(np.kron(sz, sz)),
        "ss2": jnp.asarray(ss @ ss),
        "h2": jnp.asarray(h2),
        "ez": jnp.asarray(np.diag([-1, 1, -1]).astype(np.complex64)),
    }


def _neel_state(n_sites):
    psi = np.zeros(3**n_sites, np.complex64)
    idx = 0
    for i in range(n_sites):
        idx = 3 * idx + (0 if i % 2 == 0 else 2)
    psi[idx] = 1.0
    return jnp.asarray(psi)


def _rz(phi):
    phase = jnp.exp(-1j * phi.astype(jnp.complex64))
    zero = jnp.zeros((), jnp.complex64)
    one = jnp.ones((), jnp.complex64)
    return jnp.stack(
        [
            jnp.stack([phase, zero, zero]),
            jnp.stack([zero, one, zero]),
            jnp.stack([zero, zero, jnp.conjugate(phase)]),
        ]
    )


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("plain")

    n = int(config["n_sites"])
    n_layers = int(config["n_layers"])
    beta = float(config["beta"])
    aniso = float(config["single_ion_anisotropy"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    ops = _local_ops(beta, aniso)
    psi0 = _neel_state(n)
    n_even = n // 2
    n_odd = n_even - 1
    n_single = 3 * n
    n_layer = n_single + 2 * (n - 1)
    sy = ops["sy"]
    sy2 = sy @ sy
    ident = jnp.eye(3, dtype=jnp.complex64)
    xy, zz, ss2 = ops["xy"], ops["zz"], ops["ss2"]
    h2, sz, sz2, ez = ops["h2"], ops["sz"], ops["sz2"], ops["ez"]
    beta_c = jnp.complex64(beta)
    aniso_c = jnp.float32(aniso)

    def ry(theta):
        ang = theta.astype(jnp.complex64)
        return ident + (jnp.cos(ang) - 1.0) * sy2 - 1j * jnp.sin(ang) * sy

    def two_gate(theta, phi):
        gen = theta.astype(jnp.complex64) * xy + phi.astype(jnp.complex64) * zz + beta_c * ss2
        return jax.scipy.linalg.expm(-1j * gen)

    def apply_layer(state, layer_params):
        singles = layer_params[:n_single].reshape((n, 3))
        even_p = layer_params[n_single : n_single + 2 * n_even].reshape((n_even, 2))
        odd_p = layer_params[n_single + 2 * n_even :].reshape((n_odd, 2))
        circ = tc.QuditCircuit(n, dim=3, inputs=state)
        for site in range(n):
            unitary = _rz(singles[site, 0]) @ ry(singles[site, 1]) @ _rz(singles[site, 2])
            circ.any(site, unitary=unitary)
        even_u = jax.vmap(two_gate)(even_p[:, 0], even_p[:, 1])
        odd_u = jax.vmap(two_gate)(odd_p[:, 0], odd_p[:, 1])
        for bond in range(n_even):
            circ.any(2 * bond, 2 * bond + 1, unitary=even_u[bond])
        for bond in range(n_odd):
            circ.any(2 * bond + 1, 2 * bond + 2, unitary=odd_u[bond])
        return circ.state(), None

    def circuit_state(params):
        state, _ = jax.lax.scan(apply_layer, psi0, params)
        return state

    def energy_density(params):
        state = circuit_state(params)
        energy = 0.0
        for site in range(n - 1):
            energy = energy + tc.expectation((h2, [site, site + 1]), ket=state, dim=3)
        energy = energy + aniso_c * tc.expectation((sz2, [n - 1]), ket=state, dim=3)
        return jnp.real(energy) / jnp.float32(n)

    def string_orders(params):
        state = circuit_state(params)
        vals = []
        for left, right in ((0, n - 1), (1, n - 2), (2, n - 3)):
            circ = tc.QuditCircuit(n, dim=3, inputs=state)
            for mid in range(left + 1, right):
                circ.any(mid, unitary=ez)
            str_state = circ.state()
            vals.append(
                jnp.real(
                    tc.expectation((sz, [left]), (sz, [right]), ket=str_state, bra=state, dim=3)
                )
            )
        return jnp.stack(vals)

    rng = np.random.default_rng(seed)
    params = jnp.asarray(rng.normal(0.0, scale, size=(n_layers, n_layer)).astype(np.float32))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        curr, state = carry
        value, grad = jax.value_and_grad(energy_density)(curr)
        updates, state = optimizer.update(grad, state, curr)
        curr = optax.apply_updates(curr, updates)
        return (curr, state), value

    @jax.jit
    def train(curr, state):
        return jax.lax.scan(step, (curr, state), xs=None, length=max_steps)

    energy_fn = jax.jit(energy_density)
    string_fn = jax.jit(string_orders)
    (params, _), history = train(params, opt_state)
    final_energy = energy_fn(params)
    final_strings = string_fn(params)
    return {
        "energy_density_history": np.asarray(history, dtype=np.float64),
        "final_energy_density": np.float64(final_energy),
        "final_string_orders": np.asarray(final_strings, dtype=np.float64),
    }
