import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
from jax.scipy.linalg import expm


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("greedy")
    n, depth = config["n_sites"], config["n_layers"]
    coupling = config["beta"]
    mag = jnp.array([1., 0., -1.])
    sz = jnp.diag(mag)
    raising = np.diag(np.ones(2), 1)
    sx = (raising + raising.T) / np.sqrt(2)
    sy = (raising - raising.T) / (1j * np.sqrt(2))
    xy = jnp.array((np.kron(sx, sx) + np.kron(sy, sy)).real)
    zz = jnp.kron(sz, sz)
    exchange = xy + zz
    square = exchange @ exchange
    bond_h = exchange + coupling * square
    onsite = config["single_ion_anisotropy"] * sz**2
    even = list(range(0, n - 1, 2))
    odd = list(range(1, n - 1, 2))
    yvals, ybasis = np.linalg.eigh(sy)
    ybasis = jnp.asarray(ybasis)

    def rotation(angles):
        a, b, g = angles
        ry = (ybasis * jnp.exp(-1j * b * yvals)) @ ybasis.conj().T
        return jnp.exp(-1j * a * mag)[:, None] * ry * jnp.exp(-1j * g * mag)

    def entangler(angles):
        t, p = angles
        return expm(-1j * (t * xy + p * zz + coupling * square), max_squarings=8)

    def state(params):
        c = tc.QuditCircuit(n, dim=3)
        for i in range(1, n, 2):
            c.any(i, unitary=jnp.flip(jnp.eye(3), axis=0))
        for layer in params:
            rotations = jax.vmap(rotation)(layer[:3*n].reshape(n, 3))
            gates = jax.vmap(entangler)(layer[3*n:].reshape(-1, 2))
            for b, i in enumerate(even):
                c.any(i, i+1, unitary=gates[b] @ jnp.kron(rotations[i], rotations[i+1]))
            if n % 2:
                c.any(n-1, unitary=rotations[-1])
            for b, i in enumerate(odd):
                c.any(i, i+1, unitary=gates[len(even)+b])
        return c.state()

    def energy(c):
        value = 0.
        for i in range(n-1):
            h = bond_h + jnp.kron(onsite, jnp.eye(3))
            if i == n-2:
                h = h + jnp.kron(jnp.eye(3), onsite)
            value += c.expectation((h, [i, i+1])).real
        return value / n

    def objective(params):
        return energy(tc.QuditCircuit(n, dim=3, inputs=state(params)))

    @jax.jit
    def diagnostics(params):
        c = tc.QuditCircuit(n, dim=3, inputs=state(params))
        strings = []
        parity = jnp.diag(1 - 2 * mag**2)
        for left in range(3):
            right = n - 1 - left
            factors = [(sz, [left]), (sz, [right])]
            factors.extend((parity, [i]) for i in range(left+1, right))
            strings.append(c.expectation(*factors).real)
        return energy(c), jnp.stack(strings)

    shape = (depth, 3*n + 2*(n-1))
    params = jnp.asarray(np.random.default_rng(config["seed"]).normal(
        0., config["initial_parameter_scale"], shape), dtype=jnp.float32)
    adam = optax.adam(config["learning_rate"])

    @jax.jit
    def update(params, moment):
        value, grad = jax.value_and_grad(objective)(params)
        delta, moment = adam.update(grad, moment)
        return optax.apply_updates(params, delta), moment, value

    moment = adam.init(params)
    history = []
    for _ in range(config["max_steps"]):
        params, moment, value = update(params, moment)
        history.append(value)
    final, strings = diagnostics(params)
    return dict(energy_density_history=np.asarray(history),
                final_energy_density=np.asarray(final)[()],
                final_string_orders=np.asarray(strings))
