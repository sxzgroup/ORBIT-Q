import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import optax


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    n, layers = config['n_sites'], config['n_layers']
    beta, anis = config['beta'], config['single_ion_anisotropy']
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.diag([1., 0., -1.])
    xy = np.kron(sx, sx) + np.kron(sy, sy)
    zz = np.kron(sz, sz)
    dot = xy + zz
    square = dot @ dot
    bonds = list(range(0, n-1, 2)) + list(range(1, n-1, 2))
    xy, zz, square = map(jnp.asarray, (xy, zz, square))
    h = dot + beta * np.asarray(square)
    terms = []
    for i in range(n-1):
        left = anis if i == 0 else anis / 2
        right = anis if i == n-2 else anis / 2
        terms.append(jnp.asarray(h + left*np.kron(sz@sz, np.eye(3))
                                + right*np.kron(np.eye(3), sz@sz), dtype=jnp.complex64))
    initial = tc.QuditCircuit(n, dim=3)
    flip = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.complex64)
    for i in range(1, n, 2):
        initial.unitary(i, unitary=flip)
    state0 = initial.state()

    def rotation(p):
        a, b, g = p
        c, s = jnp.cos(b), jnp.sin(b) / jnp.sqrt(2.)
        ry = jnp.array([[(1+c)/2, -s, (1-c)/2],
                        [s, c, -s], [(1-c)/2, s, (1+c)/2]])
        z = jnp.array([1., 0., -1.])
        return jnp.exp(-1j*a*z)[:, None] * ry * jnp.exp(-1j*g*z)[None, :]

    def entangler(p):
        return jl.expm(-1j*(p[0]*xy + p[1]*zz + beta*square))

    def layer(state, p):
        r = jax.vmap(rotation)(p[:3*n].reshape(n, 3))
        u = jax.vmap(entangler)(p[3*n:].reshape(n-1, 2))
        c = tc.QuditCircuit(n, dim=3, inputs=state)
        for k, i in enumerate(bonds):
            gate = u[k]
            if i % 2 == 0:
                gate = gate @ jnp.kron(r[i], r[i+1])
            c.unitary(i, i+1, unitary=gate)
        if n % 2:
            c.unitary(n-1, unitary=r[n-1])
        return c.state(), None

    def circuit(p):
        state, _ = jax.lax.scan(jax.checkpoint(layer), state0, p)
        return tc.QuditCircuit(n, dim=3, inputs=state)

    def energy(p):
        c = circuit(p)
        return jnp.real(sum(c.expectation((op, [i, i+1]))
                            for i, op in enumerate(terms))) / n

    params = jnp.asarray(np.random.default_rng(config['seed']).normal(
        0, config['initial_parameter_scale'], (layers, 5*n-2)), dtype=jnp.float32)
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, os):
        e, g = jax.value_and_grad(energy)(p)
        updates, os = optimizer.update(g, os, p)
        return optax.apply_updates(p, updates), os, e

    history = np.empty(config['max_steps'])
    for k in range(config['max_steps']):
        params, opt_state, e = step(params, opt_state)
        history[k] = float(e)

    @jax.jit
    def final(p):
        c = circuit(p)
        e = jnp.real(sum(c.expectation((op, [i, i+1]))
                         for i, op in enumerate(terms))) / n
        strings = []
        for i, j in [(0, n-1), (1, n-2), (2, n-3)]:
            ops = [(sz, [i]), (sz, [j])]
            ops += [(np.diag([-1., 1., -1.]), [k]) for k in range(i+1, j)]
            strings.append(jnp.real(c.expectation(*ops)))
        return e, jnp.stack(strings)

    e, strings = final(params)
    return {'energy_density_history': history, 'final_energy_density': float(e),
            'final_string_orders': np.asarray(strings)}
