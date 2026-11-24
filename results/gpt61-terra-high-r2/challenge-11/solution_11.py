import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    n = int(config['n_sites'])
    layers = int(config['n_layers'])
    steps = int(config['max_steps'])
    beta = float(config['beta'])
    anisotropy = float(config['single_ion_anisotropy'])
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex64) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex64) / np.sqrt(2)
    sz = np.diag([1, 0, -1]).astype(np.complex64)
    xy = jnp.asarray(np.kron(sx, sx) + np.kron(sy, sy))
    zz = jnp.asarray(np.kron(sz, sz))
    dot = xy + zz
    square = dot @ dot
    bond = tc.gates.Gate(jnp.reshape(dot + beta * square, (3, 3, 3, 3)))
    onsite = tc.gates.Gate(jnp.asarray(sz @ sz))
    z = jnp.asarray([1., 0., -1.])
    flip = jnp.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=jnp.complex64)
    bond_order = list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))

    def make_gates(params):
        a = params[:, :3*n].reshape(layers, n, 3)
        b = params[:, 3*n:].reshape(layers, n-1, 2)
        t = a[:, :, 1]
        c = jnp.cos(t)
        s = jnp.sin(t) / jnp.sqrt(2.)
        v, w = (1+c)/2, (1-c)/2
        ry = jnp.stack((v, -s, w, s, c, -s, w, s, v), axis=-1).reshape(layers, n, 3, 3)
        rotations = jnp.exp(-1j*a[:, :, 0, None]*z)[..., :, None] * ry * jnp.exp(-1j*a[:, :, 2, None]*z)[..., None, :]
        generators = -1j * (b[..., 0, None, None]*xy + b[..., 1, None, None]*zz + beta*square)
        entanglers = jax.vmap(jax.vmap(jsp.linalg.expm))(generators)
        return rotations, entanglers

    def make_circuit(params):
        rotations, entanglers = make_gates(params)
        initial = tc.QuditCircuit(n, 3)
        for site in range(1, n, 2):
            initial.any(site, unitary=flip)
        state = initial.state()
        def layer(state, gates):
            r, g = gates
            circuit = tc.QuditCircuit(n, 3, inputs=state)
            for site in range(n):
                circuit.any(site, unitary=r[site])
            for site in bond_order:
                circuit.any(site, site+1, unitary=g[site])
            return circuit.state(), None
        state, _ = jax.lax.scan(layer, state, (rotations, entanglers))
        return tc.QuditCircuit(n, 3, inputs=state)

    def energy(params):
        circuit = make_circuit(params)
        value = sum(circuit.expectation((bond, [i, i+1])) for i in range(n-1))
        value += anisotropy * sum(circuit.expectation((onsite, [i])) for i in range(n))
        return jnp.real(value) / n

    @jax.jit
    def update(params, m, v, step):
        value, grad = jax.value_and_grad(energy)(params)
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*grad*grad
        mh = m / (1 - 0.9**step)
        vh = v / (1 - 0.999**step)
        params = params - config['learning_rate'] * mh / (jnp.sqrt(vh) + 1e-8)
        return params, m, v, value

    @jax.jit
    def diagnostics(params):
        circuit = make_circuit(params)
        end = jnp.asarray(sz)
        middle = jnp.diag(jnp.asarray([-1., 1., -1.], dtype=jnp.complex64))
        strings = []
        for start, stop in ((0, n-1), (1, n-2), (2, n-3)):
            ops = [(tc.gates.Gate(end), [start])] + [(tc.gates.Gate(middle), [k]) for k in range(start+1, stop)] + [(tc.gates.Gate(end), [stop])]
            strings.append(jnp.real(circuit.expectation(*ops)))
        return jnp.stack(strings)

    rng = np.random.default_rng(config['seed'])
    shape = (layers, 3*n + 2*(n-1))
    params = jnp.asarray(rng.normal(0, config['initial_parameter_scale'], shape), dtype=jnp.float32)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    history = np.empty(steps, dtype=np.float64)
    for k in range(steps):
        params, m, v, value = update(params, m, v, k+1)
        history[k] = value
    final_energy = float(jax.jit(energy)(params))
    strings = np.asarray(diagnostics(params))
    return {'energy_density_history': history, 'final_energy_density': final_energy, 'final_string_orders': strings}
