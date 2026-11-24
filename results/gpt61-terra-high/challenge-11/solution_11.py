import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=[REDACTED] intra_op_parallelism_threads=6')
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    n = int(config['n_sites'])
    layers = int(config['n_layers'])
    beta = config['beta']
    anisotropy = config['single_ion_anisotropy']
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy_np = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    sz_np = np.diag([1, 0, -1])
    xy = jnp.asarray(np.kron(sx, sx) + np.kron(sy_np, sy_np), dtype=jnp.complex64)
    zz = jnp.asarray(np.kron(sz_np, sz_np), dtype=jnp.complex64)
    dot = xy + zz
    dot2 = dot @ dot
    bond_h = dot + beta * dot2
    sz2 = jnp.asarray(sz_np @ sz_np, dtype=jnp.complex64)
    sy = jnp.asarray(sy_np, dtype=jnp.complex64)
    z = jnp.array([1., 0., -1.])
    initial_circuit = tc.QuditCircuit(n, 3)
    for i in range(1, n, 2):
        initial_circuit.x(i)
        initial_circuit.x(i)
    initial = initial_circuit.state()
    bonds = list(range(0, n - 1, 2)) + list(range(1, n - 1, 2))

    def layer(state, p):
        circuit = tc.QuditCircuit(n, 3, inputs=state)
        for i in range(n):
            alpha, theta, gamma = p[3 * i:3 * i + 3]
            rz1 = jnp.diag(jnp.exp(-1j * alpha * z))
            ry = jla.expm(-1j * theta * sy)
            rz2 = jnp.diag(jnp.exp(-1j * gamma * z))
            circuit.any(i, unitary=rz1 @ ry @ rz2)
        for i in bonds:
            theta, phi = p[3 * n + 2 * i:3 * n + 2 * i + 2]
            unitary = jla.expm(-1j * (theta * xy + phi * zz + beta * dot2))
            circuit.any(i, i + 1, unitary=unitary)
        return circuit.state(), None

    def evolve(params):
        state, _ = jax.lax.scan(layer, initial, params)
        return tc.QuditCircuit(n, 3, inputs=state)

    def energy(params):
        circuit = evolve(params)
        result = 0.
        for i in range(n - 1):
            result = result + jnp.real(circuit.expectation((bond_h, [i, i + 1])))
        for i in range(n):
            result = result + anisotropy * jnp.real(circuit.expectation((sz2, [i])))
        return result / n

    optimizer = optax.adam(config['learning_rate'])
    @jax.jit
    def update(params, opt_state):
        value, grad = jax.value_and_grad(energy)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    rng = np.random.default_rng(config['seed'])
    params = jnp.asarray(rng.normal(0, config['initial_parameter_scale'],
                                   (layers, 3 * n + 2 * (n - 1))), dtype=jnp.float32)
    opt_state = optimizer.init(params)
    history = []
    for _ in range(int(config['max_steps'])):
        params, opt_state, value = update(params, opt_state)
        history.append(value)

    @jax.jit
    def final_diagnostics(params):
        circuit = evolve(params)
        result = 0.
        for i in range(n - 1):
            result = result + jnp.real(circuit.expectation((bond_h, [i, i + 1])))
        for i in range(n):
            result = result + anisotropy * jnp.real(circuit.expectation((sz2, [i])))
        string = []
        sz = jnp.asarray(sz_np, dtype=jnp.complex64)
        parity = jnp.diag(jnp.array([-1., 1., -1.], dtype=jnp.complex64))
        for left, right in [(0, n - 1), (1, n - 2), (2, n - 3)]:
            ops = [(sz, [left])] + [(parity, [i]) for i in range(left + 1, right)] + [(sz, [right])]
            string.append(jnp.real(circuit.expectation(*ops)))
        return result / n, jnp.stack(string)

    final_energy, strings = final_diagnostics(params)
    return {'energy_density_history': np.asarray(jnp.stack(history)),
            'final_energy_density': float(final_energy),
            'final_string_orders': np.asarray(strings)}
