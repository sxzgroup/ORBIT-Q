import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    n = config['n_qubits']
    layers = config['n_layers']
    field = config['transverse_field']
    steps = config['max_steps']
    terms, weights = [], []
    for i in range(n):
        p = [0] * n
        p[i] = 1
        terms.append(p)
        weights.append(-field / n)
    for i in range(n - 1):
        p = [0] * n
        p[i] = p[i + 1] = 3
        terms.append(p)
        weights.append(-1 / n)
    hamiltonian = tc.quantum.PauliStringSum2MVP(terms, weights)
    initial = jnp.ones((2**n,), dtype=jnp.complex64) / jnp.sqrt(float(2**n))
    eye = jnp.eye(2, dtype=jnp.complex64)
    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    zz = jnp.array([1, -1, -1, 1], dtype=jnp.float32)

    def layer(state, params, parity):
        a, b = params
        c = tc.Circuit(n, inputs=state)
        gx = jnp.cosh(a) * eye + jnp.sinh(a) * x
        gz = jnp.diag(jnp.exp(b * zz)).astype(jnp.complex64)
        for i in range(n):
            c.any(i, unitary=gx)
        for i in range(parity, n - 1, 2):
            c.any(i, i + 1, unitary=gz)
        state = c.state()
        return state / jnp.linalg.norm(state)

    def objective(params):
        def block(state, p):
            state = layer(state, p[0], 0)
            state = layer(state, p[1], 1)
            return state, None
        state, _ = jax.lax.scan(jax.checkpoint(block), initial, params.reshape((-1, 2, 2)))
        return jnp.real(jnp.vdot(state, hamiltonian(state)))

    optimizer = optax.adam(config['learning_rate'])
    params = jnp.full((layers, 2), config['initial_filter_strength'], dtype=jnp.float32)
    opt_state = optimizer.init(params)
    value_grad = jax.jit(jax.value_and_grad(objective))
    history = []
    for _ in range(steps):
        value, grad = value_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(value)
    p = np.asarray(params)
    return {'final_a': p[:, 0].reshape(-1, 2), 'final_b': p[:, 1].reshape(-1, 2),
            'energy_density_history': np.asarray(history)}
