import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    n = config['n_qubits']
    nl = config['n_layers']
    field = config['transverse_field']
    def layer(state, a, b, parity):
        c = tc.Circuit(n, inputs=state)
        xgate = tc.gates.Gate(jnp.array([[jnp.cosh(a), jnp.sinh(a)], [jnp.sinh(a), jnp.cosh(a)]], dtype=jnp.complex64))
        zzgate = tc.gates.Gate(jnp.diag(jnp.array([jnp.exp(b), jnp.exp(-b), jnp.exp(-b), jnp.exp(b)], dtype=jnp.complex64)))
        for i in range(n):
            c.apply_general_gate(tc.gates.Gate(xgate.tensor), i)
        for i in range(parity, n-1, 2):
            c.apply_general_gate(tc.gates.Gate(zzgate.tensor.reshape(2, 2, 2, 2)), i, i+1)
        state = c.state()
        return state / jnp.linalg.norm(state)
    def loss(params):
        initial = tc.Circuit(n)
        for i in range(n):
            initial.h(i)
        state = initial.state()
        def block(state, p):
            state = layer(state, p[0,0], p[1,0], 0)
            state = layer(state, p[0,1], p[1,1], 1)
            return state, None
        state, _ = jax.lax.scan(block, state, jnp.transpose(params.reshape(2, nl//2, 2), (1,0,2)))
        c = tc.Circuit(n, inputs=state)
        energy = sum(jnp.real(c.expectation_ps(z=[i,i+1])) for i in range(n-1))
        energy += field * sum(jnp.real(c.expectation_ps(x=[i])) for i in range(n))
        return -energy / n
    optimizer = optax.adam(config['learning_rate'])
    params = jnp.full((2,nl), config['initial_filter_strength'], dtype=jnp.float32)
    opt_state = optimizer.init(params)
    @jax.jit
    def update(p, s):
        value, grad = jax.value_and_grad(loss)(p)
        updates, s = optimizer.update(grad, s, p)
        return optax.apply_updates(p, updates), s, value
    history = []
    for _ in range(config['max_steps']):
        params, opt_state, val = update(params, opt_state)
        history.append(val)
    return {'final_a': np.asarray(params[0]).reshape(-1, 2), 'final_b': np.asarray(params[1]).reshape(-1, 2), 'energy_density_history': np.asarray(jnp.stack(history))}
