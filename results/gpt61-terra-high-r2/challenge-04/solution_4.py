import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import tensorcircuit as tc

tc.set_backend('jax')
tc.set_dtype('complex128')


def run_solution(config):
    n = config['n_qubits']
    angle = config['entangler_angle']

    def table(p):
        p01, p10 = p
        z = jnp.zeros((), dtype=jnp.complex128)
        k0 = jnp.diag(jnp.array([jnp.sqrt(1-p01), jnp.sqrt(1-p10)], dtype=jnp.complex128))
        k1 = jnp.array([[z, jnp.sqrt(p10)], [z, z]], dtype=jnp.complex128)
        k2 = jnp.array([[z, z], [jnp.sqrt(p01), z]], dtype=jnp.complex128)
        rows = []
        for probe in range(4):
            c = tc.DMCircuit(n)
            if probe == 0:
                c.h(0)
                for i in range(1, n): c.cnot(0, i)
            elif probe == 1:
                for i in range(0, n, 2):
                    c.x(i+1)
                    c.h(i)
                    c.cnot(i, i+1)
            elif probe == 3:
                for i in range(n): c.h(i)
            for start in (0, 1):
                for i in range(start, n-1, 2):
                    c.rxx(i, i+1, theta=angle)
                    c.general_kraus([k0,k1,k2], i)
                    c.general_kraus([k0,k1,k2], i+1)
            rows.append(jnp.stack([jnp.real(c.expectation_ps(z=[i], reuse=False)) for i in range(n)] + [jnp.real(c.expectation_ps(z=list(range(n)), reuse=False))]))
        return jnp.stack(rows)

    target = jax.jit(table)(jnp.array([config['true_p01'],config['true_p10']]))
    initial = jnp.array([config['initial_p01'],config['initial_p10']])
    params = jnp.log(initial / (1-initial))
    def objective(r):
        return jnp.mean((table(jax.nn.sigmoid(r))-target)**2)
    opt = optax.adam(config['learning_rate'])
    state = opt.init(params)
    @jax.jit
    def step(r, state):
        loss, grad = jax.value_and_grad(objective)(r)
        update, state = opt.update(grad, state, r)
        return optax.apply_updates(r, update), state, loss
    history = []
    for _ in range(config['max_steps']):
        params, state, loss = step(params, state)
        history.append(loss)
    p = jax.nn.sigmoid(params)
    return {'loss_history': np.asarray(jnp.stack(history)), 'final_probabilities': np.asarray(p), 'fitted_expectations': np.asarray(jax.jit(table)(p))}
