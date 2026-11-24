import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    n = config['n_data_qubits']
    layers = config['n_layers']
    rng = np.random.default_rng(config['seed'])
    params = jnp.array(rng.normal(0, config['initial_parameter_scale'], (layers, 6, n)), dtype=jnp.float32)
    uniforms = jnp.array(rng.random((config['n_trajectories'], layers, n)), dtype=jnp.float32)
    h = config['transverse_field']

    def trajectory(p, u):
        c = tc.Circuit(2*n)
        for layer in range(layers):
            for i in range(n):
                c.ry(i, theta=p[layer, 0, i])
                c.ry(n+i, theta=p[layer, 1, i])
                c.rzz(n+i, i, theta=p[layer, 2, i])
            for i in range(n-1):
                c.cnot(n+i, n+i+1)
            for i in range(n):
                bit = c.cond_measure(n+i, status=u[layer, i])
                angle = jnp.where(bit == 0, p[layer, 3, i], p[layer, 4, i])
                c.rzz(n+i, i, theta=angle)
            for i in range(n-1):
                c.cnot(i, i+1)
            for i in range(n):
                c.rz(i, theta=p[layer, 5, i])
        e = sum(-c.expectation_ps(z=[i,i+1]) for i in range(n-1))
        e -= h * sum(c.expectation_ps(x=[i]) for i in range(n))
        return jnp.real(e)

    batch = jax.jit(jax.vmap(trajectory, in_axes=(None, 0)))
    def loss(p):
        return jnp.mean(batch(p, uniforms))
    vg = jax.jit(jax.value_and_grad(loss))
    optimizer = optax.adam(config['learning_rate'])
    state = optimizer.init(params)
    history = []
    for _ in range(config['max_steps']):
        energy, grad = vg(params)
        history.append(float(energy))
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
    return {'energy_history': np.array(history), 'final_trajectory_energies': np.asarray(batch(params, uniforms))}
