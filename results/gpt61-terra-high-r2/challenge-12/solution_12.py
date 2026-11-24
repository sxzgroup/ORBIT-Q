import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex128')
    jax.config.update('jax_enable_x64', True)
    n = config['n_qubits']
    state = config['dmrg_state']
    tensors = []
    for i in range(n):
        t = state[i]
        phys = state.site_ind(i)
        left = state.bond(i - 1, i) if i else None
        right = state.bond(i, i + 1) if i < n - 1 else None
        inds = [x for x in (left, phys, right) if x is not None]
        a = np.asarray(t.transpose(*inds).data)
        if i == 0:
            a = a[None, ...]
        if i == n - 1:
            a = a[..., None]
        tensors.append(jnp.asarray(a))
    bonds = [(i, i + 1) for layer in range(config['n_layers'])
             for i in range(layer % 2, n - 1, 2)]
    initial = np.random.default_rng(config['seed']).normal(
        0, config['initial_parameter_scale'], (len(bonds), 15))
    bits = ''.join(str(i % 2) for i in range(n))
    def overlap(params):
        mats = jax.vmap(lambda p: tc.gates.su4_gate(p).tensor)(params)
        c = tc.Circuit(n, tensors=tensors)
        for k in range(len(bonds) - 1, -1, -1):
            u = mats[k].reshape(4, 4).conj().T
            c.any(*bonds[k], unitary=u)
        return jnp.conj(c.amplitude(bits))
    def loss(params):
        z = overlap(params)
        return 1.0 - jnp.real(z * jnp.conj(z))
    optimizer = optax.adam(config['learning_rate'])
    def step(carry, _):
        p, state = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, state = optimizer.update(grad, state, p)
        return (optax.apply_updates(p, updates), state), value
    p = jnp.asarray(initial)
    (p, _), history = jax.jit(lambda p: jax.lax.scan(
        step, (p, optimizer.init(p)), None, length=config['max_steps']))(p)
    phase = jnp.angle(jax.jit(overlap)(p))
    history = np.asarray(history)
    return {'loss_history': history, 'fidelity_history': 1 - history,
            'final_parameters': np.asarray(p).reshape(-1),
            'final_overlap_phase': float(phase)}
