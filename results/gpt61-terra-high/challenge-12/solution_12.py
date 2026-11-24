import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    tc.set_backend('jax')
    tc.set_dtype('complex128')
    jax.config.update('jax_enable_x64', True)
    tc.set_contractor('greedy')
    n = config['n_qubits']
    bonds = [(i, i + 1) for layer in range(config['n_layers'])
             for i in range(layer % 2, n - 1, 2)]
    target = tc.quantum.quimb2qop(config['dmrg_state'])
    bits = ''.join(str(i % 2) for i in range(n))
    rng = np.random.default_rng(config['seed'])
    params = jnp.asarray(rng.normal(scale=config['initial_parameter_scale'],
                                    size=(len(bonds), 15)))

    def amplitude(p):
        mats = jax.vmap(lambda t: tc.gates.su4_gate(-t).tensor)(p)
        c = tc.Circuit(n, mps_inputs=target)
        for k in reversed(range(len(bonds))):
            c.any(*bonds[k], unitary=mats[k])
        return jnp.conj(c.amplitude(bits))

    def loss(p):
        a = amplitude(p)
        f = jnp.real(a * jnp.conj(a))
        return 1.0 - f

    opt = optax.adam(config['learning_rate'])
    state = opt.init(params)
    def step(carry, _):
        p, s = carry
        value, grad = jax.value_and_grad(loss)(p)
        updates, s = opt.update(grad, s, p)
        return (optax.apply_updates(p, updates), s), value
    @jax.jit
    def train(p, s):
        return jax.lax.scan(step, (p, s), None, length=config['max_steps'])
    (params, _), losses = train(params, state)
    phase = jnp.angle(jax.jit(amplitude)(params))
    losses = np.asarray(losses)
    return {'loss_history': losses, 'fidelity_history': 1.0 - losses,
            'final_parameters': np.asarray(params).reshape(-1),
            'final_overlap_phase': float(phase)}
