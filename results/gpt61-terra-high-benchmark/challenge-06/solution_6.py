import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')

import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax
from jax.experimental.sparse import BCOO


def run_solution(config):
    n, blocks = config['n_qubits'], config['n_blocks']
    tc.set_backend('numpy')
    def operator(terms, weights):
        strings = []
        for ops in terms:
            s = [0] * n
            for i, p in ops:
                s[i] = p
            strings.append(s)
        return tc.quantum.PauliStringSum2COO(strings, weights, numpy=True).tocsr()
    xyterms = [[(i, p), (i + 1, p)] for i in range(n - 1) for p in (1, 2)]
    zterms = [[(i, 3)] for i in range(n)]
    zzterms = [[(i, 3), (i + 1, 3)] for i in range(n - 1)]
    stagger = [(-1.) ** i for i in range(n)]
    hxy = operator(xyterms, [1.] * len(xyterms))
    hz = operator(zterms, stagger)
    ht = .7 * hxy + 1.1 * operator(zzterms, [1.] * len(zzterms)) + .25 * hz
    tc.set_backend('jax')
    hxy, hz, ht = (BCOO.from_scipy_sparse(x.astype(np.complex64)) for x in (hxy, hz, ht))
    initial = tc.Circuit(n)
    for q in range(1, n, 2):
        initial.x(q)
    initial = initial.state()

    def rhs(y, time, coupling, detuning):
        return -1j * (coupling * (hxy @ y) + detuning * (hz @ y))

    def loss(p):
        def layer(state, pars):
            s, j, d, angles = pars
            t = config['t_min'] + (config['t_max'] - config['t_min']) * jax.nn.sigmoid(s)
            state = tc.timeevol.ode_evol_global(
                rhs, state, jnp.stack([jnp.zeros_like(t), t]), None,
                jnp.tanh(j), jnp.tanh(d), mode='raw',
                ode_backend='diffrax', rtol=config['ode_rtol'], atol=config['ode_atol'],
                max_steps=config['ode_max_steps'], dt0=0.03)[-1]
            c = tc.Circuit(n, inputs=state)
            for q in range(n):
                c.rz(q, theta=angles[q, 2])
                c.ry(q, theta=angles[q, 1])
                c.rz(q, theta=angles[q, 0])
            return c.state(), None
        state, _ = jax.lax.scan(layer, initial, p)
        c = tc.Circuit(n, inputs=state)
        return tc.templates.measurements.operator_expectation(c, ht) / n

    rng = np.random.default_rng(42)
    params = (jnp.zeros(blocks), jnp.full((blocks,), .1), jnp.full((blocks,), .1),
              jnp.asarray(rng.normal(0, .1, (blocks, n, 3)), dtype=jnp.float32))
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)
    @jax.jit
    def train(p, opt_state):
        def step(carry, _):
            p, opt_state = carry
            value, grad = jax.value_and_grad(loss)(p)
            updates, opt_state = optimizer.update(grad, opt_state, p)
            return (optax.apply_updates(p, updates), opt_state), value
        return jax.lax.scan(step, (p, opt_state), None, length=config['max_steps'])
    (params, _), history = train(params, opt_state)
    s, j, d, _ = params
    return {'final_analog_times': np.asarray(config['t_min'] + (config['t_max'] - config['t_min']) * jax.nn.sigmoid(s)),
            'final_analog_couplings': np.asarray(jnp.tanh(j)),
            'final_analog_detunings': np.asarray(jnp.tanh(d)),
            'energy_density_history': np.asarray(history)}
