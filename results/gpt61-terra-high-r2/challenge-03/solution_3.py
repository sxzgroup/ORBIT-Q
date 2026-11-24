import os
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex128')
    jax.config.update('jax_enable_x64', True)
    n = int(config['n_qubits'])
    steps = int(config['n_steps'])
    field = float(config['transverse_field'])
    weight = float(config['log_probability_weight'])
    selected = range(0, n, 2)
    c = tc.Circuit(n)
    for i in range(n):
        c.h(i)
    initial = c.state()

    def layer(state, params, parity):
        c = tc.Circuit(n, inputs=state)
        for k, i in enumerate(range(parity, n - 1, 2)):
            c.rxx(i, i + 1, theta=2 * params[k, 0])
            c.rzz(i, i + 1, theta=2 * params[k, 1])
        for i in range(n):
            c.rx(i, theta=params[i, 2])
        norms = []
        for i in selected:
            c.post_select(i, keep=0)
            v = c.state()
            norms.append(jnp.real(jnp.vdot(v, v)))
        norms = jnp.stack(norms)
        probs = norms / jnp.concatenate((jnp.ones((1,), dtype=norms.dtype), norms[:-1]))
        return v / jnp.sqrt(norms[-1]), probs

    def objective(params):
        def pair(state, p):
            state, l1 = layer(state, p[0], 0)
            state, l2 = layer(state, p[1], 1)
            return state, jnp.concatenate((l1, l2))
        state, probs = jax.lax.scan(pair, initial, params[:steps // 2 * 2].reshape(steps // 2, 2, n, 3))
        if steps % 2:
            state, last = layer(state, params[-1], 0)
            probs = jnp.concatenate((probs.reshape(-1), last))
        c = tc.Circuit(n, inputs=state)
        energy = -sum(jnp.real(c.expectation_ps(z=[i, i+1])) for i in range(n-1))
        energy -= field * sum(jnp.real(c.expectation_ps(x=[i])) for i in range(n))
        energy /= n
        meanlog = jnp.mean(jnp.log(probs + 1e-12))
        probability = jnp.prod(probs)
        return energy - weight * meanlog, (energy, meanlog, probability)

    opt = optax.adam(float(config['learning_rate']))
    params = jax.random.normal(jax.random.PRNGKey(13), (steps, n, 3), dtype=jnp.float64) * 0.05
    opt_state = opt.init(params)
    @jax.jit
    def update(params, opt_state):
        (loss, (energy, meanlog, probability)), grad = jax.value_and_grad(objective, has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, (energy, meanlog, probability, loss)
    history = []
    for _ in range(int(config['max_steps'])):
        params, opt_state, values = update(params, opt_state)
        history.append(values)
    energy, meanlog, probability, loss = (np.asarray(jax.device_get(jnp.stack(x))) for x in zip(*history))
    return {'energy_density_history': energy, 'success_probability_history': probability, 'mean_log_probability_history': meanlog, 'loss_history': loss}
