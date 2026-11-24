import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    jax.config.update('jax_enable_x64', True)
    tc.set_backend('jax')
    tc.set_dtype('complex128')
    n = config['n_qubits']
    steps = config['n_steps']
    field = config['transverse_field']
    weight = config['log_probability_weight']
    nb = n // 2
    # Each row has space for all even bonds, followed by the RX angles.
    initial = np.random.default_rng(17).normal(0, 0.1, (steps, 2 * nb + n))
    params = jnp.asarray(initial)
    init_circuit = tc.Circuit(n)
    for q in range(n):
        init_circuit.h(q)
    initial_state = init_circuit.state()

    def layer(state, row, parity):
        c = tc.Circuit(n, inputs=state)
        for k, q in enumerate(range(parity, n - 1, 2)):
            c.rxx(q, q + 1, theta=2 * row[2 * k])
            c.rzz(q, q + 1, theta=2 * row[2 * k + 1])
        for q in range(n):
            c.rx(q, theta=row[2 * nb + q])
        probabilities = []
        for q in range(0, n, 2):
            c.post_select(q, keep=0)
            state = c.state()
            p = jnp.real(jnp.vdot(state, state))
            probabilities.append(p)
            state = state / jnp.sqrt(p)
            c = tc.Circuit(n, inputs=state)
        return state, jnp.stack(probabilities)

    def objective(parameters):
        def pair(state, rows):
            state, a = layer(state, rows[0], 0)
            state, b = layer(state, rows[1], 1)
            return state, jnp.concatenate((a, b))
        state, probabilities = jax.lax.scan(pair, initial_state, parameters[:steps // 2 * 2].reshape((-1, 2, 2 * nb + n)))
        if steps % 2:
            state, last = layer(state, parameters[-1], 0)
            probabilities = jnp.concatenate((probabilities.reshape(-1), last))
        c = tc.Circuit(n, inputs=state)
        zz = sum(jnp.real(c.expectation((tc.gates.z(), [q]), (tc.gates.z(), [q + 1]))) for q in range(n - 1))
        x = sum(jnp.real(c.expectation((tc.gates.x(), [q]))) for q in range(n))
        energy = (-zz - field * x) / n
        mean_log = jnp.mean(jnp.log(probabilities + 1e-12))
        success_probability = jnp.prod(probabilities)
        loss = energy - weight * mean_log
        return loss, (energy, mean_log, success_probability)

    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)
    value_grad = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def train(parameters, opt_state):
        def update(carry, _):
            parameters, opt_state = carry
            (loss, (energy, mean_log, success_probability)), grad = value_grad(parameters)
            updates, opt_state = optimizer.update(grad, opt_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, opt_state), (energy, mean_log, success_probability, loss)
        return jax.lax.scan(update, (parameters, opt_state), None, length=config['max_steps'])

    _, (energy, mean_log, success_probability, loss) = train(params, opt_state)
    energy, mean_log, success_probability, loss = map(np.asarray, (energy, mean_log, success_probability, loss))
    return {'energy_density_history': energy,
            'success_probability_history': success_probability,
            'mean_log_probability_history': mean_log,
            'loss_history': loss}
