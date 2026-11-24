import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    tc.set_backend('jax')
    tc.set_contractor('greedy')
    tape = config['gate_tape']
    cones = []
    used_indices = set()
    for coefficient, term in config['pauli_terms']:
        active = {q for _, q in term}
        kept = []
        for gate in reversed(tape):
            qubits = gate[1:-1]
            if active.intersection(qubits):
                kept.append(gate)
                active.update(qubits)
                used_indices.add(gate[-1])
        cones.append((coefficient, term, sorted(active), list(reversed(kept))))
    ids = sorted(used_indices)
    index = {p: i for i, p in enumerate(ids)}
    circuits = []
    for coef, term, qubits, gates in cones:
        local = {q: i for i, q in enumerate(qubits)}
        ops = [(g[0], tuple(local[q] for q in g[1:-1]), index[g[-1]]) for g in gates]
        obs = {axis: [local[q] for a, q in term if a.lower() == axis] for axis in 'xyz'}
        circuits.append((coef, len(qubits), ops, obs))

    def objective(p):
        total = 0.0
        for coef, nq, ops, obs in circuits:
            c = tc.Circuit(nq)
            for q in range(nq):
                c.h(q)
            for name, qubits, i in ops:
                getattr(c, name)(*qubits, theta=p[i])
            total = total + coef * jnp.real(c.expectation_ps(**obs, reuse=False))
        return total

    values_grad = jax.vmap(jax.value_and_grad(objective))
    optimizer = optax.adam(config['learning_rate'])
    initial = np.stack([np.random.default_rng(config['seed'] + 100000 + r).normal(
        0, config['initial_parameter_scale'], config['parameter_count']) for r in range(config['n_restarts'])])[:, ids]
    params = jnp.asarray(initial, dtype=jnp.float32)
    state = optimizer.init(params)
    def step(carry, unused):
        p, state = carry
        value, grad = values_grad(p)
        updates, state = optimizer.update(-grad, state, p)
        return (optax.apply_updates(p, updates), state), value
    @jax.jit
    def optimize(p, state):
        return jax.lax.scan(step, (p, state), None, length=config['max_steps'])[1]
    return {'observable_history': np.asarray(optimize(params, state)).T}
