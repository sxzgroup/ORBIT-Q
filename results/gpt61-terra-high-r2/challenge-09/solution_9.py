import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import tensorcircuit as tc

    tc.set_backend('jax')
    tape = config['gate_tape']
    cones = []
    all_indices = set()
    for weight, paulis in config['pauli_terms']:
        active = {q for _, q in paulis}
        selected = []
        for gate in reversed(tape):
            qubits = gate[1:-1]
            if any(q in active for q in qubits):
                selected.append(gate)
                active.update(qubits)
        selected.reverse()
        all_indices.update(g[-1] for g in selected)
        cones.append((weight, paulis, sorted(active), selected))
    indices = sorted(all_indices)
    parameter_map = {p: i for i, p in enumerate(indices)}
    circuits = []
    for weight, paulis, qubits, gates in cones:
        qmap = {q: i for i, q in enumerate(qubits)}
        circuits.append((weight, {axis: [qmap[q] for a, q in paulis if a == axis] for axis in 'xyz'},
                         len(qubits), [(g[0], tuple(qmap[q] for q in g[1:-1]), parameter_map[g[-1]]) for g in gates]))

    def objective(parameters):
        value = 0.0
        for weight, obs, n, gates in circuits:
            c = tc.Circuit(n)
            for q in range(n):
                c.h(q)
            for name, qubits, p in gates:
                getattr(c, name)(*qubits, theta=parameters[p])
            value = value + weight * jnp.real(c.expectation_ps(reuse=False, **obs))
        return value

    values_and_grads = jax.vmap(jax.value_and_grad(objective))
    rate = config['learning_rate']
    def update(carry, step):
        params, m, v = carry
        values, grads = values_and_grads(params)
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * grads ** 2
        t = step + 1
        params = params + rate * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        return (params, m, v), values

    initial = np.stack([np.random.default_rng(config['seed'] + 100000 + r).normal(
        0, config['initial_parameter_scale'], config['parameter_count'])[indices]
        for r in range(config['n_restarts'])]).astype(np.float32)
    params = jnp.asarray(initial)
    zeros = jnp.zeros_like(params)
    @jax.jit
    def optimize(p, m, v):
        return jax.lax.scan(update, (p, m, v), jnp.arange(config['max_steps']))[1]
    return {'observable_history': np.asarray(optimize(params, zeros, zeros)).T}
