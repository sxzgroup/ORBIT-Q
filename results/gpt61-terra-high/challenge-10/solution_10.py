import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
import tensornetwork as tn


def run_solution(config):
    tc.set_backend('jax')
    n = config['n_qubits']
    layers = config['n_layers']
    selected = tuple(config['selected_qubits'])
    x = np.array([[0, 1], [1, 0]], np.complex64)
    z = np.diag([1, -1]).astype(np.complex64)
    identity = np.eye(2, dtype=np.complex64)
    # Bond-dimension-three MPO for the open-boundary TFIM Hamiltonian.
    block = np.zeros((3, 3, 2, 2), np.complex64)
    block[0, 0] = identity
    block[0, 1] = z
    block[0, 2] = -config['x_strength'] * x
    block[1, 2] = -config['zz_strength'] * z
    block[2, 2] = identity

    def objective(p):
        nodes = [tn.Node(jnp.asarray(block[0]))]
        nodes += [tn.Node(jnp.asarray(block)) for _ in range(n - 2)]
        nodes += [tn.Node(jnp.asarray(block[:, 2]))]
        for i in range(n - 1):
            (nodes[i][0] if i == 0 else nodes[i][1]) ^ nodes[i + 1][0]
        outs = [node[1] if i in (0, n - 1) else node[2]
                for i, node in enumerate(nodes)]
        ins = [node[2] if i in (0, n - 1) else node[3]
               for i, node in enumerate(nodes)]
        h = tc.quantum.QuOperator(outs, ins)
        c = tc.Circuit(n)
        for i in config['initial_ones']:
            c.x(i)
        for layer in range(layers):
            for i in range(n):
                c.rx(i, theta=p[layer, i, 0])
                c.rz(i, theta=p[layer, i, 1])
                c.ry(i, theta=p[layer, i, 2])
            c.cmz(*selected)
        return tc.templates.measurements.mpo_expectation(c, h) / n

    p = jnp.asarray(np.random.default_rng(config['seed']).normal(0, config['initial_parameter_scale'], (layers, n, 3)), dtype=jnp.float32)
    optimizer = optax.adam(config['learning_rate'])
    state = optimizer.init(p)
    @jax.jit
    def step(p, state):
        value, grad = jax.value_and_grad(objective)(p)
        updates, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, updates), state, value
    history = []
    for _ in range(config['max_steps']):
        p, state, value = step(p, state)
        history.append(value)
    return {'energy_history': np.asarray(jnp.stack(history)), 'final_parameters': np.asarray(p)}
