import numpy as np
import tensorcircuit as tc
import tensornetwork as tn
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    n = config['n_qubits']
    chosen = tuple(config['selected_qubits'])
    layers = config['n_layers']
    initial = set(config['initial_ones'])
    x = np.array([[0, 1], [1, 0]], np.complex64)
    z = np.diag([1, -1]).astype(np.complex64)
    eye = np.eye(2, dtype=np.complex64)
    w = np.zeros((3, 3, 2, 2), dtype=np.complex64)
    w[0, 0], w[0, 1], w[0, 2] = eye, z, -config['x_strength'] * x
    w[1, 2], w[2, 2] = -config['zz_strength'] * z, eye
    nodes = [tn.Node(jnp.asarray(w[0] if i == 0 else w[:, 2] if i == n-1 else w)) for i in range(n)]
    nodes[0][0] ^ nodes[1][0]
    for i in range(1, n-1):
        nodes[i][1] ^ nodes[i+1][0]
    out = [nodes[0][1]] + [nodes[i][2] for i in range(1, n-1)] + [nodes[-1][1]]
    ins = [nodes[0][2]] + [nodes[i][3] for i in range(1, n-1)] + [nodes[-1][2]]
    h = tc.quantum.QuOperator(out, ins)

    def energy(p):
        c = tc.Circuit(n)
        for i in initial:
            c.x(i)
        for l in range(layers):
            for i in range(n):
                c.rx(i, theta=p[l, i, 0])
                c.rz(i, theta=p[l, i, 1])
                c.ry(i, theta=p[l, i, 2])
            c.cmz(*chosen)
        return tc.templates.measurements.mpo_expectation(c, h) / n

    opt = optax.adam(config['learning_rate'])
    rng = np.random.default_rng(config['seed'])
    params = jnp.asarray(rng.normal(0, config['initial_parameter_scale'], (layers, n, 3)), dtype=jnp.float32)
    def update(carry, _):
        p, state = carry
        e, g = jax.value_and_grad(energy)(p)
        updates, state = opt.update(g, state, p)
        return (optax.apply_updates(p, updates), state), e
    @jax.jit
    def optimize(p):
        return jax.lax.scan(update, (p, opt.init(p)), None, length=config['max_steps'])
    (params, _), history = optimize(params)
    return {'energy_history': np.asarray(history), 'final_parameters': np.asarray(params)}
