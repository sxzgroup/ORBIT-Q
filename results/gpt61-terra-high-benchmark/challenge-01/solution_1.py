import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import optax
import tensorcircuit as tc
import tensornetwork as tn
from tensorcircuit.quantum import QuOperator
from tensorcircuit.templates.measurements import mpo_expectation


def run_solution(config):
    tc.set_backend('jax')
    tc.set_dtype('complex128')
    n = config['n_qubits']
    field = config['field']
    state = config['dmrg_state']
    arrays = []
    for i in range(n):
        t = state[i]
        inds = ([state.bond(i-1, i)] if i else []) + [state.site_ind(i)] + ([state.bond(i, i+1)] if i < n-1 else [])
        a = np.asarray(t.transpose(*inds).data)
        arrays.append(jnp.asarray(a.reshape((1 if i == 0 else a.shape[0], 2, 1 if i == n-1 else a.shape[-1])), dtype=jnp.complex128))
    eye = np.eye(2); x = np.array([[0., 1.], [1., 0.]]); z = np.diag([1., -1.])
    w = np.zeros((3, 3, 2, 2), dtype=complex)
    w[0,0] = eye; w[0,1] = z; w[0,2] = -field*x; w[1,2] = -z; w[2,2] = eye
    nodes = [tn.Node(jnp.asarray(w[0] if i == 0 else w[:,2] if i == n-1 else w)) for i in range(n)]
    nodes[0][0] ^ nodes[1][0]
    for i in range(1, n-1): nodes[i][1] ^ nodes[i+1][0]
    mpo = QuOperator([node[-2] for node in nodes], [node[-1] for node in nodes])
    def energy(p):
        c = tc.Circuit(n, tensors=arrays)
        for layer in range(config['n_layers']):
            for i in range(n):
                c.rz(i, theta=p[layer, i, 0]); c.ry(i, theta=p[layer, i, 1]); c.rz(i, theta=p[layer, i, 2])
            for i in range(layer % 2, n-1, 2):
                a = p[layer, i, 3:6]
                u = tc.gates.rxx(theta=2*a[0]).tensor.reshape(4, 4) @ tc.gates.ryy(theta=2*a[1]).tensor.reshape(4, 4) @ tc.gates.rzz(theta=2*a[2]).tensor.reshape(4, 4)
                c.any(i, i+1, unitary=u)
        return mpo_expectation(c, mpo)
    opt = optax.adam(config['learning_rate'])
    params = jnp.zeros((config['n_layers'], n, 6), dtype=jnp.float64)
    opt_state = opt.init(params)
    vg = jax.value_and_grad(energy)
    def step(carry, _):
        p, s = carry
        e, grad = vg(p)
        updates, s = opt.update(grad, s, p)
        return (optax.apply_updates(p, updates), s), e
    _, hist = jax.jit(lambda p, s: jax.lax.scan(step, (p, s), None, length=config['max_steps']))(params, opt_state)
    return {'energy_history': np.asarray(hist)}
