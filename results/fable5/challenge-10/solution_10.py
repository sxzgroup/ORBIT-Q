"""Challenge 10: 22-qubit VQE with an 18-qubit controlled-Z hyperedge.

TensorCircuit-NG dense statevector simulation (JAX backend). The fixed
18-qubit controlled-Z is applied through the framework's ``multicontrol``
gate, which acts in MPO form (17 controls + Z target), so no dense
2^18 x 2^18 matrix and no long decomposition into small gates is ever
built. Each layer applies a trainable RX-RZ-RY block on every qubit (the
three rotations are composed at the 2x2 gate level into one matrix before
application, which is exactly the same layer unitary) followed by the
controlled-Z hyperedge. The TFIM energy is evaluated through the framework's
sparse Pauli-sum operator (matrix-vector products, no dense Hamiltonian) and
minimized with Adam for exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax
import jax.numpy as jnp
import optax
import numpy as _np
from tensornetwork.matrixproductstates.mpo import FiniteMPO
from tensorcircuit import gates as g
from tensorcircuit.templates.measurements import mpo_expectation

_X = _np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = _np.diag([1.0, -1.0]).astype(complex)
_I = _np.eye(2, dtype=complex)


def _tfim_mpo(n, zz, hx):
    # H = -zz * sum Z_i Z_{i+1} - hx * sum X_i as a bond-dimension-3 MPO
    # (memory-light operator representation; expectation via matrix-vector
    # style MPO-state contraction, no dense Hamiltonian is ever built)
    w = _np.zeros((3, 3, 2, 2), dtype=complex)
    w[0, 0] = _I
    w[1, 0] = _Z
    w[2, 0] = -hx * _X
    w[2, 1] = -zz * _Z
    w[2, 2] = _I
    tensors = [w[2:3]] + [w] * (n - 2) + [w[:, 0:1]]
    return tc.quantum.tn2qop(FiniteMPO(tensors, backend="jax"))


def _rot_block(rx, rz, ry):
    # apply RX then RZ then RY on one qubit; composed as a single 2x2 unitary
    return (
        g.ry(theta=ry).tensor
        @ g.rz(theta=rz).tensor
        @ g.rx(theta=rx).tensor
    )


def run_solution(config):
    n = int(config["n_qubits"])
    selected = [int(q) for q in config["selected_qubits"]]
    ones = [int(q) for q in config["initial_ones"]]
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    ham = _tfim_mpo(n, float(config["zz_strength"]), float(config["x_strength"]))

    def initial_state():
        c = tc.Circuit(n)
        for q in ones:
            c.x(q)
        return c.state()

    psi0 = initial_state()
    zmat = g._z_matrix

    def energy_density(params):
        c = tc.Circuit(n, inputs=psi0)
        for l in range(n_layers):
            for q in range(n):
                c.any(q, unitary=_rot_block(params[l, q, 0], params[l, q, 1], params[l, q, 2]))
            # fixed 18-qubit controlled-Z hyperedge in MPO form
            c.multicontrol(*selected, ctrl=[1] * (len(selected) - 1), unitary=zmat)
        return tc.backend.real(mpo_expectation(c, ham)) / n

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(energy_density)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    params = scale * jax.random.normal(
        jax.random.PRNGKey(seed), (n_layers, n, 3), dtype=jnp.float32
    )
    opt_state = opt.init(params)

    es = []
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), dtype=np.float64)

    return {
        "energy_history": e_hist,
        "final_parameters": np.asarray(params, dtype=np.float64),
    }
