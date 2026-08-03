"""Challenge 1: DMRG-MPS input with variational brickwork-circuit refinement.

The evaluator-provided quimb DMRG MPS is loaded as the input state of a
TensorCircuit-NG MPS-simulated circuit. A 4-layer brickwork ansatz
(RZ-RY-RZ single-qubit rotations + exp[-i(txx XX + tyy YY + tzz ZZ)] bond
interactions on alternating even/odd bonds) is optimized with Adam for
exactly ``max_steps`` steps. The TFIM energy is evaluated as an
MPS-MPO-MPS contraction through TensorCircuit's QuOperator algebra.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import optax
import tensornetwork as tn
from tensorcircuit import gates as g
from tensorcircuit.quantum import QuVector
from tensornetwork.matrixproductstates.mpo import FiniteMPO

_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _tfim_mpo(n, field):
    # H = -sum_i Z_i Z_{i+1} - field * sum_i X_i as a bond-dimension-3 MPO.
    w = np.zeros((3, 3, 2, 2), dtype=complex)
    w[0, 0] = _I
    w[1, 0] = _Z
    w[2, 0] = -field * _X
    w[2, 1] = -_Z
    w[2, 2] = _I
    tensors = [w[2:3]] + [w] * (n - 2) + [w[:, 0:1]]
    return tc.quantum.tn2qop(FiniteMPO(tensors, backend="jax"))


def _mps_tensors(dmrg_state):
    # quimb MPS -> list of (left, phys, right) site tensors for tc.MPSCircuit.
    mps = dmrg_state.copy()
    mps.permute_arrays("lpr")
    arrays = list(mps.arrays)
    out = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.complex128)
        if i == 0:
            a = a.reshape(1, *a.shape)
        elif i == len(arrays) - 1:
            a = a.reshape(*a.shape, 1)
        out.append(tc.backend.convert_to_tensor(a))
    return out


def _one_q(a, b, c_):
    # RZ(a) then RY(b) then RZ(c) composed into a single 2x2 unitary.
    return g.rz(theta=c_).tensor @ g.ry(theta=b).tensor @ g.rz(theta=a).tensor


def _two_q(txx, tyy, tzz):
    # exp[-i (txx XX + tyy YY + tzz ZZ)]; the three terms commute, and
    # tc's rxx(theta) equals exp(-i theta/2 XX), hence the factor 2.
    return (
        g.rxx(theta=2.0 * txx).tensor.reshape(4, 4)
        @ g.ryy(theta=2.0 * tyy).tensor.reshape(4, 4)
        @ g.rzz(theta=2.0 * tzz).tensor.reshape(4, 4)
    )


def _quvector(tensors):
    ts = list(tensors)
    first = tc.backend.reshape(ts[0], ts[0].shape[1:])
    last = tc.backend.reshape(ts[-1], ts[-1].shape[:-1])
    nodes = [tn.Node(first)] + [tn.Node(t) for t in ts[1:-1]] + [tn.Node(last)]
    e = nodes[0][1]
    for nd in nodes[1:-1]:
        e ^ nd[0]
        e = nd[2]
    e ^ nodes[-1][0]
    return QuVector([nodes[0][0]] + [nd[1] for nd in nodes[1:]])


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["field"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    input_tensors = _mps_tensors(config["dmrg_state"])
    mpo = _tfim_mpo(n, field)
    split = {"max_singular_values": 32}

    def energy_fn(params):
        c = tc.MPSCircuit(n, tensors=input_tensors, split=split)
        p1, p2 = params["p1"], params["p2"]
        for layer in range(n_layers):
            for i in range(n):
                c.any(i, unitary=_one_q(p1[layer, i, 0], p1[layer, i, 1], p1[layer, i, 2]))
            for i in range(layer % 2, n - 1, 2):
                c.any(i, i + 1, unitary=_two_q(p2[layer, i, 0], p2[layer, i, 1], p2[layer, i, 2]))
        v = _quvector(c.get_tensors())
        e = (v.adjoint() @ mpo @ v).eval_matrix()
        # Normalized expectation: MPS truncation slightly shrinks the norm,
        # and the physical variational energy is <H> of the normalized state.
        nrm = (v.adjoint() @ v).eval_matrix()
        return tc.backend.real(e)[0, 0] / tc.backend.real(nrm)[0, 0]

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(energy_fn)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    # Zero initialization keeps the circuit at the identity, so the
    # refinement starts exactly from the supplied DMRG-MPS input state.
    params = {
        "p1": tc.backend.cast(tc.backend.zeros([n_layers, n, 3]), "float64"),
        "p2": tc.backend.cast(tc.backend.zeros([n_layers, n - 1, 3]), "float64"),
    }
    opt_state = opt.init(params)

    history = np.empty(max_steps, dtype=np.float64)
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        history[k] = float(e)

    return {"energy_history": history}
