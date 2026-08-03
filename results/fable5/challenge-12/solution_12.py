"""Challenge 12: variational circuit to MPS overlap optimization (32 qubits).

The trainable two-layer SU4 brickwork circuit is simulated exactly on
TensorCircuit-NG's MPS simulator (shallow circuit, bond dimension <= 8), and
the loss 1 - |<psi_MPS|psi_circ>|^2 is the direct tensor-network overlap
(proj_with_mps) between the evaluator-provided quimb DMRG-MPS bra (loaded as
framework MPS site tensors, never converted to a preparation circuit) and
the circuit ket. Each SU4 gate is the matrix exponential of 15 su(4) Pauli
generators. Adam runs for exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax
from jax.scipy.linalg import expm

_P = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.diag([1.0, -1.0]).astype(complex),
}
# 15 su(4) generators: all Pauli pairs except identity(x)identity
_GENS = np.stack([
    np.kron(_P[a], _P[b])
    for a in ("i", "x", "y", "z")
    for b in ("i", "x", "y", "z")
    if not (a == "i" and b == "i")
])


def _mps_tensors(dmrg_state):
    # quimb MPS -> (left, phys, right) site tensors for tc.MPSCircuit
    mps = dmrg_state.copy()
    mps.permute_arrays("lpr")
    arrays = list(mps.arrays)
    out = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.complex64)
        if i == 0:
            a = a.reshape(1, *a.shape)
        elif i == len(arrays) - 1:
            a = a.reshape(*a.shape, 1)
        out.append(tc.backend.convert_to_tensor(a))
    return out


def run_solution(config):
    n = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    target_tensors = _mps_tensors(config["dmrg_state"])
    gens = jnp.asarray(_GENS, dtype=jnp.complex64)
    split = {"max_singular_values": 16}
    layer_bonds = [
        [(i, i + 1) for i in range(l % 2, n - 1, 2)] for l in range(n_layers)
    ]
    n_gates = sum(len(b) for b in layer_bonds)

    def su4(theta15):
        return expm(-1j * jnp.tensordot(theta15.astype(jnp.complex64), gens, axes=1))

    def overlap(params):
        c = tc.MPSCircuit(n, split=split)
        for q in range(1, n, 2):
            c.x(q)  # Neel |0101...01>
        k = 0
        for bonds in layer_bonds:
            for (a, b) in bonds:
                c.any(a, b, unitary=su4(params[k]))
                k += 1
        target = tc.MPSCircuit(n, tensors=target_tensors, split=split)
        return c.proj_with_mps(target)  # <psi_MPS | psi_circ>

    def loss_fn(params):
        ov = overlap(params)
        fid = tc.backend.real(ov * jnp.conj(ov))
        return 1.0 - fid, fid

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        (loss, fid), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, fid

    params = scale * jax.random.normal(
        jax.random.PRNGKey(seed), (n_gates, 15), dtype=jnp.float32
    )
    opt_state = opt.init(params)

    losses, fids = [], []
    for k in range(max_steps):
        params, opt_state, loss, fid = step(params, opt_state)
        losses.append(loss)
        fids.append(fid)
    loss_hist = np.asarray(jax.device_get(jnp.stack(losses)), dtype=np.float64)
    fid_hist = np.asarray(jax.device_get(jnp.stack(fids)), dtype=np.float64)

    final_ov = complex(np.asarray(jax.jit(overlap)(params)))
    return {
        "loss_history": loss_hist,
        "fidelity_history": fid_hist,
        "final_parameters": np.asarray(params, dtype=np.float64).reshape(-1),
        "final_overlap_phase": float(np.angle(final_ov)),
    }
