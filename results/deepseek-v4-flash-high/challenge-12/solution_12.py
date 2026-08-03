import numpy as np

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    n = config["n_qubits"]
    n_layers = config["n_layers"]
    max_steps = config["max_steps"]
    learning_rate = config["learning_rate"]
    seed = config["seed"]
    scale = config["initial_parameter_scale"]

    # Alternating brickwork bonds: layer l acts on bonds (2j+l, 2j+l+1).
    bonds = [
        (2 * j + l, 2 * j + l + 1)
        for l in range(n_layers)
        for j in range(n // 2)
        if 2 * j + l + 1 < n
    ]
    n_params = len(bonds) * 15

    # Target MPS (quimb canonical shape (left, right, phys)) -> (left, phys, right).
    mps = config["dmrg_state"]
    tensors = [np.asarray(t.data) for t in mps.tensors]
    bra_tensors = []
    for i, a in enumerate(tensors):
        if a.ndim == 2:  # boundary tensor without the empty bond
            a = np.transpose(a)  # (phys, bond)
            if i == 0:
                a = a[None, :, :]  # (1, phys, bond)
            else:
                a = a[:, :, None]  # (bond, phys, 1)
        elif a.shape[1] != 2:
            a = np.transpose(a, (0, 2, 1))
        bra_tensors.append(a)
    bra = tc.MPSCircuit(n, tensors=bra_tensors, center_position=0)

    # Neel |0101...01> product-state MPS.
    neel = []
    for i in range(n):
        t = np.zeros((1, 2, 1), dtype=np.complex128)
        t[0, i % 2, 0] = 1.0
        neel.append(t)

    def overlap_fn(theta):
        ket = tc.MPSCircuit(n, tensors=neel)
        for k, (i1, i2) in enumerate(bonds):
            ket.apply_double_gate(tc.gates.su4_gate(theta[k]), i1, i2)
        return ket.proj_with_mps(bra)

    def loss_fn(theta):
        ov = overlap_fn(theta)
        return 1.0 - jnp.real(ov * jnp.conj(ov))

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    overlap_jit = jax.jit(overlap_fn)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.uniform(-scale, scale, size=(len(bonds), 15)))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)

    loss_history = np.empty(max_steps)
    fidelity_history = np.empty(max_steps)
    for step in range(max_steps):
        loss, grad = loss_and_grad(theta)
        loss_history[step] = float(loss)
        fidelity_history[step] = 1.0 - float(loss)
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

    final_parameters = np.asarray(theta).reshape(-1).copy()
    final_overlap = complex(np.asarray(overlap_jit(theta)))
    final_overlap_phase = float(np.angle(final_overlap))

    return {
        "loss_history": loss_history,
        "fidelity_history": fidelity_history,
        "final_parameters": final_parameters,
        "final_overlap_phase": final_overlap_phase,
    }
