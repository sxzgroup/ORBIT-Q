import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oem
import optax
import quimb.tensor as qtn
import tensorcircuit as tc


def _mps_to_bra(dmrg_state, n_qubits):
    mps = dmrg_state.copy()
    if getattr(mps, "site_ind_id", "k{}") != "k{}":
        mps.site_ind_id = "k{}"
    ordered_tn = qtn.TensorNetwork([mps[i] for i in range(n_qubits)])
    mps_ket = tc.quantum.quimb2qop(ordered_tn)
    for node in mps_ket.nodes:
        node.tensor = tc.backend.cast(node.tensor, tc.dtypestr)
    return mps_ket.adjoint()


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor(
        "custom",
        optimizer=oem.paths.greedy,
        preprocessing=True,
        use_primitives=True,
    )

    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    dmrg_state = config["dmrg_state"]

    bonds = [
        (i, i + 1)
        for layer in range(n_layers)
        for i in range(layer % 2, n_qubits - 1, 2)
    ]
    num_bonds = len(bonds)
    param_count = 15 * num_bonds

    mps_bra = _mps_to_bra(dmrg_state, n_qubits)

    def overlap_fn(params):
        gate_params = jnp.reshape(params, (num_bonds, 15))
        c = tc.Circuit(n_qubits)
        for q in range(1, n_qubits, 2):
            c.x(q)
        for idx, (q0, q1) in enumerate(bonds):
            c.su4(q0, q1, theta=gate_params[idx])
        return (mps_bra @ c.get_quvector()).eval()

    def loss_and_fidelity(params):
        ov = overlap_fn(params)
        fidelity = jnp.real(ov * jnp.conj(ov))
        loss = 1.0 - fidelity
        return loss, fidelity

    value_and_grad_fn = jax.value_and_grad(loss_and_fidelity, has_aux=True)
    optimizer = optax.adam(learning_rate=learning_rate)

    rng = np.random.default_rng(seed)
    init_params = jnp.asarray(
        rng.normal(loc=0.0, scale=init_scale, size=(param_count,)),
        dtype=jnp.float64,
    )

    @jax.jit
    def optimize(params):
        opt_state = optimizer.init(params)

        def step_fn(carry, _):
            curr_params, curr_state = carry
            (loss, fidelity), grads = value_and_grad_fn(curr_params)
            updates, next_state = optimizer.update(grads, curr_state, curr_params)
            next_params = optax.apply_updates(curr_params, updates)
            return (next_params, next_state), (loss, fidelity)

        (final_params, _), (loss_hist, fid_hist) = jax.lax.scan(
            step_fn, (params, opt_state), None, length=max_steps
        )
        final_ov = overlap_fn(final_params)
        final_phase = jnp.angle(final_ov)
        return final_params, loss_hist, fid_hist, final_phase

    final_params, loss_hist, fid_hist, final_phase = optimize(init_params)

    return {
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
        "fidelity_history": np.asarray(fid_hist, dtype=np.float64),
        "final_parameters": np.asarray(final_params, dtype=np.float64),
        "final_overlap_phase": float(np.asarray(final_phase).item()),
    }
