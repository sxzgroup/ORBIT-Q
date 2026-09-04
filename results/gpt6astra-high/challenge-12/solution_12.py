import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")
    n = config["n_qubits"]
    target = config["dmrg_state"]
    tensors = []
    for i in range(n):
        site = target[i]
        inds = ([target.bond(i - 1, i)] if i else [])
        inds += [target.site_ind(i)]
        inds += ([target.bond(i, i + 1)] if i < n - 1 else [])
        a = np.asarray(site.transpose(*inds).data)
        tensors.append(a.reshape((a.shape[0] if i else 1, 2,
                                  a.shape[-1] if i < n - 1 else 1)))
    bra = tc.Circuit(n, tensors=tensors).quvector().adjoint()
    bonds = [(i, i + 1) for layer in range(config["n_layers"])
             for i in range(layer % 2, n - 1, 2)]
    gate_batch = jax.vmap(lambda p: tc.gates.su4(theta=p).tensor)

    def overlap(parameters):
        circuit = tc.Circuit(n)
        for i in range(1, n, 2):
            circuit.x(i)
        gates = gate_batch(parameters.reshape((-1, 15)))
        for k, (i, j) in enumerate(bonds):
            circuit.any(i, j, unitary=gates[k])
        return jnp.reshape((bra @ circuit.quvector()).eval(), ())

    def objective(parameters):
        z = overlap(parameters)
        fidelity = jnp.real(z * jnp.conj(z))
        return 1.0 - fidelity, fidelity

    optimizer = optax.adam(config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    parameters = jnp.asarray(rng.normal(0, config["initial_parameter_scale"],
                                       15 * len(bonds)))
    value_grad = jax.value_and_grad(objective, has_aux=True)

    def update(carry, _):
        p, state = carry
        (loss, fidelity), grad = value_grad(p)
        updates, state = optimizer.update(grad, state, p)
        p = optax.apply_updates(p, updates)
        return (p, state), (loss, fidelity)

    @jax.jit
    def train(p):
        return jax.lax.scan(update, (p, optimizer.init(p)), None,
                            length=config["max_steps"])

    (parameters, _), (losses, fidelities) = train(parameters)
    phase = jnp.angle(jax.jit(overlap)(parameters))
    return {"loss_history": np.asarray(losses),
            "fidelity_history": np.asarray(fidelities),
            "final_parameters": np.asarray(parameters),
            "final_overlap_phase": float(phase)}
