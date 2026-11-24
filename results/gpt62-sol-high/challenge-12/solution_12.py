import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _mps_bra(state):
    state = state.copy()
    state.permute_arrays("lpr")
    tensors = [np.asarray(state[site].data) for site in range(state.L)]
    tensors[0] = tensors[0][None, ...]
    tensors[-1] = tensors[-1][..., None]
    return tc.Circuit(state.L, tensors=tensors).quvector().adjoint()


def run_solution(config):
    n_qubits = config["n_qubits"]
    bonds = [
        (site, site + 1)
        for layer in range(config["n_layers"])
        for site in range(layer % 2, n_qubits - 1, 2)
    ]
    rng = np.random.default_rng(config["seed"])
    initial = rng.normal(scale=config["initial_parameter_scale"], size=(len(bonds), 15))
    adam = optax.adam(config["learning_rate"])

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"), tc.runtime_contractor(
        "greedy", preprocessing=True
    ):
        bra = _mps_bra(config["dmrg_state"])
        gate_tensors = jax.vmap(lambda angles: tc.gates.su4_gate(angles).tensor)

        def overlap(parameters):
            circuit = tc.Circuit(n_qubits)
            for site in range(1, n_qubits, 2):
                circuit.x(site)
            for (left, right), gate in zip(bonds, gate_tensors(parameters)):
                circuit.any(left, right, unitary=gate)
            return (bra @ circuit.quvector()).eval()

        def loss(parameters):
            amplitude = overlap(parameters)
            fidelity = amplitude.real**2 + amplitude.imag**2
            return 1.0 - fidelity, fidelity

        value_and_grad = jax.value_and_grad(loss, has_aux=True)

        def step(carry, _):
            parameters, optimizer_state = carry
            metrics, gradient = value_and_grad(parameters)
            updates, optimizer_state = adam.update(gradient, optimizer_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, optimizer_state), metrics

        @jax.jit
        def train(parameters):
            (parameters, _), histories = jax.lax.scan(
                step, (parameters, adam.init(parameters)), None, length=config["max_steps"]
            )
            return parameters, histories, jnp.angle(overlap(parameters))

        parameters, (losses, fidelities), phase = train(jnp.asarray(initial))
        return {
            "loss_history": np.asarray(losses),
            "fidelity_history": np.asarray(fidelities),
            "final_parameters": np.asarray(parameters).ravel(),
            "final_overlap_phase": float(phase),
        }
