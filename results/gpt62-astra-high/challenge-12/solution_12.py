import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _mps_bra(state):
    state = state.copy()
    state.permute_arrays("lrp")
    sites = [tc.gates.Gate(jnp.asarray(state[i].data, dtype=jnp.complex128))
             for i in range(state.L)]
    for i in range(state.L - 1):
        sites[i][-2] ^ sites[i + 1][0]
    return tc.quantum.QuVector([s[-1] for s in sites]).adjoint()


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = config["n_qubits"]
    bonds = [i for layer in range(config["n_layers"])
             for i in range(layer % 2, n - 1, 2)]
    bra = _mps_bra(config["dmrg_state"])
    rng = np.random.default_rng(config["seed"])
    initial = jnp.asarray(rng.normal(0, config["initial_parameter_scale"],
                                     (len(bonds), 15)))
    adam = optax.adam(config["learning_rate"])

    def overlap(params):
        gates = jax.vmap(lambda p: tc.gates.su4(theta=p).tensor)(params)
        circuit = tc.Circuit(n)
        for i in range(1, n, 2):
            circuit.x(i)
        for i, gate in zip(bonds, gates):
            circuit.any(i, i + 1, unitary=gate)
        return (bra @ circuit.quvector()).eval().reshape(())

    def objective(params):
        amplitude = overlap(params)
        fidelity = jnp.real(amplitude * amplitude.conj())
        return 1 - fidelity, fidelity

    def step(carry, _):
        params, moments = carry
        (loss, fidelity), gradient = jax.value_and_grad(objective, has_aux=True)(params)
        updates, moments = adam.update(gradient, moments, params)
        return (optax.apply_updates(params, updates), moments), (loss, fidelity)

    @jax.jit
    def train(params):
        (params, _), history = jax.lax.scan(
            step, (params, adam.init(params)), None, length=config["max_steps"])
        return params, history, jnp.angle(overlap(params))

    params, (loss, fidelity), phase = train(initial)
    return dict(loss_history=np.asarray(loss), fidelity_history=np.asarray(fidelity),
                final_parameters=np.asarray(params).ravel(), final_overlap_phase=float(phase))
