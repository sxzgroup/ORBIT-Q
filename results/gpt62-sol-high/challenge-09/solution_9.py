import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _causal_cone(gate_tape, paulis):
    qubits = {qubit for _, qubit in paulis}
    gates = []
    # In the Heisenberg picture, a gate disjoint from the current support cancels.
    for name, *sites, parameter in reversed(gate_tape):
        if not qubits.isdisjoint(sites):
            qubits.update(sites)
            gates.append((name, sites, parameter))
    local = {qubit: i for i, qubit in enumerate(sorted(qubits))}
    tape = [
        (name, tuple(local[q] for q in sites), parameter)
        for name, sites, parameter in reversed(gates)
    ]
    operators = [(axis, local[q]) for axis, q in paulis]
    return len(local), tape, operators


def _expectation(theta, cone, parameter_slot):
    size, tape, operators = cone
    if not operators:
        return 1.0
    circuit = tc.Circuit(size)
    # Build the Hadamard at the active precision.
    for qubit in range(size):
        circuit.apply_general_gate(tc.gates.h(), qubit)
    for name, sites, parameter in tape:
        getattr(circuit, name)(*sites, theta=theta[parameter_slot[parameter]])
    paulis = [(getattr(tc.gates, axis)(), [q]) for axis, q in operators]
    # Contract the expectation network directly without forming a cone statevector.
    return jnp.real(circuit.expectation(*paulis, reuse=False))


def run_solution(config):
    restarts = config["n_restarts"]
    steps = config["max_steps"]
    if not restarts or not steps:
        return {"observable_history": np.empty((restarts, steps))}

    cones = [
        (weight, _causal_cone(config["gate_tape"], paulis))
        for weight, paulis in config["pauli_terms"]
    ]
    active = sorted({p for _, (_, tape, _) in cones for _, _, p in tape})
    parameter_slot = {p: i for i, p in enumerate(active)}
    # Draw each full vector first. Parameters outside the cones have zero gradient
    # and stay at their initial values under Adam.
    initial = np.stack([
        np.random.default_rng(config["seed"] + 100000 + r).normal(
            0.0, config["initial_parameter_scale"], config["parameter_count"]
        )[active]
        for r in range(restarts)
    ])

    def loss(theta):
        return -sum(
            weight * _expectation(theta, cone, parameter_slot)
            for weight, cone in cones
        )

    optimizer = optax.adam(config["learning_rate"])
    value_and_grad = jax.vmap(jax.value_and_grad(loss))

    def update(carry, _):
        theta, state = carry
        losses, gradients = value_and_grad(theta)
        updates, state = optimizer.update(gradients, state)
        return (optax.apply_updates(theta, updates), state), -losses

    @jax.jit
    def optimize(theta):
        _, history = jax.lax.scan(
            update, (theta, optimizer.init(theta)), None, length=steps
        )
        return history.T

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        history = np.asarray(optimize(jnp.asarray(initial)))
    return {"observable_history": history}
