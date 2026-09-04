"""Exact local tensor networks with batched, differentiable Adam updates."""

import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import tensorcircuit as tc

    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")
    tape = config["gate_tape"]
    cones = []
    active = set()
    for weight, term in config["pauli_terms"]:
        support = {q for _, q in term}
        selected = []
        for gate in reversed(tape):
            if support.intersection(gate[1:-1]):
                selected.append(gate)
                support.update(gate[1:-1])
                active.add(gate[-1])
        mapping = {q: i for i, q in enumerate(sorted(support))}
        cones.append((weight, term, mapping, selected[::-1]))
    indices = np.array(sorted(active), dtype=np.int32)
    positions = {int(p): i for i, p in enumerate(indices)}

    def objective(theta):
        value = jnp.array(0.0, dtype=jnp.float64)
        for weight, term, mapping, selected in cones:
            circuit = tc.Circuit(len(mapping))
            for q in range(len(mapping)):
                circuit.apply_general_gate(tc.gates.h(), q)
            for gate in selected:
                getattr(circuit, gate[0])(
                    *[mapping[q] for q in gate[1:-1]],
                    theta=theta[positions[gate[-1]]],
                )
            ops = [(getattr(tc.gates, axis)(), [mapping[q]]) for axis, q in term]
            value = value + weight * jnp.real(circuit.expectation(*ops, reuse=False))
        return value

    # Draw every full vector before projection, preserving the specified RNG stream.
    # Omitted coordinates have exactly zero objective gradient and Adam update.
    initial = np.stack([
        np.random.default_rng(config["seed"] + 100000 + r).normal(
            0.0, config["initial_parameter_scale"], config["parameter_count"]
        )[indices]
        for r in range(config["n_restarts"])
    ])
    value_grad = jax.vmap(jax.value_and_grad(objective))

    def step(carry, t):
        theta, m, v = carry
        values, gradient = value_grad(theta)
        gradient = -gradient  # Minimize loss = -objective.
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        mhat = m / (1.0 - 0.9**t)
        vhat = v / (1.0 - 0.999**t)
        theta = theta - config["learning_rate"] * mhat / (jnp.sqrt(vhat) + 1e-8)
        return (theta, m, v), values

    @jax.jit
    def optimize(theta):
        zeros = jnp.zeros_like(theta)
        _, history = jax.lax.scan(
            step, (theta, zeros, zeros),
            jnp.arange(1, config["max_steps"] + 1, dtype=jnp.float64),
        )
        return history.T

    return {"observable_history": np.asarray(optimize(jnp.asarray(initial)))}
