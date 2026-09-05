"""Differentiable TensorCircuit simulation of a fixed selected trajectory."""

import numpy as np


def run_solution(config):
    import jax
    import jax.numpy as jnp
    import optax
    import tensorcircuit as tc

    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = int(config["n_qubits"])
    depth = int(config["n_steps"])
    updates = int(config["max_steps"])
    field = float(config["transverse_field"])
    weight = float(config["log_probability_weight"])

    # Only active bonds have parameters; padding is ignored by the odd layer.
    bond_indices, rx_indices, count = [], [], 0
    for t in range(depth):
        nb = len(range(t % 2, n - 1, 2))
        indices = np.zeros((n // 2, 2), dtype=np.int32)
        indices[:nb] = np.arange(count, count + 2 * nb).reshape(nb, 2)
        count += 2 * nb
        bond_indices.append(indices)
        rx_indices.append(np.arange(count, count + n))
        count += n
    bond_indices = jnp.asarray(np.stack(bond_indices))
    rx_indices = jnp.asarray(np.stack(rx_indices))
    initial = tc.Circuit(n)
    for q in range(n):
        initial.h(q)
    initial_state = initial.state()
    projector = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    identity = tc.gates.i().tensor

    def unitary_layer(state, angles, rotations, parity):
        c = tc.Circuit(n, inputs=state)
        for b, q in enumerate(range(parity, n - 1, 2)):
            # XX and ZZ commute. TensorCircuit rotations include a factor 1/2.
            c.rxx(q, q + 1, theta=2 * angles[b, 0])
            c.rzz(q, q + 1, theta=2 * angles[b, 1])
        for q in range(n):
            c.rx(q, theta=rotations[q])
        return c.state()

    def step(state, data):
        t, angles, rotations = data
        state = jax.lax.cond(
            t % 2 == 0,
            lambda s: unitary_layer(s, angles, rotations, 0),
            lambda s: unitary_layer(s, angles, rotations, 1),
            state,
        )
        logs = []
        for q in range(0, n, 2):
            c = tc.Circuit(n, inputs=state)
            p = jnp.real(c.expectation((projector, [q])))
            logs.append(jnp.log(p + 1e-12))
            c.post_select(q, keep=0)
            # Normalize via a circuit operator, without manipulating amplitudes.
            c.any(q, unitary=identity / jnp.sqrt(p))
            state = c.state()
        return state, jnp.stack(logs)

    def objective(params):
        state, logs = jax.lax.scan(
            step, initial_state,
            (jnp.arange(depth), params[bond_indices], params[rx_indices]),
        )
        c = tc.Circuit(n, inputs=state)
        energy = -sum(c.expectation_ps(z=[q, q + 1]) for q in range(n - 1))
        energy -= field * sum(c.expectation_ps(x=[q]) for q in range(n))
        energy = jnp.real(energy) / n
        mean_log = jnp.mean(logs)
        loss = energy - weight * mean_log
        return loss, (energy, jnp.exp(jnp.sum(logs)), mean_log)

    optimizer = optax.adam(float(config["learning_rate"]))
    params = jnp.asarray(np.random.default_rng(3).normal(0, 0.15, count))

    def update(carry, unused):
        params, opt_state = carry
        (loss, metrics), grad = jax.value_and_grad(objective, has_aux=True)(params)
        change, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, change)
        return (params, opt_state), (*metrics, loss)

    @jax.jit
    def train(params):
        return jax.lax.scan(
            update, (params, optimizer.init(params)), None, length=updates
        )[1]

    history = train(params)
    names = ("energy_density_history", "success_probability_history",
             "mean_log_probability_history", "loss_history")
    return {name: np.asarray(values) for name, values in zip(names, history)}
