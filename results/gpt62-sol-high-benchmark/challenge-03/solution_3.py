import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    with jax.enable_x64(), tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        histories = _train(config)
        keys = (
            "energy_density_history",
            "success_probability_history",
            "mean_log_probability_history",
            "loss_history",
        )
        return dict(zip(keys, map(np.asarray, histories)))


def _train(config):
    n = config["n_qubits"]
    steps = config["n_steps"]
    field = config["transverse_field"]
    weight = config["log_probability_weight"]
    even_bonds = tuple(range(0, n - 1, 2))
    odd_bonds = tuple(range(1, n - 1, 2))

    initial = tc.Circuit(n)
    for site in range(n):
        initial.h(site)
    initial_state = initial.state()

    def layer(state, bond_angles, rx_angles, bonds):
        circuit = tc.Circuit(n, inputs=state)
        for i, site in enumerate(bonds):
            # XX and ZZ commute, and TensorCircuit rotations use half angles.
            circuit.rxx(site, site + 1, theta=2 * bond_angles[i, 0])
            circuit.rzz(site, site + 1, theta=2 * bond_angles[i, 1])
        for site in range(n):
            circuit.rx(site, theta=rx_angles[site])

        probabilities = []
        for site in range(0, n, 2):
            circuit.post_select(site, keep=0)
            projected = circuit.state()
            norm = tc.backend.norm(projected)
            probabilities.append(norm ** 2)
            state = projected / norm
            circuit = tc.Circuit(n, inputs=state)
        return state, jnp.stack(probabilities)

    def loss_fn(params):
        even_angles, odd_angles, rx_angles = params

        def pair(state, angles):
            even, odd, rx_even, rx_odd = angles
            state, p_even = layer(state, even, rx_even, even_bonds)
            state, p_odd = layer(state, odd, rx_odd, odd_bonds)
            return state, jnp.stack((p_even, p_odd))

        paired = steps - steps % 2
        state, probabilities = jax.lax.scan(
            pair, initial_state,
            (even_angles[:steps // 2], odd_angles, rx_angles[:paired:2], rx_angles[1:paired:2]),
        )
        probabilities = probabilities.ravel()
        if steps % 2:
            state, last = layer(state, even_angles[-1], rx_angles[-1], even_bonds)
            probabilities = jnp.concatenate((probabilities, last))

        circuit = tc.Circuit(n, inputs=state)
        zz = sum(circuit.expectation_ps(z=[i, i + 1]) for i in range(n - 1))
        x = sum(circuit.expectation_ps(x=[i]) for i in range(n))
        energy = jnp.real(-zz - field * x) / n
        mean_log = jnp.mean(jnp.log(probabilities + 1e-12))
        loss = energy - weight * mean_log
        return loss, (energy, jnp.prod(probabilities), mean_log)

    rng = np.random.default_rng(3)
    shapes = (((steps + 1) // 2, len(even_bonds), 2),
              (steps // 2, len(odd_bonds), 2), (steps, n))
    params = tuple(jnp.asarray(rng.normal(0, 0.15, size=shape)) for shape in shapes)
    adam = optax.adam(config["learning_rate"])
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def update(carry, _):
        params, optimizer_state = carry
        (loss, (energy, probability, mean_log)), grad = value_and_grad(params)
        updates, optimizer_state = adam.update(grad, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return (params, optimizer_state), (energy, probability, mean_log, loss)

    _, histories = jax.lax.scan(update, (params, adam.init(params)), None,
                                length=config["max_steps"])
    return histories
