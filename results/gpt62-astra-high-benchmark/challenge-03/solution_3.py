import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = config["n_qubits"]
    steps = config["n_steps"]
    pairs = n // 2
    field = config["transverse_field"]
    weight = config["log_probability_weight"]

    # At even site 2*i the odd partner is 2*i+1 or 2*i-1, by layer.
    # In odd layers the two open ends share a circuit with no interaction.
    layer = np.arange(steps)[:, None]
    site = np.arange(pairs)[None, :]
    partners = jnp.asarray((site - layer % 2) % pairs)
    active = (layer % 2 == 0) | (site != 0)
    bond_count = int(active.sum())
    bond_ids = np.full(active.shape, bond_count)
    bond_ids[active] = np.arange(bond_count)

    plus = tc.Circuit(2)
    plus.h(0)
    plus.h(1)
    initial = plus.state()
    initial = jnp.stack([initial / jnp.linalg.norm(initial)] * pairs)

    def select_pair(args):
        state, angles, rx_even, rx_odd = args
        c = tc.Circuit(2, inputs=state)
        # XX and ZZ commute. TensorCircuit rotations use half angles.
        c.rxx(0, 1, theta=2 * angles[0])
        c.rzz(0, 1, theta=2 * angles[1])
        c.rx(0, theta=rx_even)
        c.rx(1, theta=rx_odd)
        c.mid_measurement(0, keep=0)
        projected = c.state()
        prob = jnp.vdot(projected, projected).real
        return projected / jnp.sqrt(prob), prob

    def advance(states, inputs):
        angles, rx, partner = inputs
        # Each state follows its odd site. All even sites enter in the same
        # product state, so re-pairing only routes whole two-qubit circuits.
        ordered = states[partner]
        selected, probs = jax.lax.map(
            select_pair, (ordered, angles, rx[::2], rx[1::2][partner])
        )
        return states.at[partner].set(selected), probs

    def observables(state):
        c = tc.Circuit(2, inputs=state)
        x = c.expectation_ps(x=[0]) + c.expectation_ps(x=[1])
        return x.real, c.expectation_ps(z=[0]).real, c.expectation_ps(z=[1]).real

    def objective(params):
        bonds, rx = params
        # A zero angle gives the identity on the disconnected endpoints.
        angles = jnp.pad(bonds, ((0, 1), (0, 0)))[bond_ids]
        states, probs = jax.lax.scan(advance, initial, (angles, rx, partners))
        x, z_even, z_odd = jax.vmap(observables)(states)
        # After the projections all sites are separable. Every even spin is
        # up, so the pair states can again be read in canonical (2i,2i+1) order.
        zz = jnp.dot(z_even, z_odd) + jnp.dot(z_odd[:-1], z_even[1:])
        energy = -(zz + field * x.sum()) / n
        mean_log = jnp.log(probs + 1e-12).mean()
        loss = energy - weight * mean_log
        return loss, (energy, jnp.prod(probs), mean_log)

    rng = np.random.default_rng(3)
    params = tuple(jnp.asarray(rng.normal(0, 0.15, shape))
                   for shape in ((bond_count, 2), (steps, n)))
    adam = optax.adam(config["learning_rate"])
    differentiate = jax.value_and_grad(objective, has_aux=True)

    def update(carry, _):
        params, opt_state = carry
        (loss, (energy, success, mean_log)), grad = differentiate(params)
        updates, opt_state = adam.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (energy, success, mean_log, loss)

    _, history = jax.lax.scan(update, (params, adam.init(params)), None,
                              length=config["max_steps"])
    names = ("energy_density_history", "success_probability_history",
             "mean_log_probability_history", "loss_history")
    return dict(zip(names, map(np.asarray, history)))
