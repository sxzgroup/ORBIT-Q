"""Spin-1 brickwork VQE using TensorCircuit's qutrit simulator."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    n = config["n_sites"]
    layers = config["n_layers"]
    steps = config["max_steps"]
    beta = config["beta"]
    anisotropy = config["single_ion_anisotropy"]
    even = tuple(range(0, n - 1, 2))
    bonds = even + tuple(range(1, n - 1, 2))

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex64"):
        backend = tc.backend
        sx = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        sy = jnp.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
        m = jnp.array([1., 0., -1.])
        sz = jnp.diag(m)
        sz2 = sz @ sz
        identity = jnp.eye(3)
        exchange_xy = jnp.kron(sx, sx) + jnp.kron(sy, sy)
        exchange_z = jnp.kron(sz, sz)
        exchange = exchange_xy + exchange_z
        biquadratic = beta * (exchange @ exchange)
        interaction = exchange + biquadratic
        field_left = anisotropy * jnp.kron(sz2, identity)
        field_right = anisotropy * jnp.kron(identity, sz2)
        bond_terms = tuple(
            (interaction + (1 if i == 0 else 0.5) * field_left
             + (1 if i == n - 2 else 0.5) * field_right)
            for i in range(n - 1)
        )
        string_phase = jnp.diag(jnp.array([-1., 1., -1.]))

        initial_circuit = tc.QuditCircuit(n, dim=3)
        for i in range(1, n, 2):
            initial_circuit.x(i)
            initial_circuit.x(i)
        initial_state = initial_circuit.state()
        sy2 = sy @ sy

        def rotation(alpha, angle, gamma):
            ry = identity - 1j * jnp.sin(angle) * sy + (jnp.cos(angle) - 1) * sy2
            return jnp.exp(-1j * alpha * m[:, None] - 1j * gamma * m[None, :]) * ry

        def entangler(theta, phi):
            return backend.expm(-1j * (theta * exchange_xy + phi * exchange_z + biquadratic))

        def apply_layer(state, gates):
            rotations, entanglers = gates
            circuit = tc.QuditCircuit(n, dim=3, inputs=state)
            if n % 2:
                circuit.any(n - 1, unitary=rotations[-1])
            for i, bond in enumerate(bonds):
                gate = entanglers[i]
                if i < len(even):
                    # Fuse each pair of single-site rotations into its subsequent gate.
                    gate = gate @ jnp.kron(rotations[bond], rotations[bond + 1])
                circuit.any(bond, bond + 1, unitary=gate)
            return circuit.state(), None

        def state_at(params):
            local = params[:, :3 * n].reshape(-1, 3)
            pairs = params[:, 3 * n:].reshape(-1, 2)
            rotations = jax.vmap(rotation)(*local.T).reshape(layers, n, 3, 3)
            entanglers = jax.vmap(entangler)(*pairs.T).reshape(layers, n - 1, 9, 9)
            state, _ = jax.lax.scan(apply_layer, initial_state, (rotations, entanglers))
            return state

        def energy(state):
            return sum(
                tc.expectation((term, [i, i + 1]), ket=state, dim=3).real
                for i, term in enumerate(bond_terms)
            ) / n

        energy_and_grad = jax.value_and_grad(lambda params: energy(state_at(params)))
        optimizer = optax.adam(config["learning_rate"])

        @jax.jit
        def optimize(params):
            def step(carry, _):
                current, adam_state = carry
                value, gradient = energy_and_grad(current)
                updates, adam_state = optimizer.update(gradient, adam_state, current)
                return (optax.apply_updates(current, updates), adam_state), value

            (params, _), history = jax.lax.scan(
                step, (params, optimizer.init(params)), None, length=steps
            )
            return params, history

        @jax.jit
        def diagnostics(params):
            state = state_at(params)
            strings = [
                tc.expectation(
                    (sz, [i]), *((string_phase, [k]) for k in range(i + 1, j)),
                    (sz, [j]), ket=state, dim=3
                ).real
                for i, j in ((0, n - 1), (1, n - 2), (2, n - 3))
            ]
            return energy(state), jnp.stack(strings)

        rng = np.random.default_rng(config["seed"])
        params = jnp.asarray(rng.normal(
            0, config["initial_parameter_scale"], (layers, 3 * n + 2 * (n - 1))
        ), dtype=jnp.float32)
        params, history = optimize(params)
        final_energy, strings = diagnostics(params)
        return {
            "energy_density_history": np.asarray(history),
            "final_energy_density": np.asarray(final_energy)[()],
            "final_string_orders": np.asarray(strings),
        }
