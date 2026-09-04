"""Fixed-uniform, projective measurement-feedback VQE in TensorCircuit."""
import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    tc.set_backend("jax")
    jax.config.update("jax_enable_x64", True)
    tc.set_dtype("complex128")
    n = config["n_data_qubits"]
    layers = config["n_layers"]
    batch = config["n_trajectories"]
    rng = np.random.default_rng(config["seed"])
    parameters = jnp.asarray(rng.normal(
        0, config["initial_parameter_scale"], (layers, 6, n)), dtype=jnp.float64)
    uniforms = jnp.asarray(rng.random((batch, layers, n)), dtype=jnp.float64)

    def basis(bits):
        c = tc.Circuit(n)
        for i in range(n):
            c.conditional_gate(bits[i], [tc.gates.i(), tc.gates.x()], i)
        return c.quvector()

    def layer(carry, inputs):
        state, previous_bits = carry
        angles, randoms = inputs
        initial = tc.Circuit(n, inputs=state).quvector() | basis(previous_bits)
        c = tc.Circuit(2 * n, mps_inputs=initial)
        for i in range(n):
            c.ry(i, theta=angles[0, i])
            c.ry(n + i, theta=angles[1, i])
        for i in range(n):
            c.rzz(n + i, i, theta=angles[2, i])
        for i in range(n - 1):
            c.cnot(n + i, n + i + 1)
        bits, _ = c.measure(*range(n, 2 * n), status=randoms)
        bits = bits.astype(jnp.int32)
        measured = basis(bits)
        # Native normalized projection; measured ancillas remain in |bits>.
        outcome = jnp.concatenate((jnp.zeros(n, dtype=jnp.int32), bits))
        collapsed = c.projected_subsystem(outcome, tuple(range(n)))
        initial = tc.Circuit(n, inputs=collapsed).quvector() | measured
        c = tc.Circuit(2 * n, mps_inputs=initial)
        for i in range(n):
            c.conditional_gate(bits[i], [
                tc.gates.rzz(theta=angles[3, i]),
                tc.gates.rzz(theta=angles[4, i])], n + i, i)
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rz(i, theta=angles[5, i])
        # Measured ancillas are product basis states; retain their bits for
        # the next layer, using the native subsystem projection API.
        data = c.projected_subsystem(outcome, tuple(range(n)))
        return (data, bits), None

    initial_state = tc.Circuit(n).state()

    def energy(parameters, randoms):
        (state, _), _ = jax.lax.scan(layer,
            (initial_state, jnp.zeros(n, dtype=jnp.int32)), (parameters, randoms))
        c = tc.Circuit(n, inputs=state)
        value = sum(-jnp.real(c.expectation_ps(z=[i, i + 1]))
                    for i in range(n - 1))
        value -= config["transverse_field"] * sum(
            jnp.real(c.expectation_ps(x=[i])) for i in range(n))
        return value

    energies = jax.vmap(energy, in_axes=(None, 0))

    def objective(parameters):
        return jnp.mean(energies(parameters, uniforms))

    optimizer = optax.adam(config["learning_rate"])

    @jax.jit
    def train(parameters):
        def update(carry, _):
            parameters, opt_state = carry
            value, grad = jax.value_and_grad(objective)(parameters)
            changes, opt_state = optimizer.update(grad, opt_state, parameters)
            parameters = optax.apply_updates(parameters, changes)
            return (parameters, opt_state), value
        (parameters, _), history = jax.lax.scan(update,
            (parameters, optimizer.init(parameters)), None,
            length=config["max_steps"])
        return history, energies(parameters, uniforms)

    history, final = train(parameters)
    return {"energy_history": np.asarray(history),
            "final_trajectory_energies": np.asarray(final)}
