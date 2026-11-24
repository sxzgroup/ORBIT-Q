import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = config["n_data_qubits"]
    layers = config["n_layers"]
    shots = config["n_trajectories"]
    sites = tuple(range(n))
    rng = np.random.default_rng(config["seed"])
    params = jnp.asarray(rng.normal(0, config["initial_parameter_scale"],
                                    size=(layers, 6, n)))
    uniforms = jnp.asarray(rng.random((shots, layers, n)))
    projectors = [
        tc.quantum.QuVector.from_tensor(jnp.asarray(ket)).projector().eval_matrix()
        for ket in (tc.gates.zero_state, tc.gates.one_state)
    ]

    def prepare_ancillas(old, angles):
        anc = tc.Circuit(n)
        for i in sites:
            anc.select_gate(old[i], [tc.gates.i(), tc.gates.x()], i)
            anc.ry(i, theta=angles[i])
        return anc

    def ladder(c, offset):
        for i in range(offset, offset + n - 1):
            c.cnot(i, i + 1)

    def layer(carry, inputs):
        data, old = carry
        angles, status = inputs
        ry, anc_ry, ent, fb0, fb1, post = angles
        anc = prepare_ancillas(old, anc_ry)
        data_c = tc.Circuit(n, inputs=data)
        for i in sites:
            data_c.ry(i, theta=ry[i])
        joint = tc.Circuit(2 * n, mps_inputs=data_c.quvector() | anc.quvector())
        for i in sites:
            joint.rzz(n + i, i, theta=ent[i])
        ladder(joint, n)

        # The diagonal RZZ gates preserve all ancilla Z probabilities, and
        # the ladder permutes Z-basis strings. Sample the same joint marginal.
        ladder(anc, 0)
        bits, _ = anc.measure_jit(*sites, status=status)
        bits = bits.astype(jnp.int32)
        for i in sites:
            joint.select_gate(bits[i], projectors, n + i)
        for i in sites:
            joint.rzz(n + i, i, theta=jnp.where(bits[i], fb1[i], fb0[i]))
        # Factor out the measured register and normalize the branch. Its bits
        # specify the incoming ancilla ket for the next layer.
        labels = jnp.concatenate((jnp.zeros_like(bits), bits))
        projected = joint.projected_subsystem(labels, left=sites)
        out = tc.Circuit(n, inputs=projected)
        ladder(out, 0)
        for i in sites:
            out.rz(i, theta=post[i])
        return (out.state(), bits), None

    initial = (tc.Circuit(n).state(), jnp.zeros(n, dtype=jnp.int32))

    def trajectory(p, status):
        (data, _), _ = jax.lax.scan(layer, initial, (p, status))
        c = tc.Circuit(n, inputs=data)
        zz = sum(c.expectation_ps(z=[i, i + 1]) for i in range(n - 1))
        x = sum(c.expectation_ps(x=[i]) for i in sites)
        return -(zz + config["transverse_field"] * x).real

    energies = jax.vmap(trajectory, in_axes=(None, 0))
    adam = optax.adam(config["learning_rate"])

    def train(p, status):
        def step(carry, _):
            p, state = carry
            loss, grad = jax.value_and_grad(lambda q: energies(q, status).mean())(p)
            updates, state = adam.update(grad, state, p)
            return (optax.apply_updates(p, updates), state), loss

        (p, _), history = jax.lax.scan(step, (p, adam.init(p)), None,
                                       length=config["max_steps"])
        return history, energies(p, status)

    history, final = jax.jit(train)(params, uniforms)
    return dict(energy_history=np.asarray(history),
                final_trajectory_energies=np.asarray(final))
