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
    batch = config["n_trajectories"]
    sites = tuple(range(n))
    rng = np.random.default_rng(config["seed"])
    angles = jnp.asarray(rng.normal(0, config["initial_parameter_scale"], (layers, 6, n)))
    uniforms = jnp.asarray(rng.random((batch, layers, n)))
    zero_bits = jnp.zeros(n, dtype=jnp.int32)
    zero_state = tc.Circuit(n).state()
    # At full dephasing the two Kraus operators are |0><0| and |1><1|.
    projectors = tc.channels.phasedampingchannel(1.0)

    def ladder(circuit, offset=0):
        for i in range(n - 1):
            circuit.cnot(offset + i, offset + i + 1)

    def layer(carry, inputs):
        state, previous = carry
        angles_layer, uniforms_layer = inputs
        ry_data, ry_anc, entangle, feedback0, feedback1, post = angles_layer
        anc = tc.Circuit(n)
        for i in sites:
            anc.conditional_gate(previous[i], [tc.gates.i(), tc.gates.x()], i)
            anc.ry(i, theta=ry_anc[i])

        data = tc.Circuit(n, inputs=state)
        for i in sites:
            data.ry(i, theta=ry_data[i])
        joint = data.quvector().tensor_product(anc.quvector())
        circuit = tc.Circuit(2 * n, mps_inputs=joint)
        for i in sites:
            circuit.rzz(n + i, i, theta=entangle[i])
        ladder(circuit, n)

        # The RZZ entanglers leave the diagonal of the ancilla marginal
        # invariant. The CNOTs permute that diagonal in both circuits.
        ladder(anc)
        bits, _ = anc.measure(*sites, status=uniforms_layer)
        bits = bits.astype(jnp.int32)
        for i in sites:
            circuit.conditional_gate(bits[i], projectors, n + i)
        for i in sites:
            circuit.rzz(n + i, i, theta=jnp.where(bits[i], feedback1[i], feedback0[i]))
        # Extract and normalize the branch, carrying its ancilla ket as bits.
        state = circuit.projected_subsystem(jnp.concatenate((zero_bits, bits)), sites)
        data = tc.Circuit(n, inputs=state)
        ladder(data)
        for i in sites:
            data.rz(i, theta=post[i])
        return (data.state(), bits), None

    paulis = []
    weights = []
    for i in sites:
        term = [0] * n
        term[i] = 1
        paulis.append(term)
        weights.append(-config["transverse_field"])
        if i + 1 < n:
            term = [0] * n
            term[i:i + 2] = [3, 3]
            paulis.append(term)
            weights.append(-1.)
    hamiltonian = tc.quantum.PauliStringSum2MVP(paulis, weights)

    def energy(p, draws):
        (state, _), _ = jax.lax.scan(layer, (zero_state, zero_bits), (p, draws))
        return jnp.vdot(state, hamiltonian(state)).real

    energies = jax.vmap(energy, in_axes=(None, 0))
    optimizer = optax.adam(config["learning_rate"])

    def update(carry, _):
        p, adam = carry
        loss, grad = jax.value_and_grad(lambda a: energies(a, uniforms).mean())(p)
        steps, adam = optimizer.update(grad, adam, p)
        return (optax.apply_updates(p, steps), adam), loss

    @jax.jit
    def optimize(p):
        (p, _), history = jax.lax.scan(
            update, (p, optimizer.init(p)), None, length=config["max_steps"]
        )
        return history, energies(p, uniforms)

    history, final = optimize(angles)
    return {"energy_history": np.asarray(history), "final_trajectory_energies": np.asarray(final)}
