import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
from tensorcircuit.templates.measurements import sparse_expectation
from tensorcircuit.timeevol import ode_evol_global


def run_solution(config):
    backend = tc.set_backend("jax")
    tc.set_dtype("complex64")
    n = config["n_qubits"]
    blocks = config["n_blocks"]
    lo, hi = config["t_min"], config["t_max"]

    def strings(pauli, width):
        return [[pauli if i <= k < i + width else 0 for k in range(n)]
                for i in range(n - width + 1)]

    xy = strings(1, 2) + strings(2, 2)
    zz = strings(3, 2)
    z = strings(3, 1)
    signs = [(-1) ** k for k in range(n)]
    xy_action = tc.quantum.PauliStringSum2MVP(xy, [1.] * len(xy))
    field_action = tc.quantum.PauliStringSum2MVP(z, signs)
    target = tc.quantum.PauliStringSum2COO_numpy(
        xy + zz + z, [0.7] * len(xy) + [1.1] * len(zz) + [0.25 * w for w in signs])
    target = backend.coo_sparse_matrix_from_numpy(target)

    initial = tc.Circuit(n)
    for k in range(1, n, 2):
        initial.x(k)
    neel = initial.state()

    def physical(raw):
        return (lo + (hi - lo) * jax.nn.sigmoid(raw[..., 0]),
                jnp.tanh(raw[..., 1]), jnp.tanh(raw[..., 2]))

    def hamiltonian(_, exchange, field):
        return lambda state: exchange * xy_action(state) + field * field_action(state)

    def block(state, params):
        raw, angles = params
        duration, coupling, detuning = physical(raw)
        # Integrate in units of the block duration, from 0 to 1.
        evolved = ode_evol_global(
            hamiltonian, state, jnp.array([0., 1.]), None,
            duration * coupling, duration * detuning, ode_backend="diffrax",
            dt0=0.25, rtol=config["ode_rtol"], atol=config["ode_atol"],
            max_steps=config["ode_max_steps"])[-1]
        circuit = tc.Circuit(n, inputs=evolved)
        for k in range(n):
            alpha, beta, gamma = angles[k]
            circuit.rz(k, theta=gamma)
            circuit.ry(k, theta=beta)
            circuit.rz(k, theta=alpha)
        return circuit.state(), None

    def energy(params):
        state, _ = jax.lax.scan(block, neel, params)
        return sparse_expectation(tc.Circuit(n, inputs=state), target) / n

    params = (jnp.tile(jnp.array([0., 0.1, 0.1]), (blocks, 1)),
              jnp.asarray(np.random.default_rng(6).normal(
                  0., 0.1, (blocks, n, 3)), dtype=jnp.float32))
    adam = optax.adam(config["learning_rate"])

    @jax.jit
    def step(params, optimizer_state):
        value, grad = jax.value_and_grad(energy)(params)
        updates, optimizer_state = adam.update(grad, optimizer_state)
        return optax.apply_updates(params, updates), optimizer_state, value

    optimizer_state = adam.init(params)
    history = np.empty(config["max_steps"])
    for i in range(len(history)):
        params, optimizer_state, value = step(params, optimizer_state)
        history[i] = float(value)
    times, couplings, detunings = map(np.asarray, physical(params[0]))
    return dict(final_analog_times=times, final_analog_couplings=couplings,
                final_analog_detunings=detunings, energy_density_history=history)
