import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        return _train(config)


def _train(config):
    n = config["n_qubits"]
    blocks = config["n_blocks"]
    t_min, t_max = config["t_min"], config["t_max"]

    def pauli(axis, *sites):
        string = [0] * n
        for site in sites:
            string[site] = axis
        return string

    xy = [pauli(axis, i, i + 1) for i in range(n - 1) for axis in (1, 2)]
    zz = [pauli(3, i, i + 1) for i in range(n - 1)]
    field = [pauli(3, i) for i in range(n)]
    signs = [(-1) ** i for i in range(n)]
    xy_matvec = tc.quantum.PauliStringSum2MVP(xy, [1.0] * len(xy))
    field_matvec = tc.quantum.PauliStringSum2MVP(field, signs)
    target = tc.quantum.PauliStringSum2COO(
        xy + zz + field,
        [0.7] * len(xy) + [1.1] * len(zz) + [0.25 * sign for sign in signs],
        numpy=True,
    )
    target = tc.backend.coo_sparse_matrix_from_numpy(target)

    initial = tc.Circuit(n)
    for i in range(1, n, 2):
        initial.x(i)
    initial_state = initial.state()
    unit_interval = jnp.array([0.0, 1.0])

    def physical(raw):
        return t_min + (t_max - t_min) * jax.nn.sigmoid(raw[:, 0]), jnp.tanh(raw[:, 1:])

    def hamiltonian(_, hopping, detuning):
        return tc.quantum.LinearOperator(
            target.shape,
            lambda state: hopping * xy_matvec(state) + detuning * field_matvec(state),
        )

    def block(state, controls):
        time, couplings, angles = controls
        # Integrate on [0, 1], so the physical time multiplies the Hamiltonian.
        evolved = tc.timeevol.ode_evol_global(
            hamiltonian, state, unit_interval, None, time * couplings[0],
            time * couplings[1], ode_backend="diffrax", solver="Dopri8", dt0=0.25,
            rtol=config["ode_rtol"], atol=config["ode_atol"],
            max_steps=config["ode_max_steps"],
        )[-1]
        circuit = tc.Circuit(n, inputs=evolved)
        for i in range(n):
            alpha, beta, gamma = angles[i]
            circuit.rz(i, theta=gamma)
            circuit.ry(i, theta=beta)
            circuit.rz(i, theta=alpha)
        return circuit.state(), None

    def loss(params):
        raw, angles = params
        times, couplings = physical(raw)
        state, _ = jax.lax.scan(block, initial_state, (times, couplings, angles))
        energy = tc.templates.measurements.sparse_expectation(tc.Circuit(n, inputs=state), target)
        return energy / (n * jnp.vdot(state, state).real)

    # A fixed seed makes the independent normal draws reproducible.
    rng = np.random.default_rng(0)
    params = (
        jnp.tile(jnp.array([0.0, 0.1, 0.1]), (blocks, 1)),
        jnp.asarray(rng.normal(0.0, 0.1, (blocks, n, 3))),
    )
    optimizer = optax.adam(config["learning_rate"])
    loss_and_grad = jax.value_and_grad(loss)

    def update(carry, _):
        params, opt_state = carry
        value, gradients = loss_and_grad(params)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), value

    @jax.jit
    def optimize(params, opt_state):
        return jax.lax.scan(update, (params, opt_state), None, length=config["max_steps"])

    (params, _), history = optimize(params, optimizer.init(params))
    times, couplings = physical(params[0])
    return {
        "final_analog_times": np.asarray(times),
        "final_analog_couplings": np.asarray(couplings[:, 0]),
        "final_analog_detunings": np.asarray(couplings[:, 1]),
        "energy_density_history": np.asarray(history),
    }
