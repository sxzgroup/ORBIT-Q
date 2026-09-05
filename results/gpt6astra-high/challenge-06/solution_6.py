"""Hybrid VQE with TensorCircuit's adaptive Schrodinger ODE evolution."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def run_solution(config):
    jax.config.update("jax_enable_x64", True)
    tc.set_dtype("complex128")
    n, blocks = config["n_qubits"], config["n_blocks"]

    def pauli(sites, axis):
        word = [0] * n
        for k in sites:
            word[k] = axis
        return word

    xy = [pauli([k, k + 1], a) for k in range(n - 1) for a in (1, 2)]
    zz = [pauli([k, k + 1], 3) for k in range(n - 1)]
    z = [pauli([k], 3) for k in range(n)]
    signs = [(-1.0) ** k for k in range(n)]
    tc.set_backend("numpy")
    target = tc.quantum.PauliStringSum2COO(
        xy + zz + z,
        [0.7] * len(xy) + [1.1] * len(zz) + [0.25 * s for s in signs],
        numpy=True,
    )
    backend = tc.set_backend("jax")
    target = backend.sparse_csr_from_coo(backend.coo_sparse_matrix_from_numpy(target))
    xy_op = tc.quantum.PauliStringSum2MVP(xy, [1.0] * len(xy))
    z_op = tc.quantum.PauliStringSum2MVP(z, signs)
    initial = tc.Circuit(n)
    for k in range(1, n, 2):
        initial.x(k)
    initial_state = initial.state()

    def physical(p):
        times = config["t_min"] + (config["t_max"] - config["t_min"]) * jax.nn.sigmoid(p[..., 0])
        return times, jnp.tanh(p[..., 1]), jnp.tanh(p[..., 2])

    def hamiltonian(time, coupling, detuning):
        return tc.quantum.LinearOperator(
            (2**n, 2**n),
            lambda state: coupling * xy_op(state) + detuning * z_op(state),
            dtype=jnp.complex128,
        )

    def block(state, parameters):
        analog, angles = parameters
        time, coupling, detuning = physical(analog)
        c = tc.Circuit(n, inputs=state)
        c = tc.timeevol.evol_global(
            c, hamiltonian, jnp.stack([jnp.zeros_like(time), time]),
            coupling, detuning,
            ode_backend="diffrax", solver="Dopri8", dt0=0.01,
            rtol=config["ode_rtol"], atol=config["ode_atol"],
            max_steps=config["ode_max_steps"],
        )
        for k in range(n):
            # Rightmost factor acts first in RZ(alpha) RY(beta) RZ(gamma).
            c.rz(k, theta=angles[k, 2])
            c.ry(k, theta=angles[k, 1])
            c.rz(k, theta=angles[k, 0])
        return c.state(), None

    def loss(parameters):
        state, _ = jax.lax.scan(block, initial_state, parameters)
        c = tc.Circuit(n, inputs=state)
        energy = tc.templates.measurements.sparse_expectation(c, target)
        # Normalize the expectation to remove numerical ODE norm drift.
        return energy / (n * jnp.real(c.expectation()))

    analog = np.tile([0.0, 0.1, 0.1], (blocks, 1))
    angles = np.random.default_rng(0).normal(0.0, 0.1, (blocks, n, 3))
    parameters = (jnp.asarray(analog), jnp.asarray(angles))
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(parameters)

    @jax.jit
    def update(parameters, opt_state):
        energy, gradients = jax.value_and_grad(loss)(parameters)
        updates, opt_state = optimizer.update(gradients, opt_state, parameters)
        return optax.apply_updates(parameters, updates), opt_state, energy

    history = np.empty(config["max_steps"], dtype=np.float64)
    for step in range(config["max_steps"]):
        parameters, opt_state, energy = update(parameters, opt_state)
        history[step] = float(energy)
    times, couplings, detunings = physical(parameters[0])
    return {
        "final_analog_times": np.asarray(times),
        "final_analog_couplings": np.asarray(couplings),
        "final_analog_detunings": np.asarray(detunings),
        "energy_density_history": history,
    }
