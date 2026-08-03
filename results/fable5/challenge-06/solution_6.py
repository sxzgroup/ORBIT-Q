"""Challenge 6: digital-analog hybrid VQE with trainable analog blocks.

14-qubit dense statevector simulation with TensorCircuit-NG (JAX backend).
Each hybrid block evolves the state under the trainable analog Hamiltonian
H_analog = J_l sum(XX+YY) + Delta_l sum((-1)^i Z) for a bounded time t_l via
the framework's continuous-time ODE integrator (tc.timeevol.ode_evol_global,
Schrodinger equation, adaptive Dopri5), honoring the configured ode_rtol,
ode_atol, and ode_max_steps as integrator tolerances and step-control bound.
Digital RZ-RY-RZ rotations follow through the framework circuit API. The
normalized energy density of the target Hamiltonian is minimized with Adam
for exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax
from jax.experimental import sparse as jsparse
from tensorcircuit import timeevol as te
from tensorcircuit.templates.measurements import operator_expectation


def _xy_terms(n):
    out = []
    for i in range(n - 1):
        for code in (1, 2):  # XX and YY
            s = [0] * n
            s[i] = code
            s[i + 1] = code
            out.append((s, 1.0))
    return out


def _zz_terms(n):
    return [([3 if k in (i, i + 1) else 0 for k in range(n)], 1.0) for i in range(n - 1)]


def _stag_terms(n):
    out = []
    for i in range(n):
        s = [0] * n
        s[i] = 3
        out.append((s, (-1.0) ** i))
    return out


def _pauli_sum(terms):
    ls, ws = zip(*terms)
    return tc.quantum.PauliStringSum2COO(list(ls), list(ws))


def run_solution(config):
    n = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min, t_max = float(config["t_min"]), float(config["t_max"])
    rtol, atol = float(config["ode_rtol"]), float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    h_target = _pauli_sum(
        [(s, 0.7 * w) for s, w in _xy_terms(n)]
        + [(s, 1.1 * w) for s, w in _zz_terms(n)]
        + [(s, 0.25 * w) for s, w in _stag_terms(n)]
    )
    # H_analog(J, Delta) = J * sum(XX+YY) + Delta * sum((-1)^i Z), assembled
    # from two constant framework Pauli-sum sparse matrices.
    a_mat = _pauli_sum(_xy_terms(n))
    b_mat = _pauli_sum(_stag_terms(n))
    idx = jnp.concatenate([a_mat.indices, b_mat.indices], axis=0)

    def h_analog(t, J, D):
        data = jnp.concatenate([J * a_mat.data, D * b_mat.data])
        return jsparse.BCOO((data, idx), shape=(2**n, 2**n))

    def neel_state():
        c = tc.Circuit(n)
        for i in range(1, n, 2):
            c.x(i)
        return c.state()

    psi0 = neel_state()

    def energy_density(params):
        state = psi0
        for l in range(n_blocks):
            t_l = t_min + (t_max - t_min) * jax.nn.sigmoid(params["s"][l])
            j_l = jnp.tanh(params["j"][l])
            d_l = jnp.tanh(params["d"][l])
            # continuous-time Schrodinger evolution via the framework ODE solver
            state = te.ode_evol_global(
                h_analog, state, jnp.array([0.0, 1.0]) * t_l, None, j_l, d_l,
                rtol=rtol, atol=atol, max_steps=ode_max_steps, ode_backend="jaxode",
            )[-1]
            c = tc.Circuit(n, inputs=state)
            for k in range(n):
                c.rz(k, theta=params["ang"][l, k, 0])
                c.ry(k, theta=params["ang"][l, k, 1])
                c.rz(k, theta=params["ang"][l, k, 2])
            state = c.state()
        c = tc.Circuit(n, inputs=state)
        e = tc.backend.real(operator_expectation(c, h_target))
        nrm = tc.backend.real(jnp.vdot(state, state))
        return e / nrm / n

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(energy_density)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    key = jax.random.PRNGKey(42)
    params = {
        "s": jnp.zeros(n_blocks, dtype=jnp.float64),
        "j": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "d": jnp.full((n_blocks,), 0.1, dtype=jnp.float64),
        "ang": 0.1 * jax.random.normal(key, (n_blocks, n, 3), dtype=jnp.float64),
    }
    opt_state = opt.init(params)

    es = []
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), dtype=np.float64)

    t_fin = t_min + (t_max - t_min) * jax.nn.sigmoid(params["s"])
    return {
        "final_analog_times": np.asarray(t_fin, dtype=np.float64),
        "final_analog_couplings": np.asarray(jnp.tanh(params["j"]), dtype=np.float64),
        "final_analog_detunings": np.asarray(jnp.tanh(params["d"]), dtype=np.float64),
        "energy_density_history": e_hist,
    }
