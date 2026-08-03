"""Challenge 3: probability-aware post-selected many-body cooling (12 qubits).

TensorCircuit-NG dense statevector simulation (JAX backend). Ten cooling
steps: trainable exp[-i(txx XX + tzz ZZ)] on alternating brickwork bonds,
trainable RX on every qubit, then sequential post-selection of |0> on each
even-indexed qubit via the framework's ``post_select`` (branch probability =
squared norm of the unnormalized post-selected state). The loss
<H>/n - w * mean(log(p_event + 1e-12)) is minimized with Adam for exactly
``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax
from tensorcircuit import gates as g
from tensorcircuit.templates.measurements import operator_expectation


def _tfim_hamiltonian(n, field):
    # H = -sum_i Z_i Z_{i+1} - field * sum_i X_i
    ls, ws = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = 3
        s[i + 1] = 3
        ls.append(s)
        ws.append(-1.0)
    for i in range(n):
        s = [0] * n
        s[i] = 1
        ls.append(s)
        ws.append(-field)
    return tc.quantum.PauliStringSum2COO(ls, ws)


def _two_q(txx, tzz):
    # exp[-i (txx XX + tzz ZZ)]; commuting terms, and tc's rxx(theta) equals
    # exp(-i theta/2 XX), hence the factor 2.
    return (
        g.rxx(theta=2.0 * txx).tensor.reshape(4, 4)
        @ g.rzz(theta=2.0 * tzz).tensor.reshape(4, 4)
    )


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    w_logp = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    ham = _tfim_hamiltonian(n, field)
    even_bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]
    post_qubits = list(range(0, n, 2))
    n_events = n_steps * len(post_qubits)

    def plus_state():
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
        return c.state()

    psi0 = plus_state()

    def metrics(params):
        state = psi0
        log_ps = []
        for t in range(n_steps):
            bonds = even_bonds if t % 2 == 0 else odd_bonds
            c = tc.Circuit(n, inputs=state)
            for k, (a, b) in enumerate(bonds):
                c.any(a, b, unitary=_two_q(params["bond"][t, k, 0], params["bond"][t, k, 1]))
            for i in range(n):
                c.rx(i, theta=params["rx"][t, i])
            state = c.state()
            # sequential post-selection of |0> on even-indexed qubits
            for q in post_qubits:
                cm = tc.Circuit(n, inputs=state)
                cm.post_select(q, keep=0)
                branch = cm.state()
                p = tc.backend.real(jnp.vdot(branch, branch))
                log_ps.append(jnp.log(p + 1e-12))
                state = branch / jnp.sqrt(p)
        c = tc.Circuit(n, inputs=state)
        e_density = tc.backend.real(operator_expectation(c, ham)) / n
        mean_logp = jnp.mean(jnp.stack(log_ps))
        loss = e_density - w_logp * mean_logp
        success = jnp.exp(n_events * mean_logp)
        return loss, (e_density, mean_logp, success)

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        (loss, aux), grad = jax.value_and_grad(metrics, has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    # Small Gaussian initialization keeps early post-selection probabilities
    # near the |+> value of 1/2 per event while breaking parameter symmetry.
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    params = {
        "bond": 0.02 * jax.random.normal(k1, (n_steps, len(even_bonds), 2), dtype=jnp.float64),
        "rx": 0.02 * jax.random.normal(k2, (n_steps, n), dtype=jnp.float64),
    }
    opt_state = opt.init(params)

    e_hist = np.empty(max_steps, dtype=np.float64)
    p_hist = np.empty(max_steps, dtype=np.float64)
    mlp_hist = np.empty(max_steps, dtype=np.float64)
    loss_hist = np.empty(max_steps, dtype=np.float64)
    for k in range(max_steps):
        params, opt_state, loss, (e_density, mean_logp, success) = step(params, opt_state)
        e_hist[k] = float(e_density)
        p_hist[k] = float(success)
        mlp_hist[k] = float(mean_logp)
        loss_hist[k] = float(loss)

    return {
        "energy_density_history": e_hist,
        "success_probability_history": p_hist,
        "mean_log_probability_history": mlp_hist,
        "loss_history": loss_hist,
    }
