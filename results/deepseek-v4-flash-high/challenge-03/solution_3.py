import numpy as np
import jax
import jax.numpy as jnp

import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO, reduced_density_matrix
from tensorcircuit.templates.measurements import operator_expectation


def _tfim_hamiltonian(n, h):
    strings, weights = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = 3
        s[i + 1] = 3
        strings.append(s)
        weights.append(-1.0)
    for i in range(n):
        s = [0] * n
        s[i] = 1
        strings.append(s)
        weights.append(-h)
    return PauliStringSum2COO(strings, weight=weights, numpy=False)


def run_solution(config):
    tc.set_backend("jax")

    n = config["n_qubits"]
    n_steps = config["n_steps"]
    h_field = config["transverse_field"]
    w_log = config["log_probability_weight"]
    lr = config["learning_rate"]
    max_steps = config["max_steps"]
    eps_log = 1e-12

    even = list(range(0, n, 2))
    even_bonds = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)][: n // 2]
    odd_bonds = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)][: n // 2 - 1]
    hamiltonian = _tfim_hamiltonian(n, h_field)

    def metrics(thetas, rxs):
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
        logs = []
        norm_sq = jnp.asarray(1.0)
        for t in range(n_steps):
            bonds = even_bonds if t % 2 == 0 else odd_bonds
            for k, (a, b) in enumerate(bonds):
                c.rxx(a, b, theta=2 * thetas[t, 2 * k])
                c.rzz(a, b, theta=2 * thetas[t, 2 * k + 1])
            for i in range(n):
                c.rx(i, theta=rxs[t, i])
            rho = reduced_density_matrix(
                c.state(), subsystem_to_keep=even, normalize=True
            )
            diag = jnp.real(jnp.diag(rho))
            prev = jnp.asarray(1.0)
            # Kept qubits are ordered most-significant first in the reduced basis.
            for k in range(1, len(even) + 1):
                pk = jnp.sum(diag[: 2 ** (len(even) - k)])
                logs.append(jnp.log(pk / prev + eps_log))
                prev = pk
            norm_sq = norm_sq * prev
            for i in even:
                c.mid_measurement(i, keep=0)
        success = norm_sq
        energy = operator_expectation(c, hamiltonian) / success / n
        mean_log = jnp.mean(jnp.stack(logs))
        loss = energy - w_log * mean_log
        return loss, (energy, success, mean_log)

    value_and_grad = jax.jit(jax.value_and_grad(metrics, argnums=(0, 1), has_aux=True))

    key = jax.random.PRNGKey(0)
    thetas = 0.1 * jax.random.normal(key, (n_steps, n), dtype=jnp.float32)
    rxs = 0.1 * jax.random.normal(
        jax.random.fold_in(key, 1), (n_steps, n), dtype=jnp.float32
    )

    m_t = jnp.zeros_like(thetas)
    v_t = jnp.zeros_like(thetas)
    m_r = jnp.zeros_like(rxs)
    v_r = jnp.zeros_like(rxs)
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    e_hist = np.empty(max_steps)
    p_hist = np.empty(max_steps)
    log_hist = np.empty(max_steps)
    loss_hist = np.empty(max_steps)

    for step in range(1, max_steps + 1):
        (loss, (energy, success, mean_log)), (g_t, g_r) = value_and_grad(thetas, rxs)
        e_hist[step - 1] = np.asarray(energy)
        p_hist[step - 1] = np.asarray(success)
        log_hist[step - 1] = np.asarray(mean_log)
        loss_hist[step - 1] = np.asarray(loss)

        m_t = beta1 * m_t + (1 - beta1) * g_t
        v_t = beta2 * v_t + (1 - beta2) * g_t * g_t
        m_r = beta1 * m_r + (1 - beta1) * g_r
        v_r = beta2 * v_r + (1 - beta2) * g_r * g_r
        bc_t = 1 - beta1**step
        bc_v = 1 - beta2**step
        thetas = thetas - lr * (m_t / bc_t) / (jnp.sqrt(v_t / bc_v) + adam_eps)
        rxs = rxs - lr * (m_r / bc_t) / (jnp.sqrt(v_r / bc_v) + adam_eps)

    return {
        "energy_density_history": e_hist,
        "success_probability_history": p_hist,
        "mean_log_probability_history": log_hist,
        "loss_history": loss_hist,
    }
