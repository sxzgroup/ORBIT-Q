from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def asymmetric_bit_flip_kraus(p01, p10):
    e00 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    e01 = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)
    e10 = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128)
    e11 = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)

    s00 = jnp.sqrt(1.0 - p01).astype(jnp.complex128)
    s11 = jnp.sqrt(1.0 - p10).astype(jnp.complex128)
    s10 = jnp.sqrt(p10).astype(jnp.complex128)
    s01 = jnp.sqrt(p01).astype(jnp.complex128)

    k0 = tc.Gate(s00 * e00 + s11 * e11)
    k1 = tc.Gate(s10 * e01)
    k2 = tc.Gate(s01 * e10)
    return [k0, k1, k2]


@lru_cache(maxsize=None)
def _build_solver(n_qubits, max_steps):
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    # Probe state preparation schedules on |0>^n:
    # 0: GHZ state (|0...0> + |1...1>) / sqrt(2)
    # 1: Bell-like pair states (|01> + |10>) / sqrt(2) on (0,1), (2,3), ...
    # 2: |0>^n
    # 3: |+>^n
    sq_table = jnp.array(
        [
            [2] + [0] * (n_qubits - 1),
            [2 if i % 2 == 0 else 1 for i in range(n_qubits)],
            [0] * n_qubits,
            [2] * n_qubits,
        ],
        dtype=jnp.int32,
    )
    tq_table = jnp.array(
        [
            [1] * (n_qubits - 1),
            [1 if i % 2 == 0 else 0 for i in range(n_qubits - 1)],
            [0] * (n_qubits - 1),
            [0] * (n_qubits - 1),
        ],
        dtype=jnp.int32,
    )

    i_mat = np.eye(2, dtype=np.complex128)
    z_mat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    obs = np.broadcast_to(i_mat, (n_qubits + 1, n_qubits, 2, 2)).copy()
    for q in range(n_qubits):
        obs[q, q] = z_mat
    obs[n_qubits, :] = z_mat
    obs_batch = jnp.asarray(obs)

    sq_gates = [tc.gates.i(), tc.gates.x(), tc.gates.h()]
    tq_gates = [tc.gates.ii(), tc.gates.cnot()]

    def single_probe_observable(sq_idx, tq_idx, obs_ops, angle, p01, p10):
        c = tc.DMCircuit(n_qubits)
        for i in range(n_qubits):
            c.select_gate(sq_idx[i], sq_gates, i)
        for i in range(n_qubits - 1):
            c.select_gate(tq_idx[i], tq_gates, i, i + 1)

        kraus = asymmetric_bit_flip_kraus(p01, p10)

        for i in range(0, n_qubits - 1, 2):
            c.rxx(i, i + 1, theta=angle)
            c.apply_general_kraus(kraus, i)
            c.apply_general_kraus(kraus, i + 1)

        for i in range(1, n_qubits - 1, 2):
            c.rxx(i, i + 1, theta=angle)
            c.apply_general_kraus(kraus, i)
            c.apply_general_kraus(kraus, i + 1)

        exp_val = c.expectation(
            *[(obs_ops[q], [q]) for q in range(n_qubits)], reuse=False
        )
        return jnp.real(exp_val)

    @jax.jit
    def evaluate_table(angle, p01, p10):
        v_obs = jax.vmap(
            single_probe_observable, in_axes=(None, None, 0, None, None, None)
        )
        v_probe = jax.vmap(v_obs, in_axes=(0, 0, None, None, None, None))
        return v_probe(sq_table, tq_table, obs_batch, angle, p01, p10)

    @jax.jit
    def solve(angle, true_p01, true_p10, init_p01, init_p10, lr):
        target_table = evaluate_table(angle, true_p01, true_p10)
        init_probs = jnp.stack([init_p01, init_p10])
        r0 = jnp.log(init_probs / (1.0 - init_probs))

        optimizer = optax.adam(learning_rate=lr)
        opt_state0 = optimizer.init(r0)

        def loss_fn(r):
            probs = jax.nn.sigmoid(r)
            pred_table = evaluate_table(angle, probs[0], probs[1])
            return jnp.mean((pred_table - target_table) ** 2)

        value_and_grad_fn = jax.value_and_grad(loss_fn)

        def step_fn(carry, _):
            r, opt_state = carry
            loss, grads = value_and_grad_fn(r)
            updates, opt_state = optimizer.update(grads, opt_state, r)
            r = optax.apply_updates(r, updates)
            return (r, opt_state), loss

        (r_final, _), loss_history = jax.lax.scan(
            step_fn, (r0, opt_state0), None, length=max_steps
        )
        final_probs = jax.nn.sigmoid(r_final)
        fitted_table = evaluate_table(angle, final_probs[0], final_probs[1])
        return loss_history, final_probs, fitted_table

    return solve


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    max_steps = int(config["max_steps"])
    solver = _build_solver(n_qubits, max_steps)

    loss_history, final_probs, fitted_expectations = solver(
        jnp.asarray(config["entangler_angle"], dtype=jnp.float64),
        jnp.asarray(config["true_p01"], dtype=jnp.float64),
        jnp.asarray(config["true_p10"], dtype=jnp.float64),
        jnp.asarray(config["initial_p01"], dtype=jnp.float64),
        jnp.asarray(config["initial_p10"], dtype=jnp.float64),
        jnp.asarray(config["learning_rate"], dtype=jnp.float64),
    )

    return {
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "final_probabilities": np.asarray(final_probs, dtype=np.float64),
        "fitted_expectations": np.asarray(fitted_expectations, dtype=np.float64),
    }
