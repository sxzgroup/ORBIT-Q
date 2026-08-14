import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    backend = tc.backend

    n_qubits = int(config["n_qubits"])
    field = float(config["transverse_field"])
    n_steps = int(config["n_steps"])
    log_weight = float(config["log_probability_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    eps = 1e-12

    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    meas_qubits = list(range(0, n_qubits, 2))
    xx_mat = tc.gates._xx_matrix
    zz_mat = tc.gates._zz_matrix

    def apply_layer(state, xxzz, rxangs, bonds):
        circuit = tc.Circuit(n_qubits, inputs=state)
        for bond_idx, (left, right) in enumerate(bonds):
            circuit.exp1(left, right, theta=xxzz[bond_idx, 0], unitary=xx_mat)
            circuit.exp1(left, right, theta=xxzz[bond_idx, 1], unitary=zz_mat)
        for qubit in range(n_qubits):
            circuit.rx(qubit, theta=rxangs[qubit])
        logs = []
        joint = jnp.float32(1.0)
        for qubit in meas_qubits:
            z_exp = backend.real(circuit.expectation_ps(z=[qubit]))
            prob = (joint + z_exp) * jnp.float32(0.5) / joint
            logs.append(backend.log(prob + eps))
            circuit.post_select(qubit, keep=0)
            joint = joint * prob
        state = circuit.state()
        return state / backend.norm(state), backend.stack(logs)

    def even_step(state, xxzz, rxangs):
        return apply_layer(state, xxzz, rxangs, even_bonds)

    def odd_step(state, xxzz, rxangs):
        return apply_layer(state, xxzz, rxangs, odd_bonds)

    def scan_step(state, args):
        xxzz, rxangs, is_even = args
        return jax.lax.cond(
            is_even,
            lambda current: even_step(current, xxzz, rxangs),
            lambda current: odd_step(current, xxzz, rxangs),
            state,
        )

    def plus_state():
        circuit = tc.Circuit(n_qubits)
        for qubit in range(n_qubits):
            circuit.h(qubit)
        return circuit.state()

    def energy_density(state):
        circuit = tc.Circuit(n_qubits, inputs=state)
        energy = 0.0
        for qubit in range(n_qubits - 1):
            energy = energy - backend.real(
                circuit.expectation_ps(z=[qubit, qubit + 1])
            )
        for qubit in range(n_qubits):
            energy = energy - field * backend.real(circuit.expectation_ps(x=[qubit]))
        return energy / n_qubits

    def evaluate(params):
        xxzz, rxangs = params
        flags = jnp.asarray([(step % 2) == 0 for step in range(n_steps)], dtype=bool)
        state, logs = jax.lax.scan(scan_step, plus_state(), (xxzz, rxangs, flags))
        n_events = n_steps * len(meas_qubits)
        mean_log = backend.mean(logs)
        success = backend.exp(n_events * mean_log)
        density = energy_density(state)
        loss = density - jnp.float32(log_weight) * mean_log
        return loss, (density, success, mean_log)

    rng_a, rng_b = jax.random.split(jax.random.PRNGKey(0))
    xxzz = 0.05 * jax.random.normal(
        rng_a, (n_steps, len(even_bonds), 2), dtype=jnp.float32
    )
    if len(even_bonds) > len(odd_bonds):
        xxzz = xxzz.at[1::2, len(odd_bonds) :, :].set(0.0)
    rxangs = 0.05 * jax.random.normal(rng_b, (n_steps, n_qubits), dtype=jnp.float32)
    params = (xxzz, rxangs)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    def train_step(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(evaluate, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    train_step = jax.jit(train_step)
    energy_hist = []
    success_hist = []
    mean_log_hist = []
    loss_hist = []
    for _ in range(max_steps):
        params, opt_state, loss, aux = train_step(params, opt_state)
        energy_hist.append(aux[0])
        success_hist.append(aux[1])
        mean_log_hist.append(aux[2])
        loss_hist.append(loss)

    return {
        "energy_density_history": np.asarray(energy_hist, dtype=np.float64),
        "success_probability_history": np.asarray(success_hist, dtype=np.float64),
        "mean_log_probability_history": np.asarray(mean_log_hist, dtype=np.float64),
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
    }
