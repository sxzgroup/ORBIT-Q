import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _single_qubit_rx_rz_ry(angles: jnp.ndarray) -> jnp.ndarray:
    c = tc.Circuit(1)
    c.rx(0, theta=angles[0])
    c.rz(0, theta=angles[1])
    c.ry(0, theta=angles[2])
    return c.matrix()


_batched_rx_rz_ry = jax.vmap(jax.vmap(_single_qubit_rx_rz_ry))


def _build_tfim_mvp(n_qubits: int, zz_strength: float, x_strength: float):
    structures = []
    weights = []
    if zz_strength != 0.0:
        for i in range(n_qubits - 1):
            s = [0] * n_qubits
            s[i] = 3
            s[i + 1] = 3
            structures.append(s)
            weights.append(-zz_strength)
    if x_strength != 0.0:
        for i in range(n_qubits):
            s = [0] * n_qubits
            s[i] = 1
            structures.append(s)
            weights.append(-x_strength)
    return tc.quantum.PauliStringSum2MVP(structures, weights)


def run_solution(config: dict) -> dict:
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")

    n_qubits = int(config["n_qubits"])
    selected_qubits = tuple(int(q) for q in config["selected_qubits"])
    initial_ones = tuple(int(q) for q in config["initial_ones"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    rng = np.random.default_rng(seed)
    init_params_np = rng.normal(
        loc=0.0, scale=init_scale, size=(n_layers, n_qubits, 3)
    ).astype(np.float64)

    if max_steps <= 0:
        return {
            "energy_history": np.zeros((0,), dtype=np.float64),
            "final_parameters": init_params_np,
        }

    mvp = _build_tfim_mvp(n_qubits, zz_strength, x_strength)
    inv_n = 1.0 / float(n_qubits)

    def energy_density(params: jnp.ndarray) -> jnp.ndarray:
        c = tc.Circuit(n_qubits)
        for q in initial_ones:
            c.x(q)
        if n_layers > 0:
            mats = _batched_rx_rz_ry(params)
            for layer in range(n_layers):
                for q in range(n_qubits):
                    c.any(q, unitary=mats[layer, q])
                if len(selected_qubits) >= 2:
                    c.cmz(*selected_qubits)
                elif len(selected_qubits) == 1:
                    c.z(selected_qubits[0])
        psi = c.state()
        h_psi = jax.lax.stop_gradient(mvp(psi))
        eh = jnp.real(jnp.vdot(psi, h_psi)) * inv_n
        return 2.0 * eh - jax.lax.stop_gradient(eh)

    val_and_grad = jax.value_and_grad(energy_density)
    optimizer = optax.adam(learning_rate)
    init_params = jnp.asarray(init_params_np, dtype=jnp.float64)
    init_opt_state = optimizer.init(init_params)

    @jax.jit
    def optimize(params, opt_state):
        def step_fn(carry, _):
            curr_params, curr_state = carry
            energy_val, grads = val_and_grad(curr_params)
            updates, next_state = optimizer.update(grads, curr_state, curr_params)
            next_params = optax.apply_updates(curr_params, updates)
            return (next_params, next_state), energy_val

        (final_params, _), history = jax.lax.scan(
            step_fn, (params, opt_state), None, length=max_steps
        )
        return final_params, history

    final_params, history = optimize(init_params, init_opt_state)
    return {
        "energy_history": np.asarray(history, dtype=np.float64),
        "final_parameters": np.asarray(final_params, dtype=np.float64),
    }
