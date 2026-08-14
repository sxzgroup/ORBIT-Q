import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


tc.set_backend("jax")
tc.set_dtype("complex64")


def _tfim_matvec(psi, n_qubits, zz_strength, x_strength):
    psi_t = psi.reshape((2,) * n_qubits)
    out = jnp.zeros_like(psi_t)
    zvals = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    for i in range(n_qubits - 1):
        zi = zvals.reshape((1,) * i + (2,) + (1,) * (n_qubits - i - 1))
        zj = zvals.reshape((1,) * (i + 1) + (2,) + (1,) * (n_qubits - i - 2))
        out = out - zz_strength * zi * zj * psi_t
    for i in range(n_qubits):
        out = out - x_strength * jnp.flip(psi_t, axis=i)
    return out.reshape(-1)


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    selected = tuple(int(q) for q in config["selected_qubits"])
    initial_ones = tuple(int(q) for q in config["initial_ones"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    rng = np.random.default_rng(seed)
    params0 = jnp.asarray(
        rng.normal(0.0, scale, size=(n_layers, n_qubits, 3)).astype(np.float32)
    )
    ham = tc.aslinearoperator(
        lambda psi: _tfim_matvec(psi, n_qubits, zz_strength, x_strength),
        shape=(1 << n_qubits, 1 << n_qubits),
        dtype=tc.dtypestr,
    )

    def energy_density(params):
        circuit = tc.Circuit(n_qubits)
        for qubit in initial_ones:
            circuit.x(qubit)
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                circuit.rx(qubit, theta=params[layer, qubit, 0])
                circuit.rz(qubit, theta=params[layer, qubit, 1])
                circuit.ry(qubit, theta=params[layer, qubit, 2])
            circuit.cmz(*selected)
        psi = circuit.state()
        energy = jnp.vdot(psi, ham.matvec(psi))
        return jnp.real(energy) / n_qubits

    optimizer = optax.adam(learning_rate)
    opt_state0 = optimizer.init(params0)

    def train(params, opt_state):
        def body(carry, _):
            params, opt_state = carry
            energy, grads = jax.value_and_grad(energy_density)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), energy

        (params, _), history = jax.lax.scan(
            body, (params, opt_state), None, length=max_steps
        )
        return params, history

    final_parameters, energy_history = jax.jit(train)(params0, opt_state0)
    return {
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "final_parameters": np.asarray(final_parameters, dtype=np.float64),
    }
