"""22-qubit VQE with an 18-qubit controlled-Z hyperedge using TensorCircuit-NG."""

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _build_tfim_mvp(n_qubits: int, zz_strength: float, x_strength: float) -> Any:
    """Build a matrix-vector product operator for the open-boundary TFIM Hamiltonian."""
    structures: List[List[int]] = []
    weights: List[float] = []

    for i in range(n_qubits - 1):
        term = [0] * n_qubits
        term[i] = 3
        term[i + 1] = 3
        structures.append(term)
        weights.append(-float(zz_strength))

    for i in range(n_qubits):
        term = [0] * n_qubits
        term[i] = 1
        structures.append(term)
        weights.append(-float(x_strength))

    return tc.quantum.PauliStringSum2MVP(structures, weights)


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Run VQE optimization with an RX-RZ-RY + CMZ ansatz and Adam."""
    tc.set_backend("jax")

    n_qubits = int(config["n_qubits"])
    selected_qubits = tuple(int(q) for q in config["selected_qubits"])
    initial_ones = tuple(int(q) for q in config["initial_ones"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    mvp = _build_tfim_mvp(n_qubits, zz_strength, x_strength)

    def energy_density(params: jnp.ndarray) -> jnp.ndarray:
        c = tc.Circuit(n_qubits)
        for q in initial_ones:
            c.x(q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                c.rx(q, theta=params[layer, q, 0])
                c.rz(q, theta=params[layer, q, 1])
                c.ry(q, theta=params[layer, q, 2])
            if len(selected_qubits) >= 2:
                c.cmz(*selected_qubits)
            elif len(selected_qubits) == 1:
                c.z(selected_qubits[0])
        psi = c.state()
        hpsi = mvp(psi)
        return jnp.real(jnp.vdot(psi, hpsi)) / n_qubits

    rng = np.random.default_rng(seed)
    init_np = rng.normal(
        loc=0.0,
        scale=initial_scale,
        size=(n_layers, n_qubits, 3),
    ).astype(np.float32)
    init_params = jnp.asarray(init_np)

    optimizer = optax.adam(learning_rate=learning_rate)
    init_opt_state = optimizer.init(init_params)
    value_and_grad_fn = jax.value_and_grad(energy_density)

    def scan_step(carry: Any, _: Any) -> Any:
        params, opt_state = carry
        ed, grads = value_and_grad_fn(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), ed

    @jax.jit
    def optimize(params: jnp.ndarray, opt_state: Any) -> Any:
        (final_p, _), history = jax.lax.scan(
            scan_step, (params, opt_state), None, length=max_steps
        )
        return final_p, history

    final_params, energy_history = optimize(init_params, init_opt_state)
    energy_history.block_until_ready()

    return {
        "energy_history": np.asarray(energy_history, dtype=np.float64),
        "final_parameters": np.asarray(final_params, dtype=np.float64),
    }
