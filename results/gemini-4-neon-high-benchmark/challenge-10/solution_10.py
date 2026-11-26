"""Problem 10: 22-qubit VQE with an 18-qubit controlled-Z hyperedge using TensorCircuit-NG."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _build_tfim_mvp(
    n_qubits: int, zz_strength: float, x_strength: float
) -> Any:
    """Build the open-boundary TFIM Hamiltonian matrix-vector product via TensorCircuit."""
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        term = [0] * n_qubits
        term[i] = 3
        term[i + 1] = 3
        structures.append(term)
        weights.append(-zz_strength)
    for i in range(n_qubits):
        term = [0] * n_qubits
        term[i] = 1
        structures.append(term)
        weights.append(-x_strength)
    return tc.quantum.PauliStringSum2MVP(structures, weights)


def _rx_rz_ry_matrix(angles: Any) -> Any:
    """Compose a single-qubit RX-RZ-RY block using TensorCircuit gate primitives."""
    rx = tc.gates.rx_gate(angles[0]).tensor
    rz = tc.gates.rz_gate(angles[1]).tensor
    ry = tc.gates.ry_gate(angles[2]).tensor
    return ry @ rz @ rx


def _apply_controlled_z(c: tc.Circuit, selected_qubits: Sequence[int]) -> None:
    """Apply the fixed multi-qubit controlled-Z hyperedge gate on selected_qubits."""
    if len(selected_qubits) >= 2:
        c.cmz(*selected_qubits)
    elif len(selected_qubits) == 1:
        c.z(selected_qubits[0])


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Run variational optimization of the CMZ-hyperedge ansatz on the open-boundary TFIM."""
    n_qubits = int(config["n_qubits"])
    selected_qubits: Tuple[int, ...] = tuple(int(q) for q in config["selected_qubits"])
    initial_ones: Tuple[int, ...] = tuple(int(q) for q in config["initial_ones"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_parameter_scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    rng = np.random.default_rng(seed)
    init_params_np = rng.normal(
        loc=0.0,
        scale=initial_parameter_scale,
        size=(n_layers, n_qubits, 3),
    ).astype(np.float64)

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex128"),
        tc.runtime_contractor("greedy"),
    ):
        mvp = _build_tfim_mvp(n_qubits, zz_strength, x_strength)
        vmap_rot_block = jax.vmap(jax.vmap(_rx_rz_ry_matrix))

        def state_fn(params: Any) -> Any:
            rot_unitaries = vmap_rot_block(params)
            c = tc.Circuit(n_qubits)
            for q in initial_ones:
                c.x(q)
            for layer_idx in range(n_layers):
                for q in range(n_qubits):
                    c.unitary(q, unitary=rot_unitaries[layer_idx, q])
                _apply_controlled_z(c, selected_qubits)
            return c.state()

        optimizer = optax.adam(learning_rate)
        init_params = jnp.asarray(init_params_np, dtype=jnp.float64)
        init_opt_state = optimizer.init(init_params)

        def step_fn(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
            params, opt_state = carry
            psi, vjp_fn = jax.vjp(state_fn, params)
            psi_barrier = jax.lax.optimization_barrier(psi)
            h_psi = jax.lax.optimization_barrier(mvp(psi_barrier))
            energy_density = jnp.real(jnp.vdot(psi_barrier, h_psi)) / n_qubits
            # H is Hermitian, so d/dpsi Re(<psi|H|psi>)/n has JAX cotangent (2/n) * conj(H|psi>).
            (grads,) = vjp_fn((2.0 / n_qubits) * jnp.conj(h_psi))
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return (next_params, next_opt_state), energy_density

        @jax.jit
        def optimize(params: Any, opt_state: Any) -> Tuple[Any, Any]:
            (final_p, _), history = jax.lax.scan(
                step_fn, (params, opt_state), None, length=max_steps
            )
            return final_p, history

        final_params, energy_history = optimize(init_params, init_opt_state)
        final_params_np = np.asarray(jax.device_get(final_params), dtype=np.float64)
        energy_history_np = np.asarray(jax.device_get(energy_history), dtype=np.float64)

    return {
        "energy_history": energy_history_np,
        "final_parameters": final_params_np,
    }
