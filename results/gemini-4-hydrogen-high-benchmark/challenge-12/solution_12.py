"""Variational circuit to MPS overlap optimization using TensorCircuit-NG."""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _quimb_mps_to_quvector(dmrg_state: Any, dtype: str) -> tc.quantum.QuVector:
    """Convert a quimb MPS into a TensorCircuit QuVector cast to ``dtype``."""
    psi_mps = tc.quantum.quimb2qop(dmrg_state)
    for node in psi_mps.nodes:
        node.tensor = tc.backend.cast(node.tensor, dtype)
    return psi_mps


def _brickwork_bonds(n_qubits: int, n_layers: int) -> List[Tuple[int, int]]:
    """Return alternating nearest-neighbor bond pairs for brickwork layers."""
    bonds: List[Tuple[int, int]] = []
    for layer in range(n_layers):
        for q in range(layer % 2, n_qubits - 1, 2):
            bonds.append((q, q + 1))
    return bonds


def run_solution(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize a 2-layer SU(4) brickwork circuit against a target quimb MPS."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    init_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    dmrg_state = config["dmrg_state"]

    psi_mps = _quimb_mps_to_quvector(dmrg_state, tc.dtypestr)
    mps_bra = psi_mps.adjoint()
    bonds = _brickwork_bonds(n_qubits, n_layers)
    parameter_count = 15 * len(bonds)

    rng = np.random.default_rng(seed)
    init_params = jnp.asarray(
        rng.normal(0.0, init_scale, size=parameter_count),
        dtype=jnp.float64,
    )

    def overlap_fn(params: jax.Array) -> jax.Array:
        circuit = tc.Circuit(n_qubits)
        for q in range(1, n_qubits, 2):
            circuit.x(q)
        for idx, (q0, q1) in enumerate(bonds):
            start = 15 * idx
            circuit.su4(q0, q1, theta=params[start : start + 15])
        return (mps_bra @ circuit.quvector()).eval()

    def loss_fn(params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        ov = overlap_fn(params)
        fidelity = jnp.real(ov * jnp.conj(ov))
        return 1.0 - fidelity, fidelity

    optimizer = optax.adam(learning_rate=learning_rate)
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def optimize(params: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        opt_state = optimizer.init(params)

        def step(carry: Tuple[jax.Array, Any], _: Any) -> Tuple[Tuple[jax.Array, Any], Tuple[jax.Array, jax.Array]]:
            curr_params, curr_state = carry
            (loss, fidelity), grads = value_and_grad(curr_params)
            updates, next_state = optimizer.update(grads, curr_state, curr_params)
            next_params = optax.apply_updates(curr_params, updates)
            return (next_params, next_state), (loss, fidelity)

        (final_params, _), (losses, fidelities) = jax.lax.scan(
            step, (params, opt_state), None, length=max_steps
        )
        final_overlap = overlap_fn(final_params)
        return final_params, losses, fidelities, jnp.angle(final_overlap)

    final_params, loss_history, fidelity_history, final_phase = optimize(init_params)

    return {
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "fidelity_history": np.asarray(fidelity_history, dtype=np.float64),
        "final_parameters": np.asarray(final_params, dtype=np.float64),
        "final_overlap_phase": float(final_phase),
    }
