"""Variational brickwork SU(4) circuit to DMRG-MPS overlap optimization."""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _brickwork_bonds(n_qubits: int, n_layers: int) -> List[Tuple[int, int]]:
    """Return nearest-neighbor brickwork bond pairs for alternating layers."""
    return [
        (i, i + 1)
        for layer in range(n_layers)
        for i in range(layer % 2, n_qubits - 1, 2)
    ]


def _mps_to_quvector(dmrg_state: Any, n_qubits: int) -> tc.QuVector:
    """Convert a quimb MPS into a TensorCircuit QuVector aligned with sites 0..n-1."""
    mps_qop = tc.quantum.quimb2qop(dmrg_state)
    if len(mps_qop.out_edges) != n_qubits:
        raise ValueError(
            f"Expected {n_qubits} physical MPS edges, got {len(mps_qop.out_edges)}."
        )
    if hasattr(dmrg_state, "site_ind") and hasattr(dmrg_state, "tensors"):
        ind_order = [
            ind
            for tensor in dmrg_state.tensors
            for ind in tensor.inds
            if str(ind).startswith("k")
        ]
        expected_inds = [dmrg_state.site_ind(i) for i in range(n_qubits)]
        if ind_order != expected_inds and set(ind_order) == set(expected_inds):
            pos = {name: idx for idx, name in enumerate(ind_order)}
            mps_qop = tc.QuVector(
                [mps_qop.out_edges[pos[name]] for name in expected_inds],
                ref_nodes=mps_qop.ref_nodes,
                ignore_edges=mps_qop.ignore_edges,
            )
    for node in mps_qop.nodes:
        node.tensor = tc.backend.cast(
            tc.backend.convert_to_tensor(node.tensor), tc.dtypestr
        )
    return mps_qop


def run_solution(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize a brickwork SU(4) variational circuit against a target DMRG MPS."""
    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    dmrg_state = config["dmrg_state"]

    bonds = _brickwork_bonds(n_qubits, n_layers)
    n_gates = len(bonds)
    parameter_count = n_gates * 15

    rng = np.random.default_rng(seed)
    init_params_np = rng.normal(
        loc=0.0, scale=initial_scale, size=(parameter_count,)
    ).astype(np.float64)

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex128"),
        tc.runtime_contractor("greedy", preprocessing=True),
    ):
        mps_qop = _mps_to_quvector(dmrg_state, n_qubits)
        su4_batch = tc.backend.vmap(lambda theta: tc.gates.su4_gate(theta).tensor)

        def overlap_fn(params: jnp.ndarray) -> jnp.ndarray:
            circuit = tc.Circuit(n_qubits)
            for qubit in range(1, n_qubits, 2):
                circuit.x(qubit)
            gate_tensors = su4_batch(jnp.reshape(params, (n_gates, 15)))
            for idx, (q0, q1) in enumerate(bonds):
                circuit.unitary(q0, q1, unitary=gate_tensors[idx])
            return tc.expectation(ket=circuit.get_quvector(), bra=mps_qop, conj=True)

        def loss_fn(params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            overlap = overlap_fn(params)
            fidelity = jnp.real(overlap * jnp.conj(overlap))
            return 1.0 - fidelity, fidelity

        optimizer = optax.adam(learning_rate=learning_rate)
        value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

        @jax.jit
        def optimize(
            params: jnp.ndarray, opt_state: Any
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            def step(
                carry: Tuple[jnp.ndarray, Any], _: None
            ) -> Tuple[Tuple[jnp.ndarray, Any], Tuple[jnp.ndarray, jnp.ndarray]]:
                curr_params, curr_state = carry
                (loss, fidelity), grads = value_and_grad(curr_params)
                updates, next_state = optimizer.update(grads, curr_state, curr_params)
                next_params = optax.apply_updates(curr_params, updates)
                return (next_params, next_state), (loss, fidelity)

            (final_p, _), (losses, fids) = jax.lax.scan(
                step, (params, opt_state), None, length=max_steps
            )
            final_overlap = overlap_fn(final_p)
            return final_p, losses, fids, final_overlap

        init_params = jnp.asarray(init_params_np, dtype=jnp.float64)
        init_opt_state = optimizer.init(init_params)
        final_params, loss_history, fidelity_history, final_overlap = optimize(
            init_params, init_opt_state
        )

        loss_history_np = np.asarray(loss_history, dtype=np.float64)
        fidelity_history_np = np.asarray(fidelity_history, dtype=np.float64)
        final_parameters_np = np.asarray(final_params, dtype=np.float64)
        final_overlap_phase = float(np.angle(complex(final_overlap)))

    return {
        "loss_history": loss_history_np,
        "fidelity_history": fidelity_history_np,
        "final_parameters": final_parameters_np,
        "final_overlap_phase": final_overlap_phase,
    }
