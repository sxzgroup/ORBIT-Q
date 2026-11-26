"""Random local light-cone optimization using TensorCircuit-NG."""

from typing import Any, Dict, List, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _extract_causal_cone(
    gate_tape: Sequence[Tuple[Any, ...]],
    ops: Sequence[Tuple[str, int]],
) -> Tuple[int, List[Tuple[Any, ...]], List[int], List[int], List[int]]:
    """Extract and remap the backward causal light-cone subcircuit for one Pauli term."""
    paulis_by_axis: Dict[str, List[int]] = {"x": [], "y": [], "z": []}
    obs_qubits: List[int] = []
    for axis, qubit in ops:
        axis_l = axis.lower()
        if axis_l in paulis_by_axis:
            q_int = int(qubit)
            paulis_by_axis[axis_l].append(q_int)
            obs_qubits.append(q_int)

    cone = set(obs_qubits)
    kept_rev: List[Tuple[Any, ...]] = []
    for gate in reversed(gate_tape):
        if len(gate) == 3:
            if gate[1] in cone:
                kept_rev.append(gate)
        else:
            if gate[1] in cone or gate[2] in cone:
                cone.add(gate[1])
                cone.add(gate[2])
                kept_rev.append(gate)

    cone_qubits = sorted(cone)
    qmap = {q: i for i, q in enumerate(cone_qubits)}
    remapped_gates: List[Tuple[Any, ...]] = []
    for gate in reversed(kept_rev):
        if len(gate) == 3:
            remapped_gates.append((gate[0], qmap[gate[1]], int(gate[2])))
        else:
            remapped_gates.append((gate[0], qmap[gate[1]], qmap[gate[2]], int(gate[3])))

    return (
        len(cone_qubits),
        remapped_gates,
        [qmap[q] for q in paulis_by_axis["x"]],
        [qmap[q] for q in paulis_by_axis["y"]],
        [qmap[q] for q in paulis_by_axis["z"]],
    )


def _eval_cone_expectation(
    params: Any,
    n_cone: int,
    gates: Sequence[Tuple[Any, ...]],
    x_idx: Sequence[int],
    y_idx: Sequence[int],
    z_idx: Sequence[int],
) -> Any:
    """Evaluate a local Pauli expectation on its causal-cone circuit in TensorCircuit-NG."""
    c = tc.Circuit(n_cone)
    for q in range(n_cone):
        c.h(q)
    for gate in gates:
        if len(gate) == 3:
            getattr(c, gate[0])(gate[1], theta=params[gate[2]])
        else:
            getattr(c, gate[0])(gate[1], gate[2], theta=params[gate[3]])
    return tc.backend.real(
        c.expectation_ps(
            x=x_idx,
            y=y_idx,
            z=z_idx,
            reuse=False,
            enable_lightcone=True,
        )
    )


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize local Pauli observables across restarts using causal light-cones and Adam."""
    gate_tape = config["gate_tape"]
    parameter_count = int(config["parameter_count"])
    pauli_terms = config["pauli_terms"]
    n_restarts = int(config["n_restarts"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    const_offset = 0.0
    raw_specs: List[Tuple[float, int, List[Tuple[Any, ...]], List[int], List[int], List[int]]] = []
    for coeff, ops in pauli_terms:
        n_cone, gates, x_idx, y_idx, z_idx = _extract_causal_cone(gate_tape, ops)
        if n_cone == 0:
            const_offset += float(coeff)
        else:
            raw_specs.append((float(coeff), n_cone, gates, x_idx, y_idx, z_idx))

    active_params = sorted({gate[-1] for spec in raw_specs for gate in spec[2]})
    if not active_params:
        with tc.runtime_backend("jax"), tc.runtime_contractor("greedy", preprocessing=True):
            obj = const_offset
            for coeff, n_cone, gates, x_idx, y_idx, z_idx in raw_specs:
                obj = obj + coeff * float(
                    _eval_cone_expectation(jnp.zeros(0), n_cone, gates, x_idx, y_idx, z_idx)
                )
        history = np.full((n_restarts, max_steps), float(obj), dtype=np.float64)
        return {"observable_history": history}

    pmap = {p: i for i, p in enumerate(active_params)}
    term_specs = []
    for coeff, n_cone, gates, x_idx, y_idx, z_idx in raw_specs:
        remapped = [
            (g[0], g[1], pmap[g[2]]) if len(g) == 3 else (g[0], g[1], g[2], pmap[g[3]])
            for g in gates
        ]
        term_specs.append((coeff, n_cone, remapped, x_idx, y_idx, z_idx))

    active_idx = np.asarray(active_params, dtype=np.int64)
    init_active = np.empty((n_restarts, len(active_params)), dtype=np.float32)
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + 100000 + r)
        full_theta = rng.normal(0.0, initial_scale, size=parameter_count)
        init_active[r] = full_theta[active_idx]

    with tc.runtime_backend("jax"), tc.runtime_contractor("greedy", preprocessing=True):
        def loss_fn(params: Any) -> Any:
            obj = const_offset
            for coeff, n_cone, gates, x_idx, y_idx, z_idx in term_specs:
                obj = obj + coeff * _eval_cone_expectation(
                    params, n_cone, gates, x_idx, y_idx, z_idx
                )
            return -obj

        vg_fn = jax.value_and_grad(loss_fn)
        optimizer = optax.adam(learning_rate)

        @jax.jit
        def optimize_all(init_params: Any) -> Any:
            init_opt_state = jax.vmap(optimizer.init)(init_params)

            def step_fn(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
                params, opt_state = carry
                losses, grads = jax.vmap(vg_fn)(params)
                updates, new_opt_state = jax.vmap(optimizer.update)(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), -losses

            _, obj_steps = jax.lax.scan(
                step_fn, (init_params, init_opt_state), None, length=max_steps
            )
            return obj_steps.T

        history = np.asarray(optimize_all(jnp.asarray(init_active)), dtype=np.float64)

    return {"observable_history": history}
