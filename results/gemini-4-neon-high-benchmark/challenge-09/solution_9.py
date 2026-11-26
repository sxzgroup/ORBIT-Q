"""Random local light-cone variational optimization using TensorCircuit-NG."""

from typing import Any, Dict, List, Sequence, Tuple

import cotengra as ctg
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _extract_light_cone(
    gate_tape: Sequence[Tuple[Any, ...]],
    obs_qubits: Sequence[int],
) -> Tuple[List[int], List[Tuple[Any, ...]]]:
    """Return sorted qubits and ordered gates in the backward causal cone."""
    active = set(int(q) for q in obs_qubits)
    kept: List[Tuple[Any, ...]] = []
    for entry in reversed(gate_tape):
        if len(entry) == 3:
            if int(entry[1]) in active:
                kept.append(entry)
        else:
            q1, q2 = int(entry[1]), int(entry[2])
            if q1 in active or q2 in active:
                kept.append(entry)
                active.add(q1)
                active.add(q2)
    kept.reverse()
    return sorted(active), kept


def _prepare_cones(
    gate_tape: Sequence[Tuple[Any, ...]],
    pauli_terms: Sequence[Tuple[float, Sequence[Tuple[str, int]]]],
) -> Tuple[List[int], List[Tuple[float, int, List[Tuple[Any, ...]], List[Tuple[str, int]]]]]:
    """Extract per-term causal cones and build a compact active parameter map."""
    raw_cones = []
    active_param_set = set()
    for coeff, ops in pauli_terms:
        clean_ops = [(axis.lower(), int(q)) for axis, q in ops if axis.lower() != "i"]
        if not clean_ops:
            raw_cones.append((float(coeff), 0, [], []))
            continue
        cone_qubits, kept_gates = _extract_light_cone(gate_tape, [q for _, q in clean_ops])
        qmap = {q: idx for idx, q in enumerate(cone_qubits)}
        mapped_gates: List[Tuple[Any, ...]] = []
        for entry in kept_gates:
            p_idx = int(entry[-1])
            active_param_set.add(p_idx)
            gname = str(entry[0]).lower()
            if len(entry) == 3:
                mapped_gates.append((gname, qmap[int(entry[1])], p_idx))
            else:
                mapped_gates.append((gname, qmap[int(entry[1])], qmap[int(entry[2])], p_idx))
        mapped_ops = [(axis, qmap[q]) for axis, q in clean_ops]
        raw_cones.append((float(coeff), len(cone_qubits), mapped_gates, mapped_ops))

    active_params = sorted(active_param_set)
    pmap = {p: idx for idx, p in enumerate(active_params)}
    cones = [
        (coeff, nq, [(*g[:-1], pmap[g[-1]]) for g in gates], ops)
        for coeff, nq, gates, ops in raw_cones
    ]
    return active_params, cones


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize local Pauli observables via light-cone TensorCircuit-NG circuits."""
    gate_tape = config["gate_tape"]
    pauli_terms = config["pauli_terms"]
    n_restarts = int(config["n_restarts"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    parameter_count = int(
        config.get(
            "parameter_count",
            max((int(entry[-1]) for entry in gate_tape), default=-1) + 1,
        )
    )

    if n_restarts == 0 or max_steps == 0:
        return {"observable_history": np.empty((n_restarts, max_steps), dtype=np.float64)}

    active_params, cones = _prepare_cones(gate_tape, pauli_terms)

    init_thetas = np.empty((n_restarts, len(active_params)), dtype=np.float64)
    for restart_idx in range(n_restarts):
        rng = np.random.default_rng(seed + 100000 + restart_idx)
        full_theta = rng.normal(0.0, initial_scale, size=parameter_count)
        if active_params:
            init_thetas[restart_idx] = full_theta[active_params]

    jax.config.update("jax_enable_x64", True)
    optimizer = ctg.ReusableHyperOptimizer(
        methods=["greedy"],
        parallel=False,
        minimize="combo",
        max_repeats=8,
        progbar=False,
    )
    pauli_gates = {"x": tc.gates.x, "y": tc.gates.y, "z": tc.gates.z}

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex128"),
        tc.runtime_contractor("custom", optimizer=optimizer, preprocessing=True),
    ):
        def local_objective(theta: jnp.ndarray) -> jnp.ndarray:
            total = jnp.array(0.0, dtype=jnp.float64)
            for coeff, nq, gates, ops in cones:
                if not ops:
                    total = total + coeff
                    continue
                circuit = tc.Circuit(nq)
                for q in range(nq):
                    circuit.ry(q, theta=jnp.pi / 2)
                for entry in gates:
                    if len(entry) == 3:
                        gname, q, p_idx = entry
                        getattr(circuit, gname)(q, theta=theta[p_idx])
                    else:
                        gname, q1, q2, p_idx = entry
                        getattr(circuit, gname)(q1, q2, theta=theta[p_idx])
                tc_ops = [(pauli_gates[axis](), [q]) for axis, q in ops]
                exp_val = jnp.real(circuit.expectation(*tc_ops, enable_lightcone=True))
                total = total + coeff * exp_val
            return total

        if not active_params:
            const_val = float(local_objective(jnp.zeros((0,), dtype=jnp.float64)))
            return {
                "observable_history": np.full(
                    (n_restarts, max_steps), const_val, dtype=np.float64
                )
            }

        loss_and_grad = jax.value_and_grad(lambda th: -local_objective(th))
        adam = optax.adam(learning_rate=learning_rate)

        def optimize_restart(theta0: jnp.ndarray) -> jnp.ndarray:
            opt_state = adam.init(theta0)

            def step(carry: Tuple[jnp.ndarray, Any], _: Any) -> Tuple[Tuple[jnp.ndarray, Any], jnp.ndarray]:
                theta, state = carry
                loss, grads = loss_and_grad(theta)
                updates, next_state = adam.update(grads, state, theta)
                next_theta = optax.apply_updates(theta, updates)
                return (next_theta, next_state), -loss

            _, history = jax.lax.scan(step, (theta0, opt_state), None, length=max_steps)
            return history

        run_all = jax.jit(jax.vmap(optimize_restart))
        observable_history = np.asarray(
            run_all(jnp.asarray(init_thetas, dtype=jnp.float64)),
            dtype=np.float64,
        )

    return {"observable_history": observable_history}
