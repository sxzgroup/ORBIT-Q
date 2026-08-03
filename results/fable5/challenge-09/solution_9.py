"""Challenge 9: random local light-cone optimization (512 qubits).

The two measured Pauli terms depend only on finite backward causal cones of
the seeded random ladder circuit. The cones are extracted classically from
the evaluator-provided gate tape (backward support propagation), and each
cone becomes a small TensorCircuit-NG circuit (18 and 15 qubits) starting
from |+>. The cone parameter sets are disjoint, and Adam is coordinatewise,
so optimizing the full 3897-parameter vector is exactly equivalent to
optimizing the two cone slices independently: all other coordinates receive
zero gradient and never move. Restarts are vmapped and the 100 Adam updates
run inside jax.lax.scan; the recorded objective is evaluated immediately
before each update, and loss = -objective is minimized.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax


def _causal_cone(tape, support):
    # classical backward light-cone: keep gates whose qubits intersect the
    # growing support of the measured term
    supp = set(support)
    keep = []
    for entry in reversed(tape):
        qs = entry[1:-1]
        if any(q in supp for q in qs):
            keep.append(entry)
            supp.update(qs)
    return list(reversed(keep)), sorted(supp)


def _cone_objective(weight, paulis, sub_tape, qubits):
    remap = {q: i for i, q in enumerate(qubits)}
    nq = len(qubits)
    pmap = sorted({gg[-1] for gg in sub_tape})
    plocal = {p: i for i, p in enumerate(pmap)}

    def f(theta_sub):
        c = tc.Circuit(nq)
        for i in range(nq):
            c.h(i)  # |+> product input
        for gg in sub_tape:
            name = gg[0]
            if len(gg) == 3:
                getattr(c, name)(remap[gg[1]], theta=theta_sub[plocal[gg[-1]]])
            else:
                getattr(c, name)(remap[gg[1]], remap[gg[2]], theta=theta_sub[plocal[gg[-1]]])
        ops = [(getattr(tc.gates, p)(), [remap[q]]) for p, q in paulis]
        return weight * tc.backend.real(c.expectation(*ops))

    return f, pmap


def run_solution(config):
    tape = tuple(tuple(g) for g in config["gate_tape"])
    n_params = int(config["parameter_count"])
    terms = config["pauli_terms"]
    n_restarts = int(config["n_restarts"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])

    # full per-restart initial parameter vectors, as specified
    theta_full = np.stack([
        scale * np.random.default_rng(seed + 100000 + r).standard_normal(n_params)
        for r in range(n_restarts)
    ]).astype(np.float32)

    history = np.zeros((n_restarts, max_steps), dtype=np.float64)
    chunk = 40
    for weight, paulis in terms:
        sub_tape, qubits = _causal_cone(tape, [q for _, q in paulis])
        f, pmap = _cone_objective(float(weight), tuple(paulis), sub_tape, qubits)
        opt = optax.adam(lr)

        def step(carry, _):
            theta, opt_state = carry
            vals, grads = jax.vmap(jax.value_and_grad(lambda t: -f(t)))(theta)
            updates, opt_state = opt.update(grads, opt_state)
            return (optax.apply_updates(theta, updates), opt_state), -vals

        @jax.jit
        def run_chunk(theta0):
            opt_state = opt.init(theta0)
            _, hist = jax.lax.scan(step, (theta0, opt_state), None, length=max_steps)
            return hist.T  # (chunk, max_steps): objective before each update

        for start in range(0, n_restarts, chunk):
            block = theta_full[start:start + chunk][:, pmap]
            history[start:start + chunk] += np.asarray(
                jax.device_get(run_chunk(jnp.asarray(block))), dtype=np.float64
            )

    return {"observable_history": history}
