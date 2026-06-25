## Quantum Framework Constraint

The required quantum software framework for this run is TensorCircuit-NG or tensorcircuit-nightly.

You may use general-purpose Python libraries for numerical arrays, optimization, sparse linear algebra, data handling, testing, and interoperability.

However:

- The central quantum computation must be expressed through TensorCircuit-NG or tensorcircuit-nightly.
- Do not use another quantum software framework to implement the task.
- Do not replace TensorCircuit-NG or tensorcircuit-nightly with a raw NumPy/JAX/SciPy simulator from scratch.
- Do not implement the required quantum gates, circuit evolution, measurements, or state updates by manually manipulating the full state vector, amplitudes, tensor entries, or index tables as a workaround around the framework. Use the quantum software framework's own circuit/state/operator APIs for the core quantum process.
- Do not use TensorCircuit-NG or tensorcircuit-nightly only as a gate-matrix lookup helper, state initializer, or thin wrapper while the real quantum evolution is carried out by custom array code. The implementation itself must genuinely run through TensorCircuit-NG or tensorcircuit-nightly's quantum APIs.
- Do not compute an exact or near-exact oracle answer for the evaluator's target quantity and then use that oracle value to fabricate outputs that merely satisfy the verifier checks.
- Do not directly diagonalize, exactly solve, or otherwise compute the strict target optimum of the same Hamiltonian/objective when the task is asking for a variational, iterative, sampled, or framework-native optimization procedure, unless the task explicitly allows that method.
- Do not fake, smooth, interpolate, backfill, or analytically construct required return arrays such as `energy_history`, `loss_history`, gradients, entropies, measurement statistics, or trajectory records. Every returned entry must come from the actual required computation at the corresponding step.
- Do not exploit evaluator loopholes by returning values that pass surface-level checks while skipping the required optimization, update loop, circuit evolution, measurement process, or other requested algorithmic structure.
- Treat the problem description itself as part of the specification, not just the visible evaluator outputs. The code and implementation workflow themselves must be consistent with the described physics process, optimization procedure, observables, measurement or post-selection rules, and returned quantities.
- Assume there may be additional hidden or black-box evaluation derived from the problem statement, not only the visible checks in the provided evaluator script. A solution that only targets the visible checks while violating the stated physics or algorithmic process is invalid.
- Do not hardcode parameter targets, closed-form trajectories, fixed entropy profiles, synthetic measurement statistics, or any other surrogate outputs unless they are genuinely produced by the required computation described in the task.
- If the stated physical process says a quantity should come from circuit evolution, variational updates, measurements, post-selection, entropy evaluation, or trajectory averaging, then return the value produced by that process, not a proxy chosen merely to satisfy the visible thresholds.
- If the problem specifies Adam, gradient descent, stochastic gradient descent, natural gradient, coordinate descent, or any other named optimizer or update rule, then the code must actually implement that optimizer workflow on the stated objective. Do not substitute a different update rule, a hand-chosen target parameter vector, or a synthetic monotone trajectory.
- If the problem specifies one value per optimizer update, per time step, per measurement event, per block, or per trajectory, then the code must explicitly execute that workflow and record those quantities at the required points.
- Do not use online resources, hidden baselines, copied reference solutions, or task-specific hints.
- The submitted `solution_N.py` should be concise. More than 200 non-empty, non-comment Python lines is considered a failed implementation strategy.
- The submitted solution must complete the evaluator's timed `run_solution(config)` call within 300 seconds. Runtime starts losing score after 180 seconds and receives zero runtime score at 300 seconds or above.
- Optimize for end-to-end evaluator runtime, not just correctness. Prefer the most direct and efficient TensorCircuit-NG or tensorcircuit-nightly representation, reduce avoidable Python overhead, reuse compiled or precomputed objects when appropriate, and iterate on the implementation if a simpler first attempt is too slow.
- When using a JAX-backed implementation, pay attention not only to execution speed but also to compilation latency. When appropriate, prefer structured scans such as `jax.scan` over large Python-unrolled update loops or deeply unrolled layer-by-layer circuit construction, since scanning repeated circuit blocks or optimizer steps can materially reduce compile overhead.
- When there are multiple plausible approaches, test or reason about which one is likely fastest within TensorCircuit-NG or tensorcircuit-nightly, then use that path. Avoid spending time polishing slow designs that are unlikely to pass the runtime budget.

Before implementing, inspect the installed TensorCircuit-NG or tensorcircuit-nightly package, examples, and source code available in the task environment. Search for APIs, tensor-network primitives, circuit/state representations, measurement utilities, optimizers, and interoperability hooks that make the task feasible and efficient in TensorCircuit-NG or tensorcircuit-nightly. Choose an implementation strategy based on what the framework can actually express well.

If you find a substantive obstruction that makes the task impractical or impossible to solve within the required framework and runtime budget, declare the task failed clearly in your final response rather than spending the entire run on an unproductive workaround. Do not bypass the obstruction by switching frameworks, by writing a raw quantum simulator from scratch, by reverse-engineering the evaluator thresholds, or by manufacturing outputs that mimic a successful run without actually performing the required computation.
