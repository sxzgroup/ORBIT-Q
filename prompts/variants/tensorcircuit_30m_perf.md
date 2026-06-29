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
- When the required optimization workflow depends on gradients, use exact or mathematically strict gradients for the stated objective. Automatic differentiation through the required framework is preferred. Parameter-shift or another exact analytic gradient method is also allowed when correct. Finite-difference gradients, SPSA, and other approximate-gradient substitutes are not allowed.
- If the problem specifies one value per optimizer update, per time step, per measurement event, per block, or per trajectory, then the code must explicitly execute that workflow and record those quantities at the required points.
- Do not access the network, browse the web, call external APIs or services, or otherwise rely on internet connectivity during the run.
- Do not use online resources, hidden baselines, copied reference solutions, or task-specific hints.
- Do not delegate any part of the task to sub-agents, helper agents, external agents, or parallel agent workers. Complete the work within the single provided agent session.
- The submitted `solution_N.py` should be concise. More than 200 non-empty, non-comment Python lines is considered a failed implementation strategy.
- The submitted solution must complete the evaluator's timed `run_solution(config)` call within 300 seconds. Runtime starts losing score after 180 seconds and receives zero runtime score at 300 seconds or above.
- Optimize for end-to-end evaluator runtime, not just correctness. Prefer the most direct and efficient TensorCircuit-NG or tensorcircuit-nightly representation, reduce avoidable Python overhead, reuse compiled or precomputed objects when appropriate, and iterate on the implementation if a simpler first attempt is too slow.
- When using a JAX-backed implementation, pay attention not only to execution speed but also to compilation latency. When appropriate, prefer structured scans such as `jax.scan` over large Python-unrolled update loops or deeply unrolled layer-by-layer circuit construction, since scanning repeated circuit blocks or optimizer steps can materially reduce compile overhead.
- When there are multiple plausible approaches, test or reason about which one is likely fastest within TensorCircuit-NG or tensorcircuit-nightly, then use that path. Avoid spending time polishing slow designs that are unlikely to pass the runtime budget.
- Before committing to a deep implementation path, verify that the installed TensorCircuit-NG or tensorcircuit-nightly and its dependencies actually expose the primitives, APIs, and numerical behavior the task needs.
- Stop early only if, after checking realistic framework-native alternatives, you conclude that solving the task with the required framework, allowed methods, and acceptable runtime is genuinely close to impossible.
- If your investigation shows that the task is very unlikely to fit within the allowed runtime using the required framework and allowed methods, stop early and report that efficiency obstruction instead of spending the remaining budget on low-probability tuning attempts.

Before implementing, inspect the installed TensorCircuit-NG or tensorcircuit-nightly package, examples, and source code available in the task environment. Search for APIs, tensor-network primitives, circuit/state representations, measurement utilities, optimizers, and interoperability hooks that make the task feasible and efficient in TensorCircuit-NG or tensorcircuit-nightly. Choose an implementation strategy based on what the framework can actually express well.

If you find a substantive obstruction that makes the task impractical or impossible to solve within the required framework and runtime budget, declare the task failed clearly in your final response and stop work promptly rather than spending the entire run on an unproductive workaround. Do not bypass the obstruction by switching frameworks, by importing or using another quantum framework, by writing a raw quantum simulator from scratch, by reverse-engineering the evaluator thresholds, or by manufacturing outputs that mimic a successful run without actually performing the required computation.

## Time Budget

You have only 30 minutes of total solve time in this agent run. Act with urgency. Use tools, code inspection, and tightly scoped test snippets quickly and intelligently. Prioritize early detection of viable TensorCircuit-native solution paths, fast elimination of bad designs, and short empirical checks over long speculative implementation detours.

Preserve a runnable answer as early as possible. Once you have a minimally working solution that plausibly satisfies the task contract, keep that version intact and iterate from there with small, testable improvements rather than risking the whole run on a late large rewrite.

## TensorCircuit Performance Checklist

Use the following checklist as a source of optimization hypotheses and implementation ideas. These are not automatic rules. You must judge them against the concrete task.

- Prefer the most native representation that matches the physics: use TensorCircuit-native operators and abstractions when they fit, including options such as `tc.quantum.PauliStringSum2MVP`, `tc.quantum.PauliStringSum2COO`, `tc.templates.measurements.mpo_expectation`, direct MPS/MPO/qop contraction, `Circuit.post_select`, `enable_lightcone=True`, `QuditCircuit`, and built-in multi-qubit gates such as `cmz` or `su4`.
- Vectorization: replace manual Python loops with `tc.backend.vmap` or `jax.vmap` when this reduces overhead.
- JIT compilation: wrap performance-critical tensor-in/tensor-out kernels with `tc.backend.jit` or `jax.jit`, preferably around the outer optimization step rather than tiny inner fragments. Remember that JIT can be a net loss if the function only runs once.
- JIT reuse discipline: keep input shapes, dtypes, Python container structure, and static arguments stable across calls. Put structural settings in static args when needed, and do not change dtype or shape mid-loop if you want compilation reuse.
- JIT boundary for gradients and optimizers: often prefer jitting the whole train step instead of returning raw gradients from a jitted `value_and_grad` and then updating in Python.
- Move static physics and fixed data out of the hot path: prebuild Hamiltonians, MPOs, matrix-free operators, target states, fixed gate tapes, bond lists, sampled Pauli structures, and reproducible shot-status tensors once and reuse them.
- Structured parameters over flat vectors: prefer PyTrees or structured parameter blocks over a single large flattened parameter vector when that avoids costly slice-based autodiff and gradient assembly.
- Use locality-aware evaluation early: if the task only needs local observables or sparse measurements, test whether light-cone reduction such as `enable_lightcone=True` cuts work materially.
- XLA fusion pathology check: if first-call compile time is unexpectedly extreme, test whether XLA fusion/codegen is the real bottleneck rather than TensorCircuit tracing.
- Staging awareness: single-qubit-heavy unrolling can have worse staging behavior than expected, so avoid gratuitous unrolling.
- Static setup vs numeric loop: move setup work out of the timed loop, but distinguish one-time setup savings from real steady-state speedups.
- Sampling path choice matters: for large-qubit or large-shot cases, benchmark direct sampling paths such as `Circuit.sample(batch=..., allow_state=False)` when the explicit final state is unnecessary.
- Host-device transfer hygiene: avoid `.item()`, Python `float(...)`, `tc.backend.numpy(...)`, printing tensors, or Python branching on tensors inside jitted or repeatedly executed kernels.
- Dtype policy: default to `complex64` and `float32` unless the task genuinely needs higher precision. Set dtype before building long-lived constants and before tracing.
- Avoid full state instantiation when unnecessary: for larger systems, test whether settings such as `reuse=False` or `allow_state=False` prevent wasteful dense-state materialization.
- Control flow, memory, and checkpointing: for repeated blocks, consider `jax.lax.scan` to reduce graph size, and use `jax.checkpoint` or `jax.remat` only when the memory-time trade-off is favorable in practice.
- Compile-time mode vs throughput mode: treat `scan`, partial unrolling, and full unrolling as different performance modes. `scan` may lower compile cost, while manual unrolling may still win on steady-state runtime.
- Contraction optimization: for large tensor-network workloads, test contractor choice and path-search budget. OMECO is a strong early candidate, and tuned cotengra may win when the path is heavily amortized.
- Contraction slicing or hybrid workflows: if exact contraction remains too expensive, consider an unsliced strong path first and then sliced or reconfigured execution when the code path actually supports it.
- Contraction path assumption: do not assume path search is the repeated bottleneck inside a jitted workload unless profiling shows retracing or changing circuit structure.
- Contraction tuning is not always the first lever: first test representation improvements such as light-cone reduction, sparse or MPO reformulation, direct overlap, or more native operator APIs before spending a lot of effort on path search.
- Sparse, MPO, and matrix-free observables or ODEs: for large Hamiltonians or Pauli sums, benchmark sparse or matrix-free routes such as `tc.templates.measurements.sparse_expectation`, `tc.templates.measurements.mpo_expectation`, `tc.quantum.PauliStringSum2MVP`, or `tc.quantum.PauliStringSum2COO` instead of dense assembly or manual `kron`.
- Backend selection and interfacing: if the task leaves room for backend choice within TensorCircuit, JAX is usually the best first option for speed, JIT, and vectorization. If a larger PyTorch stack is required, consider bridging via `tc.interfaces.torch_interface`.

These checklist items must be evaluated by real testing on the concrete workload. Do not assume any item is inherently positive. A change can improve runtime, worsen compile latency, increase memory, or do the opposite. Use short A/B checks whenever feasible, separate one-time setup or warmup cost from steady-state execution cost, and keep only the changes that empirically help this task.
