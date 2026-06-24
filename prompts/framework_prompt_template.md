## Quantum Framework Constraint

The required quantum software framework for this run is {framework}.

You may use general-purpose Python libraries for numerical arrays, optimization, sparse linear algebra, data handling, testing, and interoperability.

However:

- The central quantum computation must be expressed through {framework}.
- Do not use another quantum software framework to implement the task.
- Do not replace {framework} with a raw NumPy/JAX/SciPy simulator from scratch.
- Do not use online resources, hidden baselines, copied reference solutions, or task-specific hints.
- The submitted `solution_N.py` should be concise. More than 200 non-empty, non-comment Python lines is considered a failed implementation strategy.
- The submitted solution must complete the evaluator's timed `run_solution(config)` call within 300 seconds. Runtime starts losing score after 180 seconds and receives zero runtime score at 300 seconds or above.
- Optimize for end-to-end evaluator runtime, not just correctness. Prefer the most direct and efficient {framework} representation, reduce avoidable Python overhead, reuse compiled or precomputed objects when appropriate, and iterate on the implementation if a simpler first attempt is too slow.
- When there are multiple plausible approaches, test or reason about which one is likely fastest within {framework}, then use that path. Avoid spending time polishing slow designs that are unlikely to pass the runtime budget.

Before implementing, inspect the installed {framework} package, examples, and source code available in the task environment. Search for APIs, tensor-network primitives, circuit/state representations, measurement utilities, optimizers, and interoperability hooks that make the task feasible and efficient in {framework}. Choose an implementation strategy based on what the framework can actually express well.

If you find a substantive obstruction that makes the task impractical or impossible to solve within the required framework and runtime budget, declare the task failed clearly in your final response rather than spending the entire run on an unproductive workaround. Do not bypass the obstruction by switching frameworks or by writing a raw quantum simulator from scratch.
