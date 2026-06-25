# Problem 12: Variational Circuit to MPS Overlap Optimization

## Goal

Optimize a variational circuit state against a target matrix-product state without converting the target into a gate-preparation circuit. The loss is the direct tensor-network overlap between a DMRG-MPS bra and a trainable circuit ket.

## Fixed Problem Configuration

The evaluator defines and passes the following configuration dictionary into `run_solution(config)`.

```python
{
    "n_qubits": 32,
    "zz_anisotropy": 1.4,
    "staggered_field": 0.2,
    "dmrg_chi": 8,
    "dmrg_sweeps": 4,
    "dmrg_tolerance": 1e-7,
    "n_layers": 2,
    "max_steps": 5000,
    "learning_rate": 0.02,
    "initial_parameter_scale": 0.02,
    "seed": 2039,
    "fidelity_threshold": 0.85,
}
```

Before timing `run_solution(config)`, the evaluator runs DMRG itself and augments this dictionary with:

- `dmrg_state`: the normalized quimb MPS target state.

This runtime key is intentionally supplied by the evaluator so the DMRG preparation time is excluded from solution timing and every solution optimizes against the same deterministic MPS target.

Use an open-boundary 32-qubit XXZ Hamiltonian

```text
H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + 1.4 Z_i Z_{i+1})
    + 0.2 sum_i (-1)^i Z_i.
```

Obtain a normalized DMRG-MPS target state for this Hamiltonian with maximum bond dimension `chi = 8` and four DMRG sweeps. Prepare a variational circuit state from the 32-qubit Neel product state `|0101...01>`. Use two brickwork variational layers. Each layer applies a trainable nearest-neighbor two-qubit `SU4` gate on alternating bonds. The first layer uses bonds `(0,1), (2,3), ...`; the second layer uses bonds `(1,2), (3,4), ...`. Each `SU4` gate has 15 trainable parameters, so the default circuit has `465` trainable parameters.

Optimize all circuit parameters to maximize

```text
F(theta) = |<psi_MPS | psi_circuit(theta)>|^2.
```

Train the loss `1 - F(theta)` with Adam at learning rate `0.02` for exactly 5000 optimizer updates. Do not use early stopping.

## Solution Interface

The solution file must be named `solution_12.py` and expose:

```python
def run_solution(config):
    ...
    return results
```

The solution should not print progress. It should perform the core computation and return only NumPy-format quantities that the evaluator checks or reports. It should consume the evaluator-provided `dmrg_state` and should not run DMRG internally.

Required result keys:

- `loss_history`: NumPy array with shape `(max_steps,)`.
- `fidelity_history`: NumPy array with shape `(max_steps,)`.
- `final_parameters`: NumPy array with shape `(parameter_count,)`.
- `final_overlap_phase`: scalar float.

Each history records one value per optimizer update, evaluated immediately before applying that update. The evaluator recomputes parameter count, derives final fidelity and final loss from the histories, and checks all shapes and scalar quantities from the returned arrays.

## Evaluation Interface

The evaluator file is `evaluate_12.py`. It dynamically imports a solution module selected by:

```bash
python evaluate_12.py --solution solution_12
```

The evaluator computes the DMRG-MPS target before timing, passes that quimb MPS into the solution through `config["dmrg_state"]`, and then consumes only the returned result dictionary. It prints the end-to-end solution time excluding evaluator-side DMRG preparation, initial and final fidelity, final overlap phase, target MPS bond dimension, history shape, returned keys, and pass/fail criteria. It does not save files or create plots by default.

## Passing Criteria

A run is considered functionally successful when all of the following hold for the default configuration:

- `loss_history.shape == fidelity_history.shape == (5000,)`.
- `final_parameters.shape == (parameter_count,)`.
- The final fidelity is larger than the initial fidelity.
- The final loss is lower than the initial loss.
- The final fidelity is at least `0.85`.
- The final fidelity is bounded by `1` up to numerical tolerance.
- The target MPS maximum bond dimension is positive and at most `dmrg_chi`.
- All returned arrays and scalar metrics contain finite values.

## Submission Requirements

Create `/root/solution_12.py` and expose:

```python
def run_solution(config):
    ...
```

The evaluator will import `solution_12` from `/root` and call
`run_solution(config)`. Do not modify files under `/tests`, `/logs`, or any
verifier reward files. Do not read hidden oracle solution files.

Your implementation should be concise. More than 200 non-empty, non-comment
Python lines in `/root/solution_12.py` will receive zero static-policy
score.

