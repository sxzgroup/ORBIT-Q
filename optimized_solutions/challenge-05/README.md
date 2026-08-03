# Task 05 — TensorCircuit-native fused cooling

> **Take-home insight:** for the fully TensorCircuit-native tensor network,
> the measured acceleration comes from giving OMECo enough search budget to
> avoid bad contraction paths. Pair-fusing `RX ⊗ RX → RZZ` makes the graph
> smaller, but its isolated contribution is not statistically established.

## Factor attribution

| Direct comparison | Mean paired speedup (95% t-CI) | Decision |
|---|---:|---|
| OMECo `4×4` instead of `1×1`, same fused circuit | **1.420x** `[1.181, 1.659]` | Keep — dominant measured factor |
| Fused instead of unfused filters, both OMECo `1×1` | 1.198x `[0.916, 1.480]` | Unresolved; do not claim |

![Task 05 direct factor-ablation comparisons](factor-ablation.svg)

*Figure — Bars show the median paired runtime ratio relative to the retained
implementation; dots are the five cold-process pairs. Panel a compares the
public expert with the final TensorCircuit-native candidate. The clipped
triangle marks one `3.14x` reference/candidate ratio caused by a `180.22 s`
reference long tail. Panel b isolates the OMECo search budget. Panel c shows
that gate fusion alone is unresolved. Ratios from different panels are not
multiplied.*

## What changed

- **TensorCircuit owns the quantum process.** The state is a `tc.Circuit`;
  cooling filters are applied through `circuit.unitary`; every layer norm is
  computed with `circuit.expectation`; and the TFIM is a
  `tc.quantum.QuOperator` evaluated by `mpo_expectation`.
- **Exact pair fusion.** On every active brickwork bond the implementation
  applies
  `RZZ(2 i b) @ (RX(2 i a) ⊗ RX(2 i a))` as one TensorCircuit `Gate`.
  The two odd-layer endpoint `RX` filters remain separate.
- **Stable exact differentiation.** Every one of the ten layer
  normalizations remains inside the differentiable graph. `complex128`
  prevents the non-unitary contraction from overflowing during 600 Adam
  updates.
- **Path search.** `tc.set_contractor("omeco-4-4")` removes the large runtime
  variance observed with a single OMECo trial/iteration.
- **Compiled optimizer loop.** `K.jaxy_scan` executes all 600 required Adam
  updates and returns every pre-update energy.

The solution contains no custom MPS update, handwritten state-vector
evolution, or manual quantum-state `einsum`. The module docstring is unchanged
from the public expert.

## End-to-end result

Five rotated-order, fresh-process triples were run in the same TensorCircuit
image on the same machine:

| Trial | Public expert (s) | Fused + OMECo `1×1` (s) | Final `4×4` (s) |
|---:|---:|---:|---:|
| 1 | 99.17 | 95.98 | 61.55 |
| 2 | 97.84 | 83.14 | 58.64 |
| 3 | 98.54 | 68.26 | 62.60 |
| 4 | 180.22 | 85.33 | 57.46 |
| 5 | 102.77 | 93.30 | 60.32 |

All 15 cells passed the official evaluator. The final candidate was faster
than the public expert in 5/5 pairs. With all pairs retained, the mean paired
speedup is `1.939x` (95% t-CI `[1.105, 2.772]`); the median paired speedup is
`1.668x`. Because trial 4 contains a visible reference long tail, the
predeclared all-pairs result is accompanied by a sensitivity check: over the
other four pairs, mean paired speedup is `1.639x`, with individual ratios
`1.574x–1.704x`.

The final `4×4` candidate itself is stable: `60.114 ± 2.092 s` mean ± sample
standard deviation, range `57.46–62.60 s`.

## Validation

- official functional evaluator: **PASS**, 600/600 updates;
- final energy density: `-1.3267847010`;
- static policy: **1.0**, 117 effective lines;
- required-framework import and raw-simulator checks: **PASS**;
- frozen candidate SHA-256:
  `bbe5768d9e3b0e1e7e28d611ad2be8392a8bc3e031fb662604adca90a9f08b79`.

The earlier custom no-QR MPS prototype is intentionally not used or claimed
here because its core state updates were outside TensorCircuit quantum APIs.
