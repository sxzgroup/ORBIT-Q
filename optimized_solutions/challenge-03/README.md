# Challenge 03: structural loophole, measured impact, and fix

## Take-home message

**Yes: the original measurement schedule admits an exact cheap reduction.**
Post-selecting every even qubit after *each* brickwork layer prevents two
surviving odd qubits from interacting before projection. The nominal
12-qubit trajectory is therefore exactly six independent one-qubit
trajectories.

This is a circuit-design degeneracy, not a fabricated-output or verifier
workaround. It is milder than Challenge 07 because all 230 parameters, ten
unitary layers, and 300 Adam updates are still evaluated.

## Why the reduction is exact

1. After every original layer, all even qubits are projected to `|0>`.
2. Every even- or odd-layer bond contains exactly one even qubit.
3. Therefore every two-qubit gate acts on one disposable even qubit and only
   one surviving odd qubit; no gate can correlate two survivors before the
   next projection.

The selected branch consequently remains

```text
|0>_0 ⊗ |ψ_1> ⊗ |0>_2 ⊗ |ψ_3> ⊗ ... ⊗ |0>_10 ⊗ |ψ_11>.
```

## Measured impact on the original workload

The strongest implementation also delays normalization to the end. Because
the evaluator uses only the product of event probabilities and their mean
logarithm, the intermediate normalization factors telescope exactly.

| Original-workload implementation | Runtime, mean ± SE | Comparison |
| --- | ---: | ---: |
| Public expert | `4.259896 ± 0.034251 s` | `1.000x` |
| Previous PR candidate | `1.009877 ± 0.016924 s` | `4.218x` vs expert |
| Product reduction + delayed normalization | `0.870991 ± 0.007915 s` | **`4.894x` vs expert** |

Six alternating matched pairs all favored the final implementation. The mean
paired speedup was **4.893978x** with a 95% t-interval of
`4.700273x–5.087684x`. Against the previous PR candidate alone, delayed
normalization contributed **1.162204x** (6/6 wins).

The complete 300-step histories agree with the public expert to at most
`1.8e-6` over energy, loss, success probability, and mean log probability.

## Factor ablation

| Factor | Incremental speedup vs its parent |
| --- | ---: |
| Exact product-state reduction | **1.660x** |
| TensorCircuit `K.vmap` for local conditional maps | **2.063x** |
| Whole training/evolution in `K.jaxy_scan` | 1.149x |
| TensorCircuit `K.vmap` for one-qubit observables | 1.093x |
| Delay normalization and telescope probabilities | **1.162x** |

![Direct factor-removal comparisons](factor-ablation.svg)

*Figure — Direct parent comparisons for the three largest factors in the
original ablation campaign. Ratios are incremental, not multiplicative shares
of the final speedup. The later delayed-normalization refinement is reported
in the table from its own six-pair comparison.*

## Design fix in this PR

Post-selection now occurs **after each complete even+odd brickwork block**,
not between its two layers.

| Property | Original task | Patched task |
| --- | --- | --- |
| Measurement placement | After every layer | After each even+odd block |
| Selected events | 60 | 30 |
| Survivor structure | Exactly six one-qubit states | Odd survivors can entangle |
| Unitary layers / parameters / Adam updates | 10 / 230 / 300 | 10 / 230 / 300 |

The odd layer can now transmit correlations through an even qubit before that
qubit is projected. At a deterministic non-perturbative parameter point, the
first block has two nonzero Schmidt coefficients across an odd-survivor cut,
`0.999136` and `0.041559`; the reduced purity is `0.996552`, rather than
`1`. Thus the product-state induction used by the cheap reduction is false.

The regenerated public expert completes all 300 updates and passes every
functional criterion (`3.81 s` in the validation run). The unchanged
product-reduction implementation fails the patched evaluator's probability
consistency check. Its old-task timing remains above solely as evidence of
the loophole's impact and is not a speedup claim for the patched task.
