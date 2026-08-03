# Challenge 10: specialize the cold contraction program

**Take-home insight.** The speedup is not caused by an intrinsically cheaper
MPO gate or by a large reduction in contraction FLOPs. It comes from replacing
a generic TensorNetwork that must be built, planned, differentiated, and
compiled in every cold process with a static sequence of local MPS/MPO
contractions. The known bond-dimension bound (`1 -> 2 -> 4`) makes this
specialization exact.

## Result and attribution

The public expert-to-candidate campaign measured **`4.898x`**
(`4.598–5.199x`, five paired runs): `18.931 s` to `3.869 s`.

Additional five-pair cold-JIT ablations separate the mechanism:

| Change | Paired speedup (95% t-CI) | Meaning |
|---|---:|---|
| Native hyperedge → same graph with a fixed path | **`1.626x`** (`1.373–1.878x`) | CopyNode-amplified path search is the largest isolated factor |
| Fixed-path hyperedge → fixed-path MPO | `1.011x` (`0.965–1.057x`) | No resolved gate-level MPO advantage |
| Fixed-path generic MPO → fused rotations | **`1.210x`** (`1.161–1.258x`) | Less graph construction and preprocessing |
| Fused fixed-path generic graph → local MPS | **`1.158x`** (`1.119–1.198x`) | A smaller, more regular traced and differentiated program |

The rows are independent paired experiments; their ratios are not additive.

![Task 10 factor ablation](factor-ablation.svg)

## What is actually cheaper

The native CMZ is already an exact bond-2 hyperedge representation. Its
`CopyNode`s make the current contractor skip `_merge_single_gates`, so OMECo
searches a 424-tensor/423-step graph instead of the preprocessed MPO's
86-tensor/85-step graph (about `3.5 s` versus `0.7 s` of search).

This does not imply a large execution-cost difference: both generic paths have
the same maximum intermediate size (`96` elements) and nearly identical
estimated FLOPs. Once the path is fixed, hyperedge and MPO runtimes are
indistinguishable.

The headline campaign was therefore a **cold specialization** win. Profiling
measured lowering plus XLA compilation at `17.514 s` for the expert and
`3.541 s` for the candidate, while compiled numerical execution took only
milliseconds. The local MPO is an implementation choice; the important change
is replacing dynamic generic graph planning and compilation with a fixed exact
contraction program.
