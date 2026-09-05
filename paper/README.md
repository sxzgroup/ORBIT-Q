# ORBIT-Q paper-update figures

This directory provides paper-ready updates to Fig. 1c, Fig. 2b, and Fig. 4
of [arXiv:2607.03105](https://arxiv.org/abs/2607.03105), plus one new summary
figure for AI-assisted optimization of the 12 human-expert implementations
and a separate paired Astra/expert runtime comparison. Updated 2026-09-05
with the complete campaigns in PRs #27, #28 and #29.
The manuscript masters are vector PDF; matched PNG files are included only for
inline GitHub/PR previews.
Fable 5 and DeepSeek max are intentionally excluded. Sol, Terra, Luna,
DeepSeek V4 Flash/Pro, Grok 4.5/4.6, and Astra use high thinking effort;
the explicitly named Sol ultra configuration uses ultra.

## Campaign results

| Model | Passed | Solve time (min) | Tokens (M) | Cost (USD) | Runtime / expert (Fig. 2b) | Source |
|---|---:|---:|---:|---:|---:|---|
| GPT-5.6 Sol | 10/12 | 197.70 | 26.071 | 25.527 | 2.541× | [#5](https://github.com/sxzgroup/ORBIT-Q/pull/5) |
| GPT-5.6 Sol ultra | 10/12 | 182.80 | 33.048 | 30.019 | 2.072× | [#6](https://github.com/sxzgroup/ORBIT-Q/pull/6) |
| GPT-5.6 Terra | 9/12 | 206.43 | 36.804 | 12.037 | 2.702× | [#20](https://github.com/sxzgroup/ORBIT-Q/pull/20) |
| GPT-5.6 Luna | 9/12 | 239.37 | 74.853 | 2.237 | 4.692× | [#21](https://github.com/sxzgroup/ORBIT-Q/pull/21) |
| DeepSeek V4 Flash | 5/12 | 290.41 | 81.585 | 0.573 | 3.476× | [#23](https://github.com/sxzgroup/ORBIT-Q/pull/23) |
| Grok 4.5 | 8/12 | 163.09 | 8.462 | 5.090 | 3.844× | [#26](https://github.com/sxzgroup/ORBIT-Q/pull/26) |
| DeepSeek V4 Pro | 5/12 | 324.22 | 67.180 | 1.101 | 2.588× | [#27](https://github.com/sxzgroup/ORBIT-Q/pull/27) |
| Grok 4.6 | 9/12 | 199.17 | 29.767 | 20.130 | 2.191× | [#28](https://github.com/sxzgroup/ORBIT-Q/pull/28) |
| GPT-6 Astra | 12/12 | 44.72 | 6.835 | 13.967 | 0.540× | [#29](https://github.com/sxzgroup/ORBIT-Q/pull/29) |

Reported outcomes retain each campaign's published audit/adjudication basis.
Astra's 12/12 is the original automatic verifier result, not a newly applied
blanket human adjudication. The maintainer's subsequent source review accepts
its Task 08 reduction ([review](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5548928438)).
Earlier campaigns' Task 08 failures are not retroactively changed.

## Updated Fig. 1c — agent–framework benchmark matrix

[PDF](updated_figures/fig1c_updated_agent_framework_matrix.pdf)

The matrix presents all reported configurations in one continuous layout,
without visually separating earlier and later evaluations. The added
TensorCircuit-NG results are Sol **10/12**, Sol ultra **10/12**, Terra **9/12**,
Luna **9/12**, DeepSeek V4 Flash **5/12**, Grok 4.5 **8/12**, DeepSeek V4 Pro
**5/12**, Grok 4.6 **9/12**, and Astra **12/12**. The count-based matrix palette
is unchanged; model-specific scatter markers and the Astra label use purple.

## Updated Fig. 2b — failure rate versus artifact runtime

[PDF](updated_figures/fig2b_updated_agent_axis.pdf)

All thirteen model points are retained. Astra uses the runtime comparison
below; other points retain their previous fixed-paper-reference values.
The linear axes add space left of zero failure and below unit runtime.

## Updated Fig. 4 — benchmark resource use

[PDF](updated_figures/fig4_updated_agent_framework_resources.pdf)

The original 2 × 3 layout is preserved. Panels (a–c) combine all fourteen agent
configurations; panels (d–f) retain the paper's original framework comparison.
Panels (b) and (e) use one consistent total-token encoding for every
configuration because the paper tables do not publish token components for
every legacy configuration. Bubble area represents total solver cost. All
three added models appear in wall-time, token, and cost panels. Their costs
per valid solution are **$0.220223** (DeepSeek Pro), **$2.236615** (Grok 4.6),
and **$1.163920** (Astra). Cost values retain the source campaigns' accounting.

## Astra / expert artifact runtime

[PDF](updated_figures/astra_expert_paired_runtime.pdf)

The [PR #29 runtime reply](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5549123804)
reports fresh runs on the same Apple M2 host, pinned TensorCircuit image and
6 CPU / 10 GiB limits. Following the [reviewer's suggestion](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5549266776),
Task 08 uses an **estimated 50 s** expert reference (25.05 s × 2, rounded),
compared with Astra's measured 16.31 s. The other eleven pairs are unchanged.
With this replacement, Astra is faster on **8/12** tasks. The geometric mean
Astra/expert ratio is **0.540×** (equivalently **1.85×** speedup), and totals
are **440.81 s / 640.11 s = 0.689×**. This includes 11 measured pairs and
one estimated expert reference, not 12 completed paired measurements.
The plot's right panel uses the reciprocal expert/Astra speedup, so >1 means
Astra is faster. Only the two exported Task 04 field names were restored in
its execution copy; the algorithms and original archive were unchanged.
The [updated table](astra_runtime_update.md) and
[plot data](source_data/astra_expert_runtime.csv) record the replacement.
The original [paired measurements](source_data/astra_expert_paired.csv),
including the missing expert Task 08 runtime, remain unchanged.

## New figure — human expert + AI co-optimization

[PDF](updated_figures/expert_optimization_updated_overview.pdf)

Panel (a) compares end-to-end runtime before and after optimization on a log
scale. Panel (b) reports the paired end-to-end speedup and the dominant insight
retained after ablation. The largest result is the exact Task 07 reduction
(**45.76×**); Task 01 reaches **9.64×**, while Tasks 03, 09, 10, and 12 reach
**3.82–4.90×**. Task 08 is included as a valid optimized implementation with a
small **1.04×** paired point estimate.
This concerns the optimized human-expert implementation and is separate from
the Task 08 benchmark outcomes of the solver campaigns above.
The legal TensorCircuit-native Task 05 result is used (**1.939× mean paired**).
This is the arithmetic mean of five paired public-expert/final-candidate runtime
ratios, not the ratio of the two reported mean runtimes. The candidate won all
five pairs; the median is **1.668×**. One `180.22 s` expert-reference long tail
raises the mean, while the four-pair sensitivity mean is **1.639×**. The direct
ablation attributes **1.420×** to the retained OMECo `4×4` path-search budget;
gate fusion alone remains unresolved.

Individual factor-removal plots remain available as supplementary evidence:

| Task | End-to-end result | Upstream PR |
|---:|---:|---:|
| 01 | **9.636×** | [#8](https://github.com/sxzgroup/ORBIT-Q/pull/8) |
| 02 | **1.116×** | [#9](https://github.com/sxzgroup/ORBIT-Q/pull/9) |
| 03 | **4.894×** | [#10](https://github.com/sxzgroup/ORBIT-Q/pull/10) |
| 04 | **2.602×** | [#11](https://github.com/sxzgroup/ORBIT-Q/pull/11) |
| 05 | **1.939× mean paired** | [#19](https://github.com/sxzgroup/ORBIT-Q/pull/19) |
| 06 | **1.504×** | [#13](https://github.com/sxzgroup/ORBIT-Q/pull/13) |
| 07 | **45.758×** | [#7](https://github.com/sxzgroup/ORBIT-Q/pull/7) |
| 08 | **1.045×** | [#18](https://github.com/sxzgroup/ORBIT-Q/pull/18) |
| 09 | **3.822×** | [#14](https://github.com/sxzgroup/ORBIT-Q/pull/14) |
| 10 | **4.898×** | [#15](https://github.com/sxzgroup/ORBIT-Q/pull/15) |
| 11 | **1.464×** | [#16](https://github.com/sxzgroup/ORBIT-Q/pull/16) |
| 12 | **3.914×** | [#17](https://github.com/sxzgroup/ORBIT-Q/pull/17) |

Source tables, adjudication notes, and the reproducible plotting script are in
[`source_data/`](source_data/) and
[`make_updated_paper_figures.py`](make_updated_paper_figures.py).
Run `python3 paper/verify_updated_results.py` from the repository root to check
the tables without plotting dependencies; see [FIGURE_QA.md](FIGURE_QA.md)
for regression tests and rendering checks.
