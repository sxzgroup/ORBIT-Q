# Source-data notes

- `paper_agent_axis.csv`: original paper agent-axis totals plus Sol, Sol ultra,
  Terra, Luna, DeepSeek V4 Flash/Pro, Grok 4.5/4.6, and Astra (14 rows). It drives updated Fig. 1c,
  and the top row of Fig. 4. Its `gm_slowdown` and `include_fig2b` fields
  drive Fig. 2b except that Astra uses `astra_expert_runtime.csv`.
  Unqualified added configurations use
  high thinking effort; Sol ultra uses ultra.
- `paper_framework_axis.csv`: original paper framework-axis totals for the
  unchanged bottom row of Fig. 4.
- `paper_expert_runtimes.csv`: original public per-task TensorCircuit expert
  references retained to validate the archived fixed-denominator calculations.
- `benchmark_models.csv`: compact new-campaign aggregate table.
- `task_outcomes.csv`: 9 configurations × 12 tasks. `raw_reward` preserves the
  verifier result; `final_pass` retains each source campaign's reported outcome,
  including its published adjudication where applicable.
- `recent_pr_evidence.json`: pinned commits, source-file SHA-256 hashes, and
  normalized per-task evidence for PRs #27–29, plus paired-run provenance.
- `astra_expert_paired.csv`: 12 fresh paired-runtime records from the
  [PR #29 reply](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5549123804).
  Task 08 has a missing expert time (OOM) and is excluded from aggregates.
- `astra_expert_runtime.csv`: derived plot data for all 12 Astra tasks.
  Eleven pairs retain the fresh measurements; Task 08 uses the estimated
  **50 s** expert reference requested in the [PR #29 discussion](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5549266776)
  (25.05 s × 2, rounded). Its measured Astra time is 16.31 s.
  This table supplies the Astra Fig. 2b point (**0.539698×**) and runtime bars.
  `expert_reference_basis` distinguishes the estimate from measurements.
- `astra_expert_adjustment.json`: replacement provenance, original-source
  hash, and recomputed aggregate values. Neither the original paired table
  nor the historical `paper_agent_axis.csv` measurements are overwritten.
- `expert_optimization.csv`: original and optimized expert runtimes plus the
  dominant factor retained after ablation.
- `insights.csv`: short take-home messages for the expert optimizations.

The original values come from the main and supplementary tables of
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105). The new campaign totals
come from ORBIT-Q PRs #5, #6, #20, #21, #23, #26, #27, #28, and #29.
Fig. 2b retains the fixed-paper-reference coordinates for all non-Astra models;
Astra uses its paired remeasurement with the Task 08 replacement described above.
The original CSV measurements are unchanged.

The original paper tables expose total token use for the legacy configurations
but not the full cache/non-cache/output decomposition. Updated Fig. 4b/e
therefore compare total tokens with one consistent color for every bar.

Task 08 is final `F` for the first eight added solver configurations. Luna and Sol ultra
retain raw reward `1` in `task_outcomes.csv`; `final_pass=0` records the
paper-facing decision. Astra retains its original 12/12 automatic result;
the maintainer subsequently accepted its Task 08 implementation in
[review](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5548928438).
The newly added campaigns are not described as uniformly human-adjudicated.
This solver-campaign adjudication is separate from the
valid Task 08 human-expert optimization recorded in `expert_optimization.csv`.
Task 05 uses the legal TensorCircuit-native result from
[PR #19](https://github.com/sxzgroup/ORBIT-Q/pull/19): **1.939×** mean paired
end-to-end speedup (median **1.668×**). This is the mean of the five paired
expert/candidate ratios; one `180.22 s` expert-reference long tail raises it,
and the four-pair sensitivity mean is **1.639×**. The candidate won 5/5 pairs.
The earlier custom no-QR MPS result is excluded.
