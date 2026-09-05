# Figure QA record

Date: 2026-09-05

- Regenerated Fig. 1c, Fig. 2b and Fig. 4; added a separate Astra/expert
  paired-runtime figure. Each has a vector PDF master and matched 300-dpi PNG.
- Rendered all four changed PDFs with Poppler and visually inspected them.
  Adjusted scatter labels to remove overlap; retained direct labels without
  colored leader lines. Astra uses purple; the count matrix, passed/failed
  bars, and total-token bars retain their shared encodings.
- All five PDFs, including the unchanged expert-optimization overview, have
  one page, embedded TrueType fonts, and no raster image objects. SVG/TIFF
  are intentionally omitted; PDF is the publication master.
- Validated 14 agent rows and 108 unique task outcomes. Added-campaign pass
  counts are **10, 10, 9, 9, 5, 8, 5, 9, 12**. Task 08 source outcomes are
  preserved: failed for the first eight campaigns, passed for Astra.
- Cross-checked the 36 new task records and resource totals against pinned
  PR #27–29 evidence; verified the three original source-file SHA-256 hashes
  against local Git objects.
- Fig. 2b retains all thirteen model points and all previous non-Astra
  coordinates. Astra alone uses the updated artifact comparison. Both axes are linear,
  with limits (-10, 72) and (-0.6, 9.2) for visual padding, not negative data.
- Inspected the revised PDF after Poppler rendering: Astra is below the
  reference anchor; no footer, dagger, numeric Astra runtime annotation,
  colored leader lines, or label overlap is present.
- Independently checked the updated ratios and reciprocal speedups: **12**
  comparisons, consisting of **11 measured pairs and one estimated reference**.
  Task 08 uses **50.00 s / 16.31 s**, giving **3.065604×** expert/Astra speedup.
  Astra is faster on **8/12** tasks; geometric mean Astra/expert is
  **0.539698×**, with totals **440.81 / 640.11 s** (ratio **0.688647×**).
  The raw Task 08 OOM record and original paired-source hash are unchanged.
- Re-rendered both runtime figures after the replacement and inspected the
  actual PDF pages. The per-task plot now contains all 24 runtime bars.
  Programmatically checked all thirteen scatter coordinates, including
  Astra at (0, 0.53969753); every non-Astra coordinate remains unchanged.
  The matrix and solver-resource PNG hashes are also unchanged.
- Twelve regression tests pass, including rejection of missing tasks, changed
  outcomes, zero Grok cost, mixed runtime baselines, filled OOM times, and
  inverted speedups, an incorrect replacement, an estimate labeled as measured,
  changes to unrelated pairs, stale aggregates, and inflated measured-pair
  counts. `git diff --check` passes.
- The earlier 72 task records and all original numeric agent rows are retained.
  The original framework and expert-reference tables, expert optimizations,
  insights, and expert-overview PDF/PNG are unchanged from the PR head.
- Fable 5 and DeepSeek max remain excluded. No benchmark was rerun for this
  paper-summary refresh, and unrelated worktree changes were not included.

Reproduce from the repository root:

```bash
python3 paper/verify_updated_results.py
python3 -m unittest discover -s paper -p test_updated_results.py
python3 paper/make_updated_paper_figures.py
```

With the pinned campaign commits available locally, additionally run
`python3 paper/verify_updated_results.py --check-git-sources`.
