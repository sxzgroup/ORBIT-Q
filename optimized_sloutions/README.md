# Optimized Solution Variants

`tasks/challenge-*/solution/solution_N.py` is the publication reference copied from the original challenge-suite source and is the default Harbor oracle artifact. Keep those files stable unless the published reference itself is intentionally revised.

This directory stores tracked, reference-derived variants for analysis and future benchmark evolution. Variant files should keep the evaluator contract (`run_solution(config)`) but are not used by default Harbor runs.

Naming convention:

- `challenge-XX/solution_N_<variant>.py` for a concrete implementation variant.
- Prefer descriptive suffixes such as `mpo`, `scan`, or `framework_primitive` over chronological names like `v2`.

To verify a variant, copy it into a temporary task as `/solution/solution_N.py` or otherwise make the variant module importable and call the evaluator directly with `--solution`.

## Local Runtime Records

Host: MacBook Pro with Apple M4 Pro, 14-core CPU, 48 GB memory. Software: JAX 0.10.0 and JAXLIB 0.10.0. Unless noted otherwise, times below are evaluator-reported `End-to-end solution time` values.

| File | Date | Command summary | End-to-end time | Result |
| --- | --- | --- | ---: | --- |
| `challenge-01/solution_1_mpo.py` | 2026-07-07 | `evaluate_1.py --solution solution_1_mpo --max-steps 500` | 19.34s | PASS |
| `challenge-05/solution_5_omeco.py` | 2026-07-07 | `evaluate_5.py --solution solution_5_omeco --max-steps 600` (reference baseline 48.27s) | 34.85s | PASS |
