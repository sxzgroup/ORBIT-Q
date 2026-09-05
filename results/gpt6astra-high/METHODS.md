# Methods and provenance

This appendix accompanies the [concise report](README.md). Candidate code,
numeric rewards and original provenance are unchanged by the reporting update.

## Execution and isolation

- Run date: 2026-09-05 Asia/Shanghai; twelve sequential jobs from
  2026-09-04 19:13:25 to 20:16:38 UTC, about 63.2 minutes including verification.
- Solver: `gpt-6-astra`, `high`; separate audit: `gpt-5.6-sol`, `high`.
- One trial per task, zero Harbor retries; 6 CPUs, 10,240 MiB memory,
  16,384 MiB storage; agent/verifier timeouts 1,800/1,200 seconds.
- Independent branch: `codex/gpt-6-astra-high-clean-benchmark`, from
  `0201238ec2983907e2891f5319f5fff2d00844d5`. Execution task copies match the
  historical Sol copies byte-for-byte after the same resource adaptation.
  Canonical tasks, prompts, templates and adapters remain unchanged.
- The derived image adds solver Codex CLI 0.153.3 because the service rejects
  Astra on the original CLI. Verifier `PATH=/usr/local/bin:/usr/bin:/bin`
  retains CLI 0.145.0. Parent image SHA starts `b059c5fa7f75`; quantum packages
  are unchanged: TensorCircuit nightly 1.8.0.dev20260726, JAX 0.10.0,
  NumPy 2.2.6. Harbor is 0.20.0. Full image hashes are in [protocol.json](protocol.json).
- Solver and verifier use fresh separate containers and Codex homes. No prior
  solutions, audit reasoning, host repository, scratch directory, conversation,
  skills or MCP servers are supplied. Only new-trial log directories are mounted.
  All twelve solver sessions are unique; no candidate was edited or given hints.
- [Container evidence](container-evidence/) and the [isolation audit](contamination-audit.json)
  record mounts, resources, session metadata and candidate/task hashes without
  credentials. They establish execution isolation, not absence of training-data exposure.

## Comparison definitions

Historical sources are pinned to `a395c1cd0444630f2244fb89e3bcecde667e3372`
(Sol high) and `552481c7d133c1bc48e7b0e379cab8fcfaf3851b` (Sol ultra).
Automatic reward is functional × static policy × LLM audit; runtime is separate.
Historical final validity includes documented adjudication; Astra has none.

The [upstream PR #28](https://github.com/sxzgroup/ORBIT-Q/pull/28) supplies the
two-figure layout: a task-outcome matrix and a three-panel resource figure
(per-task solve time, token composition, cost/time per passing solution).
The outcome matrix and panel (c) include its eight campaigns: Sol high/ultra,
Terra high, Luna high, DeepSeek V4 Flash/Pro high and Grok 4.5/4.6 high, plus
Astra high. Their published pass sets and resource totals are pinned to commit
`09476b847cc2ff53603002dbf0bf7f9c34a148cd` in
[public-model-comparison.json](public-model-comparison.json), with source paths
and cost basis. The Sol rows match the existing comparison data.
Historical outcomes retain the source's final-review basis; Astra is automatic
only. Panels (a)/(b) retain Astra's own measurements and original colors.

Reported runtime ratios use frozen Sol high, not an expert reference:
no same-run expert baseline was measured for Astra. Aggregate ratios are
`exp(mean(log(ratio)))` on tasks 02–07 and 09–12, the same ten common-passing
tasks. Tasks 01/08 remain in the outcome matrix but not the runtime aggregate.

Solver resource totals include all twelve tasks, including failed candidates.
Per-pass cost/time divides each total by that campaign's published final pass
count (5–10), or twelve Astra automatic passes, different review bases. Token
stacks are non-cache input (`input − cache`), cache input, and output; cache is
not added twice. Costs retain the values in each source archive, exclude verifier calls and are
not normalized to a common price schedule; Astra also excludes its task-free
preflight calls.

Each configuration has one trial per task. Timing differences are descriptive,
not significance tests. Dates, actual backend conditions and Astra's solver CLI
differ; the comparison is not a controlled model-only or Astra-effort experiment.

## Archive and reproduction

The submission branch `codex/gpt-6-astra-high-tc18-results` is based directly
on upstream `v2` at `2187118fecc0e275efc0763c676baa8e2df57108`; it adds only
`results/gpt6astra-high/`. This packaging base is separate from experimental
base `0201238`. The original experiment branch, candidates, numeric results,
raw audit/trial records and protocol are preserved; no solver or verifier was
rerun during packaging. See [submission.json](submission.json).

Each challenge directory contains the candidate, reward, functional output,
audit, job/trial metadata and resource/hash stamp. Full sessions and private
credentials are not committed. Harbor redacted some literal JSON booleans into
unquoted `[REDACTED]` markers: original bytes remain in `*.raw.txt`, while
parseable JSON uses `null` for unknown values and records the normalization.
Numeric scores, timings, token counts and costs are unaffected.

### Offline verification

Run from any checkout, or pass a relocated archive directory. Python 3.10+
and its standard library suffice; no Git history, Docker, Harbor, API access,
credentials, original task copies or host job/session logs are needed:

```bash
python3 results/gpt6astra-high/verify_archive.py
python3 results/gpt6astra-high/verify_archive.py --archive-dir /path/to/gpt6astra-high
python3 -B results/gpt6astra-high/test_verify_archive.py
```

The verifier is read-only. It enforces all twelve tasks, validates the file
inventory and SHA-256 checksums in [archive-manifest.json](archive-manifest.json),
checks raw/normalized JSON equivalence, candidate hashes, reward products,
model/effort/image/resource records, session metadata, token/cost/time totals,
comparison ratios, model labels, figure formats and documentation links.
Regression tests use temporary copies to ensure missing tasks, altered
artifacts, inconsistent scores/configs/resources and corrupt figures fail.

This checks **archived evidence**, not a new functional or semantic evaluation.
The pinned task-hash manifest is checked, but absent task source bytes and
unarchived solver transcripts cannot be independently reconstructed from it.
The checksum manifest detects accidental archive changes; it is not an
independent signature or proof of uncontaminated execution.

### Figure reproduction

With NumPy and Matplotlib installed, run from the worktree root:

```bash
python3 results/gpt6astra-high/make_figures.py
```

The committed exports were generated with Matplotlib 3.10.8. Byte-for-byte
reproduction was checked in a relocated archive; other plotting environments
can produce different bytes with the same data. Verify the committed archive
before overwriting exports. Do not refresh checksums just to hide a mismatch.

### Historical host-only utilities

`run_benchmark.py`, `inspect_run.py`, `summarize.py` and `check_isolation.py`
preserve the original campaign workflow and rely on the original machine's
paths, Git objects and/or job logs. They are **not** the offline verification
entry point. In particular, `check_isolation.py` recomputes the isolation
report from original logs; those logs are intentionally not in the PR.

A new experimental reproduction requires a separate checkout of experimental
base `0201238`, the pinned parent image and CLI-only [Dockerfile.cli](Dockerfile.cli),
Harbor 0.20.0, credentials, and explicit adaptation of every host/output path.
Use new task/job/archive directories: the historical runner can write result
files and must not be run against this frozen submission archive. No experiment
needs to be repeated to review or submit these results.
