# Gemini 4 Hydrogen high benchmark results

**11/12 automatic passes** on TensorCircuit 1.8 in **Antigravity** (`antigravity-cli 1.5.17`), audited by Gemini 4 Neon high in Antigravity. Selected solver executions: **135.21 awake minutes**, **$7.5309** API-equivalent recorded. Geometric mean artifact/expert runtime: **1.031×** (10 measured pairs and one estimated expert reference; lower is faster).

The reported cohort is the complete 12-task round executed in Antigravity (`antigravity-cli 1.5.17`, `gemini-4-hydrogen-high`, `high` effort) and uniformly audited in Antigravity by `gemini-4-neon-high` (`high` effort). Task 01 timed out at the 1,800 s agent limit without submitting a solution.

| Challenge | Functional | Static | Audit | Expert (s) | Artifact (s) | Artifact / expert |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0 | 0 | 0 | 59.83 | — | — |
| 02 | 1 | 1 | 1 | 4.41 | 9.88 | 2.24× |
| 03 | 1 | 1 | 1 | 3.88 | 4.85 | 1.25× |
| 04 | 1 | 1 | 1 | 14.27 | 13.90 | 0.97× |
| 05 | 1 | 1 | 1 | 93.60 | 112.36 | 1.20× |
| 06 | 1 | 1 | 1 | 44.67 | 27.72 | 0.62× |
| 07 | 1 | 1 | 1 | 121.71 | 167.29 | 1.37× |
| 08 | 1 | 1 | 1 | 50.00 (est.) | 19.66 | 0.39× |
| 09 | 1 | 1 | 1 | 25.10 | 6.97 | 0.28× |
| 10 | 1 | 1 | 1 | 18.40 | 84.85 | 4.61× |
| 11 | 1 | 1 | 1 | 165.32 | 137.93 | 0.83× |
| 12 | 1 | 1 | 1 | 9.83 | 11.68 | 1.19× |

Direct same-host measurement unavailable: 01 (no valid submission); 08 (expert exceeded the memory limit). [Replay evidence](timing-evidence.json).

Task 08 uses the [reviewer’s expert-reference replacement](https://github.com/sxzgroup/ORBIT-Q/pull/29#issuecomment-5549266776): 25.05 s × 2, rounded to **50 s**. Only the expert reference is estimated. The measured-only geometric mean is 1.135× (10 pairs).

[Methods](METHODS.md) · [Trial records](trials.json) · [Published runtimes](published-runtimes.json) · [Measured pairs](paired-runtimes.json)

![Task outcomes](figures/task-outcomes.png)

![Solver time, tokens and cost efficiency](figures/agent-resources.png)

![Square resource-efficiency panel](figures/resource-efficiency-square.png)

![Paired runtimes and speedup](figures/artifact-vs-expert.png)

![Failure rate and runtime relative to the expert](figures/artifact-efficiency.png)
