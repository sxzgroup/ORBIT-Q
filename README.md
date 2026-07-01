# ORBIT-Q

Open Research Benchmark for Integrated Tasks in Quantum Computing.

> Companion open-source benchmark framework for evaluating agentic quantum programming systems across framework-constrained implementation tasks.

ORBIT-Q is a Harbor-based benchmark suite for studying how autonomous coding agents implement nontrivial quantum-computing workloads under explicit framework constraints. The repository contains 12 canonical challenge tasks, framework-specific execution environments, solver and verifier adapters, and the tooling needed to reproduce framework-constrained agent evaluations.

This repository is intended to accompany a research manuscript. The final paper title, DOI/arXiv link, and citation metadata will be added when available.

## Overview

Modern coding agents can produce working scientific code, but quantum software evaluation requires more than numerical agreement on a visible test set. A solution may pass functional checks while bypassing the requested framework, using a hand-written simulator, exploiting evaluator details, or solving a relaxed version of the scientific problem. ORBIT-Q therefore evaluates each submission through a compound protocol:

- functional correctness on the original challenge-suite evaluator;
- timed execution of `run_solution(config)` with a 300 second cutoff;
- static policy checks for line count, imports, framework use, and obvious tampering;
- LLM-based source audit for framework compliance, problem fidelity, and simulator-bypass behavior;
- reproducible Harbor jobs with solver and verifier containers separated.

The benchmark currently covers the canonical task set:

```text
tasks/challenge-01
...
tasks/challenge-12
```

Each task is framework-neutral. The required framework is selected at run time through an appended prompt, a framework-specific Docker image, and verifier environment variables. The same canonical task directory can therefore be used to evaluate TensorCircuit, PennyLane, TorchQuantum, MindQuantum, or additional frameworks added later.

## Repository Structure

```text
.
|-- adapters/                  # Harbor adapters for agents, verifier, and Docker images
|-- frameworks/                # Per-framework Python dependency specifications
|-- images/framework/          # Shared Dockerfile for solver/verifier images
|-- prompts/frameworks/        # Generated framework-specific task instructions
|-- scripts/                   # Benchmark runners, generators, and diagnostics
|-- tasks/challenge-*/         # Canonical Harbor challenge tasks
|-- conf.toml                  # Public runner defaults
`-- templates/challenge/       # Source templates for generated tasks and verifier tests
```

The maintained source of verifier logic is `templates/challenge/tests/`. Do not hand-edit copied verifier files under `tasks/challenge-*` for lasting changes; edit the templates and regenerate tasks only when the task template or upstream problem source changes.

## Evaluation Protocol

Solutions must write `/root/solution_<id>.py` and expose:

```python
def run_solution(config):
    ...
```

The verifier computes:

```text
reward = functional_score * runtime_score * static_policy_score * llm_audit_score
```

Runtime scoring is:

```text
runtime <= 180s: runtime_score = 1
180s < runtime < 300s: linear decay
runtime >= 300s: runtime_score = 0
```

Important verifier fields include:

```json
{
  "reward": 0.0,
  "functional_score": 0.0,
  "runtime_score": 0.0,
  "runtime_sec": -1.0,
  "static_policy_score": 0.0,
  "llm_audit_score": 0.0,
  "framework_score": 0.0,
  "no_forbidden_framework_score": 0.0,
  "no_raw_simulator_bypass_score": 0.0,
  "no_static_cheating_score": 0.0
}
```

The functional evaluator prints:

```text
End-to-end solution time: XX.XXs
```

If this line is absent, the verifier records `runtime_sec = -1` and assigns zero runtime score.

## Experimental Records

Benchmark jobs write structured outputs under `jobs/`. A typical completed job contains Harbor metadata, solver logs, submitted artifacts, verifier logs, and `reward.json` files with the component scores described above.

Paper-facing aggregation tables and visualizations should be generated from version-controlled analysis code and archived job artifacts. They are not part of the canonical task definitions.

## Installation and Bootstrap

Run all commands from the repository root. The benchmark runner reads public
defaults from `conf.toml` and optional machine-local overrides from
`conf.local.toml`. `conf.local.toml` is gitignored and should contain only local
profile/model preferences, never API tokens.

The preferred local Harbor environment is:

```bash
./.conda/harbor-py312/bin/harbor --help
```

If this environment is absent, install Harbor outside the repository and keep the repository root on `PYTHONPATH` so Harbor can import the local adapters.

Build the selected framework image before running benchmark jobs:

```bash
FRAMEWORK=tensorcircuit bash scripts/build_challenge_quantum_image.sh
```

The default TensorCircuit image tag is:

```text
challenge-benchmark-quantum-tensorcircuit:py311
```

Verify that the image contains both solver CLIs:

```bash
docker run --rm challenge-benchmark-quantum-tensorcircuit:py311 \
  sh -lc 'codex --version && claude --version'
```

Framework images share `images/framework/Dockerfile`; Python dependencies are specified in `frameworks/<framework>/requirements.txt`. To build another framework image:

```bash
FRAMEWORK=pennylane bash scripts/build_challenge_quantum_image.sh
FRAMEWORK=torchquantum bash scripts/build_challenge_quantum_image.sh
FRAMEWORK=mindquantum bash scripts/build_challenge_quantum_image.sh
```

Recommended image tags follow:

```text
challenge-benchmark-quantum-<framework>:py311
```

## Running a Challenge

Use `scripts/run_harbor_challenge.py` so canonical task files remain fixed while
the framework, solver, Codex profile, and model settings come from `conf.toml`,
environment variables, local overrides, or explicit CLI arguments.

```bash
export OPENAI_API_KEY=...
FRAMEWORK=tensorcircuit

python3 scripts/run_harbor_challenge.py \
  --challenge 02 \
  --framework "$FRAMEWORK"
```

The tracked `conf.toml` uses Harbor's built-in Codex solver by default. To use a
private Codex profile, create `conf.local.toml`:

```toml
[run]
solver_agent = "codex-para"

[codex]
model = "YOUR_MODEL_NAME"
audit_model = "YOUR_AUDIT_MODEL_NAME"
profile = "your-profile"
force_auth_json = true
```

The runner passes:

```text
--extra-instruction-path prompts/frameworks/<framework>.md
--environment-import-path adapters.framework_docker:FrameworkDockerEnvironment
--environment-kwarg framework=<framework>
--environment-kwarg docker_image=challenge-benchmark-quantum-<framework>:py311
--agent-import-path harbor.agents.installed.codex:Codex
--verifier-import-path adapters.codex_para_verifier:CodexParaVerifier
--verifier-env REQUIRED_QUANTUM_FRAMEWORK=<framework>
```

To use Claude Code as the solver while keeping the verifier audit Codex-based:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL="your-claude-model"

MODEL_NAME="$ANTHROPIC_MODEL"
AUDIT_MODEL_NAME=gpt-5
FRAMEWORK=tensorcircuit

python3 scripts/run_harbor_challenge.py \
  --challenge 02 \
  --framework "$FRAMEWORK" \
  --solver-agent claude-code \
  --model "$MODEL_NAME" \
  --solver-reasoning-effort max \
  --audit-model "$AUDIT_MODEL_NAME"
```

The Claude adapter accepts either `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`
from the host and exposes `ANTHROPIC_API_KEY` inside Docker for Claude Code.

## Verifier-Only Candidate Check

To evaluate an already generated candidate without rerunning a solver, copy a task to a temporary location, replace the solution artifact, and run Harbor without an agent import path:

```bash
AUDIT_MODEL_NAME=gpt-5
FRAMEWORK=tensorcircuit
tmp_task="$(mktemp -d)/challenge-01-candidate-verify"

cp -R tasks/challenge-01 "$tmp_task"
cp jobs/<job>/<trial>/artifacts/root/solution_1.py \
  "$tmp_task/solution/solution_1.py"

PYTHONPATH="$PWD" ./.conda/harbor-py312/bin/harbor run \
  -p "$tmp_task" \
  --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
  --environment-kwarg "framework=$FRAMEWORK" \
  --environment-kwarg "docker_image=challenge-benchmark-quantum-$FRAMEWORK:py311" \
  --verifier-import-path adapters.codex_para_verifier:CodexParaVerifier \
  --verifier-kwarg "audit_model=$AUDIT_MODEL_NAME" \
  --verifier-env "REQUIRED_QUANTUM_FRAMEWORK=$FRAMEWORK" \
  -n 1 \
  -o "$PWD/jobs" \
  --job-name challenge-01-candidate-verifier \
  --yes
```

This path is useful for source-audit reruns, policy changes, and verifier debugging. It should not be used with hidden upstream baselines as the required smoke test, because some upstream baselines depend on unreleased framework features.

## Benchmark Maintenance

Regenerate framework prompts and canonical tasks only when prompt templates, task templates, or upstream problem files change:

```bash
python3 scripts/generate_framework_prompts.py \
  --framework tensorcircuit pennylane torchquantum mindquantum
python3 scripts/generate_tc_challenge_tasks.py
```

By default, the task generator expects the upstream TensorCircuit challenge suite source at:

```text
../tensorcircuit/examples/challenge_suite
```

Override it with:

```bash
export TC_CHALLENGE_SUITE_SOURCE=/path/to/tensorcircuit/examples/challenge_suite
```

Do not regenerate `tasks/challenge-*` while a Harbor job is running. Local Harbor uses the live task directory as both Docker context and compose project directory.

## Diagnostics

Inspect a completed or running job:

```bash
python3 scripts/inspect_harbor_job.py jobs/<job-name> --stale-minutes 10
```

Useful Docker checks:

```bash
docker ps --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
docker stats --no-stream
docker top <container-id> -eo pid,ppid,etime,stat,pcpu,pmem,args
```

If Harbor reports that Docker is not running while `docker ps` works, the failure is often caused by sandboxed Docker-socket access. Retry the same Harbor or Docker command with the required Docker permissions before concluding that an image is missing or Docker is unavailable.

## Reproducibility Notes

- The benchmark intentionally uses stable framework-specific local images instead of per-task Dockerfiles.
- `tasks/challenge-*` contain only framework-neutral task definitions and a structural `environment/` marker required by Harbor.
- The framework constraint enters through the appended prompt, selected Docker image, and verifier policy variable.
- `NUMBA_DISABLE_JIT=1` is set for agent and verifier containers to avoid `quimb`/`numba` illegal-instruction failures on Docker Desktop aarch64.
- Jobs are written under `jobs/`; these logs are experimental artifacts and are not required to modify benchmark source code.
- Secrets, custom Codex profiles, auth files, and model catalogs must remain outside the repository.

## Citation

If you use ORBIT-Q in academic work, please cite the accompanying manuscript once it is available. Until then, use the following provisional citation:

```bibtex
@misc{orbitq2026,
  title        = {ORBIT-Q: Open Research Benchmark for Integrated Tasks in Quantum Computing},
  author       = {{ORBIT-Q Maintainers}},
  year         = {2026},
  note         = {Open-source benchmark framework; manuscript in preparation}
}
```

## License

The public release license has not yet been finalized. Add a `LICENSE` file before archival release or external distribution.
