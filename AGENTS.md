# AGENTS.md

This repository hosts a Harbor benchmark suite for 12 standard quantum challenge tasks.

Active tasks:

```bash
tasks/challenge-01
...
tasks/challenge-12
```

Do not use the removed `physical-sciences/physics/...` smoke path as the active benchmark.

## Quick Bootstrap

Run all commands from the repository root.

Use the local Harbor Python environment if it exists:

```bash
./.conda/harbor-py312/bin/harbor --help
```

If this environment is absent, create or install Harbor outside this repository, then keep using the repo root on `PYTHONPATH` so Harbor can import local adapters.

Build or refresh the framework-specific Docker image used by both agent and verifier. This is the first command to run on a new machine after Docker is available:

```bash
FRAMEWORK=tensorcircuit bash scripts/build_challenge_quantum_image.sh
```

The default TensorCircuit tag is:

```text
challenge-benchmark-quantum-tensorcircuit:py311
```

Verify that the image contains Codex CLI:

```bash
docker run --rm challenge-benchmark-quantum-tensorcircuit:py311 codex --version
```

Generate framework prompt files and canonical tasks only when templates or upstream problem files change:

```bash
python3 scripts/generate_framework_prompts.py --framework tensorcircuit pennylane
python3 scripts/generate_tc_challenge_tasks.py
```

By default the generator expects the upstream challenge-suite source repo next to this repository at `../tensorcircuit/examples/challenge_suite`. Override it with `TC_CHALLENGE_SUITE_SOURCE` or `--source` if needed.

Do not regenerate `tasks/challenge-*` while a Harbor job is running. Local Harbor uses the live task directory as Docker context and compose project directory.

## Framework Image Strategy

The benchmark intentionally uses stable framework-specific local images instead of asking Harbor to rebuild each task environment. Tasks do not carry Dockerfiles. Image definitions and framework Python dependencies are maintained under:

```bash
images/framework/Dockerfile
frameworks/<framework>/requirements.txt
scripts/build_challenge_quantum_image.sh
```

The canonical task content is framework-neutral, and `tasks/challenge-*` do not declare framework Docker images. The run command selects the image through a small environment adapter:

```bash
FRAMEWORK=tensorcircuit bash scripts/build_challenge_quantum_image.sh
python3 scripts/run_harbor_challenge.py --challenge 01 --framework tensorcircuit
```

If only the prompt changes and the required framework/image are unchanged, do not rebuild the image or regenerate tasks. Generate or edit the prompt file and select it with `--extra-instruction-path` or `EXTRA_INSTRUCTION_PATH`.

If a new framework needs additional Python packages, do not overwrite the TensorCircuit image tag. Add `frameworks/<framework>/requirements.txt`, then build a framework-specific tag with the shared Dockerfile. Use `FRAMEWORK=<framework>` for the standard naming path, `REQUIREMENTS_FILE=<path>` to override the requirements file, or `IMAGE=<explicit-tag>` when the tag must be controlled exactly.

Recommended tag pattern:

```text
challenge-benchmark-quantum-<framework>:py311
```

Examples:

```bash
FRAMEWORK=tensorcircuit bash scripts/build_challenge_quantum_image.sh
FRAMEWORK=pennylane bash scripts/build_challenge_quantum_image.sh
FRAMEWORK=my-framework REQUIREMENTS_FILE=frameworks/my-framework/requirements.txt \
  bash scripts/build_challenge_quantum_image.sh
```

To run with another framework, build that framework's image and pass the framework on the command line:

```bash
FRAMEWORK=pennylane bash scripts/build_challenge_quantum_image.sh
python3 scripts/run_harbor_challenge.py --challenge 01 --framework pennylane
```

`scripts/run_harbor_challenge.py` passes `--environment-import-path adapters.framework_docker:FrameworkDockerEnvironment`, `--environment-kwarg framework=<framework>`, and `--verifier-env REQUIRED_QUANTUM_FRAMEWORK=<framework>`. The adapter sets the selected image in memory before Docker starts; it does not modify, copy, or patch canonical task files.

When using a private or custom Codex profile, always pass model names explicitly:

```bash
MODEL_NAME=AWS-GPT-5.5
AUDIT_MODEL_NAME=AWS-GPT-5.5
```

`MODEL_NAME` is consumed by Harbor's `-m` flag. `AUDIT_MODEL_NAME` is passed through `--verifier-kwarg "audit_model=$AUDIT_MODEL_NAME"` and becomes `CODEX_AUDIT_MODEL` inside the verifier container. If the same model should solve and audit, set both variables to the same value.

## Run A Challenge With CodexPara

Use the wrapper so `tasks/challenge-*` stay fixed while the framework is selected from command-line arguments:

```bash
MODEL_NAME=AWS-GPT-5.5
AUDIT_MODEL_NAME="$MODEL_NAME"
FRAMEWORK=tensorcircuit
python3 scripts/run_harbor_challenge.py \
  --challenge 02 \
  --framework "$FRAMEWORK" \
  --model "$MODEL_NAME" \
  --audit-model "$AUDIT_MODEL_NAME"
```

If Harbor reports Docker is not running but `docker ps` works, this is usually Codex sandbox permission around Docker preflight/socket access. Re-run the same Harbor command with escalated Docker access.

Local Harbor also requires each task to contain an `environment/` directory before `-p tasks/challenge-XX` is recognized as a task. In this benchmark that directory is intentionally just a structural marker; framework Dockerfiles and image tags stay outside the task tree.

## Verifier-Only Candidate Check

To verify an already generated solution without rerunning the Agent, copy the task to a temporary directory, replace the `solution/solution_<id>.py` file with the candidate artifact, then run Harbor without an agent import path. Harbor's default oracle path only copies `solution/` into `/root`; it does not ask an Agent to solve again.

Example for challenge 01:

```bash
AUDIT_MODEL_NAME=AWS-GPT-5.5
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

Do not use hidden upstream baselines as the required verifier smoke test. Some baselines depend on unreleased TensorCircuit-NG features such as `tc.set_contractor("omeco")`; these can fail under the pinned public `tensorcircuit-ng==1.7.0` image even when the verifier pipeline is healthy.

## What Differs From Vanilla Harbor

Tasks are generated flat as `tasks/challenge-01` through `tasks/challenge-12`, not nested by domain/subfield.

Framework constraints are injected through Harbor's supported extra instruction mechanism:

```bash
--extra-instruction-path prompts/frameworks/tensorcircuit.md
```

This avoids modifying Harbor itself for variable-like prompt content. The prompt generator can create framework-specific files, and the run command selects which one to append.

Framework images are selected by `adapters.framework_docker:FrameworkDockerEnvironment` at runtime. The 12 problem statements and canonical `task.toml` files do not encode TensorCircuit, PennyLane, or any other target framework; the selected framework enters through the extra prompt, the environment adapter, and verifier policy variables.

Within a given framework, the agent and separate verifier use the same prebuilt framework image selected by command-line kwargs:

```bash
--environment-import-path adapters.framework_docker:FrameworkDockerEnvironment
--environment-kwarg framework=<framework>
--environment-kwarg docker_image=challenge-benchmark-quantum-<framework>:py311
--verifier-env REQUIRED_QUANTUM_FRAMEWORK=<framework>
```

This avoids rebuilding heavy quantum dependencies per run and avoids storing repeated environment files under each task.

Codex is run through local adapters:

```bash
--agent-import-path adapters.codex_para:CodexPara
--verifier-import-path adapters.codex_para_verifier:CodexParaVerifier
```

`CodexParaVerifier` exists because Harbor agent and verifier interfaces are different. It reuses `CodexPara` auth/profile/catalog setup, then lets Harbor run `/tests/test.sh` and parse `/logs/verifier/reward.json`.

The shared Dockerfile bakes in common system tooling:

```text
python:3.11-slim
nodejs/npm/ripgrep
@openai/codex
build-essential, pkg-config, rustc, cargo
```

Python packages are installed from `frameworks/<framework>/requirements.txt`. For example, TensorCircuit uses `frameworks/tensorcircuit/requirements.txt`, which currently includes NumPy/SciPy/PyTest, TensorCircuit-NG, TensorNetwork-NG, JAX/JAXLIB, Optax, Quimb, and OMECo.

The build uses TUNA apt/PyPI mirrors and the adapter's npm fallback uses:

```bash
https://registry.npmmirror.com
```

`NUMBA_DISABLE_JIT=1` is set for both agent and verifier to avoid `quimb`/`numba` illegal-instruction failures on Docker Desktop aarch64/LinuxKit.

## Maintained Files

Prefer editing these source files:

```bash
scripts/generate_tc_challenge_tasks.py
images/framework/Dockerfile
frameworks/<framework>/requirements.txt
templates/challenge/
prompts/framework_prompt_template.md
scripts/generate_framework_prompts.py
prompts/frameworks/
adapters/codex_para.py
adapters/codex_para_verifier.py
adapters/framework_docker.py
scripts/inspect_harbor_job.py
scripts/build_challenge_quantum_image.sh
AGENTS.md
```

Do not hand-edit copied verifier files under `tasks/challenge-*` for lasting changes. Edit `templates/challenge/tests/` and regenerate tasks.

## Prompt Policy

The main task instruction excludes baseline and hint sections. The appended framework prompt should:

- Require the core quantum computation to use the selected quantum framework.
- Allow general support libraries such as NumPy/JAX/SciPy/Optax when appropriate.
- Forbid solving by switching to a different quantum framework.
- Discourage replacing the framework with a full raw NumPy/JAX/SciPy simulator.
- Require `/root/solution_<id>.py` with `run_solution(config)`.
- State that timed execution must finish within 300 seconds.
- Encourage inspecting installed framework source/examples/APIs before implementing.
- Encourage fast end-to-end iteration and framework-native performance.
- Allow the Agent to declare failure if the selected framework has a real obstruction.

## Scoring

Verifier output is written under `/logs/verifier/` and collected into each job.

Main reward:

```text
reward = functional_score * runtime_score * static_policy_score * llm_audit_score
```

Important fields:

```json
{
  "reward": 0.0,
  "functional_score": 0.0,
  "runtime_score": 0.0,
  "runtime_sec": -1.0,
  "static_policy_score": 0.0,
  "llm_audit_score": 0.0,
  "line_count_score": 0.0,
  "framework_score": 0.0,
  "no_forbidden_framework_score": 0.0,
  "no_raw_simulator_bypass_score": 0.0,
  "no_static_cheating_score": 0.0
}
```

Runtime scoring:

```text
<= 180s: runtime_score = 1
180s to 300s: linear decay
>= 300s: runtime_score = 0
```

The functional evaluator prints:

```text
End-to-end solution time: XX.XXs
```

`score_submission.py` parses that line. If the line is missing, `runtime_sec=-1` and runtime score is zero.

Static policy enforces a 200 effective Python line limit and checks imports, forbidden quantum frameworks, obvious test/reward tampering, and raw simulator hints.

LLM audit is performed by Codex inside the verifier container. The parser scans Codex output from the end and accepts the first JSON object containing the required policy keys. This avoids failures caused by source-code braces or repeated JSON objects in Codex CLI output.

## Diagnostics

Inspect a job:

```bash
python3 scripts/inspect_harbor_job.py jobs/<job-name> --stale-minutes 10
```

Use Docker state directly when available:

```bash
docker ps --format '{{.ID}} {{.Image}} {{.Status}} {{.Names}}'
docker stats --no-stream
docker top <container-id> -eo pid,ppid,etime,stat,pcpu,pmem,args
```

To distinguish slow execution from a hang, check whether `docker top` shows an active evaluator or Codex audit process and whether CPU usage is nonzero. For challenge 01 verifier, a normal functional stage showed `evaluate_1.py` using roughly 4-5 CPU cores and about 2.2 GiB memory.

The log line below is not fatal by itself:

```text
Skipping image OS validation for hb__<task>: docker inspect returned 1
```

If Docker Desktop still shows about `7.653GiB` memory in `docker stats`, the host Docker backend is capped near 8GB despite `memory_mb = 32768` in `task.toml`. Increase Docker Desktop or remote runner resources to actually get 32GB.

## Secrets And Profiles

The Codex adapters depend on an operator-provided Codex profile, auth file, and optional model catalog. Keep those files outside the repository and point the adapter to them with environment variables or Harbor agent/verifier kwargs.

Do not commit these files or paste process environments that might include secrets.

Default current settings:

```text
agent model: `AWS-GPT-5.5` unless overridden by `--model` / `MODEL_NAME`
profile: set by adapter kwarg, default `para`
reasoning effort: high
verifier audit model: `AWS-GPT-5.5` unless overridden by `--audit-model` / `AUDIT_MODEL_NAME`
```
