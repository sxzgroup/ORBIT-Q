# ORBIT-Q

Open Research Benchmark for Integrated Tasks in Quantum Computing.

This repository contains a Harbor-based benchmark suite for 12 open quantum implementation tasks:

```bash
tasks/challenge-01
...
tasks/challenge-12
```

Read `AGENTS.md` first for the full bootstrap, image, verifier, and scoring workflow.

## Bootstrap

Build the selected framework's agent/verifier image. Image tags are framework-specific; do not overwrite one framework's image when adding another framework. The shared Dockerfile is `images/framework/Dockerfile`; framework Python packages live in `frameworks/<framework>/requirements.txt`.

```bash
FRAMEWORK=tensorcircuit bash scripts/build_challenge_quantum_image.sh
docker run --rm challenge-benchmark-quantum-tensorcircuit:py311 codex --version
```

Regenerate framework prompts and canonical tasks only when templates or upstream problem files change. This is benchmark maintenance, not part of switching frameworks for a run:

```bash
python3 scripts/generate_framework_prompts.py --framework tensorcircuit pennylane
python3 scripts/generate_tc_challenge_tasks.py
```

If the TensorCircuit challenge-suite source is not at `../tensorcircuit/examples/challenge_suite`, set:

```bash
export TC_CHALLENGE_SUITE_SOURCE=/path/to/tensorcircuit/examples/challenge_suite
```

For another framework, build a framework-specific image tag, for example:

```bash
FRAMEWORK=pennylane bash scripts/build_challenge_quantum_image.sh
```

## Run A Challenge

Use explicit model names for both solving and verifier audit:

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

`scripts/run_harbor_challenge.py` keeps `tasks/challenge-*` fixed. It passes the matching extra prompt to Harbor and uses `adapters.framework_docker:FrameworkDockerEnvironment` to select the framework image at container creation time. No task files are copied or patched.

Each task keeps a minimal `environment/` directory only because Harbor uses it to recognize local task directories. Do not put framework Dockerfiles or framework image names there.

Shortcut script:

```bash
FRAMEWORK=tensorcircuit MODEL_NAME=AWS-GPT-5.5 AUDIT_MODEL_NAME=AWS-GPT-5.5 \
  bash scripts/run_codex_para_smoke.sh
```
