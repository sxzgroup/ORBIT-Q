#!/usr/bin/env bash
# One-valid-outcome-per-task DeepSeek V4 Pro/high Harbor campaign.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_ROOT="$ROOT/results/deepseek-v4-pro-high"
HOST_REPO="${ORBITQ_HOST_REPO:-/Users/qqy/Desktop/2026Project/ORBIT-Q}"
RUN_ROOT="${DEEPSEEK_RUN_ROOT:-$HOST_REPO/jobs/deepseek-v4-pro-high-tc18-solaudit-20260813-valid}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$HOST_REPO/jobs/gpt56terra-high-solaudit-20260731-valid}"
HARBOR="${HARBOR_BIN:-$HOST_REPO/.conda/harbor-py312/bin/harbor}"
IMAGE_TAG="challenge-benchmark-quantum-tensorcircuit:1.8.0.dev20260726-py311"
IMAGE_ID_FILE="$RESULT_ROOT/tensorcircuit-1.8-image-id.txt"
EXPECTED_TC_VERSION="1.8.0.dev20260726"
TRACKED_DEEPSEEK_CONFIG="$RESULT_ROOT/deepseek-v4-pro.config.toml"
TRACKED_DEEPSEEK_CATALOG="$RESULT_ROOT/deepseek-v4-models.json"
DEEPSEEK_CODEX_HOME="${DEEPSEEK_CODEX_HOME:-$HOME/.codex-orbitq-deepseek}"
DEEPSEEK_KEY_FILE="${DEEPSEEK_KEY_FILE:-$DEEPSEEK_CODEX_HOME/api_key}"
DEEPSEEK_CONFIG="$TRACKED_DEEPSEEK_CONFIG"
DEEPSEEK_CATALOG="$TRACKED_DEEPSEEK_CATALOG"
AUDIT_CONFIG="$RESULT_ROOT/audit-high.config.toml"
BASE_COMMIT="0201238ec2983907e2891f5319f5fff2d00844d5"
PROXY="${BENCHMARK_PROXY:-http://172.17.0.1:7892}"
EXPECTED_TASK_SHA="19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895"
SECRET_PROFILE=""

if [[ ! -s "$IMAGE_ID_FILE" ]]; then
  printf 'Pinned TensorCircuit 1.8 image ID is unavailable: %s\n' "$IMAGE_ID_FILE" >&2
  exit 2
fi
IMAGE="$(tr -d '[:space:]' <"$IMAGE_ID_FILE")"
if [[ ! "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'Invalid pinned image ID: %s\n' "$IMAGE" >&2
  exit 2
fi
actual_image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null || true)"
if [[ "$actual_image_id" != "$IMAGE" ]]; then
  printf 'TensorCircuit image ID mismatch: tag=%s expected=%s actual=%s\n' \
    "$IMAGE_TAG" "$IMAGE" "$actual_image_id" >&2
  exit 2
fi
actual_tc_version="$(docker run --rm "$IMAGE" python -c \
  'import tensorcircuit as tc; print(tc.__version__)')"
if [[ "$actual_tc_version" != "$EXPECTED_TC_VERSION" ]]; then
  printf 'TensorCircuit version mismatch: expected=%s actual=%s\n' \
    "$EXPECTED_TC_VERSION" "$actual_tc_version" >&2
  exit 2
fi

if [[ ! -x "$HARBOR" ]]; then
  printf 'Harbor executable is unavailable: %s\n' "$HARBOR" >&2
  exit 2
fi

if (($#)); then
  challenges=("$@")
else
  challenges=(01 02 03 04 05 06 07 08 09 10 11 12)
fi

if [[ ! -f "$RUN_ROOT/task-copy-manifest.json" ]]; then
  mkdir -p "$RUN_ROOT"
  python3 "$RESULT_ROOT/tools/prepare_tasks.py" \
    --run-root "$RUN_ROOT" \
    --source-run-root "$SOURCE_RUN_ROOT" \
    --base-commit "$BASE_COMMIT"
fi
mkdir -p "$RUN_ROOT/jobs"

manifest_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["aggregate_task_copy_sha256"])' "$RUN_ROOT/task-copy-manifest.json")"
if [[ "$manifest_sha" != "$EXPECTED_TASK_SHA" ]]; then
  printf 'Frozen task aggregate SHA-256 mismatch: %s\n' "$manifest_sha" >&2
  exit 2
fi

printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Frozen task base: %s\n' "$BASE_COMMIT"
printf 'Task SHA-256: %s\n' "$EXPECTED_TASK_SHA"
printf 'TensorCircuit: %s\n' "$EXPECTED_TC_VERSION"
printf 'Docker image tag: %s\n' "$IMAGE_TAG"
printf 'Docker image ID: %s\n' "$IMAGE"
printf 'Solver: deepseek-v4-pro/high; audit: gpt-5.6-sol/high\n'

if [[ "${DEEPSEEK_DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry-run validation complete; no Harbor job was started.\n'
  exit 0
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" && -s "$DEEPSEEK_KEY_FILE" ]]; then
  IFS= read -r DEEPSEEK_API_KEY <"$DEEPSEEK_KEY_FILE"
  export DEEPSEEK_API_KEY
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf 'DeepSeek credentials are unavailable; refusing to start.\n' >&2
  printf 'Export DEEPSEEK_API_KEY or store it in the private file %s.\n' \
    "$DEEPSEEK_KEY_FILE" >&2
  exit 2
fi

# Codex CLI 0.146 expects custom Responses providers to carry the bearer token
# in the provider table. Materialize that field only in a private, ephemeral
# profile; the tracked profile and archived job metadata remain secret-free.
secret_dir="$RUN_ROOT/.secrets"
SECRET_PROFILE="$secret_dir/deepseek-v4-pro.config.toml"
mkdir -p "$secret_dir"
chmod 700 "$secret_dir"
sed -E '/^env_key[[:space:]]*=/d' "$TRACKED_DEEPSEEK_CONFIG" >"$SECRET_PROFILE"
printf 'experimental_bearer_token = "%s"\n' "$DEEPSEEK_API_KEY" >>"$SECRET_PROFILE"
chmod 600 "$SECRET_PROFILE"
DEEPSEEK_CONFIG="$SECRET_PROFILE"
unset DEEPSEEK_API_KEY
trap 'rm -f "$SECRET_PROFILE"' EXIT

overall_rc=0
for raw in "${challenges[@]}"; do
  nn="$(printf '%02d' "$((10#$raw))")"
  task="$RUN_ROOT/challenge-$nn"
  valid=0
  attempt=1
  while [[ -d "$RUN_ROOT/jobs/challenge-$nn-tensorcircuit-deepseek-v4-pro-high-tc18-20260813-r$attempt" ]]; do
    attempt=$((attempt + 1))
  done

  for _retry in 1 2 3; do
    job="challenge-$nn-tensorcircuit-deepseek-v4-pro-high-tc18-20260813-r$attempt"
    log="$RUN_ROOT/host-challenge-$nn-r$attempt.log"
    printf '\n[%s] START challenge-%s attempt=%s\n' "$(date -u +%FT%TZ)" "$nn" "$attempt"

    if PYTHONPATH="$ROOT" "$HARBOR" run \
      -p "$task" \
      --extra-instruction-path "$ROOT/prompts/frameworks/tensorcircuit.md" \
      --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
      --environment-kwarg framework=tensorcircuit \
      --environment-kwarg "docker_image=$IMAGE" \
      --agent-import-path adapters.codex_para:CodexPara \
      --agent-kwarg reasoning_effort=high \
      --agent-kwarg profile=deepseek-v4-pro \
      --agent-kwarg "profile_config_path=$DEEPSEEK_CONFIG" \
      --agent-kwarg "model_catalog_path=$DEEPSEEK_CATALOG" \
      --agent-kwarg force_auth_json=false \
      --agent-env "HTTP_PROXY=$PROXY" \
      --agent-env "HTTPS_PROXY=$PROXY" \
      --agent-env NO_PROXY=localhost,127.0.0.1 \
      --verifier-import-path adapters.codex_para_verifier:CodexParaVerifier \
      --verifier-kwarg audit_model=gpt-5.6-sol \
      --verifier-kwarg force_auth_json=true \
      --verifier-kwarg "profile_config_path=$AUDIT_CONFIG" \
      --verifier-env REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit \
      --verifier-env "HTTP_PROXY=$PROXY" \
      --verifier-env "HTTPS_PROXY=$PROXY" \
      --verifier-env NO_PROXY=localhost,127.0.0.1 \
      -m deepseek-v4-pro \
      -n 1 \
      -o "$RUN_ROOT/jobs" \
      --job-name "$job" \
      --yes 2>&1 | tee "$log"
    then
      rc=0
    else
      rc=$?
    fi
    printf '[%s] END challenge-%s attempt=%s rc=%s\n' \
      "$(date -u +%FT%TZ)" "$nn" "$attempt" "$rc"

    reward="$(find "$RUN_ROOT/jobs/$job" -path '*/verifier/reward.json' -type f -print -quit 2>/dev/null)"
    agent_log="$(find "$RUN_ROOT/jobs/$job" -path '*/agent/codex.txt' -type f -print -quit 2>/dev/null)"
    infra_failure=0
    if [[ -n "$agent_log" ]] \
      && rg -q '"type":"turn.failed"' "$agent_log" \
      && rg -qi \
        'tls handshake|stream disconnected|error sending request|failed to connect|connection reset|name resolution|timed out before response' \
        "$agent_log"; then
      infra_failure=1
      printf '[%s] INFRA challenge-%s attempt=%s: terminal solver transport failure\n' \
        "$(date -u +%FT%TZ)" "$nn" "$attempt"
    fi
    if [[ -n "$reward" && -s "$reward" && "$infra_failure" -eq 0 ]]; then
      valid=1
      break
    fi
    printf '[%s] INVALID challenge-%s attempt=%s: valid outcome missing; retrying\n' \
      "$(date -u +%FT%TZ)" "$nn" "$attempt"
    attempt=$((attempt + 1))
  done

  if ((valid == 0)); then
    printf '[%s] ABORT challenge-%s: no valid outcome after 3 attempts\n' \
      "$(date -u +%FT%TZ)" "$nn"
    overall_rc=1
    break
  fi
done

exit "$overall_rc"
