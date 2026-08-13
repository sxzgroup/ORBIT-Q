#!/usr/bin/env bash
# One-valid-outcome-per-task Grok 4.6/high Harbor campaign.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_ROOT="$ROOT/results/grok-4.6-high"
RUN_ROOT="${GROK46_RUN_ROOT:-/Users/qqy/Desktop/2026Project/ORBIT-Q/jobs/grok-4.6-high-solaudit-20260813-valid}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/Users/qqy/Desktop/2026Project/ORBIT-Q/jobs/gpt56terra-high-solaudit-20260731-valid}"
HARBOR="${HARBOR_BIN:-/Users/qqy/Desktop/2026Project/ORBIT-Q/.conda/harbor-py312/bin/harbor}"
IMAGE_TAG="challenge-benchmark-quantum-tensorcircuit:1.8.0.dev20260726-py311"
IMAGE_ID_FILE="$RESULT_ROOT/tensorcircuit-1.8-image-id.txt"
EXPECTED_TC_VERSION="1.8.0.dev20260726"
GROK_CONFIG="$RESULT_ROOT/grok-4.6.config.toml"
GROK_CATALOG="$RESULT_ROOT/grok-4.6-models.json"
GROK_PROXY="$ROOT/tools/xai_responses_proxy.py"
AUDIT_CONFIG="$RESULT_ROOT/audit-high.config.toml"
BASE_COMMIT="0201238ec2983907e2891f5319f5fff2d00844d5"
EXPECTED_SHA="19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895"
KEY_FILE="${GROK_KEY_FILE:-/Users/qqy/.codex-orbitq-grok/api-key}"
NETWORK_PROXY="${BENCHMARK_PROXY:-http://172.17.0.1:7892}"

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

if [[ ! -s "$KEY_FILE" ]]; then
  printf 'Grok credential file is missing or empty: %s\n' "$KEY_FILE" >&2
  exit 2
fi
IFS= read -r XAI_API_KEY <"$KEY_FILE"
export XAI_API_KEY

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

actual_sha="$(python3 - "$RUN_ROOT/task-copy-manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
digest = hashlib.sha256()
for record in manifest["tasks"]:
    digest.update(record["execution_copy_tree_sha256"].encode())
print(digest.hexdigest())
PY
)"
manifest_sha="$(python3 - "$RUN_ROOT/task-copy-manifest.json" <<'PY'
import json
import pathlib
import sys

print(
    json.loads(pathlib.Path(sys.argv[1]).read_text()).get(
        "aggregate_task_copy_sha256", ""
    )
)
PY
)"
if [[ -n "$manifest_sha" && "$manifest_sha" != "$EXPECTED_SHA" ]]; then
  printf 'Frozen task aggregate SHA mismatch: %s\n' "$manifest_sha" >&2
  exit 2
fi

printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Frozen task base: %s\n' "$BASE_COMMIT"
printf 'Task SHA-256: %s\n' "$EXPECTED_SHA"
printf 'TensorCircuit: %s\n' "$EXPECTED_TC_VERSION"
printf 'Docker image tag: %s\n' "$IMAGE_TAG"
printf 'Docker image ID: %s\n' "$IMAGE"
printf 'Solver: grok-4.6/high; audit: gpt-5.6-sol/high\n'
printf 'Manifest digest check: %s\n' "$actual_sha"

if [[ "${GROK_DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry-run validation complete; no Harbor job was started.\n'
  exit 0
fi

overall_rc=0
for raw in "${challenges[@]}"; do
  nn="$(printf '%02d' "$((10#$raw))")"
  task="$RUN_ROOT/challenge-$nn"
  valid=0
  attempt=1
  while [[ -d "$RUN_ROOT/jobs/challenge-$nn-tensorcircuit-grok-4.6-high-20260813-r$attempt" ]]; do
    attempt=$((attempt + 1))
  done

  for _retry in 1 2 3; do
    job="challenge-$nn-tensorcircuit-grok-4.6-high-20260813-r$attempt"
    log="$RUN_ROOT/host-challenge-$nn-r$attempt.log"
    printf '\n[%s] START challenge-%s attempt=%s\n' "$(date -u +%FT%TZ)" "$nn" "$attempt"

    if PYTHONPATH="$ROOT" "$HARBOR" run \
      -p "$task" \
      --extra-instruction-path "$ROOT/prompts/frameworks/tensorcircuit.md" \
      --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
      --environment-kwarg framework=tensorcircuit \
      --environment-kwarg "docker_image=$IMAGE" \
      --agent-import-path adapters.xai_codex_para:XAICodexPara \
      --agent-kwarg reasoning_effort=high \
      --agent-kwarg profile=grok46 \
      --agent-kwarg "profile_config_path=$GROK_CONFIG" \
      --agent-kwarg "model_catalog_path=$GROK_CATALOG" \
      --agent-kwarg "responses_proxy_path=$GROK_PROXY" \
      --agent-kwarg force_auth_json=false \
      --agent-env "HTTP_PROXY=$NETWORK_PROXY" \
      --agent-env "HTTPS_PROXY=$NETWORK_PROXY" \
      --agent-env NO_PROXY=localhost,127.0.0.1 \
      --verifier-import-path adapters.codex_para_verifier:CodexParaVerifier \
      --verifier-kwarg audit_model=gpt-5.6-sol \
      --verifier-kwarg force_auth_json=true \
      --verifier-kwarg "profile_config_path=$AUDIT_CONFIG" \
      --verifier-env REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit \
      --verifier-env "HTTP_PROXY=$NETWORK_PROXY" \
      --verifier-env "HTTPS_PROXY=$NETWORK_PROXY" \
      --verifier-env NO_PROXY=localhost,127.0.0.1 \
      -m grok-4.6 \
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
        'tls handshake|stream disconnected|error sending request|failed to connect|connection reset|name resolution|timed out before response|missing environment variable' \
        "$agent_log"; then
      infra_failure=1
      printf '[%s] INFRA challenge-%s attempt=%s: terminal solver transport failure\n' \
        "$(date -u +%FT%TZ)" "$nn" "$attempt"
    fi
    if [[ -n "$agent_log" ]] \
      && rg -q 'invalid type: floating point .* expected (i32|u64)' "$agent_log"; then
      infra_failure=1
      printf '[%s] INFRA challenge-%s attempt=%s: xAI/Codex integer-tool compatibility failure\n' \
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
