#!/usr/bin/env bash
# One-valid-outcome-per-task DeepSeek V4 Flash/high Harbor campaign.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_ROOT="$ROOT/results/deepseek-v4-flash-high"
RUN_ROOT="${DEEPSEEK_RUN_ROOT:-$ROOT/jobs/deepseek-v4-flash-high-solaudit-20260802-valid}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$ROOT/jobs/gpt56terra-high-solaudit-20260731-valid}"
HARBOR="$ROOT/.conda/harbor-py312/bin/harbor"
IMAGE="challenge-benchmark-quantum-tensorcircuit:py311"
TRACKED_DEEPSEEK_CONFIG="$RESULT_ROOT/deepseek-v4-flash.config.toml"
TRACKED_DEEPSEEK_CATALOG="$RESULT_ROOT/deepseek-v4-models.json"
DEEPSEEK_CODEX_HOME="${DEEPSEEK_CODEX_HOME:-$HOME/.codex-orbitq-deepseek}"
DEEPSEEK_CONFIG="$TRACKED_DEEPSEEK_CONFIG"
DEEPSEEK_CATALOG="$TRACKED_DEEPSEEK_CATALOG"
AUDIT_CONFIG="$RESULT_ROOT/audit-high.config.toml"
BASE_COMMIT="0201238ec2983907e2891f5319f5fff2d00844d5"
PROXY="${BENCHMARK_PROXY:-http://172.17.0.1:7892}"
USE_OFFICIAL_PROFILE=0
LOCAL_PROFILE_COPY=""

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ -s "$DEEPSEEK_CODEX_HOME/config.toml" && -s "$DEEPSEEK_CODEX_HOME/models.json" ]]; then
    USE_OFFICIAL_PROFILE=1
    DEEPSEEK_CATALOG="$DEEPSEEK_CODEX_HOME/models.json"
  else
    printf 'DeepSeek credentials are unavailable; refusing to start.\n' >&2
    printf 'Set DEEPSEEK_API_KEY or use the official setup script with CODEX_HOME=%s.\n' \
      "$DEEPSEEK_CODEX_HOME" >&2
    exit 2
  fi
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

if ((USE_OFFICIAL_PROFILE == 1)); then
  secret_dir="$RUN_ROOT/.secrets"
  LOCAL_PROFILE_COPY="$secret_dir/deepseek-v4-flash.config.toml"
  mkdir -p "$secret_dir"
  chmod 700 "$secret_dir"
  sed -E \
    's|^model_catalog_json[[:space:]]*=.*$|model_catalog_json = "~/.codex/models.json"|' \
    "$DEEPSEEK_CODEX_HOME/config.toml" >"$LOCAL_PROFILE_COPY"
  chmod 600 "$LOCAL_PROFILE_COPY"
  DEEPSEEK_CONFIG="$LOCAL_PROFILE_COPY"
  trap 'rm -f "$LOCAL_PROFILE_COPY"' EXIT
fi

printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Frozen task base: %s\n' "$BASE_COMMIT"
printf 'Task SHA-256: 19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895\n'
printf 'Solver: deepseek-v4-flash/high; audit: gpt-5.6-sol/high\n'

if [[ "${DEEPSEEK_DRY_RUN:-0}" == "1" ]]; then
  printf 'Dry-run validation complete; no Harbor job was started.\n'
  exit 0
fi

overall_rc=0
for raw in "${challenges[@]}"; do
  nn="$(printf '%02d' "$((10#$raw))")"
  task="$RUN_ROOT/challenge-$nn"
  valid=0
  attempt=1
  while [[ -d "$RUN_ROOT/jobs/challenge-$nn-tensorcircuit-deepseek-v4-flash-high-20260802-r$attempt" ]]; do
    attempt=$((attempt + 1))
  done

  for _retry in 1 2 3; do
    job="challenge-$nn-tensorcircuit-deepseek-v4-flash-high-20260802-r$attempt"
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
      --agent-kwarg profile=deepseek-v4-flash \
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
      -m deepseek-v4-flash \
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
