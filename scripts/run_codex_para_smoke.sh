#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_name="${JOB_NAME:-challenge-01-$(date +%Y%m%d-%H%M%S)}"
framework="${FRAMEWORK:-tensorcircuit}"
challenge_id="${CHALLENGE_ID:-01}"

cmd=(
  python3 "${repo_root}/scripts/run_harbor_challenge.py"
  --challenge "${challenge_id}"
  --framework "${framework}"
  --job-name "${job_name}"
  --jobs-dir "${repo_root}/jobs"
  -n 1
  --yes
)

if [[ -n "${MODEL_NAME:-${MODEL:-}}" ]]; then
  cmd+=(--model "${MODEL_NAME:-${MODEL:-}}")
fi
if [[ -n "${AUDIT_MODEL_NAME:-${AUDIT_MODEL:-}}" ]]; then
  cmd+=(--audit-model "${AUDIT_MODEL_NAME:-${AUDIT_MODEL:-}}")
fi

exec "${cmd[@]}"
