#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_name="${JOB_NAME:-challenge-01-codex-para-$(date +%Y%m%d-%H%M%S)}"
model="${MODEL_NAME:-${MODEL:-AWS-GPT-5.5}}"
audit_model="${AUDIT_MODEL_NAME:-${AUDIT_MODEL:-${model}}}"
framework="${FRAMEWORK:-tensorcircuit}"
challenge_id="${CHALLENGE_ID:-01}"

exec python3 "${repo_root}/scripts/run_harbor_challenge.py" \
  --challenge "${challenge_id}" \
  --framework "${framework}" \
  --model "${model}" \
  --audit-model "${audit_model}" \
  --job-name "${job_name}" \
  --jobs-dir "${repo_root}/jobs" \
  -n 1 \
  --yes
