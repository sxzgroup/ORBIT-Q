#!/usr/bin/env bash
# Official verifier-only stamp for one Fable 5 candidate solution, on the
# maintainer's Mac (colima + local Harbor + ClashX proxy relay on :7891).
#
# Usage: bash results/fable5/tools/verify_challenge_mac.sh 01
#
# Runs Harbor's oracle path (functional evaluator + static policy + Codex LLM
# audit via ~/.codex/auth.json) against the candidate solution stored in
# results/fable5/challenge-<NN>/, then copies reward.json back next to it.
#
# The audit model is NOT hardcoded: override with AUDIT_MODEL_NAME (preferred)
# or AUDIT_MODEL when the ChatGPT-auth catalog does not serve the default
# slug, e.g.:
#   AUDIT_MODEL_NAME=gpt-5.6-sol bash results/fable5/tools/verify_challenge_mac.sh 01
set -euo pipefail

NN="${1:?usage: verify_challenge_mac.sh <challenge number, e.g. 01>}"
AUDIT_MODEL="${AUDIT_MODEL_NAME:-${AUDIT_MODEL:-gpt-5.6-sol}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SOL_DIR="results/fable5/challenge-$NN"
SOL_FILE="$(ls "$SOL_DIR"/solution_*.py | head -1)"
[ -f "$SOL_FILE" ] || { echo "No solution_*.py in $SOL_DIR" >&2; exit 1; }

WORK=".scratch/tmp-tasks/challenge-$NN-candidate-verify"
rm -rf "$WORK"
mkdir -p .scratch/tmp-tasks
cp -R "tasks/challenge-$NN" "$WORK"
rm -rf "$WORK/solution/__pycache__" "$WORK/tests/__pycache__"
cp "$SOL_FILE" "$WORK/solution/$(basename "$SOL_FILE")"

# Fit declared container resources to this machine (reservation only; does not
# change scoring). Canonical tasks/challenge-* stay untouched.
python3 - "$WORK/task.toml" <<'EOF'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
t = p.read_text()
t = re.sub(r"(?m)^cpus = \d+$", "cpus = 6", t)
t = re.sub(r"(?m)^memory_mb = \d+$", "memory_mb = 10240", t)
p.write_text(t)
EOF

JOB="challenge-$NN-fable5-stamp-$(date +%m%d%H%M%S)"
echo "Audit model: $AUDIT_MODEL"
PYTHONPATH="$PWD" ./.conda/harbor-py312/bin/harbor run \
  -p "$PWD/$WORK" \
  --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
  --environment-kwarg framework=tensorcircuit \
  --environment-kwarg docker_image=challenge-benchmark-quantum-tensorcircuit:py311 \
  --verifier-import-path adapters.codex_para_verifier:CodexParaVerifier \
  --verifier-kwarg "audit_model=$AUDIT_MODEL" \
  --verifier-kwarg force_auth_json=true \
  --verifier-env REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit \
  --verifier-env HTTP_PROXY=http://192.168.5.2:7891 \
  --verifier-env HTTPS_PROXY=http://192.168.5.2:7891 \
  --verifier-env NO_PROXY=localhost,127.0.0.1 \
  -n 1 -o "$PWD/jobs" --job-name "$JOB" --yes

REWARD="$(find "jobs/$JOB" -name reward.json | head -1)"
[ -f "$REWARD" ] || { echo "reward.json not found under jobs/$JOB" >&2; exit 1; }
cp "$REWARD" "$SOL_DIR/reward.json"
STDOUT_LOG="$(find "jobs/$JOB" -name 'functional-stdout*' | head -1 || true)"
[ -n "${STDOUT_LOG:-}" ] && cp "$STDOUT_LOG" "$SOL_DIR/functional-stdout-official.txt"
printf '{"audit_model": "%s", "job": "%s", "stamped_at_utc": "%s"}\n' \
  "$AUDIT_MODEL" "$JOB" "$(date -u +%FT%TZ)" > "$SOL_DIR/stamp-info.json"

echo
echo "=== Official reward for challenge-$NN ==="
cat "$SOL_DIR/reward.json"
