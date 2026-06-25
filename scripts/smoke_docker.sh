#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK="${FRAMEWORK:-tensorcircuit}"
FRAMEWORK_SLUG="$(
  printf '%s' "${FRAMEWORK}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+|-+$//g'
)"
IMAGE="${IMAGE:-challenge-benchmark-quantum-${FRAMEWORK_SLUG}:py311}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  FRAMEWORK="${FRAMEWORK}" IMAGE="${IMAGE}" bash "${ROOT}/scripts/build_challenge_quantum_image.sh"
fi

docker run --rm \
  -e "FRAMEWORK=${FRAMEWORK_SLUG}" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    codex --version
    python - <<'"'"'PY'"'"'
import importlib
import os

framework = os.environ["FRAMEWORK"]
required = {
    "tensorcircuit": ["tensorcircuit", "jax", "diffrax", "quimb", "omeco"],
    "pennylane": ["pennylane", "jax"],
}.get(framework, [])
for module_name in required:
    importlib.import_module(module_name)
print(f"image smoke ok: {framework}")
PY
  '
