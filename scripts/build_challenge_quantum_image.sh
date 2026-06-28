#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK="${FRAMEWORK:-${1:-tensorcircuit}}"
FRAMEWORK_SLUG="$(
  printf '%s' "${FRAMEWORK}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+|-+$//g'
)"
IMAGE="${IMAGE:-challenge-benchmark-quantum-${FRAMEWORK_SLUG}:py311}"
CODEX_VERSION="${CODEX_VERSION:-latest}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-latest}"
DOCKERFILE="${DOCKERFILE:-${ROOT}/images/framework/Dockerfile}"
BUILD_CONTEXT="${BUILD_CONTEXT:-${ROOT}}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${ROOT}/frameworks/${FRAMEWORK_SLUG}/requirements.txt}"

if [[ "${REQUIREMENTS_FILE}" != /* ]]; then
  REQUIREMENTS_FILE="${ROOT}/${REQUIREMENTS_FILE}"
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Framework requirements not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

case "${REQUIREMENTS_FILE}" in
  "${ROOT}"/*)
    FRAMEWORK_REQUIREMENTS="${REQUIREMENTS_FILE#${ROOT}/}"
    ;;
  *)
    echo "Framework requirements must be inside the build context root: ${ROOT}" >&2
    exit 1
    ;;
esac

docker build \
  -f "${DOCKERFILE}" \
  --build-arg "CODEX_VERSION=${CODEX_VERSION}" \
  --build-arg "CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION}" \
  --build-arg "FRAMEWORK_REQUIREMENTS=${FRAMEWORK_REQUIREMENTS}" \
  -t "${IMAGE}" \
  "${BUILD_CONTEXT}"

docker image inspect "${IMAGE}" >/dev/null
docker run --rm "${IMAGE}" sh -lc 'codex --version && claude --version'
echo "Built ${IMAGE}"
