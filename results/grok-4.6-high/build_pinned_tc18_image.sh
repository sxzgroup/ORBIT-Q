#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_ROOT="$ROOT/results/grok-4.6-high"
IMAGE_TAG="challenge-benchmark-quantum-tensorcircuit:1.8.0.dev20260726-py311"
EXPECTED_VERSION="1.8.0.dev20260726"

if ! rg -qx "tensorcircuit-nightly==$EXPECTED_VERSION" \
  "$ROOT/frameworks/tensorcircuit/requirements.txt"; then
  printf 'TensorCircuit requirement is not pinned to %s\n' "$EXPECTED_VERSION" >&2
  exit 2
fi

FRAMEWORK=tensorcircuit IMAGE="$IMAGE_TAG" \
  bash "$ROOT/scripts/build_challenge_quantum_image.sh"

actual_version="$(docker run --rm "$IMAGE_TAG" python -c \
  'import tensorcircuit as tc; print(tc.__version__)')"
if [[ "$actual_version" != "$EXPECTED_VERSION" ]]; then
  printf 'TensorCircuit version mismatch: expected=%s actual=%s\n' \
    "$EXPECTED_VERSION" "$actual_version" >&2
  exit 2
fi

image_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')"
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'Invalid image ID: %s\n' "$image_id" >&2
  exit 2
fi
printf '%s\n' "$image_id" >"$RESULT_ROOT/tensorcircuit-1.8-image-id.txt"
printf 'Pinned %s (%s) with TensorCircuit %s\n' \
  "$IMAGE_TAG" "$image_id" "$actual_version"
