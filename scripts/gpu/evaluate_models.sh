#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OIBC_PYTHON:-python}"
SPLIT="${OIBC_EVAL_SPLIT:-dev}"
if [ "$#" -eq 0 ]; then
  echo 'Usage: scripts/gpu/evaluate_models.sh checkpoint1.pt [checkpoint2.pt ...]' >&2
  exit 2
fi
for checkpoint in "$@"; do
  "$PYTHON" -m analysis.deep_learning.evaluate --checkpoint "$checkpoint" --split "$SPLIT"
done
