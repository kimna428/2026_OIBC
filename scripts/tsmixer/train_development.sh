#!/usr/bin/env bash
# The explicit flag is required even through this wrapper. No automatic training.
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Full training requires Linux CUDA; Mac is smoke-only.' >&2; exit 2; }
if [[ "${1:-}" != --full-train ]]; then
  echo 'Usage: bash scripts/tsmixer/train_development.sh --full-train --output runs/tsmixer/RUN [options]' >&2
  exit 2
fi
PYTHON_BIN="${OIBC_PYTHON:-python}"
: "${OIBC_DATA_ROOT:?Set the server data directory}"
"$PYTHON_BIN" -m analysis.tsmixer train --device cuda "$@"
