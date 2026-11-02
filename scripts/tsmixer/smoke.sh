#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
PYTHON_BIN="${OIBC_PYTHON:-python}"
"$PYTHON_BIN" -m analysis.tsmixer smoke "$@"
