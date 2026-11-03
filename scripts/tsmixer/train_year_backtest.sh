#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/year_backtest.yaml ]] || { echo 'Run from repository root' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Linux CUDA only; Mac runs unit tests only' >&2; exit 2; }
[[ "${1:-}" == --full-train ]] || { echo 'Explicit --full-train required' >&2; exit 2; }
"${OIBC_PYTHON:-python}" -m analysis.tsmixer.year_backtest "$@"
