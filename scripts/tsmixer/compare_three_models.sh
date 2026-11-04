#!/usr/bin/env bash
set -euo pipefail
[[ -d analysis/tsmixer/comparison ]] || { echo 'Run from repository root' >&2; exit 2; }
"${OIBC_PYTHON:-python}" -m analysis.tsmixer.comparison.three "$@"
