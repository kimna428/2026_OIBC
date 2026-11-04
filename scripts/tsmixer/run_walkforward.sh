#!/usr/bin/env bash
set -euo pipefail
[[ -d analysis/tsmixer/ensemble ]] || { echo 'Run from repository root' >&2; exit 2; }
"${OIBC_PYTHON:-python}" -m analysis.tsmixer.ensemble.walkforward "$@"
