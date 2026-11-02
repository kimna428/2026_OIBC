#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
PYTHON_BIN="${OIBC_PYTHON:-python}"
: "${OIBC_DATA_ROOT:?Set the server data directory}"
"$PYTHON_BIN" -m analysis.tsmixer prepare-cache --device cuda --checksums \
  --reference reports/tsmixer/data_inventory.json "$@"
