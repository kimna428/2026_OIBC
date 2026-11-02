#!/usr/bin/env bash
# Read-only checks. No cache generation or training happens here.
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Server preflight requires Linux CUDA.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing.' >&2; exit 2; }
PYTHON_BIN="${OIBC_PYTHON:-python}"
: "${OIBC_DATA_ROOT:?Set the server data directory}"
nvidia-smi
bash scripts/tsmixer/check_environment.sh --device cuda
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m analysis.tsmixer check-data --data-root "$OIBC_DATA_ROOT" --checksums \
  --reference reports/tsmixer/data_inventory.json
CACHE_ROOT="${OIBC_TSMIXER_CACHE_ROOT:-runs/tsmixer/cache_station_v1}"
if [[ -f "$CACHE_ROOT/train/manifest.json" ]]; then
  "$PYTHON_BIN" -m analysis.tsmixer check-data --data-root "$OIBC_DATA_ROOT" --cache-root "$CACHE_ROOT"
else
  echo "Cache missing at $CACHE_ROOT. Run prepare-cache explicitly on this server; no files created."
fi
