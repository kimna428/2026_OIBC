#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/base.yaml && -f scripts/tsmixer/remote_control.py ]] || { echo 'Run from project root.' >&2; exit 2; }
exec "${OIBC_LOCAL_PYTHON:-python3}" scripts/tsmixer/remote_control.py status "$@"
