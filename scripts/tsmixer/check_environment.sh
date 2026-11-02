#!/usr/bin/env bash
set -euo pipefail
if [[ ! -f configs/tsmixer/base.yaml || ! -d analysis/tsmixer ]]; then
  echo 'Run this script from the project root.' >&2; exit 2
fi
PYTHON_BIN="${OIBC_PYTHON:-python}"
echo 'Git commit / branch / work tree:'
git log -1 --oneline
git branch --show-current
git status --short
"$PYTHON_BIN" -m analysis.tsmixer environment "$@"
