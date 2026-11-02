#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
[[ "$(uname -s)" == Darwin ]] || { echo 'Mac installation only.' >&2; exit 2; }
PYTHON_BIN="${OIBC_BOOTSTRAP_PYTHON:-python3.13}"
VENV_DIR="${OIBC_TSMIXER_VENV:-.venv-tsmixer}"
[[ ! -e "$VENV_DIR" ]] || { echo "Refusing to overwrite $VENV_DIR; activate it and inspect first." >&2; exit 2; }
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r analysis/tsmixer/requirements-mac.txt
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" -m analysis.tsmixer environment --device cpu
