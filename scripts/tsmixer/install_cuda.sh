#!/usr/bin/env bash
set -euo pipefail
[[ -f configs/tsmixer/base.yaml ]] || { echo 'Run from project root.' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Linux installation only.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing; install/check NVIDIA driver first.' >&2; exit 2; }
nvidia-smi
: "${TORCH_VERSION:?Set TORCH_VERSION to an official driver-compatible PyTorch version}"
: "${CUDA_WHEEL_INDEX:?Set CUDA_WHEEL_INDEX to an official https://download.pytorch.org/whl/cuXXX index}"
[[ "$CUDA_WHEEL_INDEX" =~ ^https://download\.pytorch\.org/whl/cu[0-9]+$ ]] || { echo 'Use an official CUDA wheel index.' >&2; exit 2; }
PYTHON_BIN="${OIBC_BOOTSTRAP_PYTHON:-python3.13}"
VENV_DIR="${OIBC_TSMIXER_VENV:-.venv-tsmixer}"
[[ ! -e "$VENV_DIR" ]] || { echo "Refusing to overwrite $VENV_DIR; use an empty venv path." >&2; exit 2; }
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "torch==$TORCH_VERSION" --index-url "$CUDA_WHEEL_INDEX"
"$VENV_DIR/bin/python" -m pip install -r analysis/tsmixer/requirements-common.txt
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" -m analysis.tsmixer environment --device cuda
