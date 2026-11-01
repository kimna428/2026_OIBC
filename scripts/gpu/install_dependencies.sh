#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OIBC_PYTHON:-python}"
: "${PYTORCH_INDEX_URL:?Set PYTORCH_INDEX_URL to the driver-compatible official wheel index from pytorch.org/get-started/locally/}"
: "${OIBC_TORCH_VERSION:?Set OIBC_TORCH_VERSION to the version available on that official CUDA wheel index}"
case "$PYTORCH_INDEX_URL" in https://download.pytorch.org/whl/*) ;; *) echo 'Use the official PyTorch wheel index.' >&2; exit 2;; esac
"$PYTHON" -m pip install "torch==${OIBC_TORCH_VERSION}" --index-url "$PYTORCH_INDEX_URL"
"$PYTHON" -m pip install -r analysis/deep_learning/requirements.txt
"$PYTHON" -m pip freeze > "${OIBC_ENV_LOCK:-/tmp/oibc-gpu-environment.txt}"
