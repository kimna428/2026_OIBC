#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OIBC_PYTHON:-python}"
"$PYTHON" -m analysis.deep_learning.train --config "configs/deep_learning/transformer_base.yaml" "$@"
