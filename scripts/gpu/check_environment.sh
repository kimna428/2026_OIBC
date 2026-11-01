#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${OIBC_PYTHON:-python}"
"$PYTHON" -m analysis.deep_learning.environment --require-cuda --cuda-probe
"$PYTHON" -m analysis.deep_learning.prepare --check-only
