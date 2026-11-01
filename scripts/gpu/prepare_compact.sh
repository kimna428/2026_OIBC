#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OIBC_STORAGE_ROOT:?Set your private quota directory containing data/cache/results/venv}"
: "${OIBC_DATA_ROOT:?Set the existing competition data directory}"
: "${OIBC_CACHE_ROOT:?Set a NEW cache directory below your private quota directory}"
export OIBC_QUOTA_GIB="${OIBC_QUOTA_GIB:-12}"
export OIBC_CACHE_MAX_GIB="${OIBC_CACHE_MAX_GIB:-2}"
export OIBC_RESULTS_MAX_GIB="${OIBC_RESULTS_MAX_GIB:-4}"
export OIBC_MIN_FREE_GIB="${OIBC_MIN_FREE_GIB:-0.5}"
"${OIBC_PYTHON:-python}" -m analysis.deep_learning.storage
"${OIBC_PYTHON:-python}" -m analysis.deep_learning.prepare --layout compact --partition both --dry-run
"${OIBC_PYTHON:-python}" -m analysis.deep_learning.prepare --layout compact --partition both
"${OIBC_PYTHON:-python}" -m analysis.deep_learning.check_data --scan --verify-originals
