#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Sequential on one GPU. Each CLI also takes a nonblocking GPU process lock.
for model in cnn transformer cross_site; do
  "scripts/gpu/train_${model}.sh" "$@"
done
