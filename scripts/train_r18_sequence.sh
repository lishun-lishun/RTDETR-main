#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Each experiment runs in a fresh Python process. This releases GPU memory
# between jobs and makes it obvious which model failed if the sequence stops.
MODELS=(
    original-r18
    anti-r18
    anti-wtconv
    anti-hifi
    anti-p2
    anti-mrfpn
)

echo "Training sequence: ${MODELS[*]}"
echo "Note: anti-mrfpn uses the documented SBA compatibility fallback because"
echo "the author repository does not publish MCAM/AGCAU."

for model in "${MODELS[@]}"; do
    bash "${SCRIPT_DIR}/train_r18_one.sh" "${model}" "$@"
done
