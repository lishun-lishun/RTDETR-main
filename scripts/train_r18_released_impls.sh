#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This sequence omits anti-mrfpn because its YAML references MCAM/AGCAU, whose
# implementation is absent from every revision of the author repository.
MODELS=(
    original-r18
    anti-r18
    anti-wtconv
    anti-hifi
    anti-p2
)

echo "Training released/runnable implementations: ${MODELS[*]}"
for model in "${MODELS[@]}"; do
    bash "${SCRIPT_DIR}/train_r18_one.sh" "${model}" "$@"
done
