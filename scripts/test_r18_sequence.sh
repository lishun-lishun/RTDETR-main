#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
SPLIT="${SPLIT:-test}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" train_dut_anti_uav.py \
    --mode test \
    --models all \
    --split "${SPLIT}" \
    --device "${DEVICE}" \
    "$@"
