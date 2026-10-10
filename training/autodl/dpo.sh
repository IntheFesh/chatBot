#!/usr/bin/env bash
# Preference training on top of the SFT adapter (R-TRN-009).
#
#   dpo.sh <profile>
#
# Runs only when the package holds at least TWIN_DPO_MIN_PAIRS preference pairs
# (training.dpo_min_pairs, 200 by default); otherwise it says so and ends successfully without
# touching anything.  The SFT adapter is continued, the result is output/dpo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
start_logging dpo
use_data_disk

PAIRS_FILE="${TWIN_HOME}/data/dpo_train.jsonl"
MINIMUM="${TWIN_DPO_MIN_PAIRS:-200}"
pairs=0
if [ -f "${PAIRS_FILE}" ]; then
    pairs="$(wc -l <"${PAIRS_FILE}")"
fi
if [ "${pairs}" -lt "${MINIMUM}" ]; then
    log "DPO skipped: ${pairs} preference pairs, at least ${MINIMUM} are needed"
    exit 0
fi
[ -f "${TWIN_HOME}/output/sft/adapter_model.safetensors" ] || die "no SFT adapter; run train.sh first"

trap stop_gpu_sampler EXIT
start_gpu_sampler "${TWIN_HOME}/logs/gpu_memory_dpo.csv"
log "DPO on ${pairs} preference pairs"
llamafactory-cli train "${TWIN_HOME}/config/dpo.yaml"
stop_gpu_sampler
"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/summarize_train.py" \
    --output-dir "${TWIN_HOME}/output/dpo" \
    --gpu-log "${TWIN_HOME}/logs/gpu_memory_dpo.csv" \
    --gpu-name "$(probe_gpu 2>/dev/null | cut -d, -f1 || true)" \
    --out-dir "${TWIN_HOME}/artifacts/dpo"
