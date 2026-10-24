#!/usr/bin/env bash
# Supervised fine-tuning of the style model (R-TRN-009).
#
#   train.sh <profile> [--again]
#
# Runs LLaMA-Factory with config/sft.yaml.  When output/sft already holds a checkpoint the run
# continues from the last one (LLaMA-Factory resumes by itself), so running the script again
# after an interruption is safe (SFT always saves checkpoints: save_steps is eval_steps).
# --again (`twin train remote train --again`) trains anew: output/sft goes, and with it what was
# built on that adapter - output/dpo and the stack the evaluation kept (D-813).  At the end it prints the best checkpoint and the best
# validation loss, and leaves the loss curves and the metrics in artifacts/train/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
AGAIN=0
for argument in "${@:2}"; do
    case "${argument}" in
        --again) AGAIN=1 ;;
        *) die "unknown argument '${argument}'" ;;
    esac
done
start_logging train
use_data_disk

CONFIG="${TWIN_HOME}/config/sft.yaml"
OUTPUT="${TWIN_HOME}/output/sft"
CURVES="${TWIN_HOME}/artifacts/train"

[ -f "${CONFIG}" ] || die "${CONFIG} is missing; run decrypt.sh first"
[ -f "${TWIN_HOME}/.setup_done" ] || die "run setup.sh ${TWIN_PROFILE} install first"
mkdir -p "${CURVES}"

if [ "${AGAIN}" = "1" ]; then
    log "--again: the earlier training (output/sft) and what was built on it (output/dpo, the stack the evaluation kept) are deleted; training starts from the base model"
    rm -rf "${OUTPUT}" "${TWIN_HOME}/output/dpo"
    rm -f "${TWIN_HOME}/state/stack_choice"
fi

trap stop_gpu_sampler EXIT
start_gpu_sampler "${TWIN_HOME}/logs/gpu_memory.csv"

if [ -d "${OUTPUT}" ] && compgen -G "${OUTPUT}/checkpoint-*" >/dev/null; then
    log "resuming from the last checkpoint in ${OUTPUT}"
fi
log "training ${TWIN_PROFILE} with ${CONFIG}"
llamafactory-cli train "${CONFIG}"

stop_gpu_sampler
"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/summarize_train.py" \
    --output-dir "${OUTPUT}" \
    --gpu-log "${TWIN_HOME}/logs/gpu_memory.csv" \
    --gpu-name "$(probe_gpu 2>/dev/null | cut -d, -f1 || true)" \
    --out-dir "${CURVES}"
