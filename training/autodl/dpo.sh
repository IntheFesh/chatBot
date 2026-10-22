#!/usr/bin/env bash
# Preference training on top of the SFT model (R-TRN-009; D-650, D-651).
#
#   dpo.sh <profile>
#
# Runs only when the package holds at least TWIN_DPO_MIN_PAIRS preference pairs
# (training.dpo_min_pairs, 200 by default); otherwise it says so and ends successfully without
# touching anything.
#
# The reference model of the run is the SFT model, never the bare base (D-650):
#   layered   (the bf16 profiles) config/dpo.yaml has adapter_name_or_path = the SFT adapter and
#             create_new_adapter: LLaMA-Factory merges SFT into the base in memory and trains a NEW
#             adapter on it; without a ref_model the reference pass switches that adapter off, which
#             is the merged SFT model.  No second copy in VRAM.  output/dpo is the delta over SFT.
#   continued (the 4-bit profile) a quantised base cannot take a new adapter, so the SFT adapter is
#             continued (output/dpo replaces it) and config/dpo.yaml names the reference explicitly:
#             ref_model (the base), ref_model_adapters (SFT), ref_model_quantization_bit (4).  That
#             is a second model next to the policy, so a one-step trial on the longest pairs comes
#             first; on an out-of-memory failure it halves the per-device batch (the accumulation
#             doubles), and if even a batch of 1 does not fit it drops the three ref_model keys:
#             THIS profile only then falls back to the reference "base with the adapter switched off"
#             and says so loudly.  A failure that is not memory stops the script.
# artifacts/dpo/dpo_reference.json says how the reference was chosen (and why a fall-back happened);
# `twin train remote dpo` copies it into the run record.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"
start_logging dpo
use_data_disk

PAIRS_FILE="${TWIN_HOME}/data/dpo_train.jsonl"
CONFIG="${TWIN_HOME}/config/dpo.yaml"
RECORD="${TWIN_HOME}/artifacts/dpo/dpo_reference.json"
TRIAL_DATA="${TWIN_HOME}/trial_data_dpo"
TRIAL_PAIRS=64
MINIMUM="${TWIN_DPO_MIN_PAIRS:-200}"
REFERENCE_KEYS=(ref_model ref_model_adapters ref_model_quantization_bit)
pairs=0
if [ -f "${PAIRS_FILE}" ]; then
    pairs="$(wc -l <"${PAIRS_FILE}")"
fi
if [ "${pairs}" -lt "${MINIMUM}" ]; then
    log "DPO skipped: ${pairs} preference pairs, at least ${MINIMUM} are needed"
    exit 0
fi
[ -f "${TWIN_HOME}/output/sft/adapter_model.safetensors" ] || die "no SFT adapter; run train.sh first"
[ -f "${CONFIG}" ] || die "${CONFIG} is missing; build the package from a dataset that has preference pairs"

# The state of the reference model, written to RECORD.
DPO_REFERENCE=sft
DPO_FALLBACK=0
DPO_REASON=""
DPO_REF_BITS=""

# write_record BATCH ACCUM: artifacts/dpo/dpo_reference.json (plain words and numbers only).
write_record() {
    local bits="null" reason="null" fallback=false
    if [ -n "${DPO_REF_BITS}" ] && [ "${DPO_REFERENCE}" = "sft" ]; then bits="${DPO_REF_BITS}"; fi
    if [ "${DPO_FALLBACK}" = "1" ]; then fallback=true; fi
    if [ -n "${DPO_REASON}" ]; then reason="\"${DPO_REASON}\""; fi
    mkdir -p "${TWIN_HOME}/artifacts/dpo"
    printf '{\n  "mode": "%s",\n  "reference": "%s",\n  "reference_quantization_bit": %s,\n  "fallback": %s,\n  "fallback_reason": %s,\n  "per_device_batch": %s,\n  "accumulation": %s,\n  "pairs": %s\n}\n' \
        "${TWIN_DPO_MODE}" "${DPO_REFERENCE}" "${bits}" "${fallback}" "${reason}" "$1" "$2" "${pairs}" >"${RECORD}"
}

# drop_reference FILE: the explicit reference out of a config (the reference is then the base with
# the adapter switched off).  config/dpo.yaml itself only changes once the trial has succeeded.
drop_reference() {
    local key
    for key in "${REFERENCE_KEYS[@]}"; do
        yaml_drop "$1" "${key}"
    done
}

# run_trial: one step of the real config on the longest pairs (see the head of this file).
run_trial() {
    local trial="${TWIN_HOME}/config/dpo_trial.yaml" trial_log="${TWIN_HOME}/logs/dpo_trial.log"
    local batch accum next start_batch start_accum
    start_batch="$(yaml_get "${CONFIG}" per_device_train_batch_size)"
    start_accum="$(yaml_get "${CONFIG}" gradient_accumulation_steps)"
    batch="${start_batch}"
    accum="${start_accum}"
    "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/make_trial_dataset.py" \
        --data-dir "${TWIN_HOME}/data" --dataset "$(yaml_get "${CONFIG}" dataset)" \
        --out-dir "${TRIAL_DATA}" --count "${TRIAL_PAIRS}" \
        --trial-name twin_dpo_trial --trial-file dpo_trial.jsonl ||
        die "the data of the DPO trial step could not be prepared"
    while true; do
        cp "${CONFIG}" "${trial}"
        if [ "${DPO_REFERENCE}" = "base" ]; then drop_reference "${trial}"; fi
        yaml_set "${trial}" per_device_train_batch_size "${batch}"
        yaml_set "${trial}" gradient_accumulation_steps "${accum}"
        yaml_set "${trial}" max_steps 1
        yaml_set "${trial}" dataset twin_dpo_trial
        yaml_set "${trial}" dataset_dir "${TRIAL_DATA}"
        yaml_set "${trial}" output_dir "${TWIN_HOME}/output/dpo_trial"
        yaml_set "${trial}" save_strategy '"no"'
        yaml_set "${trial}" plot_loss false
        yaml_drop "${trial}" save_steps
        rm -rf "${TWIN_HOME}/output/dpo_trial"
        log "DPO trial step: reference ${DPO_REFERENCE}, per-device batch ${batch}, accumulation ${accum}"
        if llamafactory-cli train "${trial}" >"${trial_log}" 2>&1; then
            break
        fi
        if ! oom_in_log "${trial_log}"; then
            tail -n 40 "${trial_log}" >&2
            die "the DPO trial step failed for a reason other than memory (log: ${trial_log})"
        fi
        if next="$(next_smaller_batch "${batch}" "${accum}")"; then
            read -r batch accum <<<"${next}"
            log "out of memory; trying batch ${batch} with accumulation ${accum}"
            continue
        fi
        if [ "${DPO_REFERENCE}" = "sft" ]; then
            DPO_REFERENCE=base
            DPO_FALLBACK=1
            DPO_REASON="out of memory with the explicit 4-bit SFT reference even at a batch of 1"
            batch="${start_batch}"
            accum="${start_accum}"
            log "WARNING: ${DPO_REASON}; this profile falls back to the BASE model (adapter switched off) as the reference"
            continue
        fi
        die "the DPO trial step runs out of memory even with a batch of 1 and the base as the reference; choose a larger GPU"
    done
    rm -rf "${TWIN_HOME}/output/dpo_trial" "${trial}" "${TRIAL_DATA}"
    if [ "${DPO_REFERENCE}" = "base" ]; then drop_reference "${CONFIG}"; fi
    yaml_set "${CONFIG}" per_device_train_batch_size "${batch}"
    yaml_set "${CONFIG}" gradient_accumulation_steps "${accum}"
    log "DPO trial step succeeded; training will use batch ${batch} with accumulation ${accum} and the reference ${DPO_REFERENCE}"
}

cleanup() {
    stop_gpu_sampler
    rm -rf "${TRIAL_DATA}"
}
trap cleanup EXIT

if [ "${TWIN_DPO_MODE}" = "continued" ]; then
    DPO_REF_BITS="$(yaml_get "${CONFIG}" ref_model_quantization_bit)"
    if [ -z "$(yaml_get "${CONFIG}" ref_model)" ]; then
        # an earlier run of this script already dropped the reference: the fact stays
        DPO_REFERENCE=base
        DPO_FALLBACK=1
        DPO_REASON="an earlier run of dpo.sh dropped the explicit reference (out of memory)"
    fi
    run_trial
fi
write_record "$(yaml_get "${CONFIG}" per_device_train_batch_size)" "$(yaml_get "${CONFIG}" gradient_accumulation_steps)"

start_gpu_sampler "${TWIN_HOME}/logs/gpu_memory_dpo.csv"
log "DPO on ${pairs} preference pairs (${TWIN_DPO_MODE}, reference model: ${DPO_REFERENCE})"
llamafactory-cli train "${CONFIG}"
stop_gpu_sampler
"${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/summarize_train.py" \
    --output-dir "${TWIN_HOME}/output/dpo" \
    --gpu-log "${TWIN_HOME}/logs/gpu_memory_dpo.csv" \
    --gpu-name "$(probe_gpu 2>/dev/null | cut -d, -f1 || true)" \
    --dpo-reference "${RECORD}" \
    --out-dir "${TWIN_HOME}/artifacts/dpo"
