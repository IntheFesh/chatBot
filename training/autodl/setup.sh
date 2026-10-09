#!/usr/bin/env bash
# Prepare an AutoDL instance for the style model training (R-TRN-009).
#
#   setup.sh <profile> [install|verify]
#
# install (default)  checks the GPU, the data disk and the memory, makes sure PyTorch is a CUDA
#                    12.8 build for Blackwell, installs the pinned training packages and
#                    downloads the base model from ModelScope.  Run it again after an
#                    interruption: every step skips what is already done.
# verify             runs after decrypt.sh: the template consistency check (R-TRN-011) and a
#                    one-step trial run whose batch size is lowered until it no longer runs out
#                    of memory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

PROFILE_NAME="${1:-}"
MODE="${2:-install}"

install_packages() {
    log "installing the pinned packages (LLaMA-Factory ${TWIN_LLAMAFACTORY})"
    "${TWIN_PYTHON}" -m pip install \
        "llamafactory==${TWIN_LLAMAFACTORY}" \
        "transformers==${TWIN_TRANSFORMERS}" \
        "peft==${TWIN_PEFT}" \
        "trl==${TWIN_TRL}" \
        "accelerate==${TWIN_ACCELERATE}" \
        "datasets==${TWIN_DATASETS}" \
        "bitsandbytes==${TWIN_BITSANDBYTES}" \
        "modelscope==${TWIN_MODELSCOPE}" \
        "zstandard==${TWIN_ZSTANDARD}" \
        "cryptography>=${TWIN_CRYPTOGRAPHY_MIN}" \
        matplotlib
}

ensure_zstd() {
    if command -v zstd >/dev/null 2>&1; then
        return 0
    fi
    log "zstd is missing; trying apt, then the zstandard module"
    if command -v apt-get >/dev/null 2>&1 && apt-get install -y zstd >/dev/null 2>&1; then
        return 0
    fi
    "${TWIN_PYTHON}" -c 'import zstandard' 2>/dev/null ||
        "${TWIN_PYTHON}" -m pip install "zstandard==${TWIN_ZSTANDARD}"
}

ensure_torch() {
    local verdict
    verdict="$(run_torch_check)"
    log "PyTorch verdict: ${verdict}"
    [ "${verdict}" = "ok" ] && return 0
    log "installing PyTorch ${TWIN_TORCH} (CUDA ${TWIN_CUDA_SERIES}) from ${TWIN_TORCH_INDEX}"
    "${TWIN_PYTHON}" -m pip install \
        "torch==${TWIN_TORCH}" "torchvision==${TWIN_TORCHVISION}" "torchaudio==${TWIN_TORCHAUDIO}" \
        --index-url "${TWIN_TORCH_INDEX}"
    verdict="$(run_torch_check)"
    [ "${verdict}" = "ok" ] && return 0
    die "PyTorch is still not usable on this GPU: ${verdict}"
}

# base_model_complete DIR: the download is finished when config.json exists and the files add up
# to (nearly) the size of the bf16 weights.
base_model_complete() {
    local target="$1" kb minimum_gb
    [ -f "${target}/config.json" ] || return 1
    kb="$(du -sk "${target}" | awk '{ print $1 }')"
    minimum_gb=$(awk -v gb="${TWIN_BASE_GB}" 'BEGIN { printf "%d", gb * 0.95 }')
    [ "$((kb / 1048576))" -ge "${minimum_gb}" ]
}

download_base_model() {
    local target="${TWIN_MODELS}/${TWIN_BASE_MODEL}"
    mkdir -p "${target}"
    if base_model_complete "${target}"; then
        log "${TWIN_BASE_MODEL} is already on the data disk"
        return 0
    fi
    free_space_gate "download of the base model" "$((TWIN_BASE_GB_CEIL + 5))"
    log "downloading ${TWIN_BASE_MODEL} from ModelScope to ${target}"
    USE_MODELSCOPE_HUB=1 "${TWIN_PYTHON}" -c \
        'import sys; from modelscope import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' \
        "${TWIN_BASE_MODEL}" "${target}"
    base_model_complete "${target}" || die "the download of ${TWIN_BASE_MODEL} is incomplete; run setup again to resume it"
}

run_install() {
    start_logging setup
    log "profile ${TWIN_PROFILE}: ${TWIN_BASE_MODEL}, ${TWIN_METHOD}"
    mkdir -p "${TWIN_HOME}" "${TWIN_MODELS}"
    use_data_disk
    run_resource_checks || exit 1
    ensure_zstd
    ensure_torch
    with_turbo install_packages
    download_base_model
    turbo_off
    "${TWIN_PYTHON}" -m pip check || log "pip check reported conflicts (see above); training may still work"
    touch "${TWIN_HOME}/.setup_done"
    log "setup finished; next: decrypt.sh, then setup.sh ${TWIN_PROFILE} verify"
}

# The template check is the entry point round 13b adds to the training package:
#   python -m twin.training.parity_check --model-dir DIR --cases FILE
# It exits 0 when LLaMA-Factory's encoding of the sample cases equals the prompts the style
# model's prompt builder renders.  When the module is not in the package this stops with an error
# instead of skipping the check.
run_parity_check() {
    local pylib="${TWIN_HOME}/pylib"
    if ! PYTHONPATH="${pylib}" "${TWIN_PYTHON}" -c \
        'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("twin.training.parity_check") else 1)'; then
        die "the template consistency check (twin.training.parity_check) is not in this package; build the package with a twin version that includes it"
    fi
    PYTHONPATH="${pylib}" "${TWIN_PYTHON}" -m twin.training.parity_check \
        --model-dir "${TWIN_MODELS}/${TWIN_BASE_MODEL}" \
        --cases "${TWIN_HOME}/data/parity_cases.jsonl"
    log "template consistency check passed"
}

# One training step with the configured batch layout; on a CUDA out-of-memory failure the
# per-device batch is halved (and the accumulation doubled) until the step runs.
run_trial_step() {
    local config="${TWIN_HOME}/config/sft.yaml" trial="${TWIN_HOME}/config/trial.yaml"
    local batch accum next trial_log="${TWIN_HOME}/logs/trial.log"
    batch="$(yaml_get "${config}" per_device_train_batch_size)"
    accum="$(yaml_get "${config}" gradient_accumulation_steps)"
    while true; do
        cp "${config}" "${trial}"
        yaml_set "${trial}" per_device_train_batch_size "${batch}"
        yaml_set "${trial}" gradient_accumulation_steps "${accum}"
        yaml_set "${trial}" per_device_eval_batch_size "${batch}"
        yaml_set "${trial}" max_steps 1
        yaml_set "${trial}" max_samples 64
        yaml_set "${trial}" output_dir "${TWIN_HOME}/output/trial"
        yaml_set "${trial}" eval_strategy '"no"'
        yaml_set "${trial}" save_strategy '"no"'
        yaml_set "${trial}" load_best_model_at_end false
        yaml_drop "${trial}" early_stopping_steps
        yaml_drop "${trial}" eval_steps
        log "trial step with per-device batch ${batch}, accumulation ${accum}"
        if llamafactory-cli train "${trial}" >"${trial_log}" 2>&1; then
            break
        fi
        if ! oom_in_log "${trial_log}"; then
            tail -n 40 "${trial_log}" >&2
            die "the trial step failed for a reason other than memory (log: ${trial_log})"
        fi
        if ! next="$(next_smaller_batch "${batch}" "${accum}")"; then
            die "the trial step runs out of memory even with a batch of 1; choose a larger GPU"
        fi
        read -r batch accum <<<"${next}"
        log "out of memory; trying batch ${batch} with accumulation ${accum}"
    done
    for key in per_device_train_batch_size per_device_eval_batch_size; do
        yaml_set "${config}" "${key}" "${batch}"
    done
    yaml_set "${config}" gradient_accumulation_steps "${accum}"
    rm -rf "${TWIN_HOME}/output/trial" "${trial}"
    log "trial step succeeded; training will use batch ${batch} with accumulation ${accum}"
}

run_verify() {
    start_logging setup_verify
    use_data_disk
    [ -f "${TWIN_HOME}/data/dataset_info.json" ] || die "no dataset yet; run decrypt.sh first"
    [ -f "${TWIN_HOME}/.setup_done" ] || die "run setup.sh ${TWIN_PROFILE} install first"
    run_parity_check
    run_trial_step
    log "verification finished"
}

main() {
    load_env_files "${SCRIPT_DIR}"
    require_profile "${PROFILE_NAME}"
    case "${MODE}" in
        install) run_install ;;
        verify) run_verify ;;
        *) die "unknown mode '${MODE}' (install or verify)" ;;
    esac
}

main
