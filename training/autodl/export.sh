#!/usr/bin/env bash
# Merge the adapter, convert to GGUF and quantise (R-TRN-009).
#
#   export.sh <profile> [sft|dpo|auto] [--dry-run] [--redo]
#
# The model is an adapter STACK (lib.sh, D-650): "sft" is the SFT adapter, "dpo" is the stack with
# the preference run - sft,dpo for a bf16 profile (merged one after the other, in this order: the
# DPO adapter was trained on the SFT model merged into the base), the one continued adapter for the
# 4-bit profile.  "auto" (the default) is the stack the evaluation kept (state/stack_choice, written
# by `twin train remote eval`, D-655); with a DPO adapter on the disk and no such verdict it stops.
#
# Order (print it with --dry-run; nothing is touched then):
#   merge the adapter stack into the bf16 base -> build llama-quantize -> convert to an F16 GGUF
#   -> delete the merged HF weights -> quantise Q4_K_M, Q5_K_M, Q8_0 straight into
#   artifacts/gguf/ -> delete the F16 GGUF -> copy the adapter stack (two layers are also composed
#   into the one adapter vLLM serves) and write artifacts/manifest.json.
# A deletion only follows the step that made the file replacing the deleted one, and the free
# space is checked before every step that writes a large file.  The quantized files are written
# where they stay (no copy: at the peak the disk holds the base, the F16 file and the three
# quantizations, 3.2 times the bf16 size, and not a fourth set).  Running the script again skips
# the steps that are done (state/export.<step>); the adapters themselves stay for the remote vLLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"

ADAPTER_CHOICE="auto"
DRY_RUN=0
REDO=0
for argument in "${@:2}"; do
    case "${argument}" in
        --dry-run) DRY_RUN=1 ;;
        --redo) REDO=1 ;;
        sft | dpo | auto) ADAPTER_CHOICE="${argument}" ;;
        *) die "unknown argument '${argument}'" ;;
    esac
done

if [ "${DRY_RUN}" = "1" ]; then
    print_export_plan
    exit 0
fi

start_logging export
use_data_disk

KEPT="$(stack_choice "${ADAPTER_CHOICE}")" ||
    die "a DPO adapter exists but no evaluation has kept a stack (sft or dpo) yet: run eval.sh through 'twin train remote eval' (it compares the two on the same test contexts), or name the stack: export.sh ${TWIN_PROFILE} sft|dpo"

MERGED="${TWIN_HOME}/output/merged"
LLAMA_CPP="${TWIN_HOME}/llama.cpp"
GGUF_DIR="${TWIN_HOME}/gguf"
VENV_GGUF="${TWIN_HOME}/venv-gguf"
F16="${GGUF_DIR}/model-F16.gguf"
ARTIFACTS="${TWIN_HOME}/artifacts"
STATE="${TWIN_HOME}/state"
IFS=, read -r -a QUANTS <<<"${TWIN_GGUF_QUANTS}"
mkdir -p "${STATE}" "${GGUF_DIR}" "${ARTIFACTS}/gguf"
if [ "${REDO}" = "1" ]; then
    rm -f "${STATE}"/export.*
fi

step_done() { [ -f "${STATE}/export.$1" ]; }
mark_done() { touch "${STATE}/export.$1"; }

# the quantizations are written where they stay
quant_file() { echo "${ARTIFACTS}/gguf/${TWIN_PROFILE}-$1.gguf"; }

step_merge_adapter() {
    free_space_gate "merging the adapter" "$((TWIN_BASE_GB_CEIL + 3))"
    local adapters config="${TWIN_HOME}/config/export.run.yaml"
    adapters="$(stack_adapters "${KEPT}")"
    require_adapters "${adapters}"
    cp "${TWIN_HOME}/config/export.yaml" "${config}"
    yaml_set "${config}" adapter_name_or_path "${adapters}"
    rm -rf "${MERGED}"
    log "merging the ${KEPT} stack (${adapters//${TWIN_HOME}\//}) into ${TWIN_BASE_MODEL} (export_device: $(yaml_get "${config}" export_device))"
    llamafactory-cli export "${config}"
    [ -f "${MERGED}/config.json" ] || die "the merge did not produce a model in ${MERGED}"
    echo "${KEPT}" >"${STATE}/export.adapter"
}

ensure_build_tools() {
    command -v git >/dev/null 2>&1 || die "git is missing on this instance"
    if ! command -v cmake >/dev/null 2>&1; then
        "${TWIN_PYTHON}" -m pip install cmake
    fi
    if ! command -v c++ >/dev/null 2>&1; then
        apt-get install -y build-essential >/dev/null 2>&1 || die "no C++ compiler and apt-get failed"
    fi
}

step_prepare_llama_cpp() {
    free_space_gate "building llama.cpp" 6
    ensure_build_tools
    if [ ! -d "${LLAMA_CPP}/.git" ]; then
        with_turbo git clone --depth 1 --branch "${TWIN_LLAMA_CPP_TAG}" "${TWIN_LLAMA_CPP_REPO}" "${LLAMA_CPP}"
    fi
    cmake -S "${LLAMA_CPP}" -B "${LLAMA_CPP}/build" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    cmake --build "${LLAMA_CPP}/build" --target llama-quantize -j "$(nproc)"
    [ -x "${LLAMA_CPP}/build/bin/llama-quantize" ] || die "llama-quantize was not built"
    if [ ! -x "${VENV_GGUF}/bin/python" ]; then
        "${TWIN_PYTHON}" -m venv "${VENV_GGUF}"
    fi
    "${VENV_GGUF}/bin/python" -m pip install -r "${LLAMA_CPP}/requirements/requirements-convert_hf_to_gguf.txt"
}

step_convert_f16() {
    free_space_gate "the F16 conversion" "$((TWIN_BASE_GB_CEIL + 3))"
    [ -f "${MERGED}/config.json" ] || die "the merged model is missing; run the merge step again"
    rm -f "${F16}"
    "${VENV_GGUF}/bin/python" "${LLAMA_CPP}/convert_hf_to_gguf.py" "${MERGED}" --outfile "${F16}" --outtype f16
    [ -s "${F16}" ] || die "the F16 conversion produced no file"
}

step_delete_merged_hf() {
    [ -s "${F16}" ] || die "the F16 GGUF does not exist; refusing to delete the merged model"
    rm -rf "${MERGED}"
    log "merged HF weights deleted"
}

step_quantize() {
    local quant target
    for quant in "${QUANTS[@]}"; do
        target="$(quant_file "${quant}")"
        if [ -s "${target}" ]; then
            continue
        fi
        [ -s "${F16}" ] || die "the F16 GGUF is missing; run export.sh --redo to start from the merge"
        free_space_gate "quantising ${quant}" "$(quant_space_needed_gb "$(stat -c %s "${F16}")")"
        rm -f "${target}.part"
        "${LLAMA_CPP}/build/bin/llama-quantize" "${F16}" "${target}.part" "${quant}"
        mv "${target}.part" "${target}"
    done
}

step_delete_f16() {
    local quant
    for quant in "${QUANTS[@]}"; do
        [ -s "$(quant_file "${quant}")" ] || die "$(quant_file "${quant}") is missing; refusing to delete the F16 GGUF"
    done
    rm -f "${F16}"
    log "F16 GGUF deleted"
}

step_collect_artifacts() {
    # the stack the merge used (a rerun of this step must not pick another one)
    local kept name
    local -a layers=()
    kept="$(cat "${STATE}/export.adapter" 2>/dev/null || echo "${KEPT}")"
    while read -r name; do
        require_adapters "${TWIN_HOME}/output/${name}"
        layers+=(--layer "${name}=${TWIN_HOME}/output/${name}")
    done < <(stack_layers "${kept}")
    rm -rf "${ARTIFACTS}/adapter" "${ARTIFACTS}/adapters"
    # the one adapter vLLM serves (artifacts/adapter: the only layer, or the layers composed by
    # rank concatenation), the layers themselves when there are two, adapter_stack.json (D-652, D-653)
    "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/adapter_stack.py" \
        --artifacts "${ARTIFACTS}" --choice "${kept}" --mode "${TWIN_DPO_MODE}" "${layers[@]}"
    # the quantizations are already in artifacts/gguf (written there by step_quantize); this step
    # must not delete them, so that doing it again only rewrites the adapter copy and the manifest
    for quant in "${QUANTS[@]}"; do
        [ -s "$(quant_file "${quant}")" ] || die "$(quant_file "${quant}") is missing; run export.sh --redo"
    done
    "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/artifact_manifest.py" \
        --artifacts "${ARTIFACTS}" \
        --bundle-manifest "${TWIN_HOME}/manifest.json" \
        --quants "${TWIN_GGUF_QUANTS}" \
        --adapter-kind "${kept}" \
        --run-id "${TWIN_RUN_ID:-}"
}

for step in "${EXPORT_STEPS[@]}"; do
    if step_done "${step}"; then
        log "step ${step} is already done"
        continue
    fi
    log "step ${step}: $(export_step_text "${step}")"
    "step_${step}"
    mark_done "${step}"
done
log "export finished: $(find "${ARTIFACTS}/gguf" -name '*.gguf' | wc -l) GGUF files in ${ARTIFACTS}/gguf"
