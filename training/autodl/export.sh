#!/usr/bin/env bash
# Merge the adapter, convert to GGUF and quantise (R-TRN-009).
#
#   export.sh <profile> [sft|dpo|auto] [--dry-run] [--redo]
#
# Order (print it with --dry-run; nothing is touched then):
#   merge the adapter into the bf16 base -> build llama-quantize -> convert to an F16 GGUF
#   -> delete the merged HF weights -> quantise Q4_K_M, Q5_K_M, Q8_0 -> delete the F16 GGUF
#   -> copy the adapter and write artifacts/manifest.json.
# A deletion only follows the step that made the file replacing the deleted one, and the free
# space is checked before every step that writes a large file.  Running the script again skips
# the steps that are done (state/export.<step>); the adapter itself stays for the remote vLLM.
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

MERGED="${TWIN_HOME}/output/merged"
LLAMA_CPP="${TWIN_HOME}/llama.cpp"
GGUF_DIR="${TWIN_HOME}/gguf"
VENV_GGUF="${TWIN_HOME}/venv-gguf"
F16="${GGUF_DIR}/model-F16.gguf"
ARTIFACTS="${TWIN_HOME}/artifacts"
STATE="${TWIN_HOME}/state"
IFS=, read -r -a QUANTS <<<"${TWIN_GGUF_QUANTS}"
mkdir -p "${STATE}" "${GGUF_DIR}" "${ARTIFACTS}"
if [ "${REDO}" = "1" ]; then
    rm -f "${STATE}"/export.*
fi

step_done() { [ -f "${STATE}/export.$1" ]; }
mark_done() { touch "${STATE}/export.$1"; }

quant_file() { echo "${GGUF_DIR}/${TWIN_PROFILE}-$1.gguf"; }

step_merge_adapter() {
    free_space_gate "merging the adapter" "$((TWIN_BASE_GB_CEIL + 3))"
    local adapter config="${TWIN_HOME}/config/export.run.yaml"
    adapter="$(best_adapter "${ADAPTER_CHOICE}")"
    [ -f "${adapter}/adapter_model.safetensors" ] || die "no adapter in ${adapter}; run train.sh first"
    cp "${TWIN_HOME}/config/export.yaml" "${config}"
    yaml_set "${config}" adapter_name_or_path "${adapter}"
    rm -rf "${MERGED}"
    log "merging ${adapter##*/} into ${TWIN_BASE_MODEL} (export_device: $(yaml_get "${config}" export_device))"
    llamafactory-cli export "${config}"
    [ -f "${MERGED}/config.json" ] || die "the merge did not produce a model in ${MERGED}"
    echo "${adapter}" >"${STATE}/export.adapter"
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
        free_space_gate "quantising ${quant}" "$((TWIN_BASE_GB_CEIL / 2 + 2))"
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
    local adapter
    adapter="$(cat "${STATE}/export.adapter" 2>/dev/null || best_adapter "${ADAPTER_CHOICE}")"
    rm -rf "${ARTIFACTS}/adapter" "${ARTIFACTS}/gguf"
    mkdir -p "${ARTIFACTS}/adapter" "${ARTIFACTS}/gguf"
    for file in adapter_config.json adapter_model.safetensors README.md; do
        if [ -f "${adapter}/${file}" ]; then cp "${adapter}/${file}" "${ARTIFACTS}/adapter/"; fi
    done
    for quant in "${QUANTS[@]}"; do
        cp "$(quant_file "${quant}")" "${ARTIFACTS}/gguf/"
    done
    "${TWIN_PYTHON}" "${SCRIPT_DIR}/tools/artifact_manifest.py" \
        --artifacts "${ARTIFACTS}" \
        --bundle-manifest "${TWIN_HOME}/manifest.json" \
        --quants "${TWIN_GGUF_QUANTS}" \
        --adapter-kind "${adapter##*/}" \
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
