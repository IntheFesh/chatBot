#!/usr/bin/env bash
# Serve the base model with the trained LoRA adapter through vLLM (R-SRV-003; used by round 14).
#
#   serve_vllm.sh <profile> [--dry-run]
#
# vLLM lives in its own virtual environment (it pins its own PyTorch), so the training
# environment is never changed.  The server listens on 127.0.0.1 only - the local machine reaches
# it through an SSH tunnel.  The client talks to two endpoints only:
#   POST /v1/completions   with "model": "twin-style" and "prompt": the string StylePromptBuilder
#                          rendered (the server applies no chat template of its own)
#   POST /tokenize         to compare the server's token ids with the local tokenizer
# Versions (checked 2026-10-09): vLLM 0.26.0 requires torch 2.11.0, its builds list compute
# capability 12.0 (RTX 5090 and RTX PRO 6000); whether the wheels run on these two cards could not
# be tried without the hardware, so the first start prints the versions it ended up with.
# --dry-run prints the command and exits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "${SCRIPT_DIR}/lib.sh"

load_env_files "${SCRIPT_DIR}"
require_profile "${1:-}"

DRY_RUN=0
for argument in "${@:2}"; do
    case "${argument}" in
        --dry-run) DRY_RUN=1 ;;
        *) die "unknown argument '${argument}'" ;;
    esac
done

VENV="${TWIN_HOME}/venv-vllm"
PORT="${TWIN_VLLM_PORT:-8000}"
MODEL_DIR="${TWIN_MODELS}/${TWIN_BASE_MODEL}"
ADAPTER_DIR="${TWIN_HOME}/artifacts/adapter"

# vllm_arguments: the arguments of "vllm serve" (one per line).
vllm_arguments() {
    printf '%s\n' serve "${MODEL_DIR}" \
        --host 127.0.0.1 --port "${PORT}" \
        --served-model-name base \
        --enable-lora --lora-modules "twin-style=${ADAPTER_DIR}" \
        --max-lora-rank "${TWIN_LORA_RANK}" --max-loras 1 \
        --max-model-len 4096 --dtype bfloat16
    # a 14B model in bf16 does not fit 32 GB next to the cache: load it 4-bit like in training
    if [ "${TWIN_QUANT_BIT}" != "0" ]; then
        printf '%s\n' --quantization bitsandbytes
    fi
}

if [ "${DRY_RUN}" = "1" ]; then
    vllm_arguments | tr '\n' ' '
    echo
    exit 0
fi

start_logging serve_vllm
use_data_disk
[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ] || die "no adapter in ${ADAPTER_DIR}; run export.sh first"

install_vllm() {
    if [ -x "${VENV}/bin/vllm" ] && "${VENV}/bin/python" -m pip show vllm 2>/dev/null | grep -q "^Version: ${TWIN_VLLM}"; then
        return 0
    fi
    free_space_gate "installing vLLM" 20
    [ -x "${VENV}/bin/python" ] || "${TWIN_PYTHON}" -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install "torch==${TWIN_VLLM_TORCH}" --index-url "${TWIN_TORCH_INDEX}"
    local wheel="https://github.com/vllm-project/vllm/releases/download/v${TWIN_VLLM}/vllm-${TWIN_VLLM}%2Bcu128-cp38-abi3-manylinux_2_28_x86_64.whl"
    if ! with_turbo "${VENV}/bin/python" -m pip install "${wheel}" --extra-index-url "${TWIN_TORCH_INDEX}"; then
        log "the CUDA 12.8 wheel could not be installed; trying vllm==${TWIN_VLLM} from PyPI"
        "${VENV}/bin/python" -m pip install "vllm==${TWIN_VLLM}" --extra-index-url "${TWIN_TORCH_INDEX}"
    fi
}

install_vllm
"${VENV}/bin/python" -c 'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())'
mapfile -t ARGUMENTS < <(vllm_arguments)
log "starting vLLM on 127.0.0.1:${PORT} with the adapter twin-style"
exec "${VENV}/bin/vllm" "${ARGUMENTS[@]}"
