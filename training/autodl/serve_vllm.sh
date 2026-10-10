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
# Versions (checked 2026-10-10): vLLM 0.26.0 requires torch 2.11.0, torchvision 0.26.0 and
# torchaudio 2.11.0.  The GitHub release page has the wheel "vllm-0.26.0+cu129" (there is no +cu128
# and no +cu130); PyPI's default wheel of the same version is a CUDA 13 build, so it is never used.
# The three PyTorch wheels come from the cu129 index of PyTorch.  Whether this combination runs on
# the compute capability 12.0 of the RTX 5090 and the RTX PRO 6000 (and takes a LoRA adapter with
# bitsandbytes 4-bit) could not be tried without the hardware, so the first start prints the
# versions and architectures it ended up with.
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
    if [ -x "${VENV}/bin/vllm" ] && "${VENV}/bin/python" -m pip show vllm 2>/dev/null | grep -q "^Version: ${TWIN_VLLM}+${TWIN_VLLM_CUDA_TAG}"; then
        return 0
    fi
    free_space_gate "installing vLLM" 20
    [ -x "${VENV}/bin/python" ] || "${TWIN_PYTHON}" -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install "torch==${TWIN_VLLM_TORCH}" "torchvision==${TWIN_VLLM_TORCHVISION}" \
        "torchaudio==${TWIN_VLLM_TORCHAUDIO}" --index-url "${TWIN_VLLM_TORCH_INDEX}"
    local wheel="https://github.com/vllm-project/vllm/releases/download/v${TWIN_VLLM}/vllm-${TWIN_VLLM}%2B${TWIN_VLLM_CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl"
    # no fall-back to "pip install vllm==...": PyPI's default wheel is built for CUDA 13
    with_turbo "${VENV}/bin/python" -m pip install "${wheel}" --extra-index-url "${TWIN_VLLM_TORCH_INDEX}" \
        || die "the vLLM wheel ${wheel} could not be installed (see the messages above)"
}

install_vllm
"${VENV}/bin/python" -c 'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())'
mapfile -t ARGUMENTS < <(vllm_arguments)
log "starting vLLM on 127.0.0.1:${PORT} with the adapter twin-style"
exec "${VENV}/bin/vllm" "${ARGUMENTS[@]}"
