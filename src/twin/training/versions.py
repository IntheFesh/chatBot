"""The software versions pinned for the AutoDL instance (checked 2026-10-09).

Everything the instance installs is pinned here, once.  ``training/autodl/versions.env`` is the
same list as ``KEY=value`` lines for the shell scripts; a test compares the committed file with
:func:`versions_env`, so the two cannot drift.

Why these versions:

* PyTorch 2.9.1 from the ``cu128`` index: Blackwell (compute capability 12.0, ``sm_120``) needs a
  CUDA 12.8 build of PyTorch 2.7 or newer, and the build script of v2.9.1 lists ``12.0`` for
  CUDA 12.8.  The default PyPI wheels of PyTorch 2.11 and later are CUDA 13 builds, which the
  project does not use, so torch is always installed from the ``cu128`` index first and every
  later ``pip install`` finds its requirement already met.
* transformers, peft, trl, accelerate and datasets sit at the upper bound that LLaMA-Factory 0.9.5
  accepts (transformers 4.57.6 is also what llama.cpp's converter is pinned to).
* bitsandbytes 0.49.2 ships ``libbitsandbytes_cuda128.so``, built for compute capability 12.0.
* llama.cpp tag ``b11177`` (2026-09-25): ``convert_hf_to_gguf.py`` registers ``Qwen3ForCausalLM``
  (``conversion/qwen.py``) and ``llama-quantize`` is a CMake target of the tree.
* vLLM 0.26.0 requires ``torch==2.11.0`` and is installed into its own virtual environment from
  the ``cu128`` index, so the training environment never changes.
"""

from __future__ import annotations

from typing import Final

from twin.training.lf_template import LLAMAFACTORY_VERSION

TORCH: Final = "2.9.1"
TORCHVISION: Final = "0.24.1"
TORCHAUDIO: Final = "2.9.1"
TORCH_INDEX_URL: Final = "https://download.pytorch.org/whl/cu128"
TORCH_MIN_VERSION: Final = "2.7"
CUDA_SERIES: Final = "12.8"
BLACKWELL_ARCH: Final = "sm_120"
BLACKWELL_CAPABILITY: Final = "12.0"

TRANSFORMERS: Final = "4.57.6"
PEFT: Final = "0.18.1"
TRL: Final = "0.24.0"
ACCELERATE: Final = "1.11.0"
DATASETS: Final = "4.0.0"
BITSANDBYTES: Final = "0.49.2"
MODELSCOPE: Final = "1.39.1"
ZSTANDARD: Final = "0.25.0"
CRYPTOGRAPHY_MIN: Final = "43"

LLAMA_CPP_REPO: Final = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_TAG: Final = "b11177"
GGUF_QUANTS: Final = ("Q4_K_M", "Q5_K_M", "Q8_0")

VLLM: Final = "0.26.0"
VLLM_TORCH: Final = "2.11.0"

# The tokenizer of the Qwen3 family (8B, 14B and 32B ship the same ``tokenizer.json``: checked
# 2026-10-09, identical bytes on Hugging Face and on ModelScope).  The training export counts
# tokens with it, so it is pinned by hash.
QWEN3_TOKENIZER_FILE: Final = "tokenizer.json"
QWEN3_TOKENIZER_SHA256: Final = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
QWEN3_TOKENIZER_BYTES: Final = 11_422_654
QWEN3_TOKENIZER_URLS: Final = (
    "https://huggingface.co/Qwen/Qwen3-8B/resolve/"
    "b968826d9c46dd6066d109eabc6255188de91218/tokenizer.json",
    "https://modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo?Revision=master&FilePath=tokenizer.json",
)


def versions_env() -> str:
    """The pinned versions as ``KEY=value`` lines (``training/autodl/versions.env``)."""
    values = {
        "TWIN_LLAMAFACTORY": LLAMAFACTORY_VERSION,
        "TWIN_TORCH": TORCH,
        "TWIN_TORCHVISION": TORCHVISION,
        "TWIN_TORCHAUDIO": TORCHAUDIO,
        "TWIN_TORCH_INDEX": TORCH_INDEX_URL,
        "TWIN_TORCH_MIN": TORCH_MIN_VERSION,
        "TWIN_CUDA_SERIES": CUDA_SERIES,
        "TWIN_BLACKWELL_ARCH": BLACKWELL_ARCH,
        "TWIN_BLACKWELL_CAPABILITY": BLACKWELL_CAPABILITY,
        "TWIN_TRANSFORMERS": TRANSFORMERS,
        "TWIN_PEFT": PEFT,
        "TWIN_TRL": TRL,
        "TWIN_ACCELERATE": ACCELERATE,
        "TWIN_DATASETS": DATASETS,
        "TWIN_BITSANDBYTES": BITSANDBYTES,
        "TWIN_MODELSCOPE": MODELSCOPE,
        "TWIN_ZSTANDARD": ZSTANDARD,
        "TWIN_CRYPTOGRAPHY_MIN": CRYPTOGRAPHY_MIN,
        "TWIN_LLAMA_CPP_REPO": LLAMA_CPP_REPO,
        "TWIN_LLAMA_CPP_TAG": LLAMA_CPP_TAG,
        "TWIN_GGUF_QUANTS": ",".join(GGUF_QUANTS),
        "TWIN_VLLM": VLLM,
        "TWIN_VLLM_TORCH": VLLM_TORCH,
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())
