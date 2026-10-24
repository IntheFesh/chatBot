"""Which quantisation of the model a graphics card can hold (R-SRV-002).

``export.sh`` writes three GGUF files per model: Q8_0, Q5_K_M and Q4_K_M.  Bigger is closer to the
trained model, so the recommendation is the **largest that fits with 20 % of the card's memory to
spare**: the weights, plus the key/value cache for the context window (4096 tokens), plus the
buffers llama.cpp computes in, must stay within 80 % of ``memory.total``.  The 20 % is the
margin for the desktop, the driver and other programs; it is not negotiable here.

The KV cache of the Qwen3 models the training profiles use (checked on 2026-10-10 in the
``config.json`` of each model on Hugging Face) is, in 16-bit floats::

    2 (keys and values) x layers x kv heads x head dim x 2 bytes per token

=========== ======= ========= ======== ===================
model       layers  kv heads  head dim  bytes per token
=========== ======= ========= ======== ===================
Qwen3-8B    36      8         128      147,456
Qwen3-14B   40      8         128      163,840
Qwen3-32B   64      8         128      262,144
=========== ======= ========= ======== ===================

Without a card the advice is Q4_K_M and a warning about the speed; a card too small for even
Q4_K_M means the model belongs on the rented instance (``style_model.mode: vllm_completion``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

QUANT_ORDER: Final = ("Q8_0", "Q5_K_M", "Q4_K_M")
BF16_FRACTION: Final = {"Q8_0": 0.531, "Q5_K_M": 0.356, "Q4_K_M": 0.306}
"""File size as a fraction of the bf16 weights (the three together are about 1.19 times)."""
KV_BYTES_PER_TOKEN: Final = {
    "Qwen/Qwen3-8B": 2 * 36 * 8 * 128 * 2,
    "Qwen/Qwen3-14B": 2 * 40 * 8 * 128 * 2,
    "Qwen/Qwen3-32B": 2 * 64 * 8 * 128 * 2,
}
DEFAULT_KV_BYTES_PER_TOKEN: Final = KV_BYTES_PER_TOKEN[
    "Qwen/Qwen3-32B"
]  # unknown model: the largest
HEADROOM: Final = 0.20
COMPUTE_BUFFER_BYTES: Final = 512 * 1024 * 1024
CPU_BANDWIDTH_BYTES_PER_S: Final = 40e9
"""Memory bandwidth of an ordinary dual-channel desktop; generating a token reads every weight
once, so bandwidth divided by file size is the *upper bound* of the speed on a CPU."""
GIB: Final = 1024**3
GB: Final = 10**9

Mode = Literal["gpu", "cpu", "remote"]


@dataclass(frozen=True)
class QuantOption:
    """One GGUF file that could be served."""

    quant: str
    size_bytes: int


@dataclass(frozen=True)
class QuantAdvice:
    """The recommendation and the numbers behind it."""

    mode: Mode
    quant: str | None
    reason: str
    budget_bytes: int | None = None
    needed: tuple[tuple[str, int], ...] = ()  # (quant, bytes it needs), largest quant first
    notes: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        out = [self.reason]
        if self.budget_bytes is not None:
            out.append(f"usable memory (80 % of the card): {self.budget_bytes / GIB:.1f} GiB")
        out.extend(f"  {quant}: needs {need / GIB:.1f} GiB" for quant, need in self.needed)
        out.extend(self.notes)
        return out


def kv_bytes_per_token(base_model: str | None) -> int:
    """KV cache bytes per token for ``base_model`` (the largest known when it is not listed)."""
    return KV_BYTES_PER_TOKEN.get(base_model or "", DEFAULT_KV_BYTES_PER_TOKEN)


def estimated_options(base_gb: float) -> list[QuantOption]:
    """File sizes before the files exist: from the size of the bf16 weights (``base_gb`` is in
    decimal gigabytes, the way the model repository lists it)."""
    return [QuantOption(q, round(base_gb * GB * BF16_FRACTION[q])) for q in QUANT_ORDER]


def need_bytes(option: QuantOption, *, context: int, kv_per_token: int) -> int:
    """Memory a quantisation needs on the card: weights, KV cache of the context, buffers."""
    return option.size_bytes + context * kv_per_token + COMPUTE_BUFFER_BYTES


def _ranked(options: list[QuantOption]) -> list[QuantOption]:
    known = {o.quant: o for o in options if o.quant in QUANT_ORDER}
    return [known[q] for q in QUANT_ORDER if q in known]


def recommend_quant(
    options: list[QuantOption],
    *,
    vram_bytes: int | None,
    context: int = 4096,
    kv_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
    headroom: float = HEADROOM,
) -> QuantAdvice:
    """The quantisation to serve on a card of ``vram_bytes`` (``None``: no card)."""
    ranked = _ranked(options)
    if not ranked:
        return QuantAdvice("remote", None, "no GGUF file (Q8_0, Q5_K_M or Q4_K_M) to choose from")
    if vram_bytes is None:
        smallest = ranked[-1]
        chosen = next((o for o in ranked if o.quant == "Q4_K_M"), smallest)
        speed = CPU_BANDWIDTH_BYTES_PER_S / chosen.size_bytes
        notes = (
            f"no NVIDIA card: {chosen.quant} runs on the CPU, at best about {speed:.0f} tokens "
            "per second (bounded by memory bandwidth), and the long prompt takes seconds to "
            "read first - replies will be slow; a card or the rented instance is much faster",
        )
        return QuantAdvice("cpu", chosen.quant, f"{chosen.quant} for the CPU", notes=notes)
    budget = int(vram_bytes * (1.0 - headroom))
    needed = tuple(
        (o.quant, need_bytes(o, context=context, kv_per_token=kv_per_token)) for o in ranked
    )
    for quant, need in needed:
        if need <= budget:
            return QuantAdvice(
                "gpu",
                quant,
                f"{quant}: the largest that fits with {headroom:.0%} of the memory to spare",
                budget,
                needed,
            )
    return QuantAdvice(
        "remote",
        None,
        f"even {ranked[-1].quant} does not fit this card with {headroom:.0%} to spare",
        budget,
        needed,
        ("serve the model on the rented instance instead: style_model.mode vllm_completion",),
    )
