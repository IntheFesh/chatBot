"""Which quantisation a card holds, with 20 % of its memory to spare (R-SRV-002)."""

from __future__ import annotations

import pytest

from twin.serving.quant import (
    COMPUTE_BUFFER_BYTES,
    DEFAULT_KV_BYTES_PER_TOKEN,
    GB,
    GIB,
    KV_BYTES_PER_TOKEN,
    QuantOption,
    estimated_options,
    kv_bytes_per_token,
    need_bytes,
    recommend_quant,
)
from twin.training.profiles import PROFILES


def gib(value: float) -> int:
    return round(value * GIB)


def eight_b() -> list[QuantOption]:
    """The three files of a Qwen3-8B (sizes of llama.cpp's quantisations of that model)."""
    return estimated_options(PROFILES["5090-8b"].base_gb)


def test_the_kv_cache_of_the_qwen3_sizes_is_two_times_layers_heads_dimension_times_two_bytes() -> (
    None
):
    assert KV_BYTES_PER_TOKEN["Qwen/Qwen3-8B"] == 147_456
    assert KV_BYTES_PER_TOKEN["Qwen/Qwen3-14B"] == 163_840
    assert KV_BYTES_PER_TOKEN["Qwen/Qwen3-32B"] == 262_144
    assert kv_bytes_per_token("Qwen/Qwen3-14B") == 163_840
    assert kv_bytes_per_token("Some/Other") == DEFAULT_KV_BYTES_PER_TOKEN == 262_144
    assert kv_bytes_per_token(None) == DEFAULT_KV_BYTES_PER_TOKEN


def test_the_estimated_sizes_add_up_to_about_1_19_times_the_bf16_weights() -> None:
    options = estimated_options(16.4)
    assert [o.quant for o in options] == ["Q8_0", "Q5_K_M", "Q4_K_M"]
    assert sum(o.size_bytes for o in options) / (16.4 * GB) == pytest.approx(1.193, abs=0.001)
    # the files of Qwen3-8B on Hugging Face: Q8_0 8.7 GB, Q5_K_M 5.85 GB, Q4_K_M 5.03 GB
    assert [round(o.size_bytes / GB, 1) for o in options] == [8.7, 5.8, 5.0]


def test_a_32_gb_card_holds_q8_of_the_8b_model_with_the_margin() -> None:
    advice = recommend_quant(
        eight_b(),
        vram_bytes=gib(32),
        context=4096,
        kv_per_token=KV_BYTES_PER_TOKEN["Qwen/Qwen3-8B"],
    )
    assert advice.mode == "gpu" and advice.quant == "Q8_0"
    assert advice.budget_bytes == int(gib(32) * 0.8)
    assert all(need <= advice.budget_bytes for quant, need in advice.needed if quant == "Q8_0")


def test_the_margin_decides_between_neighbours() -> None:
    """Q8_0 of the 8B model needs about 9.2 GiB: a 12 GiB card (9.6 GiB usable) holds it, an
    11 GiB card (8.8 GiB usable) falls to Q5_K_M (6.5 GiB), and 8 GiB (6.4 usable) to Q4_K_M."""
    options = eight_b()
    kv = KV_BYTES_PER_TOKEN["Qwen/Qwen3-8B"]
    chosen = {
        size: recommend_quant(options, vram_bytes=gib(size), kv_per_token=kv).quant
        for size in (12, 11, 8)
    }
    assert chosen == {12: "Q8_0", 11: "Q5_K_M", 8: "Q4_K_M"}


def test_exactly_the_usable_memory_still_fits_and_one_byte_less_does_not() -> None:
    option = QuantOption("Q4_K_M", 4 * GIB)
    kv = 100_000
    need = need_bytes(option, context=4096, kv_per_token=kv)
    assert need == 4 * GIB + 4096 * kv + COMPUTE_BUFFER_BYTES
    exactly = recommend_quant([option], vram_bytes=int(need / 0.8) + 1, kv_per_token=kv)
    assert exactly.quant == "Q4_K_M" and exactly.mode == "gpu"
    short = recommend_quant([option], vram_bytes=int(need / 0.8) - 10, kv_per_token=kv)
    assert short.quant is None and short.mode == "remote"


def test_a_longer_context_needs_more_memory() -> None:
    options = [QuantOption("Q5_K_M", gib(6))]
    kv = KV_BYTES_PER_TOKEN["Qwen/Qwen3-8B"]
    assert recommend_quant(options, vram_bytes=gib(10), context=4096, kv_per_token=kv).quant
    assert (
        recommend_quant(options, vram_bytes=gib(10), context=32768, kv_per_token=kv).quant is None
    )


def test_a_card_that_cannot_hold_even_q4_sends_the_model_to_the_rented_instance() -> None:
    advice = recommend_quant(
        estimated_options(PROFILES["pro6000-32b"].base_gb),
        vram_bytes=gib(24),
        kv_per_token=KV_BYTES_PER_TOKEN["Qwen/Qwen3-32B"],
    )
    assert advice.mode == "remote" and advice.quant is None
    assert "vllm_completion" in " ".join(advice.lines())


def test_only_the_available_files_are_considered() -> None:
    only_q4 = [QuantOption("Q4_K_M", gib(5))]
    assert recommend_quant(only_q4, vram_bytes=gib(32)).quant == "Q4_K_M"
    adapter_only = [QuantOption("lora", gib(1))]
    nothing = recommend_quant(adapter_only, vram_bytes=gib(32))
    assert nothing.mode == "remote" and nothing.quant is None


def test_without_a_card_it_is_q4_and_a_warning_about_the_speed() -> None:
    advice = recommend_quant(eight_b(), vram_bytes=None)
    assert advice.mode == "cpu" and advice.quant == "Q4_K_M"
    assert any("slow" in line for line in advice.lines())
    only_q8 = recommend_quant([QuantOption("Q8_0", gib(9))], vram_bytes=None)
    assert only_q8.quant == "Q8_0"  # nothing smaller exists: the smallest file there is


def test_the_advice_lists_what_each_quantisation_needs() -> None:
    advice = recommend_quant(eight_b(), vram_bytes=gib(32))
    lines = advice.lines()
    assert lines[0].startswith("Q8_0") and any("Q4_K_M: needs" in line for line in lines)
