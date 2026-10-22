"""R-TRN-009, R17-49d (D-650, D-652): the DPO reference model and the adapter stack, for real.

LLaMA-Factory 0.9.5 builds a bf16 profile's DPO run with PEFT calls this test makes itself, on a
tiny randomly initialised Qwen3 (a few layers, a few dozen dimensions; nothing is downloaded):

* ``adapter_name_or_path`` + ``create_new_adapter`` (``model/adapter.py``): the SFT adapter is
  loaded and **merged** into the base (``PeftModel.from_pretrained(...).merge_and_unload()``), a
  new LoRA is created on the merged model (``get_peft_model``);
* without a ``ref_model`` the trainer switches the new adapter off for the reference pass
  (``train/dpo/trainer.py``: ``model.disable_adapter()``).

So the claim of the spec - "the reference model is the SFT model" - is: with the new adapter off,
the outputs are the SFT model's, not the base's.  And what the export and the remote vLLM do with
the two adapters (merge them one after the other; serve one adapter composed by rank
concatenation, ``training/autodl/tools/adapter_stack.py``) must give ``W + dSFT + dDPO``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from tests.unit.test_autodl_tools import load_tool

adapter_stack = load_tool("adapter_stack")

CONFIG = Qwen3Config(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    tie_word_embeddings=False,
)
IDS = torch.tensor([[1, 5, 9, 3, 7, 2, 11, 4]])
PROMPTS = [IDS, torch.tensor([[2, 2, 8, 30, 12, 63]])]


def base_model() -> Qwen3ForCausalLM:
    """The same random base every time (the 'downloaded' weights of the test)."""
    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(CONFIG)
    model.eval()
    return model


def logits_of(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([model(input_ids=ids).logits for ids in PROMPTS], dim=1)


def train_like(model: Any, seed: int) -> None:
    """What training does to a fresh LoRA: B leaves zero (A already has its random start)."""
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data = torch.randn(parameter.shape, generator=generator) * 0.05


def lora(rank: int, alpha: int, *, rslora: bool = False) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=rslora,
        task_type="CAUSAL_LM",
    )


def make_sft(directory: Path, rank: int = 4, alpha: int = 8, *, rslora: bool = False) -> None:
    """An SFT adapter on the base, saved the way LLaMA-Factory saves it."""
    model = get_peft_model(base_model(), lora(rank, alpha, rslora=rslora))
    train_like(model, seed=11)
    model.save_pretrained(str(directory))


def merged_sft(directory: Path) -> Qwen3ForCausalLM:
    """The base with the SFT adapter merged in: what ``create_new_adapter`` starts DPO from."""
    peft = PeftModel.from_pretrained(base_model(), str(directory))
    merged = peft.merge_and_unload()
    merged.eval()
    return merged  # type: ignore[no-any-return]


def make_dpo(sft_dir: Path, directory: Path, rank: int = 2, alpha: int = 6) -> PeftModel:
    """A NEW adapter on the merged SFT model (``create_new_adapter``), 'trained' and saved."""
    model = get_peft_model(merged_sft(sft_dir), lora(rank, alpha))
    train_like(model, seed=22)
    model.save_pretrained(str(directory))
    model.eval()
    return model


def largest_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def test_switching_the_new_adapter_off_gives_the_sft_model_and_not_the_base(
    tmp_path: Path,
) -> None:
    """R17-49d (D-650): the reference of ``create_new_adapter`` is SFT; the old one was the base.

    * the SFT model = the base with the SFT adapter (applied, not merged);
    * the policy before DPO = the merged SFT model + a new adapter: its output with the new adapter
      off equals the SFT model's, to floating-point precision;
    * the bare base (what the previous setup used as the reference) is far from it;
    * with the adapter on, the policy differs from SFT (the new adapter does something).
    """
    make_sft(tmp_path / "sft")
    sft_model = PeftModel.from_pretrained(base_model(), str(tmp_path / "sft"))
    sft_model.eval()
    sft_logits = logits_of(sft_model)
    base_logits = logits_of(base_model())
    policy = make_dpo(tmp_path / "sft", tmp_path / "dpo")
    with policy.disable_adapter():
        reference = logits_of(policy)
    assert largest_difference(reference, sft_logits) < 1e-5
    assert largest_difference(base_logits, sft_logits) > 1e-2  # the SFT adapter changes the model
    assert largest_difference(reference, base_logits) > 1e-2  # ... so the reference is not the base
    assert largest_difference(logits_of(policy), sft_logits) > 1e-2


def test_the_reference_is_still_sft_when_the_policy_has_trained_away_from_it(
    tmp_path: Path,
) -> None:
    """The adapter switched off stays SFT however far the new adapter has moved."""
    make_sft(tmp_path / "sft", rank=4, alpha=8)
    sft_logits = logits_of(PeftModel.from_pretrained(base_model(), str(tmp_path / "sft")).eval())
    policy = get_peft_model(merged_sft(tmp_path / "sft"), lora(2, 6))
    for seed, scale in ((5, 0.05), (6, 0.5), (7, 2.0)):  # B grows: DPO moves the policy
        generator = torch.Generator().manual_seed(seed)
        for name, parameter in policy.named_parameters():
            if "lora_B" in name:
                parameter.data = torch.randn(parameter.shape, generator=generator) * scale
        with policy.disable_adapter():
            assert largest_difference(logits_of(policy.eval()), sft_logits) < 1e-5


@pytest.mark.parametrize(
    ("sft_rank", "sft_alpha", "dpo_rank", "dpo_alpha", "rslora"),
    [
        (4, 8, 2, 6, False),  # different ranks and scales (2.0 and 3.0)
        (4, 4, 4, 16, False),  # scale 1.0 and 4.0
        (3, 9, 5, 5, False),  # odd ranks
        (4, 8, 2, 6, True),  # rank-stabilised SFT (scale alpha / sqrt(r))
    ],
)
def test_the_stack_is_sft_plus_dpo_merged_one_after_the_other_and_served_as_one_composed_adapter(
    tmp_path: Path, sft_rank: int, sft_alpha: int, dpo_rank: int, dpo_alpha: int, rslora: bool
) -> None:
    """R17-49d (D-650, D-652): ``sft,dpo`` merged in order = the rank-concatenated adapter.

    The first is what ``llamafactory-cli export`` does with ``adapter_name_or_path: sft,dpo``
    (``model/adapter.py``: merge each adapter into the model in turn).  The second is what the
    remote vLLM loads: ``artifacts/adapter``, whose A is ``[A1; A2]`` and whose B is
    ``[s1*B1 | s2*B2]`` with ``alpha = r1 + r2``.  Both must be ``W + dSFT + dDPO`` - the model the
    DPO adapter was trained on top of - whatever rank and alpha each layer has.
    """
    sft_dir, dpo_dir, served_dir = tmp_path / "sft", tmp_path / "dpo", tmp_path / "served"
    make_sft(sft_dir, sft_rank, sft_alpha, rslora=rslora)
    make_dpo(sft_dir, dpo_dir, dpo_rank, dpo_alpha)

    exported = PeftModel.from_pretrained(base_model(), str(sft_dir)).merge_and_unload()
    exported = PeftModel.from_pretrained(exported, str(dpo_dir)).merge_and_unload()
    exported.eval()

    config = adapter_stack.compose_files([("sft", sft_dir), ("dpo", dpo_dir)], served_dir)
    assert config["r"] == sft_rank + dpo_rank and config["lora_alpha"] == sft_rank + dpo_rank
    assert config["use_rslora"] is False  # the scales are in B now
    served = PeftModel.from_pretrained(base_model(), str(served_dir))
    served.eval()

    assert largest_difference(logits_of(served), logits_of(exported)) < 1e-5
    # and it is not just the SFT model or just the DPO delta on the base
    sft_only = PeftModel.from_pretrained(base_model(), str(sft_dir)).eval()
    assert largest_difference(logits_of(served), logits_of(sft_only)) > 1e-3
    dpo_alone = PeftModel.from_pretrained(base_model(), str(dpo_dir)).eval()
    assert largest_difference(logits_of(served), logits_of(dpo_alone)) > 1e-3


def test_a_stack_in_the_wrong_order_or_with_a_layer_twice_is_a_different_model(
    tmp_path: Path,
) -> None:
    """The DPO adapter is a delta over the merged SFT weights: only ``sft`` then ``dpo`` is it.

    LoRA deltas add, so the order does not change a sum of deltas - but a stack that listed ``sft``
    twice (what ``sft,dpo`` would be for the 4-bit profile, whose DPO adapter already contains SFT)
    counts the SFT delta twice and is not the model that was trained.
    """
    sft_dir, dpo_dir = tmp_path / "sft", tmp_path / "dpo"
    make_sft(sft_dir)
    make_dpo(sft_dir, dpo_dir)
    right = PeftModel.from_pretrained(base_model(), str(sft_dir)).merge_and_unload()
    right = PeftModel.from_pretrained(right, str(dpo_dir)).merge_and_unload()
    twice = PeftModel.from_pretrained(base_model(), str(sft_dir)).merge_and_unload()
    twice = PeftModel.from_pretrained(twice, str(sft_dir)).merge_and_unload()
    twice = PeftModel.from_pretrained(twice, str(dpo_dir)).merge_and_unload()
    assert largest_difference(logits_of(right.eval()), logits_of(twice.eval())) > 1e-2


def test_the_composed_adapter_survives_a_round_trip_through_the_collect_step(
    tmp_path: Path,
) -> None:
    """``collect`` (what ``export.sh`` runs) keeps the layers, composes, and describes all of it."""
    sft_dir, dpo_dir = tmp_path / "sft", tmp_path / "dpo"
    make_sft(sft_dir, 4, 8)
    make_dpo(sft_dir, dpo_dir, 2, 6)
    artifacts = tmp_path / "artifacts"
    stack = adapter_stack.collect(artifacts, "dpo", "layered", [("sft", sft_dir), ("dpo", dpo_dir)])
    assert stack["kind"] == "sft+dpo" and stack["served"]["composed"] is True
    assert [layer["rank"] for layer in stack["layers"]] == [4, 2] and stack["served"]["rank"] == 6
    exported = PeftModel.from_pretrained(base_model(), str(sft_dir)).merge_and_unload()
    exported = PeftModel.from_pretrained(exported, str(dpo_dir)).merge_and_unload()
    served = PeftModel.from_pretrained(base_model(), str(artifacts / "adapter"))
    assert largest_difference(logits_of(served.eval()), logits_of(exported.eval())) < 1e-5
