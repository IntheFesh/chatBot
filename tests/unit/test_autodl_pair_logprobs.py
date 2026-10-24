"""R-TRN-009, R18-E6 (D-865): ``tools/pair_logprobs.py`` scores the held-out pairs for real.

On a tiny random Qwen3 (nothing is downloaded) with real PEFT adapters - an SFT adapter, a DPO
adapter made the ``layered`` way (a new LoRA on the merged SFT model) and one made the
``continued`` way (the SFT adapter trained further) - the tool's numbers must be the log
probabilities a plain forward pass of the same model gives: the stack built here by hand, its
adapters applied (not merged), the prompt tokenised as one string (the inference form), the reply
with ``<|im_end|>`` and the newline that follow it in training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from tests.support.tiny_lm import tiny_base, tiny_tokenizer, write_tiny_base
from tests.unit.test_autodl_tools import TOOLS, load_tool
from twin.training import lf_template
from twin.training.dataset_dir import DpoSample
from twin.training.lf_template import Turn

pair_logprobs = load_tool("pair_logprobs")

SYSTEM = "她说话很短，常用 [拥抱]。\n【此刻】\n当地时间：晚上"
PAIRS = (
    DpoSample("p0", SYSTEM, (Turn("user", "你好呀 在吗"),), "在呀 想你了", "您好，请问有什么事吗"),
    DpoSample(
        "p1",
        SYSTEM,
        (Turn("user", "今天好累"), Turn("assistant", "抱抱"), Turn("user", "吃饭了吗")),
        "吃了 [表情包:开心]",
        "我已经用过晚餐了。",
    ),
    DpoSample("p2", "", (Turn("user", "晚安"),), "晚安 [亲亲]\n明天见", "祝你有一个美好的夜晚"),
)


def lora(rank: int, alpha: int) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        task_type="CAUSAL_LM",
    )


def train_like(model: Any, seed: int) -> None:
    """What training leaves: every LoRA matrix away from its start."""
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = torch.randn(parameter.shape, generator=generator) * 0.3


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The base, the three adapters and the held-out file, once for the module."""
    root = tmp_path_factory.mktemp("pairs")
    base = write_tiny_base(root / "base")
    sft = get_peft_model(tiny_base(), lora(4, 8))
    train_like(sft, 11)
    sft.save_pretrained(str(root / "sft"))
    merged = PeftModel.from_pretrained(tiny_base(), str(root / "sft")).merge_and_unload()
    layered = get_peft_model(merged, lora(2, 6))  # create_new_adapter on the merged SFT model
    train_like(layered, 22)
    layered.save_pretrained(str(root / "dpo_layered"))
    continued = PeftModel.from_pretrained(tiny_base(), str(root / "sft"), is_trainable=True)
    train_like(continued, 33)  # the 4-bit profile continues the SFT adapter
    continued.save_pretrained(str(root / "dpo_continued"))
    pairs = root / "dpo_holdout.jsonl"
    pairs.write_text(
        "".join(json.dumps(p.to_json(), ensure_ascii=False) + "\n" for p in PAIRS), "utf-8"
    )
    return {
        "root": root,
        "base": base,
        "sft": root / "sft",
        "layered": root / "dpo_layered",
        "continued": root / "dpo_continued",
        "pairs": pairs,
    }


def direct_model(world: dict[str, Path], stack: str) -> Any:
    """The stack built by hand, adapters applied on top (PEFT forward), nothing merged last."""
    if stack == "sft":
        return PeftModel.from_pretrained(tiny_base(), str(world["sft"])).eval()
    if stack == "layered":
        merged = PeftModel.from_pretrained(tiny_base(), str(world["sft"])).merge_and_unload()
        return PeftModel.from_pretrained(merged, str(world["layered"])).eval()
    return PeftModel.from_pretrained(tiny_base(), str(world["continued"])).eval()


def direct_logp(model: Any, sample: DpoSample, reply: str) -> tuple[float, int]:
    """``sum log p(reply tokens)`` by a plain forward pass over the inference prompt string."""
    tokenizer = tiny_tokenizer()
    prompt = tokenizer.encode(
        lf_template.render_prompt(sample.system, sample.turns), add_special_tokens=False
    )
    response = tokenizer.encode(f"{reply}<|im_end|>\n", add_special_tokens=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt + response])).logits[0].float()
    total = 0.0
    for offset, token in enumerate(response):
        total += float(torch.log_softmax(logits[len(prompt) + offset - 1], dim=-1)[token])
    return total, len(response)


def adapters_of(world: dict[str, Path], stack: str) -> str:
    if stack == "sft":
        return str(world["sft"])
    if stack == "layered":
        return f"{world['sft']},{world['layered']}"
    return str(world["continued"])


@pytest.mark.parametrize("stack", ["sft", "layered", "continued"])
def test_the_scores_are_the_log_probabilities_of_a_plain_forward_pass(
    world: dict[str, Path], stack: str, tmp_path: Path
) -> None:
    out = tmp_path / "pair_logps.jsonl"
    code = pair_logprobs.main(
        [
            "--pairs",
            str(world["pairs"]),
            "--base",
            str(world["base"]),
            "--adapters",
            adapters_of(world, stack),
            "--dtype",
            "float32",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["p0", "p1", "p2"]
    model = direct_model(world, stack)
    for row, sample in zip(rows, PAIRS, strict=True):
        assert set(row) == {
            "id",
            "chosen_logp",
            "rejected_logp",
            "chosen_tokens",
            "rejected_tokens",
        }
        chosen, chosen_tokens = direct_logp(model, sample, sample.chosen)
        rejected, rejected_tokens = direct_logp(model, sample, sample.rejected)
        assert row["chosen_logp"] == pytest.approx(chosen, abs=1e-4)
        assert row["rejected_logp"] == pytest.approx(rejected, abs=1e-4)
        assert (row["chosen_tokens"], row["rejected_tokens"]) == (chosen_tokens, rejected_tokens)
    text = out.read_text("utf-8")
    for sample in PAIRS:  # numbers and ids only
        assert sample.chosen not in text and sample.rejected not in text


def test_the_stacks_score_differently_and_the_base_alone_differently_again(
    world: dict[str, Path],
) -> None:
    """The adapters are really applied: SFT, SFT+DPO and the base give three different scores."""
    sample = PAIRS[1]
    found = {
        stack: direct_logp(direct_model(world, stack), sample, sample.chosen)[0]
        for stack in ("sft", "layered")
    }
    found["base"] = direct_logp(tiny_base(), sample, sample.chosen)[0]
    loaded = pair_logprobs.load_stack(
        str(world["base"]), [str(world["sft"])], quantization_bit=None, dtype="float32"
    )
    prompt, chosen, _ = pair_logprobs.encode_pair(
        tiny_tokenizer(), pair_logprobs.read_pairs(world["pairs"])[1]
    )
    assert pair_logprobs.response_logprob(loaded, prompt, chosen) == pytest.approx(
        found["sft"], abs=1e-4
    )
    assert len({round(value, 3) for value in found.values()}) == 3


def test_the_prompt_is_the_slots_tokenised_one_by_one_as_llamafactory_does(
    world: dict[str, Path],
) -> None:
    tokenizer = tiny_tokenizer()
    [_, pair, _] = pair_logprobs.read_pairs(world["pairs"])
    prompt, chosen, rejected = pair_logprobs.encode_pair(tokenizer, pair)
    pieces = [lf_template.render_system(pair.system)] + [
        (lf_template.render_user if t.role == "user" else lf_template.render_assistant)(t.content)
        for t in pair.turns
    ]
    by_slot = [i for piece in pieces for i in tokenizer.encode(piece, add_special_tokens=False)]
    assert prompt == by_slot
    whole = tokenizer.encode(
        lf_template.render_prompt(pair.system, pair.turns), add_special_tokens=False
    )
    assert prompt == whole  # R-TRN-011: the training ids are the inference prompt's
    assert tokenizer.decode(chosen) == f"{pair.chosen}<|im_end|>\n"
    assert tokenizer.decode(rejected) == f"{pair.rejected}<|im_end|>\n"
    opener = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert prompt[-len(opener) :] == opener


def test_a_quantised_base_takes_exactly_one_adapter_of_a_profile_of_ours() -> None:
    with pytest.raises(ValueError, match="exactly one adapter"):
        pair_logprobs.load_stack("unused", ["a", "b"], quantization_bit=4, dtype="bfloat16")
    with pytest.raises(ValueError, match="8-bit base"):
        pair_logprobs.load_stack("unused", ["a"], quantization_bit=8, dtype="bfloat16")


def test_a_broken_file_names_its_line_and_never_a_text(tmp_path: Path) -> None:
    secret = "她的一句话"
    broken = tmp_path / "dpo_holdout.jsonl"
    for content in (secret, json.dumps({"id": "p0", "conversations": [{"from": "x"}]})):
        broken.write_text(content + "\n", encoding="utf-8")
        with pytest.raises(pair_logprobs.PairFileError, match="line 1") as caught:
            pair_logprobs.read_pairs(broken)
        assert secret not in str(caught.value)
    unrenderable = DpoSample("p9", "", (Turn("user", "嗯"),), "好", "也好")
    row = unrenderable.to_json()
    row["conversations"].append({"from": "gpt", "value": secret})  # ends with her turn
    broken.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    [pair] = pair_logprobs.read_pairs(broken)
    with pytest.raises(pair_logprobs.PairFileError, match="pair p9") as caught:
        pair_logprobs.encode_pair(tiny_tokenizer(), pair)
    assert secret not in str(caught.value)


def test_the_command_runs_on_its_own_and_fails_without_saying_a_text(
    world: dict[str, Path], tmp_path: Path
) -> None:
    """As ``eval.sh`` runs it: a separate process, ``twin.training.lf_template`` on its path."""
    out = tmp_path / "sft" / "pair_logps.jsonl"
    env = {**os.environ, "PYTHONPATH": str(Path(lf_template.__file__).parents[2])}
    command = [
        sys.executable,
        str(TOOLS / "pair_logprobs.py"),
        "--pairs",
        str(world["pairs"]),
        "--base",
        str(world["base"]),
        "--adapters",
        str(world["sft"]),
        "--dtype",
        "float32",
        "--out",
        str(out),
    ]
    done = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", env=env, check=False, timeout=300
    )
    assert done.returncode == 0, done.stderr
    assert "scored 3 held-out preference pairs under a stack of 1 adapter(s)" in done.stdout
    assert len(out.read_text("utf-8").splitlines()) == 3
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    failed = subprocess.run(
        [*command[:3], str(empty), *command[4:]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
        timeout=300,
    )
    assert failed.returncode == 1 and "holds no pairs" in failed.stderr
