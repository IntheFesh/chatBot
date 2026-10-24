"""R18-E6 (D-865): ``tools/pair_logprobs.py`` against LLaMA-Factory 0.9.5 itself.

The tool scores the held-out preference pairs the way the DPO training saw them.  Here the same
tiny random Qwen3 with a real SFT adapter and a ``layered`` DPO adapter is loaded by
LLaMA-Factory's own code - ``get_infer_args`` + ``load_tokenizer`` + ``load_model`` (the
evaluation's way: every adapter merged in order) - the pairs are encoded by its
``qwen3_nothink`` template as the pairwise (DPO) processor does (``encode_oneturn``) and the log
probabilities summed by its ``get_batch_logps``; the tool runs in the same interpreter (the
instance's packages: transformers 4.57.6, peft 0.18.1) and must give the same numbers.

It needs the LLaMA-Factory environment of ``tests/integration/test_template_parity.py``::

    TWIN_LLAMAFACTORY_PYTHON=lfvenv/bin/python \\
        uv run pytest tests/integration/test_pair_logprobs_llamafactory.py

Without the variable the test is skipped (no tokenizer download is needed: the tiny one is used).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from peft import PeftModel, get_peft_model

from tests.support.autodl_scripts import AUTODL
from tests.support.tiny_lm import tiny_base, write_tiny_base
from tests.unit.test_autodl_pair_logprobs import PAIRS, lora, train_like
from twin.training import lf_template

pytestmark = pytest.mark.integration

PYTHON = os.environ.get("TWIN_LLAMAFACTORY_PYTHON")
needs_environment = pytest.mark.skipif(
    not PYTHON, reason="set TWIN_LLAMAFACTORY_PYTHON (see the module docstring)"
)

LF_SIDE = r"""
import json
import sys
from pathlib import Path

import torch
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.trainer_utils import get_batch_logps

base, adapters, pairs_path, tool, out = sys.argv[1:6]
model_args, data_args, finetuning_args, _ = get_infer_args(
    {
        "model_name_or_path": base,
        "adapter_name_or_path": adapters,
        "template": "qwen3_nothink",
        "finetuning_type": "lora",
        "infer_dtype": "float32",
        "flash_attn": "sdpa",
    }
)
tokenizer = load_tokenizer(model_args)["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
roles = {"human": "user", "gpt": "assistant"}
found = []
for line in Path(pairs_path).read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    prompt = [{"role": roles[m["from"]], "content": m["value"]} for m in row["conversations"]]
    row_scores = {"id": row["id"]}
    for key in ("chosen", "rejected"):
        messages = prompt + [{"role": "assistant", "content": row[key]["value"]}]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, row["system"])
        ids = torch.tensor([prompt_ids + response_ids], device=model.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + response_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.to(torch.float32)
        logps, length = get_batch_logps(logits=logits, labels=labels)
        row_scores[key + "_logp"] = float(logps[0])
        row_scores[key + "_tokens"] = int(length[0])
    found.append(row_scores)
sys.path.insert(0, str(Path(tool).parent))
import pair_logprobs
code = pair_logprobs.main(
    ["--pairs", pairs_path, "--base", base, "--adapters", adapters, "--dtype", "float32",
     "--out", out]
)
print(json.dumps({"llamafactory": found, "tool_exit": code}))
"""


@needs_environment
def test_the_tool_scores_the_pairs_as_llamafactory_trains_on_them(tmp_path: Path) -> None:
    assert PYTHON is not None
    base = write_tiny_base(tmp_path / "base")
    sft = get_peft_model(tiny_base(), lora(4, 8))
    train_like(sft, 11)
    sft.save_pretrained(str(tmp_path / "sft"))
    merged = PeftModel.from_pretrained(tiny_base(), str(tmp_path / "sft")).merge_and_unload()
    layered = get_peft_model(merged, lora(2, 6))
    train_like(layered, 22)
    layered.save_pretrained(str(tmp_path / "dpo"))
    pairs = tmp_path / "dpo_holdout.jsonl"
    pairs.write_text(
        "".join(json.dumps(p.to_json(), ensure_ascii=False) + "\n" for p in PAIRS), "utf-8"
    )
    script = tmp_path / "lf_side.py"
    script.write_text(LF_SIDE, encoding="utf-8")
    out = tmp_path / "pair_logps.jsonl"
    pylib = str(Path(lf_template.__file__).parents[2])
    done = subprocess.run(
        [
            PYTHON,
            str(script),
            str(base),
            f"{tmp_path / 'sft'},{tmp_path / 'dpo'}",
            str(pairs),
            str(AUTODL / "tools" / "pair_logprobs.py"),
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": pylib},
        check=False,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["tool_exit"] == 0
    tool = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    expected = result["llamafactory"]
    assert [row["id"] for row in tool] == [row["id"] for row in expected] == ["p0", "p1", "p2"]
    for ours, theirs in zip(tool, expected, strict=True):
        assert ours["chosen_tokens"] == theirs["chosen_tokens"]
        assert ours["rejected_tokens"] == theirs["rejected_tokens"]
        assert ours["chosen_logp"] == pytest.approx(theirs["chosen_logp"], abs=1e-4)
        assert ours["rejected_logp"] == pytest.approx(theirs["rejected_logp"], abs=1e-4)
