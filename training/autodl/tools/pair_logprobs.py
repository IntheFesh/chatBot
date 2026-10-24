#!/usr/bin/env python3
"""Score the held-out preference pairs under one adapter stack (R-TRN-009; D-862, D-865).

    PYTHONPATH=<workdir>/pylib pair_logprobs.py --pairs data/dpo_holdout.jsonl \
        --base /root/autodl-tmp/models/Qwen/Qwen3-8B --adapters output/sft[,output/dpo] \
        [--quantization-bit 4] [--dtype bfloat16] --out artifacts/eval/<stack>/pair_logps.jsonl

``twin train export-dpo`` keeps a fifth of the preference pairs out of the DPO training
(``dpo_holdout.jsonl``, the same ShareGPT form as ``dpo_train.jsonl``).  ``eval.sh`` runs this tool
once per adapter stack; ``twin train remote eval`` compares the two stacks on its output
(``twin.training.stack_choice``).  For every pair it writes one line - the pair's id and four
numbers, never a text::

    {"id": ..., "chosen_logp": ..., "rejected_logp": ...,
     "chosen_tokens": ..., "rejected_tokens": ...}

``*_logp`` is the sum of the log probabilities of the reply's tokens after the prompt, ``*_tokens``
their number.

*The sequence* is the one LLaMA-Factory 0.9.5 builds for the pair when it trains DPO
(``data/processor/pairwise.py``): the prompt ids of ``encode_oneturn`` - the system block, every
earlier turn and the last user turn with the assistant opener, each slot tokenised on its own
(``data/template.py::_convert_elements_to_ids``) - followed by the reply's ids, the tokens of
``<reply><|im_end|>\\n`` (``efficient_eos`` is off for ``qwen3_nothink``).  The slots are rendered
by the project's one implementation of that template, ``twin.training.lf_template`` (shipped in
``pylib/``; R-TRN-011's check proves it equal to LLaMA-Factory's).  The log probability is summed
over the reply's ids, as ``train/trainer_utils.py::get_batch_logps`` does for ``pref_loss:
sigmoid`` (logits in float32).  Nothing is cut: the export keeps replies under 256 tokens and the
system segment inside the style budget, so the training did not cut these pairs either.

*The model* is loaded the way LLaMA-Factory loads it for the evaluation (``model/adapter.py``,
not trainable; ``model/model_utils/quantization.py``): the base in bfloat16 with ``sdpa``
attention, or - when the profile quantises - in 4 bit (bitsandbytes NF4, double quantisation,
compute and storage in bfloat16); then every adapter of the stack is merged into it in the order
given (``PeftModel.from_pretrained(...).merge_and_unload()``).  A quantised base takes exactly one
adapter, which stays unmerged.

Needs ``torch``, ``transformers`` and ``peft`` (the instance has them once the packages are
installed).  Error messages never contain a text of the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twin.training import lf_template

DTYPES = ("bfloat16", "float16", "float32")
ROLES = {lf_template.ROLE_USER: "user", lf_template.ROLE_ASSISTANT: "assistant"}


class PairFileError(ValueError):
    """The held-out pairs cannot be read (the message names a line, never its text)."""


@dataclass(frozen=True)
class Pair:
    """One held-out preference pair: the context and the two replies."""

    id: str
    system: str
    turns: tuple[lf_template.Turn, ...]
    chosen: str
    rejected: str


def read_pairs(path: Path) -> list[Pair]:
    """The pairs of a ``dpo_holdout.jsonl`` in file order."""
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                turns = tuple(
                    lf_template.Turn(ROLES[message["from"]], str(message["value"]))
                    for message in row["conversations"]
                )
                pairs.append(
                    Pair(
                        str(row["id"]),
                        str(row.get("system") or ""),
                        turns,
                        str(row["chosen"]["value"]),
                        str(row["rejected"]["value"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                raise PairFileError(
                    f"line {number} of {path.name} is not a preference pair"
                ) from None
    return pairs


def encode_pair(tokenizer: Any, pair: Pair) -> tuple[list[int], list[int], list[int]]:
    """``(prompt ids, chosen ids, rejected ids)`` as LLaMA-Factory's DPO training builds them."""

    def ids(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    try:
        lf_template.check_alternation(pair.turns)
        prompt = ids(lf_template.render_system(pair.system)) if pair.system else []
        for turn in pair.turns:
            render = (
                lf_template.render_user if turn.role == "user" else lf_template.render_assistant
            )
            prompt.extend(ids(render(turn.content)))
        chosen = ids(lf_template.render_response(pair.chosen))
        rejected = ids(lf_template.render_response(pair.rejected))
    except lf_template.TemplateError as exc:
        raise PairFileError(f"pair {pair.id} cannot be rendered: {exc}") from None
    return prompt, chosen, rejected


def response_logprob(model: Any, prompt: list[int], response: list[int]) -> float:
    """The sum of ``log p(response token | everything before it)`` over the response."""
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits.to(torch.float32)
    logprobs = logits[0, :-1].log_softmax(-1)
    start = len(prompt) - 1
    targets = ids[0, 1:][start:].unsqueeze(1)
    return float(logprobs[start:].gather(1, targets).sum())


def load_stack(
    base: str,
    adapters: list[str],
    *,
    quantization_bit: int | None,
    dtype: str,
    attention: str = "sdpa",
) -> Any:
    """The base with the stack's adapters, loaded as LLaMA-Factory's evaluation loads it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    compute = getattr(torch, dtype)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    options: dict[str, Any] = {"dtype": compute, "attn_implementation": attention}
    if quantization_bit is not None:
        if quantization_bit != 4:
            raise ValueError(f"a {quantization_bit}-bit base is not a profile of this project")
        if len(adapters) != 1:
            raise ValueError("a quantised base takes exactly one adapter (LLaMA-Factory's rule)")
        from transformers import BitsAndBytesConfig

        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=compute,
        )
        options["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(base, **options)
    if quantization_bit is None:
        model = model.to(device)
        for adapter in adapters:
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    else:
        model = PeftModel.from_pretrained(model, adapters[0])
    model.eval()
    return model


def score_pairs(model: Any, tokenizer: Any, pairs: list[Pair]) -> list[dict[str, Any]]:
    """One row of numbers per pair (see the module description)."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        prompt, chosen, rejected = encode_pair(tokenizer, pair)
        rows.append(
            {
                "id": pair.id,
                "chosen_logp": response_logprob(model, prompt, chosen),
                "rejected_logp": response_logprob(model, prompt, rejected),
                "chosen_tokens": len(chosen),
                "rejected_tokens": len(rejected),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="comma-separated, in the stack's order")
    parser.add_argument("--quantization-bit", type=int, default=None)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    adapters = [part for part in args.adapters.split(",") if part]
    try:
        pairs = read_pairs(args.pairs)
        if not pairs:
            raise PairFileError(f"{args.pairs.name} holds no pairs")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base)
        model = load_stack(
            args.base,
            adapters,
            quantization_bit=args.quantization_bit,
            dtype=args.dtype,
            attention=args.attention,
        )
        rows = score_pairs(model, tokenizer, pairs)
    except (OSError, ValueError, ImportError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    write_rows(args.out, rows)
    sys.stdout.write(
        f"scored {len(rows)} held-out preference pairs under a stack of {len(adapters)} "
        "adapter(s)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
