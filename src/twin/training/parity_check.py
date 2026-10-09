"""Training = inference, token for token (R-TRN-011): the check that runs on the instance.

What the style model is trained on and what it is shown must be the same sequence of token ids.
Training builds it from the ShareGPT sample with LLaMA-Factory (``qwen3_nothink``, ``mask_history``,
``cutoff_len`` 2048); the running bot renders the prompt with ``StylePromptBuilder`` and the
inference server tokenises that string.  For every case of ``parity_cases.jsonl`` (a sample of the
exported set with ``prompt``, the string inference would send for its context) this check asserts:

1. **The prompt.**  The tokens of ``prompt`` are all the tokens of the turns before the last one
   (prompt and response of each, as LLaMA-Factory encodes them) followed by the prompt tokens of
   the last turn.  The string is also the one :mod:`twin.training.lf_template` renders for the
   conversation.
2. **The response.**  The response tokens of the last turn decode to the reply, ``<|im_end|>`` and
   the newline of the template, and nothing else.
3. **The loss.**  What LLaMA-Factory's processor makes of the sample has the whole sequence as
   ``input_ids`` and, with ``mask_history``, labels for the last response only; every other
   position is ``-100``.
4. **The length.**  The sequence is not longer than ``cutoff_len``, so nothing is cut.

It does *not* compare with ``tokenizer.apply_chat_template``: the official Qwen3 template writes an
empty think block before the last answer, which this training format deliberately does not have.

Two engines.  ``llamafactory`` (the default, what the instance runs) calls the installed
LLaMA-Factory: ``get_template_and_fix_tokenizer``, the ShareGPT converter, ``encode_multiturn`` and
the supervised processor.  ``slots`` re-implements the same encoding rules from
:mod:`twin.training.lf_template` for environments without LLaMA-Factory; it proves the
construction (pieces encoded one by one equal the whole prompt encoded at once), not the
LLaMA-Factory release, and says so in its output.

Usage (exit code 0 = every case passes, 1 = a case differs, 2 = bad arguments)::

    python -m twin.training.parity_check --model-dir <base model folder> --cases <cases.jsonl>

This module imports nothing from the project but :mod:`twin.training.lf_template`, because the
instance has only the modules listed in ``twin.training.bundle.PYLIB_MODULES``.  The output holds
ids, counts and token positions - never the text of a sample.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from twin.training import lf_template

EXIT_OK: Final = 0
EXIT_DIFFERENT: Final = 1
EXIT_USAGE: Final = 2

CUTOFF_LEN: Final = 2048  # profiles.CUTOFF_LEN (a test compares them; this module stays standalone)
IGNORE_INDEX: Final = -100
TEMPLATE: Final = lf_template.LF_TEMPLATE_NAME
ENGINES: Final = ("llamafactory", "slots")
TOKENIZER_FILE: Final = "tokenizer.json"


class TokenizerLike(Protocol):
    """The two methods of a Hugging Face tokenizer that LLaMA-Factory uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


class ParityError(RuntimeError):
    """The check cannot run (a missing file or package); not a verdict on the cases."""


Pair = tuple[list[int], list[int]]


@dataclass
class Case:
    """One line of ``parity_cases.jsonl``."""

    id: str
    system: str
    conversations: list[dict[str, str]]
    prompt: str
    tags: list[str] = field(default_factory=list)

    @property
    def reply(self) -> str:
        return self.conversations[-1]["value"]

    def turns(self) -> list[lf_template.Turn]:
        """The conversation before the reply, as :class:`~twin.training.lf_template.Turn`."""
        return [
            lf_template.Turn("user" if index % 2 == 0 else "assistant", message["value"])
            for index, message in enumerate(self.conversations[:-1])
        ]


@dataclass
class CaseResult:
    case_id: str
    tags: list[str]
    tokens: int
    problems: list[str]

    @property
    def passed(self) -> bool:
        return not self.problems


# ------------------------------------------------------------------------------ the engines


class Engine(Protocol):
    """How a sample becomes token ids on the training side."""

    name: str

    def pairs(self, case: Case) -> list[Pair]: ...

    def training(self, case: Case) -> tuple[list[int], list[int]]: ...


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """LLaMA-Factory's rule for cutting a pair that does not fit (``processor_utils``)."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    new_source_len = min(max(cutoff_len - new_target_len, 0), source_len)
    return new_source_len, new_target_len


class SlotEngine:
    """The ``qwen3_nothink`` encoding written out from :mod:`twin.training.lf_template`.

    Every slot of the template is tokenised on its own and the ids are concatenated, as
    ``Template._convert_elements_to_ids`` does; the training example follows the supervised
    processor with ``mask_history`` (newest pair first, only that pair has labels).
    """

    name = "slots"

    def __init__(self, tokenizer: TokenizerLike, cutoff_len: int = CUTOFF_LEN) -> None:
        self._tokenizer = tokenizer
        self._cutoff = cutoff_len

    def _ids(self, slot: str) -> list[int]:
        return list(self._tokenizer.encode(slot, add_special_tokens=False)) if slot else []

    def pairs(self, case: Case) -> list[Pair]:
        encoded: list[list[int]] = []
        for index, message in enumerate(case.conversations):
            ids: list[int] = []
            if index == 0 and case.system:
                ids += self._ids(lf_template.render_system(case.system))
            if message["from"] == lf_template.ROLE_USER:
                ids += self._ids(lf_template.render_user(message["value"]))
            else:
                ids += self._ids(lf_template.render_assistant(message["value"]))
            encoded.append(ids)
        return [(encoded[i], encoded[i + 1]) for i in range(0, len(encoded), 2)]

    def training(self, case: Case) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        labels: list[int] = []
        total = 0
        for turn_index, (source, target) in enumerate(reversed(self.pairs(case))):
            if total >= self._cutoff:
                break
            source_len, target_len = infer_seqlen(len(source), len(target), self._cutoff - total)
            source, target = source[:source_len], target[:target_len]
            total += source_len + target_len
            target_labels = [IGNORE_INDEX] * target_len if turn_index else list(target)
            input_ids = source + target + input_ids
            labels = [IGNORE_INDEX] * source_len + target_labels + labels
        return input_ids, labels


class LlamaFactoryEngine:
    """The installed LLaMA-Factory, called the way the training run calls it."""

    name = "llamafactory"

    def __init__(self, tokenizer: Any, cutoff_len: int = CUTOFF_LEN) -> None:
        try:
            template_module = importlib.import_module("llamafactory.data.template")
            converter_module = importlib.import_module("llamafactory.data.converter")
            parser_module = importlib.import_module("llamafactory.data.parser")
            processor_module = importlib.import_module("llamafactory.data.processor.supervised")
            hparams = importlib.import_module("llamafactory.hparams")
        except ImportError as exc:
            raise ParityError(
                f"LLaMA-Factory cannot be imported ({exc.name}); run this check in the "
                "environment that trains, or use --engine slots"
            ) from exc
        self._tokenizer = tokenizer
        self._data_args = hparams.DataArguments(
            template=TEMPLATE, cutoff_len=cutoff_len, mask_history=True
        )
        self.template = template_module.get_template_and_fix_tokenizer(tokenizer, self._data_args)
        attribute = parser_module.DatasetAttr("file", "parity_cases.jsonl")
        attribute.join(
            {
                "formatting": "sharegpt",
                "columns": {"messages": "conversations", "system": "system"},
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": lf_template.ROLE_USER,
                    "assistant_tag": lf_template.ROLE_ASSISTANT,
                    "system_tag": lf_template.ROLE_SYSTEM,
                },
            }
        )
        self._converter = converter_module.get_dataset_converter(
            "sharegpt", attribute, self._data_args
        )
        self._processor = processor_module.SupervisedDatasetProcessor(
            template=self.template,
            tokenizer=tokenizer,
            processor=None,
            data_args=self._data_args,
        )

    def _converted(self, case: Case) -> dict[str, Any]:
        converted: dict[str, Any] = self._converter(
            {"system": case.system, "conversations": case.conversations}
        )
        return converted

    def pairs(self, case: Case) -> list[Pair]:
        converted = self._converted(case)
        discard = self._data_args.mask_history and not getattr(
            self.template, "preserve_thinking", False
        )
        encoded = self.template.encode_multiturn(
            self._tokenizer,
            converted["_prompt"] + converted["_response"],
            converted["_system"],
            converted["_tools"],
            discard,
        )
        return [(list(source), list(target)) for source, target in encoded]

    def training(self, case: Case) -> tuple[list[int], list[int]]:
        converted = self._converted(case)
        columns = {
            "_prompt": [converted["_prompt"]],
            "_response": [converted["_response"]],
            "_system": [converted["_system"]],
            "_tools": [converted["_tools"]],
            "_images": [None],
            "_videos": [None],
            "_audios": [None],
        }
        made = self._processor.preprocess_dataset(columns)
        if not made["input_ids"]:
            raise ParityError("LLaMA-Factory dropped the sample as invalid")
        return list(made["input_ids"][0]), list(made["labels"][0])

    def template_facts(self) -> list[str]:
        """Problems with the template as LLaMA-Factory resolved it (the stop word, the EOS)."""
        problems: list[str] = []
        if list(self.template.stop_words) != [lf_template.STOP_TOKEN]:
            problems.append("the template does not stop at <|im_end|>")
        if self._tokenizer.eos_token != lf_template.STOP_TOKEN:
            problems.append("the tokenizer's end-of-sequence token is not <|im_end|>")
        return problems


# --------------------------------------------------------------------------------- the checks


def _first_difference(left: Sequence[int], right: Sequence[int]) -> str:
    for position, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return f"first difference at token {position}: {a} against {b}"
    return f"one is a prefix of the other ({len(left)} and {len(right)} tokens)"


def check_case(case: Case, tokenizer: TokenizerLike, engine: Engine, cutoff_len: int) -> CaseResult:
    """All checks of one case; the result lists what differs (it is empty when the case passes)."""
    try:
        return _check_case(case, tokenizer, engine, cutoff_len)
    except lf_template.TemplateError as exc:
        return CaseResult(case.id, case.tags, 0, [f"the sample cannot be rendered: {exc}"])


def _check_case(
    case: Case, tokenizer: TokenizerLike, engine: Engine, cutoff_len: int
) -> CaseResult:
    problems: list[str] = []
    expected_prompt = lf_template.render_prompt(case.system, case.turns())
    if case.prompt != expected_prompt:
        problems.append("the prompt of the case is not the template rendering of its conversation")

    pairs = engine.pairs(case)
    if len(pairs) * 2 != len(case.conversations):
        problems.append(f"{len(pairs)} turn pair(s) for {len(case.conversations)} messages")
        return CaseResult(case.id, case.tags, 0, problems)

    earlier = [token for source, target in pairs[:-1] for token in (*source, *target)]
    expected_ids = earlier + pairs[-1][0]
    actual_ids = tokenizer.encode(case.prompt, add_special_tokens=False)
    if actual_ids != expected_ids:
        problems.append(
            "the inference prompt does not tokenise like the training sequence: "
            + _first_difference(actual_ids, expected_ids)
        )

    response = pairs[-1][1]
    wanted = lf_template.render_response(case.reply)
    decoded = tokenizer.decode(response, skip_special_tokens=False)
    if decoded != wanted:
        problems.append("the response tokens do not decode to the reply, <|im_end|> and a newline")
    if tokenizer.encode(wanted, add_special_tokens=False) != response:
        problems.append("the response is tokenised differently than the reply on its own")

    input_ids, labels = engine.training(case)
    sequence = [token for source, target in pairs for token in (*source, *target)]
    if input_ids != sequence:
        problems.append(
            "the training sequence is not the pairs in order: "
            + _first_difference(input_ids, sequence)
        )
    masked = len(input_ids) - len(response)
    if labels != [IGNORE_INDEX] * masked + response:
        problems.append("with mask_history only the last response must carry labels")
    if len(input_ids) > cutoff_len:
        problems.append(f"{len(input_ids)} tokens: longer than cutoff_len {cutoff_len}")
    return CaseResult(case.id, case.tags, len(input_ids), problems)


def run_checks(
    cases: Sequence[Case], tokenizer: TokenizerLike, engine: Engine, cutoff_len: int = CUTOFF_LEN
) -> list[CaseResult]:
    return [check_case(case, tokenizer, engine, cutoff_len) for case in cases]


# ------------------------------------------------------------------------------ files and CLI


def load_cases(path: Path) -> list[Case]:
    if not path.is_file():
        raise ParityError(f"{path} does not exist")
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cases.append(
                    Case(
                        id=str(row["id"]),
                        system=str(row.get("system", "")),
                        conversations=[
                            {"from": str(m["from"]), "value": str(m["value"])}
                            for m in row["conversations"]
                        ],
                        prompt=str(row["prompt"]),
                        tags=[str(tag) for tag in row.get("tags", [])],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ParityError(f"{path.name} line {number} is not a case ({exc})") from None
    if not cases:
        raise ParityError(f"{path} holds no case")
    return cases


class _FileTokenizer:
    """``tokenizer.json`` behind the two methods the checks use (the ``slots`` engine only)."""

    def __init__(self, path: Path) -> None:
        tokenizers = importlib.import_module("tokenizers")
        self._tokenizer = tokenizers.Tokenizer.from_file(str(path))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))


def open_engine(name: str, model_dir: Path, cutoff_len: int) -> tuple[TokenizerLike, Engine]:
    """The tokenizer of ``model_dir`` and the engine named ``name``."""
    if name == "slots":
        file = model_dir / TOKENIZER_FILE if model_dir.is_dir() else model_dir
        if not file.is_file():
            raise ParityError(f"{file} does not exist")
        tokenizer: TokenizerLike = _FileTokenizer(file)
        return tokenizer, SlotEngine(tokenizer, cutoff_len)
    if importlib.util.find_spec("llamafactory") is None:
        raise ParityError(
            "LLaMA-Factory is not installed in this environment; run this check where the "
            "model is trained, or use --engine slots"
        )
    try:
        transformers = importlib.import_module("transformers")
        hub_tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_dir), use_fast=True, split_special_tokens=False, padding_side="right"
        )
    except Exception as exc:
        raise ParityError(
            f"the tokenizer in {model_dir} cannot be loaded ({type(exc).__name__})"
        ) from exc
    return hub_tokenizer, LlamaFactoryEngine(hub_tokenizer, cutoff_len)


def report(results: Sequence[CaseResult], engine: Engine) -> list[str]:
    """The lines to print: one per failing case, then the summary (no text of any sample)."""
    lines = [
        f"FAILED {result.case_id} [{', '.join(result.tags)}]: {problem}"
        for result in results
        for problem in result.problems
    ]
    failed = sum(1 for result in results if not result.passed)
    longest = max((result.tokens for result in results), default=0)
    lines.append(
        f"template check ({engine.name}): {len(results) - failed} of {len(results)} case(s) "
        f"token-identical, longest {longest} tokens"
    )
    if engine.name == "slots":
        lines.append(
            "note: the slots engine does not call LLaMA-Factory; it checks the construction"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin.training.parity_check")
    parser.add_argument("--model-dir", type=Path, required=True, help="folder of the base model")
    parser.add_argument("--cases", type=Path, required=True, help="parity_cases.jsonl")
    parser.add_argument("--engine", choices=ENGINES, default="llamafactory")
    parser.add_argument("--cutoff-len", type=int, default=CUTOFF_LEN)
    parser.add_argument("--limit", type=int, default=0, help="check only the first N cases")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    try:
        cases = load_cases(args.cases)
        if args.limit > 0:
            cases = cases[: args.limit]
        tokenizer, engine = open_engine(args.engine, args.model_dir, args.cutoff_len)
        results = run_checks(cases, tokenizer, engine, args.cutoff_len)
        facts = getattr(engine, "template_facts", None)
        extra = list(facts()) if callable(facts) else []
    except ParityError as exc:
        sys.stderr.write(f"template check cannot run: {exc}\n")
        return EXIT_USAGE
    for problem in extra:
        sys.stdout.write(f"FAILED template: {problem}\n")
    for line in report(results, engine):
        sys.stdout.write(line + "\n")
    return EXIT_OK if all(r.passed for r in results) and not extra else EXIT_DIFFERENT


if __name__ == "__main__":
    sys.exit(main())
