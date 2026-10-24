"""The tokenizer comparison before a model is used (R-TRN-011.4, R-LLM-011)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.support.style_models import ScriptedStyleClient
from tests.support.tiny_tokenizer import build_merging_tokenizer, tiny_qwen_tokenizer
from twin.llm.errors import StyleModelError
from twin.llm.redaction import redact_text
from twin.serving.tokencheck import (
    MAX_REPORTED,
    SamplePrompt,
    TokenizationReport,
    TokenizeCheckError,
    check_tokenization,
    compare_ids,
    sample_prompts,
)
from twin.training import lf_template
from twin.training.tokenizer import QwenTokenizer

TOKENIZER = tiny_qwen_tokenizer()
BOS = 7


class ServerLike(ScriptedStyleClient):
    """A style client whose ``tokenize`` is a function the test chooses."""

    def __init__(self, tokenize: Callable[[str], list[int]]) -> None:
        super().__init__()
        self._tokenize = tokenize

    async def tokenize(self, text: str) -> list[int]:
        return self._tokenize(text)


def like_the_trainer(text: str) -> list[int]:
    return TOKENIZER.encode(text)


async def test_a_server_that_tokenizes_like_the_trainer_passes() -> None:
    report = await check_tokenization(ServerLike(like_the_trainer), TOKENIZER)
    assert report.ok and report.samples == len(sample_prompts()) and report.tokens > 500
    assert report.lines() == [
        f"tokenization matches: {report.samples} prompts, {report.tokens} tokens"
    ]


async def test_a_bos_token_in_front_is_found_at_position_zero_in_every_prompt() -> None:
    server = ServerLike(lambda text: [BOS, *TOKENIZER.encode(text)])
    report = await check_tokenization(server, TOKENIZER)
    assert not report.ok and len(report.differences) == report.samples
    first = report.differences[0]
    assert first.kind == "extra_prefix" and first.position == 0 and first.server_id == BOS
    assert first.local_id == TOKENIZER.token_id(lf_template.IM_START)
    assert first.server_count == first.local_count + 1
    assert "BOS" in first.line() and "token 0" in first.line()


async def test_control_tokens_that_fall_apart_are_found() -> None:
    def split(text: str) -> list[int]:
        for token in lf_template.CONTROL_TOKENS:
            text = text.replace(token, token.replace("|", "│"))
        return TOKENIZER.encode(text)

    report = await check_tokenization(ServerLike(split), TOKENIZER)
    assert not report.ok
    assert {d.kind for d in report.differences} == {"split_special"}
    assert report.differences[0].position == 0
    assert "control token was split" in report.differences[0].line()


async def test_a_different_vocabulary_is_a_plain_difference_with_the_position() -> None:
    other = QwenTokenizer(build_merging_tokenizer(), "1" * 64)
    report = await check_tokenization(ServerLike(other.encode), TOKENIZER)
    assert not report.ok
    kinds = {d.kind for d in report.differences}
    assert kinds <= {"different", "length", "split_special"}
    assert all(d.position >= 0 for d in report.differences)


async def test_ids_that_stop_early_or_start_late_are_told_apart() -> None:
    cut = ServerLike(lambda text: TOKENIZER.encode(text)[:-2])
    report = await check_tokenization(cut, TOKENIZER)
    assert {d.kind for d in report.differences} == {"length"}
    assert report.differences[0].server_id is None
    dropped = ServerLike(lambda text: TOKENIZER.encode(text)[1:])
    kinds = {d.kind for d in (await check_tokenization(dropped, TOKENIZER)).differences}
    assert kinds == {"missing_prefix"}


async def test_the_report_names_the_prompts_and_cuts_a_long_list() -> None:
    server = ServerLike(lambda text: [BOS, *TOKENIZER.encode(text)])
    report = await check_tokenization(server, TOKENIZER)
    lines = report.lines()
    assert lines[0] == f"tokenization differs in {report.samples} of {report.samples} prompts:"
    assert len(lines) == 1 + MAX_REPORTED + 1 and lines[-1].startswith("... and ")
    assert lines[1].startswith(report.differences[0].sample + ":")


async def test_a_server_that_cannot_tokenize_is_not_a_mismatch_but_an_error() -> None:
    class Broken(ServerLike):
        async def tokenize(self, text: str) -> list[int]:
            raise StyleModelError("connection refused", kind="unavailable")

    with pytest.raises(TokenizeCheckError, match="could not tokenize 'plain'"):
        await check_tokenization(Broken(like_the_trainer), TOKENIZER)


async def test_given_prompts_replace_the_standard_set() -> None:
    prompts = [
        SamplePrompt("one", lf_template.render_prompt("", [lf_template.Turn("user", "你好")]))
    ]
    report = await check_tokenization(ServerLike(like_the_trainer), TOKENIZER, prompts)
    assert report.ok and report.samples == 1


def test_the_standard_prompts_are_ChatML_of_the_training_template_with_every_kind_of_text() -> None:
    prompts = sample_prompts()
    names = [p.name for p in prompts]
    assert names == [
        "plain",
        "chat",
        "stickers_and_codes",
        "quote_and_plan",
        "mixed_scripts",
        "whitespace",
        "long",
    ]
    for prompt in prompts:
        assert prompt.text.startswith(lf_template.SYSTEM_OPEN)
        assert prompt.text.endswith(lf_template.ASSISTANT_OPEN)
        assert "<think>" not in prompt.text
    joined = "\n".join(p.text for p in prompts)
    for expected in (
        "[表情包:",
        "[微笑]",
        "[引用:",
        "【规划】",
        "【前文】",
        "😀",
        "https://",
        "\t",
    ):
        assert expected in joined, expected
    assert len(prompts[-1].text) > 1500  # the long conversation


def test_the_prompts_hold_no_personal_data_the_outbound_filter_would_change() -> None:
    """The remote client redacts what it sends; a sample it would change could never match."""
    for prompt in sample_prompts():
        assert redact_text(prompt.text) == prompt.text, prompt.name


def test_the_comparison_of_equal_ids_has_no_difference() -> None:
    assert compare_ids(TOKENIZER, "x", [1, 2, 3], [1, 2, 3]) is None
    found = compare_ids(TOKENIZER, "x", [1, 2, 3], [1, 9, 3])
    assert found is not None and found.position == 1 and found.kind == "different"
    assert found.to_json()["kind"] == "different" and found.to_json()["server_id"] == 9
    assert TokenizationReport(1, 3).ok
