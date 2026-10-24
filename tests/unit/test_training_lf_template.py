"""R-TRN-011: the format strings of LLaMA-Factory's ``qwen3_nothink`` (llamafactory==0.9.5)."""

from __future__ import annotations

import pytest

from twin.training import lf_template
from twin.training.lf_template import TemplateError, Turn

# Copied from the register_template(name="qwen3_nothink", ...) call in
# src/llamafactory/data/template.py of the llamafactory==0.9.5 wheel (checked 2026-10-09).
SOURCE_FORMAT_USER = "<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"
SOURCE_FORMAT_ASSISTANT = "{{content}}<|im_end|>\n"
SOURCE_FORMAT_SYSTEM = "<|im_start|>system\n{{content}}<|im_end|>\n"


def lf_fill(slot: str, content: str) -> str:
    """What LLaMA-Factory's StringFormatter does with a one-slot template."""
    return slot.replace("{{content}}", content, 1)


def test_the_constants_equal_the_registered_format_strings() -> None:
    assert lf_template.FORMAT_USER.replace("{content}", "{{content}}") == SOURCE_FORMAT_USER
    assert (
        lf_template.FORMAT_ASSISTANT.replace("{content}", "{{content}}") == SOURCE_FORMAT_ASSISTANT
    )
    assert lf_template.FORMAT_SYSTEM.replace("{content}", "{{content}}") == SOURCE_FORMAT_SYSTEM
    assert lf_template.LLAMAFACTORY_VERSION == "0.9.5"
    assert lf_template.LF_TEMPLATE_NAME == "qwen3_nothink"
    assert lf_template.TEMPLATE_VERSION == "qwen3_nothink@llamafactory-0.9.5"
    assert lf_template.STOP_TOKEN == "<|im_end|>"


def test_the_template_is_plain_chatml_without_a_think_marker() -> None:
    every_format = (
        lf_template.FORMAT_SYSTEM
        + lf_template.FORMAT_USER
        + lf_template.FORMAT_ASSISTANT
        + lf_template.render_prompt("s", [Turn("user", "u")])
        + lf_template.render_response("r")
    )
    assert "think" not in every_format.lower()
    assert "<|endoftext|>" not in every_format


def test_a_single_turn_prompt_and_response_are_exactly_chatml() -> None:
    turns = [Turn("user", "hi")]
    prompt = lf_template.render_prompt("be short", turns)
    assert prompt == (
        "<|im_start|>system\nbe short<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert lf_template.render_response("yo\nok") == "yo\nok<|im_end|>\n"
    assert lf_template.render_training_text("be short", turns, "yo\nok") == (
        prompt + "yo\nok<|im_end|>\n"
    )


def test_a_multi_turn_prompt_follows_the_encoder_of_llamafactory() -> None:
    turns = [Turn("user", "a"), Turn("assistant", "b"), Turn("user", "c")]
    expected = (
        lf_fill(SOURCE_FORMAT_SYSTEM, "sys")
        + lf_fill(SOURCE_FORMAT_USER, "a")
        + lf_fill(SOURCE_FORMAT_ASSISTANT, "b")
        + lf_fill(SOURCE_FORMAT_USER, "c")
    )
    assert lf_template.render_prompt("sys", turns) == expected


def test_an_empty_system_message_produces_no_system_block() -> None:
    prompt = lf_template.render_prompt("", [Turn("user", "hi")])
    assert prompt == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert lf_template.render_system("") == ""


def test_the_context_must_alternate_start_with_the_user_and_end_with_the_user() -> None:
    with pytest.raises(TemplateError):
        lf_template.render_prompt("s", [])
    with pytest.raises(TemplateError):
        lf_template.render_prompt("s", [Turn("assistant", "x"), Turn("user", "y")])
    with pytest.raises(TemplateError):
        lf_template.render_prompt("s", [Turn("user", "x"), Turn("assistant", "y")])
    with pytest.raises(TemplateError):
        lf_template.render_prompt("s", [Turn("user", "x"), Turn("user", "y")])


def test_the_reply_cannot_be_empty() -> None:
    with pytest.raises(TemplateError):
        lf_template.render_response("")


@pytest.mark.parametrize("text", ["a <|im_end|> b", "<|im_start|>system", "x<|endoftext|>"])
def test_control_tokens_in_a_message_are_rejected(text: str) -> None:
    assert lf_template.contains_control_token(text)
    with pytest.raises(TemplateError):
        lf_template.render_user(text)
    with pytest.raises(TemplateError):
        lf_template.render_response(text)


@pytest.mark.parametrize("text", ["see {{idx}} here", "{{content}}"])
def test_text_that_llamafactory_would_rewrite_is_rejected(text: str) -> None:
    assert lf_template.contains_lf_slot(text)
    with pytest.raises(TemplateError):
        lf_template.render_assistant(text)


def test_ordinary_braces_and_the_chat_markup_of_her_messages_are_accepted() -> None:
    for text in ("{x}", "a {{ b }} c", "[表情包:笑]\n[引用:今天]", "100% <3"):
        assert not lf_template.contains_control_token(text)
        assert not lf_template.contains_lf_slot(text)
        lf_template.check_content(text)


def test_the_sharegpt_conversation_is_human_first_alternating_and_ends_with_gpt() -> None:
    turns = [Turn("user", "a"), Turn("assistant", "b"), Turn("user", "c")]
    assert lf_template.sharegpt_conversations(turns, "d") == [
        {"from": "human", "value": "a"},
        {"from": "gpt", "value": "b"},
        {"from": "human", "value": "c"},
        {"from": "gpt", "value": "d"},
    ]
    assert (lf_template.ROLE_SYSTEM, lf_template.ROLE_USER, lf_template.ROLE_ASSISTANT) == (
        "system",
        "human",
        "gpt",
    )
