"""When a chat reply thinks, and what counts as a refusal (R-LLM-002, R-LLM-008, R-SAFE-005)."""

from __future__ import annotations

import pytest

from twin.engine.refusal import looks_like_refusal
from twin.engine.thinking import LONG_TEXT_CHARS, resolve_thinking, wants_thinking


@pytest.mark.parametrize(
    "text",
    [
        "你吃饭了吗",
        "周末去哪里玩？",
        "明天几点见",
        "我今天好难过",
        "气死我了！！",
        "好想你",
        "字" * (LONG_TEXT_CHARS + 1),
    ],
)
def test_questions_feelings_and_long_messages_make_auto_think(text: str) -> None:
    assert wants_thinking(text)


@pytest.mark.parametrize(
    "text", ["好的", "哈哈哈哈", "嗯嗯晚安", "我到家了", "字" * LONG_TEXT_CHARS]
)
def test_small_talk_does_not(text: str) -> None:
    assert not wants_thinking(text)


def test_the_setting_decides() -> None:
    assert resolve_thinking("on", "好的")
    assert not resolve_thinking("off", "你吃了吗？")
    assert resolve_thinking("auto", "你吃了吗？")
    assert not resolve_thinking("auto", "好的")
    # thinking.auto_rules off: auto is the same as off
    assert not resolve_thinking("auto", "你吃了吗？", auto_rules=False)
    assert resolve_thinking("on", "好的", auto_rules=False)


@pytest.mark.parametrize(
    "text",
    [
        "抱歉，我无法继续这个话题",
        "对不起，我不能回答这个问题。",
        "我无法协助你完成这个请求",
        "这违反了相关政策",
        "I can't assist with that request.",
        "I cannot help with this.",
        "As an AI language model, I...",
        "Content Exists Risk",
    ],
)
def test_the_wording_of_a_refusal_is_recognised(text: str) -> None:
    assert looks_like_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "对不起，我不能继续聊了，要睡了",
        "抱歉抱歉我刚才在洗澡",
        "我没办法陪你出去了，明天要上班",
        "好的",
        "",
        "   ",
    ],
)
def test_a_girlfriend_ending_the_chat_is_not_a_refusal(text: str) -> None:
    assert not looks_like_refusal(text)
