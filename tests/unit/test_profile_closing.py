"""Closing replies and how often she lets them be the last word (R-ENG-007)."""

from __future__ import annotations

from datetime import date

import pytest

from twin.profile.closing import is_closing_message
from twin.profile.localtime import LocalStamp
from twin.profile.metrics import WindowCollector
from twin.profile.units import Rec

STAMP = LocalStamp(date(2026, 1, 5), 600.0, 40, "UTC")


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("text", "好的"),
        ("text", "嗯嗯"),
        ("text", "晚安"),
        ("text", "哈哈哈哈"),
        ("text", "好哒[捂脸]"),
        ("text", "[拥抱]"),
        ("text", "嗯嗯。"),
        ("text", "ok"),
        ("quote", "知道啦"),
        ("sticker", None),
    ],
)
def test_short_answers_that_ask_nothing_are_closing(kind: str, text: str | None) -> None:
    assert is_closing_message(kind, text)


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        ("text", "你吃饭了吗"),
        ("text", "怎么了"),
        ("text", "在干嘛"),
        ("text", "那我们周末去哪里玩呀"),
        ("text", "今天好累啊真的不想上班了"),
        ("text", "好的?"),
        ("text", "？"),
        ("text", ""),
        ("text", None),
        ("image", None),
        ("voice", "好的"),
    ],
)
def test_questions_long_messages_and_media_are_not_closing(kind: str, text: str | None) -> None:
    assert not is_closing_message(kind, text)


def rec(ts: float, her: bool, text: str = "好的", kind: str = "text") -> Rec:
    return Rec(f"m{ts}", ts, her, kind, text, None, STAMP, "workday")


def closing_rate(messages: list[Rec]) -> tuple[float, int]:
    collector = WindowCollector(120, 3600, ())
    for message in messages:
        collector.feed(message)
    leaf = collector.leaves()["her"]["closing_no_reply_rate"]
    return leaf.value, leaf.n  # type: ignore[union-attr]


def test_the_rate_counts_closing_messages_she_did_not_answer() -> None:
    # (1) closing, answered   (2) closing, nobody answers before the next burst of the user
    # (3) closing, the next message comes after the segment gap   (4) a question: not counted
    value, count = closing_rate(
        [
            rec(0, False, "好的"),
            rec(30, True, "嗯嗯"),
            rec(500, False, "晚安"),
            rec(900, False, "还没睡吗"),  # a burst of the user after more than the burst gap
            rec(940, True, "还没"),
            rec(10_000, False, "嗯嗯"),
            rec(14_000, False, "哈哈"),
        ]
    )
    assert count == 3  # 好的 (answered), 晚安 (the user spoke again), 嗯嗯 (new segment follows)
    assert value == pytest.approx(2 / 3)


def test_the_last_open_burst_is_not_judged() -> None:
    value, count = closing_rate(
        [rec(0, False, "好的"), rec(10, True, "嗯"), rec(100, False, "嗯嗯")]
    )
    assert count == 1 and value == 0.0  # the final "嗯嗯" has not been followed by anything yet


def test_no_closing_messages_means_no_data() -> None:
    value, count = closing_rate([rec(0, False, "你好吗"), rec(10, True, "好呀")])
    assert count == 0 and value == 0.0
