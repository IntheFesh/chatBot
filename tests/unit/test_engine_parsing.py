"""Reading the model's output: the four forms of the convention (R-ENG-007)."""

from __future__ import annotations

import pytest

from twin.engine.parsing import ParsedLine, parse_reply

TEXT = "text"


def kinds(raw: str) -> list[tuple[str, str]]:
    return [(line.kind, line.text) for line in parse_reply(raw).lines]


def test_every_line_is_a_bubble() -> None:
    parsed = parse_reply("好呀\n等我一下\n\n马上来")
    assert parsed.quote is None
    assert [line.text for line in parsed.lines] == ["好呀", "等我一下", "马上来"]
    assert all(line.kind == TEXT for line in parsed.lines)


def test_sticker_lines_carry_their_tag() -> None:
    assert kinds("哈哈\n[表情包:开心]") == [(TEXT, "哈哈"), ("sticker", "开心")]
    assert kinds("[表情包： 撒娇 ]") == [("sticker", "撒娇")]  # full-width colon and spaces
    assert kinds("【表情包:晚安】") == [("sticker", "晚安")]


def test_a_sticker_marker_inside_a_text_line_becomes_a_line_of_its_own() -> None:
    assert kinds("好呀[表情包:开心]等下见") == [
        (TEXT, "好呀"),
        ("sticker", "开心"),
        (TEXT, "等下见"),
    ]
    assert kinds("[表情包:开心][表情包:大笑]") == [("sticker", "开心"), ("sticker", "大笑")]


def test_only_the_first_line_may_quote() -> None:
    parsed = parse_reply("[引用:今天吃什么]\n吃面吧")
    assert parsed.quote == "今天吃什么" and [line.text for line in parsed.lines] == ["吃面吧"]
    stray = parse_reply("吃面吧\n[引用:今天吃什么]\n你呢")
    assert stray.quote is None and stray.stray_quotes == 1
    assert [line.text for line in stray.lines] == ["吃面吧", "你呢"]
    second = parse_reply("[引用:一]\n[引用:二]\n好")
    assert second.quote == "一" and second.stray_quotes == 1
    assert parse_reply("[引用:你说[拥抱]那句]\n嗯").quote == "你说[拥抱]那句"
    assert parse_reply("[引用:]\n嗯").quote is None  # nothing to quote


def test_a_long_quote_is_cut_to_a_fragment() -> None:
    assert len(parse_reply("[引用:" + "字" * 200 + "]\n嗯").quote or "") == 60


def test_the_silence_marker_is_a_line_kind() -> None:
    assert parse_reply("[不回]").lines == (ParsedLine("no_reply"),)
    assert kinds("【不回】") == [("no_reply", "")]
    assert kinds("好的\n[不回]") == [(TEXT, "好的"), ("no_reply", "")]


def test_a_speaker_label_is_taken_off() -> None:
    parsed = parse_reply("她：好呀\n她的回复：等下\n回复：嗯")
    assert [line.text for line in parsed.lines] == ["好呀", "等下", "嗯"]
    assert parsed.labels_removed == 3
    assert kinds("她：") == []


def test_invisible_characters_and_line_endings_do_not_matter() -> None:
    assert kinds("﻿好呀\r\n​嗯\r") == [(TEXT, "好呀"), (TEXT, "嗯")]


@pytest.mark.parametrize("raw", ["", "  \n \n", "​"])
def test_nothing_in_nothing_out(raw: str) -> None:
    assert parse_reply(raw).empty


def test_text_is_never_dropped() -> None:
    raw = "你好[拥抱]，今天怎么样？\n[其他括号]也是文字"
    assert [line.text for line in parse_reply(raw).lines] == [
        "你好[拥抱]，今天怎么样？",
        "[其他括号]也是文字",
    ]
