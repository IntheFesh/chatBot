"""One way to show a reply, hers or the bot's (R-EVAL-001)."""

from __future__ import annotations

import pytest

from twin.eval.render import (
    Candidate,
    CandidateLine,
    candidate_from_bubbles,
    candidate_from_messages,
    has_event_text,
    message_lines,
    quote_fragment,
    render_candidate,
    render_turn,
    text_lines,
)
from twin.retrieval.records import MessageData

STICKERS = {"a" * 32: "笑得很开心的猫", "b" * 32: "委屈的小狗"}


def label(md5: str) -> str | None:
    return STICKERS.get(md5)


def message(
    kind: str = "text",
    text: str | None = None,
    *,
    sent: bool = False,
    md5: str | None = None,
    quote: dict[str, str] | None = None,
    call_status: str | None = None,
) -> MessageData:
    return MessageData(
        id=f"m-{kind}-{text or md5}",
        kind=kind,
        is_sent=sent,
        text=text,
        quote=quote,
        sticker_md5=md5,
        call_status=call_status,
        call_duration_s=None,
        voice_seconds=None,
        has_transcript=False,
    )


def test_a_real_reply_and_a_bot_reply_with_the_same_content_render_identically() -> None:
    """The structural guarantee of the blind test: nothing but the style can give a side away."""
    real = candidate_from_messages(
        [
            message("quote", "好呀[拥抱]", quote={"quoteContent": "晚上一起吃饭吗"}),
            message("sticker", md5="a" * 32),
            message("text", "那就六点\n不见不散"),
        ]
    )
    bot = candidate_from_bubbles(
        [
            ("text", None, "好呀[拥抱]"),
            ("sticker", "a" * 32, "[表情包:开心]"),
            ("text", None, "那就六点"),
            ("text", None, "不见不散"),
        ],
        "晚上一起吃饭吗",
    )
    expected = "[引用：晚上一起吃饭吗]\n好呀[拥抱]\n[表情包：笑得很开心的猫]\n那就六点\n不见不散"
    assert render_candidate(real, label) == expected
    assert render_candidate(bot, label) == expected


def test_a_sticker_is_always_shown_by_the_description_the_library_has() -> None:
    first = Candidate((CandidateLine("sticker", sticker_md5="b" * 32),))
    assert render_candidate(first, label) == "[表情包：委屈的小狗]"
    unknown = Candidate((CandidateLine("sticker", sticker_md5="c" * 32),))
    assert render_candidate(unknown, label) == "[表情包：未标注]"  # hers and the bot's alike
    assert render_candidate(unknown, lambda md5: "  ") == "[表情包：未标注]"
    with pytest.raises(ValueError, match="MD5"):
        CandidateLine("sticker")


def test_the_quote_is_one_first_line_of_at_most_thirty_characters() -> None:
    long = "这是一段很长很长的被引用的话" * 4
    shown = render_candidate(Candidate((CandidateLine("text", text="好"),), long), label)
    first, second = shown.split("\n")
    assert first.startswith("[引用：") and first.endswith("]") and second == "好"
    assert len(first) == len("[引用：]") + 30
    assert quote_fragment("a]b") == "a)b" and quote_fragment("  \n ") is None
    # a quote without any bubble is still just the header; without a fragment there is no header
    assert render_candidate(Candidate((), "你好"), label) == "[引用：你好]"
    assert render_candidate(Candidate((CandidateLine("text", text="好"),), "  "), label) == "好"


def test_a_line_break_inside_a_message_is_a_line_like_a_second_bubble() -> None:
    assert text_lines("一\n二\r\n\n  三  ") == ["一", "二", "三"]
    one_message = candidate_from_messages([message("text", "一\n二")])
    two_bubbles = candidate_from_bubbles([("text", None, "一"), ("text", None, "二")], None)
    assert render_candidate(one_message, label) == render_candidate(two_bubbles, label) == "一\n二"
    assert text_lines(None) == [] and text_lines("\x07\x1f") == []  # control characters go


def test_only_what_the_bot_could_send_is_taken_from_a_real_reply() -> None:
    reply = candidate_from_messages(
        [message("image"), message("text", "哈哈"), message("voice"), message("sticker", md5=None)]
    )
    assert [line.text for line in reply.lines] == ["哈哈"] and reply.quote is None
    assert candidate_from_messages([message("call")]).empty
    # the quote of a later message does not become the header of the reply
    late = candidate_from_messages(
        [message("text", "先说"), message("quote", "再说", quote={"quoteContent": "旧话"})]
    )
    assert late.quote is None and [line.text for line in late.lines] == ["先说", "再说"]


def test_event_text_and_media_placeholders_are_recognised_in_a_candidate() -> None:
    assert has_event_text(Candidate((CandidateLine("text", text="[图片]"),)))
    assert has_event_text(Candidate((CandidateLine("text", text="[链接]"),)))
    assert not has_event_text(Candidate((CandidateLine("text", text="好的[拥抱]"),)))
    assert not has_event_text(Candidate((CandidateLine("sticker", sticker_md5="a" * 32),)))


def test_a_candidate_survives_the_database_round_trip() -> None:
    original = candidate_from_bubbles(
        [("text", None, "你好"), ("sticker", "a" * 32, "")], "引用的话"
    )
    assert Candidate.from_json(original.to_json()) == original
    assert Candidate.from_json({"lines": []}) == Candidate(())
    assert original.stickers == ("a" * 32,) and not original.empty


def test_the_context_is_shown_by_the_same_rules() -> None:
    assert message_lines(message("text", "在吗\n有事", sent=True), label) == ["在吗", "有事"]
    assert message_lines(message("sticker", md5="a" * 32), label) == ["[表情包：笑得很开心的猫]"]
    assert message_lines(message("sticker"), label) == []
    quoted = message("quote", "我也是", quote={"quoteTitle": "他", "quoteContent": "好久不见"})
    assert message_lines(quoted, label) == ["[引用：好久不见]", "我也是"]
    assert message_lines(message("image"), label) == ["[图片]"]
    assert message_lines(message("call", call_status="missed"), label) != []
    assert message_lines(message("system", "对方撤回了一条消息"), label) == []
    assert render_turn("她", ["一", "二"]) == "她：一\n　　二" and render_turn("我", []) == ""
