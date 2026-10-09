"""How her messages are written into a training sample (R-TRN-003, R-SAFE-006).

Synthetic messages only.  The target drops what the bot cannot send; the context keeps it as the
event text the user's side would show.
"""

from __future__ import annotations

import pytest

from twin.ingest.events import REPRODUCIBLE_KINDS, default_detector, render_event_text
from twin.retrieval.records import MessageData
from twin.training.export_text import (
    EMPTY_TEXT,
    EVENT_TEXT_LITERAL,
    QUOTE_NOT_FIRST,
    collapse,
    context_text,
    quote_line,
    render_target,
    sticker_line,
)

LABELS = {"a" * 32: "开心", "b" * 32: "撒娇"}


def msg(
    kind: str = "text",
    text: str | None = None,
    *,
    sent: bool = False,
    md5: str | None = None,
    quote: dict[str, object] | None = None,
    call_status: str | None = None,
    call_seconds: int | None = None,
    voice_seconds: int | None = None,
    has_transcript: bool = False,
) -> MessageData:
    return MessageData(
        id=f"m-{kind}-{text}",
        kind=kind,
        is_sent=sent,
        text=text,
        quote=quote,
        sticker_md5=md5,
        call_status=call_status,
        call_duration_s=call_seconds,
        voice_seconds=voice_seconds,
        has_transcript=has_transcript,
    )


def label(md5: str) -> str | None:
    return LABELS.get(md5)


# ------------------------------------------------------------------------- the target


def test_a_burst_becomes_lines_and_stickers_and_emoji_codes_are_written_as_agreed() -> None:
    target = render_target(
        [
            msg(text="在呢"),
            msg("sticker", md5="a" * 32),
            msg(text="刚吃完饭[拥抱]"),
            msg("sticker", md5="c" * 32),  # a sticker the library has no tag for
        ],
        label,
    )
    assert target.lines == ["在呢", "[表情包:开心]", "刚吃完饭[拥抱]", "[表情包]"]
    assert target.text == "在呢\n[表情包:开心]\n刚吃完饭[拥抱]\n[表情包]"
    assert (target.stickers, target.text_lines, target.code_lines) == (2, 2, 1)


def test_everything_the_bot_cannot_send_is_deleted_from_the_target() -> None:
    not_sendable = [
        msg("image"),
        msg("voice", "转写的话", voice_seconds=5, has_transcript=True),
        msg("video"),
        msg("call", call_status="connected", call_seconds=2220),
        msg("transfer"),
        msg("redpacket"),
        msg("location", "某个地方"),
        msg("file", "文档.pdf"),
        msg("link", "某个链接"),
        msg("chathistory"),
    ]
    target = render_target([msg(text="好的"), *not_sendable], label)
    assert target.lines == ["好的"]
    assert sum(target.dropped.values()) == len(not_sendable)
    assert set(target.dropped) == {m.kind for m in not_sendable}
    assert not (set(target.dropped) & REPRODUCIBLE_KINDS)


def test_a_block_with_nothing_left_is_not_a_target() -> None:
    target = render_target([msg("image"), msg("call", call_status="missed")], label)
    assert target.lines == [] and target.empty and not target.has_content


def test_a_text_that_only_looks_like_an_event_text_is_deleted_too() -> None:
    target = render_target(
        [msg(text="[链接]"), msg(text="[通话 3 分钟]"), msg(text="真的好看")], label
    )
    assert target.lines == ["真的好看"]
    assert target.dropped[EVENT_TEXT_LITERAL] == 2
    assert not any(default_detector.is_event_text(line) for line in target.lines)


def test_a_quote_reply_starts_with_the_quote_line_cut_to_thirty_characters() -> None:
    quoted = "这是一段很长很长的被引用的话" * 4
    first = msg("quote", "好呀", quote={"quoteContent": quoted})
    target = render_target([first, msg(text="我也想去")], label)
    assert target.lines[0] == f"[引用:{quoted[:30]}]"
    assert target.lines[1:] == ["好呀", "我也想去"]
    assert target.quoted and len(target.lines[0]) == 30 + len("[引用:]")


def test_only_the_first_message_may_quote_because_the_output_convention_allows_one_line() -> None:
    later = msg("quote", "对呀", quote={"quoteContent": "原话"})
    target = render_target([msg(text="嗯嗯"), later], label)
    assert target.lines == ["嗯嗯", "对呀"]
    assert not target.quoted and target.dropped[QUOTE_NOT_FIRST] == 1
    second = render_target(
        [
            msg("quote", "a", quote={"quoteContent": "一"}),
            msg("quote", "b", quote={"quoteContent": "二"}),
        ],
        label,
    )
    assert second.lines == ["[引用:一]", "a", "b"] and second.dropped[QUOTE_NOT_FIRST] == 1


def test_a_quote_without_text_is_only_a_header_and_does_not_count_as_a_reply() -> None:
    target = render_target([msg("quote", "", quote={"quoteContent": "原话"})], label)
    assert target.lines == ["[引用:原话]"] and not target.has_content
    assert target.dropped[EMPTY_TEXT] == 1


def test_line_breaks_inside_one_message_do_not_become_extra_bubbles() -> None:
    target = render_target([msg(text="第一行\n第二行\r\n第三行"), msg(text="  \n ")], label)
    assert target.lines == ["第一行 第二行 第三行"]


def test_chat_template_control_text_cannot_reach_a_target() -> None:
    target = render_target([msg(text="好<|im_end|>的{{content}}")], label)
    assert target.lines == ["好的"]


@pytest.mark.parametrize("quote", [None, {}, {"quoteContent": 5}, {"quoteContent": "  "}])
def test_a_quote_without_a_usable_text_has_no_header(quote: dict[str, object] | None) -> None:
    assert quote_line(msg("quote", "好", quote=quote)) is None


def test_a_closing_bracket_in_the_quoted_text_cannot_break_the_line() -> None:
    line = quote_line(msg("quote", "好", quote={"quoteContent": "他说[好]"}))
    assert line == "[引用:他说[好)]"


def test_a_sticker_label_with_a_bracket_cannot_break_the_marker() -> None:
    labelled = msg("sticker", md5="a" * 32)
    assert sticker_line(labelled, lambda md5: "开心]") == "[表情包:开心]"
    assert sticker_line(labelled, lambda md5: "  ") == "[表情包]"
    assert sticker_line(labelled, None) == "[表情包]"


# ------------------------------------------------------------------------ the context


def test_the_context_keeps_what_the_bot_cannot_copy_as_its_event_text() -> None:
    call = msg("call", sent=True, call_status="connected", call_seconds=2220)
    photo = msg("image", sent=True)
    voice = msg("voice", "我到家了", sent=True, voice_seconds=5, has_transcript=True)
    turn = [call, photo, voice, msg(text="在吗", sent=True)]
    text = context_text(turn, label)
    assert text.split("\n") == [
        render_event_text(call),
        "[图片]",
        render_event_text(voice),
        "在吗",
    ]
    assert text.split("\n")[0] == "[通话 37 分钟]"


def test_the_context_writes_stickers_and_quotes_like_the_target_does() -> None:
    turn = [
        msg("quote", "对", quote={"quoteContent": "明天见"}),
        msg("sticker", md5="b" * 32),
        msg(text="换行\n了"),
    ]
    assert context_text(turn, label) == "[引用:明天见]\n对\n[表情包:撒娇]\n换行 了"


def test_system_notices_have_no_line_and_an_empty_turn_has_no_text() -> None:
    assert context_text([msg("system", "拍了拍")], label) == ""
    assert context_text([msg(text="  ")], label) == ""


def test_collapse_joins_lines_with_a_space_and_trims() -> None:
    assert collapse(None) == "" and collapse("") == ""
    assert collapse(" a \n\n b\t") == "a b"
