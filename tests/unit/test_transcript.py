"""Transcript lines and the corpus statements round 06 added (R-STO-007, R-IMP-007, R-TRN-013)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import inspect, select

from tests.support.embedding import H, Msg, U, day, write_dialogue
from tests.support.persona import attach_files, sticker_files
from twin.ingest.corpus import (
    her_bubble_skeleton,
    her_message_count,
    messages_between,
)
from twin.ingest.transcript import (
    MESSAGE_CHAR_CAP,
    STICKER_TEXT,
    TranscriptLine,
    message_text,
    transcript_lines,
)
from twin.services import Services
from twin.storage.chat_models import Message


@pytest.fixture
def chat(services: Services) -> Services:
    files = sticker_files(1)
    md5 = next(iter(files))
    write_dialogue(
        services,
        [
            (
                day(0),
                [
                    U("你好"),
                    H("嗯嗯"),
                    H(kind="sticker", md5=md5),
                    U(kind="image"),
                    H(kind="image"),
                    Msg("u", "系统提示", "system"),
                    U("看看"),
                    H("  好   的\n呀  "),
                ],
            ),
            (day(2), [U("再见"), H("拜拜"), H("明天见")]),
        ],
    )
    attach_files(services, files)
    return services


def lines_of(services: Services, **options: object) -> list[TranscriptLine]:
    with services.db.session() as session:
        found = session.scalars(select(Message).order_by(Message.create_time_utc, Message.sort_seq))
        return transcript_lines(found, **options)  # type: ignore[arg-type]


def sticker_md5(services: Services) -> str:
    with services.db.session() as session:
        return str(
            session.scalar(select(Message.sticker_md5).where(Message.sticker_md5.is_not(None)))
        )


def test_text_is_one_line_and_events_are_their_event_text(chat: Services) -> None:
    lines = lines_of(chat)
    assert [(line.her, line.text) for line in lines[:8]] == [
        (False, "你好"),
        (True, "嗯嗯"),
        (True, STICKER_TEXT),
        (False, "[图片]"),
        (True, "[图片]"),
        (False, "看看"),
        (True, "好 的 呀"),
        (False, "再见"),
    ]
    assert all(line.message_id for line in lines) and "系统提示" not in [x.text for x in lines]


def test_a_known_sticker_shows_its_tag(chat: Services) -> None:
    md5 = sticker_md5(chat)
    plain = [x.text for x in lines_of(chat)]
    tagged = [x.text for x in lines_of(chat, sticker_tag_of={md5: "开心"}.get)]
    assert STICKER_TEXT in plain and "[表情包:开心]" in tagged and STICKER_TEXT not in tagged
    other = [x.text for x in lines_of(chat, sticker_tag_of={"x": "y"}.get)]
    assert STICKER_TEXT in other


def test_a_long_message_is_shortened_and_a_system_notice_has_no_line(chat: Services) -> None:
    long_message = Message(
        id="m", conversation_id="c", create_time_utc=day(0), is_sent=False, kind="text",
        text="很长" * 100, raw={}, source_export_id="x", has_transcript=False,
    )  # fmt: skip
    text = message_text(long_message)
    assert text is not None and len(text) == MESSAGE_CHAR_CAP and text.endswith("…")
    assert message_text(long_message, limit=5) == "很长很长…"
    empty = Message(
        id="e", conversation_id="c", create_time_utc=day(0), is_sent=False, kind="text",
        text="  ", raw={}, source_export_id="x", has_transcript=False,
    )  # fmt: skip
    assert message_text(empty) is None
    with chat.db.session() as session:
        notice = session.scalars(select(Message).where(Message.kind == "system")).one()
        assert message_text(notice) is None


def test_a_line_renders_with_the_labels() -> None:
    line = TranscriptLine("m1", True, "好")
    assert line.render() == "她：好" and TranscriptLine("m2", False, "嗯").render() == "对方：嗯"
    assert line.render(her_label="A", user_label="B") == "A：好"


# --------------------------------------------------------------- the statements


def test_her_messages_are_counted_without_system_notices_and_before_a_moment(
    chat: Services,
) -> None:
    with chat.db.session() as session:
        total = session.scalar(her_message_count())
        before = session.scalar(her_message_count(day(1)))
        none = session.scalar(her_message_count(day(0)))
    assert total == 6 and before == 4 and none == 0  # 嗯嗯, sticker, image, 好的; then two on day 2


def test_her_bubbles_carry_only_id_time_kind_and_sticker(chat: Services) -> None:
    with chat.db.session() as session:
        found = list(session.scalars(her_bubble_skeleton()))
        assert [row.kind for row in found] == ["text", "sticker", "image", "text", "text", "text"]
        assert all(not row.is_sent for row in found)
        assert sum(1 for row in found if row.sticker_md5) == 1
        unloaded = inspect(found[0]).unloaded  # the words are not even read from the database
        assert {"text_ct", "raw_ct", "quote_ct"} <= unloaded
        earlier = list(session.scalars(her_bubble_skeleton(day(1))))
    assert len(found) == 6 and len(earlier) == 4


def test_messages_between_are_both_sides_in_time_order_with_an_exclusive_bound(
    chat: Services,
) -> None:
    with chat.db.session() as session:
        everything = list(session.scalars(messages_between(day(0), day(0) + timedelta(hours=1))))
        assert [m.kind for m in everything].count("system") == 1
        times = [m.create_time_utc for m in everything]
        assert times == sorted(times) and {m.is_sent for m in everything} == {True, False}
        bounded = list(session.scalars(messages_between(day(0), day(3), before=day(1))))
        assert [m.id for m in bounded] == [m.id for m in everything]
        wide = list(session.scalars(messages_between(day(0), day(3))))
        assert len(wide) == len(everything) + 3
        first = everything[0].create_time_utc
        assert list(session.scalars(messages_between(day(0), day(3), before=first))) == []
