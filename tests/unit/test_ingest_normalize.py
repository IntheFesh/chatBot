"""Normalisation of export messages (R-IMP-007, R-IMP-009, R-SCOPE-002)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from tests.fixtures.synth_export import message
from twin.config.settings import TimeConfig, TzRange
from twin.ingest.events import Kind
from twin.ingest.normalize import (
    InvalidMessage,
    NormalizeContext,
    NormalizedMessage,
    as_id,
    as_int,
    classify_media,
    kind_of,
    normalize_message,
    parse_call,
    parse_voice_seconds,
)
from twin.ingest.schema import ExportMessage
from twin.ingest.times import SourceTime, epoch_to_utc

USERNAME = "wxid_" + "target0001"
EPOCH = 1_735_732_800  # 2025-01-01 12:00:00 UTC
CONTEXT = NormalizeContext(USERNAME, SourceTime("America/Chicago"))


def norm(**fields: Any) -> NormalizedMessage:
    base = message(id="m1", createTime=EPOCH, isSent=False, renderType="text")
    base.update(fields)
    return normalize_message(base, ExportMessage.model_validate(base), CONTEXT)


@pytest.mark.parametrize(
    ("render_type", "kind"),
    [
        ("text", Kind.TEXT),
        ("emoji", Kind.STICKER),
        ("quote", Kind.QUOTE),
        ("image", Kind.IMAGE),
        ("voice", Kind.VOICE),
        ("voip", Kind.CALL),
        ("system", Kind.SYSTEM),
        ("transfer", Kind.TRANSFER),
        ("redPacket", Kind.REDPACKET),
        ("link", Kind.LINK),
        ("video", Kind.VIDEO),
        ("file", Kind.FILE),
        ("location", Kind.LOCATION),
        ("chathistory", Kind.CHATHISTORY),
        ("RED_PACKET", Kind.REDPACKET),
    ],
)
def test_every_render_type_maps_to_its_kind(render_type: str, kind: Kind) -> None:
    item = norm(renderType=render_type)
    assert item.kind == kind.value and item.render_type_known


def test_an_unknown_render_type_is_kept_as_unknown_and_flagged() -> None:
    item = norm(renderType="holographic", novelField=1)
    assert item.kind == "unknown" and not item.render_type_known
    assert item.unknown_fields == {"novelField"}
    assert item.raw["novelField"] == 1  # nothing is lost


def test_the_numeric_type_decides_when_render_type_is_missing() -> None:
    assert kind_of(None, 47) == (Kind.STICKER, True)
    assert kind_of(None, "3") == (Kind.IMAGE, True)
    assert kind_of("", 10000) == (Kind.SYSTEM, True)
    assert kind_of("holographic", 1) == (Kind.TEXT, False)
    assert kind_of(None, None) == (Kind.UNKNOWN, True)
    assert kind_of("holographic", 999) == (Kind.UNKNOWN, False)


def test_text_messages_keep_their_content() -> None:
    item = norm(content="今天吃什么")
    assert (item.kind, item.text, item.is_sent) == ("text", "今天吃什么", False)
    assert item.media == [] and item.quote is None


def test_is_sent_decides_who_spoke() -> None:
    assert norm(isSent=False).is_sent is False  # her
    assert norm(isSent=True).is_sent is True  # the user
    assert norm(isSent=1).is_sent is True
    assert norm(isSent="true").is_sent is True
    assert norm(isSent="0").is_sent is False


def test_the_sender_name_decides_when_is_sent_is_missing() -> None:
    base = message(id="m", createTime=EPOCH, renderType="text")
    for sender, expected in ((USERNAME, False), ("wxid_" + "someone9999", True)):
        raw = {**base, "senderUsername": sender}
        item = normalize_message(raw, ExportMessage.model_validate(raw), CONTEXT)
        assert item.is_sent is expected
    with pytest.raises(InvalidMessage) as info:
        normalize_message(base, ExportMessage.model_validate(base), CONTEXT)
    assert info.value.reason == "no_sender"


def test_ids_come_from_id_then_local_id() -> None:
    assert norm(id=12345).id == "12345"
    raw = message(localId=77, createTime=EPOCH, isSent=True, renderType="text")
    item = normalize_message(raw, ExportMessage.model_validate(raw), CONTEXT)
    assert item.id == f"{USERNAME}:77"
    nothing = message(createTime=EPOCH, isSent=True)
    with pytest.raises(InvalidMessage) as info:
        normalize_message(nothing, ExportMessage.model_validate(nothing), CONTEXT)
    assert info.value.reason == "no_id"


def test_times_accept_seconds_milliseconds_and_the_local_text() -> None:
    assert norm(createTime=EPOCH).create_time == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert norm(createTime=EPOCH * 1000).create_time == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    assert norm(createTime=str(EPOCH)).create_time == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    from_text = norm(createTime=None, createTimeText="2025-01-01 06:00:00")
    assert from_text.create_time == datetime(2025, 1, 1, 12, 0, tzinfo=UTC)  # Chicago is UTC-6
    raw = message(id="x", isSent=False)
    with pytest.raises(InvalidMessage) as info:
        normalize_message(raw, ExportMessage.model_validate(raw), CONTEXT)
    assert info.value.reason == "no_time"


def test_a_text_that_disagrees_with_the_epoch_is_flagged() -> None:
    assert not norm(createTimeText="2025-01-01 06:00:00").time_mismatch
    assert norm(createTimeText="2025-01-01 12:00:00").time_mismatch  # six hours off
    assert not norm(createTimeText="not a time").time_mismatch


def test_source_time_ranges_override_the_default_zone() -> None:
    config = TimeConfig(
        source_timezone="America/Chicago",
        source_timezone_ranges=[
            TzRange(**{"from": date(2025, 1, 1), "to": date(2025, 1, 31), "tz": "Asia/Shanghai"})
        ],
    )
    source = SourceTime.from_config(config)
    assert source.parse_local_text("2025-01-01 20:00:00") == datetime(2025, 1, 1, 12, tzinfo=UTC)
    assert source.parse_local_text("2025-02-01 06:00:00") == datetime(2025, 2, 1, 12, tzinfo=UTC)
    assert source.local_date(datetime(2025, 1, 1, 20, tzinfo=UTC)) == date(2025, 1, 2)
    assert source.local_date(datetime(2025, 2, 1, 20, tzinfo=UTC)) == date(2025, 2, 1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4200", 4),
        ("800", 1),
        (3200, 3),
        ("59000", 59),
        ("0", None),
        ("", None),
        ("abc", None),
        (None, None),
        (-5, None),
        ("1500.0", 2),
        (True, None),
    ],
)
def test_voice_length_is_milliseconds_and_tolerates_junk(value: Any, expected: int | None) -> None:
    assert parse_voice_seconds(value) == expected


def test_voice_with_and_without_transcript() -> None:
    spoken = norm(renderType="voice", voiceLength="4600", voiceTranscript="我到了")
    assert (spoken.voice_seconds, spoken.has_transcript, spoken.text) == (5, True, "我到了")
    silent = norm(renderType="voice", voiceLength="4600", voiceTranscriptStatus="none")
    assert (silent.voice_seconds, silent.has_transcript, silent.text) == (5, False, None)
    blank = norm(renderType="voice", voiceLength="900", voiceTranscript="   ")
    assert not blank.has_transcript and blank.text is None


@pytest.mark.parametrize(
    ("content", "status", "seconds"),
    [
        ("通话时长 37:12", "connected", 2232),
        ("通话时长 01:05", "connected", 65),
        ("通话时长 1:02:03", "connected", 3723),
        ("通话时长 00:00", "unknown", None),
        ("对方已取消", "cancelled", None),
        ("已拒绝", "rejected", None),
        ("未应答", "missed", None),
        ("已在其它设备接听", "other_device", None),
        ("已在其他设备接听", "other_device", None),
        ("", "unknown", None),
        (None, "unknown", None),
    ],
)
def test_call_content_is_parsed(content: str | None, status: str, seconds: int | None) -> None:
    assert parse_call(content) == (status, seconds)
    item = norm(renderType="voip", content=content)
    assert (item.call_status, item.call_duration_s) == (status, seconds)


def test_stickers_carry_md5_and_url_and_ignore_malformed_md5() -> None:
    md5 = "ab" * 16
    item = norm(renderType="emoji", emojiMd5=md5.upper(), emojiUrl="https://example.test/e")
    assert (item.sticker_md5, item.sticker_url) == (md5, "https://example.test/e")
    assert norm(renderType="emoji", emojiMd5="nothex").sticker_md5 is None


def test_quote_fields_are_collected() -> None:
    item = norm(
        renderType="quote",
        content="好的",
        quoteTitle="她",
        quoteContent="晚上吃饭吗",
        quoteType=1,
    )
    assert item.kind == "quote" and item.text == "好的"
    assert item.quote == {"quoteTitle": "她", "quoteContent": "晚上吃饭吗", "quoteType": 1}
    assert norm(renderType="quote", title="标题里的回复").text == "标题里的回复"


def test_main_text_per_kind() -> None:
    assert norm(renderType="link", title="文章", content="<xml/>").text == "文章"
    assert norm(renderType="file", title="a.pdf").text == "a.pdf"
    assert norm(renderType="location", locationPoiname="公园", locationLabel="x").text == "公园"
    assert norm(renderType="location", locationLabel="北路").text == "北路"
    assert norm(renderType="system", content="你撤回了一条消息").text == "你撤回了一条消息"
    assert norm(renderType="image", content="[图片]").text is None
    assert norm(renderType="transfer", content="x", amount="1").text is None


def test_media_entries_are_classified() -> None:
    entries = [
        message(kind="image", path="media/images/a.jpg", md5="1", fileId="f"),
        message(kind="image_thumb", path="media/images/a_t.jpg", md5="2", fileId="g"),
    ]
    item = norm(renderType="image", offlineMedia=entries)
    assert [(m.kind, m.path) for m in item.media] == [
        ("image", "media/images/a.jpg"),
        ("skip", "media/images/a_t.jpg"),  # the full picture makes the thumbnail unnecessary
    ]
    only_thumb = norm(renderType="image", offlineMedia=entries[1:])
    assert [m.kind for m in only_thumb.media] == ["image"]
    video = norm(
        renderType="video",
        offlineMedia=[
            message(kind="video", path="media/v.mp4", md5=None, fileId="v"),
            message(kind="video_thumb", path="media/v.jpg", md5=None, fileId="t"),
        ],
    )
    assert [m.kind for m in video.media] == ["skip", "video_cover"]
    assert item.sender_avatar_path is None
    assert norm(isSent=True, senderAvatarPath="media/avatars/me.png").sender_avatar_path


@pytest.mark.parametrize(
    ("declared", "parent", "expected"),
    [
        ("emoji", Kind.STICKER, "sticker"),
        ("avatar", Kind.TEXT, "avatar"),
        ("voice", Kind.VOICE, "voice"),
        ("audio", Kind.VOICE, "voice"),
        ("file", Kind.FILE, "skip"),
        ("thumb", Kind.VIDEO, "video_cover"),
        ("cover", Kind.VIDEO, "video_cover"),
        ("photo", Kind.IMAGE, "image"),
        ("mystery", Kind.IMAGE, "skip"),
        (None, Kind.IMAGE, "image"),
        (None, Kind.LINK, "skip"),
    ],
)
def test_classify_media(declared: str | None, parent: Kind, expected: str) -> None:
    assert classify_media(declared, parent) == expected


def test_helper_coercions() -> None:
    assert as_int("12") == 12 and as_int(3.0) == 3 and as_int("x") is None and as_int(None) is None
    assert as_int(True) is None and as_int(float("nan")) is None
    assert as_id(7) == "7" and as_id(7.0) == "7" and as_id(" a ") == "a" and as_id([]) is None
    assert epoch_to_utc(0) is None and epoch_to_utc("x") is None and epoch_to_utc(1e30) is None
    assert epoch_to_utc(True) is None


def test_the_model_rejects_a_wrong_type_in_a_typed_field() -> None:
    from pydantic import ValidationError

    odd = message(id="m", createTime=EPOCH, isSent=False, renderType=["text"])
    with pytest.raises(ValidationError):
        ExportMessage.model_validate(odd)
