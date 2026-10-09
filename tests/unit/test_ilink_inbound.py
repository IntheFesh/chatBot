"""Inbound parsing of every message form, media decryption and quotes (R-CH-005)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    CDN,
    Harness,
    b64_hex,
    b64_raw,
    drive,
    encrypt,
    gif_bytes,
    image_bytes,
    image_item,
    make_harness,
    message,
    now_ms,
    text_item,
    updates,
)
from twin.channel.base import (
    FLAG_MEDIA_UNAVAILABLE,
    FLAG_UNKNOWN_ITEM_TYPE,
    FLAG_VIDEO_NO_COVER,
    FLAG_VOICE_UNTRANSCRIBED,
    InboundMessage,
    MessageKind,
)
from twin.channel.ilink.inbound import InboundConverter, extract_partial, message_time
from twin.channel.ilink.wire import PartialText, WireMessage
from twin.storage.db import Database
from twin.storage.media import MediaKind

GET_UPDATES = f"{API}/ilink/bot/getupdates"
KEY = bytes(range(1, 17))
OTHER_KEY = bytes(range(101, 117))


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def h(db: Database, clock: ManualClock, tmp_path: Path) -> AsyncIterator[Harness]:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    harness.bind()
    yield harness
    await harness.channel.stop()


def serve(api: respx.MockRouter, param: str, body: bytes, status: int = 200) -> respx.Route:
    return api.get(f"{CDN}/download", params={"encrypted_query_param": param}).respond(
        status, content=body
    )


async def receive(
    h: Harness, api: respx.MockRouter, *items: dict[str, Any], mid: int = 77, **kwargs: Any
) -> list[InboundMessage]:
    raw = message(*items, mid=mid, created_ms=now_ms(h.clock), **kwargs)
    api.post(GET_UPDATES).respond(200, json=updates([raw], cursor=f"C-{mid}"))
    await drive(h.channel.poll_once(), h.clock)  # type: ignore[arg-type]
    return h.store.inbox()


# ------------------------------------------------------------------ images


@pytest.mark.parametrize(
    "picture",
    [
        pytest.param(image_bytes("PNG"), id="png"),
        pytest.param(image_bytes("JPEG", size=40), id="jpeg"),
        pytest.param(gif_bytes(), id="gif"),
        pytest.param(b"", id="empty"),
        pytest.param(bytes(range(16)), id="one block"),
        pytest.param(bytes(range(32)), id="two blocks"),
    ],
)
async def test_an_image_is_downloaded_decrypted_and_stored_encrypted(
    picture: bytes, api: respx.MockRouter, h: Harness
) -> None:
    serve(api, "DL-1", encrypt(KEY, picture))
    [received] = await receive(
        h, api, image_item("DL-1", hex_key=KEY.hex(), media_key=b64_raw(KEY))
    )
    assert received.kind is MessageKind.IMAGE and received.flags == frozenset()
    ref = received.media_ref
    assert ref is not None
    assert ref.sha256 == hashlib.sha256(picture).hexdigest() and ref.size == len(picture)
    assert ref.kind is MediaKind.IMAGE
    assert h.media.read_bytes(ref.sha256) == picture  # decrypted bytes match the original
    stored_file = h.media.path_for(ref.sha256)
    assert picture not in stored_file.read_bytes() or picture == b""  # the store keeps it sealed


@pytest.mark.parametrize(
    ("label", "item_keys"),
    [
        ("the hex key on the item", {"hex_key": KEY.hex()}),
        ("a base64 raw 16-byte key", {"media_key": b64_raw(KEY)}),
        ("a base64 32-digit hex key", {"media_key": b64_hex(KEY)}),
        (
            "the item key beats a wrong media key",
            {"hex_key": KEY.hex(), "media_key": b64_raw(OTHER_KEY)},
        ),
        (
            "a bad item key falls back to the media key",
            {"hex_key": "zz", "media_key": b64_raw(KEY)},
        ),
    ],
)
async def test_every_documented_key_form_decrypts_the_download(
    label: str, item_keys: dict[str, str], api: respx.MockRouter, h: Harness
) -> None:
    picture = image_bytes("PNG", size=20)
    serve(api, "DL-K", encrypt(KEY, picture))
    [received] = await receive(h, api, image_item("DL-K", **item_keys))
    assert received.media_ref is not None, label
    assert h.media.read_bytes(received.media_ref.sha256) == picture


async def test_without_any_key_the_downloaded_bytes_are_the_picture(
    api: respx.MockRouter, h: Harness
) -> None:
    picture = image_bytes("JPEG")
    serve(api, "DL-PLAIN", picture)
    [received] = await receive(h, api, image_item("DL-PLAIN"))
    assert received.media_ref is not None
    assert h.media.read_bytes(received.media_ref.sha256) == picture


async def test_a_full_download_url_wins_over_the_parameter(
    api: respx.MockRouter, h: Harness
) -> None:
    picture = image_bytes("PNG")
    route = api.get("https://cdn.example.com/full/1").respond(200, content=encrypt(KEY, picture))
    [received] = await receive(
        h, api, image_item("IGNORED", hex_key=KEY.hex(), full_url="https://cdn.example.com/full/1")
    )
    assert route.call_count == 1 and received.media_ref is not None


@pytest.mark.parametrize(
    ("label", "build", "counter"),
    [
        (
            "missing file",
            lambda api: serve(api, "DL-X", b"", status=404),
            "image.download_http_404",
        ),
        (
            "wrong key",
            lambda api: serve(api, "DL-X", encrypt(OTHER_KEY, b"x" * 40)),
            "image.decrypt_failed",
        ),
        (
            "not a whole number of blocks",
            lambda api: serve(api, "DL-X", b"short"),
            "image.decrypt_failed",
        ),
    ],
)
async def test_a_media_problem_never_loses_the_message(
    label: str, build: Any, counter: str, api: respx.MockRouter, h: Harness
) -> None:
    build(api)
    [received] = await receive(h, api, image_item("DL-X", hex_key=KEY.hex()))
    assert received.kind is MessageKind.IMAGE and received.media_ref is None, label
    assert FLAG_MEDIA_UNAVAILABLE in received.flags
    assert h.store.item_stats().failures[counter] == 1
    assert list(h.media.iter_hashes()) == []


async def test_an_image_without_an_address_or_with_an_unreadable_key_is_unavailable(
    api: respx.MockRouter, h: Harness
) -> None:
    item = {"type": 2, "image_item": {"media": {"aes_key": b64_raw(KEY)}}}
    [no_address] = await receive(h, api, item, mid=1)
    assert no_address.media_ref is None and FLAG_MEDIA_UNAVAILABLE in no_address.flags
    serve(api, "DL-B", b"whatever")
    [_, bad_key] = await receive(h, api, image_item("DL-B", hex_key="zz", media_key="!!"), mid=2)
    assert bad_key.media_ref is None
    failures = h.store.item_stats().failures
    assert failures["image.no_download_address"] == 1 and failures["image.bad_key"] == 1


async def test_an_image_item_without_media_is_still_a_message(
    api: respx.MockRouter, h: Harness
) -> None:
    [received] = await receive(h, api, {"type": 2, "image_item": {}})
    assert received.kind is MessageKind.IMAGE and FLAG_MEDIA_UNAVAILABLE in received.flags


# ------------------------------------------------------- voice, file, video


async def test_a_voice_message_carries_the_cloud_transcription(
    api: respx.MockRouter, h: Harness
) -> None:
    item = {
        "type": 3,
        "voice_item": {"media": {"encrypt_query_param": "V"}, "encode_type": 6, "text": "合成转写"},
    }
    [received] = await receive(h, api, item)
    assert received.kind is MessageKind.VOICE and received.text == "合成转写"
    assert received.media_ref is None and not received.flags  # the audio is never downloaded


async def test_a_voice_message_without_a_transcription_is_marked(
    api: respx.MockRouter, h: Harness
) -> None:
    item = {"type": 3, "voice_item": {"media": {"encrypt_query_param": "V"}, "playtime": 1500}}
    [received] = await receive(h, api, item)
    assert received.kind is MessageKind.VOICE and received.text is None
    assert received.flags == frozenset({FLAG_VOICE_UNTRANSCRIBED})


async def test_a_file_message_gives_the_file_name_only(api: respx.MockRouter, h: Harness) -> None:
    route = api.get(f"{CDN}/download").respond(200, content=b"never requested")
    item = {
        "type": 4,
        "file_item": {
            "media": {"encrypt_query_param": "F"},
            "file_name": "合成名.pdf",
            "len": "1024",
        },
    }
    [received] = await receive(h, api, item)
    assert received.kind is MessageKind.FILE and received.text == "合成名.pdf"
    assert received.media_ref is None and route.call_count == 0


async def test_a_video_message_gives_its_cover_picture(api: respx.MockRouter, h: Harness) -> None:
    cover = image_bytes("JPEG", size=30)
    serve(api, "COVER-1", encrypt(KEY, cover))
    video = api.get(f"{CDN}/download", params={"encrypted_query_param": "VIDEO-1"}).respond(200)
    item = {
        "type": 5,
        "video_item": {
            "media": {"encrypt_query_param": "VIDEO-1", "aes_key": b64_raw(OTHER_KEY)},
            "video_size": 999,
            "play_length": 3000,
            "thumb_media": {"encrypt_query_param": "COVER-1", "aes_key": b64_raw(KEY)},
            "thumb_width": 240,
            "thumb_height": 135,
        },
    }
    [received] = await receive(h, api, item)
    assert received.kind is MessageKind.VIDEO and not received.flags
    assert received.media_ref is not None and received.media_ref.kind is MediaKind.IMAGE
    assert h.media.read_bytes(received.media_ref.sha256) == cover
    assert video.call_count == 0  # the video itself is not fetched


async def test_a_video_without_a_usable_cover_is_marked(api: respx.MockRouter, h: Harness) -> None:
    serve(api, "COVER-2", b"", status=500)
    no_thumb = {"type": 5, "video_item": {"media": {"encrypt_query_param": "V"}}}
    bad_thumb = {
        "type": 5,
        "video_item": {"thumb_media": {"encrypt_query_param": "COVER-2", "aes_key": b64_raw(KEY)}},
    }
    first = await receive(h, api, no_thumb, mid=1)
    both = await receive(h, api, bad_thumb, mid=2)
    assert [m.kind for m in both] == [MessageKind.VIDEO, MessageKind.VIDEO]
    assert all(m.flags == frozenset({FLAG_VIDEO_NO_COVER}) for m in first + both[1:])


async def test_unknown_item_types_are_recorded_by_number_only(
    api: respx.MockRouter, h: Harness
) -> None:
    [received] = await receive(h, api, {"type": 99, "secret_payload": {"content": "private"}})
    assert received.kind is MessageKind.UNKNOWN and received.item_type == 99
    assert received.flags == frozenset({FLAG_UNKNOWN_ITEM_TYPE})
    assert received.text is None and received.media_ref is None


async def test_tool_call_items_are_ignored(api: respx.MockRouter, h: Harness) -> None:
    tool = {"type": 11, "tool_call_start_item": {"name": "x"}}
    inbox = await receive(h, api, tool, text_item("真正的话"), mid=5)
    assert [m.text for m in inbox] == ["真正的话"]
    assert inbox[0].id == "5"  # one real item: the plain id


async def test_a_message_with_several_items_becomes_several_messages(
    api: respx.MockRouter, h: Harness
) -> None:
    serve(api, "DL-M", encrypt(KEY, image_bytes("PNG")))
    inbox = await receive(h, api, text_item("看这个"), image_item("DL-M", hex_key=KEY.hex()), mid=6)
    assert [(m.id, m.kind) for m in inbox] == [
        ("6#0", MessageKind.TEXT),
        ("6#1", MessageKind.IMAGE),
    ]


async def test_an_empty_text_item_produces_nothing(api: respx.MockRouter, h: Harness) -> None:
    assert await receive(h, api, text_item("")) == []
    assert h.store.cursor() == "C-77"


# ------------------------------------------------------------------ times


def test_message_times_accept_milliseconds_seconds_and_nothing(clock: ManualClock) -> None:
    received = clock.now_utc()
    in_ms = WireMessage(create_time_ms=1_790_000_000_000)
    in_seconds = WireMessage(create_time_ms=1_790_000_000)
    assert message_time(in_ms, received).timestamp() == 1_790_000_000
    assert message_time(in_seconds, received).timestamp() == 1_790_000_000
    assert message_time(WireMessage(), received) == received
    assert message_time(WireMessage(create_time_ms=0), received) == received


# ----------------------------------------------------------------- quotes


async def test_a_quote_with_its_text_is_resolved(api: respx.MockRouter, h: Harness) -> None:
    item = text_item(
        "我的回复",
        ref_msg={
            "title": "摘要",
            "message_item": {"type": 1, "msg_id": "Q-1", "text_item": {"text": "被引用的话"}},
        },
    )
    [received] = await receive(h, api, item)
    assert received.text == "我的回复"
    quote = received.quote
    assert quote is not None and quote.resolved
    assert quote.text == "被引用的话" and quote.title == "摘要"


async def test_a_quote_with_only_an_id_is_looked_up_in_the_message_index(
    api: respx.MockRouter, h: Harness
) -> None:
    await receive(h, api, text_item("早先的一句话"), mid=500)
    item = text_item("回复它", ref_msg={"svr_id": 500})
    [_, reply] = await receive(h, api, item, mid=501)
    assert reply.quote is not None and reply.quote.resolved
    assert reply.quote.text == "早先的一句话" and reply.quote.svr_id == "500"


async def test_a_quote_of_an_unknown_message_is_marked_unresolved(
    api: respx.MockRouter, h: Harness
) -> None:
    item = text_item("回复", ref_msg={"svr_id": "404-404", "title": "只有标题"})
    [received] = await receive(h, api, item)
    assert received.quote is not None and not received.quote.resolved
    assert received.quote.text is None and received.quote.title == "只有标题"


async def test_a_quoted_picture_is_downloaded(api: respx.MockRouter, h: Harness) -> None:
    picture = image_bytes("PNG", size=24)
    serve(api, "QI-1", encrypt(KEY, picture))
    quoted = image_item("QI-1", hex_key=KEY.hex())
    quoted["msg_id"] = "Q-IMG"
    [received] = await receive(h, api, text_item("这张图", ref_msg={"message_item": quoted}))
    assert received.quote is not None and received.quote.media_ref is not None
    assert received.quote.resolved
    assert h.media.read_bytes(received.quote.media_ref.sha256) == picture


async def test_a_quote_attaches_to_the_item_that_carries_it(
    api: respx.MockRouter, h: Harness
) -> None:
    serve(api, "DL-Q", encrypt(KEY, image_bytes("PNG")))
    quoted = {"type": 1, "text_item": {"text": "旧话"}}
    inbox = await receive(
        h,
        api,
        image_item("DL-Q", hex_key=KEY.hex()),
        text_item("引用在这里", ref_msg={"message_item": quoted}),
        mid=9,
    )
    assert [m.quote is not None for m in inbox] == [False, True]


async def test_a_quoted_voice_message_uses_its_transcript(
    api: respx.MockRouter, h: Harness
) -> None:
    quoted = {"type": 3, "voice_item": {"text": "语音里的话"}}
    [received] = await receive(h, api, text_item("嗯", ref_msg={"message_item": quoted}))
    assert received.quote is not None and received.quote.text == "语音里的话"


def test_a_partial_quote_selects_the_marked_part() -> None:
    full = "今天天气很好，今天适合出门，明天下雨"
    # the second "今天" up to the first "门"
    part = PartialText(start="今天", end="门", startindex=1, endindex=0)
    assert extract_partial(full, part) == "今天适合出门"


def test_the_hash_chooses_between_the_two_readings_of_the_end_index() -> None:
    full = "甲乙丙甲乙丙甲乙丙"
    # "the 1st 丙 in the whole text" vs "the 1st 丙 after the start"
    global_reading = "甲乙丙甲乙丙"
    after_start = "乙丙甲乙丙"
    first = PartialText(start="甲", end="丙", startindex=0, endindex=1)
    assert extract_partial(full, first) == global_reading  # no hash: the global reading
    md5 = hashlib.md5(global_reading.encode(), usedforsecurity=False).hexdigest()
    assert extract_partial(full, first.model_copy(update={"quotemd5": md5})) == global_reading
    other = PartialText(start="乙", end="丙", startindex=0, endindex=1)
    md5_other = hashlib.md5(after_start.encode(), usedforsecurity=False).hexdigest()
    chosen = extract_partial(full, other.model_copy(update={"quotemd5": md5_other}))
    assert chosen == after_start


def test_a_partial_quote_that_cannot_be_placed_returns_the_whole_text() -> None:
    full = "一二三"
    assert extract_partial(full, PartialText(start="九", end="三")) == full
    assert extract_partial(full, PartialText(start="一")) == full
    assert extract_partial(full, PartialText(start="三", end="一")) == full
    wrong_hash = PartialText(start="一", end="三", quotemd5="0" * 32)
    assert extract_partial(full, wrong_hash) == full


async def test_the_partial_selection_is_applied_to_a_resolved_quote(
    api: respx.MockRouter, h: Harness
) -> None:
    ref = {
        "message_item": {"type": 1, "text_item": {"text": "前面 重点内容 后面"}},
        "partial_text": {"start": "重点", "end": "内容", "startindex": 0, "endindex": 0},
    }
    [received] = await receive(h, api, text_item("这段", ref_msg=ref))
    assert received.quote is not None and received.quote.text == "重点内容"


async def test_a_flaky_download_is_retried_before_giving_up(
    api: respx.MockRouter, h: Harness
) -> None:
    picture = image_bytes("PNG")
    route = api.get(f"{CDN}/download", params={"encrypted_query_param": "DL-R"}).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=encrypt(KEY, picture))]
    )
    [received] = await receive(h, api, image_item("DL-R", hex_key=KEY.hex()))
    assert route.call_count == 2 and received.media_ref is not None


async def test_a_crash_while_converting_one_message_does_not_stop_the_batch(
    api: respx.MockRouter, h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = InboundConverter.convert

    async def flaky(self: InboundConverter, wire: WireMessage, message_id: str) -> Any:
        if message_id == "1":
            raise RuntimeError("converter bug")
        return await real(self, wire, message_id)

    monkeypatch.setattr(InboundConverter, "convert", flaky)
    now = now_ms(h.clock)
    batch = [
        message(text_item("坏"), mid=1, created_ms=now),
        message(text_item("好"), mid=2, created_ms=now),
    ]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C"))
    await h.channel.poll_once()
    assert [m.text for m in h.store.inbox()] == ["好"]
    assert h.store.item_stats().failures["convert_error"] == 1
