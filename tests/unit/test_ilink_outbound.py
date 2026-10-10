"""Sending: text, images, typing, window and quota gates, error classes (R-CH-006 to R-CH-008)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    CDN,
    CTX,
    OTHER,
    TOKEN,
    USER,
    AllowingBypass,
    Harness,
    SetAllowList,
    b64_hex,
    gif_bytes,
    image_bytes,
    make_harness,
    message,
    now_ms,
    request_json,
    text_item,
    updates,
)
from tests.support.synthetic import mobile
from tests.support.waiting import wait_until
from twin.channel.base import (
    BypassRefused,
    CapabilityNotSupported,
    MediaNotAllowed,
    OutboundKind,
    QuoteTarget,
    RecipientNotAllowed,
)
from twin.channel.ilink.outbound import (
    TICKET_TTL_S,
    TYPING_KEEPALIVE_S,
    TYPING_MAX_S,
    IlinkSender,
)
from twin.channel.policy import CompositeMediaPolicy, sha256_hex
from twin.storage.db import Database

SEND = f"{API}/ilink/bot/sendmessage"
GET_UPDATES = f"{API}/ilink/bot/getupdates"
UPLOAD_URL = f"{API}/ilink/bot/getuploadurl"
GETCONFIG = f"{API}/ilink/bot/getconfig"
SENDTYPING = f"{API}/ilink/bot/sendtyping"
STICKER = image_bytes("GIF", (9, 99, 199))
ANIMATED = gif_bytes()
OK = httpx.Response(200, json={})


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def h(db: Database, clock: ManualClock, tmp_path: Path) -> AsyncIterator[Harness]:
    policy = CompositeMediaPolicy([SetAllowList(sha256_hex(STICKER), sha256_hex(ANIMATED))])
    harness = make_harness(db, clock, tmp_path, RecordingAlerts(), policy=policy)
    harness.login()
    harness.bind()
    yield harness
    await harness.channel.stop()


def decrypt(key: bytes, data: bytes) -> bytes:
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()  # noqa: S305
    unpadder = padding.PKCS7(128).unpadder()
    padded = decryptor.update(data) + decryptor.finalize()
    return unpadder.update(padded) + unpadder.finalize()


# ------------------------------------------------------------------- text


async def test_a_text_message_is_one_item_with_the_context_token(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    result = await h.channel.send_text("你好")
    assert result.ok and result.kind is OutboundKind.OK and result.message_id is None

    request = route.calls.last.request
    body = request_json(request)
    msg = body["msg"]
    assert msg["item_list"] == [{"type": 1, "text_item": {"text": "你好"}}]
    assert msg["message_type"] == 2 and msg["message_state"] == 2
    assert msg["from_user_id"] == "" and msg["to_user_id"] == USER
    assert msg["context_token"] == CTX
    assert msg["client_id"].startswith("wechat-twin:") and result.client_id == msg["client_id"]
    assert body["base_info"]["channel_version"]
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert h.store.window_state().outbound_since_inbound == 1


async def test_a_success_body_with_a_message_id_indexes_the_text_for_later_quotes(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(SEND).respond(200, json={"ret": 0, "message_id": 123})
    result = await h.channel.send_text("机器人说的话")
    assert result.ok and result.message_id == "123"
    assert h.store.quote_text("123") == "机器人说的话"


async def test_every_bubble_gets_its_own_client_id_and_counts_once(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    await h.channel.send_text("一")
    await h.channel.send_text("二")
    ids = [request_json(call.request)["msg"]["client_id"] for call in route.calls]
    assert len(set(ids)) == 2
    assert h.channel.session_state().outbound_since_inbound == 2


async def test_only_the_bound_user_can_be_named_as_the_recipient(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    with pytest.raises(RecipientNotAllowed):
        await h.channel.send_text("给别人", recipient=OTHER)
    with pytest.raises(RecipientNotAllowed):
        await h.channel.send_image(STICKER, "image/gif", recipient=OTHER)
    with pytest.raises(RecipientNotAllowed):
        await h.channel.send_typing(True, recipient=OTHER)
    assert route.call_count == 0
    assert (await h.channel.send_text("给他", recipient=USER)).ok


async def test_nothing_can_be_sent_while_nobody_is_bound(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    route = api.post(SEND).respond(200, json={})
    try:
        with pytest.raises(RecipientNotAllowed, match="no user is bound"):
            await harness.channel.send_text("嗨")
        assert route.call_count == 0
    finally:
        await harness.channel.stop()


async def test_empty_and_oversized_text_is_refused_before_any_request(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    empty = await h.channel.send_text("  \n ")
    assert not empty.ok and empty.reason == "empty_text"
    too_long = await h.channel.send_text("长" * 4001)
    assert not too_long.ok and too_long.reason == "text_too_long"
    assert route.call_count == 0
    assert (await h.channel.send_text("长" * 4000)).ok


async def test_quotes_cannot_be_sent_because_the_protocol_has_no_such_item(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    caps = h.channel.capabilities()
    assert caps.supports_quote is False and caps.supports_typing is True
    assert caps.gif_animated is None
    assert caps.proactive_window_h is None and caps.outbound_quota is None  # not measured yet
    with pytest.raises(CapabilityNotSupported, match="no way to send a quote"):
        await h.channel.send_text("回复", quote=QuoteTarget("1", "旧话"))
    assert route.call_count == 0


# -------------------------------------------------------- local gate checks


async def test_without_a_context_token_nothing_is_sent(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    harness.bind(with_context=False)
    route = api.post(SEND).respond(200, json={})
    try:
        result = await harness.channel.send_text("嗨")
        assert not result.ok and result.kind is OutboundKind.WINDOW_REJECTED
        assert result.reason == "no_context_token" and route.call_count == 0
    finally:
        await harness.channel.stop()


async def test_without_a_login_or_after_it_expired_nothing_is_sent(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    route = api.post(SEND).respond(200, json={})
    try:
        harness.bind()
        first = await harness.channel.send_text("嗨")
        assert first.kind is OutboundKind.AUTH_EXPIRED and first.reason == "not_logged_in"
        harness.login()
        harness.store.mark_needs_relogin(-14, None)
        second = await harness.channel.send_text("嗨")
        assert second.kind is OutboundKind.AUTH_EXPIRED and second.reason == "needs_relogin"
        assert route.call_count == 0
    finally:
        await harness.channel.stop()


async def test_the_safe_quota_stops_the_ninth_bubble_and_an_inbound_message_restores_it(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    for number in range(8):
        assert (await h.channel.send_text(f"第{number}条")).ok
    assert h.channel.session_state().remaining_quota == 0
    refused = await h.channel.send_text("第9条")
    assert not refused.ok and refused.kind is OutboundKind.WINDOW_REJECTED
    assert refused.reason == "quota_exhausted" and route.call_count == 8

    batch = [message(text_item("我回来了"), mid=1, created_ms=now_ms(clock))]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()
    assert h.channel.session_state().remaining_quota == 8
    assert (await h.channel.send_text("好的")).ok


async def test_the_safe_window_stops_sending_after_twenty_two_hours(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    clock.tick(21 * 3600 + 59 * 60)
    assert (await h.channel.send_text("还在窗口内")).ok
    clock.tick(2 * 60)
    refused = await h.channel.send_text("窗口过了")
    assert refused.reason == "window_elapsed" and route.call_count == 1
    state = h.channel.session_state()
    assert state.window_remaining is not None and state.window_remaining.total_seconds() < 0


# --------------------------------------------------------- server answers


async def test_a_platform_refusal_closes_the_window_and_is_never_retried(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={"ret": -2, "errmsg": "prepare failed"})
    result = await h.channel.send_text("嗨")
    assert not result.ok and result.kind is OutboundKind.WINDOW_REJECTED
    assert result.session_expired is True and result.code == -2
    assert result.errmsg == "prepare failed"
    assert route.call_count == 1  # one attempt, no retry
    state = h.channel.session_state()
    assert state.expired and h.store.window_state().last_error["code"] == -2  # type: ignore[index]

    for _ in range(3):  # later tries are refused locally: no loop against the server
        again = await h.channel.send_text("再试")
        assert again.reason == "session_expired" and again.session_expired
    assert route.call_count == 1

    api.post(GET_UPDATES).respond(
        200,
        json=updates([message(text_item("我又来了"), mid=2, created_ms=now_ms(clock))], cursor="C"),
    )
    await h.channel.poll_once()  # the user wrote again: the window reopens
    route.respond(200, json={})
    assert not h.channel.session_state().expired
    assert (await h.channel.send_text("欢迎回来")).ok


async def test_other_server_errors_are_reported_without_closing_the_window(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={"ret": 1, "errmsg": f"bad request from {mobile()}"})
    result = await h.channel.send_text("嗨")
    assert result.kind is OutboundKind.REJECTED and result.code == 1 and not result.session_expired
    assert mobile() not in (result.errmsg or "")  # the message is redacted before it is kept
    assert not h.channel.session_state().expired
    assert h.channel.session_state().outbound_since_inbound == 0  # a refusal costs no quota
    assert route.call_count == 1


async def test_a_dead_login_during_a_send_raises_the_alert(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(SEND).respond(200, json={"ret": -14})
    result = await h.channel.send_text("嗨")
    assert result.kind is OutboundKind.AUTH_EXPIRED and result.code == -14
    assert h.alerts.categories() == ["channel.auth_expired"]
    assert h.store.auth_record().state.value == "needs_relogin"
    again = await h.channel.send_text("嗨")
    assert again.reason == "needs_relogin"


@pytest.mark.parametrize(
    ("failure", "kind", "counted"),
    [
        (httpx.Response(500), OutboundKind.AMBIGUOUS, True),
        (httpx.Response(502), OutboundKind.AMBIGUOUS, True),
        (httpx.ReadTimeout("slow"), OutboundKind.AMBIGUOUS, True),
        (httpx.ReadError("reset"), OutboundKind.AMBIGUOUS, True),
        (httpx.Response(200, text="<html>gateway</html>"), OutboundKind.AMBIGUOUS, True),
        (httpx.ConnectError("down"), OutboundKind.NETWORK, False),
        (httpx.ConnectTimeout("slow"), OutboundKind.NETWORK, False),
        (httpx.Response(400), OutboundKind.REJECTED, False),
        (httpx.Response(403), OutboundKind.REJECTED, False),
    ],
)
async def test_unknown_outcomes_are_not_resent_and_only_possible_deliveries_use_quota(
    failure: httpx.Response | Exception,
    kind: OutboundKind,
    counted: bool,
    api: respx.MockRouter,
    h: Harness,
) -> None:
    route = api.post(SEND).mock(side_effect=[failure])
    result = await h.channel.send_text("嗨")
    assert not result.ok and result.kind is kind
    assert route.call_count == 1
    assert h.channel.session_state().outbound_since_inbound == (1 if counted else 0)
    assert h.store.window_state().last_error is not None
    assert result.session_expired is False


async def test_concurrent_sends_keep_their_order_and_are_counted_exactly(
    api: respx.MockRouter, h: Harness
) -> None:
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request_json(request)["msg"]["item_list"][0]["text_item"]["text"])
        return httpx.Response(200, json={})

    api.post(SEND).mock(side_effect=record)
    results = await asyncio.gather(*(h.channel.send_text(f"消息{i}") for i in range(5)))
    assert all(r.ok for r in results)
    assert sorted(seen) == [f"消息{i}" for i in range(5)] and len(seen) == 5
    assert h.channel.session_state().outbound_since_inbound == 5


# ------------------------------------------------------------------ images


async def sticker_routes(api: respx.MockRouter, picture: bytes) -> dict[str, respx.Route]:
    return {
        "upload_url": api.post(UPLOAD_URL).respond(200, json={"upload_param": "UP-1"}),
        "cdn": api.post(url__startswith=f"{CDN}/upload").respond(
            200, headers={"x-encrypted-param": "DL-OUT"}
        ),
        "send": api.post(SEND).respond(200, json={}),
    }


async def test_an_allowed_image_is_encrypted_uploaded_and_sent_as_one_item(
    api: respx.MockRouter, h: Harness
) -> None:
    routes = await sticker_routes(api, STICKER)
    result = await h.channel.send_image(STICKER, "image/gif")
    assert result.ok

    request_body = request_json(routes["upload_url"].calls.last.request)
    key = bytes.fromhex(request_body["aeskey"])
    assert len(key) == 16 and request_body["media_type"] == 1
    assert request_body["to_user_id"] == USER and request_body["no_need_thumb"] is True
    assert request_body["rawsize"] == len(STICKER)
    assert request_body["rawfilemd5"] == hashlib.md5(STICKER, usedforsecurity=False).hexdigest()
    uploaded = routes["cdn"].calls.last.request.content
    assert len(uploaded) == request_body["filesize"] == (len(STICKER) // 16 + 1) * 16
    assert decrypt(key, uploaded) == STICKER  # the CDN receives the picture encrypted with the key

    item = request_json(routes["send"].calls.last.request)["msg"]["item_list"]
    assert len(item) == 1 and item[0]["type"] == 2
    media = item[0]["image_item"]["media"]
    assert media["encrypt_query_param"] == "DL-OUT" and media["encrypt_type"] == 1
    assert media["aes_key"] == b64_hex(key)
    assert base64.b64decode(media["aes_key"]).decode() == key.hex()
    assert item[0]["image_item"]["mid_size"] == len(uploaded)
    assert request_json(routes["send"].calls.last.request)["msg"]["context_token"] == CTX
    assert h.channel.session_state().outbound_since_inbound == 1


async def test_a_picture_that_is_not_allowed_never_leaves_the_machine(
    api: respx.MockRouter, h: Harness
) -> None:
    routes = await sticker_routes(api, STICKER)
    photo = image_bytes("JPEG", (1, 1, 1), size=64)
    with pytest.raises(MediaNotAllowed, match="neither a sticker"):
        await h.channel.send_image(photo, "image/jpeg")
    assert sum(route.call_count for route in routes.values()) == 0
    assert h.channel.session_state().outbound_since_inbound == 0


async def test_the_declared_type_must_match_the_bytes_and_be_an_image_type(
    api: respx.MockRouter, h: Harness
) -> None:
    routes = await sticker_routes(api, STICKER)
    with pytest.raises(MediaNotAllowed, match="declared type"):
        await h.channel.send_image(STICKER, "image/png")  # the bytes are a GIF
    with pytest.raises(MediaNotAllowed, match="not sent"):
        await h.channel.send_image(STICKER, "application/pdf")
    assert sum(route.call_count for route in routes.values()) == 0


async def test_an_image_can_come_from_a_file_and_jpg_is_an_alias_of_jpeg(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    jpeg = image_bytes("JPEG", (5, 5, 5), size=33)
    policy = CompositeMediaPolicy([SetAllowList(sha256_hex(jpeg))])
    harness = make_harness(db, clock, tmp_path, RecordingAlerts(), policy=policy)
    harness.login()
    harness.bind()
    path = tmp_path / "sticker.jpg"
    path.write_bytes(jpeg)
    routes = await sticker_routes(api, jpeg)
    try:
        assert (await harness.channel.send_image(path, "image/jpg")).ok
        assert routes["cdn"].call_count == 1
    finally:
        await harness.channel.stop()


async def test_an_animated_gif_is_sent_unchanged(api: respx.MockRouter, h: Harness) -> None:
    routes = await sticker_routes(api, ANIMATED)
    assert (await h.channel.send_image(ANIMATED, "image/gif")).ok
    key = bytes.fromhex(request_json(routes["upload_url"].calls.last.request)["aeskey"])
    assert decrypt(key, routes["cdn"].calls.last.request.content) == ANIMATED


async def test_an_upload_problem_stops_before_the_message_is_sent(
    api: respx.MockRouter, h: Harness
) -> None:
    send = api.post(SEND).respond(200, json={})
    api.post(UPLOAD_URL).respond(200, json={})
    no_address = await h.channel.send_image(STICKER, "image/gif")
    assert (
        no_address.kind is OutboundKind.UPLOAD_FAILED and "no upload address" in no_address.reason
    )

    api.post(UPLOAD_URL).respond(200, json={"upload_param": "P"})
    cdn = api.post(url__startswith=f"{CDN}/upload").respond(400)
    refused = await h.channel.send_image(STICKER, "image/gif")
    assert refused.kind is OutboundKind.UPLOAD_FAILED and refused.http_status == 400
    assert cdn.call_count == 1  # a client error is final
    assert send.call_count == 0
    assert h.channel.session_state().outbound_since_inbound == 0


async def test_a_dead_login_at_the_upload_address_request_is_reported(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(UPLOAD_URL).respond(200, json={"ret": -14})
    result = await h.channel.send_image(STICKER, "image/gif")
    assert result.kind is OutboundKind.AUTH_EXPIRED
    assert h.alerts.categories() == ["channel.auth_expired"]


async def test_images_obey_the_same_window_gate_as_text(api: respx.MockRouter, h: Harness) -> None:
    routes = await sticker_routes(api, STICKER)
    h.store.mark_needs_relogin(-14, None)
    result = await h.channel.send_image(STICKER, "image/gif")
    assert result.reason == "needs_relogin"
    assert routes["upload_url"].call_count == 0


# ------------------------------------------------------------------ typing


async def test_typing_starts_with_a_ticket_keeps_alive_every_five_seconds_and_cancels(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    config = api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "TICKET-1"})
    typing = api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    assert request_json(config.calls.last.request)["ilink_user_id"] == USER
    assert request_json(config.calls.last.request)["context_token"] == CTX
    first = request_json(typing.calls.last.request)
    assert first["status"] == 1 and first["typing_ticket"] == "TICKET-1"
    assert first["ilink_user_id"] == USER

    await wait_until(lambda: clock.pending_sleepers >= 1)
    await clock.advance(5)
    await wait_until(lambda: typing.call_count >= 2)
    assert request_json(typing.calls.last.request)["status"] == 1  # keep-alive

    await h.channel.send_typing(False)
    assert request_json(typing.calls.last.request)["status"] == 2  # cancel
    count = typing.call_count
    await clock.advance(30)
    assert typing.call_count == count  # the keep-alive is gone
    assert config.call_count == 1  # the ticket was cached


async def test_the_keep_alive_ends_by_itself_after_three_minutes_with_a_cancel(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    """The bot that started typing and was never told to stop does not type for ever."""
    assert isinstance(h.channel._sender, IlinkSender)  # the sender the channel is made of
    api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "TICKET-1"})
    typing = api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    await wait_until(lambda: clock.pending_sleepers >= 1)
    beats = int(TYPING_MAX_S / TYPING_KEEPALIVE_S)
    for _ in range(beats):
        await clock.advance(TYPING_KEEPALIVE_S)
    await wait_until(lambda: typing.call_count >= beats + 2)
    statuses = [request_json(call.request)["status"] for call in typing.calls]
    assert statuses == [1] * (beats + 1) + [2]  # the start, a beat every 5 s for 180 s, a cancel
    count = typing.call_count
    await clock.advance(60)
    assert typing.call_count == count  # and then nothing


async def test_the_ticket_is_asked_for_again_after_ten_minutes(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    config = api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "TICKET-1"})
    api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    await h.channel.send_typing(False)
    await clock.advance(TICKET_TTL_S - 1)
    await h.channel.send_typing(True)
    await h.channel.send_typing(False)
    assert config.call_count == 1  # a minute short of ten: the ticket is still good
    await clock.advance(2)
    config.respond(200, json={"ret": 0, "typing_ticket": "TICKET-2"})
    await h.channel.send_typing(True)
    assert config.call_count == 2  # ten minutes of the (monotonic) clock: a new one
    await h.channel.send_typing(False)


async def test_typing_without_a_ticket_is_silent(api: respx.MockRouter, h: Harness) -> None:
    typing = api.post(SENDTYPING).respond(200, json={})
    api.post(GETCONFIG).mock(
        side_effect=[
            httpx.Response(200, json={"ret": 0}),
            httpx.Response(200, json={"ret": 1, "errmsg": "no"}),
            httpx.ConnectError("down"),
        ]
    )
    for _ in range(3):
        await h.channel.send_typing(True)
    await h.channel.send_typing(False)
    assert typing.call_count == 0


async def test_typing_request_failures_are_swallowed(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    typing = api.post(SENDTYPING).mock(side_effect=httpx.ConnectError("down"))
    await h.channel.send_typing(True)
    await h.channel.send_typing(False)
    assert typing.call_count == 2


async def test_a_dead_login_noticed_through_typing_is_recorded(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(GETCONFIG).respond(200, json={"ret": -14})
    await h.channel.send_typing(True)
    assert h.store.auth_record().state.value == "needs_relogin"
    h.store.save_credentials(h.store.credentials())  # type: ignore[arg-type]
    api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    api.post(SENDTYPING).respond(200, json={"ret": -14})
    h.channel._sender._tickets.clear()
    await h.channel.send_typing(True)
    assert h.store.auth_record().state.value == "needs_relogin"


async def test_typing_is_skipped_while_logged_out(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.bind()
    config = api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    try:
        await harness.channel.send_typing(True)
        await harness.channel.send_typing(False)
        assert config.call_count == 0
    finally:
        await harness.channel.stop()


async def test_typing_stops_by_itself_after_three_minutes(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    typing = api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    for _ in range(36):
        await wait_until(lambda: clock.pending_sleepers >= 1)
        await clock.advance(5)
    await wait_until(lambda: request_json(typing.calls.last.request)["status"] == 2)
    count = typing.call_count
    await clock.advance(60)
    assert typing.call_count == count


# ------------------------------------------------------------ safe bypass


async def test_a_bypass_lets_the_probe_pass_the_safe_thresholds_only(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    clock.tick(23 * 3600)
    refused = await h.channel.send_text("[测试]晚了")
    assert refused.reason == "window_elapsed" and route.call_count == 0

    bypass = AllowingBypass()
    allowed = await h.channel.send_text("[测试]晚了", bypass=bypass)
    assert allowed.ok and route.call_count == 1
    [request] = bypass.requests
    assert request.kind == "text" and request.text == "[测试]晚了"
    assert request.gate_reason == "window_elapsed"  # what was bypassed is on record


async def test_a_bypass_is_also_consulted_when_nothing_needs_bypassing(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(SEND).respond(200, json={})
    bypass = AllowingBypass()
    await h.channel.send_text("[测试]", bypass=bypass)
    assert bypass.requests[0].gate_reason is None


async def test_a_refused_bypass_sends_nothing(api: respx.MockRouter, h: Harness) -> None:
    route = api.post(SEND).respond(200, json={})
    with pytest.raises(BypassRefused):
        await h.channel.send_text("不是测试消息", bypass=AllowingBypass(refuse=True))
    assert route.call_count == 0


async def test_a_bypass_can_exceed_the_safe_quota_but_never_reopen_a_refused_session(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(SEND).respond(200, json={})
    bypass = AllowingBypass()
    for number in range(10):
        assert (await h.channel.send_text(f"[测试]{number}", bypass=bypass)).ok
    assert route.call_count == 10
    assert bypass.requests[8].gate_reason == "quota_exhausted"

    route.respond(200, json={"ret": -2, "errmsg": "prepare failed"})
    assert (await h.channel.send_text("[测试]x", bypass=bypass)).session_expired
    calls = route.call_count
    after = await h.channel.send_text("[测试]y", bypass=bypass)
    assert after.reason == "session_expired" and route.call_count == calls  # no retry loop


async def test_a_bypass_does_not_replace_the_context_token(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    harness.bind(with_context=False)
    route = api.post(SEND).respond(200, json={})
    try:
        result = await harness.channel.send_text("[测试]", bypass=AllowingBypass())
        assert result.reason == "no_context_token" and route.call_count == 0
    finally:
        await harness.channel.stop()


async def test_every_outgoing_request_goes_to_the_official_hosts_only(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(SEND).respond(200, json={})
    api.post(UPLOAD_URL).respond(200, json={"upload_param": "P"})
    api.post(url__startswith=f"{CDN}/upload").respond(200, headers={"x-encrypted-param": "D"})
    await h.channel.send_text("文字")
    await h.channel.send_image(STICKER, "image/gif")
    hosts = {call.request.url.host for route in api.routes for call in route.calls}
    assert hosts <= {"ilinkai.weixin.qq.com", "novac2c.cdn.weixin.qq.com"}


async def test_the_typing_ticket_is_reused_until_it_gets_old(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    config = api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    await h.channel.send_typing(True)  # a second "typing" before the first ended restarts it
    assert config.call_count == 1
    await h.channel.send_typing(False)
    clock.tick(11 * 60)
    await h.channel.send_typing(True)
    assert config.call_count == 2  # the cached ticket expired
    await h.channel.send_typing(False)


async def test_a_typing_notice_after_the_login_vanished_sends_nothing(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "T"})
    typing = api.post(SENDTYPING).respond(200, json={})
    await h.channel.send_typing(True)
    h.state.delete("ilink.credentials")
    await h.channel.send_typing(False)
    assert typing.call_count == 1  # only the start; nothing could be sent without a login
