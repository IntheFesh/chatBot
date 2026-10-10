"""Long polling, cursor, de-duplication, back-off and login expiry (R-CH-004, R-CH-003)."""

from __future__ import annotations

import asyncio
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
    CTX,
    OTHER,
    TOKEN,
    USER,
    Harness,
    make_harness,
    message,
    now_ms,
    request_json,
    run_until,
    text_item,
    updates,
)
from twin.channel.base import MessageKind
from twin.channel.ilink.poller import PollOutcome, backoff_delay
from twin.channel.ilink.store import IlinkStore
from twin.storage.db import Database

GET_UPDATES = f"{API}/ilink/bot/getupdates"
BIG_ID = 9_223_372_036_854_775_001


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        for notice in ("notifystart", "notifystop"):
            router.post(f"{API}/ilink/bot/msg/{notice}").respond(200, json={"ret": 0, "errmsg": ""})
        yield router


@pytest.fixture
async def h(db: Database, clock: ManualClock, tmp_path: Path) -> AsyncIterator[Harness]:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    harness.bind()
    yield harness
    await harness.channel.stop()


def hello(clock: ManualClock, text: str = "你好", mid: int | str | None = BIG_ID) -> dict[str, Any]:
    return message(text_item(text), mid=mid, created_ms=now_ms(clock))


# ------------------------------------------------------------------ a batch


async def test_a_batch_is_stored_and_the_cursor_moves_after_it(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(GET_UPDATES).respond(200, json=updates([hello(clock)], cursor="CURSOR-2"))
    assert await h.channel.poll_once() is PollOutcome.MESSAGES

    request = route.calls.last.request
    body = request_json(request)
    assert body["get_updates_buf"] == "" and "base_info" in body
    assert body["base_info"]["bot_agent"].startswith("WechatTwin/")
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["authorizationtype"] == "ilink_bot_token"
    assert request.headers["ilink-app-id"] == "bot"

    [stored] = h.store.inbox()
    assert stored.id == str(BIG_ID)  # a 64-bit id survives exactly, as text
    assert stored.kind is MessageKind.TEXT and stored.text == "你好"
    assert h.store.cursor() == "CURSOR-2"
    assert h.store.seen_ids() == [str(BIG_ID)]
    token = h.store.context_token()
    assert token is not None and token.token == CTX


async def test_the_saved_cursor_is_sent_on_the_next_poll_even_after_a_restart(
    api: respx.MockRouter, h: Harness, clock: ManualClock, db: Database, tmp_path: Path
) -> None:
    route = api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json=updates([hello(clock)], cursor="CURSOR-2")),
            httpx.Response(200, json=updates(cursor="CURSOR-3")),
        ]
    )
    await h.channel.poll_once()
    restarted = make_harness(db, clock, tmp_path / "again", RecordingAlerts())
    try:
        assert await restarted.channel.poll_once() is PollOutcome.EMPTY
    finally:
        await restarted.channel.stop()
    assert request_json(route.calls[1].request)["get_updates_buf"] == "CURSOR-2"
    assert h.store.cursor() == "CURSOR-3"


async def test_an_empty_answer_and_a_bare_object_change_nothing(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json=updates(cursor=None)),
            httpx.Response(200, json={}),
            httpx.Response(200, json={"ret": 0, "msgs": None}),
        ]
    )
    for _ in range(3):
        assert await h.channel.poll_once() is PollOutcome.EMPTY
    assert h.store.cursor() == "" and h.store.inbox() == []


async def test_the_server_suggested_timeout_is_used_for_the_next_poll(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(GET_UPDATES).respond(
        200, json=updates(cursor="C1", longpolling_timeout_ms=20_000)
    )
    await h.channel.poll_once()
    await h.channel.poll_once()
    first = route.calls[0].request.extensions["timeout"]["read"]
    second = route.calls[1].request.extensions["timeout"]["read"]
    assert first == 40.0  # 35 s default plus 5 s margin
    assert second == 25.0  # 20 s from the server plus 5 s margin


# ------------------------------------------------------------ de-duplication


async def test_a_message_seen_twice_is_delivered_once(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json=updates([hello(clock)], cursor="C1")),
            httpx.Response(
                200, json=updates([hello(clock), hello(clock, "二", BIG_ID + 1)], cursor="C2")
            ),
        ]
    )
    await h.channel.poll_once()
    await h.channel.poll_once()
    assert [m.text for m in h.store.inbox()] == ["你好", "二"]


async def test_the_item_message_id_is_used_when_the_message_has_none(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    raw = message(text_item("甲", msg_id="ITEM-1"), mid=None, created_ms=now_ms(clock))
    api.post(GET_UPDATES).respond(200, json=updates([raw, raw], cursor="C1"))
    await h.channel.poll_once()
    assert [m.id for m in h.store.inbox()] == ["ITEM-1"]
    assert h.store.seen_ids() == ["ITEM-1"]


async def test_a_message_without_any_id_still_deduplicates_by_content(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    raw = message(text_item("乙"), mid=None, created_ms=now_ms(clock))
    api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json=updates([raw], cursor="C1")),
            httpx.Response(200, json=updates([raw], cursor="C2")),
        ]
    )
    await h.channel.poll_once()
    await h.channel.poll_once()
    [only] = h.store.inbox()
    assert only.id.startswith("h-") and only.text == "乙"


async def test_the_seen_list_keeps_only_the_most_recent_ids(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    batch = [hello(clock, f"m{i}", 1000 + i) for i in range(520)]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()
    seen = h.store.seen_ids()
    assert len(seen) == 500 and seen[-1] == "1519" and seen[0] == "1020"


# -------------------------------------------------------------- atomicity


async def test_a_failure_while_saving_leaves_cursor_ids_and_inbox_untouched(
    api: respx.MockRouter,
    h: Harness,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = api.post(GET_UPDATES).respond(200, json=updates([hello(clock)], cursor="CURSOR-2"))

    def explode(self: IlinkStore, tx: object, entries: object) -> None:
        raise RuntimeError("the disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(IlinkStore, "_add_quotes", explode)
        with pytest.raises(RuntimeError, match="disk went away"):
            await h.channel.poll_once()
    assert h.store.cursor() == "" and h.store.seen_ids() == [] and h.store.inbox() == []

    # the server sends the same batch again because the cursor did not move; nothing is lost
    assert await h.channel.poll_once() is PollOutcome.MESSAGES
    assert [m.text for m in h.store.inbox()] == ["你好"]
    assert request_json(route.calls[-1].request)["get_updates_buf"] == ""
    assert h.store.cursor() == "CURSOR-2"


async def test_the_consumer_acknowledges_by_asking_for_the_next_message(
    api: respx.MockRouter, h: Harness, clock: ManualClock, db: Database, tmp_path: Path
) -> None:
    batch = [hello(clock, "一", 1), hello(clock, "二", 2)]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()

    first_run = h.channel.incoming()
    first = await anext(first_run)
    await first_run.aclose()  # the consumer stops before it is done with the first message
    assert first.text == "一"
    assert [m.text for m in h.store.inbox()] == ["一", "二"]

    restarted = make_harness(db, clock, tmp_path / "again", RecordingAlerts())
    try:
        second_run = restarted.channel.incoming()
        assert (await anext(second_run)).text == "一"  # delivered again after the restart
        assert (await anext(second_run)).text == "二"  # asking for more acknowledged the first
        assert [m.text for m in h.store.inbox()] == ["二"]
        await second_run.aclose()
    finally:
        await restarted.channel.stop()


async def test_incoming_waits_for_news_and_ends_when_the_channel_stops(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GET_UPDATES).respond(200, json=updates([hello(clock)], cursor="C1"))
    await h.channel.start()
    received: list[str] = []

    async def consume() -> None:
        async for incoming in h.channel.incoming():
            received.append(incoming.text or "")

    consumer = asyncio.create_task(consume())
    try:
        await run_until_received(h, clock, received)
        await h.channel.stop()
        await asyncio.wait_for(consumer, 5)
    finally:
        consumer.cancel()
    assert received == ["你好"]


async def run_until_received(h: Harness, clock: ManualClock, received: list[str]) -> None:
    from tests.support.waiting import wait_until

    for _ in range(50):
        if received:
            return
        await wait_until(lambda: bool(received) or clock.pending_sleepers >= 1)
        if not received:
            await clock.advance(1.0)
    raise AssertionError("the message was never delivered")


# ---------------------------------------------------------- who is accepted


async def test_echoes_and_strangers_are_dropped_but_marked_as_seen(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    batch = [
        message(text_item("机器人自己的话"), mid=11, message_type=2, created_ms=now_ms(clock)),
        message(text_item("陌生人"), mid=12, sender=OTHER, created_ms=now_ms(clock)),
        hello(clock, "用户", 13),
    ]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()
    assert [m.text for m in h.store.inbox()] == ["用户"]
    assert h.store.seen_ids() == ["11", "12", "13"]
    failures = h.store.item_stats().failures
    assert failures["bot_echo_skipped"] == 1 and failures["other_sender_dropped"] == 1


async def test_a_malformed_message_is_counted_and_the_rest_still_arrive(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    batch: list[Any] = [{"message_id": 5, "item_list": "not a list"}, "junk", hello(clock)]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()
    assert [m.text for m in h.store.inbox()] == ["你好"]
    assert h.store.item_stats().failures["invalid_message"] == 2


async def test_nothing_is_processed_until_a_user_is_bound(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    try:
        batch = [
            message(text_item("第一条"), mid=1, sender=OTHER, context_token="CTX-OTHER"),
            message(text_item("第二条"), mid=2, sender=USER),
        ]
        api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
        await harness.channel.poll_once()
        assert harness.store.inbox() == []  # nothing is handed to the engine
        assert harness.store.context_token() is None
        pending = harness.store.pending_binding()
        assert pending is not None and pending.user_id == OTHER  # only the first sender
        assert pending.context_token == "CTX-OTHER"
        assert pending.matches_expected is False  # the login said the scanner was USER
        assert harness.store.cursor() == "C1"
    finally:
        await harness.channel.stop()


async def test_a_candidate_that_matches_the_scanner_is_marked_as_matching(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    try:
        api.post(GET_UPDATES).respond(
            200, json=updates([message(text_item("hi"), mid=1)], cursor="C1")
        )
        await harness.channel.poll_once()
        pending = harness.store.pending_binding()
        assert pending is not None and pending.matches_expected is True
        harness.store.bind(pending.user_id, context_token=pending.context_token)
        assert harness.store.pending_binding() is None
        assert harness.store.context_token() is not None
        api.post(GET_UPDATES).respond(
            200, json=updates([message(text_item("再来"), mid=2)], cursor="C2")
        )
        await harness.channel.poll_once()
        assert [m.text for m in harness.store.inbox()] == ["再来"]
    finally:
        await harness.channel.stop()


async def test_a_login_without_a_scanner_id_cannot_be_compared(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login(expected_user=None)
    try:
        api.post(GET_UPDATES).respond(
            200, json=updates([message(text_item("hi"), mid=1)], cursor="C1")
        )
        await harness.channel.poll_once()
        pending = harness.store.pending_binding()
        assert pending is not None and pending.matches_expected is None
    finally:
        await harness.channel.stop()


async def test_polling_without_a_login_does_nothing(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    route = api.post(GET_UPDATES).respond(200, json=updates())
    try:
        assert await harness.channel.poll_once() is PollOutcome.NOT_LOGGED_IN
        assert route.call_count == 0
    finally:
        await harness.channel.stop()


async def test_the_api_host_from_the_login_is_used_for_polling(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login(base_url="https://api-host.example.com")
    harness.bind()
    route = api.post("https://api-host.example.com/ilink/bot/getupdates").respond(
        200, json=updates()
    )
    try:
        await harness.channel.poll_once()
        assert route.call_count == 1
    finally:
        await harness.channel.stop()


async def test_item_type_numbers_are_counted_without_any_content(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    batch = [
        message(text_item("文字"), mid=1, created_ms=now_ms(clock)),
        message({"type": 99, "mystery_item": {"x": 1}}, mid=2, created_ms=now_ms(clock)),
    ]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C1"))
    await h.channel.poll_once()
    stats = h.store.item_stats()
    assert stats.counts() == {1: 1, 99: 1}
    assert stats.failures == {"unknown_item_type": 1}


# ------------------------------------------------------------------ errors


async def test_a_read_timeout_is_a_quiet_empty_result(api: respx.MockRouter, h: Harness) -> None:
    api.post(GET_UPDATES).mock(side_effect=httpx.ReadTimeout("nothing to say"))
    assert await h.channel.poll_once() is PollOutcome.TIMEOUT
    assert h.store.poll_status().consecutive_failures == 0
    assert h.alerts.alerts == []


async def test_a_poll_that_waited_out_its_time_is_a_working_connection(
    api: respx.MockRouter, h: Harness
) -> None:
    """R-OPS-003: a quiet long poll must keep the "last successful poll" fresh."""
    api.post(GET_UPDATES).mock(side_effect=httpx.ReadTimeout("nothing to say"))
    assert h.store.poll_status().last_ok_at is None
    assert await h.channel.poll_once() is PollOutcome.TIMEOUT
    assert h.store.poll_status().last_ok_at == h.clock.now_utc()


async def test_the_success_time_is_written_at_most_once_a_minute_and_at_once_after_failures(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    h.store.record_poll_success()
    first = h.store.poll_status().last_ok_at
    assert first == clock.now_utc()
    clock.tick(30)
    h.store.record_poll_success()
    assert h.store.poll_status().last_ok_at == first  # too soon: nothing is written
    clock.tick(31)
    h.store.record_poll_success()
    assert h.store.poll_status().last_ok_at == clock.now_utc()  # a minute has passed
    h.store.record_poll_failure("network", None, "down")
    clock.tick(1)
    h.store.record_poll_success()  # recovering from a failure is written at once
    status = h.store.poll_status()
    assert status.last_ok_at == clock.now_utc() and status.consecutive_failures == 0


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("down"),
        httpx.ConnectTimeout("slow"),
        httpx.ReadError("reset"),
        httpx.Response(500),
        httpx.Response(502),
        httpx.Response(524),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"ret": 1, "errmsg": "ERRMSG"}),
        httpx.Response(200, json={"errcode": 7}),
    ],
)
async def test_network_and_server_errors_are_counted_as_failures(
    failure: httpx.Response | Exception, api: respx.MockRouter, h: Harness
) -> None:
    api.post(GET_UPDATES).mock(side_effect=[failure])
    assert await h.channel.poll_once() is PollOutcome.ERROR
    status = h.store.poll_status()
    assert status.consecutive_failures == 1 and status.last_error is not None
    assert h.store.auth_record().state.value == "ok"  # not an expired login


async def test_three_failures_in_a_row_raise_one_warning_and_a_success_clears_the_count(
    api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(500)] * 4 + [httpx.Response(200, json=updates())]
    )
    for _ in range(4):
        assert await h.channel.poll_once() is PollOutcome.ERROR
    assert h.alerts.categories() == ["channel.poll_failing"]  # once, at the third failure
    assert h.alerts.alerts[0].severity == "warning"
    assert h.store.poll_status().consecutive_failures == 4
    assert await h.channel.poll_once() is PollOutcome.EMPTY
    assert h.store.poll_status().consecutive_failures == 0
    assert route.call_count == 5


def test_the_delay_doubles_from_one_second_to_a_sixty_second_cap() -> None:
    assert [backoff_delay(n, 1.0) for n in range(1, 9)] == [1, 2, 4, 8, 16, 32, 60, 60]
    assert backoff_delay(50, 1.5) == 60  # jitter can never push it past the cap
    assert backoff_delay(1, 0.5) == 0.5 and backoff_delay(3, 1.5) == 6.0
    with pytest.raises(ValueError, match="starts at 1"):
        backoff_delay(0, 1.0)


async def test_the_loop_backs_off_between_failures_and_starts_over_after_a_success(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(500)] * 3
        + [httpx.Response(200, json=updates([hello(clock)], cursor="C1"))]
        + [httpx.Response(500)] * 2
        + [httpx.Response(200, json=updates())] * 50
    )
    await run_until(
        lambda stop: h.channel._poller.run(stop),
        clock,
        lambda: len(h.store.inbox()) == 1 and len(clock.sleeps) >= 6,
    )
    waits = clock.sleeps[:5]
    assert waits[:3] == [1.0, 2.0, 4.0]  # three failures
    assert waits[3:5] == [1.0, 2.0]  # after the success the sequence starts again


async def test_an_instant_empty_answer_does_not_make_the_loop_spin(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GET_UPDATES).respond(200, json=updates())
    await run_until(lambda stop: h.channel._poller.run(stop), clock, lambda: len(clock.sleeps) >= 3)
    assert all(s == 1.0 for s in clock.sleeps[:3])  # paced at one poll per second


async def test_read_timeouts_do_not_trigger_the_back_off(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    calls = 0

    def slow(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("held for the full time", request=request)

    api.post(GET_UPDATES).mock(side_effect=slow)

    async def start(stop: asyncio.Event) -> None:
        await h.channel._poller.run(stop)

    await run_until(start, clock, lambda: calls >= 4)
    assert h.store.poll_status().consecutive_failures == 0
    assert all(s <= 1.0 for s in clock.sleeps)  # only the anti-spin pause, never 2, 4, 8 ...


# ----------------------------------------------------------- expired login


@pytest.mark.parametrize(
    "answer",
    [
        {"ret": -14, "errmsg": "ERRMSG"},
        {"errcode": -14},
        {"ret": 0, "errcode": -14},
        {"ret": -14, "errcode": 5},
    ],
)
async def test_error_minus_14_ends_polling_and_asks_for_a_new_login(
    answer: dict[str, Any], api: respx.MockRouter, h: Harness
) -> None:
    route = api.post(GET_UPDATES).respond(200, json=answer)
    assert await h.channel.poll_once() is PollOutcome.AUTH_EXPIRED

    record = h.store.auth_record()
    assert record.state.value == "needs_relogin" and record.code == -14
    assert h.alerts.categories() == ["channel.auth_expired"]
    alert = h.alerts.alerts[0]
    assert alert.severity == "critical" and alert.dedup_key == "channel.auth_expired"
    assert "twin channel login" in alert.title
    [(title, lines)] = h.banner.shown
    assert "expired" in title and any("twin channel login" in line for line in lines)

    # nothing more is sent to the server and the alert is not repeated
    assert await h.channel.poll_once() is PollOutcome.NEEDS_RELOGIN
    assert route.call_count == 1 and len(h.alerts.alerts) == 1


async def test_the_state_survives_a_restart_and_a_new_login_resumes_polling(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json={"ret": -14}),
            httpx.Response(200, json=updates([hello(clock)], cursor="FRESH")),
        ]
    )
    await h.channel.poll_once()
    assert h.store.context_token() is not None
    h.login()  # `twin channel login` stores the new credentials
    assert h.store.auth_record().state.value == "ok"
    assert h.store.cursor() == "" and h.store.context_token() is None  # reset by the login
    assert await h.channel.poll_once() is PollOutcome.MESSAGES
    assert request_json(route.calls[1].request)["get_updates_buf"] == ""


async def test_the_old_token_is_tried_again_after_an_hour_and_recovery_is_reported(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(GET_UPDATES).mock(
        side_effect=[
            httpx.Response(200, json={"ret": -14}),
            httpx.Response(200, json={"ret": -14}),  # the first probe: still dead
            httpx.Response(200, json=updates(cursor="C9")),  # the second probe: alive
        ]
    )
    await h.channel.poll_once()
    clock.tick(30 * 60)
    assert await h.channel.poll_once() is PollOutcome.NEEDS_RELOGIN  # too early
    clock.tick(31 * 60)
    assert await h.channel.poll_once() is PollOutcome.AUTH_EXPIRED  # probe 1 fails quietly
    assert len(h.alerts.alerts) == 1
    clock.tick(30 * 60)
    assert await h.channel.poll_once() is PollOutcome.NEEDS_RELOGIN
    clock.tick(31 * 60)
    assert await h.channel.poll_once() is PollOutcome.EMPTY  # probe 2 succeeds
    assert h.store.auth_record().state.value == "ok"
    assert h.alerts.categories() == ["channel.auth_expired", "channel.auth_recovered"]
    assert route.call_count == 3


async def test_a_probe_that_gets_a_read_timeout_also_counts_as_recovered(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(200, json={"ret": -14}), httpx.ReadTimeout("quiet")]
    )
    await h.channel.poll_once()
    clock.tick(61 * 60)
    assert await h.channel.poll_once() is PollOutcome.TIMEOUT
    assert h.store.auth_record().state.value == "ok"


async def test_the_loop_waits_quietly_while_a_login_is_needed(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    route = api.post(GET_UPDATES).respond(200, json={"ret": -14})
    await run_until(lambda stop: h.channel._poller.run(stop), clock, lambda: len(clock.sleeps) >= 4)
    assert route.call_count == 1
    assert set(clock.sleeps[:4]) == {5.0}


async def test_an_expired_login_without_a_timestamp_is_probed_at_once(
    api: respx.MockRouter, h: Harness
) -> None:
    h.state.put("ilink.auth_state", {"state": "needs_relogin", "code": -14})
    route = api.post(GET_UPDATES).respond(200, json=updates(cursor="C1"))
    assert await h.channel.poll_once() is PollOutcome.EMPTY
    assert route.call_count == 1 and h.store.auth_record().state.value == "ok"


async def test_the_guard_reports_an_expiry_only_once_per_episode(h: Harness) -> None:
    from twin.channel.ilink.auth import AuthGuard

    guard = AuthGuard(h.store, h.alerts, h.banner)
    assert await guard.on_expired(source="test", code=-14, errmsg=None) is True
    assert await guard.on_expired(source="test", code=-14, errmsg=None) is False
    assert len(h.alerts.alerts) == 1 and len(h.banner.shown) == 1
    assert await guard.on_recovered() is True
    assert await guard.on_recovered() is False


async def test_a_message_without_content_still_renews_the_token_and_the_window(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    clock.tick(5 * 3600)
    empty = message(
        {"type": 11, "tool_call_start_item": {}},
        mid=70,
        context_token="CTX-FRESH",
        created_ms=now_ms(clock),
    )
    api.post(GET_UPDATES).respond(200, json=updates([empty], cursor="C1"))
    await h.channel.poll_once()
    assert h.store.inbox() == []  # nothing to hand over
    token = h.store.context_token()
    assert token is not None and token.token == "CTX-FRESH"
    assert h.channel.session_state().last_inbound_at == clock.now_utc()
