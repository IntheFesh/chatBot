"""The channel as a whole: lifecycle, crash safety, stored state (R-CH-002, R-CH-004, R-CH-007)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import respx

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    BOT,
    CTX,
    OTHER,
    TOKEN,
    USER,
    Harness,
    advance_until,
    make_harness,
    message,
    now_ms,
    request_json,
    text_item,
    updates,
)
from twin.app import Application, HealthStatus
from twin.channel.base import AuthState, Channel, InboundMessage
from twin.channel.component import register_channel
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.ilink.store import QUOTE_INDEX_MAX, Credentials, IlinkStore
from twin.config.loader import load_settings
from twin.services import Services
from twin.storage.db import Database

GET_UPDATES = f"{API}/ilink/bot/getupdates"
START = f"{API}/ilink/bot/msg/notifystart"
STOP = f"{API}/ilink/bot/msg/notifystop"


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


# ----------------------------------------------------------------- lifecycle


async def test_the_channel_implements_the_shared_interface(h: Harness) -> None:
    assert isinstance(h.channel, Channel)
    assert h.channel.health().status is HealthStatus.OK


async def test_start_announces_the_bot_and_stop_says_goodbye(
    api: respx.MockRouter, h: Harness
) -> None:
    started = api.post(START).respond(200, json={"ret": 0, "errmsg": ""})
    stopped = api.post(STOP).respond(200, json={"ret": 0, "errmsg": ""})
    api.post(GET_UPDATES).respond(200, json=updates())
    await h.channel.start()
    await h.channel.start()  # a second start changes nothing
    assert started.call_count == 1
    sent = started.calls.last.request
    assert sent.headers["authorization"] == f"Bearer {TOKEN}"
    assert "base_info" in request_json(sent)
    await h.channel.stop()
    await h.channel.stop()
    assert stopped.call_count == 1


async def test_failing_lifecycle_notices_do_not_stop_the_channel(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(START).respond(500)
    api.post(STOP).mock(side_effect=httpx.ConnectError("down"))
    api.post(GET_UPDATES).respond(200, json=updates())
    await h.channel.start()
    await h.channel.stop()  # no exception


async def test_no_notices_are_sent_without_a_login(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    started = api.post(START).respond(200, json={})
    api.post(GET_UPDATES).respond(200, json=updates())
    await harness.channel.start()
    await harness.channel.stop()
    assert started.call_count == 0


async def test_a_send_only_channel_neither_polls_nor_announces_itself(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts(), poll=False)
    harness.login()
    harness.bind()
    started = api.post(START).respond(200, json={})
    stopped = api.post(STOP).respond(200, json={})
    polled = api.post(GET_UPDATES).respond(200, json=updates())
    send = api.post(f"{API}/ilink/bot/sendmessage").respond(200, json={})
    await harness.channel.start()
    assert (await harness.channel.send_text("[测试]你好")).ok
    await harness.channel.stop()
    assert (started.call_count, stopped.call_count, polled.call_count) == (0, 0, 0)
    assert send.call_count == 1


async def test_the_running_channel_polls_in_the_background_and_delivers(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    api.post(START).respond(200, json={})
    api.post(STOP).respond(200, json={})
    batch = [message(text_item("后台收到"), mid=31, created_ms=now_ms(clock))]
    api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(200, json=updates(batch, cursor="C1"))]
        + [httpx.Response(200, json=updates(cursor="C1"))] * 100
    )
    await h.channel.start()
    received: list[InboundMessage] = []

    async def consume() -> None:
        async for incoming in h.channel.incoming():
            received.append(incoming)

    consumer = asyncio.create_task(consume())
    try:
        await advance_until(clock, lambda: bool(received))
    finally:
        await h.channel.stop()
        await asyncio.wait_for(consumer, 5)
    assert [m.text for m in received] == ["后台收到"]
    assert h.store.inbox() == []  # acknowledged when the loop ended


async def test_a_crash_between_receiving_and_saving_loses_nothing(
    api: respx.MockRouter, h: Harness, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.post(START).respond(200, json={})
    api.post(STOP).respond(200, json={})
    batch = [message(text_item("不能丢"), mid=41, created_ms=now_ms(clock))]
    api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(200, json=updates(batch, cursor="C1"))]
        + [httpx.Response(200, json=updates(batch, cursor="C1"))]  # the server repeats it
        + [httpx.Response(200, json=updates(cursor="C1"))] * 100
    )
    real_commit = IlinkStore.commit_batch
    failures = {"left": 1}

    def flaky(self: IlinkStore, batch_: object) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("power cut while saving")
        real_commit(self, batch_)  # type: ignore[arg-type]

    monkeypatch.setattr(IlinkStore, "commit_batch", flaky)
    await h.channel.start()
    try:
        await advance_until(
            clock, lambda: [m.text for m in h.store.inbox()] == ["不能丢"], step_s=2.0
        )
    finally:
        await h.channel.stop()
    assert "task_crashed" in h.alerts.categories()  # the supervisor noticed and restarted it
    assert h.store.seen_ids() == ["41"]


# ------------------------------------------------------------ session state


async def test_the_session_state_reports_login_binding_and_window(
    api: respx.MockRouter, db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts(), quota=5)
    try:
        state = harness.channel.session_state()
        assert state.auth is AuthState.NOT_LOGGED_IN and not state.bound
        assert state.last_inbound_at is None and state.window_remaining is None
        assert state.remaining_quota == 5 and not state.has_context_token

        harness.login()
        harness.bind()
        state = harness.channel.session_state()
        assert state.auth is AuthState.OK and state.bound and state.has_context_token
        assert state.last_inbound_at == clock.now_utc()
        assert state.window_remaining is not None
        assert state.window_remaining.total_seconds() == 22 * 3600

        harness.store.mark_needs_relogin(-14, "x")
        assert harness.channel.session_state().auth is AuthState.NEEDS_RELOGIN
    finally:
        await harness.channel.stop()


async def test_the_channel_takes_its_limits_from_the_configuration(services: Services) -> None:
    channel = IlinkChannel.from_services(services, poll=False)
    try:
        window = channel.window()
        assert window.window_h == services.settings.channel.proactive_window_safe_h == 22
        assert window.quota == services.settings.channel.outbound_quota_safe == 8
    finally:
        await channel.stop()


async def test_measured_limits_appear_in_the_capabilities(
    db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    channel = IlinkChannel(
        db=db,
        clock=clock,
        media=make_harness(db, clock, tmp_path, RecordingAlerts()).media,
        alerts=RecordingAlerts(),
        window_h=22,
        quota=8,
        measured_window_h=24.0,
        measured_quota=10,
        gif_animated=True,
    )
    try:
        caps = channel.capabilities()
        assert (caps.proactive_window_h, caps.outbound_quota, caps.gif_animated) == (24.0, 10, True)
        assert caps.max_text_chars == 4000
    finally:
        await channel.stop()


# ---------------------------------------------------- stored state and binding


async def test_unbinding_forgets_the_conversation_but_keeps_the_login(
    api: respx.MockRouter, h: Harness, clock: ManualClock
) -> None:
    batch = [message(text_item("还没读"), mid=51, created_ms=now_ms(clock))]
    api.post(GET_UPDATES).respond(200, json=updates(batch, cursor="C5"))
    await h.channel.poll_once()
    assert h.store.inbox() and h.store.context_token() is not None
    assert h.store.unbind() is True
    assert h.store.bound_user() is None and h.store.inbox() == []
    assert h.store.context_token() is None and h.store.window_state().last_inbound_at is None
    assert h.store.credentials() is not None and h.store.cursor() == "C5"
    assert h.store.unbind() is False


async def test_the_quote_index_is_bounded_and_replaces_repeated_ids(h: Harness) -> None:
    for number in range(QUOTE_INDEX_MAX + 20):
        h.store.add_quote(f"id-{number}", f"text {number}")
    assert h.store.quote_text("id-0") is None  # the oldest entries were dropped
    assert h.store.quote_text(f"id-{QUOTE_INDEX_MAX + 19}") == f"text {QUOTE_INDEX_MAX + 19}"
    h.store.add_quote("id-300", "changed")
    assert h.store.quote_text("id-300") == "changed"
    h.store.add_quote("long", "x" * 1000)
    assert len(h.store.quote_text("long") or "") == 200


async def test_the_quote_index_forgets_old_entries(h: Harness, clock: ManualClock) -> None:
    h.store.add_quote("old", "an old line")
    clock.tick(31 * 24 * 3600)
    h.store.add_quote("new", "a new line")
    assert h.store.quote_text("old") is None and h.store.quote_text("new") == "a new line"


async def test_acknowledging_an_unknown_message_changes_nothing(h: Harness) -> None:
    assert h.store.ack("nothing") is False


async def test_credentials_are_stored_encrypted_not_in_plaintext(h: Harness, db: Database) -> None:
    with closing(sqlite3.connect(db.path)) as raw:
        rows = raw.execute("SELECT key, value FROM channel_state").fetchall()
    assert {key for key, _ in rows} >= {"ilink.credentials", "ilink.bound_user"}
    for _key, blob in rows:
        assert TOKEN.encode() not in blob and USER.encode() not in blob and BOT.encode() not in blob
        assert CTX.encode() not in blob and OTHER.encode() not in blob


# ------------------------------------------------------------- the component


async def test_the_application_runs_the_channel_and_reports_the_login_in_its_health(
    services: Services,
) -> None:
    application = Application()
    component = register_channel(application, services)
    assert component is not None
    assert [c.name for c in application.start_order()] == ["channel"]
    await application.start()
    try:
        health = application.health()["channel"]
        assert health.status is HealthStatus.DEGRADED and "not logged in" in health.detail
        component.channel.store.save_credentials(
            Credentials(TOKEN, BOT, USER, API, services.clock.now_utc().isoformat())
        )
        assert application.health()["channel"].status is HealthStatus.OK
        component.channel.store.mark_needs_relogin(-14, None)
        expired = application.health()["channel"]
        assert expired.status is HealthStatus.DEGRADED and "login expired" in expired.detail
    finally:
        await application.stop()


async def test_the_channel_component_is_not_added_for_the_console_channel(
    services: Services, tmp_path: Path
) -> None:
    settings = load_settings(
        None, {"paths": {"data_dir": str(tmp_path / "console")}, "channel": {"kind": "console"}}
    )
    application = Application()
    assert register_channel(application, replace(services, settings=settings)) is None
    assert application.components == {}
