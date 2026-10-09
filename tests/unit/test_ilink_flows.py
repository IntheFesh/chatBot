"""The interactive login and binding flows (R-CH-003, R-CH-007, D-012)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    BOT,
    CTX,
    OTHER,
    TOKEN,
    USER,
    ScriptedPrompter,
    advance_until,
    drive,
    message,
    text_item,
    updates,
)
from twin.channel.base import AuthState
from twin.channel.ilink import flows as flows_module
from twin.channel.ilink.flows import ConsoleLoginUI, run_binding, run_login
from twin.channel.ilink.login import LoginError
from twin.channel.ilink.store import BatchCommit, Credentials, IlinkStore, PendingBinding
from twin.channel.state import ChannelStateStore
from twin.services import Services

QR = f"{API}/ilink/bot/get_bot_qrcode"
STATUS = f"{API}/ilink/bot/get_qrcode_status"
GET_UPDATES = f"{API}/ilink/bot/getupdates"
NEW_TOKEN = "BOT-TOKEN-SECOND-LOGIN"
OLD_TOKEN = "OLD-TOKEN"


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        for notice in ("notifystart", "notifystop"):
            router.post(f"{API}/ilink/bot/msg/{notice}").respond(200, json={})
        yield router


def store_of(services: Services) -> IlinkStore:
    return IlinkStore(ChannelStateStore(services.db), services.clock)


def scan_succeeds(api: respx.MockRouter, *, token: str = TOKEN, scanner: str | None = USER) -> None:
    api.post(QR).respond(200, json={"qrcode": "Q1", "qrcode_img_content": "https://qr.example/1"})
    body: dict[str, Any] = {
        "status": "confirmed",
        "bot_token": token,
        "ilink_bot_id": BOT,
        "baseurl": API,
    }
    if scanner:
        body["ilink_user_id"] = scanner
    api.get(STATUS).respond(200, json=body)


def seed_login(services: Services, token: str = OLD_TOKEN) -> IlinkStore:
    store = store_of(services)
    store.save_credentials(Credentials(token, BOT, USER, API, services.clock.now_utc().isoformat()))
    return store


def user_writes(api: respx.MockRouter, services: Services, sender: str = USER) -> None:
    raw = message(text_item("第一条消息"), mid=1, sender=sender, context_token=CTX)
    api.post(GET_UPDATES).mock(
        side_effect=[httpx.Response(200, json=updates([raw], cursor="C1"))]
        + [httpx.Response(200, json=updates(cursor="C1"))] * 200
    )


# ------------------------------------------------------------------ login


async def test_scanning_and_then_writing_to_the_bot_binds_that_account(
    api: respx.MockRouter, services: Services
) -> None:
    scan_succeeds(api)
    user_writes(api, services)
    prompter = ScriptedPrompter(True)
    outcome = await drive(
        run_login(services, prompter, open_viewer=False, app_running=False, bind_wait_s=60),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert outcome.logged_in_now and outcome.bound
    store = store_of(services)
    credentials = store.credentials()
    assert credentials is not None and credentials.bot_token == TOKEN
    assert credentials.ilink_user_id == USER
    bound = store.bound_user()
    assert bound is not None and bound.user_id == USER
    context = store.context_token()
    assert context is not None and context.token == CTX  # kept from the first message
    assert store.inbox() == []  # that first message itself is not processed
    assert store.pending_binding() is None
    transcript = prompter.transcript
    assert "Scan this code" in transcript and "https://qr.example/1" in transcript
    assert USER not in transcript  # the full id is never printed
    assert not list(services.paths.tmp_dir.glob("ilink-login-*.png"))  # the picture is removed


async def test_the_running_application_records_the_sender_and_the_command_only_waits(
    api: respx.MockRouter, services: Services
) -> None:
    scan_succeeds(api)
    polled = api.post(GET_UPDATES).respond(200, json=updates())
    store = store_of(services)
    prompter = ScriptedPrompter(True)

    async def application_sees_a_message() -> None:
        batch = BatchCommit(
            candidate=PendingBinding(USER, services.clock.now_utc(), CTX, True), new_cursor="C"
        )
        store.commit_batch(batch)

    task = asyncio.ensure_future(
        run_login(services, prompter, open_viewer=False, app_running=True, bind_wait_s=60)
    )
    clock: ManualClock = services.clock  # type: ignore[assignment]
    await advance_until(clock, lambda: "Now send any message" in prompter.transcript)
    await application_sees_a_message()
    await advance_until(clock, task.done)
    outcome = await task
    assert outcome.bound and store.bound_user() is not None
    assert polled.call_count == 0  # the command did not poll: the application owns the cursor


async def test_a_sender_that_is_not_the_scanner_is_refused_by_default(
    api: respx.MockRouter, services: Services
) -> None:
    scan_succeeds(api, scanner=USER)
    user_writes(api, services, sender=OTHER)
    prompter = ScriptedPrompter("no")
    outcome = await drive(
        run_login(services, prompter, open_viewer=False, bind_wait_s=60),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert outcome.logged_in_now and not outcome.bound
    store = store_of(services)
    assert store.bound_user() is None and store.pending_binding() is None
    assert "NOT the account that scanned" in prompter.transcript


async def test_a_sender_that_is_not_the_scanner_can_still_be_bound_by_typing_the_word(
    api: respx.MockRouter, services: Services
) -> None:
    scan_succeeds(api, scanner=USER)
    user_writes(api, services, sender=OTHER)
    outcome = await drive(
        run_login(services, ScriptedPrompter("bind"), open_viewer=False, bind_wait_s=60),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert outcome.bound
    bound = store_of(services).bound_user()
    assert bound is not None and bound.user_id == OTHER


async def test_a_working_login_is_reused_unless_forced(
    api: respx.MockRouter, services: Services
) -> None:
    store = seed_login(services)
    store.bind(USER, context_token=CTX)
    qr = api.post(QR).respond(200, json={"qrcode": "Q1", "qrcode_img_content": "https://qr/1"})
    prompter = ScriptedPrompter()
    outcome = await run_login(services, prompter, open_viewer=False, bind_wait_s=5)
    assert not outcome.logged_in_now and outcome.bound
    assert "Already logged in" in prompter.transcript and qr.call_count == 0
    assert store.credentials().bot_token == OLD_TOKEN  # type: ignore[union-attr]


async def test_forcing_a_new_scan_replaces_the_token_and_resets_the_poll_state(
    api: respx.MockRouter, services: Services
) -> None:
    store = seed_login(services)
    store.bind(USER, context_token=CTX)
    store.state.put("ilink.cursor", "STALE")
    scan_succeeds(api, token=NEW_TOKEN)
    outcome = await drive(
        run_login(services, ScriptedPrompter(), force=True, open_viewer=False, bind_wait_s=5),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert outcome.logged_in_now and outcome.bound
    assert store.credentials().bot_token == NEW_TOKEN  # type: ignore[union-attr]
    assert store.cursor() == "" and store.context_token() is None  # reset by the new login
    assert store.bound_user() is not None  # the binding survives a re-login
    assert store.auth_record().state is AuthState.OK


async def test_a_different_account_scanning_for_a_bound_bot_is_rejected_without_changes(
    api: respx.MockRouter, services: Services
) -> None:
    store = seed_login(services)
    store.bind(USER, context_token=CTX)
    scan_succeeds(api, token=NEW_TOKEN, scanner=OTHER)
    with pytest.raises(LoginError, match="not the bound account"):
        await drive(
            run_login(services, ScriptedPrompter(), force=True, open_viewer=False),
            services.clock,  # type: ignore[arg-type]
            step_s=1.0,
        )
    assert store.credentials().bot_token == OLD_TOKEN  # type: ignore[union-attr]
    assert store.context_token() is not None  # nothing was reset


async def test_a_login_after_expiry_clears_the_needs_relogin_state(
    api: respx.MockRouter, services: Services
) -> None:
    store = seed_login(services)
    store.bind(USER, context_token=CTX)
    store.mark_needs_relogin(-14, None)
    scan_succeeds(api, token=NEW_TOKEN)
    outcome = await drive(
        run_login(services, ScriptedPrompter(), open_viewer=False),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert outcome.logged_in_now
    assert store.auth_record().state is AuthState.OK


# ---------------------------------------------------------------- binding


async def test_a_waiting_binding_gives_up_after_the_time_limit(
    api: respx.MockRouter, services: Services
) -> None:
    seed_login(services)
    api.post(GET_UPDATES).respond(200, json=updates())
    prompter = ScriptedPrompter()
    bound = await drive(
        run_binding(services, prompter, app_running=False, wait_s=6),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert bound is False and "No message arrived" in prompter.transcript


async def test_binding_stops_waiting_when_the_login_turns_out_to_be_dead(
    api: respx.MockRouter, services: Services
) -> None:
    seed_login(services)
    api.post(GET_UPDATES).respond(200, json={"ret": -14})
    prompter = ScriptedPrompter()
    bound = await drive(
        run_binding(services, prompter, app_running=False, wait_s=600),
        services.clock,  # type: ignore[arg-type]
        step_s=1.0,
    )
    assert bound is False and "twin channel login --force" in prompter.transcript


async def test_a_candidate_found_earlier_is_confirmed_without_waiting(
    services: Services,
) -> None:
    store = seed_login(services)
    store.commit_batch(
        BatchCommit(candidate=PendingBinding(USER, services.clock.now_utc(), CTX, True))
    )
    prompter = ScriptedPrompter(True)
    assert await run_binding(services, prompter, app_running=True, wait_s=5)
    assert "Now send any message" not in prompter.transcript
    assert store.bound_user() is not None


async def test_declining_the_confirmation_leaves_nobody_bound(services: Services) -> None:
    store = seed_login(services)
    store.commit_batch(
        BatchCommit(candidate=PendingBinding(USER, services.clock.now_utc(), CTX, True))
    )
    assert not await run_binding(services, ScriptedPrompter(False), app_running=True, wait_s=5)
    assert store.bound_user() is None and store.pending_binding() is None


# ------------------------------------------------------------ console UI


def test_the_console_ui_prints_the_code_saves_the_picture_and_cleans_up(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Any] = []
    monkeypatch.setattr(flows_module, "open_in_viewer", lambda path: opened.append(path) or True)
    prompter = ScriptedPrompter("1234")
    ui = ConsoleLoginUI(prompter, services.paths.tmp_dir, open_viewer=True)
    ui.show_qr("https://qr.example/9", number=2, total=3)
    pictures = list(services.paths.tmp_dir.glob("ilink-login-*.png"))
    assert len(pictures) == 1 and opened == pictures
    assert "code 2 of 3" in prompter.transcript and "https://qr.example/9" in prompter.transcript
    assert "opened" in prompter.transcript
    ui.show_qr("https://qr.example/10", number=3, total=3)
    assert len(list(services.paths.tmp_dir.glob("ilink-login-*.png"))) == 1  # old one removed
    assert ui.ask_verify_code(previous_was_wrong=True) == "1234"
    assert "not accepted" in prompter.transcript
    ui.info("hello")
    ui.cleanup()
    assert not list(services.paths.tmp_dir.glob("ilink-login-*.png"))


def test_the_console_ui_still_works_when_the_picture_cannot_be_saved(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(content: str, directory: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(flows_module, "save_png", broken)
    prompter = ScriptedPrompter()
    ConsoleLoginUI(prompter, services.paths.tmp_dir, open_viewer=False).show_qr(
        "https://qr.example/1", number=1, total=3
    )
    assert "could not save the picture" in prompter.transcript
    assert "https://qr.example/1" in prompter.transcript  # the link is the fallback


def test_the_viewer_can_be_switched_off(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(path: object) -> bool:
        raise AssertionError("the viewer must not be opened")

    monkeypatch.setattr(flows_module, "open_in_viewer", never)
    ConsoleLoginUI(ScriptedPrompter(), services.paths.tmp_dir, open_viewer=False).show_qr(
        "https://qr.example/1", number=1, total=3
    )
