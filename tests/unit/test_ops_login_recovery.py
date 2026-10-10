"""A lost login fixes itself when someone scans: the QR window, never a mail (R-OPS-004)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.ilink import API, BOT, OTHER, TOKEN, USER
from tests.support.ops import RecordingMailer, RecordingNotifier
from tests.support.waiting import wait_until
from twin.channel.base import AuthState
from twin.channel.ilink.login import LoginError, LoginResult
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore
from twin.ops import login_recovery as recovery_module
from twin.ops.alert_delivery import AlertDelivery
from twin.ops.login_recovery import (
    CHECK_S,
    LoginRecovery,
    ViewerQrWindow,
    WindowLoginUI,
    default_window,
)
from twin.services import Services

QR = f"{API}/ilink/bot/get_bot_qrcode"
STATUS = f"{API}/ilink/bot/get_qrcode_status"
QR_CONTENT = "https://qr.example/secret-login-code-1"
NEW_TOKEN = "BOT-TOKEN-AFTER-THE-SCAN"
OLD_TOKEN = "OLD-TOKEN"


@dataclass
class RecordingWindow:
    """A window that remembers what it was asked and answers the verification box."""

    answers: list[str | None] = field(default_factory=list)
    shown: list[tuple[str, bool]] = field(default_factory=list)  # (title, picture existed)
    pictures: list[Path] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    closed: int = 0

    def show(self, png: Path, title: str) -> None:
        self.pictures.append(png)
        self.shown.append((title, png.is_file()))

    def status(self, text: str) -> None:
        self.statuses.append(text)

    def ask_code(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else None

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def store_of(services: Services) -> IlinkStore:
    return IlinkStore(ChannelStateStore(services.db), services.clock)


def lost_login(services: Services, bound: str = USER) -> IlinkStore:
    store = store_of(services)
    store.save_credentials(
        Credentials(OLD_TOKEN, BOT, bound, API, services.clock.now_utc().isoformat())
    )
    store.bind(bound)
    store.mark_needs_relogin(-14, "session timeout")
    services.alerts.raise_alert(
        "channel.auth_expired", "the WeChat login expired", severity="critical", dedup_key="login"
    )
    return store


def scan_succeeds(api: respx.MockRouter, scanner: str = USER) -> None:
    api.post(QR).respond(200, json={"qrcode": "Q1", "qrcode_img_content": QR_CONTENT})
    api.get(STATUS).respond(
        200,
        json={
            "status": "confirmed",
            "bot_token": NEW_TOKEN,
            "ilink_bot_id": BOT,
            "ilink_user_id": scanner,
            "baseurl": API,
        },
    )


def make(services: Services, window: RecordingWindow, **options: Any) -> LoginRecovery:
    return LoginRecovery(services, window=lambda: window, **options)


async def test_a_scan_restores_the_login_and_closes_the_alert(
    api: respx.MockRouter, services: Services
) -> None:
    store = lost_login(services)
    scan_succeeds(api)
    window = RecordingWindow()
    recovery = make(services, window)
    assert recovery.needed()
    assert await recovery.check_once() is True
    credentials = store.credentials()
    assert credentials is not None and credentials.bot_token == NEW_TOKEN
    assert store.auth_record().state is AuthState.OK and not recovery.needed()
    assert recovery.attempts == 1 and recovery.recoveries == 1
    (title, existed) = window.shown[0]
    assert existed and "第 1/" in title  # the picture was there while the window showed it
    assert window.closed == 1 and not window.pictures[0].exists()  # and is removed afterwards
    assert services.alerts.open_alerts() == []
    recovered = [a for a in services.alerts.recent() if a.kind == "recovery"]
    assert len(recovered) == 1 and recovered[0].category == "login_lost"
    assert await recovery.check_once() is False and recovery.attempts == 1  # nothing to do now


async def test_the_qr_code_never_reaches_a_mail_or_a_notification(
    api: respx.MockRouter, services: Services, clock: ManualClock
) -> None:
    lost_login(services)
    scan_succeeds(api)
    notifier, mailer = RecordingNotifier(), RecordingMailer()
    delivery = AlertDelivery(
        services.alerts,
        clock,
        notifier=notifier,
        mailer=mailer,
        recipient=lambda: "me@example.org",
        zone=lambda: ZoneInfo("America/Chicago"),
    )
    window = RecordingWindow()
    await make(services, window).check_once()
    await delivery.deliver_once()
    assert notifier.shown and mailer.sent  # the alert and the recovery notice went out
    everything = " ".join(
        [t + " " + b for t, b in notifier.shown]
        + [m.subject + m.text + (m.html or "") for m in mailer.sent]
    )
    for secret in (QR_CONTENT, "qr.example", "qrcode", NEW_TOKEN, OLD_TOKEN, "png"):
        assert secret not in everything, secret
    assert any("扫码" in m.text for m in mailer.sent)  # it says to scan, on the computer


async def test_the_verification_number_is_asked_in_the_window(
    api: respx.MockRouter, services: Services
) -> None:
    store = lost_login(services)
    api.post(QR).respond(200, json={"qrcode": "Q1", "qrcode_img_content": QR_CONTENT})
    confirmed = {
        "status": "confirmed",
        "bot_token": NEW_TOKEN,
        "ilink_bot_id": BOT,
        "ilink_user_id": USER,
        "baseurl": API,
    }
    api.get(STATUS).mock(
        side_effect=[
            httpx.Response(200, json={"status": "need_verifycode"}),
            httpx.Response(200, json=confirmed),
        ]
    )
    window = RecordingWindow(answers=["123456"])
    assert await make(services, window).check_once() is True
    assert window.prompts == ["请输入手机上显示的数字"]
    sent = [
        str(call.request.url) for call in api.calls if "get_qrcode_status" in str(call.request.url)
    ]
    assert any("verify_code=123456" in url for url in sent)
    credentials = store.credentials()
    assert credentials is not None and credentials.bot_token == NEW_TOKEN


async def test_no_number_typed_means_a_failed_attempt_and_the_alert_stays(
    api: respx.MockRouter, services: Services
) -> None:
    store = lost_login(services)
    api.post(QR).respond(200, json={"qrcode": "Q1", "qrcode_img_content": QR_CONTENT})
    api.get(STATUS).respond(200, json={"status": "need_verifycode"})
    window = RecordingWindow(answers=[None])
    recovery = make(services, window)
    assert await recovery.check_once() is False
    assert recovery.attempts == 1 and recovery.recoveries == 0
    assert store.auth_record().state is AuthState.NEEDS_RELOGIN
    assert [a.category for a in services.alerts.open_alerts()] == ["login_lost"]
    assert window.closed == 1


async def test_the_account_that_scanned_must_be_the_bound_one(
    api: respx.MockRouter, services: Services
) -> None:
    store = lost_login(services)
    scan_succeeds(api, scanner=OTHER)
    recovery = make(services, RecordingWindow())
    assert await recovery.check_once() is False
    credentials = store.credentials()
    assert credentials is not None and credentials.bot_token == OLD_TOKEN  # nothing was changed
    assert store.auth_record().state is AuthState.NEEDS_RELOGIN
    assert TOKEN not in str(services.alerts.recent())


async def test_a_failed_attempt_is_repeated_after_the_retry_time(
    services: Services, clock: ManualClock
) -> None:
    lost_login(services)
    calls: list[int] = []

    async def runner(ui: WindowLoginUI) -> LoginResult:
        calls.append(1)
        if len(calls) == 1:
            raise LoginError("timed out waiting for the scan")
        return LoginResult(NEW_TOKEN, BOT, USER, API)

    recovery = make(services, RecordingWindow(), runner=runner, retry_s=300)
    assert await recovery.check_once() is False and len(calls) == 1
    clock.tick(299)
    assert await recovery.check_once() is False and len(calls) == 1  # too early
    clock.tick(2)
    # the runner above does not store anything; the real flow does, so the state stays "lost":
    assert await recovery.check_once() is True and len(calls) == 2


async def test_the_component_watches_the_login_by_itself(
    api: respx.MockRouter, services: Services, clock: ManualClock
) -> None:
    store = store_of(services)
    store.save_credentials(Credentials(OLD_TOKEN, BOT, USER, API, clock.now_utc().isoformat()))
    store.bind(USER)
    scan_succeeds(api)
    window = RecordingWindow()
    recovery = make(services, window)
    await recovery.start()
    try:
        await clock.advance(CHECK_S)
        assert recovery.attempts == 0  # the login is fine: nothing to do
        store.mark_needs_relogin(-14, "session timeout")
        await clock.advance(CHECK_S)
        await wait_until(lambda: recovery.recoveries == 1, limit_s=10)
        assert recovery.health().status.value == "ok"
    finally:
        await recovery.stop()
    assert not recovery.needed() and window.closed == 1


# ----------------------------------------------------------------------------- the windows


def test_without_tk_the_picture_opens_in_the_default_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(recovery_module, "open_in_viewer", lambda path: opened.append(path) or True)
    window = default_window("linux")
    assert isinstance(window, ViewerQrWindow)
    picture = tmp_path / "ilink-login-1.png"
    window.show(picture, "title")
    window.status("scanned")
    window.close()
    assert opened == [picture] and window.ask_code("number?") is None


def test_the_login_screens_use_the_window_and_clean_up_the_pictures(tmp_path: Path) -> None:
    window = RecordingWindow(answers=["42", None])
    ui = WindowLoginUI(window, tmp_path)
    ui.show_qr(QR_CONTENT, number=2, total=5)
    assert window.shown == [("微信登录失效：请用手机微信扫码（第 2/5 张）", True)]
    assert ui.ask_verify_code(previous_was_wrong=False) == "42"
    with pytest.raises(LoginError, match="not entered"):
        ui.ask_verify_code(previous_was_wrong=True)
    assert window.prompts[1].endswith("（上一次没有被接受）")
    ui.info("scanned")
    assert window.statuses == ["scanned"]
    ui.cleanup()
    assert not list(tmp_path.glob("*.png")) and window.closed == 1


def test_a_second_picture_replaces_the_first(tmp_path: Path) -> None:
    window = RecordingWindow()
    ui = WindowLoginUI(window, tmp_path)
    ui.show_qr("one", number=1, total=5)
    ui.show_qr("two", number=2, total=5)
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert asyncio.iscoroutinefunction(LoginRecovery.recover)
