"""``twin channel``: login, status, send-test, unbind (R-CH-003, R-CH-007, R-CH-008)."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from tests.support.ilink import (
    API,
    BOT,
    CDN,
    CTX,
    OTHER,
    TOKEN,
    USER,
    b64_raw,
    encrypt,
    image_bytes,
    image_item,
    message,
    request_json,
    text_item,
    updates,
)
from tests.support.persona import sticker_scenario
from twin.channel.base import AuthState
from twin.channel.ilink.store import BatchCommit, Credentials, IlinkStore, PendingBinding
from twin.channel.state import ChannelStateStore
from twin.cli import app
from twin.clock import SystemClock
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.process_model import CommandKind, get_spec, iter_commands
from twin.services import Services, build_services
from twin.storage.state import read_state_version

runner = CliRunner()
SEND = f"{API}/ilink/bot/sendmessage"
QR = f"{API}/ilink/bot/get_bot_qrcode"
STATUS = f"{API}/ilink/bot/get_qrcode_status"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    return path


@pytest.fixture
def svc(data_dir: Path) -> Iterator[Services]:
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    yield services
    services.close()


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def store_of(services: Services) -> IlinkStore:
    return IlinkStore(ChannelStateStore(services.db), services.clock)


def logged_in(services: Services, *, bound: bool = True) -> IlinkStore:
    store = store_of(services)
    store.save_credentials(Credentials(TOKEN, BOT, USER, API, services.clock.now_utc().isoformat()))
    if bound:
        store.bind(USER, context_token=CTX)
    return store


# ------------------------------------------------------------------ help


def test_the_channel_commands_are_registered_with_their_process_model_class() -> None:
    result = runner.invoke(app, ["channel", "--help"])
    assert result.exit_code == 0
    for name in ("login", "status", "send-test", "unbind", "listen"):
        assert name in result.output
    commands = dict(iter_commands(app))
    kinds = {
        "channel login": CommandKind.LIGHT,
        "channel status": CommandKind.READ,
        "channel send-test": CommandKind.LIGHT,
        "channel unbind": CommandKind.LIGHT,
        "channel listen": CommandKind.EXCLUSIVE,
    }
    for name, kind in kinds.items():
        spec = get_spec(commands[name])
        assert spec is not None and spec.kind is kind, name


# ---------------------------------------------------------------- status


def test_status_of_a_fresh_installation_says_what_to_do(svc: Services) -> None:
    result = runner.invoke(app, ["channel", "status"])
    assert result.exit_code == 0, result.output
    assert "NOT LOGGED IN" in result.output and "twin channel login" in result.output
    assert "NOBODY" in result.output
    assert "application: not running" in result.output
    assert "recent inbound item types: none yet" in result.output


def test_status_shows_login_binding_window_and_item_types_without_any_content(
    svc: Services,
) -> None:
    store = logged_in(svc)
    store.commit_batch(
        BatchCommit(
            item_types=[1, 1, 2, 3, 99],
            failures=Counter({"image.decrypt_failed": 2}),
            quote_entries=[("m1", "some private words")],
        )
    )
    before = read_state_version_of(svc)
    result = runner.invoke(app, ["channel", "status"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "logged in as" in out and BOT not in out  # ids are masked
    assert "bound user: " in out and USER not in out
    assert "context token: present" in out
    assert "window: safe limit 22 h" in out and "expired by the platform: no" in out
    assert "quota: sent 0 since the last inbound, 8 of 8 left" in out
    assert "type 1 (text): 2" in out and "type 99 (UNKNOWN): 1" in out
    assert "image.decrypt_failed: 2" in out
    assert "some private words" not in out and TOKEN not in out
    assert read_state_version_of(svc) == before  # a READ command writes nothing


def test_status_reports_an_expired_login_and_a_waiting_candidate(svc: Services) -> None:
    store = logged_in(svc, bound=False)
    store.commit_batch(
        BatchCommit(
            candidate=PendingBinding(OTHER, svc.clock.now_utc(), CTX, matches_expected=False)
        )
    )
    waiting = runner.invoke(app, ["channel", "status"]).output
    assert "waiting for confirmation of" in waiting and "DOES NOT MATCH" in waiting
    store.mark_needs_relogin(-14, "ERRMSG")
    expired = runner.invoke(app, ["channel", "status"]).output
    assert "NEEDS RE-LOGIN" in expired and "(code -14)" in expired
    assert "twin channel login --force" in expired


# ----------------------------------------------------------------- login


def test_login_scans_binds_and_stores_the_credentials(svc: Services, api: respx.MockRouter) -> None:
    api.post(QR).respond(200, json={"qrcode": "Q", "qrcode_img_content": "https://qr.example/1"})
    api.get(STATUS).respond(
        200,
        json={
            "status": "confirmed",
            "bot_token": TOKEN,
            "ilink_bot_id": BOT,
            "baseurl": API,
            "ilink_user_id": USER,
        },
    )
    # R18-A3 (D-767): the first message arrives after the login, through the command's own poll.
    # (This test used to plant a candidate before any login existed and expect the new login to
    # keep it - the stale-candidate behaviour that a new login now clears.)
    api.post(f"{API}/ilink/bot/msg/notifystart").respond(200, json={})
    api.post(f"{API}/ilink/bot/msg/notifystop").respond(200, json={})
    first = message(text_item("第一条消息"), mid=1, sender=USER, context_token=CTX)
    api.post(f"{API}/ilink/bot/getupdates").mock(
        side_effect=[httpx.Response(200, json=updates([first], cursor="C1"))]
        + [httpx.Response(200, json=updates(cursor="C1"))] * 200
    )
    store = store_of(svc)
    result = runner.invoke(app, ["channel", "login", "--no-open"], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Scan this code with WeChat" in result.output
    assert "https://qr.example/1" in result.output
    assert "Logged in" in result.output and "Bound to" in result.output
    assert USER not in result.output and TOKEN not in result.output
    assert store.credentials() is not None and store.bound_user() is not None
    assert store.auth_record().state is AuthState.OK
    assert not list(svc.paths.tmp_dir.glob("ilink-login-*.png"))


def test_login_without_a_confirmed_binding_exits_with_an_error(
    svc: Services, api: respx.MockRouter
) -> None:
    logged_in(svc, bound=False)
    store_of(svc).commit_batch(
        BatchCommit(candidate=PendingBinding(USER, svc.clock.now_utc(), CTX, matches_expected=True))
    )
    result = runner.invoke(app, ["channel", "login"], input="n\n")
    assert result.exit_code == 1
    assert "no user is bound yet" in result.output
    assert store_of(svc).bound_user() is None


def test_login_reports_an_unreachable_service_without_a_traceback(
    svc: Services, api: respx.MockRouter
) -> None:
    api.post(QR).mock(side_effect=httpx.ConnectError("down"))
    result = runner.invoke(app, ["channel", "login", "--force", "--no-open"])
    assert result.exit_code == 1
    assert "cannot reach the login service" in result.output
    assert "Traceback" not in result.output


# ------------------------------------------------------------- send-test


def test_send_test_puts_the_test_prefix_in_front_and_reports_success(
    svc: Services, api: respx.MockRouter
) -> None:
    logged_in(svc)
    route = api.post(SEND).respond(200, json={"ret": 0, "message_id": 77})
    result = runner.invoke(app, ["channel", "send-test", "你好"])
    assert result.exit_code == 0, result.output
    body = request_json(route.calls.last.request)["msg"]
    assert body["item_list"][0]["text_item"]["text"] == "[测试]你好"
    assert body["to_user_id"] == USER
    assert "sent to the bound user" in result.output and "[测试]你好" in result.output
    again = runner.invoke(app, ["channel", "send-test", "[测试]已有前缀"])
    assert again.exit_code == 0
    assert (
        request_json(route.calls.last.request)["msg"]["item_list"][0]["text_item"]["text"]
        == "[测试]已有前缀"
    )
    assert read_state_version_of(svc) >= 1  # LIGHT: the window write bumped the state version


def read_state_version_of(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


def test_send_test_refuses_when_nobody_is_bound(svc: Services, api: respx.MockRouter) -> None:
    logged_in(svc, bound=False)
    route = api.post(SEND).respond(200, json={})
    result = runner.invoke(app, ["channel", "send-test", "你好"])
    assert result.exit_code == 1 and "no user is bound yet" in result.output
    assert route.call_count == 0


def test_send_test_explains_why_nothing_was_sent(svc: Services, api: respx.MockRouter) -> None:
    store = logged_in(svc)
    route = api.post(SEND).respond(200, json={})
    old = svc.clock.now_utc() - timedelta(hours=30)
    store.state.put(
        "ilink.window",
        {"last_inbound_at": old.isoformat(), "outbound_since_inbound": 0, "expired": False},
    )
    result = runner.invoke(app, ["channel", "send-test", "你好"])
    assert result.exit_code == 1
    assert "window_rejected: window_elapsed" in result.output
    assert "send the bot a message from your phone first" in result.output
    assert route.call_count == 0


def test_send_test_shows_the_platform_error(svc: Services, api: respx.MockRouter) -> None:
    logged_in(svc)
    api.post(SEND).respond(200, json={"ret": -2, "errmsg": "prepare failed"})
    result = runner.invoke(app, ["channel", "send-test", "你好"])
    assert result.exit_code == 1
    assert "code -2" in result.output and "prepare failed" in result.output


def test_send_test_before_login_points_to_the_login_command(
    svc: Services, api: respx.MockRouter
) -> None:
    store_of(svc).bind(USER, context_token=CTX)
    result = runner.invoke(app, ["channel", "send-test", "你好"])
    assert result.exit_code == 1 and "twin channel login" in result.output


# ---------------------------------------------------------------- unbind


def test_unbind_needs_two_confirmations_and_forgets_the_conversation(svc: Services) -> None:
    store = logged_in(svc)
    before = read_state_version_of(svc)
    result = runner.invoke(app, ["channel", "unbind"], input="y\nunbind\n")
    assert result.exit_code == 0, result.output
    assert "unbound" in result.output and USER not in result.output
    assert store.bound_user() is None and store.context_token() is None
    # R17-53a (D-743, user decision): the login goes with the binding - the bot token is deleted
    assert store.credentials() is None and store.cursor() == ""
    assert store.auth_record().state is AuthState.NOT_LOGGED_IN
    assert "forgot the login" in result.output and TOKEN not in result.output
    assert read_state_version_of(svc) > before  # the running application is told


@pytest.mark.parametrize("answers", ["n\n", "y\nnope\n", "\n"])
def test_unbind_changes_nothing_unless_both_questions_are_answered(
    svc: Services, answers: str
) -> None:
    store = logged_in(svc)
    result = runner.invoke(app, ["channel", "unbind"], input=answers)
    assert result.exit_code == 1 and "nothing changed" in result.output
    assert store.bound_user() is not None


def test_unbind_with_nobody_bound_is_a_no_op(svc: Services) -> None:
    result = runner.invoke(app, ["channel", "unbind"])
    assert result.exit_code == 0 and "nobody is bound" in result.output


def test_status_shows_problems_and_ages_in_the_right_units(svc: Services) -> None:
    store = logged_in(svc)
    store.record_poll_failure("http_502", 502, None)
    store.update_window(
        22, 8, lambda w: w.record_error(svc.clock.now_utc(), kind="network", code=None, errmsg="x")
    )
    out = runner.invoke(app, ["channel", "status"]).output
    assert "polling: 1 failure(s) in a row" in out
    assert "last poll problem: http_502 code=502" in out
    assert "last send problem: network" in out
    assert "login saved: " in out


def test_status_describes_ages_from_seconds_to_hours() -> None:
    from datetime import UTC, datetime

    from twin.channel.ilink.status import _age, _hours

    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    assert _age(now, None) == "never"
    assert _age(now, now - timedelta(seconds=30)) == "30 s ago"
    assert _age(now, now - timedelta(minutes=10)) == "10 min ago"
    assert _age(now, now - timedelta(hours=5)) == "5.0 h ago"
    assert _hours(timedelta(hours=21, minutes=5)) == "21h05m"
    assert _hours(timedelta(hours=-1, minutes=-30)) == "-1h30m"


# ---------------------------------------------------------------- listen


def test_listen_prints_the_form_of_each_message_and_never_its_content(
    svc: Services, api: respx.MockRouter
) -> None:
    logged_in(svc)
    api.post(f"{API}/ilink/bot/msg/notifystart").respond(200, json={})
    api.post(f"{API}/ilink/bot/msg/notifystop").respond(200, json={})
    created = int(svc.clock.now_utc().timestamp() * 1000)
    batch = [
        message(text_item("很私密的话"), mid=1, created_ms=created),
        message({"type": 99, "x": "很私密的话"}, mid=2, created_ms=created),
        message(text_item("回复", ref_msg={"svr_id": "404"}), mid=3, created_ms=created),
    ]
    api.post(f"{API}/ilink/bot/getupdates").mock(
        side_effect=[httpx.Response(200, json=updates(batch, cursor="C1"))]
        + [httpx.Response(200, json=updates(cursor="C1"))] * 50
    )
    result = runner.invoke(app, ["channel", "listen", "--count", "3"])
    assert result.exit_code == 0, result.output
    assert "kind=text  item_type=1  text_chars=5" in result.output
    assert (
        "kind=unknown  item_type=99" in result.output and "flags=unknown_item_type" in result.output
    )
    assert "quote=unresolved" in result.output
    assert "很私密的话" not in result.output


def test_listen_needs_a_bound_user_and_the_wechat_channel(svc: Services) -> None:
    result = runner.invoke(app, ["channel", "listen", "--count", "1"])
    assert result.exit_code == 1 and "nobody is bound yet" in result.output
    console = runner.invoke(app, ["--set", "channel.kind=console", "channel", "listen"])
    assert console.exit_code == 1 and "not 'ilink'" in console.output


def test_listen_cannot_run_next_to_the_application(svc: Services) -> None:
    logged_in(svc)
    lock = InstanceLock(LOCK_RUN, locks_dir=svc.paths.locks_dir)
    assert lock.acquire()
    try:
        result = runner.invoke(app, ["channel", "listen", "--count", "1"])
    finally:
        lock.release()
    assert result.exit_code == 4 and "already running" in result.output


# ----------------------------------------------- the token is nowhere in the clear (D-127, D-743)


def test_the_token_is_in_no_output_and_no_log_of_the_commands(
    svc: Services, api: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17-53a: login, status, send-test, config show and doctor never print or log the token."""
    from twin.ops.logging import configure_logging, shutdown_logging

    api.post(QR).respond(200, json={"qrcode": "Q", "qrcode_img_content": "https://qr.example/1"})
    odd = httpx.Response(200, json={"status": "odd", "bot_token": TOKEN})  # logged by its status
    confirmed = httpx.Response(
        200,
        json={
            "status": "confirmed",
            "bot_token": TOKEN,
            "ilink_bot_id": BOT,
            "baseurl": API,
            "ilink_user_id": USER,
        },
    )
    api.get(STATUS).mock(side_effect=[odd, confirmed])
    api.post(SEND).respond(200, json={"ret": 0, "message_id": 77})

    async def no_wait(self: object, seconds: float) -> None:
        return None

    monkeypatch.setattr(SystemClock, "sleep", no_wait)  # the poll gap of the login
    store_of(svc).commit_batch(
        BatchCommit(candidate=PendingBinding(USER, svc.clock.now_utc(), CTX, matches_expected=True))
    )
    configure_logging(svc.paths.logs_dir, level="DEBUG", role="run", console=False)
    try:
        outputs = [
            runner.invoke(app, ["channel", "login", "--no-open"], input="y\n").output,
            runner.invoke(app, ["channel", "status"]).output,
            runner.invoke(app, ["channel", "send-test", "你好"]).output,
            runner.invoke(app, ["config", "show"]).output,
            runner.invoke(app, ["doctor"]).output,
        ]
    finally:
        shutdown_logging()
    assert store_of(svc).credentials() is not None  # (the login did store it)
    for output in outputs:
        assert TOKEN not in output
    logs = [path.read_bytes() for path in svc.paths.logs_dir.rglob("*") if path.is_file()]
    assert logs and b"unknown_login_status" in logs[0]  # (the log was written, and read)
    assert all(TOKEN.encode() not in raw for raw in logs)


def test_the_token_is_not_readable_in_the_database_file(svc: Services) -> None:
    """R17-53a: it is kept (AES-256-GCM) in ``channel_state``; the file holds ciphertext only."""
    from sqlalchemy import text

    store = logged_in(svc)
    assert store.credentials() is not None and store.credentials().bot_token == TOKEN  # type: ignore[union-attr]
    with svc.db.session() as session:  # fold the write-ahead log into the file
        session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
    files = [path for path in svc.paths.data_dir.glob("twin.db*") if path.is_file()]
    assert files
    for path in files:
        raw = path.read_bytes()
        assert TOKEN.encode() not in raw and BOT.encode() not in raw, path.name


def test_listen_says_whether_a_picture_is_in_the_sticker_library_and_never_shows_it(
    svc: Services, api: respx.MockRouter
) -> None:
    """R17-5: the first real-device run learns what iLink delivers for a sticker of the user.

    The line holds the structure of a picture - type, size, the start of its SHA-256 and whether
    the library knows it (by the file's hash, by its MD5, or not at all) - and nothing of the
    picture or of anyone's words.
    """
    logged_in(svc)
    scenario = sticker_scenario(svc)
    known = scenario.files[scenario.md5s[0]]
    photo = image_bytes("PNG", colour=(10, 200, 30), size=12)
    key = bytes(range(1, 17))
    api.post(f"{API}/ilink/bot/msg/notifystart").respond(200, json={})
    api.post(f"{API}/ilink/bot/msg/notifystop").respond(200, json={})
    for name, data in (("DL-S", known), ("DL-P", photo)):
        api.get(f"{CDN}/download", params={"encrypted_query_param": name}).respond(
            200, content=encrypt(key, data)
        )
    created = int(svc.clock.now_utc().timestamp() * 1000)
    batch = [
        message(image_item("DL-S", media_key=b64_raw(key)), mid=1, created_ms=created),
        message(image_item("DL-P", media_key=b64_raw(key)), mid=2, created_ms=created),
    ]
    api.post(f"{API}/ilink/bot/getupdates").mock(
        side_effect=[httpx.Response(200, json=updates(batch, cursor="C1"))]
        + [httpx.Response(200, json=updates(cursor="C1"))] * 50
    )
    result = runner.invoke(app, ["channel", "listen", "--count", "2"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("#")]
    assert len(lines) == 2
    library, stranger = lines
    assert "mime=image/png" in library and "size=64x64" in library
    assert f"sha256={hashlib.sha256(known).hexdigest()[:8]}" in library
    assert "library=sha256" in library
    assert "mime=image/png" in stranger and "size=12x12" in stranger
    assert f"sha256={hashlib.sha256(photo).hexdigest()[:8]}" in stranger
    assert "library=none" in stranger
    assert hashlib.sha256(known).hexdigest()[:9] not in result.output  # only the first 8 digits


def test_a_picture_is_described_by_its_structure_only(svc: Services) -> None:
    """R17-5: ``library=md5`` for a sticker without a stored file; no facts for non-pictures."""
    from datetime import UTC, datetime

    from twin.channel.base import InboundMessage, MediaRef, MessageKind
    from twin.channel.cli import describe_inbound
    from twin.storage.chat_models import Sticker
    from twin.storage.media import MediaKind

    data = image_bytes("PNG", colour=(1, 2, 3), size=9)
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    now = svc.clock.now_utc()
    with svc.db.transaction(bump_state=False) as session:
        session.add(
            Sticker(
                md5=md5,
                status="unavailable",
                reason="download_failed",
                attempts=1,
                her_uses=1,
                user_uses=0,
                context_uses=0,
                disabled=False,
                origin="import",
                created_at=now,
                updated_at=now,
            )
        )
    stored = svc.media.put(data, MediaKind.IMAGE)
    at = datetime(2026, 10, 9, 12, tzinfo=UTC)
    ref = MediaRef(stored.sha256, MediaKind.IMAGE, stored.size, "image/png", None)
    line = describe_inbound(InboundMessage("m", at, MessageKind.IMAGE, media_ref=ref), svc)
    assert "library=md5" in line and "size=9x9" in line and "mime=image/png" in line
    plain = describe_inbound(InboundMessage("m", at, MessageKind.IMAGE, media_ref=ref))
    assert "library=" not in plain  # no database to ask: the old line, unchanged
    junk = svc.media.put(b"not a picture at all" * 20, MediaKind.IMAGE)
    odd = MediaRef(junk.sha256, MediaKind.IMAGE, junk.size, None, None)
    assert "mime=unknown" in describe_inbound(
        InboundMessage("m", at, MessageKind.IMAGE, media_ref=odd), svc
    )
    voice = MediaRef(junk.sha256, MediaKind.VOICE, junk.size, None, None)
    line = describe_inbound(InboundMessage("m", at, MessageKind.VOICE, media_ref=voice), svc)
    assert "library=" not in line and "mime=" not in line


def test_the_edges_of_the_listen_diagnostic_and_of_the_conversation_count(
    svc: Services, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sqlite3
    from datetime import UTC, datetime

    from twin.channel.base import InboundMessage, MediaRef, MessageKind
    from twin.channel.cli import describe_inbound
    from twin.ingest.conversations import count_conversations_in_file
    from twin.storage.media import MediaKind

    ref = MediaRef("ab" * 32, MediaKind.IMAGE, 100, None, None)  # nothing of it is stored
    at = datetime(2026, 10, 9, 12, tzinfo=UTC)
    line = describe_inbound(InboundMessage("m", at, MessageKind.IMAGE, media_ref=ref), svc)
    assert "file=unreadable" in line and "library=none" in line
    assert f"sha256={'ab' * 4}" in line

    database = tmp_path / "twin.db"
    database.write_bytes(b"x")

    def refuse(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(sqlite3, "connect", refuse)
    assert count_conversations_in_file(database) is None
