"""``LocalConsoleChannel`` and ``twin chat --local`` (R-CH-011, R-ARCH-005)."""

from __future__ import annotations

import ast
import asyncio
import io
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from PIL import Image
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.console import FixedLabels, RecordingOutput, ScriptedInput
from tests.support.ilink import (
    API,
    BOT,
    CTX,
    TOKEN,
    USER,
    AllowingBypass,
    SetAllowList,
    gif_bytes,
    image_bytes,
    message,
    now_ms,
    request_json,
    text_item,
    updates,
)
from twin.channel.base import (
    AuthState,
    BypassRefused,
    CapabilityNotSupported,
    Channel,
    InboundMessage,
    MediaNotAllowed,
    MessageKind,
    OutboundKind,
    QuoteTarget,
    RecipientNotAllowed,
)
from twin.channel.chat import run_local_chat
from twin.channel.echo import EchoHandler
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.local import (
    HELP_LINES,
    LOCAL_USER_ID,
    TYPING_TEXT,
    LocalConsoleChannel,
    StreamInput,
    StreamOutput,
)
from twin.channel.policy import CompositeMediaPolicy, ProbeImageManifest, sha256_hex
from twin.channel.state import ChannelStateStore
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.services import Services, build_services
from twin.storage.db import Database
from twin.storage.media import MediaKind, MediaStore

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
runner = CliRunner()


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


STICKER = gif_bytes()
PROBE = image_bytes("PNG", (10, 20, 30), size=12)
PHOTO = image_bytes("JPEG", (200, 100, 50), size=24)


class Console:
    """A started channel with its scripted input and recorded output."""

    def __init__(
        self,
        channel: LocalConsoleChannel,
        inp: ScriptedInput,
        out: RecordingOutput,
        media: MediaStore,
        shown: Path,
    ) -> None:
        self.channel = channel
        self.input = inp
        self.out = out
        self.media = media
        self.shown = shown
        self._iterator = channel.incoming()

    async def receive(self) -> InboundMessage:
        return await asyncio.wait_for(self._iterator.__anext__(), timeout=5)

    async def ended(self) -> bool:
        try:
            await asyncio.wait_for(self._iterator.__anext__(), timeout=5)
        except StopAsyncIteration:
            return True
        return False


def build(
    db: Database,
    clock: ManualClock,
    tmp_path: Path,
    *,
    quota: int = 3,
    window_h: float = 22,
    labels: FixedLabels | None = None,
) -> Console:
    media = MediaStore(tmp_path / "media", tmp_path / "tmp")
    inp, out = ScriptedInput(), RecordingOutput()
    shown = tmp_path / "shown"
    policy = CompositeMediaPolicy(
        [SetAllowList(sha256_hex(STICKER)), ProbeImageManifest(ChannelStateStore(db))]
    )
    ProbeImageManifest(ChannelStateStore(db)).register_bytes(PROBE)
    channel = LocalConsoleChannel(
        clock=clock,
        media=media,
        input=inp,
        output=out,
        shown_dir=shown,
        window_h=window_h,
        quota=quota,
        media_policy=policy,
        stickers=SetAllowList(sha256_hex(STICKER)),
        labels=labels,
    )
    return Console(channel, inp, out, media, shown)


@pytest.fixture
async def console(db: Database, clock: ManualClock, tmp_path: Path) -> AsyncIterator[Console]:
    built = build(db, clock, tmp_path, labels=FixedLabels(**{sha256_hex(STICKER): "开心"}))
    await built.channel.start()
    yield built
    await built.channel.stop()


# ---------------------------------------------------------------------- inbound


async def test_a_typed_line_becomes_an_inbound_message_that_opens_the_window(
    console: Console, clock: ManualClock
) -> None:
    console.input.feed("你好  呀")
    message_ = await console.receive()
    assert message_.kind is MessageKind.TEXT and message_.text == "你好  呀"
    assert message_.id == "local-1" and message_.at == clock.now_utc()
    state = console.channel.session_state()
    assert state.last_inbound_at == clock.now_utc() and state.outbound_since_inbound == 0


async def test_blank_lines_are_ignored_and_line_endings_are_removed(console: Console) -> None:
    for line in ("", "   ", "\r", "hello\r"):
        console.input.feed(line)
    console.input.feed("second")
    assert (await console.receive()).text == "hello"
    assert (await console.receive()).text == "second"


async def test_slash_lines_that_are_not_channel_commands_are_ordinary_messages(
    console: Console,
) -> None:
    for line in ("/暂停", "/img2 x", "/images"):
        console.input.feed(line)
    assert [(await console.receive()).text for _ in range(3)] == ["/暂停", "/img2 x", "/images"]


def write_picture(path: Path, colour: tuple[int, int, int] = (9, 99, 199)) -> bytes:
    Image.new("RGB", (16, 16), colour).save(path, "PNG")
    return path.read_bytes()


async def test_img_sends_a_picture_of_yours_stored_encrypted(
    console: Console, tmp_path: Path
) -> None:
    path = tmp_path / "我的图片.png"
    data = write_picture(path)
    console.input.feed(f"/img {path}")
    received = await console.receive()
    assert received.kind is MessageKind.IMAGE and received.text is None
    ref = received.media_ref
    assert ref is not None and ref.kind is MediaKind.IMAGE and ref.mime == "image/png"
    assert (
        ref.sha256 == sha256_hex(data) and ref.size == len(data) and ref.file_name == "我的图片.png"
    )
    assert console.media.read_bytes(ref.sha256) == data
    assert b"PNG" not in console.media.path_for(ref.sha256).read_bytes()[:64]  # not plaintext


async def test_img_accepts_a_quoted_path_as_pasted_from_a_file_manager(
    console: Console, tmp_path: Path
) -> None:
    path = tmp_path / "a b.png"
    write_picture(path)
    console.input.feed(f'/img "{path}"')
    assert (await console.receive()).kind is MessageKind.IMAGE


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("missing", "cannot read nothing-here.png"),
        ("text", "not sent: the file is not a PNG, JPEG, GIF or WebP picture"),
        ("empty", "usage: /img <path of a picture>"),
    ],
)
async def test_img_problems_are_explained_and_send_nothing(
    console: Console, tmp_path: Path, setup: str, expected: str
) -> None:
    if setup == "missing":
        console.input.feed(f"/img {tmp_path / 'nothing-here.png'}")
    elif setup == "text":
        notes = tmp_path / "notes.txt"
        notes.write_text("not a picture", encoding="utf-8")
        console.input.feed(f"/img {notes}")
    else:
        console.input.feed("/img")
    console.input.feed("after")
    assert (await console.receive()).text == "after"  # the failed command produced no message
    assert any(expected in line for line in console.out.lines)


async def test_img_refuses_a_picture_over_the_size_limit(
    console: Console, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("twin.channel.local.MAX_IMAGE_BYTES", 10)
    path = tmp_path / "big.png"
    write_picture(path)
    console.input.feed(f"/img {path}")
    console.input.feed("after")
    assert (await console.receive()).text == "after"
    assert any("larger than" in line for line in console.out.lines)


async def test_help_lists_the_commands(console: Console) -> None:
    console.input.feed("/help")
    console.input.feed("next")
    await console.receive()
    shown = console.out.text
    assert all(line in shown for line in HELP_LINES)
    assert "/img <path>" in shown and "/quit" in shown


@pytest.mark.parametrize("ending", ["/quit", "/EXIT", "  /quit  "])
async def test_quit_ends_the_input(console: Console, ending: str) -> None:
    console.input.feed("one")
    console.input.feed(ending)
    console.input.feed("never seen")
    assert (await console.receive()).text == "one"
    assert await console.ended()
    assert [m async for m in console.channel.incoming()] == []  # a new iteration ends at once


async def test_the_end_of_the_input_ends_the_conversation(console: Console) -> None:
    console.input.feed("last")
    console.input.close()
    assert (await console.receive()).text == "last"
    assert await console.ended()


# --------------------------------------------------------------------- outbound


async def test_the_bot_text_is_printed_with_its_label_and_counts_against_the_quota(
    console: Console,
) -> None:
    console.input.feed("hi")
    await console.receive()
    result = await console.channel.send_text("你好呀\n第二行")
    assert result.ok and result.kind is OutboundKind.OK and result.message_id
    assert console.out.lines == ["bot: 你好呀", "     第二行"]
    assert console.channel.session_state().remaining_quota == 2


async def test_an_empty_or_oversized_text_is_rejected_without_output(console: Console) -> None:
    empty = await console.channel.send_text("   ")
    assert empty.reason == "empty_text" and empty.kind is OutboundKind.REJECTED
    huge = await console.channel.send_text("x" * 4001)
    assert huge.reason == "text_too_long"
    assert console.out.lines == []


async def test_typing_is_shown_once_and_cleared_by_the_next_message(console: Console) -> None:
    await console.channel.send_typing(True)
    await console.channel.send_typing(True)
    assert console.out.lines == [TYPING_TEXT]
    assert console.channel.session_state().extra["typing"] is True
    console.input.feed("hi")
    await console.receive()
    await console.channel.send_text("done")
    assert console.channel.session_state().extra["typing"] is False
    await console.channel.send_typing(False)  # clearing prints nothing
    assert console.out.lines == [TYPING_TEXT, "bot: done"]
    await console.channel.send_typing(True)
    assert console.out.lines[-1] == TYPING_TEXT  # and it can be shown again


async def test_a_sticker_is_shown_as_its_label_and_the_path_of_a_copy(console: Console) -> None:
    console.input.feed("hi")
    await console.receive()
    result = await console.channel.send_image(STICKER, "image/gif")
    assert result.ok
    [line] = console.out.lines
    assert line.startswith("bot: [表情包：开心] ")
    shown = Path(line.removeprefix("bot: [表情包：开心] "))
    assert file_bytes(shown) == STICKER and shown.suffix == ".gif" and shown.parent == console.shown


async def test_a_sticker_without_a_label_says_so(
    db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    built = build(db, clock, tmp_path)
    built.input.feed("hi")
    await built.channel.start()
    try:
        await built.receive()
        await built.channel.send_image(STICKER, "image/gif")
    finally:
        await built.channel.stop()
    assert built.out.lines[0].startswith("bot: [表情包：未标注] ")


async def test_a_probe_picture_is_shown_as_a_picture_not_a_sticker(console: Console) -> None:
    console.input.feed("hi")
    await console.receive()
    assert (await console.channel.send_image(PROBE, "image/png")).ok
    assert console.out.lines[0].startswith("bot: [图片] ")


async def test_a_picture_can_be_given_as_a_file(console: Console, tmp_path: Path) -> None:
    console.input.feed("hi")
    await console.receive()
    source = tmp_path / "sticker.gif"
    source.write_bytes(STICKER)
    assert (await console.channel.send_image(source, "image/gif")).ok
    assert "[表情包：开心]" in console.out.lines[0]


async def test_pictures_that_are_not_on_the_allow_list_never_reach_the_terminal(
    console: Console,
) -> None:
    console.input.feed("hi")
    await console.receive()
    with pytest.raises(MediaNotAllowed, match="neither a sticker"):
        await console.channel.send_image(PHOTO, "image/jpeg")
    with pytest.raises(MediaNotAllowed, match="declared type"):
        await console.channel.send_image(STICKER, "image/png")
    with pytest.raises(MediaNotAllowed, match="not sent"):
        await console.channel.send_image(STICKER, "application/pdf")
    assert console.out.lines == [] and not console.shown.exists()
    assert console.channel.session_state().outbound_since_inbound == 0


async def test_the_copies_shown_to_the_user_are_removed_when_the_channel_stops(
    db: Database, clock: ManualClock, tmp_path: Path
) -> None:
    built = build(db, clock, tmp_path)
    built.input.feed("hi")
    await built.channel.start()
    await built.receive()
    await built.channel.send_image(STICKER, "image/gif")
    assert len(list(built.shown.iterdir())) == 1
    await built.channel.stop()
    assert list(built.shown.iterdir()) == []


async def test_the_same_picture_is_written_once(console: Console) -> None:
    console.input.feed("hi")
    await console.receive()
    await console.channel.send_image(STICKER, "image/gif")
    await console.channel.send_image(STICKER, "image/gif")
    assert len(list(console.shown.iterdir())) == 1
    assert console.out.lines[0] == console.out.lines[1]


# ----------------------------------------------------------------- the guards


async def test_only_the_local_user_can_be_named_as_recipient(console: Console) -> None:
    console.input.feed("hi")
    await console.receive()
    for call in (
        console.channel.send_text("x", recipient="someone-else"),
        console.channel.send_image(STICKER, "image/gif", recipient="someone-else"),
        console.channel.send_typing(True, recipient="someone-else"),
    ):
        with pytest.raises(RecipientNotAllowed):
            await call
    assert console.out.lines == []
    assert (await console.channel.send_text("ok", recipient=LOCAL_USER_ID)).ok


async def test_quotes_are_not_supported_like_the_wechat_protocol(console: Console) -> None:
    with pytest.raises(CapabilityNotSupported, match="no quotes"):
        await console.channel.send_text("x", QuoteTarget("m1", "old"))
    assert console.channel.capabilities().supports_quote is False


# ------------------------------------------------- the simulated window and count


async def test_the_simulated_count_refuses_the_next_bubble_and_a_message_restores_it(
    console: Console, clock: ManualClock
) -> None:
    console.input.feed("hi")
    await console.receive()
    for number in range(3):
        assert (await console.channel.send_text(f"b{number}")).ok
    refused = await console.channel.send_text("b3")
    assert refused.kind is OutboundKind.WINDOW_REJECTED and refused.reason == "quota_exhausted"
    assert not refused.session_expired and "b3" not in console.out.text
    window = console.channel.window()
    assert window.remaining_quota() == 0 and not window.can_send_proactive(clock.now_utc())
    console.input.feed("again")
    await console.receive()
    assert console.channel.session_state().remaining_quota == 3
    assert (await console.channel.send_text("b3")).ok


async def test_a_picture_is_refused_like_a_text_when_the_count_is_used_up(
    console: Console,
) -> None:
    console.input.feed("hi")
    await console.receive()
    for number in range(3):
        await console.channel.send_text(f"b{number}")
    refused = await console.channel.send_image(STICKER, "image/gif")
    assert refused.kind is OutboundKind.WINDOW_REJECTED and refused.reason == "quota_exhausted"
    assert not console.shown.exists()  # no copy was written for a picture that was not shown


async def test_the_simulated_window_closes_after_its_hours(
    console: Console, clock: ManualClock
) -> None:
    console.input.feed("hi")
    await console.receive()
    clock.tick(21 * 3600 + 59 * 60)
    assert (await console.channel.send_text("still in")).ok
    clock.tick(2 * 60)
    refused = await console.channel.send_text("too late")
    assert refused.reason == "window_elapsed" and refused.kind is OutboundKind.WINDOW_REJECTED
    remaining = console.channel.session_state().window_remaining
    assert remaining is not None and remaining.total_seconds() < 0


async def test_nothing_proactive_can_be_sent_before_the_user_has_written(
    console: Console,
) -> None:
    refused = await console.channel.send_text("hello?")
    assert refused.reason == "no_inbound_yet"
    state = console.channel.session_state()
    assert state.last_inbound_at is None and state.window_remaining is None


async def test_a_bypass_skips_the_thresholds_and_is_asked_about_each_send(
    console: Console,
) -> None:
    console.input.feed("hi")
    await console.receive()
    bypass = AllowingBypass()
    for number in range(5):
        assert (await console.channel.send_text(f"[测试]{number}", bypass=bypass)).ok
    assert [r.gate_reason for r in bypass.requests] == [
        None,
        None,
        None,
        "quota_exhausted",
        "quota_exhausted",
    ]
    assert (await console.channel.send_image(PROBE, "image/png", bypass=bypass)).ok
    assert bypass.requests[-1].kind == "image" and bypass.requests[-1].text is None
    with pytest.raises(BypassRefused):
        await console.channel.send_text("[测试]x", bypass=AllowingBypass(refuse=True))


async def test_capabilities_and_state_describe_the_simulation(console: Console) -> None:
    capabilities = console.channel.capabilities()
    assert capabilities.supports_typing is True and capabilities.proactive_window_h == 22
    assert capabilities.outbound_quota == 3 and capabilities.max_text_chars == 4000
    assert capabilities.gif_animated is None
    state = console.channel.session_state()
    assert state.auth is AuthState.OK and state.bound and state.has_context_token
    assert state.extra["simulated"] is True and state.remaining_quota == 3


# ------------------------------------------------------------ the echo diagnostic


async def test_the_chat_component_reports_the_health_of_its_task(
    services: Services, console: Console
) -> None:
    from twin.app import HealthStatus
    from twin.channel.chat import LocalChatComponent

    component = LocalChatComponent(
        console.channel,
        EchoHandler(console.out),
        services.clock,
        services.alerts,
        on_finished=lambda: None,
    )
    assert component.health().status is HealthStatus.OK


async def test_the_echo_names_the_kind_of_message_it_cannot_repeat(console: Console) -> None:
    console.input.feed("hi")
    await console.receive()
    picture = InboundMessage("m2", datetime(2026, 1, 1, tzinfo=UTC), MessageKind.IMAGE)
    await EchoHandler(console.out)(picture, console.channel)
    assert console.out.lines == ["bot: [测试]回显:收到一条image消息"]


async def test_the_echo_says_when_the_channel_refused_it(console: Console) -> None:
    handler = EchoHandler(console.out)
    text = InboundMessage("m3", datetime(2026, 1, 1, tzinfo=UTC), MessageKind.TEXT, text="x")
    await handler(text, console.channel)  # the user has not written: no window yet
    assert console.out.lines == ["  (the echo was not sent - window_rejected: no_inbound_yet)"]


# ---------------------------------------------------------- real streams and setup


async def test_a_stream_that_breaks_ends_the_input_like_the_end_of_the_file() -> None:
    class Broken(io.StringIO):
        def readline(self, size: int | None = -1) -> str:
            raise ValueError("I/O operation on closed file")

    assert await StreamInput(Broken()).readline() is None


async def test_a_text_stream_is_read_line_by_line_until_its_end() -> None:
    reader = StreamInput(io.StringIO("a\nb 你好\n"))
    assert [await reader.readline(), await reader.readline(), await reader.readline()] == [
        "a\n",
        "b 你好\n",
        None,
    ]


def test_output_lines_are_written_and_flushed() -> None:
    stream = io.StringIO()
    StreamOutput(stream).write_line("bot: hi")
    assert stream.getvalue() == "bot: hi\n"


async def test_starting_twice_does_not_read_the_input_twice(console: Console) -> None:
    await console.channel.start()
    console.input.feed("once")
    assert (await console.receive()).text == "once"
    console.input.feed("twice")
    assert (await console.receive()).text == "twice"


async def test_the_channel_built_from_services_uses_the_configured_limits_and_allow_list(
    services: Services,
) -> None:
    inp, out = ScriptedInput(), RecordingOutput()
    channel = LocalConsoleChannel.from_services(services, input=inp, output=out)
    capabilities = channel.capabilities()
    assert capabilities.proactive_window_h == services.settings.channel.proactive_window_safe_h
    assert capabilities.outbound_quota == services.settings.channel.outbound_quota_safe
    tight = LocalConsoleChannel.from_services(
        services, input=inp, output=out, window_h=1.5, quota=2
    )
    assert (tight.capabilities().proactive_window_h, tight.capabilities().outbound_quota) == (
        1.5,
        2,
    )
    await channel.start()
    try:
        inp.feed("hi")
        async for _message in channel.incoming():
            break
        with pytest.raises(MediaNotAllowed):  # the sticker library is empty: nothing may go
            await channel.send_image(STICKER, "image/gif")
        ProbeImageManifest(ChannelStateStore(services.db)).register_bytes(PROBE)
        assert (await channel.send_image(PROBE, "image/png")).ok
    finally:
        await channel.stop()


# ----------------------------------------------------------------- twin chat


async def test_the_chat_application_answers_through_the_engine_at_her_pace(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from tests.support.clock import InstantClock
    from tests.support.engine_harness import ScriptedWriter, make_draft
    from twin.llm.runtime import DEEPSEEK_SECRET

    services.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    extracted: list[list[str]] = []
    monkeypatch.setattr(
        "twin.engine.component.queue_bot_extraction",
        lambda _services, turns: extracted.append([t.text for t in turns]),
    )
    instant = replace(services, clock=InstantClock())  # her waiting costs no real time
    writer = ScriptedWriter(make_draft("你好呀", "在做什么"))
    inp, out = ScriptedInput("你好", close=True), RecordingOutput()
    handled = await asyncio.wait_for(
        run_local_chat(instant, input=inp, output=out, signals=False, pipeline=writer, drain=True),
        timeout=60,
    )
    assert handled == 1
    assert "type /help for the commands" in out.lines[0]
    assert out.lines.count(TYPING_TEXT) >= 1  # she shows that she is typing
    assert [line for line in out.lines if line.startswith("bot:")][:1] == ["bot: 你好呀"]
    assert "bot: 在做什么" in out.lines
    assert writer.contexts[0].user_text == "你好"
    handed_over = [text for turns in extracted for text in turns]
    conversation = ["你好", "你好呀", "在做什么"]
    assert handed_over and handed_over == conversation[: len(handed_over)]  # once each, in order


async def test_the_chat_application_needs_the_deepseek_key_and_says_how_to_set_it(
    services: Services,
) -> None:
    from twin.config.secrets import SecretStoreError

    with pytest.raises(SecretStoreError, match="twin secrets set deepseek_api_key"):
        await asyncio.wait_for(
            run_local_chat(
                services, input=ScriptedInput(close=True), output=RecordingOutput(), signals=False
            ),
            timeout=20,
        )


async def test_a_handler_receives_the_messages_and_answers_through_the_channel(
    services: Services,
) -> None:
    inp, out = ScriptedInput("one", "two", close=True), RecordingOutput()
    handled = await asyncio.wait_for(
        run_local_chat(services, input=inp, output=out, handler=EchoHandler(out), signals=False),
        timeout=20,
    )
    assert handled == 2
    assert "bot: [测试]回显:one" in out.lines and "bot: [测试]回显:two" in out.lines


async def test_one_failing_message_does_not_end_the_conversation(services: Services) -> None:
    seen: list[str] = []

    async def handler(message: InboundMessage, channel: Channel) -> None:
        seen.append(message.text or "")
        if message.text == "boom":
            raise RuntimeError("the handler broke")
        await channel.send_text(f"ok {message.text}")

    inp, out = ScriptedInput("boom", "fine", close=True), RecordingOutput()
    handled = await asyncio.wait_for(
        run_local_chat(services, input=inp, output=out, handler=handler, signals=False), timeout=20
    )
    assert seen == ["boom", "fine"] and handled == 2
    assert out.lines.count("bot: ok fine") == 1


async def test_the_chat_can_stop_after_a_number_of_messages(services: Services) -> None:
    inp, out = ScriptedInput("a", "b", "c"), RecordingOutput()
    handled = await asyncio.wait_for(
        run_local_chat(
            services, input=inp, output=out, handler=EchoHandler(out), limit=2, signals=False
        ),
        timeout=20,
    )
    assert handled == 2 and "bot: [测试]回显:c" not in out.lines


async def test_the_simulated_limits_apply_to_what_a_handler_sends(services: Services) -> None:
    async def chatty(message: InboundMessage, channel: Channel) -> None:
        first = await channel.send_text("first")
        second = await channel.send_text("second")
        assert first.ok and not second.ok and second.reason == "quota_exhausted"

    inp, out = ScriptedInput("hi", close=True), RecordingOutput()
    await asyncio.wait_for(
        run_local_chat(
            services, input=inp, output=out, handler=chatty, quota=1, window_h=1, signals=False
        ),
        timeout=20,
    )
    assert "bot: first" in out.lines and "bot: second" not in out.lines


# ---------------------------------------------------------------- the commands


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
        for notice in ("notifystart", "notifystop"):
            router.post(f"{API}/ilink/bot/msg/{notice}").respond(200, json={"ret": 0})
        yield router


def test_chat_needs_the_local_flag(svc: Services) -> None:
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 2 and "choose --local" in result.output


def test_chat_local_runs_the_application_with_the_terminal_channel(svc: Services) -> None:
    from twin.llm.runtime import DEEPSEEK_SECRET

    svc.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    result = runner.invoke(app, ["chat", "--local"], input="/quit\n")
    assert result.exit_code == 0, result.output
    assert "type /help for the commands" in result.output
    assert "not connected" not in result.output  # the engine is part of the application now
    assert svc.runtime.snapshot()["engine.paused_until"] is None  # settings were seeded


def test_chat_local_without_the_deepseek_key_says_how_to_set_it(svc: Services) -> None:
    result = runner.invoke(app, ["chat", "--local"], input="/quit\n")
    assert result.exit_code == 6, result.output  # the secrets exit code
    assert "twin secrets set deepseek_api_key" in result.output


def test_chat_local_takes_the_simulated_limits_from_options(svc: Services) -> None:
    from twin.llm.runtime import DEEPSEEK_SECRET

    svc.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    result = runner.invoke(
        app, ["chat", "--local", "--window-h", "2", "--quota", "1"], input="/quit\n"
    )
    assert result.exit_code == 0, result.output


def test_chat_refuses_to_start_while_the_application_runs(svc: Services) -> None:
    lock = InstanceLock(LOCK_RUN, locks_dir=svc.paths.locks_dir)
    assert lock.acquire()
    try:
        result = runner.invoke(app, ["chat", "--local"], input="/quit\n")
    finally:
        lock.release()
    assert result.exit_code == 4 and "'run' instance" in result.output


def test_echo_test_local_answers_with_the_prefix_and_is_a_diagnostic(svc: Services) -> None:
    result = runner.invoke(app, ["channel", "echo-test", "--local", "--count", "1"], input="ping\n")
    assert result.exit_code == 0, result.output
    assert "bot: [测试]回显:ping" in result.output


def test_echo_test_needs_a_bound_user_over_wechat(svc: Services) -> None:
    result = runner.invoke(app, ["channel", "echo-test"])
    assert result.exit_code == 1 and "nobody is bound" in result.output


def test_echo_test_refuses_the_console_kind_without_local(svc: Services) -> None:
    result = runner.invoke(app, ["--set", "channel.kind=console", "channel", "echo-test"])
    assert result.exit_code == 1 and "use --local" in result.output


def test_echo_test_over_wechat_answers_only_the_bound_user_inside_the_limits(
    svc: Services, api: respx.MockRouter
) -> None:
    store = IlinkStore(ChannelStateStore(svc.db), svc.clock)
    store.save_credentials(Credentials(TOKEN, BOT, USER, API, svc.clock.now_utc().isoformat()))
    store.bind(USER, context_token=CTX)
    given: list[int] = []

    def answer(_request: httpx.Request) -> httpx.Response:
        if not given:
            given.append(1)
            batch = [message(text_item("ping"), created_ms=now_ms(svc.clock))]
            return httpx.Response(200, json=updates(batch, cursor="C-1"))
        return httpx.Response(200, json=updates(cursor="C-2"))

    api.post(f"{API}/ilink/bot/getupdates").mock(side_effect=answer)
    send = api.post(f"{API}/ilink/bot/sendmessage").respond(200, json={})
    result = runner.invoke(app, ["channel", "echo-test", "--count", "1"])
    assert result.exit_code == 0, result.output
    assert "#1  " in result.output and "kind=text" in result.output and "ping" not in result.output
    body = request_json(send.calls.last.request)["msg"]
    assert body["to_user_id"] == USER
    assert body["item_list"][0]["text_item"]["text"] == "[测试]回显:ping"


# --------------------------------------------------- the echo is not a reply path


def importers_of(module: str) -> set[str]:
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if module in names:
                found.add(str(path.relative_to(SRC)).replace("\\", "/"))
    return found


def test_the_echo_diagnostic_is_used_by_the_diagnostic_command_and_nothing_else() -> None:
    assert importers_of("twin.channel.echo") == {"channel/cli.py"}


def test_the_terminal_channel_is_not_the_product_channel() -> None:
    # the product starts the WeChat channel (`twin run`); the terminal channel is for the chat and
    # echo commands, and for `twin run` only where `channel.kind=console` asks for it
    assert importers_of("twin.channel.local") <= {
        "channel/chat.py",
        "channel/cli.py",
        "channel/echo.py",
        "cli.py",
    }


def test_nothing_in_the_repository_answers_with_an_echo() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "echo.py" and "回显" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
