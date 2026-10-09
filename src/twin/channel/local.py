"""``LocalConsoleChannel``: the terminal as a chat application (R-CH-011, R-ARCH-005).

It implements the same :class:`~twin.channel.base.Channel` interface as the WeChat channel, so
the engine, the proactive scheduler and the evaluation tools can be exercised without a phone.
What it keeps from the real channel, on purpose:

* **one user, one id** (:data:`LOCAL_USER_ID`), and every send resolves its recipient through
  the same :class:`~twin.channel.binding.RecipientGuard`: naming anyone else raises
  :class:`~twin.channel.base.RecipientNotAllowed`;
* **the same picture allow list** (R-SAFE-006): only library stickers and the probe's own test
  pictures may be sent; anything else raises :class:`~twin.channel.base.MediaNotAllowed`;
* **a simulated platform window and message count** (:class:`~twin.channel.window.SessionWindow`,
  configurable) so proactive logic can be tried against tight limits: a send past them is
  refused exactly like the WeChat channel's own gate refuses it;
* **no quotes** (the WeChat protocol has none, so the engine must not rely on them).

What is terminal-specific: lines you type are your messages; ``/img <path>`` sends a picture of
yours (stored encrypted in the media store, like a WeChat picture); the bot's text is printed
with a ``bot:`` label; a sticker is shown as ``[表情包：<label>] <file path>`` (the file is a
plain copy for the session only, deleted when the channel stops); "typing" shows as
``对方正在输入…``.  Input and output are injected, so tests drive it with scripted streams.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO

from twin.channel.base import (
    AuthState,
    BypassRequest,
    CapabilityNotSupported,
    Channel,
    ChannelCapabilities,
    InboundMessage,
    MediaNotAllowed,
    MediaRef,
    MessageKind,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    SendBypass,
    SessionState,
)
from twin.channel.binding import RecipientGuard
from twin.channel.ilink.media import sniff_image_mime
from twin.channel.ilink.outbound import IMAGE_MIMES
from twin.channel.ilink.wire import TEXT_CHUNK_LIMIT
from twin.channel.policy import (
    CompositeMediaPolicy,
    DenyAllMediaPolicy,
    OutboundMediaPolicy,
    ProbeImageManifest,
    StickerAllowList,
    ensure_media_allowed,
    sha256_hex,
)
from twin.channel.state import ChannelStateStore
from twin.channel.window import SessionWindow
from twin.clock import Clock
from twin.ops.logging import get_logger
from twin.stickers.library import StickerLibraryAllowList
from twin.storage.media import MediaKind, MediaStore

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.channel.local")

LOCAL_USER_ID = "local-console-user"
BOT_LABEL = "bot"
TYPING_TEXT = "对方正在输入…"
IMG_COMMAND = "/img"
QUIT_COMMANDS = frozenset({"/quit", "/exit"})
HELP_COMMAND = "/help"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
NOT_SUPPORTED_QUOTE = "the local console channel has no quotes, like the WeChat protocol"
HELP_LINES = (
    "type a message and press Enter to send it",
    f"{IMG_COMMAND} <path>   send a picture file from this computer",
    f"{HELP_COMMAND}          this help",
    f"{' or '.join(sorted(QUIT_COMMANDS))}   leave (Ctrl+D / Ctrl+Z works too)",
)


class TextInput(Protocol):
    """Where the user's lines come from; ``None`` means the input ended."""

    async def readline(self) -> str | None: ...


class TextOutput(Protocol):
    """Where the bot's lines go."""

    def write_line(self, text: str) -> None: ...


class StickerLabels(Protocol):
    """The label of a sticker (round 06 provides the real one); ``None`` if it has none."""

    def label(self, sha256: str) -> str | None: ...


class StreamInput:
    """Lines from a text stream, read on a daemon thread so a blocked read never stops exit."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._queue: asyncio.Queue[str | None] | None = None

    def _start(self) -> asyncio.Queue[str | None]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream = self._stream

        def pump() -> None:
            try:
                while line := stream.readline():
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            except (OSError, ValueError):
                pass  # the stream was closed under us: same as the end of the input
            finally:
                with contextlib.suppress(RuntimeError):  # the event loop may already be closed
                    loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=pump, name="console-input", daemon=True).start()
        return queue

    async def readline(self) -> str | None:
        if self._queue is None:
            self._queue = self._start()
        return await self._queue.get()


class StreamOutput:
    """Lines to a text stream, flushed at once."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write_line(self, text: str) -> None:
        self._stream.write(text + "\n")
        self._stream.flush()


class LocalConsoleChannel(Channel):
    """The terminal as a channel: same interface, same guards, simulated platform limits."""

    name = "local_console_channel"

    def __init__(
        self,
        *,
        clock: Clock,
        media: MediaStore,
        input: TextInput,
        output: TextOutput,
        shown_dir: Path,
        window_h: float = 22.0,
        quota: int = 8,
        media_policy: OutboundMediaPolicy | None = None,
        stickers: StickerAllowList | None = None,
        labels: StickerLabels | None = None,
        user_id: str = LOCAL_USER_ID,
    ) -> None:
        self._clock = clock
        self._media = media
        self._input = input
        self._output = output
        self._shown_dir = shown_dir
        self._media_policy: OutboundMediaPolicy = media_policy or DenyAllMediaPolicy()
        self._stickers = stickers
        self._labels = labels
        self._user_id = user_id
        self._window = SessionWindow(window_h=window_h, quota=quota)
        self._window_h = window_h
        self._quota = quota
        self.guard = RecipientGuard(lambda: self._user_id)
        self._inbox: asyncio.Queue[InboundMessage | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._counter = 0
        self._typing = False
        self._shown: list[Path] = []
        self._ended = False

    @classmethod
    def from_services(
        cls,
        services: Services,
        *,
        input: TextInput,
        output: TextOutput,
        window_h: float | None = None,
        quota: int | None = None,
        stickers: StickerAllowList | None = None,
        labels: StickerLabels | None = None,
    ) -> LocalConsoleChannel:
        """Build the channel; the simulated limits default to the configured safe thresholds."""
        config = services.settings.channel
        allow = stickers if stickers is not None else StickerLibraryAllowList(services.db)
        return cls(
            clock=services.clock,
            media=services.media,
            input=input,
            output=output,
            shown_dir=services.paths.tmp_dir / "console-out",
            window_h=window_h if window_h is not None else config.proactive_window_safe_h,
            quota=quota if quota is not None else config.outbound_quota_safe,
            media_policy=CompositeMediaPolicy(
                [allow, ProbeImageManifest(ChannelStateStore(services.db))]
            ),
            stickers=allow,
            labels=labels,
        )

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._reader is None:
            self._ended = False
            self._reader = asyncio.create_task(self._read_loop(), name="local-console-reader")

    async def stop(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        self._finish_input()
        for path in self._shown:
            path.unlink(missing_ok=True)
        self._shown.clear()

    def _finish_input(self) -> None:
        if not self._ended:
            self._ended = True
            self._inbox.put_nowait(None)

    # ------------------------------------------------------------- inbound

    async def _read_loop(self) -> None:
        while (line := await self._input.readline()) is not None:
            if not await self._handle_line(line):
                break
        self._finish_input()

    async def _handle_line(self, raw: str) -> bool:
        """Act on one typed line; ``False`` when the user asked to leave."""
        text = raw.rstrip("\r\n")
        command = text.strip()
        lowered = command.lower()
        if not command:
            return True
        if lowered in QUIT_COMMANDS:
            return False
        if lowered == HELP_COMMAND:
            for line in HELP_LINES:
                self._output.write_line(f"  {line}")
            return True
        if lowered == IMG_COMMAND or lowered.startswith(IMG_COMMAND + " "):
            await self._handle_image(command[len(IMG_COMMAND) :].strip())
            return True
        self._deliver(MessageKind.TEXT, text=text)
        return True

    async def _handle_image(self, argument: str) -> None:
        argument = argument.strip("\"'")
        if not argument:
            self._output.write_line(f"  usage: {IMG_COMMAND} <path of a picture>")
            return
        name = Path(argument).name
        try:
            data = await asyncio.to_thread(self._read_picture, argument)
        except OSError as exc:
            self._output.write_line(f"  cannot read {name}: {exc.strerror or 'error'}")
            return
        except ValueError as exc:
            self._output.write_line(f"  not sent: {exc}")
            return
        stored = await asyncio.to_thread(self._media.put, data, MediaKind.IMAGE)
        mime = sniff_image_mime(data)
        self._deliver(
            MessageKind.IMAGE,
            media=MediaRef(stored.sha256, MediaKind.IMAGE, stored.size, mime, name),
        )

    @staticmethod
    def _read_picture(argument: str) -> bytes:
        path = Path(argument).expanduser()
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError(f"the picture is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        data = path.read_bytes()
        if sniff_image_mime(data) is None:
            raise ValueError("the file is not a PNG, JPEG, GIF or WebP picture")
        return data

    def _deliver(
        self, kind: MessageKind, *, text: str | None = None, media: MediaRef | None = None
    ) -> None:
        self._counter += 1
        now = self._clock.now_utc()
        self._window.on_inbound(now)
        self._inbox.put_nowait(
            InboundMessage(
                id=f"local-{self._counter}", at=now, kind=kind, text=text, media_ref=media
            )
        )

    async def incoming(self) -> AsyncIterator[InboundMessage]:
        while True:
            item = await self._inbox.get()
            if item is None:
                self._inbox.put_nowait(None)  # a second iteration also ends
                return
            yield item

    # ------------------------------------------------------------ outbound

    def _gate(
        self, kind: str, text: str | None, bypass: SendBypass | None
    ) -> OutboundResult | None:
        """The simulated platform limits; a bypass (the probe) may skip the thresholds."""
        now = self._clock.now_utc()
        reason = self._window.refusal_reason(now)
        if bypass is not None:
            bypass.authorize(
                BypassRequest("text" if kind == "text" else "image", text, now, reason)
            )
            return None
        if reason is not None:
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, reason)
        return None

    def _sent(self) -> OutboundResult:
        self._window.on_outbound(1)
        self._counter += 1
        return OutboundResult.success(message_id=f"local-out-{self._counter}")

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        if quote is not None:
            raise CapabilityNotSupported(NOT_SUPPORTED_QUOTE)
        self.guard.resolve(recipient)
        if not text.strip():
            return OutboundResult.failure(OutboundKind.REJECTED, "empty_text")
        if len(text) > TEXT_CHUNK_LIMIT:
            return OutboundResult.failure(OutboundKind.REJECTED, "text_too_long")
        async with self._lock:
            refused = self._gate("text", text, bypass)
            if refused is not None:
                return refused
            self._typing = False
            first, *rest = text.split("\n")
            self._output.write_line(f"{BOT_LABEL}: {first}")
            for line in rest:
                self._output.write_line(f"{' ' * (len(BOT_LABEL) + 2)}{line}")
            return self._sent()

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        self.guard.resolve(recipient)
        raw = data if isinstance(data, bytes) else await asyncio.to_thread(Path(data).read_bytes)
        canonical = IMAGE_MIMES.get(mime.lower())
        if canonical is None:
            raise MediaNotAllowed(f"images of type {mime!r} are not sent")
        if sniff_image_mime(raw) != canonical:
            raise MediaNotAllowed("the bytes are not an image of the declared type")
        await asyncio.to_thread(ensure_media_allowed, self._media_policy, raw)
        async with self._lock:
            refused = self._gate("image", None, bypass)
            if refused is not None:
                return refused
            digest = sha256_hex(raw)
            path = await asyncio.to_thread(self._write_shown, digest, canonical, raw)
            self._typing = False
            label = self._sticker_label(digest)
            shown = f"[表情包：{label}]" if label is not None else "[图片]"
            self._output.write_line(f"{BOT_LABEL}: {shown} {path}")
            return self._sent()

    def _sticker_label(self, sha256: str) -> str | None:
        """The label to show for a library sticker, or ``None`` if this is not one."""
        if self._stickers is None or not self._stickers.allowed(sha256):
            return None
        return (self._labels.label(sha256) if self._labels else None) or "未标注"

    def _write_shown(self, sha256: str, mime: str, data: bytes) -> Path:
        """A plain copy of the picture so its path can be opened; removed when the channel stops."""
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif"}.get(mime, ".img")
        self._shown_dir.mkdir(parents=True, exist_ok=True)
        path = self._shown_dir / f"{sha256[:16]}{suffix}"
        if not path.exists():
            path.write_bytes(data)
            self._shown.append(path)
        return path

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        self.guard.resolve(recipient)
        if active and not self._typing:
            self._output.write_line(TYPING_TEXT)
        self._typing = active

    # -------------------------------------------------------- introspection

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_quote=False,
            supports_typing=True,
            gif_animated=None,
            proactive_window_h=self._window_h,
            outbound_quota=self._quota,
            max_text_chars=TEXT_CHUNK_LIMIT,
        )

    def window(self) -> SessionWindow:
        """The simulated conversation window (a snapshot)."""
        return SessionWindow(window_h=self._window_h, quota=self._quota, state=self._window.state)

    def session_state(self) -> SessionState:
        window = self._window
        now = self._clock.now_utc()
        return SessionState(
            auth=AuthState.OK,
            bound=True,
            last_inbound_at=window.last_inbound_at,
            outbound_since_inbound=window.outbound_since_inbound,
            expired=window.expired,
            remaining_quota=window.remaining_quota(),
            window_remaining=window.window_remaining(now),
            has_context_token=True,
            extra={"simulated": True, "typing": self._typing},
        )
