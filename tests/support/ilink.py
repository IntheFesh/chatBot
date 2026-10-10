"""Synthetic iLink traffic and test doubles (built from docs/ILINK_PROTOCOL.md section 10.7).

Every id, token and URL is a made-up value.  Encryption here is an independent
implementation of the protocol's AES-128-ECB/PKCS7 (not the production functions), so a bug
in the production code cannot hide behind the same bug in the test.
"""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from PIL import Image

from tests.support.clock import ManualClock
from tests.support.waiting import wait_until
from twin.channel.base import BypassRefused, BypassRequest
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore
from twin.clock import Clock
from twin.storage.db import Database
from twin.storage.media import MediaStore

API = "https://ilinkai.weixin.qq.com"
CDN = "https://novac2c.cdn.weixin.qq.com/c2c"
USER = "u1synthetic0000000000000001@im.wechat"
OTHER = "u2synthetic0000000000000002@im.wechat"
BOT = "b1synthetic0000000000000001@im.bot"
TOKEN = "BOT-TOKEN-SYNTHETIC"
CTX = "CTX-SYNTHETIC-1"


# ---------------------------------------------------------------- crypto


def encrypt(key: bytes, data: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()  # noqa: S305
    return encryptor.update(padded) + encryptor.finalize()


def b64_raw(key: bytes) -> str:
    return base64.b64encode(key).decode("ascii")


def b64_hex(key: bytes) -> str:
    return base64.b64encode(key.hex().encode("ascii")).decode("ascii")


# ---------------------------------------------------------------- images


def image_bytes(
    fmt: str = "PNG", colour: tuple[int, int, int] = (200, 30, 30), size: int = 8
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, fmt)
    return buffer.getvalue()


def gif_bytes() -> bytes:
    frames = [Image.new("P", (6, 6), shade) for shade in (0, 120, 240)]
    buffer = io.BytesIO()
    frames[0].save(buffer, "GIF", save_all=True, append_images=frames[1:], duration=80, loop=0)
    return buffer.getvalue()


# --------------------------------------------------------- wire builders


def text_item(text: str, **extra: Any) -> dict[str, Any]:
    return {"type": 1, "text_item": {"text": text}, **extra}


def image_item(
    download_param: str,
    *,
    hex_key: str | None = None,
    media_key: str | None = None,
    full_url: str | None = None,
) -> dict[str, Any]:
    media: dict[str, Any] = {"encrypt_query_param": download_param, "encrypt_type": 1}
    if media_key is not None:
        media["aes_key"] = media_key
    if full_url is not None:
        media["full_url"] = full_url
    item: dict[str, Any] = {"type": 2, "image_item": {"media": media, "mid_size": 100}}
    if hex_key is not None:
        item["image_item"]["aeskey"] = hex_key
    return item


def message(
    *items: dict[str, Any],
    mid: int | str | None = 9_223_372_036_854_775_001,
    sender: str = USER,
    context_token: str | None = CTX,
    created_ms: int | None = None,
    message_type: int = 1,
    seq: int = 1,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "seq": seq,
        "from_user_id": sender,
        "to_user_id": BOT,
        "message_type": message_type,
        "message_state": 2,
        "item_list": list(items),
    }
    if mid is not None:
        body["message_id"] = mid
    if context_token is not None:
        body["context_token"] = context_token
    if created_ms is not None:
        body["create_time_ms"] = created_ms
    return body


def updates(
    msgs: Sequence[dict[str, Any]] = (), cursor: str | None = "CURSOR-1", **extra: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {"ret": 0, "msgs": list(msgs), **extra}
    if cursor is not None:
        body["get_updates_buf"] = cursor
    return body


def now_ms(clock: Clock, *, minus_s: float = 0.0) -> int:
    return int((clock.now_utc().timestamp() - minus_s) * 1000)


def request_json(request: httpx.Request) -> dict[str, Any]:
    import json

    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body


# ----------------------------------------------------------- test doubles


@dataclass
class RecordingBanner:
    shown: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def show(self, title: str, lines: Sequence[str]) -> None:
        self.shown.append((title, tuple(lines)))


class ScriptedPrompter:
    """Answers questions from a script; fails the test on an unexpected question."""

    def __init__(self, *answers: bool | str) -> None:
        self._answers = list(answers)
        self.said: list[str] = []
        self.asked: list[str] = []

    def confirm(self, message: str, *, default: bool = False) -> bool:
        self.asked.append(message)
        assert self._answers, f"unexpected question: {message}"
        answer = self._answers.pop(0)
        assert isinstance(answer, bool), f"expected a yes/no answer for: {message}"
        return answer

    def ask(self, message: str) -> str:
        self.asked.append(message)
        assert self._answers, f"unexpected question: {message}"
        answer = self._answers.pop(0)
        assert isinstance(answer, str), f"expected text for: {message}"
        return answer

    def say(self, message: str) -> None:
        self.said.append(message)

    @property
    def transcript(self) -> str:
        return "\n".join(self.said)

    def remaining(self) -> int:
        return len(self._answers)


class AllowingBypass:
    """A ``SendBypass`` that records each request and allows everything (probe stand-in)."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.requests: list[BypassRequest] = []
        self._refuse = refuse

    def authorize(self, request: BypassRequest) -> None:
        self.requests.append(request)
        if self._refuse:
            raise BypassRefused("not allowed in this test")


class SetAllowList:
    """A ``StickerAllowList`` backed by a set of hashes."""

    def __init__(self, *hashes: str) -> None:
        self.hashes = set(hashes)

    def allowed(self, sha256: str) -> bool:
        return sha256 in self.hashes


# ------------------------------------------------------------- the harness


@dataclass
class Harness:
    """A channel on a migrated database, with handles to its parts."""

    channel: IlinkChannel
    store: IlinkStore
    state: ChannelStateStore
    db: Database
    clock: Clock
    media: MediaStore
    alerts: Any
    banner: RecordingBanner
    window_h: float
    quota: int

    def login(self, *, expected_user: str | None = USER, base_url: str = API) -> None:
        self.store.save_credentials(
            Credentials(
                bot_token=TOKEN,
                ilink_bot_id=BOT,
                ilink_user_id=expected_user,
                api_base_url=base_url,
                saved_at=self.clock.now_utc().isoformat(),
            )
        )

    def bind(self, user: str = USER, *, with_context: bool = True) -> None:
        self.store.bind(user, context_token=CTX if with_context else None)


def make_harness(
    db: Database,
    clock: Clock,
    tmp_path: Path,
    alerts: Any,
    *,
    window_h: float = 22,
    quota: int = 8,
    policy: Any = None,
    poll: bool = True,
    jitter: Callable[[], float] = lambda: 1.0,
    client: httpx.AsyncClient | None = None,
) -> Harness:
    media = MediaStore(tmp_path / "media", tmp_path / "tmp")
    banner = RecordingBanner()
    channel = IlinkChannel(
        db=db,
        clock=clock,
        media=media,
        alerts=alerts,
        window_h=window_h,
        quota=quota,
        media_policy=policy,
        banner=banner,
        jitter=jitter,
        poll=poll,
        http_client=client,
    )
    return Harness(
        channel=channel,
        store=channel.store,
        state=channel.state,
        db=db,
        clock=clock,
        media=media,
        alerts=alerts,
        banner=banner,
        window_h=window_h,
        quota=quota,
    )


def aware(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    from datetime import UTC

    return datetime(year, month, day, hour, minute, tzinfo=UTC)


async def drive[T](
    work: Awaitable[T], clock: ManualClock, *, step_s: float = 5.0, limit_s: float = 5.0
) -> T:
    """Run ``work`` to completion, advancing the manual clock whenever it sleeps."""
    task = asyncio.ensure_future(work)
    try:
        while not task.done():
            await wait_until(lambda: task.done() or clock.pending_sleepers >= 1, limit_s=limit_s)
            if not task.done():
                await clock.advance(step_s)
        return await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def run_until(
    start: Callable[[asyncio.Event], Awaitable[None]],
    clock: ManualClock,
    done: Callable[[], bool],
    *,
    step_s: float = 1.0,
    limit_s: float = 5.0,
) -> None:
    """Run a stoppable loop, advancing the manual clock while it sleeps, until ``done()``."""
    stop = asyncio.Event()
    task = asyncio.ensure_future(start(stop))
    try:
        while not done():
            if task.done():
                await task  # surfaces a crash instead of waiting forever
                raise AssertionError("the loop ended before the condition was met")
            await wait_until(
                lambda: done() or task.done() or clock.pending_sleepers >= 1, limit_s=limit_s
            )
            if not done() and not task.done():
                await clock.advance(step_s)
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def advance_until(
    clock: ManualClock,
    done: Callable[[], bool],
    *,
    step_s: float = 1.0,
    limit_s: float = 5.0,
    max_steps: int = 500,
) -> None:
    """Advance the manual clock whenever something sleeps, until ``done()`` is true."""
    for _ in range(max_steps):
        if done():
            return
        await wait_until(lambda: done() or clock.pending_sleepers >= 1, limit_s=limit_s)
        if not done():
            await clock.advance(step_s)
    raise AssertionError("the condition was not met")
