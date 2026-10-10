"""``InMemoryChannel``: a complete channel that keeps everything in memory (R-EVAL-009)."""

from __future__ import annotations

import ast
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fixtures.synth_export import make_image_bytes
from tests.support.clock import ManualClock
from twin.channel.base import (
    AuthState,
    Channel,
    MediaNotAllowed,
    OutboundKind,
    QuoteTarget,
    RecipientNotAllowed,
)
from twin.channel.policy import sha256_hex
from twin.eval.channel import EVAL_USER_ID, InMemoryChannel

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"
PNG = make_image_bytes(random.Random(1), "PNG")
JPEG = make_image_bytes(random.Random(2), "JPEG")


class Allowed:
    def __init__(self, *digests: str) -> None:
        self.digests = set(digests)

    def allowed(self, sha256: str) -> bool:
        return sha256 in self.digests


def channel(clock: ManualClock, **options: object) -> InMemoryChannel:
    return InMemoryChannel(clock=clock, **options)  # type: ignore[arg-type]


async def test_it_implements_the_channel_interface_and_keeps_what_is_sent(
    clock: ManualClock,
) -> None:
    memory = channel(clock)
    assert isinstance(memory, Channel)
    await memory.start()
    first = await memory.send_text("你好")
    second = await memory.send_text("在吗", QuoteTarget("in-1", "吃了吗"))
    assert first.ok and second.ok and first.message_id != second.message_id
    assert [(m.kind, m.text) for m in memory.sent] == [("text", "你好"), ("text", "在吗")]
    assert memory.sent[1].quote == QuoteTarget("in-1", "吃了吗") and memory.sent[0].quote is None
    assert memory.sent[0].at == clock.now_utc()
    state = memory.session_state()
    assert state.auth is AuthState.OK and state.bound and state.outbound_since_inbound == 2
    assert state.extra["in_memory"] is True
    memory.clear()
    assert memory.sent == []
    await memory.stop()
    await memory.stop()  # twice is fine


async def test_it_only_talks_to_the_one_bound_user(clock: ManualClock) -> None:
    memory = channel(clock)
    assert (await memory.send_text("嗨", recipient=EVAL_USER_ID)).ok
    with pytest.raises(RecipientNotAllowed):
        await memory.send_text("嗨", recipient="someone-else")
    with pytest.raises(RecipientNotAllowed):
        await memory.send_image(PNG, "image/png", recipient="someone-else")
    with pytest.raises(RecipientNotAllowed):
        await memory.send_typing(True, recipient="someone-else")
    assert len(memory.sent) == 1


async def test_empty_and_overlong_texts_are_refused_like_the_real_channels_do(
    clock: ManualClock,
) -> None:
    memory = channel(clock)
    empty = await memory.send_text("  \n")
    assert not empty.ok and empty.kind is OutboundKind.REJECTED and empty.reason == "empty_text"
    long = await memory.send_text("字" * 5000)
    assert not long.ok and long.reason == "text_too_long" and memory.sent == []


async def test_pictures_go_through_the_same_allow_list_as_the_real_channels(
    clock: ManualClock,
) -> None:
    with_nothing = channel(clock)
    with pytest.raises(MediaNotAllowed):
        await with_nothing.send_image(PNG, "image/png")  # the default allows no picture
    memory = channel(clock, media_policy=Allowed(sha256_hex(PNG)))
    sent = await memory.send_image(PNG, "image/png")
    assert sent.ok and memory.sent[0].kind == "image" and memory.sent[0].sha256 == sha256_hex(PNG)
    with pytest.raises(MediaNotAllowed):
        await memory.send_image(JPEG, "image/jpeg")  # a picture that is not on the list
    with pytest.raises(MediaNotAllowed, match="declared type"):
        await memory.send_image(PNG, "image/jpeg")  # bytes that are not what they say
    with pytest.raises(MediaNotAllowed, match="not sent"):
        await memory.send_image(PNG, "application/pdf")
    assert len(memory.sent) == 1


async def test_a_picture_can_be_given_as_a_path(clock: ManualClock, tmp_path: Path) -> None:
    path = tmp_path / "sticker.png"
    path.write_bytes(PNG)
    memory = channel(clock, media_policy=Allowed(sha256_hex(PNG)))
    assert (await memory.send_image(path, "image/png")).ok


async def test_messages_of_the_user_arrive_through_incoming(clock: ManualClock) -> None:
    memory = channel(clock)
    await memory.start()
    when = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    pushed = memory.push("在干嘛", at=when)
    memory.push("喂")
    await memory.stop()
    received = [message async for message in memory.incoming()]
    assert [m.text for m in received] == ["在干嘛", "喂"] and received[0] == pushed
    assert received[0].at == when and received[1].at == clock.now_utc()
    assert [m async for m in memory.incoming()] == []  # a second pass ends at once
    assert memory.session_state().last_inbound_at == clock.now_utc()


async def test_typing_and_capabilities(clock: ManualClock) -> None:
    memory = channel(clock)
    await memory.send_typing(True)
    assert memory.typing and memory.session_state().extra["typing"] is True
    await memory.send_text("好")
    assert not memory.typing  # sending ends the typing, as on the real channels
    caps = memory.capabilities()
    assert caps.supports_quote and caps.supports_typing and caps.max_text_chars
    assert not channel(clock, supports_quote=False).capabilities().supports_quote


def test_the_evaluation_never_builds_the_wechat_channel() -> None:
    """R-EVAL-009: no module of the evaluation names, imports or builds ``IlinkChannel``."""
    offenders: list[str] = []
    for path in sorted((SRC / "eval").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("twin.channel.ilink") and any(
                    a.name in {"IlinkChannel", "channel"} for a in node.names
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
                if node.module == "twin.channel.ilink.channel":
                    offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "IlinkChannel":
                offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "IlinkChannel":
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
