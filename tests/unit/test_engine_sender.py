"""SENDING: her pace, typing, library stickers only, errors by kind (R-ENG-009, R-SAFE-006)."""

from __future__ import annotations

import ast
import inspect
import random
from pathlib import Path

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_extras import PICTURES, add_sticker
from tests.support.engine_harness import Alerts, Out, ScriptedChannel, reference_pacing
from twin.channel.base import (
    AuthState,
    CapabilityNotSupported,
    MediaNotAllowed,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    RecipientNotAllowed,
)
from twin.engine.pacing import MIN_INTERVAL_S, PacingModel
from twin.engine.sender import BubbleSender, OutBubble, SentBubble, StopReason
from twin.engine.sticker_sender import Sticker, StickerSender
from twin.services import Services
from twin.stickers.catalog import StickerCatalog

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"


class Rig:
    """A sender on a scripted channel; ``waits`` records the pauses it asked for."""

    def __init__(self, services: Services, clock: ManualClock, **channel_options: object) -> None:
        self.clock = clock
        self.channel = ScriptedChannel(clock, **channel_options)  # type: ignore[arg-type]
        self.alerts = Alerts()
        self.catalog = StickerCatalog(services)
        self.services = services
        self.sender = BubbleSender(
            self.channel,
            StickerSender(self.channel, services.media),
            self.catalog.get,
            clock,
            self.alerts,
            random.Random(4),
        )
        self.waits: list[float] = []
        self.interrupt_at: int | None = None
        self.stored: list[SentBubble] = []
        self.channel.push("hi")  # the window is open

    async def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        return self.interrupt_at is not None and len(self.waits) >= self.interrupt_at

    async def keep(self, sent: SentBubble) -> None:
        self.stored.append(sent)

    async def send(self, *bubbles: OutBubble, **options: object):  # type: ignore[no-untyped-def]
        pacing = options.pop("pacing", reference_pacing())
        return await self.sender.send(
            bubbles,
            pacing=pacing,
            wait=self.wait,
            on_sent=self.keep,
            **options,  # type: ignore[arg-type]
        )


def text(value: str) -> OutBubble:
    return OutBubble("text", value)


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    return Rig(services, clock)


# ----------------------------------------------------------------------- the pace


async def test_the_first_bubble_waits_for_its_typing_and_later_ones_for_a_pause_too(
    rig: Rig,
) -> None:
    report = await rig.send(text("你好呀"), text("在做什么呢"))
    assert report.stop is None and [s.bubble.text for s in report.sent] == ["你好呀", "在做什么呢"]
    # typing three characters at 0.5 s; then her pause of 4 s and typing five characters
    assert rig.waits == [1.5, 4.0 + 2.5]


async def test_no_bubble_follows_another_faster_than_a_second(
    services: Services, clock: ManualClock
) -> None:
    quick = Rig(services, clock)
    pacing = reference_pacing(gap_s=0.0, seconds_per_char=None)
    report = await quick.send(text("好"), text("的"), pacing=pacing)
    assert report.stop is None and quick.waits == [MIN_INTERVAL_S, MIN_INTERVAL_S]


async def test_a_continuing_reply_does_not_start_like_a_new_one(rig: Rig) -> None:
    await rig.send(text("接着说"), first_of_reply=False)
    assert rig.waits == [4.0 + 1.5]  # the pause between her messages is there from the start


async def test_typing_is_shown_before_each_bubble_where_the_channel_can(rig: Rig) -> None:
    await rig.send(text("一"), text("二"))
    kinds = [(o.kind, o.active) for o in rig.channel.out]
    assert kinds == [("typing", True), ("text", None), ("typing", True), ("text", None)]


async def test_an_unpaced_reply_goes_out_without_waiting_or_typing(rig: Rig) -> None:
    report = await rig.send(text("一"), text("二"), paced=False)
    assert len(report.sent) == 2 and rig.waits == []
    assert [o.kind for o in rig.channel.out] == ["text", "text"]


async def test_a_user_who_writes_stops_the_reply_and_the_rest_is_handed_back(rig: Rig) -> None:
    rig.interrupt_at = 2
    report = await rig.send(text("一"), text("二"), text("三"))
    assert report.stop is StopReason.INTERRUPTED
    assert [s.bubble.text for s in report.sent] == ["一"]
    assert [b.text for b in report.rest] == ["二", "三"]
    assert rig.channel.out[-1].active is False  # the typing indicator is cleared


async def test_every_bubble_is_reported_before_the_next_one_starts(rig: Rig) -> None:
    seen_at_wait: list[int] = []

    async def wait(seconds: float) -> bool:
        seen_at_wait.append(len(rig.stored))
        return False

    await rig.sender.send(
        [text("一"), text("二")], pacing=reference_pacing(), wait=wait, on_sent=rig.keep
    )
    assert seen_at_wait == [0, 1]


async def test_the_quote_goes_with_the_first_text_bubble_only(
    services: Services, clock: ManualClock
) -> None:
    rig = Rig(services, clock, supports_quote=True)
    target = QuoteTarget("m1", "去看电影")
    await rig.send(text("好呀"), text("我也想"), quote=target)
    quotes = [o.quote for o in rig.channel.out if o.kind == "text"]
    assert quotes == [target, None]


# ---------------------------------------------------------------------- the checks


@pytest.mark.parametrize(
    ("change", "reason", "category"),
    [
        ({"expired": True}, StopReason.EXPIRED, "channel_session_expired"),
        ({"auth": AuthState.NEEDS_RELOGIN}, StopReason.EXPIRED, "channel_session_expired"),
        ({"bound": False}, StopReason.UNBOUND, "channel_unbound"),
        ({"remaining": 0}, StopReason.QUOTA, "channel_quota_exhausted"),
    ],
)
async def test_a_dead_session_stops_the_reply_with_an_alert_instead_of_trying(
    rig: Rig, change: dict[str, object], reason: StopReason, category: str
) -> None:
    for name, value in change.items():
        setattr(rig.channel, name, value)
    report = await rig.send(text("一"), text("二"))
    assert report.stop is reason and report.sent == [] and len(report.rest) == 2
    assert rig.alerts.categories == [category]
    assert rig.channel.texts == []


async def test_a_session_that_ends_in_the_middle_stops_there(rig: Rig) -> None:
    def expire(item: Out) -> None:
        if item.kind == "text":
            rig.channel.expired = True

    rig.channel.on_send = expire
    report = await rig.send(text("一"), text("二"))
    assert [s.bubble.text for s in report.sent] == ["一"]
    assert report.stop is StopReason.EXPIRED and [b.text for b in report.rest] == ["二"]


# --------------------------------------------------------------- what the channel says


async def test_a_send_that_never_left_the_machine_is_repeated_twice(rig: Rig) -> None:
    network = OutboundResult.failure(OutboundKind.NETWORK, "connect_error")
    rig.channel.results.extend([network, network])
    report = await rig.send(text("一"))
    assert report.stop is None and rig.channel.texts == ["一"]
    assert len(rig.waits) == 3 and all(5.0 <= pause <= 15.0 for pause in rig.waits[1:])


async def test_a_send_that_keeps_failing_ends_the_reply_with_an_alert(rig: Rig) -> None:
    network = OutboundResult.failure(OutboundKind.NETWORK, "connect_error")
    rig.channel.results.extend([network, network, network])
    report = await rig.send(text("一"), text("二"))
    assert report.stop is StopReason.FAILED and rig.alerts.categories == ["channel_send_failed"]
    assert report.sent == [] and len(report.rest) == 2


async def test_a_send_with_an_unknown_outcome_is_not_repeated(rig: Rig) -> None:
    rig.channel.results.append(OutboundResult.failure(OutboundKind.AMBIGUOUS, "read_timeout"))
    report = await rig.send(text("一"), text("二"))
    assert report.stop is None and rig.channel.texts == ["一", "二"]
    assert report.sent[0].ambiguous and not report.sent[1].ambiguous


@pytest.mark.parametrize("kind", [OutboundKind.WINDOW_REJECTED, OutboundKind.AUTH_EXPIRED])
async def test_a_refused_window_or_login_ends_the_reply(rig: Rig, kind: OutboundKind) -> None:
    rig.channel.results.append(OutboundResult.failure(kind, "session_expired"))
    report = await rig.send(text("一"), text("二"))
    assert report.stop is StopReason.EXPIRED and rig.alerts.categories == [
        "channel_session_expired"
    ]
    assert len(rig.channel.results) == 0 and report.sent == []


async def test_a_rejected_text_ends_the_reply_but_a_rejected_sticker_is_skipped(
    services: Services, clock: ManualClock
) -> None:
    rig = Rig(services, clock)
    rig.channel.results.append(OutboundResult.failure(OutboundKind.REJECTED, "bad_request"))
    report = await rig.send(text("一"))
    assert report.stop is StopReason.FAILED
    sticker = add_sticker(services)
    rig.channel.results.append(OutboundResult.failure(OutboundKind.UPLOAD_FAILED, "cdn"))
    report = await rig.send(OutBubble("sticker", "[表情包:开心]", sticker.md5), text("好"))
    assert report.stop is None and [b.text for b in report.skipped] == ["[表情包:开心]"]
    assert rig.channel.texts == ["好"]


async def test_a_recipient_the_channel_refuses_ends_the_reply(rig: Rig) -> None:
    async def refuse(*args: object, **kwargs: object) -> OutboundResult:
        raise RecipientNotAllowed("someone else")

    rig.channel.send_text = refuse  # type: ignore[method-assign]
    report = await rig.send(text("一"))
    assert report.stop is StopReason.UNBOUND and rig.alerts.categories == ["channel_unbound"]


async def test_a_capability_the_channel_lacks_skips_the_bubble(rig: Rig) -> None:
    sticker = add_sticker(rig.services)
    rig.channel.image_error = CapabilityNotSupported("no images here")
    report = await rig.send(OutBubble("sticker", "[表情包:开心]", sticker.md5), text("好"))
    assert report.stop is None and len(report.skipped) == 1 and rig.channel.texts == ["好"]


# --------------------------------------------------------------------------- stickers


async def test_a_sticker_goes_out_as_the_picture_of_the_library(rig: Rig) -> None:
    sticker = add_sticker(rig.services, seed=3)
    report = await rig.send(OutBubble("sticker", "[表情包:开心]", sticker.md5))
    assert report.stop is None
    assert rig.channel.images == [PICTURES[sticker.md5]]  # the bytes as they are


async def test_a_sticker_the_library_does_not_know_is_skipped(rig: Rig) -> None:
    report = await rig.send(OutBubble("sticker", "[表情包:开心]", "f" * 32), text("好"))
    assert [b.sticker_md5 for b in report.skipped] == ["f" * 32] and rig.channel.texts == ["好"]
    assert rig.channel.images == []


async def test_only_a_usable_sticker_can_be_handed_to_the_sticker_sender(
    services: Services, clock: ManualClock
) -> None:
    channel = ScriptedChannel(clock)
    sender = StickerSender(channel, services.media)
    good = add_sticker(services, seed=5)
    assert (await sender.send_sticker(good)).ok and len(channel.images) == 1
    from dataclasses import replace

    for broken in (
        replace(good, status="pending"),
        replace(good, status="md5_mismatch"),
        replace(good, disabled=True),
        replace(good, sha256=None),
        replace(good, mime=None),
    ):
        with pytest.raises(MediaNotAllowed):
            await sender.send_sticker(broken)
    assert len(channel.images) == 1


def test_the_sticker_sender_has_no_way_to_take_a_path_or_bytes() -> None:
    signature = inspect.signature(StickerSender.send_sticker)
    assert list(signature.parameters) == ["self", "sticker"]
    assert signature.parameters["sticker"].annotation in ("Sticker", Sticker)
    assert Sticker.__name__ == "StickerRecord"  # a record of the library, not a file


def test_no_module_of_the_engine_but_the_sticker_sender_calls_send_image() -> None:
    callers = []
    for path in sorted((SRC / "engine").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "send_image":
                callers.append(path.relative_to(SRC).as_posix())
    assert set(callers) == {"engine/sticker_sender.py"}


def test_the_pacing_floor_is_one_second() -> None:
    assert PacingModel().bubble_interval(random.Random(1), 0, first=True) == 1.0
