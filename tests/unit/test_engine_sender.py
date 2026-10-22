"""SENDING: her pace, typing, library stickers only, errors by kind (R-ENG-009, R-SAFE-006)."""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import random
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tests.fixtures.synth_export import make_image_bytes
from tests.support.clock import ManualClock
from tests.support.engine_extras import PICTURES, add_sticker
from tests.support.engine_harness import Alerts, Out, ScriptedChannel, reference_pacing
from tests.support.gifs import animated_gif
from twin.channel.base import (
    PROBE_RUNNING,
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
from twin.llm.images import sniff_mime
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.library import store_sticker_file
from twin.storage.chat_models import Sticker as StickerRow

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


async def test_a_channel_that_holds_its_sends_for_a_probe_ends_the_reply_without_an_alert(
    rig: Rig,
) -> None:
    """R17-4 (D-684): the platform said nothing, so no "session over" alert and no ``expired``."""
    rig.channel.results.append(OutboundResult.failure(OutboundKind.WINDOW_REJECTED, PROBE_RUNNING))
    report = await rig.send(text("一"), text("二"))
    assert report.stop is StopReason.PROBE and rig.alerts.categories == []
    assert report.sent == [] and [b.text for b in report.rest] == ["一", "二"]


async def test_an_urgent_reply_goes_out_through_the_urgent_method_of_the_channel(rig: Rig) -> None:
    """R17-4 (D-685): ``urgent=True`` (the crisis answer) is passed on for every text bubble."""
    report = await rig.send(text("一"), text("二"), urgent=True, paced=False)
    assert report.stop is None and rig.channel.urgent == ["一", "二"]
    assert rig.channel.texts == ["一", "二"]
    plain = await rig.send(text("三"))
    assert plain.stop is None and rig.channel.urgent == ["一", "二"]  # an ordinary reply is not


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


# ------------------------------------------------ R17-38: a GIF as it is, or its first frame


class MimeChannel(ScriptedChannel):
    """A scripted channel that also notes the type each picture was sent as."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.mimes: list[str] = []

    async def send_image(self, data, mime, **options):  # type: ignore[no-untyped-def]
        self.mimes.append(mime)
        return await super().send_image(data, mime, **options)


def gif_sticker(services: Services, *, shades: int = 2, tags: tuple[str, ...] = ("开心",)):  # type: ignore[no-untyped-def]
    """A library sticker whose file is a GIF that moves; its first frame is red."""
    data = animated_gif(tuple((255 - 90 * n, 20 * n, 0) for n in range(shades + 1)), (10, 10))
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    with services.db.transaction() as session:
        row = StickerRow(md5=md5, status="pending", attempts=0, her_uses=3, user_uses=0)
        session.add(row)
        session.flush()
        store_sticker_file(row, data, services.media, services.clock.now_utc())
        row.tags = list(tags)
    record = StickerCatalog(services).get(md5)
    assert record is not None and record.available and record.mime == "image/gif"
    return record, data


async def test_a_gif_goes_out_as_it_is_unless_the_first_frame_is_asked_for(
    services: Services, clock: ManualClock
) -> None:
    sticker, data = gif_sticker(services)
    channel = MimeChannel(clock)
    for sender in (
        StickerSender(channel, services.media),  # no setting given: the file as it is
        StickerSender.from_services(channel, services),  # the default of the configuration
    ):
        assert (await sender.send_sticker(sticker)).ok
    assert services.settings.stickers.gif_send_mode == "as_is"
    assert channel.images == [data, data] and channel.mimes == ["image/gif", "image/gif"]


async def test_in_first_frame_mode_a_gif_goes_out_as_the_png_of_its_first_frame(
    services: Services, clock: ManualClock
) -> None:
    sticker, data = gif_sticker(services)
    assert sticker.first_frame_sha256
    still = services.media.read_bytes(sticker.first_frame_sha256)
    services.settings.stickers.gif_send_mode = "first_frame"
    channel = MimeChannel(clock)
    sender = StickerSender.from_services(channel, services)
    assert (await sender.send_sticker(sticker)).ok
    assert channel.images == [still] and channel.mimes == ["image/png"]
    assert channel.images[0] != data and sniff_mime(channel.images[0]) == "image/png"
    with Image.open(io.BytesIO(channel.images[0])) as frame:
        assert frame.convert("RGB").getpixel((0, 0)) == (255, 0, 0)  # the first frame
    # a picture that is not a GIF is not touched by the setting
    flat = add_sticker(services, seed=8)
    assert (await sender.send_sticker(flat)).ok
    assert channel.images[1] == PICTURES[flat.md5] and channel.mimes[1] == "image/png"


async def test_the_setting_is_read_when_the_sticker_is_sent(
    services: Services, clock: ManualClock
) -> None:
    sticker, data = gif_sticker(services)
    channel = MimeChannel(clock)
    sender = StickerSender.from_services(channel, services)
    services.settings.stickers.gif_send_mode = "first_frame"
    await sender.send_sticker(sticker)
    services.settings.stickers.gif_send_mode = "as_is"
    await sender.send_sticker(sticker)
    assert channel.mimes == ["image/png", "image/gif"] and channel.images[1] == data


async def test_a_gif_stored_before_first_frames_existed_gets_one_when_it_is_sent(
    services: Services, clock: ManualClock
) -> None:
    from twin.stickers.library import StickerLibraryAllowList

    sticker, _ = gif_sticker(services)
    with services.db.transaction() as session:
        row = session.get(StickerRow, sticker.md5)
        assert row is not None and row.first_frame_sha256
        services.media.delete(row.first_frame_sha256)
        row.first_frame_sha256 = None  # the library of an older version
    old = StickerCatalog(services).require(sticker.md5)
    assert old.first_frame_sha256 is None
    services.settings.stickers.gif_send_mode = "first_frame"
    channel = MimeChannel(clock)
    assert (await StickerSender.from_services(channel, services).send_sticker(old)).ok
    made = StickerCatalog(services).require(sticker.md5).first_frame_sha256
    assert made and channel.mimes == ["image/png"]
    assert hashlib.sha256(channel.images[0]).hexdigest() == made  # what went out is what was made
    assert StickerLibraryAllowList(services.db).allowed(made)  # and the channel will accept it


async def test_a_first_frame_that_cannot_be_made_is_never_replaced_by_the_animated_file(
    services: Services, clock: ManualClock
) -> None:
    broken = b"GIF89a" + bytes(range(60))  # a GIF by its header, nothing readable after it
    md5 = hashlib.md5(broken, usedforsecurity=False).hexdigest()
    with services.db.transaction() as session:
        row = StickerRow(md5=md5, status="pending", attempts=0, her_uses=3, user_uses=0)
        session.add(row)
        session.flush()
        store_sticker_file(row, broken, services.media, services.clock.now_utc())
        row.tags = ["开心"]
    sticker = StickerCatalog(services).require(md5)
    assert sticker.available and sticker.first_frame_sha256 is None
    services.settings.stickers.gif_send_mode = "first_frame"
    channel = MimeChannel(clock)
    sender = StickerSender.from_services(channel, services)
    with pytest.raises(MediaNotAllowed, match="not sent in its place"):
        await sender.send_sticker(sticker)
    assert channel.images == []  # nothing went out - not the animated file either
    # in the sender of a reply this is a bubble that is skipped, as for any sticker not sendable
    rig = Rig(services, clock)
    rig.sender = BubbleSender(
        rig.channel,
        StickerSender.from_services(rig.channel, services),
        rig.catalog.get,
        clock,
        rig.alerts,
        random.Random(4),
    )
    report = await rig.send(OutBubble("sticker", "[表情包:开心]", md5), text("好"))
    assert [b.sticker_md5 for b in report.skipped] == [md5] and rig.channel.texts == ["好"]
    assert rig.channel.images == []


async def test_a_sender_without_a_way_to_make_the_first_frame_refuses_rather_than_send_the_gif(
    services: Services, clock: ManualClock
) -> None:
    sticker, _ = gif_sticker(services)
    bare = replace(sticker, first_frame_sha256=None)
    channel = MimeChannel(clock)
    sender = StickerSender(channel, services.media, gif_send_mode=lambda: "first_frame")
    with pytest.raises(MediaNotAllowed, match="not sent in its place"):
        await sender.send_sticker(bare)
    assert channel.images == []


async def test_the_first_frame_passes_the_allow_list_of_the_terminal_channel_and_shows_its_label(
    services: Services, clock: ManualClock
) -> None:
    from tests.support.console import RecordingOutput, ScriptedInput
    from twin.channel.local import LocalConsoleChannel

    sticker, _ = gif_sticker(services, tags=("撒娇",))
    services.settings.stickers.gif_send_mode = "first_frame"
    out = RecordingOutput()
    channel_input = ScriptedInput()
    channel = LocalConsoleChannel.from_services(services, input=channel_input, output=out)
    await channel.start()
    try:
        channel_input.feed("hi")
        async for _ in channel.incoming():
            break
        assert (await StickerSender.from_services(channel, services).send_sticker(sticker)).ok
        stranger = make_image_bytes(random.Random(77), "PNG")  # a picture that is no sticker
        with pytest.raises(MediaNotAllowed, match="neither a sticker"):
            await channel.send_image(stranger, "image/png")
    finally:
        await channel.stop()
    [line] = out.lines
    assert line.startswith("bot: [表情包：撒娇] ") and line.rstrip().endswith(".png")
