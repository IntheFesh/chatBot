"""A reply that the channel could not start, or that a long stop cut in two (round 18, B2-B4).

* B2 (D-751): the first bubble does not leave the machine (the network is down for half a minute):
  the round is tried again 2 to 10 minutes later like a failed generation (R-ENG-010), at most
  ``MAX_RETRIES`` times; before, the round ended and the message was never answered.
* B3 (D-752): a channel probe starts while the first bubble is being typed: the channel refuses
  with ``probe_running`` and nothing is out; the round goes back to wait for the probe like a
  draft that the probe found before its first bubble (D-686); before, the round ended.
* B4 (D-753): a reply cut by a stop is resumed only while it is still the same moment: after a
  long downtime, or into her deep sleep, the rest is dropped (``send_dropped_stale``).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_extras import restart
from tests.support.engine_harness import (
    START,
    FixedDay,
    Harness,
    ScriptedChannel,
    bubbles_written,
    build_harness,
    make_draft,
    run_to_idle,
)
from twin.channel.base import (
    PROBE_RUNNING,
    OutboundKind,
    OutboundResult,
    QuoteTarget,
    SendBypass,
)
from twin.engine.decision import MAX_RETRIES
from twin.engine.machine import STALE_REST_S
from twin.services import Services
from twin.storage.engine_models import BotTurn

NETWORK = OutboundResult.failure(OutboundKind.NETWORK, "connect_error")


def steps(harness: Harness) -> list[dict[str, Any]]:
    with harness.services.db.session() as session:
        rows = session.scalars(select(BotTurn).where(BotTurn.direction == "out"))
        return [dict(a) for row in rows for a in row.actions or []]


def text_times(harness: Harness) -> list[Any]:
    return [o.at for o in harness.channel.out if o.kind == "text"]


# ------------------------------------------------------------------------------- B2


async def test_r18_b2_a_network_blip_on_the_first_bubble_is_answered_minutes_later(
    services: Services, clock: ManualClock
) -> None:
    """R18-B2: three sends of the first bubble never leave the machine (about half a minute).

    Before: the round ended - "在吗" was never answered, also when the network was back.  Now the
    round is tried again 2 to 10 minutes later (a new draft, R-ENG-010) and the answer goes out.
    """
    rig = build_harness(services, clock)
    rig.channel.results.extend([NETWORK] * 3)
    rig.writer.add(make_draft("在的", "怎么啦"), make_draft("在的", "怎么了"))
    await rig.engine.start()
    try:
        await rig.message("在吗")
        await run_to_idle(rig.engine, clock, until=lambda: not rig.channel.results)
        failed_at = clock.now_utc()
        await run_to_idle(rig.engine, clock)
        assert rig.channel.texts == ["在的", "怎么了"]
        waited = (text_times(rig)[0] - failed_at).total_seconds()
        assert 120 <= waited <= 600 + 60  # the retry of R-ENG-010, then her typing
        assert rig.writer.calls == 2
        assert {"step": "retried_later", "count": 1} in steps(rig)
        snap = rig.engine.snapshot()
        assert snap.state == "IDLE" and snap.pending == ()
    finally:
        await rig.engine.stop()


async def test_r18_b2_a_network_that_never_comes_back_ends_the_round_after_the_retries(
    services: Services, clock: ManualClock
) -> None:
    """R18-B2: the retries are bounded; then the round ends the way it always did (no loop)."""
    rig = build_harness(services, clock)
    rig.channel.results.extend([NETWORK] * (3 * (MAX_RETRIES + 1)))
    await rig.engine.start()
    try:
        await rig.message("在吗")
        await run_to_idle(rig.engine, clock)
        assert rig.channel.texts == [] and not rig.channel.results  # every try, and no more
        assert rig.writer.calls == MAX_RETRIES + 1
        assert "channel_send_failed" in rig.alerts.categories
        snap = rig.engine.snapshot()
        assert snap.state == "IDLE" and snap.pending == ()
    finally:
        await rig.engine.stop()


async def test_r18_b2_a_refused_first_bubble_still_ends_the_round_at_once(
    services: Services, clock: ManualClock
) -> None:
    """R18-B2: only a send that never left the machine is tried again, not a refusal."""
    rig = build_harness(services, clock)
    rig.channel.results.append(OutboundResult.failure(OutboundKind.REJECTED, "bad_request"))
    await rig.engine.start()
    try:
        await rig.message("在吗")
        await run_to_idle(rig.engine, clock)
        assert rig.channel.texts == [] and rig.writer.calls == 1
        assert rig.engine.snapshot().state == "IDLE"
    finally:
        await rig.engine.stop()


# ------------------------------------------------------------------------------- B3


class ProbeChannel(ScriptedChannel):
    """A channel whose probe starts the moment the first bubble begins to be typed."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.probe_on = False
        self.start_on_typing = True
        self.refused = 0

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        if self.probe_on:
            self.refused += 1
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, PROBE_RUNNING)
        return await super().send_text(text, quote, recipient=recipient, bypass=bypass)

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        await super().send_typing(active, recipient=recipient)
        if active and self.start_on_typing:
            self.probe_on = True  # the user starts ``twin channel probe start`` right now


async def test_r18_b3_a_probe_that_starts_during_the_first_bubble_does_not_drop_the_round(
    services: Services, clock: ManualClock
) -> None:
    """R18-B3: before, the refused first bubble ended the round and "在吗" was never answered."""
    channel = ProbeChannel(clock)
    rig = build_harness(services, clock, channel=channel, probe_active=lambda: channel.probe_on)
    rig.writer.add(make_draft("在的", "怎么啦"), make_draft("在的", "怎么了"))
    await rig.engine.start()
    try:
        await rig.message("在吗")
        await run_to_idle(
            rig.engine,
            clock,
            until=lambda: (
                channel.refused == 1
                and rig.engine.snapshot().state == "COLLECTING"
                and rig.engine.waiting_until is not None
            ),
        )
        snap = rig.engine.snapshot()
        assert channel.texts == [] and len(snap.pending) == 1  # the message still waits
        assert "channel_session_expired" not in rig.alerts.categories
        channel.probe_on = channel.start_on_typing = False  # the probe ends
        await rig.engine.on_state_change(0, 1)
        await run_to_idle(rig.engine, clock)
        assert channel.texts == ["在的", "怎么了"] and rig.writer.calls == 2
    finally:
        await rig.engine.stop()


# ------------------------------------------------------------------------------- B4


def sleepy_day() -> FixedDay:
    """Free until START+6h, deep sleep until START+20h."""
    return FixedDay(
        (START - timedelta(days=1), "free"),
        (START + timedelta(hours=6), "deep_sleep"),
        (START + timedelta(hours=20), "free"),
    )


async def cut_after_one(services: Services, clock: ManualClock) -> Harness:
    rig = build_harness(services, clock, day=sleepy_day())
    await rig.engine.start()
    rig.writer.add(make_draft("一", "二", "三"))
    await rig.message("数数")
    await run_to_idle(rig.engine, clock, until=bubbles_written(rig.engine, 1))
    assert rig.engine.snapshot().state == "SENDING"
    return rig


async def test_r18_b4_the_rest_of_a_reply_is_not_sent_into_her_deep_sleep_after_a_downtime(
    services: Services, clock: ManualClock
) -> None:
    """R18-B4: stopped after bubble 1 of 3, back ten hours later inside her deep sleep.

    Before: "二" and "三" went out at once, at night, hours out of context.  Now the rest is
    dropped and noted on the reply.
    """
    rig = await cut_after_one(services, clock)
    second = await restart(rig, clock_jump_s=10 * 3600)
    try:
        assert second.day.her_state(clock.now_utc()).kind == "deep_sleep"
        await run_to_idle(second.engine, clock)
        assert second.channel.texts == []
        assert {"step": "send_dropped_stale", "count": 2, "detail": "deep_sleep"} in steps(second)
        snap = second.engine.snapshot()
        assert snap.state == "IDLE" and snap.pending == ()
    finally:
        await second.engine.stop()


async def test_r18_b4_the_rest_of_a_reply_is_dropped_after_a_long_downtime_while_she_is_awake(
    services: Services, clock: ManualClock
) -> None:
    """R18-B4: back after longer than ``STALE_REST_S``, she is awake: still stale, dropped."""
    rig = await cut_after_one(services, clock)
    second = await restart(rig, clock_jump_s=STALE_REST_S + 60)
    try:
        assert second.day.her_state(clock.now_utc()).kind == "free"
        await run_to_idle(second.engine, clock)
        assert second.channel.texts == []
        assert {"step": "send_dropped_stale", "count": 2, "detail": "gap"} in steps(second)
    finally:
        await second.engine.stop()


async def test_r18_b4_a_short_stop_while_she_is_awake_still_sends_the_rest(
    services: Services, clock: ManualClock
) -> None:
    """R18-B4 (control): a stop of a few minutes, she is awake: the reply goes on as before."""
    rig = await cut_after_one(services, clock)
    second = await restart(rig, clock_jump_s=300)
    try:
        await run_to_idle(second.engine, clock)
        assert second.channel.texts == ["二", "三"]
        assert all(step["step"] != "send_dropped_stale" for step in steps(second))
    finally:
        await second.engine.stop()


async def test_r18_b4_a_reply_with_nothing_out_waits_for_her_waking_after_a_downtime(
    services: Services, clock: ManualClock
) -> None:
    """R18-B4 (the item-32 re-judgement for a reply that is made but has nothing out yet).

    The reply was ready (SENDING, its first bubble not yet typed) when the process stopped; it
    comes back ten hours later inside her deep sleep.  Before: both bubbles went out at once, at
    night.  Now the reply is decided again from that moment, like a reply that a plan change
    put into her sleep (D-722): queued to her waking, and made anew then ("you just woke up").
    """
    day = sleepy_day()
    rig = build_harness(services, clock, day=day)
    await rig.engine.start()
    rig.writer.add(make_draft("一", "二"))
    await rig.message("数数")
    await run_to_idle(rig.engine, clock, until=lambda: rig.engine.snapshot().state == "SENDING")
    second = await restart(rig, clock_jump_s=10 * 3600)
    try:
        second.writer.add(make_draft("刚醒", "怎么啦"))
        await run_to_idle(second.engine, clock)
        sent = [(o.text, o.at) for o in second.channel.out if o.kind == "text"]
        assert [text for text, _ in sent] == ["刚醒", "怎么啦"]
        assert all(day.her_state(at).kind == "free" for _, at in sent)
        assert second.writer.calls == 1 and second.writer.contexts[0].woke_up
    finally:
        await second.engine.stop()
