"""The answer to a crisis is finished, whatever stops it halfway - never the persona (R-SAFE-001).

Round 18, item B1 (D-750).  ``_screen`` marks the messages "screened" before the out-of-role answer
goes out.  Before this item nothing else on disk said that an answer to a crisis was under way: a
process that died after its first bubble (a kill, a stop whose grace ran out on a send that never
answered) or a single failed write of a bubble left the round "screened" and the next pass gave
the crisis message to the persona, which answered it in role - and the line with the hotline was
never sent.  Now the answer is written into the round notes (``RoundData.crisis``) in the same step
that marks the messages screened, before its first bubble; every pass of the driver finishes it
before anything else, and its bookkeeping writes are best-effort.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.engine_extras import restart
from tests.support.engine_harness import (
    Harness,
    ScriptedChannel,
    ScriptedCrisis,
    build_harness,
    make_draft,
    run_to_idle,
)
from tests.support.waiting import wait_until
from twin.channel.base import OutboundKind, OutboundResult, QuoteTarget, SendBypass
from twin.engine.decision import MAX_RETRIES
from twin.engine.roundstate import CrisisAnswer, RoundData
from twin.engine.state_store import ConversationSnapshot
from twin.services import Services
from twin.storage.engine_models import BotTurn

ANSWER = ["我想先停一下。", "你现在还好吗？", "也可以联系 988。"]


class HangingSecondSend(ScriptedChannel):
    """The first send goes out; the second one never returns (the process dies inside it)."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.calls = 0
        self.hang = asyncio.Event()

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        self.calls += 1
        if self.calls == 2:
            await self.hang.wait()  # never set: a send that never answers
        return await super().send_text(text, quote, recipient=recipient, bypass=bypass)


def safety_rows(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        rows = list(
            session.scalars(
                select(BotTurn)
                .where(BotTurn.direction == "out", BotTurn.kind == "text")
                .order_by(BotTurn.at, BotTurn.id)
            )
        )
        session.expunge_all()
    return rows


async def test_r18_b1_a_crisis_answer_cut_by_a_kill_is_finished_after_the_restart(
    services: Services, clock: ManualClock
) -> None:
    """R18-B1: the first crisis bubble is out, the second never returns, the process dies.

    Before: the restart found the message "screened" and the persona answered it in role
    ("哈哈 怎么啦"); the hotline was never sent.  Now the rest of the answer goes out (urgent, at
    once), the persona is never asked, the message is not judged a second time.
    """
    channel = HangingSecondSend(clock)
    rig = build_harness(
        services, clock, channel=channel, crisis=ScriptedCrisis("不想活"), stop_grace_s=0.05
    )
    await rig.engine.start()
    await rig.message("我真的不想活了")
    await run_to_idle(rig.engine, clock, until=lambda: channel.calls >= 2)
    await wait_until(lambda: channel.calls >= 2, limit_s=5.0)
    stored = RoundData.of(rig.engine.snapshot()).crisis
    assert stored is not None and list(stored.unsent) == ANSWER[1:]  # written before it is sent
    second = await restart(rig, crisis=ScriptedCrisis("不想活"))
    second.writer.add(make_draft("哈哈 怎么啦"))
    try:
        await run_to_idle(second.engine, clock)
        assert second.channel.texts == ANSWER[1:]
        assert second.channel.urgent == ANSWER[1:]  # still the urgent send of the answer
        assert second.writer.calls == 0  # the persona never sees the crisis message
        assert second.crisis.handled == []  # judged once, before the first restart
        snap = second.engine.snapshot()
        assert snap.state == "IDLE" and snap.pending == ()
        rows = safety_rows(second)
        assert [row.text for row in rows] == ANSWER  # one reply of the safety backend
        assert {row.backend for row in rows if row.backend} == {"safety"}
        assert len({row.reply_id for row in rows}) == 1
    finally:
        await second.engine.stop()


async def test_r18_b1_a_failed_write_during_the_crisis_answer_does_not_stop_it(
    services: Services, clock: ManualClock
) -> None:
    """R18-B1: one transient database error while the first crisis bubble is written down.

    Before: the step failed after the first bubble, the next pass found the message screened and
    the persona answered in role; "也可以联系 988。" was never sent.  Now the write is best-effort:
    the whole answer goes out and the persona says nothing.
    """
    rig = build_harness(services, clock, crisis=ScriptedCrisis("不想活"))
    original = rig.store.add_bubble
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return original(*args, **kwargs)

    rig.store.add_bubble = flaky  # type: ignore[method-assign]
    rig.writer.add(make_draft("哈哈 怎么啦"))
    await rig.engine.start()
    try:
        await rig.message("我真的不想活了")
        await run_to_idle(rig.engine, clock)
        assert rig.channel.urgent == ANSWER and rig.channel.texts == ANSWER
        assert rig.writer.calls == 0
        assert rig.engine.snapshot().state == "IDLE"
    finally:
        await rig.engine.stop()


async def test_r18_b1_a_finished_crisis_answer_is_not_sent_again_after_a_restart(
    services: Services, clock: ManualClock
) -> None:
    """R18-B1: once the answer is out the round is over: a restart sends nothing more."""
    rig = build_harness(services, clock, crisis=ScriptedCrisis("不想活"))
    await rig.engine.start()
    await rig.message("我真的不想活了")
    await run_to_idle(rig.engine, clock)
    assert rig.channel.texts == ANSWER
    assert RoundData.of(rig.engine.snapshot()).crisis is None
    second = await restart(rig, crisis=ScriptedCrisis("不想活"))
    try:
        await run_to_idle(second.engine, clock)
        assert second.channel.out == [] and second.writer.calls == 0
        assert [row.text for row in safety_rows(second)] == ANSWER
    finally:
        await second.engine.stop()


async def test_r18_b1_a_network_failure_of_the_answer_is_tried_again_later_never_in_role(
    services: Services, clock: ManualClock
) -> None:
    """R18-B1: the channel cannot send the hotline line (three network failures in a row).

    Before: the round ended with the line unsent.  Now the rest of the answer is tried again a
    few minutes later (the retry of R-ENG-010) and goes out; the persona is never asked, and a
    message the user writes meanwhile waits for the next round.
    """
    rig = build_harness(services, clock, crisis=ScriptedCrisis("不想活"))
    rig.channel.results.extend(
        [OutboundResult.success(message_id="c1")]
        + [OutboundResult.failure(OutboundKind.NETWORK, "connect_error")] * 3
    )
    await rig.engine.start()
    try:
        await rig.message("我真的不想活了")
        await run_to_idle(
            rig.engine,
            clock,
            until=lambda: (
                (c := RoundData.of(rig.engine.snapshot()).crisis) is not None and c.retries == 1
            ),
        )
        failed_at = clock.now_utc()
        assert rig.channel.texts == ANSWER[:1]
        await run_to_idle(rig.engine, clock)
        assert rig.channel.texts == ANSWER and rig.channel.urgent[-2:] == ANSWER[1:]
        later = [o.at for o in rig.channel.out if o.kind == "text"][1]
        assert 120 <= (later - failed_at).total_seconds() <= 600 + 60
        assert rig.writer.calls == 0
    finally:
        await rig.engine.stop()


async def test_r18_b1_an_answer_the_channel_never_takes_ends_the_round_without_the_persona(
    services: Services, clock: ManualClock
) -> None:
    """R18-B1: the network never comes back: a bounded number of tries, then the round ends."""
    rig = build_harness(services, clock, crisis=ScriptedCrisis("不想活"))
    tries = 3 * (MAX_RETRIES + 1)
    rig.channel.results.extend(
        [OutboundResult.failure(OutboundKind.NETWORK, "connect_error")] * tries
    )
    await rig.engine.start()
    try:
        await rig.message("我真的不想活了")
        await run_to_idle(rig.engine, clock)
        assert rig.channel.texts == [] and rig.writer.calls == 0
        assert not rig.channel.results  # every try was made, and no more
        snap = rig.engine.snapshot()
        assert snap.state == "IDLE" and snap.pending == ()
    finally:
        await rig.engine.stop()


async def test_r18_b1_repeated_step_failures_during_a_crisis_never_give_a_persona_answer(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-B1: the end of the round keeps failing; the way out (D-720) is not her short answer.

    The fifth failure in a row gives the round up (D-720 sends one of her short answers first);
    a round that is answering a crisis gets no word of the persona: it is dropped as before.
    """
    rig = build_harness(services, clock, crisis=ScriptedCrisis("不想活"))
    engine = rig.engine

    async def broken(_answered: Any) -> None:
        raise RuntimeError("the round cannot be closed")

    monkeypatch.setattr(engine, "_complete", broken)
    await engine.start()
    try:
        await rig.message("我真的不想活了")
        await run_to_idle(engine, clock)
        assert rig.channel.texts == ANSWER  # the answer, once; no "嗯嗯" after it
        assert rig.writer.calls == 0
        assert "reply_failed" not in rig.alerts.categories
    finally:
        await engine.stop()


def test_r18_b1_the_crisis_note_round_trips_and_old_rows_have_none() -> None:
    """R18-B1: ``RoundData.crisis`` is written and read back; a row of before has none."""
    moment = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    note = CrisisAnswer(
        ("你现在还好吗？",), ("t1", "t2"), reply_id="r1", retries=1, retry_at=moment
    )
    data = RoundData(screened=("t1", "t2"), crisis=note)
    again = RoundData.of(ConversationSnapshot("DECIDING", moment, data=data.to_json()))
    assert again.crisis == note
    old = data.to_json()
    del old["crisis"]
    assert RoundData.of(ConversationSnapshot("DECIDING", moment, data=old)).crisis is None
