"""The state machine: transitions, the quiet window, merging, cancelling (R-ENG-001/002/004)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, ok
from tests.support.engine_harness import (
    START,
    Harness,
    ScriptedWriter,
    bubbles_written,
    build_harness,
    make_draft,
    reference_pacing,
    run_to_idle,
    wait_for_state,
)
from tests.support.persona import sticker_png
from tests.support.waiting import wait_until
from twin.channel.base import InboundMessage, MediaRef, MessageKind
from twin.engine.inbound import InboundRenderer
from twin.engine.roundstate import RoundData
from twin.engine.state_store import ConversationSnapshot
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.services import Services
from twin.storage.engine_models import BotTurn
from twin.storage.media import MediaKind


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(services, clock)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


def rows(harness: Harness) -> list[BotTurn]:
    with harness.services.db.session() as session:
        found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
        session.expunge_all()
    return found


def table(harness: Harness) -> list[tuple[str, str, str]]:
    """``(direction, kind, text)`` of every row of ``bot_turns``."""
    return [(row.direction, row.kind, row.text) for row in rows(harness)]


async def waits_until(harness: Harness, moment: datetime) -> None:
    await wait_until(lambda: harness.engine.waiting_until == moment, limit_s=10.0, interval=0.002)


def at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


# ------------------------------------------------------------------ the whole round


async def test_a_message_goes_through_every_state_and_is_answered(
    rig: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    visited: list[str] = []
    original = rig.state.transition

    def record(state: str, **fields: object) -> ConversationSnapshot:
        visited.append(state)
        return original(state, **fields)

    monkeypatch.setattr(rig.state, "transition", record)
    rig.writer.add(make_draft("好呀", "我也想去"))
    await rig.message("周末去看电影吧")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀", "我也想去"]
    assert visited == ["COLLECTING", "DECIDING", "GENERATING", "SENDING", "IDLE"]
    assert [r[:2] for r in table(rig)] == [("in", "text"), ("out", "text"), ("out", "text")]
    final = rig.engine.snapshot()
    assert final.state == "IDLE" and final.pending == () and final.sent == ()


async def test_each_transition_is_written_before_the_next_step(rig: Harness) -> None:
    rig.writer.gate = asyncio.Event()
    rig.writer.add(make_draft("好呀"))
    await rig.message("在吗")
    await waits_until(rig, at(15))
    collecting = rig.engine.snapshot()
    assert collecting.state == "COLLECTING" and len(collecting.pending) == 1
    await rig.clock.advance(15)
    await wait_for_state(rig.engine, "DECIDING")
    deciding = rig.engine.snapshot()
    assert deciding.planned_send_at is not None
    decision = RoundData.of(deciding).decision
    assert decision is not None and decision.mode == "free"
    assert deciding.planned_send_at == decision.send_at
    await run_to_idle(rig.engine, rig.clock, until=rig.writer.started.is_set)
    generating = rig.engine.snapshot()
    assert (
        generating.state == "GENERATING"
        and RoundData.of(generating).answering == generating.pending
    )
    rig.writer.gate.set()
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好呀"]


async def test_the_first_bubble_waits_for_her_delay_and_the_next_for_her_pause(
    rig: Harness,
) -> None:
    rig.writer.add(make_draft("好呀", "我也想去"))
    await rig.message("周末去看电影吧")
    await run_to_idle(rig.engine, rig.clock)
    first, second = (o.at for o in rig.channel.out if o.kind == "text")
    # her latency (20 s) + reading seven characters (0.7 s), counted from the message; then she
    # types "好呀" (2 characters at 0.5 s) before it appears
    assert first - START == timedelta(seconds=20.7 + 1.0)
    # the second bubble: her pause between messages (4 s) + typing four characters (2 s)
    assert second - first == timedelta(seconds=6)
    typing = [o.active for o in rig.channel.out if o.kind == "typing"]
    assert typing[0] is True and len(typing) == 2


async def test_typing_is_not_shown_where_the_channel_cannot_show_it(
    services: Services, clock: ManualClock
) -> None:
    from tests.support.engine_harness import ScriptedChannel

    channel = ScriptedChannel(clock, supports_typing=False)
    harness = build_harness(services, clock, channel=channel)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀"))
        await harness.message("在吗")
        await run_to_idle(harness.engine, clock)
        assert channel.texts == ["好呀"] and not [o for o in channel.out if o.kind == "typing"]
    finally:
        await harness.engine.stop()


# --------------------------------------------------------------------- COLLECTING


async def test_a_new_message_restarts_the_quiet_window(rig: Harness) -> None:
    await rig.message("第一条")
    await waits_until(rig, at(15))
    await rig.clock.advance(10)
    await rig.message("第二条")
    await waits_until(rig, at(25))  # fifteen seconds after the newest message
    assert rig.engine.snapshot().state == "COLLECTING" and len(rig.engine.snapshot().pending) == 2
    rig.writer.add(make_draft("看到了"))
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.contexts[0].user_text == "第一条\n第二条"
    assert rig.channel.texts == ["看到了"]


async def test_collecting_never_lasts_longer_than_the_maximum_wait(
    services: Services, clock: ManualClock
) -> None:
    services.settings.engine.max_wait_s = 40
    harness = build_harness(services, clock)
    await harness.engine.start()
    try:
        await harness.message("0")
        for second, window_end in ((10, 25), (20, 35)):
            await clock.advance(10)
            await harness.message(str(second))
            await waits_until(harness, at(window_end))
        await clock.advance(10)  # t = 30: the quiet window would end at 45
        await harness.message("30")
        await waits_until(harness, at(40))  # but 40 s after the first message is the limit
        await clock.advance(10)
        await wait_for_state(harness.engine, "DECIDING")
        assert len(harness.engine.snapshot().pending) == 4
    finally:
        await harness.engine.stop()


async def test_the_quiet_window_follows_the_users_pauses_when_adaptive(
    services: Services, clock: ManualClock
) -> None:
    services.settings.engine.quiet_window_adaptive = True
    harness = build_harness(services, clock, pacing=reference_pacing(user_gap_s=33.0))
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await waits_until(harness, at(33))  # the user's 75th percentile pause, not the 15 s default
        info = await harness.engine.quiet_window()
        assert (info.configured_s, info.suggested_s, info.effective_s) == (15.0, 33.0, 33.0)
    finally:
        await harness.engine.stop()


async def test_the_adaptive_window_has_a_ceiling_and_is_only_a_suggestion_by_default(
    services: Services, clock: ManualClock
) -> None:
    harness = build_harness(services, clock, pacing=reference_pacing(user_gap_s=300.0))
    info = await harness.engine.quiet_window()
    assert (info.configured_s, info.suggested_s, info.adaptive, info.effective_s) == (
        15.0,
        45.0,
        False,
        15.0,
    )
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await waits_until(harness, at(15))  # the setting is off: the configured window stands
    finally:
        await harness.engine.stop()


# ---------------------------------------------------------------------- DECIDING


async def test_a_message_while_she_waits_joins_the_round_without_a_new_draw(rig: Harness) -> None:
    await rig.message("周末去看电影吧")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    planned = rig.engine.snapshot().planned_send_at
    assert planned == at(20.7)  # the message is 0.7 s worth of reading
    await rig.clock.advance(1)  # t = 16 s: 78 % of the 20.7 s wait
    await rig.message("还有一件事")
    await wait_until(lambda: len(rig.engine.snapshot().pending) == 2)
    await wait_until(lambda: RoundData.of(rig.engine.snapshot()).decided_for == 2)
    assert rig.engine.snapshot().planned_send_at == planned  # before 80 %: the drawn time stands
    rig.writer.add(make_draft("好呀"))
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.contexts[0].user_text == "周末去看电影吧\n还有一件事"
    assert rig.channel.texts == ["好呀"]


async def test_after_eighty_percent_of_the_wait_a_short_pause_is_added(rig: Harness) -> None:
    await rig.message("周末去看电影吧")
    await run_to_idle(
        rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "DECIDING"
    )
    await rig.clock.advance(3)  # t = 18 s: past 80 % of 20.7 s
    await rig.message("等一下")
    await wait_until(lambda: RoundData.of(rig.engine.snapshot()).decided_for == 2)
    extended = rig.engine.snapshot().planned_send_at
    assert extended is not None
    # reading three characters (0.5 s) plus her pause between messages (4 s) after the new one
    assert extended == at(18 + 0.5 + 4)
    assert extended > at(20.7)


# --------------------------------------------------------------------- GENERATING


async def test_a_message_during_generation_cancels_it_and_joins_the_round(rig: Harness) -> None:
    rig.writer.gate = asyncio.Event()
    await rig.message("周末去看电影吧")
    await run_to_idle(rig.engine, rig.clock, until=rig.writer.started.is_set)
    assert rig.engine.snapshot().state == "GENERATING"
    await rig.message("算了先不说这个")
    await wait_until(lambda: rig.writer.cancelled == 1)
    await wait_for_state(rig.engine, "COLLECTING")
    cancelled = rig.engine.snapshot()
    assert len(cancelled.pending) == 2 and RoundData.of(cancelled).cancelled == 1
    assert rig.channel.texts == []  # nothing was said
    rig.writer.gate.set()
    rig.writer.add(make_draft("好呀"))
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.calls == 2
    assert rig.writer.contexts[1].user_text == "周末去看电影吧\n算了先不说这个"
    assert rig.channel.texts == ["好呀"]
    first_bubble = next(row for row in rows(rig) if row.direction == "out")
    assert {"step": "generation_cancelled", "count": 1} in (first_bubble.actions or [])


async def test_a_reply_that_is_ready_just_as_the_user_writes_is_thrown_away(
    services: Services, clock: ManualClock
) -> None:
    holder: dict[str, Harness] = {}

    class WritesThenUserWrites(ScriptedWriter):
        async def run(self, context, data):  # type: ignore[no-untyped-def]
            draft = await super().run(context, data)
            if self.calls == 1:  # the user's message lands between the draft and the check
                mine = holder["h"]
                await mine.engine.handle_message(mine.channel.push("还有"))
            return draft

    writer = WritesThenUserWrites(make_draft("旧的回复"), make_draft("新的回复"))
    harness = build_harness(services, clock, writer=writer)
    holder["h"] = harness
    await harness.engine.start()
    try:
        await harness.message("在吗")
        await run_to_idle(harness.engine, clock)
        assert harness.channel.texts == ["新的回复"]
        assert writer.contexts[-1].user_text == "在吗\n还有"
    finally:
        await harness.engine.stop()


async def test_the_thinking_mode_and_the_backend_of_the_settings_go_into_the_context(
    rig: Harness,
) -> None:
    from twin.config.runtime import BACKEND_ACTIVE, THINKING_CHAT

    rig.services.runtime.set(THINKING_CHAT, "on")
    rig.services.runtime.set(BACKEND_ACTIVE, "style")
    rig.writer.add(make_draft("好"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    context = rig.writer.contexts[0]
    assert (context.thinking_mode, context.backend) == ("on", "style")


async def test_the_platform_quota_limits_the_bubbles_of_the_reply(rig: Harness) -> None:
    rig.writer.add(make_draft("好"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.contexts[0].limits.max_bubbles == 6  # 8 left, 2 kept for what she starts
    rig.channel.quota = 3
    rig.writer.add(make_draft("好"))
    await rig.message("还在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.writer.contexts[1].limits.max_bubbles == 1  # never fewer than one
    assert rig.writer.contexts[1].limits.supports_quote is False


async def test_the_recent_stickers_and_what_was_already_said_reach_the_context(
    rig: Harness,
) -> None:
    rig.writer.add(make_draft("一", "二"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    rig.writer.add(make_draft("三"))
    await rig.message("再说一遍")
    await run_to_idle(rig.engine, rig.clock)
    second = rig.writer.contexts[1]
    assert second.recent_stickers == (None, None)
    assert second.already_said == ()
    assert [turn.text for turn in second.history] == ["在吗", "一\n二"]


# ------------------------------------------------------------------------ SENDING


async def test_a_message_while_she_sends_keeps_what_is_out_and_continues(rig: Harness) -> None:
    rig.writer.add(make_draft("第一条", "第二条", "第三条"), make_draft("好的接着说"))
    await rig.message("给我讲个故事")
    await run_to_idle(rig.engine, rig.clock, until=bubbles_written(rig.engine, 1))
    await rig.message("等等")  # she is about to type the second bubble
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["第一条", "好的接着说"]  # the unsent two were dropped
    second = rig.writer.contexts[1]
    assert second.already_said == ("第一条",)
    assert [item.text for item in second.inbound] == ["等等"]  # the first message is answered
    assert [turn.text for turn in second.history][-2:] == ["给我讲个故事", "第一条"]
    interrupted = next(row for row in rows(rig) if row.direction == "out" and row.bubble_index == 0)
    assert {"step": "send_interrupted", "count": 2} in (interrupted.actions or [])
    final = rig.engine.snapshot()
    assert final.state == "IDLE" and final.sent == ()


async def test_a_message_that_comes_just_before_the_first_bubble_drops_the_whole_draft(
    rig: Harness,
) -> None:
    rig.writer.add(make_draft("旧的"), make_draft("新的"))
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock, until=lambda: rig.engine.snapshot().state == "SENDING")
    await rig.message("等等")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["新的"]
    assert rig.writer.contexts[1].user_text == "在吗\n等等"  # nothing was answered yet


async def test_silence_is_stored_as_a_row_without_text(rig: Harness) -> None:
    rig.writer.add(make_draft(no_reply=True))
    await rig.message("嗯嗯")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == []
    assert [(r[0], r[1]) for r in table(rig)] == [("in", "text"), ("out", "no_reply")]
    assert rig.engine.snapshot().state == "IDLE"


async def test_the_numbers_of_the_reply_are_stored_on_its_first_bubble(rig: Harness) -> None:
    rig.writer.add(make_draft("好呀", "我也想去", cost=0.0042, thinking=True))
    await rig.message("周末去看电影吧")
    await run_to_idle(rig.engine, rig.clock)
    first = next(row for row in rows(rig) if row.direction == "out")
    assert first.backend == "deepseek" and first.thinking is True and first.cost_usd == 0.0042
    assert first.timings["generate"] == 20 and first.timings["delay"] == 20_700
    assert {"step": "decided_free", "count": 1} in first.actions
    second = [row for row in rows(rig) if row.direction == "out"][1]
    assert second.backend is None and second.reply_id == first.reply_id


async def test_a_quote_goes_with_the_first_text_bubble_where_the_channel_can_quote(
    services: Services, clock: ManualClock
) -> None:
    from tests.support.engine_harness import ScriptedChannel

    channel = ScriptedChannel(clock, supports_quote=True)
    harness = build_harness(services, clock, channel=channel)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀", "我也想去", quote="去看电影"))
        message = channel.push("周末去看电影吧")
        await harness.engine.handle_message(message)
        await run_to_idle(harness.engine, clock)
        quotes = [o.quote for o in channel.out if o.kind == "text"]
        assert quotes[0] is not None and quotes[0].text == "去看电影"
        assert quotes[0].message_id == message.id and quotes[1] is None
        assert harness.writer.contexts[0].limits.supports_quote is True
    finally:
        await harness.engine.stop()


# ----------------------------------------------------------------------- redelivery


async def test_a_message_the_channel_hands_over_twice_is_answered_once(rig: Harness) -> None:
    message = rig.channel.push("在吗", message_id="same")
    await rig.engine.handle_message(message)
    await rig.engine.handle_message(message)
    assert len(rig.engine.snapshot().pending) == 1
    await run_to_idle(rig.engine, rig.clock)
    await rig.engine.handle_message(message)  # after the reply, once more
    assert rig.engine.snapshot().pending == ()
    assert len([r for r in table(rig) if r[0] == "in"]) == 1
    assert rig.writer.calls == 1


async def test_a_message_that_was_stored_but_never_queued_is_picked_up_again(rig: Harness) -> None:
    stored = rig.store.add_inbound(
        at=rig.clock.now_utc(), kind="text", text="在吗", external_id="x1"
    )
    assert stored.created and rig.engine.snapshot().pending == ()  # a crash before it was queued
    await rig.engine.handle_message(rig.channel.push("在吗", message_id="x1"))
    assert rig.engine.snapshot().pending == (stored.record.id,)
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好的"]


# --------------------------------------------------------------- a failing step


async def test_a_step_that_fails_is_alerted_and_tried_again(
    rig: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = rig.engine._rounds.inbound_items
    calls = {"n": 0}

    def flaky(ids: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the database hiccupped")
        return original(ids)  # type: ignore[arg-type]

    monkeypatch.setattr(rig.engine._rounds, "inbound_items", flaky)
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["好的"]
    assert "engine_error" in rig.alerts.categories


async def test_a_step_that_keeps_failing_gives_up_the_round_with_a_critical_alert(
    rig: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(ids: object) -> object:
        raise RuntimeError("always")

    monkeypatch.setattr(rig.engine._rounds, "inbound_items", broken)
    await rig.message("在吗")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.engine.snapshot().state == "IDLE" and rig.channel.texts == []
    assert rig.alerts.categories.count("engine_error") >= 5


async def test_a_picture_reaches_the_prompt_as_its_description(
    services: Services, clock: ManualClock
) -> None:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    runtime = build_llm_runtime(services)
    data = sticker_png(3)
    stored = services.media.put(data, MediaKind.IMAGE)
    picture = InboundMessage(
        "pic1",
        clock.now_utc(),
        MessageKind.IMAGE,
        media_ref=MediaRef(stored.sha256, MediaKind.IMAGE, len(data), "image/png", None),
    )
    renderer = InboundRenderer(services, client=runtime.client)
    rig = build_harness(services, clock, renderer=renderer)
    with respx.mock(assert_all_called=False) as api:
        api.post(API).mock(return_value=ok(content="一只趴在窗台上的橘猫"))
        await rig.engine.start()
        try:
            await rig.engine.handle_message(picture)
            await run_to_idle(rig.engine, rig.clock)
        finally:
            await rig.engine.stop()
            await renderer.aclose()
            await runtime.client.aclose()
    described = "[图片：一只趴在窗台上的橘猫]"
    assert rig.writer.contexts[0].user_text == described
    assert table(rig)[0] == ("in", "image", described)  # the stored line is the stable one
