"""The stored round: its JSON notes, lookups on ``bot_turns``, stickers and logs (R-ENG-011)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_extras import PICTURES, add_sticker
from tests.support.engine_harness import (
    START,
    Harness,
    build_harness,
    make_draft,
    run_to_idle,
)
from twin.config.runtime import ENGINE_PAUSED_UNTIL, registered_settings
from twin.engine.decision import Decision
from twin.engine.rounds import RoundStore
from twin.engine.roundstate import Outgoing, RoundData, meta_from_json, meta_to_json
from twin.engine.sender import OutBubble
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.services import Services


@pytest.fixture
async def rig(services: Services, clock: ManualClock) -> AsyncIterator[Harness]:
    harness = build_harness(services, clock)
    await harness.engine.start()
    yield harness
    await harness.engine.stop()


# --------------------------------------------------------------------------- the notes


def test_the_notes_of_a_round_survive_the_trip_through_json() -> None:
    meta = ReplyMeta(
        "hybrid",
        thinking=True,
        plan={"reply": True},
        cost_usd=0.002,
        timings_ms={"generate": 5},
        actions=({"step": "quota_merge", "count": 2},),
    )
    outgoing = Outgoing(
        meta,
        (OutBubble("text", "好"), OutBubble("sticker", "[表情包:开心]", "a" * 32)),
        reply_id="r1",
        quote_id="m1",
        quote_text="去看电影",
        extra_actions=({"step": "decided_free", "count": 1},),
    )
    decision = Decision("free", START, START + timedelta(seconds=20), state="free")
    data = RoundData(
        quiet_from=START,
        collect_from=START,
        answering=("a",),
        screened=("a", "b"),
        decision=decision,
        decided_for=2,
        retries=1,
        cancelled=3,
        continuation=True,
        skip_quiet=True,
        outgoing=outgoing,
    )
    from twin.engine.state_store import ConversationSnapshot

    again = RoundData.of(ConversationSnapshot("SENDING", START, data=data.to_json()))
    assert again == data
    assert meta_from_json(meta_to_json(meta)) == meta
    assert outgoing.final_meta().actions == (
        {"step": "quota_merge", "count": 2},
        {"step": "decided_free", "count": 1},
    )
    assert RoundData.of(ConversationSnapshot("IDLE", START)) == RoundData()


def test_a_sticker_bubble_needs_its_sticker() -> None:
    with pytest.raises(ValueError, match="MD5"):
        OutBubble("sticker", "[表情包:开心]")
    assert OutBubble("text", "你好").chars == 2 and OutBubble("sticker", "x", "a" * 32).chars == 0


# --------------------------------------------------------------------------- the lookups


def test_the_lookups_on_the_conversation(services: Services, clock: ManualClock) -> None:
    store = BotTurnStore(services.db, clock)
    rounds = RoundStore(services.db)
    first = store.add_inbound(at=START, kind="text", text="一", external_id="a").record
    second = store.add_inbound(at=START + timedelta(seconds=5), kind="text", text="二").record
    command = store.add_inbound(
        at=START + timedelta(seconds=6), kind="text", text="/帮助", is_command=True
    ).record
    reply = store.add_reply(
        [OutboundBubble("好", START + timedelta(seconds=30))], ReplyMeta("deepseek")
    )
    third = store.add_inbound(at=START + timedelta(seconds=60), kind="text", text="三").record
    items = rounds.inbound_items([third.id, first.id, second.id, command.id, "nobody"])
    assert [i.text for i in items] == ["一", "二", "三"]  # oldest first, no command, no stranger
    assert items[0].id == "a" and items[1].id == second.id and items[0].turn_id == first.id
    assert rounds.inbound_items([]) == []
    assert rounds.answered(first.id) and rounds.answered(second.id) and rounds.answered("nobody")
    assert not rounds.answered(third.id)
    assert rounds.last_answer_at() == START + timedelta(seconds=30)
    assert rounds.inbound_of_reply(reply[0].reply_id or "") == [first.id, second.id]
    assert rounds.inbound_of_reply("missing") == []
    rounds.set_command(third.id, True)
    assert rounds.inbound_items([third.id]) == []
    rounds.set_command(third.id, False)
    assert [i.text for i in rounds.inbound_items([third.id])] == ["三"]


def test_a_second_reply_answers_only_what_came_after_the_first(
    services: Services, clock: ManualClock
) -> None:
    store = BotTurnStore(services.db, clock)
    rounds = RoundStore(services.db)
    store.add_inbound(at=START, kind="text", text="一")
    store.add_reply([OutboundBubble("好", START + timedelta(seconds=30))], ReplyMeta("deepseek"))
    later = store.add_inbound(at=START + timedelta(seconds=90), kind="text", text="二").record
    second = store.add_reply(
        [OutboundBubble("嗯", START + timedelta(seconds=120))], ReplyMeta("deepseek")
    )
    assert rounds.inbound_of_reply(second[0].reply_id or "") == [later.id]
    assert RoundStore(services.db).last_answer_at() == START + timedelta(seconds=120)


def test_nothing_has_been_answered_in_an_empty_conversation(services: Services) -> None:
    assert RoundStore(services.db).last_answer_at() is None


# ----------------------------------------------------------------------------- stickers


async def test_a_sticker_of_the_library_goes_out_as_its_picture_and_is_stored(
    rig: Harness,
) -> None:
    sticker = add_sticker(rig.services, seed=9)
    rig.writer.add(make_draft("哈哈", "[表情包:开心]", stickers={"[表情包:开心]": sticker.md5}))
    await rig.message("今天好开心")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["哈哈"] and rig.channel.images == [PICTURES[sticker.md5]]
    from sqlalchemy import select

    from twin.storage.engine_models import BotTurn

    with rig.services.db.session() as session:
        rows = [
            (r.kind, r.sticker_md5)
            for r in session.scalars(select(BotTurn))
            if r.direction == "out"
        ]
    assert rows == [("text", None), ("sticker", sticker.md5)]
    assert rig.store.recent_stickers(5) == [None, sticker.md5]


async def test_a_sticker_the_channel_refuses_is_dropped_and_the_reply_goes_on(rig: Harness) -> None:
    from twin.channel.base import MediaNotAllowed

    sticker = add_sticker(rig.services, seed=10)
    rig.channel.image_error = MediaNotAllowed("not on the allow list")
    rig.writer.add(make_draft("[表情包:开心]", "哈哈", stickers={"[表情包:开心]": sticker.md5}))
    await rig.message("今天好开心")
    await run_to_idle(rig.engine, rig.clock)
    assert rig.channel.texts == ["哈哈"] and rig.channel.images == []
    from sqlalchemy import select

    from twin.storage.engine_models import BotTurn

    with rig.services.db.session() as session:
        actions = next(
            r.actions
            for r in session.scalars(select(BotTurn))
            if r.direction == "out" and r.bubble_index == 0
        )
    assert {"step": "bubble_skipped", "count": 1} in actions


# --------------------------------------------------------------------------------- logs


async def test_no_log_line_at_info_level_carries_what_was_said(
    rig: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    rig.writer.add(make_draft("秘密回复一", "秘密回复二"))
    await rig.message("秘密的话")
    await run_to_idle(rig.engine, rig.clock)
    logged = "\n".join(
        record.getMessage() + " " + str(getattr(record, "msg", "")) for record in caplog.records
    )
    assert logged and not any(word in logged for word in ("秘密的话", "秘密回复"))


# ----------------------------------------------------------------------------- settings


def test_the_pause_is_a_runtime_setting_with_a_time_or_nothing(services: Services) -> None:
    assert "engine.paused_until" in registered_settings()
    assert services.runtime.get(ENGINE_PAUSED_UNTIL) is None
    until = START + timedelta(hours=2)
    assert services.runtime.set(ENGINE_PAUSED_UNTIL, until)
    assert services.runtime.get(ENGINE_PAUSED_UNTIL) == until
    assert services.runtime.set(ENGINE_PAUSED_UNTIL, None)
    assert services.runtime.get(ENGINE_PAUSED_UNTIL) is None
    from twin.config.runtime import SettingValueError

    with pytest.raises(SettingValueError):
        services.runtime.set(ENGINE_PAUSED_UNTIL, "tomorrow")  # type: ignore[arg-type]
    with pytest.raises(SettingValueError):
        services.runtime.set(ENGINE_PAUSED_UNTIL, __import__("datetime").datetime(2026, 1, 1))
