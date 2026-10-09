"""The bot's conversation on disk: bot_turns, its reader, the state row, the feedback."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from twin.engine.feedback import FeedbackStore
from twin.engine.history import HistoryLoader
from twin.engine.state_store import ConversationStateStore
from twin.engine.turns import (
    BotTurnMessages,
    BotTurnStore,
    OutboundBubble,
    ReplyMeta,
    install_bot_turn_reader,
)
from twin.memory.recent import HistoryWindow, bot_turn_reader
from twin.schedule.summary import summary_scopes
from twin.services import Services
from twin.storage.engine_models import BotTurn

START = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)


@pytest.fixture
def store(services: Services) -> BotTurnStore:
    return BotTurnStore(services.db, services.clock)


def at(minutes: float) -> datetime:
    return START + timedelta(minutes=minutes)


def test_an_inbound_message_is_stored_once_per_channel_id(store: BotTurnStore) -> None:
    first = store.add_inbound(at=at(0), kind="text", text="在吗", external_id="m-1")
    again = store.add_inbound(at=at(5), kind="text", text="在吗", external_id="m-1")
    assert first.created and not again.created
    assert first.record.inbound and first.record.counts_as_conversation
    assert again.record.id == first.record.id and again.record.at == at(0)
    other = store.add_inbound(at=at(1), kind="text", text="哈哈")  # no channel id: always new
    assert other.created and other.record.external_id is None
    assert store.count(direction="in") == 2
    with pytest.raises(ValueError, match="kind"):
        store.add_inbound(at=at(2), kind="no_reply", text="")


def test_the_text_is_sealed_on_disk(services: Services, store: BotTurnStore) -> None:
    secret = "今晚七点在小区东门见"
    store.add_inbound(at=at(0), kind="text", text=secret, media={"quote": {"text": secret}})
    store.add_reply(
        [OutboundBubble(secret, at(1))],
        ReplyMeta("deepseek", plan={"idea": secret}, actions=({"step": "x"},)),
    )
    connection = sqlite3.connect(services.paths.db_path)
    try:
        dump = b"".join(
            bytes(value) if isinstance(value, bytes | bytearray) else str(value).encode()
            for row in connection.execute("SELECT * FROM bot_turns")
            for value in row
        )
    finally:
        connection.close()
    assert secret.encode() not in dump and "小区".encode() not in dump
    with services.db.session() as session:
        rows = list(session.scalars(select(BotTurn).order_by(BotTurn.at)))
        assert [row.text for row in rows] == [secret, secret]
        assert rows[1].plan == {"idea": secret}


def test_a_reply_is_a_group_of_bubbles_with_its_numbers_on_the_first(store: BotTurnStore) -> None:
    meta = ReplyMeta(
        "deepseek",
        thinking=False,
        plan={"reply": True},
        cost_usd=0.0012,
        timings_ms={"generate": 800, "post": 4},
        actions=({"step": "dedupe", "count": 1},),
    )
    rows = store.add_reply(
        [
            OutboundBubble("好呀", at(1)),
            OutboundBubble("[表情包:开心]", at(1.1), "sticker", "b" * 32),
            OutboundBubble("等下", at(1.2)),
        ],
        meta,
    )
    assert [r.bubble_index for r in rows] == [0, 1, 2]
    assert {r.reply_id for r in rows} == {rows[0].id}
    assert rows[0].cost_usd == 0.0012 and rows[0].actions[0]["step"] == "dedupe"
    assert rows[1].backend is None and rows[1].cost_usd is None  # stored once per reply
    assert rows[1].sticker_md5 == "b" * 32 and rows[1].kind == "sticker"
    assert [r.text for r in store.reply(rows[0].id)] == ["好呀", "[表情包:开心]", "等下"]
    with pytest.raises(ValueError, match="at least one"):
        store.add_reply([], meta)


def test_bubbles_can_be_stored_one_at_a_time_as_they_are_sent(store: BotTurnStore) -> None:
    first = store.add_bubble(OutboundBubble("嗯嗯", at(1)), meta=ReplyMeta("deepseek"))
    second = store.add_bubble(OutboundBubble("好的", at(2)), reply_id=first.reply_id)
    assert (first.bubble_index, second.bubble_index) == (0, 1)
    assert store.update_meta(first.id, ReplyMeta("deepseek", cost_usd=0.5))
    assert store.reply(first.id)[0].cost_usd == 0.5
    assert not store.update_meta("no-such-reply", ReplyMeta("deepseek"))
    with pytest.raises(ValueError, match="sticker"):
        OutboundBubble("[表情包:x]", at(0), "sticker")
    with pytest.raises(ValueError, match="text or a sticker"):
        OutboundBubble("x", at(0), "image")
    with pytest.raises(ValueError, match="backend"):
        ReplyMeta("telepathy")


def test_silence_is_one_row_without_text(store: BotTurnStore) -> None:
    row = store.add_no_reply(at(1), ReplyMeta("deepseek", actions=({"step": "no_reply"},)))
    assert row.kind == "no_reply" and row.text == "" and row.reply_id == row.id
    assert store.latest_reply() == [row]


def test_the_reader_gives_the_memory_the_conversation_only(
    services: Services, store: BotTurnStore
) -> None:
    store.add_inbound(at=at(0), kind="text", text="早", external_id="1")
    kept = store.add_reply([OutboundBubble("早呀", at(1))], ReplyMeta("deepseek"))
    store.add_inbound(at=at(2), kind="text", text="/状态", external_id="2", is_command=True)
    store.add_reply([OutboundBubble("⚙️ 一切正常", at(3))], ReplyMeta("command"), is_command=True)
    store.add_no_reply(at(4), ReplyMeta("deepseek"))
    thrown = store.add_reply([OutboundBubble("算了吧", at(5))], ReplyMeta("deepseek"))
    store.reject_reply(thrown[0].reply_id or "")
    store.add_inbound(at=at(6), kind="image", text="[图片：一只猫]", external_id="3")

    install_bot_turn_reader()  # what importing twin.engine.turns has done in a fresh process
    reader = bot_turn_reader(services)
    assert isinstance(reader, BotTurnMessages)
    messages = reader.messages_since(None)
    assert [(m.role, m.text) for m in messages] == [
        ("user", "早"),
        ("bot", "早呀"),
        ("user", "[图片：一只猫]"),
    ]
    assert messages[1].id == kept[0].id
    assert [m.text for m in reader.messages_between(at(0), at(2))] == ["早", "早呀"]
    assert [m.text for m in reader.messages_since(at(1))] == ["早呀", "[图片：一只猫]"]
    assert [m.text for m in reader.messages_since(None, 2)] == ["早呀", "[图片：一只猫]"]


def test_a_rejected_reply_leaves_the_conversation_but_stays_on_disk(store: BotTurnStore) -> None:
    store.add_reply([OutboundBubble("第一版", at(1))], ReplyMeta("deepseek"))
    second = store.add_reply(
        [OutboundBubble("第二版", at(2)), OutboundBubble("嗯", at(2.1))], ReplyMeta("deepseek")
    )
    assert [r.text for r in store.latest_reply()] == ["第二版", "嗯"]
    assert store.reject_reply(second[0].reply_id or "", at(3)) == 2
    assert store.reject_reply(second[0].reply_id or "", at(3)) == 0  # already thrown away
    assert [r.text for r in store.latest_reply()] == ["第一版"]
    thrown = store.latest_reply(include_rejected=True)
    assert [r.text for r in thrown] == ["第二版", "嗯"] and thrown[0].rejected_at == at(3)
    assert not thrown[0].counts_as_conversation
    assert store.last_message_at("out") == at(1)


def test_the_latest_reply_ignores_command_replies(store: BotTurnStore) -> None:
    store.add_reply([OutboundBubble("真正的回复", at(1))], ReplyMeta("deepseek"))
    store.add_reply([OutboundBubble("⚙️ 帮助", at(2))], ReplyMeta("command"), is_command=True)
    assert [r.text for r in store.latest_reply()] == ["真正的回复"]
    assert BotTurnStore.latest_reply.__name__ == "latest_reply"


def test_recent_stickers_list_the_last_bubbles_oldest_first(store: BotTurnStore) -> None:
    store.add_reply(
        [
            OutboundBubble("a", at(1)),
            OutboundBubble("[表情包:开心]", at(1.1), "sticker", "c" * 32),
            OutboundBubble("b", at(1.2)),
        ],
        ReplyMeta("deepseek"),
    )
    assert store.recent_stickers(10) == [None, "c" * 32, None]
    assert store.recent_stickers(2) == ["c" * 32, None]
    assert store.recent_stickers(0) == []


def test_the_last_message_times_follow_the_conversation(store: BotTurnStore) -> None:
    assert store.last_message_at() is None
    store.add_inbound(at=at(0), kind="text", text="a")
    store.add_reply([OutboundBubble("b", at(2))], ReplyMeta("deepseek"))
    assert store.last_message_at() == at(2) and store.last_message_at("in") == at(0)
    assert store.get("nope") is None and store.by_external("in", "x") is None


# ----------------------------------------------------------------- conversation state


def test_a_new_conversation_is_idle(services: Services) -> None:
    state = ConversationStateStore(services.db, services.clock)
    snapshot = state.load()
    assert snapshot.idle and snapshot.pending == () and snapshot.window_start_at is None
    assert snapshot.state_since == services.clock.now_utc()


def test_transitions_are_written_and_survive_a_restart(
    services: Services, clock: ManualClock
) -> None:
    state = ConversationStateStore(services.db, services.clock)
    state.transition("COLLECTING", pending=["a", "b"], last_inbound_at=at(0), round_id="r1")
    clock.tick(30)
    state.transition(
        "SENDING",
        sent=[{"text": "好呀", "at": at(1).isoformat()}],
        planned_send_at=at(2),
        data={"woke": True},
    )
    restarted = ConversationStateStore(services.db, services.clock).load()
    assert restarted.state == "SENDING" and restarted.pending == ("a", "b")
    assert restarted.sent == ({"text": "好呀", "at": at(1).isoformat()},)
    assert restarted.planned_send_at == at(2) and restarted.data == {"woke": True}
    assert restarted.state_since == services.clock.now_utc()
    assert restarted.last_inbound_at == at(0) and restarted.round_id == "r1"


def test_update_changes_only_the_named_fields(services: Services) -> None:
    state = ConversationStateStore(services.db, services.clock)
    state.transition("DECIDING", pending=["x"], planned_send_at=at(3))
    state.update(last_outbound_at=at(4), planned_send_at=None)
    snapshot = state.load()
    assert snapshot.state == "DECIDING" and snapshot.pending == ("x",)
    assert snapshot.planned_send_at is None and snapshot.last_outbound_at == at(4)
    with pytest.raises(ValueError, match="not fields"):
        state.update(mood="happy")
    with pytest.raises(ValueError, match="unknown conversation state"):
        state.transition("DREAMING")


def test_reset_goes_idle_and_keeps_the_window_start(services: Services) -> None:
    state = ConversationStateStore(services.db, services.clock)
    state.set_window_start(at(-60))
    state.transition("GENERATING", pending=["x"], sent=[{"text": "a"}], round_id="r")
    snapshot = state.reset()
    assert snapshot.idle and snapshot.pending == () and snapshot.sent == ()
    assert snapshot.round_id is None and snapshot.window_start_at == at(-60)


def test_naive_times_are_refused(services: Services) -> None:
    state = ConversationStateStore(services.db, services.clock)
    with pytest.raises(ValueError, match="naive"):
        state.update(planned_send_at=datetime(2026, 1, 1, 12, 0))  # noqa: DTZ001


# ------------------------------------------------------------------- history window


def fill(store: BotTurnStore, turns: int, *, first: int = 0) -> list[str]:
    """Turns ``first`` ... ``first + turns - 1`` of an alternating conversation (user first).

    Turn ``n`` is two minutes after turn ``n - 1``; returns the ids of the user messages.
    """
    user_ids: list[str] = []
    for number in range(first, first + turns):
        if number % 2 == 0:
            added = store.add_inbound(at=at(number * 2), kind="text", text=f"问{number}")
            user_ids.append(added.record.id)
        else:
            store.add_reply([OutboundBubble(f"答{number}", at(number * 2))], ReplyMeta("deepseek"))
    return user_ids


def test_the_window_start_is_stored_and_moves_in_one_step(services: Services) -> None:
    store = BotTurnStore(services.db, services.clock)
    state = ConversationStateStore(services.db, services.clock)
    reader = BotTurnMessages(services.db)
    loader = HistoryLoader(reader, HistoryWindow(30, 40), state)
    fill(store, 31)
    first = loader.load()
    assert len(first.turns) == 30 and first.turns[0].text == "答1"  # the newest 30 turns
    assert state.load().window_start_at == first.turns[0].at
    start = state.load().window_start_at

    fill(store, 9, first=31)  # 40 turns in the window: the start stays
    grown = loader.load()
    assert len(grown.turns) == 39 and not grown.shifted and state.load().window_start_at == start
    assert grown.turns[0] == first.turns[0]

    fill(store, 3, first=40)  # now more than 40: one step of ten turns
    moved = loader.load()
    assert moved.shifted and len(moved.turns) == 32
    assert state.load().window_start_at == moved.turns[0].at != start
    again = HistoryLoader(
        reader, HistoryWindow(30, 40), ConversationStateStore(services.db, services.clock)
    )
    assert again.load().turns == moved.turns  # a restart shows the same window


def test_the_messages_of_the_current_round_are_not_history(services: Services) -> None:
    store = BotTurnStore(services.db, services.clock)
    state = ConversationStateStore(services.db, services.clock)
    loader = HistoryLoader(BotTurnMessages(services.db), HistoryWindow(30, 40), state)
    fill(store, 4)
    current = store.add_inbound(at=at(20), kind="text", text="新的一句").record
    result = loader.load(exclude_ids={current.id})
    assert [t.text for t in result.turns][-1] == "答3"
    assert all("新的一句" not in t.text for t in result.turns)
    assert "新的一句" in [t.text for t in loader.load().turns][-1]


def test_an_empty_conversation_has_no_window_start(services: Services) -> None:
    state = ConversationStateStore(services.db, services.clock)
    loader = HistoryLoader(BotTurnMessages(services.db), HistoryWindow(30, 40), state)
    result = loader.load()
    assert result.turns == () and result.start_at is None and state.load().window_start_at is None
    assert loader.policy.maximum == 40


# ------------------------------------------------------------------------- feedback


def test_feedback_is_recorded_and_consumed_once(services: Services, store: BotTurnStore) -> None:
    rows = store.add_reply([OutboundBubble("不像她", at(1))], ReplyMeta("deepseek"))
    feedback = FeedbackStore(services.db, services.clock)
    redo = feedback.add("redo", rows[0].reply_id or "", bot_turn_id=rows[0].id)
    liked = feedback.add(
        "not_like", rows[0].reply_id or "", bot_turn_id=rows[0].id, correction="  那你说嘛  "
    )
    assert redo.correction is None and liked.correction == "那你说嘛"
    assert [f.id for f in feedback.for_reply(rows[0].reply_id or "")] == [redo.id, liked.id]
    assert [f.type for f in feedback.unprocessed()] == ["redo", "not_like"]
    assert [f.id for f in feedback.unprocessed("not_like")] == [liked.id]
    assert feedback.mark_processed([redo.id, "nope"]) == 1
    assert feedback.mark_processed([redo.id]) == 0 and feedback.mark_processed([]) == 0
    assert [f.id for f in feedback.unprocessed()] == [liked.id]
    with pytest.raises(ValueError, match="feedback type"):
        feedback.add("praise", "r")


def test_importing_the_engine_registers_the_reader_in_a_fresh_process() -> None:
    """The job worker only imports the handler modules; that must be enough (R-MEM-002)."""
    code = (
        "from twin.ops.jobs import load_handlers; load_handlers(['twin.engine.turns']);"
        "from twin.memory.recent import _factory; print(_factory is not None)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert done.stdout.strip() == "True"


def test_the_bots_daily_summary_is_queued_only_for_a_day_with_a_conversation(
    services: Services, store: BotTurnStore
) -> None:
    """R-MEM-002: with ``bot_turns`` behind the reader, an empty day queues no summary."""
    install_bot_turn_reader()
    assert summary_scopes(services, date(2026, 10, 8)) == ()
    store.add_inbound(at=datetime(2026, 10, 8, 20, 30, tzinfo=UTC), kind="text", text="晚上好")
    assert summary_scopes(services, date(2026, 10, 8)) == ("bot",)
    assert summary_scopes(services, date(2026, 10, 7)) == ()
    store.add_inbound(
        at=datetime(2026, 10, 9, 1, 0, tzinfo=UTC), kind="text", text="/状态", is_command=True
    )
    assert summary_scopes(services, date(2026, 10, 9)) == ()  # a command is not conversation
