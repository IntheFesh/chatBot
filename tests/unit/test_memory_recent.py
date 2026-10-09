"""The bot's recent turns: merged blocks in a batched window (R-MEM-001, R-LLM-010)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.support.bot_turns import DEFAULT_START, ListBotTurnReader, conversation
from twin.memory.recent import (
    BotMessage,
    HistoryWindow,
    RecentTurns,
    Turn,
    bot_turn_reader,
    merge_turns,
    register_bot_turn_reader,
)
from twin.services import Services

START = DEFAULT_START


def test_consecutive_messages_of_one_side_are_one_turn() -> None:
    reader = ListBotTurnReader()
    for role, text, minutes in [
        ("user", "在吗", 0),
        ("user", "我有件事", 0),
        ("bot", "怎么啦", 1),
        ("user", "考试通过了", 2),
        ("bot", "太好了", 3),
        ("bot", "请你吃饭", 3),
    ]:
        reader.add(role, text, START + timedelta(minutes=minutes))  # type: ignore[arg-type]
    turns = merge_turns(reader.messages_since(None))
    assert [(t.role, t.text) for t in turns] == [
        ("user", "在吗\n我有件事"),
        ("bot", "怎么啦"),
        ("user", "考试通过了"),
        ("bot", "太好了\n请你吃饭"),
    ]
    first = turns[0]
    assert first.id == "b1" and first.message_ids == ("b1", "b2") and first.at <= first.last_at
    assert merge_turns([]) == []


def test_a_new_conversation_shows_its_newest_thirty_turns() -> None:
    window = HistoryWindow()
    turns = merge_turns(conversation(36).messages_since(None))
    result = window.advance(turns, None)
    assert len(result.turns) == 30 and result.turns[0].text == "第7轮第1句"
    assert result.start_at == result.turns[0].at and result.shifted is False
    short = window.advance(turns[:12], None)
    assert len(short.turns) == 12 and short.turns[0] is turns[0]
    assert window.advance([], None).turns == () and window.advance([], None).start_at is None


def test_the_start_stays_put_until_more_than_forty_turns_and_then_moves_ten_at_once() -> None:
    window = HistoryWindow()
    turns = merge_turns(conversation(60).messages_since(None))
    start = turns[0].at
    for count in range(1, 41):  # up to 40 turns: nothing moves, every request extends the last
        result = window.advance(turns[:count], start)
        assert result.start_at == start and result.shifted is False and len(result.turns) == count
    moved = window.advance(turns[:41], start)  # the 41st turn: back to 31 in one step
    assert moved.shifted is True and len(moved.turns) == 31 and moved.start_at == turns[10].at
    for count in range(42, 51):  # from there it grows again to 40 without moving
        result = window.advance(turns[:count], moved.start_at)
        assert result.start_at == moved.start_at and result.shifted is False
    again = window.advance(turns[:51], moved.start_at)
    assert again.shifted is True and again.start_at == turns[20].at and len(again.turns) == 31


def test_consecutive_requests_share_their_prefix_between_two_moves() -> None:
    """The point of the batching (R-LLM-010): the prompt grows at the end only."""
    window = HistoryWindow()
    turns = merge_turns(conversation(60).messages_since(None))
    start: datetime | None = turns[0].at
    previous: tuple[Turn, ...] = ()
    shifts = 0
    for count in range(1, 61):
        result = window.advance(turns[:count], start)
        start = result.start_at
        if result.shifted:
            shifts += 1
        else:
            assert result.turns[: len(previous)] == previous
        previous = result.turns
    assert shifts == 2  # 60 turns: moved when the 41st and the 51st arrived


def test_a_large_jump_moves_the_start_as_often_as_it_must() -> None:
    window = HistoryWindow(minimum=30, maximum=40)
    turns = merge_turns(conversation(100).messages_since(None))
    result = window.advance(turns, turns[0].at)
    assert len(result.turns) <= 40 and result.shifted and len(turns) - len(result.turns) % 10 >= 0
    assert (len(turns) - len(result.turns)) % window.step == 0


def test_a_stored_start_that_no_longer_exists_falls_back_to_the_newest_turns() -> None:
    window = HistoryWindow()
    turns = merge_turns(conversation(50).messages_since(None))
    result = window.advance(turns, START - timedelta(days=1) + timedelta(days=400))
    assert len(result.turns) == 30


def test_the_window_needs_a_minimum_below_the_maximum() -> None:
    with pytest.raises(ValueError, match="minimum"):
        HistoryWindow(minimum=40, maximum=40)
    with pytest.raises(ValueError, match="minimum"):
        HistoryWindow(minimum=0, maximum=10)
    assert HistoryWindow(10, 25).step == 15


def test_recent_turns_reads_through_the_reader_and_returns_the_start_to_store() -> None:
    reader = conversation(45, per_turn=2)
    recent = RecentTurns(reader)
    first = recent.window(None)
    assert len(first.turns) == 30 and first.turns[-1].message_ids == ("b89", "b90")
    stored = first.start_at
    reader.add("bot", "新的一句", START + timedelta(days=1))
    second = recent.window(stored)
    assert second.start_at == stored and len(second.turns) == 31 and second.shifted is False
    for number in range(20):
        reader.add(
            "bot" if number % 2 == 0 else "user",
            f"又来{number}",
            START + timedelta(days=2, minutes=number),
        )
    third = recent.window(stored)
    assert third.shifted is True and third.turns[0].at > stored


def test_the_policy_comes_from_the_engine_settings(services: Services) -> None:
    services.settings.engine.history_turns_min = 6
    services.settings.engine.history_turns_max = 9
    recent = RecentTurns.from_settings(conversation(20), services)
    assert (recent.policy.minimum, recent.policy.maximum) == (6, 9)
    assert len(recent.window(None).turns) == 6


def test_the_reader_that_is_registered_is_the_one_the_memory_asks(services: Services) -> None:
    """Round 09 registers the ``bot_turns`` reader on import; any other can be put in its place."""
    reader = conversation(4)
    register_bot_turn_reader(lambda _services: reader)
    try:
        assert bot_turn_reader(services) is reader
    finally:
        register_bot_turn_reader(None)
    assert bot_turn_reader(services) is None
    message = BotMessage("x", "bot", "好", START)
    assert message.role == "bot" and ListBotTurnReader([message]).messages_since(None, 5) == [
        message
    ]
