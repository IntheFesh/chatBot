"""R-EVAL-004: what an audit is shown - the life line, her words about herself, the facts.

The pack is read through the memory's own views and the reader of the bot's conversation; the
tests check what is in it, what is not, and that building it writes nothing.
"""

from __future__ import annotations

from datetime import timedelta

from tests.support.consistency_world import (
    BEIJING,
    HOME_ALL_DAY,
    LIBRARY,
    MEETING,
    OLD_INVENTION,
    SECRET_MARK,
    SHANGHAI,
    STAYED_HOME_OLD,
    USER_JOB,
    build_week,
)
from tests.support.memory import add_fact
from twin.eval.consistency_pack import build_pack, render_pack
from twin.eval.isolation import changes, snapshot
from twin.memory.recent import register_bot_turn_reader, registered_bot_turn_reader
from twin.services import Services


def texts(pack_items: tuple[object, ...]) -> list[str]:
    return [getattr(item, "text") for item in pack_items]  # noqa: B009 - items are Evidence


def test_the_pack_holds_the_window_of_the_life_line_in_time_order(services: Services) -> None:
    build_week(services)
    pack = build_pack(services, days=7)
    lifeline = pack.of("lifeline")
    assert [item.ref for item in lifeline] == ["L1", "L2", "L3"]
    assert LIBRARY in lifeline[0].text and MEETING in lifeline[1].text
    assert "去医院复查" in lifeline[2].text
    assert "10-06（周二）" in lifeline[0].text  # the date and the weekday are in the text
    assert lifeline[0].source == "plan" and lifeline[2].source == "improvised"
    assert lifeline[0].at < lifeline[1].at < lifeline[2].at
    everything = " ".join(item.text for item in pack.items)
    assert "在咖啡店看书" not in everything  # before the window
    assert "在跑步机上跑步" not in everything  # invalidated: the memory does not show it


def test_only_what_the_bot_said_is_handed_over_and_only_inside_the_window(
    services: Services,
) -> None:
    build_week(services)
    pack = build_pack(services, days=7)
    replies = pack.of("reply")
    assert len(replies) == 3  # three exchanges in the window; the old one is out
    joined = " ".join(item.text for item in replies)
    assert HOME_ALL_DAY in joined and SECRET_MARK in joined
    assert STAYED_HOME_OLD not in joined
    for typed in ("今天干嘛了", "你今天忙吗", "周末有安排吗"):
        assert typed not in joined  # what the user wrote is not handed over
    assert "刚下班，好累 / 晚上想早点睡" in joined  # the bubbles of one turn are one record
    first = replies[0]
    assert len(first.message_ids) == 2 and first.item_id == first.message_ids[0]


def test_replies_are_cut_from_the_old_end_when_they_are_too_long(services: Services) -> None:
    build_week(services)
    services.settings.eval.consistency_reply_chars = 405
    store_text = "很长的话" * 100  # 400 characters, the cap of one turn
    from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta

    store = BotTurnStore(services.db, services.clock)
    at = services.clock.now_utc() - timedelta(hours=3)
    store.add_inbound(at=at, kind="text", text="说说看")
    store.add_reply([OutboundBubble(store_text, at + timedelta(seconds=20))], ReplyMeta("deepseek"))
    pack = build_pack(services, days=7)
    replies = pack.of("reply")
    assert len(replies) == 1 and len(replies[0].text) == 400  # the newest turn is kept, cut
    assert pack.reply_turns_cut == 3  # the three older turns did not fit next to it


def test_facts_are_those_about_her_that_are_close_or_freshly_invented(services: Services) -> None:
    week = build_week(services)
    pack = build_pack(services, days=7)
    facts = pack.of("fact")
    shown = {item.text: item for item in facts}
    assert SHANGHAI in shown  # invented inside the window
    assert BEIJING in shown  # a real fact: "北京" is close to nothing said, yet it is by keyword
    assert shown[SHANGHAI].source == "bot_invented" and shown[BEIJING].source == "real_record"
    assert shown[SHANGHAI].number == week.invented_fact.number
    assert shown[BEIJING].known_at == week.real_fact.known_at
    assert USER_JOB not in shown  # about him, not her
    assert OLD_INVENTION not in shown  # old and not close to anything
    assert [item.ref for item in facts] == [f"F{n}" for n in range(1, len(facts) + 1)]


def test_a_fact_that_a_record_is_close_by_keyword_is_included(services: Services) -> None:
    week = build_week(services)
    add_fact(
        week.memory,
        "她一周会去几次图书馆看书",
        week.real_fact.known_at,
        source="real_record",
        embed=False,
    )
    pack = build_pack(services, days=7)
    assert "她一周会去几次图书馆看书" in [item.text for item in pack.of("fact")]


def test_the_number_of_facts_is_limited_and_may_be_zero(services: Services) -> None:
    week = build_week(services)
    for number in range(5):
        add_fact(
            week.memory,
            f"她今天在图书馆看书的第{number}件事",
            week.real_fact.known_at,
            source="real_record",
            embed=False,
        )
    services.settings.eval.consistency_facts = 2
    assert len(build_pack(services, days=7).of("fact")) == 2
    services.settings.eval.consistency_facts = 0
    assert build_pack(services, days=7).of("fact") == ()


def test_a_week_with_nothing_in_it_is_empty(services: Services) -> None:
    pack = build_pack(services, days=7)
    assert pack.empty and pack.items == () and pack.chars == 0
    week = build_week(services)
    assert not build_pack(services, days=7).empty
    far = services.clock.now_utc() + timedelta(days=60)
    assert build_pack(services, days=7, now=far, memory=week.memory).empty


def test_without_a_reader_of_the_conversation_there_are_no_replies(services: Services) -> None:
    build_week(services)
    previous = registered_bot_turn_reader()
    register_bot_turn_reader(None)
    try:
        pack = build_pack(services, days=7)
    finally:
        register_bot_turn_reader(previous)
    assert pack.of("reply") == () and len(pack.of("lifeline")) == 3


def test_the_window_is_the_days_asked_for(services: Services) -> None:
    build_week(services)
    wide = build_pack(services, days=14)
    assert any(STAYED_HOME_OLD in item.text for item in wide.of("reply"))
    assert any("在咖啡店看书" in item.text for item in wide.of("lifeline"))
    narrow = build_pack(services, days=1)
    assert [SECRET_MARK in item.text for item in narrow.of("reply")] == [True]  # the last day only
    assert len(narrow.of("lifeline")) == 1  # the entry of the 8th
    assert narrow.days == 1 and narrow.zone == "America/Chicago"
    assert narrow.window_end - narrow.window_start == timedelta(days=1)


def test_the_prompt_fields_name_every_record_with_its_number(services: Services) -> None:
    build_week(services)
    fields = render_pack(build_pack(services, days=7))
    assert set(fields) == {"window", "days", "lifeline", "replies", "facts"}
    assert fields["days"] == "7" and "2026-10-02" in fields["window"]
    assert fields["lifeline"].splitlines()[0].startswith("L1 10-06（周二）")
    assert all(
        line.startswith(f"R{n} 10-") for n, line in enumerate(fields["replies"].splitlines(), 1)
    )
    assert "（来源：机器人自己说的；2026-10-07 知道）" in fields["facts"]
    assert "（来源：真实聊天记录；2026-09-01 知道）" in fields["facts"]
    empty = render_pack(
        build_pack(services, days=7, now=services.clock.now_utc() + timedelta(days=90))
    )
    assert empty["lifeline"] == "（没有）" and empty["replies"] == "（没有）"


def test_building_the_pack_writes_nothing(services: Services) -> None:
    build_week(services)
    before = snapshot(services.db)
    build_pack(services, days=7)
    assert changes(before, snapshot(services.db)).tables == frozenset()
