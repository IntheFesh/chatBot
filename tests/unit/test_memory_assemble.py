"""The memory block of a reply: recall, scoring, the budget, dates (R-MEM-008, R-LLM-008)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.memory import (
    add_event,
    add_fact,
    add_followup,
    add_summary,
    make_memory,
    utc,
)
from twin.llm.budget import BudgetLimits
from twin.llm.tokens import TokenEstimator
from twin.memory.assemble import MAX_MANDATORY, MemoryAssembler
from twin.memory.blocks import MemoryQuery
from twin.memory.memory import Memory
from twin.services import Services

NOW = utc(2026, 3, 10, 18)  # 13:00 on Tuesday 10 March in Chicago
TODAY = date(2026, 3, 10)


@pytest.fixture
def memory(services: Services, embedder: HashingBackend) -> Memory:
    services.settings.memory.recall_min_similarity = 0.45  # the toy embedding collides now and then
    return make_memory(services)


def limits(factor: float, level: int = 0) -> BudgetLimits:
    return BudgetLimits(
        level=level,
        examples_k=8,
        memory_budget_factor=factor,
        chat_thinking_allowed=True,
        planner_thinking_allowed=True,
        proactive_allowed=True,
        prefer_style_backend=False,
        minimal_context=False,
    )


def build(memory: Memory, topic: str, budget: int | None = None, **kwargs: object):  # type: ignore[no-untyped-def]
    assembler = MemoryAssembler(memory, **kwargs)  # type: ignore[arg-type]
    return assembler.build(MemoryQuery(topic), NOW, budget)


# ------------------------------------------------------------------------ recall


def test_the_block_holds_what_fits_the_topic_under_small_headings(memory: Memory) -> None:
    add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1), importance=3)
    add_fact(memory, "她每天早上跑步", utc(2026, 3, 2), importance=1)
    block = build(memory, "晚上想吃火锅")
    assert "【相关的事】" in block.text and "- 她：她喜欢吃火锅" in block.text
    assert block.sections[0] == "相关的事" and not block.empty
    assert block.ids("fact") and block.tokens > 0 and block.budget_tokens == 800


def test_an_empty_memory_gives_an_empty_block(memory: Memory) -> None:
    block = build(memory, "你好")
    assert (block.text, block.items, block.tokens, block.sections) == ("", (), 0, ())
    assert block.empty


def test_facts_found_by_both_searches_appear_once(memory: Memory) -> None:
    fact = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    block = build(memory, "她喜欢吃火锅")
    assert block.ids("fact") == [fact.id]
    assert block.text.count("她喜欢吃火锅") == 1


def test_the_subject_limits_the_facts(memory: Memory) -> None:
    hers = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1), subject="her")
    his = add_fact(memory, "对方喜欢吃火锅", utc(2026, 3, 1), subject="user")
    both = MemoryAssembler(memory).build(MemoryQuery("火锅"), NOW)
    assert set(both.ids("fact")) == {hers.id, his.id}
    only_hers = MemoryAssembler(memory).build(MemoryQuery("火锅", frozenset({"her"})), NOW)
    assert only_hers.ids("fact") == [hers.id]


def test_important_facts_are_always_in_and_the_better_source_and_importance_rank_first(
    memory: Memory,
) -> None:
    core = add_fact(memory, "她叫对方宝宝", utc(2026, 1, 1), category="nickname", importance=5)
    invented = add_fact(
        memory, "她喜欢吃火锅鸡", utc(2026, 3, 1), source="bot_invented", importance=2
    )
    real = add_fact(memory, "她喜欢吃火锅面", utc(2026, 3, 1), source="real_record", importance=2)
    block = build(memory, "想吃火锅")
    ids = block.ids("fact")
    assert core.id in ids  # nothing in the topic mentions it, yet it is there
    assert ids.index(real.id) < ids.index(invented.id)  # same match, the real record first
    assert "（我自己说过的，没有记录证实）" in block.text.split("她喜欢吃火锅鸡")[1].splitlines()[0]
    assert "（我自己说过" not in block.text.split("她喜欢吃火锅面")[1].splitlines()[0]


def test_the_weights_of_the_score_come_from_the_settings(
    services: Services, memory: Memory
) -> None:
    important = add_fact(
        memory, "她在准备考研", utc(2026, 3, 1), importance=5, source="bot_invented"
    )
    close = add_fact(memory, "她想吃火锅", utc(2026, 3, 1), importance=1, source="real_record")
    first = build(memory, "想吃火锅", 1000).ids("fact")
    assert first.index(close.id) < first.index(important.id)
    services.settings.memory.weights.similarity = 0.0
    services.settings.memory.weights.importance = 1.0
    again = build(memory, "想吃火锅", 1000).ids("fact")
    assert again.index(important.id) < again.index(close.id)


def test_newer_facts_rank_above_older_ones_of_equal_weight(memory: Memory) -> None:
    old = add_fact(memory, "她想吃火锅", utc(2024, 3, 1), importance=2)
    new = add_fact(
        memory, "她想吃火锅啊", utc(2026, 3, 8), importance=2
    )  # the same words to the toy model
    ids = build(memory, "想吃火锅", 1000).ids("fact")
    assert ids.index(new.id) < ids.index(old.id)


# -------------------------------------------------------------------------- budget


def test_the_block_stays_within_the_budget_and_the_better_items_win(memory: Memory) -> None:
    for number in range(30):
        add_fact(memory, f"她喜欢吃火锅配料{number}号很多种类", utc(2026, 3, 1), importance=2)
    best = add_fact(memory, "她最爱吃火锅", utc(2026, 3, 9), importance=5)
    estimator = TokenEstimator()
    block = build(memory, "想吃火锅", 80, estimator=estimator)
    assert block.tokens <= 80 and block.budget_tokens == 80 and block.dropped > 0
    assert estimator.estimate_text(block.text) <= 80 + 6 * len(block.items)  # lines add a dash
    assert best.id in block.ids("fact")
    wide = build(memory, "想吃火锅", 2000)
    assert (
        len(wide.ids("fact")) > len(block.ids("fact")) and wide.dropped == 0
    ) or wide.dropped < block.dropped


def test_a_degraded_budget_level_cuts_the_block(services: Services, memory: Memory) -> None:
    services.settings.memory.recall_facts = 40
    for number in range(30):
        add_fact(memory, f"她喜欢吃火锅配料{number}号很多种类", utc(2026, 3, 1))
    full = build(memory, "想吃火锅", 400, limits=lambda: limits(1.0))
    half = build(memory, "想吃火锅", 400, limits=lambda: limits(0.5, level=2))
    quarter = build(memory, "想吃火锅", 400, limits=lambda: limits(0.25, level=4))
    assert (full.budget_tokens, half.budget_tokens, quarter.budget_tokens) == (400, 200, 100)
    assert full.tokens > half.tokens > quarter.tokens
    default = MemoryAssembler(memory, limits=lambda: limits(0.5, level=2)).build(
        MemoryQuery("火锅"), NOW
    )
    assert default.budget_tokens == 400  # the configured 800, halved


# --------------------------------------------------------------------- dates


def test_a_follow_up_due_today_is_in_the_block_whatever_the_budget(memory: Memory) -> None:
    due = utc(2026, 3, 10, 21)  # 16:00 today
    follow = add_followup(memory, "她下午四点要考试", due, utc(2026, 3, 9))
    later = add_followup(memory, "她周五去医院", utc(2026, 3, 13, 20), utc(2026, 3, 9))
    for number in range(20):
        add_fact(memory, f"她喜欢吃火锅配料{number}号很多种类", utc(2026, 3, 1))
    block = build(memory, "想吃火锅", 1)
    assert follow.id in block.ids("followup") and block.text.startswith("【今天要留意的】")
    assert "她下午四点要考试（3月10日 16:00）" in block.text
    assert next(i for i in block.items if i.item_id == follow.id).mandatory
    assert later.id not in block.ids("followup")  # three days away: not near, not due today
    assert block.ids("fact") == []  # the budget of 1 token left room for nothing else


def test_a_near_follow_up_is_in_when_the_budget_allows_and_an_old_one_is_not(
    memory: Memory,
) -> None:
    tomorrow = add_followup(memory, "她明天面试", utc(2026, 3, 11, 20), utc(2026, 3, 9))
    gone = add_followup(memory, "她上周去看牙", utc(2026, 3, 2, 20), utc(2026, 3, 1))
    block = build(memory, "你好")
    assert tomorrow.id in block.ids("followup") and gone.id not in block.ids("followup")
    done = add_followup(
        memory, "她今天去开会", utc(2026, 3, 10, 20), utc(2026, 3, 9), close_at=utc(2026, 3, 10, 12)
    )
    assert done.id not in build(memory, "你好").ids("followup")  # closed follow-ups are not asked


@pytest.mark.parametrize("year", [2026, 2027, 2028, 2029, 2031])
def test_a_yearly_anniversary_is_in_the_block_on_its_day_every_year(
    memory: Memory, year: int
) -> None:
    birthday = add_fact(
        memory,
        "对方的生日",
        utc(2025, 1, 1),
        subject="user",
        category="anniversary",
        event_date=date(1999, 3, 10),
        recurrence="yearly",
        importance=4,
    )
    on_the_day = MemoryAssembler(memory).build(
        MemoryQuery("你好"), datetime(year, 3, 10, 18, tzinfo=NOW.tzinfo)
    )
    assert birthday.id in on_the_day.ids("fact")
    assert "对方：对方的生日（就是今天）" in on_the_day.text
    assert next(i for i in on_the_day.items if i.item_id == birthday.id).mandatory
    eve = MemoryAssembler(memory).build(
        MemoryQuery("你好"), datetime(year, 3, 9, 18, tzinfo=NOW.tzinfo)
    )
    assert "（就是明天）" in eve.text
    far = MemoryAssembler(memory).build(
        MemoryQuery("你好"), datetime(year, 6, 9, 18, tzinfo=NOW.tzinfo)
    )
    assert birthday.id not in far.ids("fact")


def test_the_day_is_her_local_day_not_the_utc_day(memory: Memory) -> None:
    exam = add_fact(
        memory,
        "她今天考试",
        utc(2026, 3, 1),
        event_date=TODAY,
        category="anniversary",
        importance=4,
    )
    late_evening = MemoryAssembler(memory).build(
        MemoryQuery("你好"), utc(2026, 3, 11, 3)
    )  # 22:00 on the 10th
    assert exam.id in late_evening.ids("fact") and "（就是今天）" in late_evening.text
    next_day = MemoryAssembler(memory).build(MemoryQuery("你好"), utc(2026, 3, 11, 12))
    assert "（是昨天）" in next_day.text


def test_a_one_off_date_and_a_monthly_one(memory: Memory) -> None:
    monthly = add_fact(
        memory,
        "每月十号发工资",
        utc(2026, 1, 1),
        event_date=date(2026, 1, 10),
        recurrence="monthly",
    )
    block = build(memory, "你好")
    assert monthly.id in block.ids("fact") and "（就是今天）" in block.text
    only_a_few = [
        add_fact(
            memory,
            f"周{n}有活动",
            utc(2026, 3, 1),
            event_date=TODAY + timedelta(days=1),
            importance=n % 5 + 1,
        )
        for n in range(MAX_MANDATORY + 3)
    ]
    crowded = build(memory, "你好", 1)
    assert sum(1 for i in crowded.items if i.mandatory) <= MAX_MANDATORY + 1
    assert len(only_a_few) > MAX_MANDATORY


# -------------------------------------------------------------------- summaries


def test_the_last_days_are_summarised_and_older_days_come_in_when_the_topic_fits(
    memory: Memory,
) -> None:
    recent = [
        add_summary(memory, "real", date(2026, 3, 9), "昨天他们聊了天气"),
        add_summary(memory, "real", date(2026, 3, 8), "前天他们聊了电影"),
        add_summary(memory, "bot", date(2026, 3, 7), "三天前他们聊了周末"),
    ]
    add_summary(memory, "real", date(2026, 3, 6), "四天前他们聊了新家")  # beyond three days
    topic = add_summary(memory, "real", date(2026, 2, 1), "那天他们聊了潜水课的事")
    memory.store.mark_bot_online(utc(2026, 3, 1))
    memory.refresh()
    block = build(memory, "想报潜水课", 2000)
    assert set(block.ids("summary")) >= {s.id for s in recent} | {topic.id}
    assert "【最近几天】" in block.text and "【更早的相关日子】" in block.text
    assert "3月9日周一（聊天记录）：昨天他们聊了天气" in block.text
    assert "3月7日周六（和对方的聊天）" in block.text
    plain = build(memory, "你好", 2000)
    assert topic.id not in plain.ids("summary")  # nothing links the topic to that day


def test_a_summary_of_today_or_the_future_is_never_in(memory: Memory) -> None:
    add_summary(memory, "real", TODAY, "今天自己的摘要")
    add_summary(memory, "real", date(2026, 3, 11), "明天的摘要")
    assert build(memory, "摘要", 2000).ids("summary") == []


# ---------------------------------------------------------------------- life line


def test_the_life_line_of_today_is_in_the_live_block_but_not_in_a_view_without_it(
    memory: Memory,
) -> None:
    memory.store.mark_bot_online(utc(2026, 3, 1))
    add_event(
        memory, TODAY, "去图书馆自习", start="09:00", end="11:00", created_at=utc(2026, 3, 10, 12)
    )
    add_event(
        memory, TODAY, "和朋友吃饭", start="12:00", end="13:00", created_at=utc(2026, 3, 10, 12)
    )
    add_event(memory, date(2026, 3, 9), "昨天的事", created_at=utc(2026, 3, 9))
    live = build(memory, "你好", 1000)
    assert "【今天的生活线】" in live.text
    assert live.text.index("去图书馆自习") < live.text.index("和朋友吃饭")
    assert "昨天的事" not in live.text and "09:00-11:00 去图书馆自习" in live.text
    without = MemoryAssembler(memory).view(NOW, include_lifeline=False).render(MemoryQuery("你好"))
    assert "生活线" not in without.text


async def test_building_can_run_off_the_event_loop(memory: Memory) -> None:
    add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    block = await MemoryAssembler(memory).abuild(MemoryQuery("火锅"), NOW)
    assert "她喜欢吃火锅" in block.text


# ------------------------------------------------- a vector backend that fails (D-731)


class IndexDown(OSError):
    """What a vector table that cannot be read raises (the message is not logged)."""


def break_vectors(memory: Memory) -> list[str]:
    """Make every vector search of ``memory`` fail; the list holds the searches that were made."""
    searches: list[str] = []

    def search(kind: object, text: str, *args: object, **kwargs: object) -> object:
        searches.append(text)
        raise IndexDown("the index cannot be read")

    memory.vectors.search = search  # type: ignore[method-assign,assignment]
    return searches


def lived_memory(memory: Memory) -> dict[str, str]:
    """A memory with what a block needs even without vectors: a follow-up, a birthday, a topic."""
    memory.store.mark_bot_online(utc(2026, 3, 1))
    ids = {
        "followup": add_followup(
            memory, "她下午四点要考试", utc(2026, 3, 10, 21), utc(2026, 3, 9)
        ).id,
        "birthday": add_fact(
            memory,
            "对方的生日",
            utc(2025, 1, 1),
            subject="user",
            category="anniversary",
            event_date=date(1999, 3, 10),
            recurrence="yearly",
            importance=4,
        ).id,
        "keyword": add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1), importance=2).id,
        "summary": add_summary(memory, "real", date(2026, 2, 1), "那天他们聊了潜水课的事").id,
    }
    memory.refresh()
    return ids


def test_a_failing_vector_search_leaves_the_block_with_what_needs_no_vectors(
    memory: Memory, caplog: pytest.LogCaptureFixture
) -> None:
    """R17-35: the follow-up, the birthday, the keyword hits stay; the block says it is degraded."""
    ids = lived_memory(memory)
    healthy = MemoryAssembler(memory).build(MemoryQuery("火锅 潜水课"), NOW)
    assert not healthy.degraded
    assert {ids["followup"], ids["birthday"], ids["keyword"], ids["summary"]} <= {
        item.item_id for item in healthy.items
    }
    searches = break_vectors(memory)
    with caplog.at_level("INFO"):
        block = MemoryAssembler(memory).build(MemoryQuery("火锅 潜水课"), NOW)
    assert searches  # the backend was asked and it failed
    assert block.degraded and not block.empty
    assert block.ids("followup") == [ids["followup"]]  # due today: mandatory, no vectors needed
    assert ids["birthday"] in block.ids("fact") and "对方的生日（就是今天）" in block.text
    assert ids["keyword"] in block.ids("fact")  # the keyword search still finds it
    assert block.ids("summary") == [ids["summary"]]  # and the summary of the topic
    assert "memory_vectors_unavailable" in caplog.text
    assert "火锅" not in caplog.text and "the index cannot be read" not in caplog.text


def test_the_marker_of_a_healthy_block_is_not_set(memory: Memory) -> None:
    lived_memory(memory)
    assert MemoryAssembler(memory).build(MemoryQuery("想吃火锅"), NOW).degraded is False
    assert MemoryAssembler(memory).build(MemoryQuery("  "), NOW).degraded is False  # nothing asked


def test_a_view_made_for_a_past_moment_still_raises_when_the_vectors_fail(
    services: Services, memory: Memory
) -> None:
    """R17-35, iron rule 13: the training export and the evaluation never get a thinner block."""
    from twin.memory.api import memory_view
    from twin.memory.asof import AsOfSource

    lived_memory(memory)
    break_vectors(memory)
    query = MemoryQuery("想吃火锅")
    view = MemoryAssembler(memory).view(NOW)
    assert not view.vectors_degraded
    with pytest.raises(IndexDown):
        view.search_facts("想吃火锅", 5)
    with pytest.raises(IndexDown):
        view.search_summaries("想报潜水课", 5)
    with pytest.raises(IndexDown):
        view.render(query)
    with pytest.raises(IndexDown):
        memory_view(services, NOW, memory=memory).render(query)
    source = AsOfSource(services, memory=memory)
    with pytest.raises(IndexDown):
        source.at(NOW).memory_block(query)
    assert not view.vectors_degraded  # and nothing was quietly noted either


def test_the_view_of_the_live_assembler_is_the_only_one_that_degrades(memory: Memory) -> None:
    lived_memory(memory)
    break_vectors(memory)
    live = MemoryAssembler(memory).view(NOW, degrade_vectors=True)
    assert live.search_facts("想吃火锅", 5)  # the keyword hit alone
    assert live.vectors_degraded
    again = MemoryAssembler(memory).view(NOW)
    with pytest.raises(IndexDown):
        again.search_facts("想吃火锅", 5)


def test_a_backend_that_failed_is_not_asked_again_for_five_minutes(
    services: Services, memory: Memory
) -> None:
    """D-731: a model that cannot be loaded does not cost its load time on every reply."""
    clock = services.clock
    assert isinstance(clock, ManualClock)
    lived_memory(memory)
    assembler = MemoryAssembler(memory)
    original = memory.vectors.search
    searches = break_vectors(memory)
    first = assembler.build(MemoryQuery("火锅 潜水课"), NOW)
    assert first.degraded and len(searches) == 1  # the summaries' search was not even tried
    clock.tick(299)
    second = assembler.build(MemoryQuery("火锅 潜水课"), NOW)
    assert second.degraded and len(searches) == 1  # within the five minutes: not asked
    assert second.ids("followup") == first.ids("followup")
    assert second.ids("fact") == first.ids("fact")
    memory.vectors.search = original  # type: ignore[method-assign]
    clock.tick(2)  # five minutes have passed since the failure
    third = assembler.build(MemoryQuery("火锅 潜水课"), NOW)
    assert not third.degraded  # asked again, answered, and the block is whole again


def test_a_backend_that_fails_again_after_the_wait_is_marked_again(
    services: Services, memory: Memory
) -> None:
    clock = services.clock
    assert isinstance(clock, ManualClock)
    lived_memory(memory)
    assembler = MemoryAssembler(memory)
    searches = break_vectors(memory)
    assert assembler.build(MemoryQuery("想吃火锅"), NOW).degraded
    clock.tick(301)
    assert assembler.build(MemoryQuery("想吃火锅"), NOW).degraded
    assert len(searches) == 2  # one try per five minutes


async def test_the_degraded_block_comes_through_the_data_source_of_the_running_bot(
    services: Services, memory: Memory
) -> None:
    from twin.engine.dataview import LiveDataSource

    ids = lived_memory(memory)
    break_vectors(memory)
    source = LiveDataSource(services, memory=memory)
    try:
        block = source.view(NOW).memory_block(MemoryQuery("想吃火锅"))
        assert block.degraded and ids["followup"] in block.ids("followup")
    finally:
        await source.aclose()
