"""The budget runs out step by step and comes back with the next day (R-LLM-008, R-LLM-006).

The ledger is the only thing that counts: the made-up DeepSeek says how many tokens each answer
used, so a few answers spend a day's dollar - then one and a quarter, one and a half.  The
application of ``twin run`` degrades in the order the specification gives, and never stops
answering him:

======  ===========================================================================
level   what the next answers show
======  ===========================================================================
0       thinking on (he asked for it with ``/思考 开``), eight examples of her real replies
1       thinking off, although he asked for it; eight examples
2       three examples instead of eight; the memory block is half as large
3       no proactive message at all (refused for the budget); he is still answered
next    a new local day: level 0 again - thinking, eight examples, the greeting of the morning
======  ===========================================================================
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.proactive_world import opening_curve, proactive_model

pytestmark = pytest.mark.integration

FRIDAY, SATURDAY = date(2026, 10, 9), date(2026, 10, 10)
EXAMPLE = re.compile(r"^例子 \d+（", re.MULTILINE)
PRICE_PER_MTOK = 0.30  # deepseek-flash, a token that was not in the cache
CHEAP = 400


def tokens_for(dollars: float) -> int:
    return round(dollars / PRICE_PER_MTOK * 1_000_000)


async def chat(world, text: str) -> tuple[dict, str]:
    """He writes, she answers: the request that was sent for it, and her reply."""
    spoken = len(world.persona_said)
    await world.say(text)
    await world.run_until_idle()
    assert len(world.persona_said) > spoken, "she did not answer"
    request = world.deepseek.of_kind("reply")[-1]
    return request.body, request.last_user


async def spend(world, dollars: float, text: str) -> None:
    """One answer that costs ``dollars`` (the made-up model says it read that many tokens)."""
    world.deepseek.bill(tokens_for(dollars))
    await chat(world, text)
    world.deepseek.bill(CHEAP)


async def test_the_budget_degrades_in_order_and_recovers_the_next_day(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 15, 0, tzinfo=UTC),  # 10:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    budget = world.assembly.llm.budget
    assert world.services.settings.budget.daily_usd == 1.0
    await world.say("/思考 开")

    # ---- level 0: everything on -----------------------------------------------------------------
    body, prompt = await chat(world, "今天天气怎么样")
    assert body["thinking"] == {"type": "enabled"} and len(EXAMPLE.findall(prompt)) == 8
    assert budget.limits().level == 0

    # ---- a dollar is spent: level 1, thinking goes first ----------------------------------------
    await spend(world, 1.02, "你在干嘛")
    assert budget.limits().level == 1
    body, prompt = await chat(world, "吃饭了没")
    assert body["thinking"] == {"type": "disabled"}  # he asked for thinking; the budget says no
    assert len(EXAMPLE.findall(prompt)) == 8

    # ---- a dollar and a quarter: level 2, a smaller context --------------------------------------
    await spend(world, 0.26, "晚上看电影吗")
    assert budget.limits().level == 2
    body, prompt = await chat(world, "看什么好")
    assert body["thinking"] == {"type": "disabled"} and len(EXAMPLE.findall(prompt)) == 3
    assert budget.limits().memory_budget_factor == 0.5

    # ---- a dollar and a half: level 3, she stops writing first -----------------------------------
    await spend(world, 0.26, "我想看喜剧")
    assert budget.limits().level == 3 and not budget.limits().proactive_allowed
    sent_before = len(world.proactive_rows(outcomes=["sent"]))
    spoken = len(world.persona_said)
    await world.run_until(world.local(17, 30, on=FRIDAY))  # lunch and a long silence go by
    assert len(world.proactive_rows(outcomes=["sent"])) == sent_before
    refused = [r for r in world.proactive_rows(outcomes=["rejected"]) if r.reason == "budget"]
    assert refused, "nothing was refused for the budget"
    assert len(world.persona_said) >= spoken  # (her last answers only)
    body, _ = await chat(world, "你还在吗")  # ... but she answers him, whatever it costs
    assert body["thinking"] == {"type": "disabled"}
    await world.say("/状态")
    status = world.system_said[-1].text
    assert "预算级别" in status
    told = world.alerts()  # the 80% mark, then every level entered, once each
    assert [c for c, _ in told] == [
        "budget_80",
        "budget_level_n",
        "budget_level_n",
        "budget_level_n",
    ]
    assert told[-1] == ("budget_level_n", "critical")  # level 3 is the loud one

    # ---- the next day: the budget starts again, and she writes first again ----------------------
    await world.run_until(world.local(11, 0, on=SATURDAY))
    assert budget.limits().level == 0
    morning = [r for r in world.proactive_rows(outcomes=["sent"]) if r.local_date == SATURDAY]
    assert [r.kind for r in morning][:1] == ["greeting"]  # the planner was asked, and answered
    body, prompt = await chat(world, "早呀")
    assert body["thinking"] == {"type": "enabled"} and len(EXAMPLE.findall(prompt)) == 8
    assert not [
        r for r in world.proactive_rows() if r.reason == "budget" and r.local_date == SATURDAY
    ]

    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
