"""How the consistency audit screen shows the time of a statement (R-EVAL-004, R-NFR-005).

The screen shows a reply by the clock of the place where the bot lives and a fact by the day on
which the bot got to know it.  The moments come from a manual clock; the zone is the one the audit
run recorded, so the same moments read differently in Chicago and in Shanghai, and the hour in which
Chicago's clocks go back on 2026-11-01 is told apart by the text around it, not by the clock face.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tests.support.clock import ManualClock
from twin.eval.consistency_model import Statement
from twin.eval.consistency_ui import describe
from twin.memory.conflict import SOURCE_NAMES

CHICAGO = ZoneInfo("America/Chicago")
SHANGHAI = ZoneInfo("Asia/Shanghai")
UTC_START = datetime.fromisoformat("2026-11-01T06:30:00+00:00")  # 01:30 CDT, the first of two


def reply(at: datetime, text: str = "我在图书馆") -> Statement:
    return Statement(ref="B1", kind="reply", item_id="turn-1", text=text, source=None, at=at)


def fact(at: datetime, known_at: datetime | None, number: int | None = 7) -> Statement:
    return Statement(
        ref="F1",
        kind="fact",
        item_id="fact-1",
        text="她周三有考试",
        source="user_said",
        at=at,
        known_at=known_at,
        number=number,
    )


def test_a_reply_is_shown_by_the_wall_clock_of_the_zone_of_the_run() -> None:
    clock = ManualClock(UTC_START)
    first = describe(reply(clock.now_utc()), CHICAGO).plain
    assert first == "她说过的话（11-01 01:30）：我在图书馆"
    assert (
        describe(reply(clock.now_utc()), SHANGHAI).plain == "她说过的话（11-01 14:30）：我在图书馆"
    )


def test_the_repeated_hour_reads_the_same_in_chicago_and_differently_in_shanghai() -> None:
    clock = ManualClock(UTC_START)
    before = clock.now_utc()
    clock.set_time(UTC_START + timedelta(hours=1))  # 07:30 UTC = 01:30 CST: the face repeats
    after = clock.now_utc()
    assert after - before == timedelta(hours=1)
    assert describe(reply(before), CHICAGO).plain == describe(reply(after), CHICAGO).plain
    assert describe(reply(before), SHANGHAI).plain == "她说过的话（11-01 14:30）：我在图书馆"
    assert describe(reply(after), SHANGHAI).plain == "她说过的话（11-01 15:30）：我在图书馆"


def test_a_fact_is_dated_by_the_day_she_got_to_know_it_in_the_zone_of_the_run() -> None:
    clock = ManualClock(datetime.fromisoformat("2026-11-01T04:30:00+00:00"))  # 23:30 on 10-31 CDT
    known = clock.now_utc()
    text_chicago = describe(fact(known, known), CHICAGO).plain
    text_shanghai = describe(fact(known, known), SHANGHAI).plain
    assert (
        text_chicago
        == f"记忆里的事实 #7（{SOURCE_NAMES['user_said']}；2026-10-31 知道）：她周三有考试"
    )
    assert (
        text_shanghai
        == f"记忆里的事实 #7（{SOURCE_NAMES['user_said']}；2026-11-01 知道）：她周三有考试"
    )


def test_a_fact_without_a_known_at_falls_back_to_its_own_moment() -> None:
    clock = ManualClock(UTC_START)
    plain = describe(fact(clock.now_utc(), None, number=None), CHICAGO).plain
    assert plain == f"记忆里的事实（{SOURCE_NAMES['user_said']}；2026-11-01 知道）：她周三有考试"


def test_the_life_line_is_shown_without_a_time() -> None:
    clock = ManualClock(UTC_START)
    statement = Statement(
        ref="L1",
        kind="lifeline",
        item_id="life-1",
        text="下午在图书馆",
        source=None,
        at=clock.now_utc(),
    )
    assert describe(statement, CHICAGO).plain == "生活安排：下午在图书馆"
