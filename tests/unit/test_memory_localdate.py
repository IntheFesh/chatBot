"""The bot's local days follow the zone it lived in at the time, not the one it lives in now.

R17-33 (D-732).  ``MemoryClock`` used to date every moment of the bot's conversation in the bot's
*current* zone, so after ``/时区 Asia/Shanghai`` the evening of 5 October in Chicago was the 6th,
summaries of old days were cut in the new zone and the extractor was told the wrong zone.  The
bot's zone is now a timeline built from ``timezone_history`` (``BotZoneTimeline``): a moment is
dated in the zone in force at that moment, a day is cut in the zone in force at 12:00 UTC of its
date.  Summaries that were written under the old rule are not corrected by themselves
(``twin memory summarize --force``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import date, timedelta

import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.memory import CHICAGO, ScriptedMemoryModel, make_memory, utc
from tests.support.routine import Rig, fixed_model
from twin.config.runtime import BOT_TIMEZONE
from twin.ingest.times import SourceTime
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY
from twin.memory import localdate
from twin.memory.extract import DialogueLine, FactExtractor
from twin.memory.localdate import BOT, REAL, MemoryClock
from twin.memory.memory import Memory
from twin.memory.summarize import DailySummarizer
from twin.schedule.service import schedule_kit
from twin.schedule.store import TimezoneHistory
from twin.schedule.time_service import BotTimeService
from twin.schedule.zone_timeline import BotZoneTimeline, ZoneSwitch
from twin.services import Services

SHANGHAI = "Asia/Shanghai"
SWITCH = utc(2026, 10, 6, 6, 0)  # Chicago to Shanghai: 01:00 on the 6th in Chicago, 14:00 there
EVENING = utc(2026, 10, 6, 4, 30)  # 23:30 on the 5th in Chicago, 12:30 on the 6th in Shanghai
CHICAGO_TO_SHANGHAI = ZoneSwitch(SWITCH, CHICAGO, SHANGHAI)


class Zone:
    """The bot's zone as the runtime setting says it (a test can change it)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> str:
        return self.name


def switched(clock: ManualClock, now: str = SHANGHAI) -> MemoryClock:
    """A calendar of a bot that was switched from Chicago to Shanghai at ``SWITCH``."""
    zone = Zone(now)
    timeline = BotZoneTimeline([CHICAGO_TO_SHANGHAI], zone)
    return MemoryClock(SourceTime(CHICAGO), BotTimeService(clock, zone), lambda: timeline)


# ---------------------------------------------------------------------------- the timeline


def test_the_memory_module_offers_the_timeline_under_its_own_name() -> None:
    assert localdate.BotZoneTimeline is BotZoneTimeline and localdate.ZoneSwitch is ZoneSwitch


def test_before_the_switch_the_bot_lived_in_the_zone_it_was_switched_from() -> None:
    timeline = BotZoneTimeline([CHICAGO_TO_SHANGHAI], Zone(SHANGHAI))
    assert timeline.zone_for(EVENING).key == CHICAGO
    assert timeline.zone_for(SWITCH - timedelta(microseconds=1)).key == CHICAGO
    assert timeline.zone_for(utc(2020, 1, 1)).key == CHICAGO  # however long ago
    assert timeline.zone_for(SWITCH).key == SHANGHAI  # a switch is in force from its own instant
    assert timeline.zone_for(SWITCH + timedelta(hours=1)).key == SHANGHAI
    assert timeline.zone_for(utc(2030, 1, 1)).key == SHANGHAI


def test_two_switches_make_three_stretches_and_the_order_of_the_rows_does_not_matter() -> None:
    back = ZoneSwitch(utc(2026, 10, 20, 6), SHANGHAI, CHICAGO)
    for rows in ([CHICAGO_TO_SHANGHAI, back], [back, CHICAGO_TO_SHANGHAI]):
        timeline = BotZoneTimeline(rows, Zone(CHICAGO))
        assert timeline.zone_for(utc(2026, 10, 1)).key == CHICAGO
        assert timeline.zone_for(utc(2026, 10, 10)).key == SHANGHAI
        assert timeline.zone_for(utc(2026, 10, 25)).key == CHICAGO
        assert [s.to_zone for s in timeline.switches] == [SHANGHAI, CHICAGO]


def test_a_bot_that_was_never_switched_lives_in_the_zone_it_is_in() -> None:
    where = Zone(CHICAGO)
    timeline = BotZoneTimeline([], where)
    assert timeline.zone_for(EVENING).key == CHICAGO
    where.name = SHANGHAI  # nothing records when: the present zone is all there is to go by
    assert timeline.zone_for(EVENING).key == SHANGHAI and timeline.zone_on(
        date(2026, 1, 1)
    ).key == (SHANGHAI)


def test_the_open_stretch_after_the_last_switch_is_the_zone_of_the_present() -> None:
    where = Zone(SHANGHAI)
    timeline = BotZoneTimeline([CHICAGO_TO_SHANGHAI], where)
    assert timeline.zone_for(utc(2026, 11, 1)).key == SHANGHAI
    where.name = "Europe/London"  # the setting changed in a way the history does not know of
    assert timeline.zone_for(utc(2026, 11, 1)).key == "Europe/London"
    assert timeline.zone_for(EVENING).key == CHICAGO  # the past stays as the history says


def test_a_day_is_cut_in_the_zone_in_force_at_noon_utc_of_its_date() -> None:
    timeline = BotZoneTimeline([CHICAGO_TO_SHANGHAI], Zone(SHANGHAI))
    assert timeline.zone_on(date(2026, 10, 5)).key == CHICAGO  # noon UTC on the 5th: before
    assert timeline.zone_on(date(2026, 10, 6)).key == SHANGHAI  # noon UTC on the 6th: after
    late = BotZoneTimeline(  # a switch at 13:00 UTC: after that day's noon
        [ZoneSwitch(utc(2026, 10, 6, 13), CHICAGO, SHANGHAI)], Zone(SHANGHAI)
    )
    assert late.zone_on(date(2026, 10, 6)).key == CHICAGO  # the whole day, switch or not


# --------------------------------------------------------------------------- the calendar


def test_the_evening_before_the_switch_is_still_the_5th_after_it(clock: ManualClock) -> None:
    """R17-33: it was the 6th - the evening of the 5th in Chicago is the 6th in Shanghai."""
    memory = switched(clock)  # the bot lives in Shanghai now
    assert memory.date_of(BOT, EVENING) == date(2026, 10, 5)
    assert memory.bot_date(EVENING) == date(2026, 10, 5)
    assert memory.zone_of(BOT, EVENING).key == CHICAGO
    assert memory.date_of(BOT, SWITCH) == date(2026, 10, 6)  # from the switch on: Shanghai
    assert memory.date_of(BOT, utc(2026, 10, 7, 3)) == date(2026, 10, 7)
    assert memory.bot_zone().key == SHANGHAI  # "now" is still the zone of the present
    assert memory.real_date(EVENING) == date(2026, 10, 5)  # hers never depended on the bot's


def test_the_days_around_the_switch_are_cut_in_their_own_zones(clock: ManualClock) -> None:
    memory = switched(clock)
    assert memory.zone_key_for_day(BOT, date(2026, 10, 5)) == CHICAGO
    assert memory.zone_key_for_day(BOT, date(2026, 10, 6)) == SHANGHAI
    chicago_day = memory.bot_bounds(date(2026, 10, 5))
    assert chicago_day == (utc(2026, 10, 5, 5), utc(2026, 10, 6, 5))
    assert memory.bounds_of(BOT, date(2026, 10, 5)) == chicago_day
    assert chicago_day[0] <= EVENING < chicago_day[1]  # the evening is in the 5th's lines
    assert memory.bot_bounds(date(2026, 10, 6)) == (utc(2026, 10, 5, 16), utc(2026, 10, 6, 16))
    assert memory.bot_bounds(date(2026, 9, 1)) == (utc(2026, 9, 1, 5), utc(2026, 9, 2, 5))


def test_the_known_limitation_a_jump_of_the_clock_overlaps_or_leaves_a_gap(
    clock: ManualClock,
) -> None:
    """D-732: a day has one zone, so the hours the clock jumped at a switch are not partitioned.

    Chicago to Shanghai jumps forward: the last Chicago day and the first Shanghai day share
    thirteen hours, which both days read.  Shanghai to Chicago jumps back: thirteen hours belong
    to neither day.  Nothing is dated wrongly - a moment is always dated in its own zone - and a
    day is still cut in one zone, which is what a summary says it is.
    """
    memory = switched(clock)
    chicago_day = memory.bot_bounds(date(2026, 10, 5))
    shanghai_day = memory.bot_bounds(date(2026, 10, 6))
    assert shanghai_day[0] < chicago_day[1]  # overlap
    assert chicago_day[1] - shanghai_day[0] == timedelta(hours=13)
    back = BotZoneTimeline(
        [CHICAGO_TO_SHANGHAI, ZoneSwitch(utc(2026, 10, 20, 6), SHANGHAI, CHICAGO)], Zone(CHICAGO)
    )
    memory = MemoryClock(SourceTime(CHICAGO), BotTimeService(clock, Zone(CHICAGO)), lambda: back)
    last_shanghai_day = memory.bot_bounds(date(2026, 10, 19))
    first_chicago_day = memory.bot_bounds(date(2026, 10, 20))
    assert first_chicago_day[0] - last_shanghai_day[1] == timedelta(hours=13)  # a gap


def test_a_bot_without_a_history_keeps_dating_everything_in_its_zone(clock: ManualClock) -> None:
    """No switch on record: the calendar made without a timeline is the old one, as documented."""
    where = Zone(CHICAGO)
    memory = MemoryClock(SourceTime(CHICAGO), BotTimeService(clock, where))
    moment = utc(2026, 11, 1, 18)  # noon in Chicago, 02:00 the next day in Shanghai
    assert memory.bot_date(moment) == date(2026, 11, 1)
    where.name = SHANGHAI  # `/时区 Asia/Shanghai`, and no row says when
    assert memory.bot_date(moment) == date(2026, 11, 2)
    assert memory.bot_bounds(date(2026, 11, 1)) == (utc(2026, 10, 31, 16), utc(2026, 11, 1, 16))


# ------------------------------------------------------------------ the days of the clock change


def test_the_day_the_clocks_go_back_is_25_hours_for_the_bot_and_dated_by_the_wall_clock(
    clock: ManualClock,
) -> None:
    """2026-11-01 in Chicago: 01:30 happens twice, the day has 25 hours."""
    where = Zone(CHICAGO)
    memory = MemoryClock(
        SourceTime(CHICAGO), BotTimeService(clock, where), lambda: BotZoneTimeline([], where)
    )
    start, end = memory.bot_bounds(date(2026, 11, 1))
    assert (start, end) == (utc(2026, 11, 1, 5), utc(2026, 11, 2, 6))
    assert end - start == timedelta(hours=25)
    assert memory.bot_date(utc(2026, 11, 2, 5, 59)) == date(2026, 11, 1)  # 23:59 CST
    assert memory.bot_date(utc(2026, 11, 2, 6, 0)) == date(2026, 11, 2)  # midnight CST
    assert memory.bot_date(utc(2026, 11, 1, 6, 30)) == date(
        2026, 11, 1
    )  # 01:30 CDT, the first time
    assert memory.bot_date(utc(2026, 11, 1, 7, 30)) == date(2026, 11, 1)  # 01:30 CST, the second
    spring_start, spring_end = memory.bot_bounds(date(2026, 3, 8))
    assert spring_end - spring_start == timedelta(hours=23)


def test_a_switch_on_the_day_the_clocks_go_back_dates_both_halves_by_their_own_zone(
    clock: ManualClock,
) -> None:
    move = ZoneSwitch(utc(2026, 11, 1, 20), CHICAGO, SHANGHAI)  # 14:00 CST that Sunday
    timeline = BotZoneTimeline([move], Zone(SHANGHAI))
    memory = MemoryClock(
        SourceTime(CHICAGO), BotTimeService(clock, Zone(SHANGHAI)), lambda: timeline
    )
    assert memory.bot_bounds(date(2026, 11, 1)) == (utc(2026, 11, 1, 5), utc(2026, 11, 2, 6))
    assert memory.zone_key_for_day(BOT, date(2026, 11, 1)) == CHICAGO  # the 25-hour day
    assert memory.date_of(BOT, utc(2026, 11, 1, 19, 59)) == date(2026, 11, 1)  # 13:59 CST
    assert memory.date_of(BOT, utc(2026, 11, 1, 20, 0)) == date(2026, 11, 2)  # 04:00 in Shanghai
    assert memory.zone_key_for_day(BOT, date(2026, 11, 2)) == SHANGHAI


# --------------------------------------------------------------- built from the real history


@pytest.fixture
def rig(services: Services, clock: ManualClock) -> Rig:
    clock.set_time(SWITCH)
    return Rig.build(services, clock, fixed_model())


def test_a_switch_made_by_the_planner_is_in_the_timeline_at_once(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    """The cached timeline is dropped with the planner's caches, after the row is written."""
    assert rig.kit.zones().switches == ()  # read, and cached, before the switch
    assert rig.kit.zones().zone_for(EVENING).key == CHICAGO  # nothing switched: the zone it is in
    outcome = rig.planner.switch_timezone(SHANGHAI, source="cli")
    assert outcome.changed
    timeline = rig.kit.zones()
    assert [(s.at, s.from_zone, s.to_zone) for s in timeline.switches] == [
        (SWITCH, CHICAGO, SHANGHAI)
    ]
    memory = MemoryClock(SourceTime(CHICAGO), rig.kit.time, rig.kit.zones)
    assert memory.date_of(BOT, EVENING) == date(2026, 10, 5)
    assert memory.zone_key_for_day(BOT, date(2026, 10, 5)) == CHICAGO
    assert memory.bot_zone().key == SHANGHAI


def test_a_switch_that_another_process_wrote_is_seen_when_the_caches_are_dropped(
    rig: Rig, services: Services, clock: ManualClock
) -> None:
    assert rig.kit.zones().switches == ()
    TimezoneHistory(services.db, clock).add(
        at=SWITCH, old=CHICAGO, new=SHANGHAI, source="cli", plan_id=None, last_greeting_at=None
    )
    services.runtime.set(BOT_TIMEZONE, SHANGHAI, by="cli")  # as `twin timezone set` does
    assert rig.kit.zones().switches == ()  # still what it read before
    rig.kit.drop_caches()  # what the state watcher does when the version of the state moves
    assert [s.to_zone for s in rig.kit.zones().switches] == [SHANGHAI]
    assert rig.kit.zones().zone_for(EVENING).key == CHICAGO


def test_the_memory_of_a_container_dates_the_evening_before_a_switch_as_the_5th(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> None:
    """R17-33 end to end: the container's own kit and memory, as ``twin run`` builds them."""
    clock.set_time(SWITCH)
    memory = make_memory(services)
    assert memory.clock.date_of(BOT, EVENING) == date(2026, 10, 5)  # nothing switched yet
    outcome = schedule_kit(services).planner.switch_timezone(SHANGHAI, source="command")
    assert outcome.changed
    assert services.runtime.get(BOT_TIMEZONE) == SHANGHAI
    assert memory.clock.date_of(BOT, EVENING) == date(2026, 10, 5)  # before the fix: 6 October
    assert memory.clock.date_of(BOT, SWITCH + timedelta(hours=1)) == date(2026, 10, 6)
    assert memory.clock.zone_of(BOT, EVENING).key == CHICAGO
    assert memory.clock.bounds_of(BOT, date(2026, 10, 5)) == (
        utc(2026, 10, 5, 5),
        utc(2026, 10, 6, 5),
    )
    assert memory.clock.date_of(REAL, EVENING) == date(2026, 10, 5)
    # a second memory of the same container reads the same history
    assert make_memory(services).clock.date_of(BOT, EVENING) == date(2026, 10, 5)


# ----------------------------------------------------------- what is built on the calendar


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def runtime(services: Services, embedder: HashingBackend) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


def evening_lines() -> list[DialogueLine]:
    return [
        DialogueLine("t1", "user", "我刚下班", EVENING),
        DialogueLine("t2", "bot", "辛苦啦", EVENING + timedelta(minutes=1)),
    ]


async def test_the_summary_of_the_5th_is_cut_and_told_in_chicago_after_the_switch(
    services: Services,
    clock: ManualClock,
    runtime: LlmRuntime,
    api: respx.MockRouter,
    embedder: HashingBackend,
) -> None:
    """A summary written after the switch for a day before it: the zone of that day."""
    clock.set_time(SWITCH)
    memory = make_memory(services)
    schedule_kit(services).planner.switch_timezone(SHANGHAI, source="cli")
    model = ScriptedMemoryModel(summaries={"2026-10-05": "这天他们聊了下班。"})
    api.post(API).mock(side_effect=model)
    result = await DailySummarizer(memory, runtime.client).summarize(
        BOT, date(2026, 10, 5), evening_lines(), DAILY
    )
    record = result.record
    assert record is not None and record.timezone == CHICAGO  # before the fix: Asia/Shanghai
    assert (record.utc_start, record.utc_end) == (utc(2026, 10, 5, 5), utc(2026, 10, 6, 5))
    prompt = model.requests[0]["messages"][1]["content"]
    assert "日期：2026-10-05（周一，时区 America/Chicago）" in prompt
    assert "[23:30] 对方：我刚下班" in prompt  # the wall clock of Chicago, not "12:30"


async def test_the_extractor_is_told_the_zone_the_bot_lived_in_when_the_lines_were_said(
    services: Services, clock: ManualClock, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    """``FactExtractor`` asked the current zone for a moment in the past (extract.py)."""
    model = ScriptedMemoryModel()
    api.post(API).mock(side_effect=model)
    extractor = FactExtractor(services, runtime.client, switched(clock))
    await extractor.extract(evening_lines(), mode="bot", tag=DAILY)
    prompt = model.requests[0]["messages"][1]["content"]
    assert "对话发生的时区：America/Chicago" in prompt  # before the fix: Asia/Shanghai
    assert "对话开始的当地时间：2026-10-05 23:30（周一）" in prompt
    assert "1. [10-05 23:30] 对方：我刚下班" in prompt
    after = [DialogueLine("t3", "user", "起床了", SWITCH + timedelta(hours=3))]
    await extractor.extract(after, mode="bot", tag=DAILY)
    assert "对话发生的时区：Asia/Shanghai" in model.requests[1]["messages"][1]["content"]


def test_a_summary_already_written_under_the_old_rule_keeps_its_date(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> None:
    """Nothing re-dates what is stored: ``twin memory summarize --force`` is how it is redone."""
    clock.set_time(SWITCH)
    memory = make_memory(services)
    stored = memory.store.add_summary(
        scope=BOT,
        local_date=date(2026, 10, 6),  # the old rule's date of the evening of the 5th
        timezone=SHANGHAI,
        utc_start=utc(2026, 10, 5, 16),
        utc_end=utc(2026, 10, 6, 16),
        text="旧规则写下的摘要",
        input_hash="old",
        template_version=None,
    )
    schedule_kit(services).planner.switch_timezone(SHANGHAI, source="cli")
    memory.refresh()
    kept = memory.store.current_summary(BOT, date(2026, 10, 6))
    assert kept is not None and kept.id == stored.id and kept.timezone == SHANGHAI
    assert memory.store.current_summary(BOT, date(2026, 10, 5)) is None  # no one wrote it again


def test_the_memory_of_a_fresh_installation_has_a_clock_that_follows_the_setting(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> None:
    memory: Memory = make_memory(services)
    assert memory.clock.bot_zone().key == CHICAGO
    assert memory.clock.date_of(BOT, EVENING) == date(2026, 10, 5)
    assert memory.clock.zone_key_for_day(BOT, date(2026, 10, 5)) == CHICAGO
