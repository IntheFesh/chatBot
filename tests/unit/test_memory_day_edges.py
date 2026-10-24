"""Her local days at the edges of a ``time.source_timezone_ranges`` entry (round 18 item C4, D-782).

A day used to be cut in the zone of its own date at both ends, while a message was dated in the
zone of its moment.  Where a trip begins (Chicago to Shanghai) the last day at home and the first
day away then shared thirteen hours, and where it ends thirteen hours belonged to no day: the
replay's scan and the day loader disagreed, the two days around each edge stayed "changed" for
ever and every import replayed them again (and sent their text to DeepSeek again).

Now a day runs from its own midnight (in the zone of its date) to the next day's midnight (in the
zone of *that* date): the days fit together, and a moment belongs to the day whose bounds hold
it.  The days at an edge are 11 or 37 hours long instead of 24.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.support.clock import ManualClock
from tests.support.memory import CHICAGO, utc
from tests.support.synth_chat import MessageWriter
from twin.config.settings import TzRange
from twin.ingest.times import SourceTime
from twin.llm.tokens import TokenEstimator
from twin.memory.dayload import lines_hash, real_day_lines
from twin.memory.localdate import MemoryClock
from twin.memory.replay import ReplayRequest, scan_days
from twin.memory.store import MemoryStore
from twin.schedule.time_service import BotTimeService
from twin.services import Services

SHANGHAI = "Asia/Shanghai"
SUMMER = TzRange.model_validate({"from": "2025-06-01", "to": "2025-08-31", "tz": SHANGHAI})
ZONES = (
    CHICAGO,
    SHANGHAI,
    "Europe/London",
    "Australia/Lord_Howe",  # a half-hour daylight saving change
    "America/Santiago",  # the clock changes at midnight
    "Asia/Kolkata",
    "Pacific/Kiritimati",  # UTC+14
    "Pacific/Pago_Pago",  # UTC-11: next to Kiritimati the clock jumps by 25 hours
)


def test_the_days_at_the_edges_of_a_trip_fit_together() -> None:
    """R18-C4: no hour is in two days, none in no day; the edge days are 11 and 37 hours."""
    source = SourceTime(CHICAGO, [SUMMER])
    before, first = source.day_bounds(date(2025, 5, 31)), source.day_bounds(date(2025, 6, 1))
    assert before == (utc(2025, 5, 31, 5), utc(2025, 5, 31, 16))  # Chicago midnight to Shanghai's
    assert first == (utc(2025, 5, 31, 16), utc(2025, 6, 1, 16))
    last, after = source.day_bounds(date(2025, 8, 31)), source.day_bounds(date(2025, 9, 1))
    assert last == (utc(2025, 8, 30, 16), utc(2025, 9, 1, 5))  # Shanghai midnight to Chicago's
    assert after == (utc(2025, 9, 1, 5), utc(2025, 9, 2, 5))
    moment = utc(2025, 5, 31, 18)  # 13:00 in Chicago, already 02:00 on 1 June in Shanghai
    assert source.local_date(moment) == date(2025, 6, 1)
    moment = utc(2025, 8, 31, 20)  # 04:00 on 1 September in Shanghai, 15:00 on 31 August at home
    assert source.local_date(moment) == date(2025, 8, 31)


def test_a_trip_that_ends_on_the_day_the_clocks_change_at_home() -> None:
    """R18-C4: the edge day and the 23-hour day of daylight saving time follow each other."""
    trip = TzRange.model_validate({"from": "2026-03-01", "to": "2026-03-07", "tz": SHANGHAI})
    source = SourceTime(CHICAGO, [trip])
    assert source.day_bounds(date(2026, 3, 7)) == (utc(2026, 3, 6, 16), utc(2026, 3, 8, 6))
    start, end = source.day_bounds(date(2026, 3, 8))
    assert (start, end - start) == (utc(2026, 3, 8, 6), timedelta(hours=23))


def test_two_trips_next_to_each_other_are_cut_by_the_dates_not_by_the_order_of_the_list() -> None:
    """R18-C4: the first range that contains a moment's date used to decide its day."""
    shanghai = TzRange.model_validate({"from": "2026-06-01", "to": "2026-06-30", "tz": SHANGHAI})
    tokyo = TzRange.model_validate({"from": "2026-07-01", "to": "2026-07-31", "tz": "Asia/Tokyo"})
    source = SourceTime(CHICAGO, [shanghai, tokyo])
    moment = utc(2026, 6, 30, 15, 30)  # 23:30 on 30 June in Shanghai, 00:30 on 1 July in Tokyo
    assert source.local_date(moment) == date(2026, 7, 1)
    start, end = source.day_bounds(date(2026, 7, 1))
    assert start <= moment < end


def test_the_replay_scan_and_the_day_loader_read_the_same_lines(services: Services) -> None:
    """R18-C4: the audit's case - after a full replay no day is "changed" any more."""
    services.settings.time.source_timezone_ranges = [SUMMER]
    writer = MessageWriter(services)
    moment, number = datetime(2025, 5, 30, tzinfo=UTC), 0
    while moment < datetime(2025, 9, 3, tzinfo=UTC):
        if moment < datetime(2025, 6, 3, tzinfo=UTC) or moment > datetime(2025, 8, 29, tzinfo=UTC):
            writer.add(moment, number % 2 == 0, "text", f"合成消息{number}")
        moment += timedelta(hours=2)
        number += 1
    writer.store()
    clock = MemoryClock.from_services(services)
    stats = {d.day: d for d in scan_days(services, clock, ReplayRequest(), TokenEstimator())}
    loaded: list[str] = []
    store = MemoryStore(services)
    for day, stat in sorted(stats.items()):
        lines = real_day_lines(services, clock, day)
        assert (len(lines), lines_hash(lines)) == (stat.lines, stat.input_hash), day
        loaded.extend(line.ref for line in lines)
        store.mark_replayed(
            day,
            input_hash=lines_hash(lines),
            lines=len(lines),
            facts_added=0,
            followups_added=0,
            summary_version=None,
            batch_id=None,
        )
    assert len(loaded) == len(set(loaded)) == sum(s.lines for s in stats.values())
    again = scan_days(services, clock, ReplayRequest(), TokenEstimator())
    assert {d.day: d.state for d in again if d.state != "done"} == {}


ranges = st.lists(
    st.tuples(
        st.dates(min_value=date(2025, 1, 1), max_value=date(2026, 12, 31)),
        st.integers(0, 40),
        st.sampled_from(ZONES),
    ).map(
        lambda t: TzRange.model_validate(
            {"from": t[0], "to": t[0] + timedelta(days=t[1]), "tz": t[2]}
        )
    ),
    max_size=4,
)


@settings(max_examples=300, deadline=None)
@given(
    default=st.sampled_from(ZONES),
    trips=ranges,
    moments=st.lists(
        st.integers(
            int(datetime(2024, 12, 25, tzinfo=UTC).timestamp()),
            int(datetime(2027, 1, 5, tzinfo=UTC).timestamp()),
        ).map(lambda seconds: datetime.fromtimestamp(seconds, UTC)),
        min_size=1,
        max_size=20,
    ),
)
def test_every_moment_belongs_to_exactly_one_day(
    default: str, trips: list[TzRange], moments: list[datetime]
) -> None:
    """R18-C4 (hypothesis): random trips (overlapping ones too), zones and moments."""
    calendar = MemoryClock(
        SourceTime(default, trips), BotTimeService(ManualClock(), lambda: CHICAGO)
    )
    for moment in moments:
        day = calendar.real_date(moment)
        start, end = calendar.real_bounds(day)
        assert start <= moment < end, (moment, day, start, end)
        for neighbour in (day - timedelta(days=1), day + timedelta(days=1)):
            low, high = calendar.real_bounds(neighbour)
            assert not low <= moment < high, (moment, neighbour)
        assert calendar.real_bounds(day - timedelta(days=1))[1] == start  # the days fit together
        assert calendar.real_bounds(day + timedelta(days=1))[0] == end
