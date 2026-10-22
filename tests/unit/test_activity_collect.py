"""What the collector of the routine model counts, message by message (R-ACT-002).

R17-40b (D-734).  The reply latency is counted by the slot in which the message she answers
arrived, for all days and for each kind of day (workday, weekend, holiday); a day of one kind
must not leak into another's distribution, and the type that counts is the one of the day the
answered message arrived on.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

from hypothesis import given
from hypothesis import strategies as st

from twin.profile.activity_collect import ActivityCollector, ActivityRaw
from twin.profile.localtime import LocalStamp
from twin.profile.units import Rec, Step

DAY = date(2026, 10, 12)  # a Monday
KINDS = ("workday", "weekend", "holiday")


def rec(number: int, *, her: bool, slot: int, kind: str, day: date = DAY) -> Rec:
    stamp = LocalStamp(day, slot * 15.0 + 1.0, slot, "America/Chicago")
    return Rec(f"m{number}", 1_700_000_000.0 + number, her, "text", "x", None, stamp, kind)


def answer(
    collector: ActivityCollector, number: int, *, arrived: Rec, latency: float, kind: str
) -> None:
    """She answers ``arrived`` after ``latency`` seconds (on a day of the type ``kind``)."""
    reply = rec(number, her=True, slot=arrived.stamp.slot, kind=kind)
    collector.feed(arrived, Step(arrived))
    collector.feed(reply, Step(reply, latency_s=latency, answered=arrived))


def collect(*replies: tuple[str, int, float]) -> ActivityRaw:
    """One reply for each ``(type of the day, slot of the message she answers, latency)``."""
    collector = ActivityCollector()
    for number, (kind, slot, latency) in enumerate(replies, start=1):
        arrived = rec(1000 + number, her=False, slot=slot, kind=kind)
        answer(collector, number, arrived=arrived, latency=latency, kind=kind)
    return collector.finish()


def test_each_kind_of_day_has_its_own_latency_counts_beside_the_all_day_one() -> None:
    raw = collect(
        ("workday", 60, 900.0),
        ("workday", 60, 900.0),
        ("weekend", 60, 20.0),
        ("weekend", 61, 25.0),
        ("holiday", 40, 7.0),
    )
    assert raw.latency == {
        60: Counter({900.0: 2, 20.0: 1}),
        61: Counter({25.0: 1}),
        40: Counter({7.0: 1}),
    }
    assert raw.latency_workday == {60: Counter({900.0: 2})}
    assert raw.latency_weekend == {60: Counter({20.0: 1}), 61: Counter({25.0: 1})}
    assert raw.latency_holiday == {40: Counter({7.0: 1})}


def test_the_type_is_that_of_the_message_she_answers_not_of_her_reply() -> None:
    """A message that arrived on Friday night and is answered on Saturday is a workday's."""
    collector = ActivityCollector()
    friday = rec(1, her=False, slot=90, kind="workday", day=date(2026, 10, 9))
    reply = rec(2, her=True, slot=2, kind="weekend", day=date(2026, 10, 10))
    collector.feed(friday, Step(friday))
    collector.feed(reply, Step(reply, latency_s=600.0, answered=friday))
    raw = collector.finish()
    assert raw.latency_workday == {90: Counter({600.0: 1})}
    assert raw.latency_weekend == {} and raw.latency_holiday == {}


def test_a_message_that_answers_nothing_adds_no_latency_and_a_collector_starts_empty() -> None:
    collector = ActivityCollector()
    opening = rec(1, her=True, slot=30, kind="weekend")
    collector.feed(opening, Step(opening, initiation=True))
    raw = collector.finish()
    assert raw.latency == {} and raw.latency_workday == {}
    assert raw.latency_weekend == {} and raw.latency_holiday == {}
    assert raw.initiations == 1 and raw.her_messages == 1


def test_a_day_type_the_model_does_not_know_counts_for_all_days_only() -> None:
    raw = collect(("festival", 60, 30.0))
    assert raw.latency == {60: Counter({30.0: 1})}
    assert raw.latency_workday == raw.latency_weekend == raw.latency_holiday == {}


def test_the_latency_is_rounded_to_whole_seconds_like_the_all_day_one() -> None:
    raw = collect(("weekend", 60, 19.6), ("weekend", 60, 20.4))
    assert raw.latency_weekend == {60: Counter({20.0: 2})} == raw.latency


@given(
    st.lists(
        st.tuples(
            st.sampled_from([*KINDS, "other"]),
            st.integers(min_value=0, max_value=95),
            st.integers(min_value=0, max_value=3000).map(float),
        ),
        max_size=40,
    )
)
def test_the_three_kinds_of_day_partition_the_known_replies_of_the_all_day_counts(
    replies: list[tuple[str, int, float]],
) -> None:
    raw = collect(*replies)

    def total(split: dict[int, Counter[float]]) -> int:
        return sum(sum(counter.values()) for counter in split.values())

    known = [r for r in replies if r[0] in KINDS]
    assert total(raw.latency) == len(replies)
    assert total(raw.latency_workday) + total(raw.latency_weekend) + total(
        raw.latency_holiday
    ) == len(known)
    splits = (raw.latency_workday, raw.latency_weekend, raw.latency_holiday)
    for kind, split in zip(KINDS, splits, strict=True):
        expected: dict[int, Counter[float]] = {}
        for found, slot, latency in replies:
            if found == kind:
                expected.setdefault(slot, Counter())[latency] += 1
        assert split == expected
