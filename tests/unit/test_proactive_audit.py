"""The audit of the proactive messages: what the log of the last days says (R-EVAL-005)."""

from __future__ import annotations

import pytest

from tests.support.clock import ManualClock
from tests.support.proactive_books import Books, Sent
from twin.eval.proactive_audit import Audit, audit_days, audit_recent
from twin.services import Services


@pytest.fixture
def books(services: Services, clock: ManualClock) -> Books:
    return Books(services, clock)


def audit(books: Books, days: int = 7, *, today: bool = False) -> Audit:
    return audit_recent(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        books.time,
        books.clock,
        days=days,
        include_today=today,
    )


def test_a_week_that_kept_every_rule_is_compliant_and_counted_by_the_local_hour(
    books: Books,
) -> None:
    books.good_week()
    result = audit(books)
    assert result.compliant and result.complete and result.streak == 7
    assert (result.first_day, result.last_day) == (books.ago(7), books.ago(1))
    assert [day.sent for day in result.days] == [3] * 7
    assert result.sent == 21 and result.deep_sleep == 0
    assert result.spacing_violations == 0 and result.chase_violations == 0
    assert result.edge_max_week == 0 and result.edge_ok
    assert result.hours == {10: 7, 15: 7, 20: 7}
    assert all(day.problems == [] and day.compliant for day in result.days)


def test_only_completed_days_count_unless_today_is_asked_for(books: Books) -> None:
    books.good_week()
    books.day(books.today, (Sent(9, 0),))
    assert audit(books).last_day == books.ago(1)
    now = audit(books, 3, today=True)
    assert now.last_day == books.today and now.days[-1].sent == 1


def test_an_audit_covers_at_least_one_day(books: Books) -> None:
    with pytest.raises(ValueError, match="at least one day"):
        audit(books, 0)


def test_a_message_in_deep_sleep_is_a_violation_whatever_else_is_right(books: Books) -> None:
    books.good_week()
    books.row(books.ago(3), 3, 30, kind="share", outcome="sent", state="deep_sleep")
    result = audit(books)
    bad = next(day for day in result.days if day.day == books.ago(3))
    assert bad.deep_sleep == 1 and not bad.compliant
    assert "深睡时段发了 1 条" in bad.problems
    assert result.deep_sleep == 1 and not result.compliant
    assert [day.compliant for day in result.days].count(False) == 1


def test_the_count_must_be_inside_the_range_the_day_was_planned_with(books: Books) -> None:
    books.good_week(days=5, last=3)
    books.day(books.ago(2), tuple(Sent(8 + 2 * n) for n in range(7)), high=6)
    books.day(books.ago(1), (), low=1, high=6)
    result = audit(books, 7).days
    assert result[-2].sent == 7 and not result[-2].count_ok  # one over the maximum
    assert result[-1].sent == 0 and not result[-1].count_ok  # none, with nothing to excuse it
    assert "发了 7 条，不在 1-6 范围内" in result[-2].problems
    assert result[0].count_ok


@pytest.mark.parametrize(
    ("reason", "excused"),
    [
        ("window_closed", True),
        ("quota_exhausted", True),
        ("channel_unavailable", True),
        ("paused", True),
        ("disabled", True),
        ("budget", True),
        ("chase_limit", True),
        ("spacing", False),
        ("deep_sleep", False),
        ("planner_declined", False),
    ],
)
def test_a_day_below_its_minimum_is_excused_only_by_what_the_schedule_cannot_help(
    books: Books, reason: str, excused: bool
) -> None:
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (), refused=(reason,))
    last = audit(books).days[-1]
    assert last.excused is excused and last.count_ok is excused
    assert last.compliant is excused


def test_a_day_above_its_maximum_is_never_excused(books: Books) -> None:
    books.good_week(days=6, last=2)
    books.day(books.ago(1), tuple(Sent(7 + n) for n in range(8)), refused=("window_closed",))
    assert not audit(books).days[-1].count_ok


def test_what_the_window_suppressed_is_counted(books: Books) -> None:
    books.good_week(days=6, last=2)
    books.day(
        books.ago(1),
        (Sent(9),),
        refused=("window_closed", "window_closed", "quota_exhausted", "spacing"),
    )
    result = audit(books)
    last = result.days[-1]
    assert last.suppressed_window == 3 and last.refused == {
        "window_closed": 2,
        "quota_exhausted": 1,
        "spacing": 1,
    }
    assert result.suppressed_window == 3


def test_a_day_with_the_switch_off_may_send_nothing_and_only_nothing(books: Books) -> None:
    books.good_week(days=5, last=3)
    books.day(books.ago(2), (), enabled=False)
    books.day(books.ago(1), (Sent(9),), enabled=False)
    result = audit(books).days
    assert result[-2].compliant and result[-2].count_ok
    assert not result[-1].count_ok and "不在 0-0 范围内" in result[-1].problems[0]


def test_the_range_of_a_day_is_the_newest_the_log_knows(books: Books) -> None:
    books.good_week(days=6, last=2)
    day = books.ago(1)
    books.day(day, (Sent(9), Sent(12)), low=1, high=6)
    books.row(day, 14, 0, kind="day", outcome="opened", low=3, high=3, enabled=True)
    last = audit(books).days[-1]
    assert (last.low, last.high) == (3, 3) and not last.count_ok


def test_two_messages_closer_than_the_spacing_are_a_violation_even_across_midnight(
    books: Books,
) -> None:
    books.good_week(days=5, last=3)
    books.day(books.ago(2), (Sent(22, 30), Sent(23, 30, chase=0)))  # an hour apart: fine
    books.day(books.ago(1), (Sent(0, 10), Sent(12, 0)), low=1)  # 40 minutes after the last night
    result = audit(books)
    assert result.days[-2].spacing_violations == 0
    assert result.days[-1].spacing_violations == 1
    assert result.spacing_violations == 1 and not result.days[-1].compliant


def test_the_spacing_is_measured_from_the_message_before_the_audited_days(books: Books) -> None:
    books.day(books.ago(8), (Sent(23, 50),))
    books.day(books.ago(7), (Sent(0, 20),))
    first = audit(books, 1)  # only one day is audited, the message before it is not
    assert first.spacing_violations == 0
    both = audit(books, 7)
    assert both.days[0].spacing_violations == 1


def test_a_chase_beyond_the_limit_is_a_violation(books: Books) -> None:
    books.good_week(days=6, last=2)
    books.day(books.ago(1), (Sent(9), Sent(11, chase=1), Sent(14, chase=2)))
    last = audit(books).days[-1]
    assert last.chase_violations == 1 and not last.compliant
    assert "1 次追发超限" in last.problems


def test_the_edge_of_sleep_is_limited_in_any_seven_days_not_per_calendar_week(
    books: Books,
) -> None:
    books.good_week()
    for back in (7, 5, 4):  # three in five days
        books.row(books.ago(back), 23, 50, kind="edge", outcome="sent", state="sleep_edge")
    result = audit(books)
    assert result.edge_max_week == 3 and not result.edge_ok
    assert not result.compliant  # every day passes by itself, the week does not
    assert all(day.compliant for day in result.days)
    assert result.to_json()["edge_max_week"] == 3


def test_two_edge_messages_a_week_are_inside_the_allowance(books: Books) -> None:
    books.good_week()
    for back in (7, 3):
        books.row(books.ago(back), 23, 50, kind="edge", outcome="sent", state="sleep_edge")
    assert audit(books).compliant


def test_a_day_the_scheduler_did_not_run_is_not_judged_and_ends_the_streak(books: Books) -> None:
    books.good_week(days=2, last=1)
    books.day(books.ago(3), (Sent(10),), opened=False)  # messages, but no opening mark
    books.good_week(days=4, last=4)
    result = audit(books)
    assert [day.observed for day in result.days] == [True] * 4 + [False, True, True]
    assert result.streak == 2 and not result.complete and result.missing_days == 5
    unobserved = result.days[4]
    assert unobserved.problems == ["程序没有运行（当天没有记录）"] and not unobserved.compliant


def test_ratings_of_the_period_are_averaged_and_older_ones_are_left_out(books: Books) -> None:
    books.good_week()
    books.rate(5, days_ago=0.1)
    books.rate(3, days_ago=2)
    books.rate(1, days_ago=9)  # before the first audited day
    result = audit(books)
    assert [r.score for r in result.ratings] == [3, 5]
    assert result.rating_mean == 4.0


def test_without_any_rating_there_is_no_mean(books: Books) -> None:
    books.good_week()
    result = audit(books)
    assert result.ratings == () and result.rating_mean is None


def test_audit_days_takes_an_explicit_period(books: Books) -> None:
    books.good_week()
    result = audit_days(
        books.log,
        books.ratings,
        books.services.settings.proactive,
        first_day=books.ago(3),
        last_day=books.ago(2),
        now=books.clock.now_utc(),
    )
    assert len(result.days) == 2 and result.complete and result.sent == 6


def test_the_audit_is_written_out_as_plain_json(books: Books) -> None:
    books.good_week()
    books.rate(4)
    data = audit(books).to_json()
    assert data["compliant"] is True and data["complete"] is True and data["streak"] == 7
    assert data["rating_count"] == 1 and data["rating_mean"] == 4.0
    assert data["hours"] == {"10": 7, "15": 7, "20": 7}
    assert len(data["days"]) == 7 and data["days"][0]["hours"] == {"10": 1, "15": 1, "20": 1}


def test_an_empty_log_has_a_streak_of_zero(books: Books) -> None:
    result = audit(books)
    assert result.streak == 0 and result.sent == 0 and result.missing_days == 7
    assert not result.complete and not result.compliant
