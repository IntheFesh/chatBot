"""What the memory looked like at a moment: the one definition of "visible at t" (R-MEM-010)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.memory import (
    fact_record,
    followup_record,
    memory_clock,
    summary_record,
    utc,
)
from twin.memory.records import LifelineRecord
from twin.memory.visible import (
    BotEra,
    date_score,
    fact_visible,
    followup_visible,
    lifeline_visible,
    occurrence_offset,
    project_fact,
    project_followup,
    summary_visible,
)

T = utc(2026, 3, 10, 18)
NO_BOT = BotEra(None)
BOT_ONLINE = BotEra(utc(2026, 3, 8))


# ------------------------------------------------------------------------ facts


def test_a_fact_is_known_strictly_before_the_moment() -> None:
    fact = fact_record(known_at=T)
    assert not fact_visible(fact, T, NO_BOT)  # proved by the very message at t: not known before it
    assert fact_visible(fact, T + timedelta(seconds=1), NO_BOT)
    assert not fact_visible(fact_record(known_at=T + timedelta(hours=1)), T, NO_BOT)


def test_validity_bounds_are_inclusive_and_open_ends_are_unbounded() -> None:
    bounded = fact_record(
        known_at=utc(2026, 3, 1), valid_from=utc(2026, 3, 5), valid_to=utc(2026, 3, 12)
    )
    assert not fact_visible(bounded, utc(2026, 3, 4), NO_BOT)
    assert fact_visible(bounded, utc(2026, 3, 5), NO_BOT)
    assert fact_visible(bounded, utc(2026, 3, 12), NO_BOT)
    assert not fact_visible(bounded, utc(2026, 3, 12) + timedelta(seconds=1), NO_BOT)
    assert not fact_visible(bounded, utc(2026, 3, 13), NO_BOT)


def test_a_replacement_counts_from_the_moment_it_became_known() -> None:
    old = fact_record(
        known_at=utc(2026, 3, 1),
        superseded_by="F2",
        superseded_at=utc(2026, 3, 6),
        valid_to=utc(2026, 3, 6),
    )
    assert fact_visible(old, utc(2026, 3, 6), NO_BOT)  # the replacement is not known *before* t
    assert fact_visible(old, utc(2026, 3, 3), NO_BOT)
    assert not fact_visible(old, utc(2026, 3, 7), NO_BOT)


def test_rejected_candidates_are_never_visible() -> None:
    assert not fact_visible(fact_record(status="rejected"), T, NO_BOT)


def test_a_fact_of_the_future_replacement_is_projected_as_it_stood_then() -> None:
    old = fact_record(
        superseded_by="F2",
        superseded_at=utc(2026, 3, 20),
        valid_to=utc(2026, 3, 20),
    )
    seen = project_fact(old, T)
    assert (seen.superseded_by, seen.superseded_at, seen.valid_to) == (None, None, None)
    declared = fact_record(
        superseded_by="F2", superseded_at=utc(2026, 3, 20), valid_to=utc(2026, 3, 15)
    )
    assert project_fact(declared, T).valid_to == utc(2026, 3, 15)  # the fact's own bound stays
    plain = fact_record()
    assert project_fact(plain, T) is plain


@pytest.mark.parametrize("source", ["user_said", "bot_invented", "user_command"])
def test_what_exists_because_of_the_bot_is_empty_before_it_went_online(source: str) -> None:
    fact = fact_record(source=source, known_at=utc(2026, 3, 9))
    assert fact_visible(fact, T, BOT_ONLINE)
    assert not fact_visible(fact, T, NO_BOT)
    assert not fact_visible(fact, T, BotEra(T + timedelta(days=1)))  # the bot is not online yet
    assert fact_visible(fact_record(source="real_record", known_at=utc(2026, 3, 1)), T, NO_BOT)


def test_the_bot_era_starts_strictly_after_going_online() -> None:
    era = BotEra(utc(2026, 3, 8))
    assert not era.reached(utc(2026, 3, 8))
    assert era.reached(utc(2026, 3, 8, 12, 1))
    assert not BotEra(None).reached(utc(2030, 1, 1))


# --------------------------------------------------------------------- summaries


def test_a_summary_is_visible_from_the_next_local_day_on(clock: ManualClock) -> None:
    calendar = memory_clock(clock)
    day = date(2026, 3, 10)
    start, end = calendar.real_bounds(day)
    summary = summary_record(local_date=day, utc_start=start, utc_end=end)
    assert not summary_visible(summary, utc(2026, 3, 10, 18), NO_BOT, calendar)  # the same day
    assert not summary_visible(summary, end - timedelta(seconds=1), NO_BOT, calendar)
    assert summary_visible(summary, end, NO_BOT, calendar)  # the next local day begins
    assert summary_visible(summary, utc(2026, 3, 15), NO_BOT, calendar)


def test_the_local_day_of_the_moment_decides_not_the_utc_day(clock: ManualClock) -> None:
    """01:00 UTC on the 11th is still the evening of the 10th in Chicago: the 10th is not over."""
    calendar = memory_clock(clock)
    start, end = calendar.real_bounds(date(2026, 3, 10))
    summary = summary_record(local_date=date(2026, 3, 10), utc_start=start, utc_end=end)
    evening = datetime(2026, 3, 11, 1, 0, tzinfo=T.tzinfo)
    assert not summary_visible(summary, evening, NO_BOT, calendar)


def test_old_versions_of_a_summary_and_bot_summaries_before_the_bot_are_hidden(
    clock: ManualClock,
) -> None:
    calendar = memory_clock(clock)
    long_ago = utc(2026, 3, 20)
    assert summary_visible(summary_record(), long_ago, NO_BOT, calendar)
    assert not summary_visible(summary_record(is_current=False), long_ago, NO_BOT, calendar)
    bot = summary_record(scope="bot")
    assert not summary_visible(bot, long_ago, NO_BOT, calendar)
    assert summary_visible(bot, long_ago, BOT_ONLINE, calendar)


# --------------------------------------------------------------------- follow-ups


def test_a_follow_up_closed_later_is_open_at_the_moment() -> None:
    created = utc(2026, 3, 9)
    closed = followup_record(
        created_at=created, status="done", closed_at=utc(2026, 3, 11), close_reason="mentioned"
    )
    assert followup_visible(closed, T, NO_BOT)
    shown = project_followup(closed, T)
    assert (shown.status, shown.closed_at, shown.close_reason) == ("open", None, None)
    assert project_followup(followup_record(), T).status == "open"


def test_a_follow_up_is_not_visible_before_it_exists_or_after_it_was_closed() -> None:
    assert not followup_visible(followup_record(created_at=T), T, NO_BOT)
    assert not followup_visible(followup_record(created_at=utc(2026, 3, 11)), T, NO_BOT)
    gone = followup_record(created_at=utc(2026, 3, 1), status="done", closed_at=utc(2026, 3, 10))
    assert not followup_visible(gone, T, NO_BOT)
    assert followup_visible(gone, utc(2026, 3, 10), NO_BOT)  # closed *at* t: still open before it


def test_follow_ups_of_the_bot_conversation_need_the_bot_era() -> None:
    from_bot = followup_record(origin="bot_session", created_at=utc(2026, 3, 9))
    assert not followup_visible(from_bot, T, NO_BOT)
    assert followup_visible(from_bot, T, BOT_ONLINE)


# ----------------------------------------------------------------------- life line


def test_the_life_line_is_empty_before_the_bot_and_after_an_entry_was_invalidated() -> None:
    def event(**fields: object) -> LifelineRecord:
        base: dict[str, object] = {
            "id": "L1",
            "rev": 1,
            "local_date": date(2026, 3, 9),
            "timezone": "America/Chicago",
            "start_local": "09:00",
            "end_local": "10:00",
            "activity": "去图书馆",
            "place": None,
            "mood": None,
            "detail": None,
            "source": "plan",
            "status": "active",
            "consistency_checked_at": None,
            "invalidated_at": None,
            "invalidated_by": None,
            "fact_id": None,
            "created_at": utc(2026, 3, 9),
            "updated_at": utc(2026, 3, 9),
        }
        base.update(fields)
        return LifelineRecord(**base)  # type: ignore[arg-type]

    assert not lifeline_visible(event(), T, NO_BOT)
    assert lifeline_visible(event(), T, BOT_ONLINE)
    assert not lifeline_visible(event(created_at=utc(2026, 3, 11)), T, BOT_ONLINE)
    invalid = event(status="invalidated", invalidated_at=utc(2026, 3, 11))
    assert lifeline_visible(invalid, T, BOT_ONLINE)  # it was still believed at t
    assert not lifeline_visible(event(invalidated_at=utc(2026, 3, 9, 12)), T, BOT_ONLINE)


# ------------------------------------------------------------------ date relevance


@pytest.mark.parametrize(
    ("event_day", "recurrence", "today", "offset"),
    [
        (date(2026, 3, 10), "none", date(2026, 3, 10), 0),
        (date(2026, 3, 11), "none", date(2026, 3, 10), 1),
        (date(2026, 3, 9), "none", date(2026, 3, 10), -1),
        (date(1999, 3, 11), "yearly", date(2026, 3, 10), 1),  # a birthday: every year
        (date(1999, 3, 10), "yearly", date(2031, 3, 10), 0),
        (date(1999, 12, 31), "yearly", date(2026, 1, 1), -1),  # across New Year
        (date(1999, 1, 1), "yearly", date(2026, 12, 31), 1),
        (date(2024, 2, 29), "yearly", date(2026, 2, 28), 0),  # 29 February keeps on the 28th
        (date(2026, 1, 15), "monthly", date(2026, 3, 14), 1),
        (date(2026, 1, 31), "monthly", date(2026, 4, 30), 0),  # a short month's last day
        (date(2026, 1, 1), "monthly", date(2026, 12, 31), 1),
        (date(2026, 6, 1), "none", date(2026, 3, 10), (date(2026, 6, 1) - date(2026, 3, 10)).days),
    ],
)
def test_the_distance_to_an_anniversary_follows_its_recurrence(
    event_day: date, recurrence: str, today: date, offset: int
) -> None:
    assert occurrence_offset(event_day, recurrence, today) == offset


def test_the_score_of_a_date_peaks_today_and_fades_with_distance() -> None:
    scores = [date_score(offset) for offset in (0, 1, -1, 2, 7, 8, -2, 30)]
    assert scores[0] == 1.0 > scores[1] > scores[2] > 0
    assert scores[3] > scores[4] > 0
    assert scores[5:] == [0.0, 0.0, 0.0]
