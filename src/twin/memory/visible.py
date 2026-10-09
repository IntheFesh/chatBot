"""What the memory looked like at a moment: the one definition of "visible at t" (R-MEM-010).

Everything that reads the memory for a past moment - ``memory_view(as_of=t)``, the assembler
when it is asked to build a block as of ``t``, ``AsOfView(t)`` of the training export and the
evaluation sandbox (R-TRN-013) - decides what exists at ``t`` with the functions below and
nowhere else:

* a **fact** is visible when it is active, ``known_at < t`` (strictly: a fact proved by the very
  message at ``t`` is not known *before* ``t``), valid at ``t`` (``valid_from <= t <= valid_to``)
  and not yet replaced (a replacement counts from the moment it became known);
* a **summary** is visible when its local day is earlier than the local day of ``t`` and the day
  is over (``utc_end <= t``); of several versions of a day the current one counts;
* a **follow-up** is visible when it was created before ``t`` and was still open at ``t``
  (``closed_at`` is ``None`` or ``>= t``); it is shown open, as it was then;
* **what exists only because the bot's conversation exists** - the bot's facts, the user's
  statements to the bot, the bot's summaries, its life line and the follow-ups of its
  conversation - is empty for a ``t`` before the bot went online.

Two helpers belong with it because the assembler and the view share them: the distance in days
from today to an anniversary or due date (:func:`occurrence_offset`) and the score it earns
(:func:`date_score`).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, replace
from datetime import date, datetime

from twin.memory.localdate import MemoryClock
from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord

BOT_FOLLOWUP_ORIGINS = frozenset({"bot_session", "user_command"})


@dataclass(frozen=True)
class BotEra:
    """When the bot's conversation began (``None``: it has not)."""

    online_at: datetime | None

    def reached(self, moment: datetime) -> bool:
        """True if the bot's conversation had begun before ``moment``."""
        return self.online_at is not None and self.online_at < moment


# ----------------------------------------------------------------------- facts


def fact_visible(fact: FactRecord, moment: datetime, era: BotEra) -> bool:
    if fact.status != "active" or not fact.known_at < moment:
        return False
    if fact.bot_conversation and not era.reached(moment):
        return False
    if fact.valid_from is not None and fact.valid_from > moment:
        return False
    if fact.valid_to is not None and fact.valid_to < moment:
        return False
    return not (fact.superseded_at is not None and fact.superseded_at < moment)


def project_fact(fact: FactRecord, moment: datetime) -> FactRecord:
    """The fact as it stood at ``moment``: a replacement that came later is not shown."""
    if fact.superseded_at is None or fact.superseded_at < moment:
        return fact
    valid_to = None if fact.valid_to == fact.superseded_at else fact.valid_to
    return replace(fact, superseded_by=None, superseded_at=None, valid_to=valid_to)


# -------------------------------------------------------------------- summaries


def summary_visible(
    summary: SummaryRecord, moment: datetime, era: BotEra, clock: MemoryClock
) -> bool:
    if not summary.is_current:
        return False
    if summary.scope == "bot" and not era.reached(moment):
        return False
    if summary.utc_end > moment:
        return False
    return summary.local_date < clock.date_of(summary.scope, moment)


# -------------------------------------------------------------------- follow-ups


def followup_visible(followup: FollowupRecord, moment: datetime, era: BotEra) -> bool:
    """Created before ``moment`` and still open then (see :func:`project_followup`)."""
    if not followup.created_at < moment:
        return False
    if followup.origin in BOT_FOLLOWUP_ORIGINS and not era.reached(moment):
        return False
    return followup.closed_at is None or followup.closed_at >= moment


def project_followup(followup: FollowupRecord, moment: datetime) -> FollowupRecord:
    """The follow-up as it was at ``moment``: open, whatever happened to it afterwards."""
    if followup.status == "open":
        return followup
    return replace(followup, status="open", closed_at=None, close_reason=None)


# --------------------------------------------------------------------- life line


def lifeline_visible(event: LifelineRecord, moment: datetime, era: BotEra) -> bool:
    if not era.reached(moment) or not event.created_at < moment:
        return False
    return event.invalidated_at is None or event.invalidated_at >= moment


# ------------------------------------------------------------------ date relevance


def _clamped(year: int, month: int, day: int) -> date:
    """``day`` of the month, or the month's last day when it has fewer (29 February: 28)."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def occurrence_offset(event_date: date, recurrence: str, today: date) -> int:
    """Days from ``today`` to the occurrence of the event nearest to it (negative: past).

    A yearly event repeats on its month and day, a monthly one on its day of the month; a
    one-off event happens once.  Of two equally near occurrences the coming one is chosen.
    """
    candidates: list[date]
    if recurrence == "yearly":
        candidates = [
            _clamped(year, event_date.month, event_date.day)
            for year in (today.year - 1, today.year, today.year + 1)
        ]
    elif recurrence == "monthly":
        candidates = []
        for shift in (-1, 0, 1):
            index = today.year * 12 + today.month - 1 + shift
            year, month = divmod(index, 12)
            candidates.append(_clamped(year, month + 1, event_date.day))
    else:
        candidates = [event_date]
    offsets = [(candidate - today).days for candidate in candidates]
    return min(offsets, key=lambda offset: (abs(offset), -offset))


def date_score(offset: int) -> float:
    """How much an event ``offset`` days away matters today, in ``[0, 1]``."""
    if offset == 0:
        return 1.0
    if offset == 1:
        return 0.85
    if offset == -1:
        return 0.6
    if 2 <= offset <= 7:
        return 0.5 - 0.3 * (offset - 2) / 5
    return 0.0
