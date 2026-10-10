"""The messages whose time her day fixes: greeting, meals, goodnight, follow-ups (R-PRO-004).

:func:`routine_slots` lays out the day of a plan.  Every slot is drawn from a random generator
seeded with the plan's seed and the name of the slot, so the same day always gets the same slots -
a restart, a rebuilt plan or a second call changes nothing - and each one is made at most once
per local day and zone (the table's key).

* **wake-up greeting** - once, a random moment in the window the plan allows (5-40 minutes after
  she wakes, and not within 18 hours of the last greeting; that rule was applied when the plan was
  made, R-SCH-002).
* **a meal** - around the mealtime of the plan, a random moment within ``proactive.meal_window_min``
  on either side; it happens with the probability that she opened a conversation in that window
  on such a day in her history (:func:`~twin.schedule.proactive.curve.window_share`).
* **goodnight** - 15 to 60 minutes before she falls asleep (``proactive.bedtime_lead_min``), with
  the probability of her history in that window.
* Two slots never lie closer than ``proactive.min_spacing_min``: a slot that would is moved later,
  behind the one it was too close to, and one that cannot be moved inside its own window is dropped
  (the other one is its moment).  The wake-up greeting is placed first, so it is always the first
  message of the day, and a "have you eaten" does not push in front of it.
* The slots count against the day's quota: if there are more of them than the plan leaves room
  for, the greeting stays and the others are chosen by a seeded draw.

A **follow-up** (R-MEM-006) is not part of the day.  :func:`followup_slot` makes the candidate for
one that came due: it goes out a little after the event - a fraction of the follow-up's window
that is drawn from the follow-up's id - so she does not ask about the exam during the exam.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from twin.config.settings import ProactiveConfig
from twin.memory.records import FollowupRecord
from twin.profile.activity_model import ActivityModel
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.curve import AWAKE_STATES, window_share
from twin.schedule.proactive.store import NewCandidate
from twin.schedule.proactive.types import PRIORITY, TriggerKind

FALLBACK_SHARE = 1.0 / 3.0  # a meal or goodnight happens this often when nothing is known of her
FOLLOWUP_EARLIEST = 0.10  # a follow-up goes out this share of its window after the event ...
FOLLOWUP_LATEST = 0.40  # ... to this share
MIN_WINDOW = timedelta(minutes=5)


def _stream(plan: DailyPlan, part: str) -> random.Random:
    return random.Random(f"{plan.seed}:proactive:{part}")  # noqa: S311 - a draw, not security


def _slot(
    plan: DailyPlan,
    key: str,
    kind: TriggerKind,
    planned_at: datetime,
    window_end: datetime,
    detail: dict[str, object],
) -> NewCandidate:
    return NewCandidate(
        local_date=plan.local_date,
        timezone=plan.timezone,
        key=key,
        kind=kind,
        priority=PRIORITY[kind],
        planned_at=planned_at,
        window_end=window_end,
        plan_id=plan.id,
        detail=detail,
    )


def _awake_from(plan: DailyPlan, moment: datetime, limit: datetime) -> datetime | None:
    """``moment``, or the first instant after it at which she is awake; ``None`` after ``limit``.

    A message is not planned for a time at which she sleeps: a breakfast message that falls before
    she wakes goes out when she is up, if her mealtime window is not over by then.
    """
    found = moment
    for _ in range(len(plan.segments) + 1):
        segment = plan.segment_at(found)
        if segment is None or segment.kind in AWAKE_STATES:
            return found if found < limit else None
        found = segment.end
    return None


def keep_apart(slots: Sequence[NewCandidate], spacing: timedelta) -> list[NewCandidate]:
    """``slots`` in time order, none closer than ``spacing`` to another (see the module text).

    The slots are placed in the order of their priority (the greeting first, then by time): one
    that is too close to a slot already placed is moved to just after it, and dropped if that is
    past the end of its own window.
    """
    placed: list[NewCandidate] = []
    for slot in sorted(slots, key=lambda found: (found.priority, found.planned_at)):
        planned = slot.planned_at
        moved = True
        while moved:
            moved = False
            for other in placed:
                if (
                    abs(planned - other.planned_at) < spacing
                    and planned < other.planned_at + spacing
                ):
                    planned = other.planned_at + spacing
                    moved = True
        if planned > slot.window_end:
            continue
        placed.append(slot if planned == slot.planned_at else replace(slot, planned_at=planned))
    return sorted(placed, key=lambda found: found.planned_at)


def routine_slots(
    plan: DailyPlan,
    *,
    model: ActivityModel | None,
    day_type: str,
    config: ProactiveConfig,
) -> list[NewCandidate]:
    """The fixed moments of ``plan``'s day, as many as the day's quota has room for."""
    allowance = plan.quota.for_plan
    if not plan.quota.enabled or allowance <= 0:
        return []
    zone = ZoneInfo(plan.timezone)
    slots: list[NewCandidate] = []

    window = plan.greeting
    if window.allowed and window.earliest is not None and window.latest is not None:
        rng = _stream(plan, "greeting")
        span = max(timedelta(0), window.latest - window.earliest)
        planned = window.earliest + timedelta(seconds=rng.uniform(0.0, span.total_seconds()))
        slots.append(
            _slot(
                plan,
                "greeting",
                TriggerKind.GREETING,
                planned,
                window.latest,
                {"wake": plan.wake.isoformat() if plan.wake else None},
            )
        )

    around = timedelta(minutes=config.meal_window_min)
    for meal in plan.meals:
        rng = _stream(plan, f"meal:{meal.kind}")
        chance = window_share(
            model, zone, day_type, meal.at - around, meal.at + around, fallback=FALLBACK_SHARE
        )
        if rng.random() >= chance:
            continue
        offset = rng.uniform(-around.total_seconds(), around.total_seconds())
        awake = _awake_from(plan, meal.at + timedelta(seconds=offset), meal.at + around)
        if awake is None:  # she sleeps through the whole window
            continue
        slots.append(
            _slot(
                plan,
                f"meal:{meal.kind}",
                TriggerKind.MEAL,
                awake,
                meal.at + around,
                {"meal": meal.kind, "meal_at": meal.at.isoformat()},
            )
        )

    night = plan.night
    if night is not None:
        shortest, longest = config.bedtime_lead_min
        rng = _stream(plan, "bedtime")
        chance = window_share(
            model,
            zone,
            day_type,
            night.onset - timedelta(minutes=longest),
            night.onset - timedelta(minutes=shortest),
            fallback=FALLBACK_SHARE,
        )
        if rng.random() < chance:
            lead = rng.uniform(shortest, longest)
            slots.append(
                _slot(
                    plan,
                    "bedtime",
                    TriggerKind.BEDTIME,
                    night.onset - timedelta(minutes=lead),
                    night.onset - timedelta(minutes=shortest),
                    {"onset": night.onset.isoformat()},
                )
            )

    slots = keep_apart(slots, timedelta(minutes=config.min_spacing_min))
    if len(slots) > allowance:
        keep = [slot for slot in slots if slot.kind is TriggerKind.GREETING][:allowance]
        others = [slot for slot in slots if slot not in keep]
        chosen = _stream(plan, "cap").sample(others, allowance - len(keep))
        slots = [slot for slot in slots if slot in keep or slot in chosen]
    return slots


def followup_slot(followup: FollowupRecord, *, plan: DailyPlan, now: datetime) -> NewCandidate:
    """The candidate for a follow-up that came due (see the module description)."""
    rng = random.Random(f"proactive:followup:{followup.id}")  # noqa: S311 - a draw, not security
    window = timedelta(minutes=followup.window_minutes)
    planned = followup.due_at + window * rng.uniform(FOLLOWUP_EARLIEST, FOLLOWUP_LATEST)
    end = max(followup.window_end, planned + MIN_WINDOW)
    return NewCandidate(
        local_date=plan.local_date,
        timezone=plan.timezone,
        key=f"followup:{followup.id}",
        kind=TriggerKind.FOLLOWUP,
        priority=PRIORITY[TriggerKind.FOLLOWUP],
        planned_at=max(planned, now),
        window_end=end,
        plan_id=plan.id,
        detail={"followup_id": followup.id, "due_at": followup.due_at.isoformat()},
    )
