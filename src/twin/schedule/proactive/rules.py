"""The hard constraints of a proactive message (R-PRO-003), as one pure function.

:func:`check` answers "may this kind of message go out now?" from a :class:`Situation` - plain
facts the scheduler gathers (her state, the platform window, what was sent today, how many
messages went unanswered).  It is called when a candidate is considered **and again right before
the first bubble goes out** (after the planner and the writing have taken their time), and it is
the only place the constraints live.  The order is the order of the reasons in the log:

1. proactive messages are off (``/主动 关``) or paused (``/暂停``);
2. no day plan - nobody can tell whether she is asleep;
3. **deep sleep: never** (no setting reaches this line, R-PRO-003);
4. the edge of her sleep: only the edge message, and only ``edge_of_sleep_weekly_max`` a week;
5. the budget has paused proactive messages (R-LLM-008, level 3);
6. the platform: login, window, messages left (R-CH-008) - "suppressed by the window";
7. the spacing to the last proactive message (``min_spacing_min``);
8. a message that nobody answered: the last one has to be ``unanswered_after_min`` old before it
   counts, and at most ``max_chase`` more may follow it;
9. the day's maximum (``daily_max``), which holds for every kind including follow-ups;
10. the day's quota: once the draw of the plan is used up only follow-ups may still go out.

:func:`window_refusal` is the platform window as the session state tells it, the same rule as
:meth:`twin.channel.window.SessionWindow.refusal_reason`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from twin.channel.base import AuthState, SessionState
from twin.schedule.proactive.types import Reason, Refusal, TriggerKind


@dataclass(frozen=True)
class Situation:
    """The facts a constraint check reads (all gathered by the scheduler)."""

    now: datetime
    state: str | None  # her state: deep_sleep | sleep_edge | busy | free; None: no day plan
    enabled: bool
    paused: bool
    budget_allowed: Callable[[], bool]  # asked only if the checks before it pass
    window: Reason | None  # why the platform refuses a message now, or None
    window_detail: str | None
    last_sent_at: datetime | None  # the last proactive message that went out
    unanswered: int  # proactive messages sent since the user last wrote
    last_unanswered_at: datetime | None  # when the newest of them went out
    sent_today: int  # proactive messages of the local day so far
    daily_max: int
    spent_since_plan: int  # proactive messages since the day plan in force took effect
    allowance: int  # the share of the day's draw that belongs to that plan (``Quota.for_plan``)
    edge_sent_week: int  # messages sent at the edge of her sleep in the last seven days
    edge_weekly_max: int
    min_spacing: timedelta
    max_chase: int
    unanswered_after: timedelta
    need: int = 1  # bubbles the platform must still have room for


def window_refusal(state: SessionState, need: int = 1) -> tuple[Reason, str] | None:
    """Why the platform refuses ``need`` more bubbles now (``None``: it does not).

    The window rule of :meth:`~twin.channel.window.SessionWindow.refusal_reason` read from the
    session state: expired or never opened (no message of the user yet), the safe window over,
    too little of the count left; and the login (an unbound or logged-out channel cannot send).
    """
    if not state.bound:
        return Reason.CHANNEL_UNAVAILABLE, "unbound"
    if state.auth is not AuthState.OK:
        return Reason.CHANNEL_UNAVAILABLE, state.auth.value
    if state.expired:
        return Reason.WINDOW_CLOSED, "session_expired"
    if state.last_inbound_at is None:
        return Reason.WINDOW_CLOSED, "no_inbound_yet"
    if state.window_remaining is None or state.window_remaining <= timedelta(0):
        return Reason.WINDOW_CLOSED, "window_elapsed"
    if state.remaining_quota < max(1, need):
        return Reason.QUOTA_EXHAUSTED, f"remaining {state.remaining_quota}"
    return None


def check(kind: TriggerKind, situation: Situation) -> Refusal | None:
    """The first constraint that refuses a message of ``kind`` now, or ``None`` (see above)."""
    s = situation
    if not s.enabled:
        return Refusal(Reason.DISABLED)
    if s.paused:
        return Refusal(Reason.PAUSED)
    if s.state is None:
        return Refusal(Reason.NO_PLAN)
    if s.state == "deep_sleep":
        return Refusal(Reason.DEEP_SLEEP)
    if s.state == "sleep_edge":
        if kind is not TriggerKind.EDGE:
            return Refusal(Reason.SLEEP_EDGE)
        if s.edge_sent_week >= s.edge_weekly_max:
            return Refusal(Reason.EDGE_WEEKLY, f"{s.edge_sent_week}/{s.edge_weekly_max}")
    elif kind is TriggerKind.EDGE:
        return Refusal(Reason.STATE_CHANGED, s.state)
    if not s.budget_allowed():
        return Refusal(Reason.BUDGET)
    if s.window is not None:
        return Refusal(s.window, s.window_detail)
    if s.last_sent_at is not None and s.now - s.last_sent_at < s.min_spacing:
        return Refusal(Reason.SPACING)
    if s.unanswered >= 1:
        if s.last_unanswered_at is not None and s.now - s.last_unanswered_at < s.unanswered_after:
            return Refusal(Reason.AWAITING_REPLY)
        if s.unanswered - 1 >= s.max_chase:
            return Refusal(Reason.CHASE_LIMIT, f"{s.unanswered} unanswered")
    if s.sent_today >= s.daily_max:
        return Refusal(Reason.DAILY_MAX, f"{s.sent_today}/{s.daily_max}")
    if kind not in (TriggerKind.FOLLOWUP, TriggerKind.EDGE) and s.spent_since_plan >= s.allowance:
        return Refusal(Reason.QUOTA_SPENT, f"{s.spent_since_plan}/{s.allowance}")
    return None
