"""The hard constraints of a proactive message as one pure function (R-PRO-003)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from twin.channel.base import AuthState, SessionState
from twin.channel.window import SessionWindow
from twin.schedule.proactive.rules import Situation, check, window_refusal
from twin.schedule.proactive.types import (
    OUTSIDE_REASONS,
    REASON_LABELS,
    WINDOW_REASONS,
    Reason,
    TriggerKind,
)

NOW = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)


def situation(**changes: object) -> Situation:
    base = Situation(
        now=NOW,
        state="free",
        enabled=True,
        paused=False,
        budget_allowed=lambda: True,
        window=None,
        window_detail=None,
        last_sent_at=None,
        unanswered=0,
        last_unanswered_at=None,
        sent_today=0,
        daily_max=6,
        spent_since_plan=0,
        allowance=4,
        edge_sent_week=0,
        edge_weekly_max=2,
        min_spacing=timedelta(minutes=60),
        max_chase=1,
        unanswered_after=timedelta(minutes=30),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def reason_of(kind: TriggerKind, **changes: object) -> Reason | None:
    refusal = check(kind, situation(**changes))
    return refusal.reason if refusal else None


def test_a_free_moment_with_nothing_in_the_way_is_allowed_for_every_kind() -> None:
    for kind in TriggerKind:
        state = "sleep_edge" if kind is TriggerKind.EDGE else "free"
        assert reason_of(kind, state=state) is None, kind


def test_deep_sleep_refuses_every_kind_and_no_setting_reaches_it() -> None:
    for kind in TriggerKind:
        assert reason_of(kind, state="deep_sleep") is Reason.DEEP_SLEEP, kind
    # all the other switches in their most generous position: still refused
    generous = {
        "state": "deep_sleep",
        "daily_max": 99,
        "allowance": 99,
        "min_spacing": timedelta(0),
        "max_chase": 99,
        "edge_weekly_max": 99,
    }
    assert reason_of(TriggerKind.FOLLOWUP, **generous) is Reason.DEEP_SLEEP
    assert reason_of(TriggerKind.EDGE, **generous) is Reason.DEEP_SLEEP


def test_at_the_edge_of_sleep_only_the_edge_message_goes_and_only_within_its_weekly_allowance() -> (
    None
):
    for kind in TriggerKind:
        if kind is not TriggerKind.EDGE:
            assert reason_of(kind, state="sleep_edge") is Reason.SLEEP_EDGE, kind
    assert reason_of(TriggerKind.EDGE, state="sleep_edge") is None
    assert reason_of(TriggerKind.EDGE, state="sleep_edge", edge_sent_week=1) is None
    assert reason_of(TriggerKind.EDGE, state="sleep_edge", edge_sent_week=2) is Reason.EDGE_WEEKLY
    # an edge message while she is awake is nonsense
    assert reason_of(TriggerKind.EDGE, state="free") is Reason.STATE_CHANGED


def test_without_a_day_plan_nothing_goes_out() -> None:
    assert reason_of(TriggerKind.FOLLOWUP, state=None) is Reason.NO_PLAN


def test_switched_off_and_paused() -> None:
    assert reason_of(TriggerKind.FOLLOWUP, enabled=False) is Reason.DISABLED
    assert reason_of(TriggerKind.MEAL, paused=True) is Reason.PAUSED
    assert reason_of(TriggerKind.MEAL, enabled=False, paused=True) is Reason.DISABLED


def test_the_budget_pauses_proactive_messages_and_is_asked_only_when_it_matters() -> None:
    asked: list[int] = []

    def allowed() -> bool:
        asked.append(1)
        return False

    assert reason_of(TriggerKind.SHARE, budget_allowed=allowed) is Reason.BUDGET
    assert asked == [1]
    asked.clear()
    assert reason_of(TriggerKind.SHARE, budget_allowed=allowed, state="deep_sleep") is (
        Reason.DEEP_SLEEP
    )
    assert asked == []  # a refusal before it never asks


def test_the_platform_window_refuses_with_its_own_reasons() -> None:
    assert reason_of(TriggerKind.MEAL, window=Reason.WINDOW_CLOSED) is Reason.WINDOW_CLOSED
    assert reason_of(TriggerKind.MEAL, window=Reason.QUOTA_EXHAUSTED) is Reason.QUOTA_EXHAUSTED
    assert reason_of(TriggerKind.MEAL, window=Reason.CHANNEL_UNAVAILABLE) is (
        Reason.CHANNEL_UNAVAILABLE
    )


def test_two_messages_are_at_least_the_spacing_apart() -> None:
    assert reason_of(TriggerKind.MEAL, last_sent_at=NOW - timedelta(minutes=59)) is Reason.SPACING
    assert reason_of(TriggerKind.MEAL, last_sent_at=NOW - timedelta(minutes=60)) is None
    assert (
        reason_of(TriggerKind.FOLLOWUP, last_sent_at=NOW - timedelta(minutes=5)) is Reason.SPACING
    )


def test_an_unanswered_message_counts_after_thirty_minutes_and_allows_max_chase_more() -> None:
    long_ago = NOW - timedelta(hours=2)
    # one unanswered message: this is the first chase, allowed when max_chase is 1
    assert reason_of(TriggerKind.SILENCE, unanswered=1, last_unanswered_at=long_ago) is None
    # two unanswered: the chase has been used
    assert (
        reason_of(TriggerKind.SILENCE, unanswered=2, last_unanswered_at=long_ago)
        is Reason.CHASE_LIMIT
    )
    # no chase at all when max_chase is 0
    assert (
        reason_of(TriggerKind.SILENCE, unanswered=1, last_unanswered_at=long_ago, max_chase=0)
        is Reason.CHASE_LIMIT
    )
    # a message sent a moment ago is not unanswered yet
    assert (
        reason_of(
            TriggerKind.SILENCE,
            unanswered=1,
            last_unanswered_at=NOW - timedelta(minutes=10),
            min_spacing=timedelta(minutes=5),
        )
        is Reason.AWAITING_REPLY
    )


def test_the_daily_maximum_holds_for_every_kind_the_day_quota_only_for_the_random_ones() -> None:
    for kind in (TriggerKind.FOLLOWUP, TriggerKind.MEAL, TriggerKind.SHARE):
        assert reason_of(kind, sent_today=6, daily_max=6) is Reason.DAILY_MAX, kind
    assert reason_of(TriggerKind.MEAL, spent_since_plan=4, allowance=4) is Reason.QUOTA_SPENT
    assert reason_of(TriggerKind.SHARE, spent_since_plan=4, allowance=4) is Reason.QUOTA_SPENT
    # a follow-up may still go out once the day's draw is used up (R-PRO-004)
    assert reason_of(TriggerKind.FOLLOWUP, spent_since_plan=4, allowance=4) is None
    assert reason_of(TriggerKind.FOLLOWUP, spent_since_plan=9, allowance=4, sent_today=5) is None


def test_the_order_of_the_checks_decides_which_reason_is_logged() -> None:
    everything = {
        "enabled": True,
        "paused": True,
        "state": "free",
        "window": Reason.WINDOW_CLOSED,
        "last_sent_at": NOW,
        "sent_today": 6,
        "spent_since_plan": 9,
    }
    assert reason_of(TriggerKind.MEAL, **everything) is Reason.PAUSED
    assert reason_of(TriggerKind.MEAL, **(everything | {"paused": False})) is Reason.WINDOW_CLOSED
    rest = everything | {"paused": False, "window": None}
    assert reason_of(TriggerKind.MEAL, **rest) is Reason.SPACING
    rest = rest | {"last_sent_at": None}
    assert reason_of(TriggerKind.MEAL, **rest) is Reason.DAILY_MAX


def state_of(**changes: object) -> SessionState:
    base = SessionState(
        auth=AuthState.OK,
        bound=True,
        last_inbound_at=NOW - timedelta(hours=1),
        outbound_since_inbound=2,
        expired=False,
        remaining_quota=5,
        window_remaining=timedelta(hours=21),
        has_context_token=True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_the_window_rule_of_the_session_state_matches_the_session_window() -> None:
    """``window_refusal`` is :meth:`SessionWindow.refusal_reason` read from the session state."""
    window = SessionWindow(window_h=22, quota=8)
    cases = [
        ("no_inbound_yet", None),
        ("window_elapsed", NOW - timedelta(hours=23)),
        (None, NOW - timedelta(hours=3)),
    ]
    for expected, inbound in cases:
        window = SessionWindow(window_h=22, quota=8)
        if inbound is not None:
            window.on_inbound(inbound)
        assert window.refusal_reason(NOW) == expected
        state = SessionState(
            auth=AuthState.OK,
            bound=True,
            last_inbound_at=window.last_inbound_at,
            outbound_since_inbound=window.outbound_since_inbound,
            expired=window.expired,
            remaining_quota=window.remaining_quota(),
            window_remaining=window.window_remaining(NOW),
            has_context_token=True,
        )
        found = window_refusal(state)
        assert (found[1] if found else None) == expected
    spent = SessionWindow(window_h=22, quota=3)
    spent.on_inbound(NOW - timedelta(hours=1))
    spent.on_outbound(3)
    assert spent.refusal_reason(NOW) == "quota_exhausted"
    spent.mark_expired(NOW)
    assert spent.refusal_reason(NOW) == "session_expired"


def test_window_refusal_reasons_and_the_login() -> None:
    assert window_refusal(state_of()) is None
    assert window_refusal(state_of(bound=False)) == (Reason.CHANNEL_UNAVAILABLE, "unbound")
    assert window_refusal(state_of(auth=AuthState.NEEDS_RELOGIN))[0] is Reason.CHANNEL_UNAVAILABLE  # type: ignore[index]
    assert window_refusal(state_of(expired=True)) == (Reason.WINDOW_CLOSED, "session_expired")
    assert window_refusal(state_of(last_inbound_at=None)) == (
        Reason.WINDOW_CLOSED,
        "no_inbound_yet",
    )
    assert window_refusal(state_of(window_remaining=timedelta(0))) == (
        Reason.WINDOW_CLOSED,
        "window_elapsed",
    )
    assert window_refusal(state_of(remaining_quota=0))[0] is Reason.QUOTA_EXHAUSTED  # type: ignore[index]
    assert window_refusal(state_of(remaining_quota=1), need=2)[0] is Reason.QUOTA_EXHAUSTED  # type: ignore[index]
    assert window_refusal(state_of(remaining_quota=2), need=2) is None


def test_every_reason_has_a_label_and_the_window_groups_are_reasons() -> None:
    assert {reason.value for reason in Reason} <= set(REASON_LABELS)
    assert WINDOW_REASONS <= OUTSIDE_REASONS
    assert Reason.DEEP_SLEEP not in OUTSIDE_REASONS  # a deep-sleep message is never excused


@pytest.mark.parametrize("kind", list(TriggerKind))
def test_the_priority_order_is_followup_routine_silence_share(kind: TriggerKind) -> None:
    from twin.schedule.proactive.types import PRIORITY

    assert PRIORITY[TriggerKind.FOLLOWUP] < PRIORITY[TriggerKind.GREETING]
    assert PRIORITY[TriggerKind.MEAL] < PRIORITY[TriggerKind.SILENCE] < PRIORITY[TriggerKind.SHARE]
    assert kind in PRIORITY
