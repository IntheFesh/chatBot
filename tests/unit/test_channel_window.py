"""The conversation window: last inbound, bubble count, quota, expiry (R-CH-008)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from twin.channel.window import SessionWindow, WindowState

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def window(window_h: float = 22, quota: int = 8) -> SessionWindow:
    return SessionWindow(window_h=window_h, quota=quota)


def test_nothing_can_be_sent_before_the_first_inbound_message() -> None:
    w = window()
    assert w.can_send_proactive(T0) is False
    assert w.refusal_reason(T0) == "no_inbound_yet"
    assert w.window_remaining(T0) is None
    assert w.remaining_quota() == 8


@pytest.mark.parametrize(
    ("hours", "allowed"),
    [
        (0, True),
        (1, True),
        (6, True),
        (12, True),
        (20, True),
        (21.99, True),
        (22, False),  # the safe window is a strict "less than"
        (23, False),
        (25, False),
        (26, False),
    ],
)
def test_the_safe_window_is_checked_from_zero_to_twenty_six_hours(
    hours: float, allowed: bool
) -> None:
    w = window()
    w.on_inbound(T0)
    now = T0 + timedelta(hours=hours)
    assert w.can_send_proactive(now) is allowed
    assert w.refusal_reason(now) == (None if allowed else "window_elapsed")


def test_remaining_quota_counts_down_to_zero_and_never_below() -> None:
    w = window(quota=3)
    w.on_inbound(T0)
    assert w.remaining_quota() == 3
    for expected in (2, 1, 0, 0):
        w.on_outbound()
        assert w.remaining_quota() == expected
    assert w.outbound_since_inbound == 4
    assert w.refusal_reason(T0) == "quota_exhausted"


def test_the_requested_number_of_bubbles_must_fit_in_the_remaining_quota() -> None:
    w = window(quota=8)
    w.on_inbound(T0)
    w.on_outbound(5)
    assert w.can_send_proactive(T0, 3) is True
    assert w.can_send_proactive(T0, 4) is False
    with pytest.raises(ValueError, match="at least 1"):
        w.can_send_proactive(T0, 0)


def test_an_inbound_message_resets_the_count_and_the_expiry() -> None:
    w = window(quota=2)
    w.on_inbound(T0)
    w.on_outbound(2)
    w.mark_expired(T0, code=-2, errmsg="refused")
    assert w.expired and w.can_send_proactive(T0) is False
    later = T0 + timedelta(hours=3)
    w.on_inbound(later)
    assert not w.expired and w.outbound_since_inbound == 0
    assert w.last_inbound_at == later
    assert w.can_send_proactive(later) is True
    assert w.state.expired_at is None


def test_an_expired_window_stays_closed_inside_the_time_window() -> None:
    w = window()
    w.on_inbound(T0)
    w.mark_expired(T0 + timedelta(minutes=5), code=-2, errmsg="refused")
    assert w.refusal_reason(T0 + timedelta(minutes=6)) == "session_expired"
    assert w.state.last_error is not None and w.state.last_error["code"] == -2


def test_a_late_arriving_older_message_does_not_move_the_window_back() -> None:
    w = window()
    w.on_inbound(T0)
    w.on_outbound(2)
    w.on_inbound(T0 - timedelta(hours=1))
    assert w.last_inbound_at == T0
    assert w.outbound_since_inbound == 2


def test_window_remaining_is_negative_after_the_window() -> None:
    w = window(window_h=2)
    w.on_inbound(T0)
    assert w.window_remaining(T0 + timedelta(hours=1)) == timedelta(hours=1)
    assert w.window_remaining(T0 + timedelta(hours=3)) == timedelta(hours=-1)
    assert w.within_window(T0 + timedelta(hours=1)) is True
    assert w.within_window(T0 + timedelta(hours=3)) is False


def test_simulated_limits_can_be_injected_for_other_channels() -> None:
    w = SessionWindow(window_h=0.5, quota=1)
    w.on_inbound(T0)
    assert w.can_send_proactive(T0 + timedelta(minutes=29)) is True
    w.on_outbound()
    assert w.can_send_proactive(T0 + timedelta(minutes=29)) is False
    assert SessionWindow(window_h=1, quota=0).remaining_quota() == 0


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="window_h"):
        SessionWindow(window_h=0, quota=1)
    with pytest.raises(ValueError, match="quota"):
        SessionWindow(window_h=1, quota=-1)
    with pytest.raises(ValueError, match="negative"):
        window().on_outbound(-1)


def test_the_state_survives_a_json_round_trip_and_rejects_naive_times() -> None:
    w = window()
    w.on_inbound(T0)
    w.on_outbound(3)
    w.record_error(T0, kind="network", code=None, errmsg="x")
    restored = WindowState.from_dict(w.state.to_dict())
    assert restored == w.state
    assert WindowState.from_dict(None) == WindowState()
    with pytest.raises(ValueError, match="naive"):
        w.on_inbound(datetime(2026, 1, 1))  # noqa: DTZ001 - the point of the test
    resumed = SessionWindow(window_h=22, quota=8, state=restored)
    assert resumed.outbound_since_inbound == 3
    resumed.reset()
    assert resumed.state == WindowState()
