"""Conversation window tracking (R-CH-008).

The platform lets a bot talk only while the user has recently written (how long and how many
messages is what the M0 probe measures).  :class:`SessionWindow` keeps the numbers:

* ``last_inbound_at`` and ``outbound_since_inbound``: every inbound message resets the count;
* ``remaining_quota() = quota - outbound_since_inbound`` (never below zero);
* ``can_send_proactive(now, n)``: inside the window, not expired and at least ``n`` left;
* ``expired``: set when the platform refuses a send for window or quota reasons; it stays
  set (nothing is retried) until the next inbound message.

The window and the quota are constructor arguments, so tests and the local console channel
can simulate any platform limit.  The class holds no clock and does no I/O: callers pass
``now`` and persist :meth:`SessionWindow.state` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from twin.clock import ensure_aware


@dataclass(frozen=True)
class WindowState:
    """The persisted part of a window (JSON-friendly, UTC)."""

    last_inbound_at: datetime | None = None
    outbound_since_inbound: int = 0
    expired: bool = False
    expired_at: datetime | None = None
    last_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_inbound_at": _iso(self.last_inbound_at),
            "outbound_since_inbound": self.outbound_since_inbound,
            "expired": self.expired,
            "expired_at": _iso(self.expired_at),
            "last_error": self.last_error,
        }

    def after_inbound(self, at: datetime) -> WindowState:
        """The state once a message from the user arrived at ``at`` (count and expiry reset)."""
        at = ensure_aware(at)
        if self.last_inbound_at is not None and at < self.last_inbound_at:
            return self  # an older message arriving late does not move the window back
        return replace(
            self,
            last_inbound_at=at,
            outbound_since_inbound=0,
            expired=False,
            expired_at=None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WindowState:
        if not data:
            return cls()
        return cls(
            last_inbound_at=_parse(data.get("last_inbound_at")),
            outbound_since_inbound=int(data.get("outbound_since_inbound", 0)),
            expired=bool(data.get("expired", False)),
            expired_at=_parse(data.get("expired_at")),
            last_error=data.get("last_error"),
        )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else ensure_aware(value).isoformat()


def _parse(value: str | None) -> datetime | None:
    return None if not value else ensure_aware(datetime.fromisoformat(value))


class SessionWindow:
    """Window and quota accounting for one conversation."""

    def __init__(self, *, window_h: float, quota: int, state: WindowState | None = None) -> None:
        if window_h <= 0:
            raise ValueError("window_h must be positive")
        if quota < 0:
            raise ValueError("quota must not be negative")
        self.window_h = window_h
        self.quota = quota
        self._state = state or WindowState()

    @property
    def state(self) -> WindowState:
        return self._state

    @property
    def last_inbound_at(self) -> datetime | None:
        return self._state.last_inbound_at

    @property
    def outbound_since_inbound(self) -> int:
        return self._state.outbound_since_inbound

    @property
    def expired(self) -> bool:
        return self._state.expired

    # ------------------------------------------------------------- updates

    def on_inbound(self, at: datetime) -> None:
        """A message from the user arrived: the window restarts and the count resets."""
        self._state = self._state.after_inbound(at)

    def on_outbound(self, count: int = 1) -> None:
        """``count`` bubbles were sent (or may have been sent)."""
        if count < 0:
            raise ValueError("count must not be negative")
        self._state = replace(
            self._state, outbound_since_inbound=self._state.outbound_since_inbound + count
        )

    def mark_expired(
        self, now: datetime, *, code: int | None = None, errmsg: str | None = None
    ) -> None:
        """The platform refused a send because of the session or its limits (R-CH-008)."""
        now = ensure_aware(now)
        self._state = replace(
            self._state,
            expired=True,
            expired_at=now,
            last_error={"at": now.isoformat(), "code": code, "errmsg": errmsg, "kind": "window"},
        )

    def record_error(
        self, now: datetime, *, kind: str, code: int | None = None, errmsg: str | None = None
    ) -> None:
        """Remember the latest send problem that did not expire the window."""
        now = ensure_aware(now)
        self._state = replace(
            self._state,
            last_error={"at": now.isoformat(), "code": code, "errmsg": errmsg, "kind": kind},
        )

    def reset(self) -> None:
        """Forget everything (a new login or a changed binding)."""
        self._state = WindowState()

    # -------------------------------------------------------------- queries

    def remaining_quota(self) -> int:
        """``quota - outbound_since_inbound``, never below zero."""
        return max(0, self.quota - self._state.outbound_since_inbound)

    def window_remaining(self, now: datetime) -> timedelta | None:
        """Time left in the window, or ``None`` before the first inbound message."""
        last = self._state.last_inbound_at
        if last is None:
            return None
        return timedelta(hours=self.window_h) - (ensure_aware(now) - last)

    def within_window(self, now: datetime) -> bool:
        left = self.window_remaining(now)
        return left is not None and left > timedelta(0)

    def refusal_reason(self, now: datetime, n: int = 1) -> str | None:
        """Why ``n`` more bubbles may not be sent now, or ``None`` if they may."""
        if n < 1:
            raise ValueError("n must be at least 1")
        if self._state.expired:
            return "session_expired"
        if self._state.last_inbound_at is None:
            return "no_inbound_yet"
        if not self.within_window(now):
            return "window_elapsed"
        if self.remaining_quota() < n:
            return "quota_exhausted"
        return None

    def can_send_proactive(self, now: datetime, n: int = 1) -> bool:
        """Inside the window, not expired and at least ``n`` bubbles of quota left."""
        return self.refusal_reason(now, n) is None
