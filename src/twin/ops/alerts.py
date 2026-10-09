"""Alert hook and its database sink (R-ARCH-003, R-ARCH-004, R-OPS-004).

Round 00 defines the hook (:class:`AlertSink`) that the job worker and the task
supervisor call when something needs attention, and a sink that records alerts in
the ``alerts`` table.  Delivery by e-mail and Windows notification, rate
limiting and the category list belong to round 12 and plug in behind the same
protocol.  Alert details never contain chat content.
"""

from __future__ import annotations

from typing import Any, Protocol

from twin.clock import Clock
from twin.ops.logging import get_logger
from twin.storage.db import Database
from twin.storage.models import ALERT_SEVERITIES, Alert

log = get_logger("twin.alerts")


class AlertSink(Protocol):
    """Receives operational alerts."""

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None: ...


class DbAlertSink:
    """Stores alerts in the ``alerts`` table and logs a one-line summary."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None:
        if severity not in ALERT_SEVERITIES:
            raise ValueError(f"unknown alert severity {severity!r}")
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            session.add(
                Alert(
                    category=category,
                    severity=severity,
                    title=title[:200],
                    detail=detail,
                    dedup_key=dedup_key,
                    created_at=now,
                    updated_at=now,
                )
            )
        log.warning("alert", category=category, severity=severity, alert_title=title[:200])
