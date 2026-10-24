"""An alert sink that remembers what it was given (a test double for ``AlertSink``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordedAlert:
    category: str
    title: str
    severity: str
    detail: dict[str, Any] | None
    dedup_key: str | None


@dataclass
class RecordingAlerts:
    alerts: list[RecordedAlert] = field(default_factory=list)

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None:
        self.alerts.append(RecordedAlert(category, title, severity, detail, dedup_key))

    def categories(self) -> list[str]:
        return [alert.category for alert in self.alerts]

    def keys(self) -> list[str | None]:
        return [alert.dedup_key for alert in self.alerts]
