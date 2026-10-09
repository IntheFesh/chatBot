"""Off-peak policies for tests (production policy: ``twin.llm.pricing.CalendarOffPeakPolicy``)."""

from __future__ import annotations

from datetime import datetime


class AlwaysOffPeak:
    """Off-peak jobs may run at any time."""

    def allows(self, now: datetime) -> bool:
        return True


class NeverOffPeak:
    """Off-peak jobs never run (until their deadline)."""

    def allows(self, now: datetime) -> bool:
        return False
