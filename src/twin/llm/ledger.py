"""The cost ledger: one row per paid model call, and the summaries read from it.

Writes (R-LLM-006) and queries (R-OPS-005) for the ``cost_ledger`` table.  Every row belongs
to an *account* (R-LLM-014): ``daily`` rows count against the daily and monthly budget,
``one_time`` rows (batch jobs, the M0 probe) are tracked per ``batch_id`` and shown
separately.  Day and month boundaries come from a :class:`~twin.schedule.time_service.TimeService`
so they follow the bot's local calendar.  The report commands (``twin cost report``,
``/费用``) arrive in round 12 and are built on these queries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.llm.types import DAILY, CostBreakdown, LedgerTag, Usage
from twin.schedule.time_service import TimeService
from twin.storage.db import Database
from twin.storage.models import CostLedger


@dataclass(frozen=True)
class LedgerRecord:
    """What is written for one call."""

    provider: str
    model: str
    purpose: str
    usage: Usage
    cost: CostBreakdown
    thinking: bool
    latency_ms: int
    at: datetime
    request_id: str | None = None
    tag: LedgerTag = DAILY
    image_count: int = 0


@dataclass(frozen=True)
class SpendSummary:
    """Totals for a group of ledger rows."""

    key: str
    calls: int
    cost_usd: float
    cache_hit_tokens: int
    cache_miss_tokens: int
    completion_tokens: int
    reasoning_tokens: int

    @property
    def cache_hit_ratio(self) -> float:
        """Cache hits as a share of all prompt tokens (0 when there were none)."""
        prompt = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / prompt if prompt else 0.0


def _empty(key: str) -> SpendSummary:
    return SpendSummary(key, 0, 0.0, 0, 0, 0, 0)


def _add(total: SpendSummary, row: CostLedger) -> SpendSummary:
    return SpendSummary(
        total.key,
        total.calls + 1,
        total.cost_usd + row.cost_usd,
        total.cache_hit_tokens + row.cache_hit_tokens,
        total.cache_miss_tokens + row.cache_miss_tokens,
        total.completion_tokens + row.completion_tokens,
        total.reasoning_tokens + row.reasoning_tokens,
    )


class LedgerStore:
    """Reads and writes ``cost_ledger``."""

    def __init__(self, db: Database, clock: Clock, time_service: TimeService) -> None:
        self._db = db
        self._clock = clock
        self._time = time_service

    # ------------------------------------------------------------------ writes

    def record(self, entry: LedgerRecord) -> str:
        """Insert a row and return its id.  Does not bump ``settings.state_version``."""
        row = CostLedger(
            at=entry.at,
            provider=entry.provider,
            model=entry.model,
            purpose=entry.purpose,
            cache_hit_tokens=entry.usage.cache_hit_tokens,
            cache_miss_tokens=entry.usage.cache_miss_tokens,
            completion_tokens=entry.usage.completion_tokens,
            reasoning_tokens=entry.usage.reasoning_tokens,
            cost_usd=entry.cost.total_usd,
            peak=entry.cost.peak,
            thinking=entry.thinking,
            latency_ms=entry.latency_ms,
            request_id=entry.request_id,
            account=entry.tag.account,
            batch_id=entry.tag.batch_id,
            image_count=entry.image_count,
            created_at=self._clock.now_utc(),
            updated_at=self._clock.now_utc(),
        )
        with self._db.transaction(bump_state=False) as session:
            session.add(row)
            ledger_id = row.id
        return ledger_id

    # ----------------------------------------------------------------- queries

    def _rows(
        self,
        session: Session,
        start: datetime,
        end: datetime,
        *,
        account: str | None,
        purpose: str | None = None,
        batch_id: str | None = None,
    ) -> list[CostLedger]:
        stmt = select(CostLedger).where(CostLedger.at >= start, CostLedger.at < end)
        if account is not None:
            stmt = stmt.where(CostLedger.account == account)
        if purpose is not None:
            stmt = stmt.where(CostLedger.purpose == purpose)
        if batch_id is not None:
            stmt = stmt.where(CostLedger.batch_id == batch_id)
        return list(session.execute(stmt.order_by(CostLedger.at)).scalars())

    def total_usd(
        self,
        start: datetime,
        end: datetime,
        *,
        account: str | None = "daily",
        purpose: str | None = None,
    ) -> float:
        """Sum of ``cost_usd`` in ``[start, end)``; ``account=None`` adds both accounts."""
        stmt = select(func.coalesce(func.sum(CostLedger.cost_usd), 0.0)).where(
            CostLedger.at >= start, CostLedger.at < end
        )
        if account is not None:
            stmt = stmt.where(CostLedger.account == account)
        if purpose is not None:
            stmt = stmt.where(CostLedger.purpose == purpose)
        with self._db.session() as session:
            return float(session.execute(stmt).scalar_one())

    def spent_on_day(self, day: date, *, account: str | None = "daily") -> float:
        start, end = self._time.day_bounds_utc(day)
        return self.total_usd(start, end, account=account)

    def spent_in_month(self, day: date, *, account: str | None = "daily") -> float:
        start, end = self._time.month_bounds_utc(day)
        return self.total_usd(start, end, account=account)

    def batch_spent_usd(self, batch_id: str) -> float:
        """Actual spend of a one-time batch so far."""
        stmt = select(func.coalesce(func.sum(CostLedger.cost_usd), 0.0)).where(
            CostLedger.batch_id == batch_id
        )
        with self._db.session() as session:
            return float(session.execute(stmt).scalar_one())

    def _grouped(
        self,
        start: datetime,
        end: datetime,
        key_of: Callable[[CostLedger], str],
        *,
        account: str | None,
        purpose: str | None = None,
    ) -> list[SpendSummary]:
        groups: dict[str, SpendSummary] = {}
        with self._db.session() as session:  # rows expire when the session ends: fold them here
            for row in self._rows(session, start, end, account=account, purpose=purpose):
                key = key_of(row)
                groups[key] = _add(groups.get(key, _empty(key)), row)
        return [groups[key] for key in sorted(groups)]

    def by_day(
        self, first: date, last: date, *, account: str | None = "daily"
    ) -> list[SpendSummary]:
        """One summary per local day with spending, ``first`` to ``last`` inclusive."""
        start, _ = self._time.day_bounds_utc(first)
        _, end = self._time.day_bounds_utc(last)
        zone = self._time.bot_timezone()
        return self._grouped(
            start, end, lambda row: row.at.astimezone(zone).date().isoformat(), account=account
        )

    def by_month(
        self, first: date, last: date, *, account: str | None = "daily"
    ) -> list[SpendSummary]:
        """One summary per local month with spending."""
        start, _ = self._time.month_bounds_utc(first)
        _, end = self._time.month_bounds_utc(last)
        zone = self._time.bot_timezone()
        return self._grouped(
            start,
            end,
            lambda row: row.at.astimezone(zone).strftime("%Y-%m"),
            account=account,
        )

    def by_purpose(
        self, start: datetime, end: datetime, *, account: str | None = None
    ) -> list[SpendSummary]:
        """Spending per ``purpose`` in ``[start, end)``."""
        return self._grouped(start, end, lambda row: row.purpose, account=account)

    def by_model(
        self, start: datetime, end: datetime, *, account: str | None = None
    ) -> list[SpendSummary]:
        return self._grouped(start, end, lambda row: row.model, account=account)

    def totals(
        self,
        start: datetime,
        end: datetime,
        *,
        account: str | None = None,
        purpose: str | None = None,
    ) -> SpendSummary:
        """All rows in the range as one summary (key ``"total"``)."""
        groups = self._grouped(start, end, lambda row: "total", account=account, purpose=purpose)
        return groups[0] if groups else _empty("total")

    def cache_hit_ratio(
        self, start: datetime, end: datetime, *, purpose: str | None = None
    ) -> float:
        """Share of prompt tokens served from the cache in ``[start, end)`` (all accounts)."""
        return self.totals(start, end, purpose=purpose).cache_hit_ratio

    def recent_cache_hit_ratio(self, *, days: int = 1, purpose: str | None = None) -> float:
        """Cache hit ratio over the last ``days`` days up to now (shown by ``/费用``)."""
        end = self._clock.now_utc()
        return self.cache_hit_ratio(end - timedelta(days=days), end, purpose=purpose)
