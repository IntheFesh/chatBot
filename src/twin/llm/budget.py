"""Budget levels and graceful degradation (R-LLM-008).

Spending on the ``daily`` account is compared with the daily and the monthly budget (days and
months follow the bot's local calendar, :class:`~twin.schedule.time_service.TimeService`;
``one_time`` batch spending never counts, R-LLM-014).  The larger of the two ratios decides the
level, using ``budget.degrade_ratios`` (default 1.0, 1.25, 1.5, 2.0):

======  =====================================================================================
level   effect (cumulative)
======  =====================================================================================
0       normal
1       chat thinking is switched off
2       retrieved examples 8 -> 3, memory context budget halved
3       proactive messages (and their planner) are paused
4       switch to the style model backend if an activated model passed the launch gate and is
        healthy (:class:`StyleBackendStatus`, round 14); otherwise stay on DeepSeek with a
        minimal context and no thinking.  Background work on the daily account is held back too.
======  =====================================================================================

Replying to the user is never refused (``allow("reply")`` is always true).  An alert is raised
once when a period reaches ``budget.alert_ratio`` and once for every level entered; the "already
told" marks live in the ``settings`` table so a restart does not repeat them, and they are keyed
by day and month so they start afresh with the next period.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from enum import IntEnum
from typing import Protocol

from twin.clock import Clock
from twin.config.settings import BudgetConfig
from twin.llm.ledger import LedgerStore
from twin.llm.types import Purpose
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.schedule.time_service import TimeService
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.llm.budget")

NOTIFIED_KEY = "budget.notified"
LEVEL_KEY = "budget.level"
CACHE_TTL_S = 5.0
MAX_EVENT_LOG = 200
LEVEL_REDUCED_EXAMPLES = 3
LEVEL_MINIMAL_EXAMPLES = 2


class BudgetLevel(IntEnum):
    NORMAL = 0
    NO_CHAT_THINKING = 1
    REDUCED_CONTEXT = 2
    PROACTIVE_PAUSED = 3
    STYLE_OR_MINIMAL = 4


class StyleBackendStatus(Protocol):
    """Answers "may the style model take over?" (round 14 provides the real answer)."""

    def available(self) -> bool:
        """True if an activated style model passed the launch gate (R-SRV-005) and is healthy."""
        ...


class NoActivatedStyleBackend:
    """In force until a style model is activated: the style backend is never available."""

    def available(self) -> bool:
        return False


@dataclass(frozen=True)
class BudgetLimits:
    """What the current level allows the reply and planning code to use."""

    level: int
    examples_k: int
    memory_budget_factor: float
    chat_thinking_allowed: bool
    planner_thinking_allowed: bool
    proactive_allowed: bool
    prefer_style_backend: bool
    minimal_context: bool


@dataclass(frozen=True)
class BudgetStatus:
    day: date
    daily_spent: float
    daily_budget: float
    monthly_spent: float
    monthly_budget: float
    daily_level: int
    monthly_level: int

    @property
    def level(self) -> int:
        return max(self.daily_level, self.monthly_level)

    @property
    def daily_ratio(self) -> float:
        return self.daily_spent / self.daily_budget if self.daily_budget > 0 else 0.0

    @property
    def monthly_ratio(self) -> float:
        return self.monthly_spent / self.monthly_budget if self.monthly_budget > 0 else 0.0


@dataclass(frozen=True)
class BudgetEvent:
    """Something worth telling other components about: a level change or the 80% mark."""

    kind: str  # "level_up" | "level_down" | "alert_ratio"
    old_level: int
    new_level: int
    period: str  # "daily" | "monthly"
    ratio: float
    at: datetime


class BudgetGate(Protocol):
    """What the DeepSeek client needs from the budget."""

    def allow(self, purpose: str) -> bool: ...

    def limits(self) -> BudgetLimits: ...

    def note_spend(self) -> list[BudgetEvent]: ...


class BudgetManager:
    """Computes the degradation level from the ledger and announces changes."""

    def __init__(
        self,
        budget: BudgetConfig,
        *,
        examples_k: int,
        ledger: LedgerStore,
        time_service: TimeService,
        clock: Clock,
        db: Database,
        alerts: AlertSink | None = None,
        style: StyleBackendStatus | None = None,
        ttl_s: float = CACHE_TTL_S,
    ) -> None:
        self._config = budget
        self._examples_k = examples_k
        self._ledger = ledger
        self._time = time_service
        self._clock = clock
        self._db = db
        self._alerts = alerts
        self._style = style or NoActivatedStyleBackend()
        self._ttl_s = ttl_s
        self._subscribers: list[Callable[[BudgetEvent], None]] = []
        self._cached: BudgetStatus | None = None
        self._cached_at = 0.0
        self._last_level: int | None = None
        self._event_log: list[BudgetEvent] = []
        # ledger writes finish on worker threads while the event loop asks for the level
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ subscribe

    def subscribe(self, callback: Callable[[BudgetEvent], None]) -> None:
        """Call ``callback`` for every :class:`BudgetEvent` (level changes, 80% mark)."""
        self._subscribers.append(callback)

    def set_style_status(self, style: StyleBackendStatus) -> None:
        self._style = style

    # --------------------------------------------------------------------- status

    def _level_for(self, ratio: float) -> int:
        return sum(1 for threshold in self._config.degrade_ratios if ratio >= threshold)

    def compute(self) -> BudgetStatus:
        """Read the ledger and work out the levels (no side effects)."""
        today = self._time.local_date()
        daily_spent = self._ledger.spent_on_day(today)
        monthly_spent = self._ledger.spent_in_month(today)
        daily_budget = self._config.daily_usd
        monthly_budget = self._config.monthly_usd
        daily_ratio = daily_spent / daily_budget if daily_budget > 0 else 0.0
        monthly_ratio = monthly_spent / monthly_budget if monthly_budget > 0 else 0.0
        return BudgetStatus(
            day=today,
            daily_spent=daily_spent,
            daily_budget=daily_budget,
            monthly_spent=monthly_spent,
            monthly_budget=monthly_budget,
            daily_level=self._level_for(daily_ratio),
            monthly_level=self._level_for(monthly_ratio),
        )

    def status(self, *, force: bool = False) -> BudgetStatus:
        """Current status; cached for a few seconds and recomputed on a new local day."""
        with self._lock:
            now = self._clock.monotonic()
            cached = self._cached
            if (
                cached is None
                or force
                or now - self._cached_at >= self._ttl_s
                or cached.day != self._time.local_date()
            ):
                cached = self.compute()
                self._cached = cached
                self._cached_at = now
                self._announce(cached)
            return cached

    def current_level(self) -> int:
        return self.status().level

    # ----------------------------------------------------------------- decisions

    def allow(self, purpose: str) -> bool:
        """May a ``daily``-account call with this purpose go out at the current level?"""
        level = self.current_level()
        if purpose == Purpose.REPLY:
            return True
        if level >= BudgetLevel.STYLE_OR_MINIMAL:
            return False
        if level >= BudgetLevel.PROACTIVE_PAUSED:
            return Purpose(purpose) not in (Purpose.PLAN, Purpose.PROACTIVE)
        return True

    def limits(self) -> BudgetLimits:
        level = self.current_level()
        style_ready = level >= BudgetLevel.STYLE_OR_MINIMAL and self._style.available()
        minimal = level >= BudgetLevel.STYLE_OR_MINIMAL and not style_ready
        if level >= BudgetLevel.STYLE_OR_MINIMAL:
            examples = min(LEVEL_MINIMAL_EXAMPLES, self._examples_k)
            memory = 0.25
        elif level >= BudgetLevel.REDUCED_CONTEXT:
            examples = min(LEVEL_REDUCED_EXAMPLES, self._examples_k)
            memory = 0.5
        else:
            examples = self._examples_k
            memory = 1.0
        return BudgetLimits(
            level=level,
            examples_k=examples,
            memory_budget_factor=memory,
            chat_thinking_allowed=level < BudgetLevel.NO_CHAT_THINKING,
            planner_thinking_allowed=level < BudgetLevel.PROACTIVE_PAUSED,
            proactive_allowed=level < BudgetLevel.PROACTIVE_PAUSED,
            prefer_style_backend=style_ready,
            minimal_context=minimal,
        )

    # ------------------------------------------------------------------- events

    def note_spend(self) -> list[BudgetEvent]:
        """Recompute after money was spent; returns the events this produced."""
        with self._lock:
            start = len(self._event_log)
            self.status(force=True)
            return self._event_log[start:]

    def _load_notified(self) -> dict[str, bool]:
        with self._db.session() as session:
            raw = get_setting(session, NOTIFIED_KEY, {})
        return {str(k): bool(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def _announce(self, status: BudgetStatus) -> None:
        """Raise alerts and events for what ``status`` newly shows."""
        now = self._clock.now_utc()
        if self._last_level is None:
            with self._db.session() as session:
                stored = get_setting(session, LEVEL_KEY, 0)
            self._last_level = int(stored) if isinstance(stored, int) else 0
        notified = self._load_notified()
        day_key = status.day.isoformat()
        month_key = status.day.strftime("%Y-%m")
        current_keys = {day_key, month_key}
        # forget marks of earlier days and months
        notified = {k: v for k, v in notified.items() if k.split("|", 1)[0] in current_keys}
        fresh: list[tuple[str, str, str, int | None, float]] = []
        for period, key, ratio, level in (
            ("daily", day_key, status.daily_ratio, status.daily_level),
            ("monthly", month_key, status.monthly_ratio, status.monthly_level),
        ):
            if ratio >= self._config.alert_ratio:
                mark = f"{key}|{period}|ratio"
                if mark not in notified:
                    notified[mark] = True
                    fresh.append((period, mark, "alert_ratio", None, ratio))
            for reached in range(1, level + 1):
                mark = f"{key}|{period}|L{reached}"
                if mark not in notified:
                    notified[mark] = True
                    fresh.append((period, mark, "level", reached, ratio))
        events: list[BudgetEvent] = []
        for period, mark, kind, entered, ratio in fresh:
            self._alert(status, period, mark, kind, entered, ratio)
            if kind == "alert_ratio":
                events.append(
                    BudgetEvent("alert_ratio", status.level, status.level, period, ratio, now)
                )
        new_level = status.level
        previous_level = self._last_level
        if new_level != previous_level:
            kind = "level_up" if new_level > previous_level else "level_down"
            period = "daily" if status.daily_level >= status.monthly_level else "monthly"
            ratio = status.daily_ratio if period == "daily" else status.monthly_ratio
            events.append(BudgetEvent(kind, previous_level, new_level, period, ratio, now))
            log.info("budget_level_changed", old=previous_level, new=new_level, period=period)
            self._last_level = new_level
        if fresh or events:
            with self._db.transaction(bump_state=False) as session:
                put_setting(
                    session, NOTIFIED_KEY, notified, clock=self._clock, record_history=False
                )
                put_setting(session, LEVEL_KEY, new_level, clock=self._clock, record_history=False)
        for event in events:
            self._event_log.append(event)
            del self._event_log[:-MAX_EVENT_LOG]
            for callback in self._subscribers:
                callback(event)

    def _alert(
        self,
        status: BudgetStatus,
        period: str,
        mark: str,
        kind: str,
        level: int | None,
        ratio: float,
    ) -> None:
        spent = status.daily_spent if period == "daily" else status.monthly_spent
        budget = status.daily_budget if period == "daily" else status.monthly_budget
        if kind == "alert_ratio":
            title = f"{period} budget at {ratio:.0%} (${spent:.2f} of ${budget:.2f})"
            severity = "warning"
        else:
            title = (
                f"{period} budget exceeded: degradation level {level} "
                f"(${spent:.2f} of ${budget:.2f})"
            )
            severity = "critical" if (level or 0) >= BudgetLevel.PROACTIVE_PAUSED else "warning"
        log.warning("budget_alert", period=period, kind=kind, level=level, ratio=round(ratio, 3))
        if self._alerts is not None:
            self._alerts.raise_alert(
                "budget",
                title,
                severity=severity,
                detail={"period": period, "ratio": round(ratio, 4), "level": level},
                dedup_key=f"budget:{mark}",
            )
