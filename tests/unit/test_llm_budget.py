"""Budget levels, alerts and degradation (R-LLM-008)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from twin.config.settings import BudgetConfig
from twin.llm.budget import (
    BudgetEvent,
    BudgetLevel,
    BudgetManager,
    NoActivatedStyleBackend,
    StyleBackendStatus,
)
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.types import CostBreakdown, LedgerTag, Purpose, Usage
from twin.schedule.time_service import BotTimeService
from twin.storage.db import Database

NOON_CHICAGO = datetime(2026, 10, 9, 17, 0, tzinfo=UTC)  # 12:00 CDT on a Friday


class StyleStatus:
    def __init__(self, available: bool) -> None:
        self.ready = available

    def available(self) -> bool:
        return self.ready


class Rig:
    """A budget manager with a ledger, recording alerts and a manual clock."""

    def __init__(
        self,
        db: Database,
        clock: ManualClock,
        budget: BudgetConfig | None = None,
        *,
        style: StyleBackendStatus | None = None,
        ttl_s: float = 0.0,
        examples_k: int = 8,
    ) -> None:
        self.db = db
        self.clock = clock
        clock.set_time(NOON_CHICAGO)
        self.time = BotTimeService(clock, lambda: "America/Chicago")
        self.ledger = LedgerStore(db, clock, self.time)
        self.alerts = RecordingAlerts()
        self.config = budget or BudgetConfig()
        self.examples_k = examples_k
        self.style = style
        self.ttl_s = ttl_s
        self.manager = self.build()
        self.spent_today = 0.0

    def build(self) -> BudgetManager:
        return BudgetManager(
            self.config,
            examples_k=self.examples_k,
            ledger=self.ledger,
            time_service=self.time,
            clock=self.clock,
            db=self.db,
            alerts=self.alerts,
            style=self.style,
            ttl_s=self.ttl_s,
        )

    def spend(self, usd: float, *, tag: LedgerTag | None = None) -> None:
        self.ledger.record(
            LedgerRecord(
                provider="deepseek",
                model="deepseek-flash",
                purpose="reply",
                usage=Usage(prompt_tokens=1, completion_tokens=1, cache_miss_tokens=1),
                cost=CostBreakdown(usd, 0.0, 0.0, True, 1.0),
                thinking=False,
                latency_ms=1,
                at=self.clock.now_utc(),
                tag=tag or LedgerTag(),
            )
        )

    def reach(self, total_today: float) -> list[BudgetEvent]:
        """Spend so that today's total is ``total_today``; returns the events produced."""
        self.spend(total_today - self.spent_today)
        self.spent_today = total_today
        return self.manager.note_spend()


@pytest.fixture
def rig(db: Database, clock: ManualClock) -> Rig:
    return Rig(db, clock)


# ------------------------------------------------------------------------ levels


def test_levels_change_at_exactly_the_configured_ratios(rig: Rig) -> None:
    expectations = [
        (0.50, 0),
        (0.79, 0),
        (0.99, 0),
        (1.00, 1),
        (1.24, 1),
        (1.25, 2),
        (1.49, 2),
        (1.50, 3),
        (1.99, 3),
        (2.00, 4),
        (9.00, 4),
    ]
    for total, level in expectations:
        rig.reach(total)
        assert rig.manager.current_level() == level, total
        assert rig.manager.status().level == level


def test_the_monthly_budget_can_decide_the_level_on_its_own(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock, BudgetConfig(daily_usd=100.0, monthly_usd=15.0))
    rig.reach(14.0)
    assert rig.manager.current_level() == 0
    rig.reach(15.0)
    status = rig.manager.status()
    assert status.level == 1 and status.daily_level == 0 and status.monthly_level == 1
    assert status.monthly_ratio == pytest.approx(1.0)
    assert status.daily_ratio == pytest.approx(0.15)


def test_one_time_spending_never_counts(rig: Rig) -> None:
    rig.spend(50.0, tag=LedgerTag("one_time", "memory-replay"))
    rig.manager.note_spend()
    assert rig.manager.current_level() == 0
    assert rig.manager.status().daily_spent == 0.0
    assert rig.alerts.alerts == []


def test_a_zero_budget_switches_that_period_off(db: Database, clock: ManualClock) -> None:
    rig = Rig(db, clock, BudgetConfig(daily_usd=0.0, monthly_usd=0.0))
    rig.reach(100.0)
    assert rig.manager.current_level() == 0 and rig.manager.status().daily_ratio == 0.0


def test_a_new_local_day_resets_the_daily_level(rig: Rig) -> None:
    rig.reach(2.5)
    assert rig.manager.current_level() == 4
    rig.clock.set_time(NOON_CHICAGO + timedelta(days=1))  # next local day, same month
    rig.spent_today = 0.0
    assert rig.manager.current_level() == 0
    assert rig.manager.status().daily_spent == 0.0


def test_the_day_boundary_is_the_local_midnight_of_the_bot_time_zone(
    db: Database, clock: ManualClock
) -> None:
    zone = ["America/Chicago"]
    rig = Rig(db, clock)
    rig.time = BotTimeService(clock, lambda: zone[0])
    rig.ledger = LedgerStore(db, clock, rig.time)
    rig.manager = rig.build()
    clock.set_time(datetime(2026, 10, 9, 4, 30, tzinfo=UTC))  # 23:30 on the 8th in Chicago
    rig.spend(1.0)
    assert rig.manager.note_spend() and rig.manager.current_level() == 1
    clock.set_time(datetime(2026, 10, 9, 5, 30, tzinfo=UTC))  # 00:30 on the 9th in Chicago
    assert rig.manager.current_level() == 0
    zone[0] = "Asia/Shanghai"  # 13:30 on the 9th in Shanghai: the earlier spend is today again
    assert rig.manager.current_level() == 1


# ------------------------------------------------------------------ alerts, events


def test_alerts_fire_once_for_the_80_percent_mark_and_for_each_level(rig: Rig) -> None:
    assert rig.reach(0.79) == [] and rig.alerts.alerts == []
    events = rig.reach(0.80)
    assert [e.kind for e in events] == ["alert_ratio"]
    assert len(rig.alerts.alerts) == 1 and "80%" in rig.alerts.alerts[0].title
    assert rig.alerts.alerts[0].category == "budget"
    rig.reach(0.9)
    assert len(rig.alerts.alerts) == 1  # not repeated

    events = rig.reach(1.0)
    assert [(e.kind, e.old_level, e.new_level) for e in events] == [("level_up", 0, 1)]
    for total, entered in ((1.25, 2), (1.5, 3), (2.0, 4)):
        events = rig.reach(total)
        assert [(e.kind, e.new_level) for e in events] == [("level_up", entered)]
    titles = [alert.title for alert in rig.alerts.alerts]
    assert len(titles) == 5  # ratio mark + levels 1..4
    assert sum("degradation level" in t for t in titles) == 4
    severities = [alert.severity for alert in rig.alerts.alerts]
    assert severities == ["warning", "warning", "warning", "critical", "critical"]
    rig.reach(3.0)  # still level 4: nothing new
    assert len(rig.alerts.alerts) == 5


def test_a_jump_over_several_levels_alerts_for_each_level_but_emits_one_event(rig: Rig) -> None:
    events = rig.reach(2.5)
    assert [(e.kind, e.old_level, e.new_level) for e in events if e.kind != "alert_ratio"] == [
        ("level_up", 0, 4)
    ]
    keys = [a.dedup_key for a in rig.alerts.alerts]
    assert len(keys) == 5 and len(set(keys)) == 5
    assert all(a.detail is not None and "period" in a.detail for a in rig.alerts.alerts)


def test_dropping_back_emits_a_level_down_event_without_an_alert(rig: Rig) -> None:
    rig.reach(1.6)
    alerts_before = len(rig.alerts.alerts)
    seen: list[BudgetEvent] = []
    rig.manager.subscribe(seen.append)
    rig.clock.set_time(NOON_CHICAGO + timedelta(days=1))
    rig.spent_today = 0.0
    assert rig.manager.current_level() == 0
    assert [(e.kind, e.old_level, e.new_level) for e in seen] == [("level_down", 3, 0)]
    assert len(rig.alerts.alerts) == alerts_before


def test_the_next_day_alerts_again(rig: Rig) -> None:
    rig.reach(1.0)
    first_day = len(rig.alerts.alerts)
    rig.clock.set_time(NOON_CHICAGO + timedelta(days=1))
    rig.spent_today = 0.0
    rig.manager.current_level()
    rig.reach(1.0)
    assert len(rig.alerts.alerts) == 2 * first_day


def test_a_restart_does_not_repeat_alerts(rig: Rig) -> None:
    rig.reach(1.3)
    count = len(rig.alerts.alerts)
    restarted = rig.build()
    restarted.note_spend()
    assert len(rig.alerts.alerts) == count
    assert restarted.current_level() == 2


def test_subscribers_receive_events(rig: Rig) -> None:
    seen: list[BudgetEvent] = []
    rig.manager.subscribe(seen.append)
    rig.reach(1.0)
    assert [e.kind for e in seen] == ["alert_ratio", "level_up"]
    assert all(e.at == NOON_CHICAGO for e in seen)


def test_status_is_cached_until_the_ttl_or_a_forced_refresh(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock, ttl_s=5.0)
    assert rig.manager.current_level() == 0
    rig.spend(3.0)
    assert rig.manager.current_level() == 0  # cached
    clock.tick(5.1)
    assert rig.manager.current_level() == 4
    rig.spend(1.0)
    assert rig.manager.note_spend() is not None
    assert rig.manager.status().daily_spent == pytest.approx(4.0)


# ------------------------------------------------------------------ what is allowed


def test_replies_are_always_allowed_and_other_work_is_held_back_in_order(rig: Rig) -> None:
    for total in (0.0, 1.0, 1.25, 1.5, 2.0, 10.0):
        rig.reach(total)
        assert rig.manager.allow(Purpose.REPLY)
        assert rig.manager.allow("reply")
    rig.reach(0.0 + 10.0)  # level 4
    assert not any(rig.manager.allow(p) for p in Purpose if p is not Purpose.REPLY)


def test_proactive_work_stops_at_level_three_and_background_work_at_level_four(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock)
    rig.reach(1.25)  # level 2
    assert all(rig.manager.allow(p) for p in Purpose)
    rig.reach(1.5)  # level 3
    assert not rig.manager.allow(Purpose.PROACTIVE) and not rig.manager.allow(Purpose.PLAN)
    assert rig.manager.allow(Purpose.SUMMARY) and rig.manager.allow(Purpose.EXTRACT)
    rig.reach(2.0)  # level 4
    assert not rig.manager.allow(Purpose.SUMMARY) and not rig.manager.allow(Purpose.CAPTION)
    assert rig.manager.allow(Purpose.REPLY)


def test_limits_shrink_level_by_level(db: Database, clock: ManualClock) -> None:
    rig = Rig(db, clock, examples_k=8)
    rows = {}
    for total in (0.0, 1.0, 1.25, 1.5, 2.0):
        rig.reach(total)
        rows[rig.manager.current_level()] = rig.manager.limits()
    zero, one, two, three, four = (rows[i] for i in range(5))
    assert (zero.examples_k, zero.memory_budget_factor) == (8, 1.0)
    assert zero.chat_thinking_allowed and zero.planner_thinking_allowed and zero.proactive_allowed
    assert not one.chat_thinking_allowed and one.planner_thinking_allowed and one.examples_k == 8
    assert (two.examples_k, two.memory_budget_factor) == (3, 0.5) and two.proactive_allowed
    assert not three.proactive_allowed and not three.planner_thinking_allowed
    assert (three.examples_k, three.memory_budget_factor) == (3, 0.5)
    assert four.examples_k == 2 and four.memory_budget_factor == 0.25
    assert BudgetLevel(four.level) is BudgetLevel.STYLE_OR_MINIMAL


def test_examples_never_exceed_the_configured_number(db: Database, clock: ManualClock) -> None:
    rig = Rig(db, clock, examples_k=2)
    rig.reach(1.25)
    assert rig.manager.limits().examples_k == 2
    rig.reach(2.0)
    assert rig.manager.limits().examples_k == 2
    one = Rig(db, clock, examples_k=1)
    one.reach(9.0)
    assert one.manager.limits().examples_k == 1


@pytest.mark.parametrize(
    ("style", "prefer_style", "minimal"),
    [
        (None, False, True),  # nothing activated yet: DeepSeek with a minimal context
        (StyleStatus(False), False, True),  # activated but not passed / not healthy
        (StyleStatus(True), True, False),  # passed the gate and healthy: switch backend
    ],
)
def test_level_four_uses_the_style_model_only_when_it_is_ready(
    db: Database,
    clock: ManualClock,
    style: StyleStatus | None,
    prefer_style: bool,
    minimal: bool,
) -> None:
    rig = Rig(db, clock, style=style)
    rig.reach(2.0)
    limits = rig.manager.limits()
    assert limits.level == 4
    assert limits.prefer_style_backend is prefer_style
    assert limits.minimal_context is minimal
    assert not limits.chat_thinking_allowed
    assert rig.manager.allow("reply")


def test_below_level_four_the_style_backend_is_never_preferred(
    db: Database, clock: ManualClock
) -> None:
    rig = Rig(db, clock, style=StyleStatus(True))
    rig.reach(1.5)
    limits = rig.manager.limits()
    assert limits.level == 3 and not limits.prefer_style_backend and not limits.minimal_context


def test_the_style_status_can_be_replaced_later(db: Database, clock: ManualClock) -> None:
    rig = Rig(db, clock)
    rig.reach(2.0)
    assert rig.manager.limits().minimal_context
    rig.manager.set_style_status(StyleStatus(True))
    assert rig.manager.limits().prefer_style_backend
    assert NoActivatedStyleBackend().available() is False


# ------------------------------------------------------------------------ config


def test_degrade_ratios_are_validated() -> None:
    assert BudgetConfig().degrade_ratios == [1.0, 1.25, 1.5, 2.0]
    for bad in ([1.0, 1.5, 2.0], [1.0, 1.0, 1.5, 2.0], [2.0, 1.5, 1.25, 1.0], [0.0, 1.0, 2.0, 3.0]):
        with pytest.raises(ValidationError):
            BudgetConfig(degrade_ratios=bad)
    custom = BudgetConfig(degrade_ratios=[0.9, 1.0, 1.1, 1.2])
    assert custom.degrade_ratios[0] == 0.9
