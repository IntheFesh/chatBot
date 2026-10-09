"""The cost ledger and the time service it relies on (R-LLM-006, R-OPS-005, R-LLM-014)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.types import CostBreakdown, LedgerTag, Usage
from twin.schedule.time_service import ConfiguredTimeService
from twin.storage.db import Database, ReadOnlyViolationError, WritePolicy, use_write_policy
from twin.storage.models import CostLedger


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_service(clock: ManualClock, zone: str = "America/Chicago") -> ConfiguredTimeService:
    return ConfiguredTimeService(clock, lambda: zone)


def entry(
    at: datetime,
    cost: float,
    *,
    purpose: str = "reply",
    tag: LedgerTag | None = None,
    hit: int = 0,
    miss: int = 100,
    completion: int = 10,
    model: str = "deepseek-flash",
) -> LedgerRecord:
    return LedgerRecord(
        provider="deepseek",
        model=model,
        purpose=purpose,
        usage=Usage(
            prompt_tokens=hit + miss,
            completion_tokens=completion,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            reasoning_tokens=3,
        ),
        cost=CostBreakdown(cost, 0.0, 0.0, True, 1.0),
        thinking=False,
        latency_ms=250,
        at=at,
        request_id="req-1",
        tag=tag or LedgerTag(),
        image_count=2,
    )


@pytest.fixture
def store(db: Database, clock: ManualClock) -> LedgerStore:
    return LedgerStore(db, clock, make_service(clock))


# --------------------------------------------------------------- time service


def test_local_dates_follow_the_current_bot_time_zone(clock: ManualClock) -> None:
    zone = ["America/Chicago"]
    service = ConfiguredTimeService(clock, lambda: zone[0])
    moment = utc(2026, 10, 9, 3, 30)  # 22:30 on the 8th in Chicago, 11:30 on the 9th in Shanghai
    assert service.local_date(moment) == date(2026, 10, 8)
    zone[0] = "Asia/Shanghai"
    assert service.local_date(moment) == date(2026, 10, 9)
    assert str(service.bot_timezone()) == "Asia/Shanghai"
    clock.set_time(moment)
    assert service.local_date() == date(2026, 10, 9)


def test_day_and_month_bounds_handle_daylight_saving_time(clock: ManualClock) -> None:
    service = make_service(clock)
    start, end = service.day_bounds_utc(date(2026, 10, 9))
    assert (start, end) == (utc(2026, 10, 9, 5), utc(2026, 10, 10, 5))
    # clocks go back on 2026-11-01: that local day has 25 hours
    start, end = service.day_bounds_utc(date(2026, 11, 1))
    assert end - start == timedelta(hours=25)
    month_start, month_end = service.month_bounds_utc(date(2026, 10, 20))
    assert (month_start, month_end) == (utc(2026, 10, 1, 5), utc(2026, 11, 1, 5))
    december_start, december_end = service.month_bounds_utc(date(2026, 12, 31))
    assert december_end == utc(2027, 1, 1, 6)
    assert december_start == utc(2026, 12, 1, 6)


def test_naive_moments_are_refused(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="naive"):
        make_service(clock).local_date(datetime(2026, 10, 9, 12))  # noqa: DTZ001


# --------------------------------------------------------------------- writing


def test_a_record_stores_every_field(store: LedgerStore, db: Database) -> None:
    ledger_id = store.record(entry(utc(2026, 10, 9, 15), 0.25, hit=40, miss=60, completion=7))
    with db.session() as session:
        row = session.execute(select(CostLedger).where(CostLedger.id == ledger_id)).scalar_one()
        assert (row.provider, row.model, row.purpose) == ("deepseek", "deepseek-flash", "reply")
        assert (row.cache_hit_tokens, row.cache_miss_tokens, row.completion_tokens) == (40, 60, 7)
        assert row.reasoning_tokens == 3 and row.image_count == 2
        assert row.cost_usd == pytest.approx(0.25) and row.peak is True
        assert row.thinking is False
        assert row.latency_ms == 250 and row.request_id == "req-1"
        assert (row.account, row.batch_id) == ("daily", None)
        assert row.at == utc(2026, 10, 9, 15)


def test_one_time_rows_carry_their_batch(store: LedgerStore, db: Database) -> None:
    tag = LedgerTag("one_time", "memory-replay-1")
    ledger_id = store.record(entry(utc(2026, 10, 9, 15), 0.5, tag=tag))
    with db.session() as session:
        row = session.get(CostLedger, ledger_id)
        assert row is not None and (row.account, row.batch_id) == ("one_time", "memory-replay-1")


def test_ledger_tags_validate_themselves() -> None:
    with pytest.raises(ValueError, match="batch id"):
        LedgerTag("one_time")
    with pytest.raises(ValueError, match="only one_time"):
        LedgerTag("daily", "b1")


def test_a_read_only_command_cannot_write_the_ledger(store: LedgerStore) -> None:
    with use_write_policy(WritePolicy(read_only=True)), pytest.raises(ReadOnlyViolationError):
        store.record(entry(utc(2026, 10, 9, 15), 0.1))


def test_recording_does_not_touch_the_state_version(store: LedgerStore, db: Database) -> None:
    from twin.storage.state import read_state_version

    with db.session() as session:
        before = read_state_version(session)
    with use_write_policy(WritePolicy(bump_state=True)):
        store.record(entry(utc(2026, 10, 9, 15), 0.1))
    with db.session() as session:
        assert read_state_version(session) == before


# --------------------------------------------------------------------- queries


def test_daily_totals_use_the_local_calendar_day(store: LedgerStore) -> None:
    store.record(entry(utc(2026, 10, 9, 4, 59), 1.0))  # 23:59 on the 8th in Chicago
    store.record(entry(utc(2026, 10, 9, 5, 0), 2.0))  # 00:00 on the 9th
    store.record(entry(utc(2026, 10, 10, 4, 59), 4.0))  # 23:59 on the 9th
    store.record(entry(utc(2026, 10, 10, 5, 0), 8.0))  # 00:00 on the 10th
    assert store.spent_on_day(date(2026, 10, 8)) == pytest.approx(1.0)
    assert store.spent_on_day(date(2026, 10, 9)) == pytest.approx(6.0)
    assert store.spent_on_day(date(2026, 10, 10)) == pytest.approx(8.0)
    assert store.spent_in_month(date(2026, 10, 9)) == pytest.approx(15.0)
    assert store.spent_in_month(date(2026, 9, 30)) == 0.0


def test_one_time_spending_is_kept_apart(store: LedgerStore) -> None:
    store.record(entry(utc(2026, 10, 9, 15), 1.0))
    store.record(entry(utc(2026, 10, 9, 16), 10.0, tag=LedgerTag("one_time", "b1")))
    store.record(entry(utc(2026, 10, 9, 17), 5.0, tag=LedgerTag("one_time", "b2")))
    day = date(2026, 10, 9)
    assert store.spent_on_day(day) == pytest.approx(1.0)
    assert store.spent_on_day(day, account="one_time") == pytest.approx(15.0)
    assert store.spent_on_day(day, account=None) == pytest.approx(16.0)
    assert store.batch_spent_usd("b1") == pytest.approx(10.0)
    assert store.batch_spent_usd("nope") == 0.0
    assert [s.key for s in store.by_day(day, day, account="one_time")] == ["2026-10-09"]


def test_summaries_by_day_month_purpose_and_model(store: LedgerStore) -> None:
    store.record(entry(utc(2026, 10, 9, 15), 1.0, purpose="reply", hit=50, miss=50))
    store.record(entry(utc(2026, 10, 9, 16), 2.0, purpose="summary", model="deepseek-v4-pro"))
    store.record(entry(utc(2026, 10, 12, 16), 3.0, purpose="reply"))
    store.record(entry(utc(2026, 11, 2, 16), 4.0, purpose="reply"))
    days = store.by_day(date(2026, 10, 1), date(2026, 10, 31))
    assert [(d.key, d.calls, round(d.cost_usd, 2)) for d in days] == [
        ("2026-10-09", 2, 3.0),
        ("2026-10-12", 1, 3.0),
    ]
    months = store.by_month(date(2026, 10, 1), date(2026, 11, 30))
    assert [(m.key, round(m.cost_usd, 2)) for m in months] == [("2026-10", 6.0), ("2026-11", 4.0)]
    start, end = utc(2026, 10, 1), utc(2026, 12, 1)
    purposes = {p.key: p for p in store.by_purpose(start, end)}
    assert set(purposes) == {"reply", "summary"} and purposes["reply"].calls == 3
    assert {m.key for m in store.by_model(start, end)} == {"deepseek-flash", "deepseek-v4-pro"}
    total = store.totals(start, end)
    assert total.calls == 4 and total.cost_usd == pytest.approx(10.0)
    assert total.reasoning_tokens == 12 and total.completion_tokens == 40
    assert store.totals(utc(2025, 1, 1), utc(2025, 2, 1)).calls == 0


def test_cache_hit_ratio_over_a_period_and_per_purpose(
    store: LedgerStore, clock: ManualClock
) -> None:
    store.record(entry(utc(2026, 10, 9, 10), 0.1, purpose="reply", hit=90, miss=10))
    store.record(entry(utc(2026, 10, 9, 11), 0.1, purpose="reply", hit=10, miss=90))
    store.record(entry(utc(2026, 10, 9, 11), 0.1, purpose="summary", hit=0, miss=100))
    start, end = utc(2026, 10, 9), utc(2026, 10, 10)
    assert store.cache_hit_ratio(start, end) == pytest.approx(100 / 300)
    assert store.cache_hit_ratio(start, end, purpose="reply") == pytest.approx(0.5)
    assert store.cache_hit_ratio(utc(2020, 1, 1), utc(2020, 1, 2)) == 0.0
    clock.set_time(utc(2026, 10, 9, 12))
    assert store.recent_cache_hit_ratio(days=1) == pytest.approx(100 / 300)
    assert store.recent_cache_hit_ratio(days=1, purpose="summary") == 0.0
