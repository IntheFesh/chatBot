"""The cost report: by day, purpose and model, cache hits, peak share, budget (R-OPS-005)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from twin.cli import app
from twin.config.secrets import SecretStore
from twin.config.settings import BudgetConfig
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.types import CostBreakdown, LedgerTag, Usage
from twin.ops.cost import (
    CostReport,
    CostReportError,
    build_report,
    parse_month,
    previous_month,
    render_html,
    render_text,
    report_to_json,
)
from twin.schedule.service import time_service_for
from twin.services import CliContext, Services, set_cli_context

runner = CliRunner()


def call(
    at: datetime,
    cost: float,
    *,
    purpose: str = "reply",
    model: str = "deepseek-flash",
    hit: int = 400,
    miss: int = 100,
    peak: bool = False,
    tag: LedgerTag | None = None,
) -> LedgerRecord:
    return LedgerRecord(
        provider="deepseek",
        model=model,
        purpose=purpose,
        usage=Usage(
            prompt_tokens=hit + miss,
            completion_tokens=20,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            reasoning_tokens=0,
        ),
        cost=CostBreakdown(cost, 0.0, 0.0, peak, 1.0),
        thinking=False,
        latency_ms=200,
        at=at,
        tag=tag or LedgerTag(),
    )


@pytest.fixture
def ledger(services: Services) -> LedgerStore:
    store = LedgerStore(services.db, services.clock, time_service_for(services))
    # September 2026 in Chicago (UTC-5): three days of calls
    entries = [
        call(datetime(2026, 9, 3, 15, tzinfo=UTC), 0.30, purpose="reply", peak=True),
        call(datetime(2026, 9, 3, 16, tzinfo=UTC), 0.20, purpose="reply", peak=True),
        call(datetime(2026, 9, 10, 3, tzinfo=UTC), 0.05, purpose="memory", model="deepseek-pro"),
        call(datetime(2026, 9, 10, 4, tzinfo=UTC), 0.05, purpose="memory", model="deepseek-pro"),
        call(datetime(2026, 9, 28, 12, tzinfo=UTC), 0.40, purpose="plan", hit=0, miss=500),
        # outside the month, and an account that is not the budget's
        call(datetime(2026, 8, 31, 12, tzinfo=UTC), 9.0),
        call(datetime(2026, 10, 1, 12, tzinfo=UTC), 9.0),
        call(
            datetime(2026, 9, 12, 12, tzinfo=UTC),
            1.5,
            purpose="batch",
            tag=LedgerTag("one_time", "batch-1"),
        ),
    ]
    for entry in entries:
        store.record(entry)
    return store


def september(services: Services, ledger: LedgerStore, **budget: float) -> CostReport:
    config = BudgetConfig(**{"monthly_usd": 2.0, "daily_usd": 0.4, **budget})
    return build_report(ledger, time_service_for(services), config, date(2026, 9, 1))


def test_a_month_is_added_up_by_day_purpose_and_model(
    services: Services, ledger: LedgerStore
) -> None:
    report = september(services, ledger)
    assert report.month == "2026-09" and report.zone == "America/Chicago"
    assert report.total.calls == 5 and report.total.cost_usd == pytest.approx(1.0)
    assert [d.key for d in report.days] == ["2026-09-03", "2026-09-09", "2026-09-28"]
    spent = {d.key: round(d.cost_usd, 2) for d in report.days}
    assert spent == {"2026-09-03": 0.5, "2026-09-09": 0.1, "2026-09-28": 0.4}
    assert {p.key: round(p.cost_usd, 2) for p in report.purposes} == {
        "reply": 0.5,
        "plan": 0.4,
        "memory": 0.1,
    }
    assert [p.key for p in report.purposes] == ["reply", "plan", "memory"]  # most expensive first
    assert {m.key for m in report.models} == {"deepseek-flash", "deepseek-pro"}
    assert report.models[0].key == "deepseek-flash"


def test_the_one_time_account_is_apart_from_the_budget(
    services: Services, ledger: LedgerStore
) -> None:
    report = september(services, ledger)
    assert report.one_time.calls == 1 and report.one_time.cost_usd == pytest.approx(1.5)
    assert report.total.cost_usd == pytest.approx(1.0)  # the batch is not in the total
    assert "一次性任务（不占预算）$1.50，1 次调用" in render_text(report)


def test_the_budget_comparison_and_the_days_over_the_daily_limit(
    services: Services, ledger: LedgerStore
) -> None:
    report = september(services, ledger)
    assert report.month_ratio == pytest.approx(0.5) and report.days_over_daily == 1  # 09-03: $0.50
    free = september(services, ledger, monthly_usd=0.0, daily_usd=0.0)
    assert free.month_ratio == 0.0 and free.days_over_daily == 0  # no limit set


def test_cache_hits_and_the_peak_share(services: Services, ledger: LedgerStore) -> None:
    report = september(services, ledger)
    assert (
        report.total.cache_hit_tokens == 4 * 400 and report.total.cache_miss_tokens == 4 * 100 + 500
    )
    assert report.total.cache_hit_ratio == pytest.approx(1600 / 2500)
    assert report.peak.calls == 2 and report.peak.cost_usd == pytest.approx(0.5)
    assert report.offpeak.calls == 3 and report.offpeak.cost_usd == pytest.approx(0.5)
    assert report.peak_share == pytest.approx(0.5) and report.peak_call_share == pytest.approx(0.4)


def test_an_empty_month_is_a_report_not_an_error(services: Services, ledger: LedgerStore) -> None:
    report = build_report(ledger, time_service_for(services), BudgetConfig(), date(2026, 1, 1))
    text = render_text(report)
    assert report.total.calls == 0 and report.peak_share == 0.0 and report.peak_call_share == 0.0
    assert text.count("（没有调用）") == 3 and "缓存命中率 0%" in text


def test_the_text_and_the_html_carry_numbers_and_names_only(
    services: Services, ledger: LedgerStore
) -> None:
    report = september(services, ledger)
    text = render_text(report)
    assert "费用报告 2026-09（America/Chicago）" in text
    assert "本月合计 $1.00 / 月预算 $2.00（50%），5 次调用；超过日预算 $0.4000 的有 1 天" in text
    assert "缓存命中率 64%" in text and "高峰占花费 50%、占调用 40%" in text
    assert "按用途" in text and "按模型" in text and "按日" in text
    page = render_html(report)
    assert page.startswith("<!doctype html>") and "<p>费用报告 2026-09" in page
    assert "&" not in page.replace("&amp;", "").replace("&lt;", "").replace("&gt;", "")


def test_the_json_has_the_same_numbers(services: Services, ledger: LedgerStore) -> None:
    data = report_to_json(september(services, ledger))
    assert data["month"] == "2026-09" and data["calls"] == 5
    assert data["total_usd"] == pytest.approx(1.0) and data["peak_share"] == 0.5
    assert data["one_time_usd"] == 1.5 and data["days_over_daily"] == 1
    by_purpose = {row["key"]: row for row in data["by_purpose"]}  # type: ignore[attr-defined]
    assert by_purpose["reply"]["calls"] == 2 and by_purpose["reply"]["cost_usd"] == 0.5
    json.dumps(data)  # plain data


def test_months_are_parsed_strictly() -> None:
    assert parse_month("2026-09") == date(2026, 9, 1) and parse_month(" 2027-12 ") == date(
        2027, 12, 1
    )
    for bad in ("2026-13", "2026-9", "09-2026", "2026-00", "september", ""):
        with pytest.raises(CostReportError, match="YYYY-MM"):
            parse_month(bad)
    assert previous_month(date(2026, 10, 15)) == date(2026, 9, 1)
    assert previous_month(date(2026, 1, 1)) == date(2025, 12, 1)


def test_the_peak_split_is_a_ledger_query(services: Services, ledger: LedgerStore) -> None:
    start, end = (
        datetime(2026, 9, 1, 5, tzinfo=UTC),
        datetime(2026, 10, 1, 5, tzinfo=UTC),
    )
    parts = ledger.by_peak(start, end)
    assert parts["peak"].calls == 2 and parts["offpeak"].calls == 3
    assert ledger.by_peak(start, end, account="one_time")["offpeak"].calls == 1
    assert ledger.by_peak(start, end, account=None)["peak"].calls == 2


# ----------------------------------------------------------------------------- the command


def twin(services: Services, *args: str) -> tuple[int, str]:
    result = runner.invoke(
        app, ["--set", f"paths.data_dir={services.paths.data_dir}", "cost", "report", *args]
    )
    return result.exit_code, result.output


@pytest.fixture
def cli(services: Services, ledger: LedgerStore, secret_store: SecretStore) -> Services:
    # the command reads the clock of the tests ("this month" is the month of its data), not the
    # day on which the suite runs
    set_cli_context(CliContext(secrets=secret_store, clock=services.clock))
    return services


def test_cost_report_prints_a_month(cli: Services) -> None:
    code, out = twin(cli, "--month", "2026-09")
    assert code == 0, out
    assert "费用报告 2026-09" in out and "按日" in out and "reply" in out


def test_cost_report_defaults_to_this_month_and_can_print_json(
    cli: Services, clock: ManualClock
) -> None:
    code, out = twin(cli)
    assert code == 0 and "费用报告 2026-10" in out  # the clock of the tests says October
    code, out = twin(cli, "--month", "2026-09", "--json")
    data = json.loads(out)
    assert code == 0 and data["month"] == "2026-09" and data["calls"] == 5


def test_a_wrong_month_is_a_usage_error(cli: Services) -> None:
    code, out = twin(cli, "--month", "2026-13")
    assert code == 2 and "YYYY-MM" in out
