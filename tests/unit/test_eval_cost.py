"""R-EVAL-007: the month costs at most 15 US dollars; one-time batches are shown apart.

The ledger is built by hand: calls on the daily account and on the one-time account, in the month
asked for and next to it.  The boundary is tested at the cent and below.
"""

from __future__ import annotations

import io
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.eval.cli import use_interaction
from twin.eval.cost_gate import (
    MONTHLY_LIMIT_USD,
    evaluate_cost,
    judge_month,
    render_lines,
    usd_of,
)
from twin.eval.store import EvalStore
from twin.eval.ui import LineKeys
from twin.llm.ledger import LedgerRecord, LedgerStore
from twin.llm.types import CostBreakdown, LedgerTag, Usage
from twin.schedule.service import time_service_for
from twin.services import Services, build_services

SPEC = Path(__file__).resolve().parents[2] / "docs" / "SPEC.md"
runner = CliRunner()
SEPTEMBER = date(2026, 9, 1)
OCTOBER = date(2026, 10, 1)


def call(
    at: datetime,
    cost: float,
    *,
    purpose: str = "reply",
    model: str = "deepseek-flash",
    hit: int = 400,
    miss: int = 100,
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
        cost=CostBreakdown(cost, 0.0, 0.0, False, 1.0),
        thinking=False,
        latency_ms=200,
        at=at,
        tag=tag or LedgerTag(),
    )


def book(services: Services, *entries: LedgerRecord) -> LedgerStore:
    ledger = LedgerStore(services.db, services.clock, time_service_for(services))
    for entry in entries:
        ledger.record(entry)
    return ledger


def noon(day: int, month: int = 9) -> datetime:
    return datetime(2026, month, day, 17, tzinfo=UTC)


# ------------------------------------------------------------------------------ the rule


def test_the_ceiling_is_the_one_in_the_spec() -> None:
    line = next(
        row
        for row in SPEC.read_text(encoding="utf-8").splitlines()
        if row.startswith("- **R-EVAL-007**")
    )
    found = re.search(r"月费用 ≤ (\d+) 美元", line)
    assert found is not None and Decimal(found.group(1)) == MONTHLY_LIMIT_USD == Decimal("15.00")


@pytest.mark.parametrize(
    ("total", "calls", "complete", "verdict"),
    [
        (14.99, 100, True, "passed"),
        (15.0, 100, True, "passed"),  # exactly the ceiling is within it
        (15.000000000000002, 100, True, "passed"),  # float noise of many small sums
        (15.0000004, 100, True, "passed"),  # below a millionth of a dollar
        (15.000001, 100, True, "failed"),
        (15.01, 100, True, "failed"),
        (15.01, 100, False, "failed"),  # the money is spent, the month need not be over
        (3.0, 100, False, "insufficient"),  # "so far" is not a month
        (0.0, 0, True, "insufficient"),  # a month nobody used has shown nothing
        (0.0, 0, False, "insufficient"),
    ],
)
def test_the_verdict_at_the_boundary(
    total: float, calls: int, complete: bool, verdict: str
) -> None:
    assert judge_month(total, calls, complete=complete)[0] == verdict


def test_the_reasons_say_what_is_missing() -> None:
    assert "还没结束" in judge_month(3.0, 10, complete=False)[1]
    assert "没有任何调用" in judge_month(0.0, 0, complete=True)[1]
    assert "$16.00" in judge_month(16.0, 5, complete=True)[1]
    assert usd_of(0.1 + 0.2) == Decimal("0.300000")


# ------------------------------------------------------------------------- the evaluation


def test_daily_and_one_time_spending_are_counted_apart(services: Services) -> None:
    book(
        services,
        call(noon(3), 4.0, purpose="reply"),
        call(noon(4), 2.5, purpose="plan", model="deepseek-pro", hit=100, miss=400),
        call(noon(5), 0.5, purpose="extract"),
        call(noon(6), 30.0, purpose="persona", tag=LedgerTag("one_time", "batch-1")),
        call(noon(7), 2.0, purpose="eval", tag=LedgerTag("one_time", "batch-2")),
        call(datetime(2026, 8, 31, 12, tzinfo=UTC), 99.0),  # the month before
        call(datetime(2026, 10, 1, 12, tzinfo=UTC), 99.0),  # the month after
    )
    result = evaluate_cost(services, SEPTEMBER)
    assert result.verdict == "passed" and result.complete
    report = result.report
    assert report.total.cost_usd == pytest.approx(7.0) and report.total.calls == 3
    assert report.one_time.cost_usd == pytest.approx(32.0) and report.one_time.calls == 2
    assert [p.key for p in report.one_time_purposes] == ["persona", "eval"]  # dearest first
    summary = result.run.summary if result.run else {}
    assert summary["total_usd"] == pytest.approx(7.0) and summary["calls"] == 3
    assert summary["one_time_usd"] == pytest.approx(32.0) and summary["one_time_calls"] == 2
    assert [(r["key"], r["calls"]) for r in summary["one_time_by_purpose"]] == [
        ("persona", 1),
        ("eval", 1),
    ]
    assert [r["key"] for r in summary["by_purpose"]] == ["reply", "plan", "extract"]
    assert {r["key"] for r in summary["by_model"]} == {"deepseek-flash", "deepseek-pro"}
    assert summary["limit_usd"] == 15.0 and summary["complete"] is True


def test_the_batches_do_not_count_against_the_ceiling_however_large(services: Services) -> None:
    book(
        services,
        call(noon(3), 1.0),
        call(noon(4), 500.0, purpose="memory", tag=LedgerTag("one_time", "replay")),
    )
    assert evaluate_cost(services, SEPTEMBER).verdict == "passed"


def test_the_cache_hit_rate_is_per_purpose_and_model_and_overall(services: Services) -> None:
    book(
        services,
        call(noon(3), 1.0, purpose="reply", hit=900, miss=100),
        call(noon(4), 1.0, purpose="reply", hit=900, miss=100),
        call(noon(5), 1.0, purpose="plan", model="deepseek-pro", hit=0, miss=1000),
    )
    result = evaluate_cost(services, SEPTEMBER)
    summary = result.run.summary if result.run else {}
    assert summary["cache_hit_ratio"] == pytest.approx(1800 / 3000)
    by_purpose = {r["key"]: r["cache_hit_ratio"] for r in summary["by_purpose"]}
    assert by_purpose == {"reply": 0.9, "plan": 0.0}
    by_model = {r["key"]: r["cache_hit_ratio"] for r in summary["by_model"]}
    assert by_model == {"deepseek-flash": 0.9, "deepseek-pro": 0.0}


def test_calls_that_add_up_to_exactly_the_ceiling_pass_despite_float_noise(
    services: Services,
) -> None:
    # 1.9 + 3.6 + 1.1 + 0.2 + 1.6 + 3.7 + 2.9 is 15 to the cent and 15.000000000000002 in floats
    book(
        services,
        *(call(noon(1 + n), cost) for n, cost in enumerate((1.9, 3.6, 1.1, 0.2, 1.6, 3.7, 2.9))),
    )
    result = evaluate_cost(services, SEPTEMBER)
    assert result.report.total.cost_usd > 15.0  # the noise is really there ...
    assert result.verdict == "passed"  # ... and does not count
    book(services, call(noon(9), 0.01))
    assert evaluate_cost(services, SEPTEMBER).verdict == "failed"  # a cent more does


def test_a_month_that_is_not_over_cannot_pass_and_shows_the_pace(
    services: Services, clock: ManualClock
) -> None:
    assert clock.now_utc() == datetime(2026, 10, 9, 12, tzinfo=UTC)
    book(services, call(noon(2, 10), 4.5), call(noon(5, 10), 4.5))
    result = evaluate_cost(services, OCTOBER)
    assert result.verdict == "insufficient" and not result.complete
    assert result.days_elapsed == 9 and result.days_in_month == 31
    assert result.projected_usd == pytest.approx(9.0 / 9 * 31)  # for the eye, not for the verdict
    assert "还没结束" in result.reason and "月底后" in result.reason
    book(services, call(noon(8, 10), 7.0))  # over before the month is out
    assert evaluate_cost(services, OCTOBER).verdict == "failed"


def test_a_month_without_any_call_is_not_a_pass(services: Services) -> None:
    result = evaluate_cost(services, SEPTEMBER)
    assert result.verdict == "insufficient" and result.report.total.calls == 0
    assert result.projected_usd is None and "没有任何调用" in result.reason


def test_a_budget_setting_does_not_move_the_ceiling(services: Services) -> None:
    book(services, call(noon(3), 20.0))
    services.settings.budget.monthly_usd = 100.0
    assert evaluate_cost(services, SEPTEMBER).verdict == "failed"
    services.settings.budget.monthly_usd = 1.0
    assert evaluate_cost(services, date(2026, 7, 1), record=False).verdict == "insufficient"
    lines = render_lines(evaluate_cost(services, SEPTEMBER, record=False))
    assert any("门槛固定为 $15.00，不随预算设置变化" in line for line in lines)


def test_the_run_is_stored_with_numbers_only(services: Services) -> None:
    book(services, call(noon(3), 2.0, purpose="reply"))
    result = evaluate_cost(services, SEPTEMBER)
    run = EvalStore(services.db, services.clock).latest_run("cost")
    assert run is not None and result.run is not None and run.id == result.run.id
    assert run.kind == "cost" and run.status == "done" and run.verdict == "passed"
    assert run.params == {"month": "2026-09", "zone": "America/Chicago"}
    assert run.mode is None and run.milestone is None
    assert evaluate_cost(services, SEPTEMBER, record=False).run is None
    assert len(EvalStore(services.db, services.clock).list_runs("cost")) == 1


def test_the_screen_is_the_cost_report_and_the_verdict(services: Services) -> None:
    book(
        services,
        call(noon(3), 6.0),
        call(noon(4), 3.0, purpose="persona", tag=LedgerTag("one_time", "b")),
    )
    text = "\n".join(render_lines(evaluate_cost(services, SEPTEMBER)))
    assert "费用报告 2026-09" in text and "按用途" in text and "按模型" in text
    assert "缓存命中率" in text and "一次性任务（不占预算）$3.00" in text
    assert "一次性 · persona" in text
    assert "R-EVAL-007 成本门槛：通过。" in text and "评估记录：" in text


# -------------------------------------------------------------------------------- the CLI


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    return path


def eval_cli(*args: str) -> tuple[int, str, str]:
    """``(exit code, what the command printed, the screen)`` of ``twin eval <args>``."""
    screen = io.StringIO()
    console = Console(file=screen, width=160, color_system=None, highlight=False)
    with use_interaction(console, LineKeys(io.StringIO(""))):
        result = runner.invoke(app, ["eval", *args])
    return result.exit_code, result.output, screen.getvalue()


def seed(*entries: LedgerRecord) -> None:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)
    try:
        book(services, *entries)
    finally:
        services.close()


def test_the_command_passes_a_cheap_month_and_fails_an_expensive_one(data_dir: Path) -> None:
    seed(
        call(datetime(2026, 3, 3, 17, tzinfo=UTC), 5.0),
        call(datetime(2026, 4, 3, 17, tzinfo=UTC), 15.01),
        call(datetime(2026, 4, 4, 17, tzinfo=UTC), 3.0, tag=LedgerTag("one_time", "b")),
    )
    code, _, screen = eval_cli("cost", "--month", "2026-03")
    assert code == 0 and "成本门槛：通过" in screen and "日常账目 $5.00" in screen
    code, _, screen = eval_cli("cost", "--month", "2026-04")
    assert code == 1 and "成本门槛：未通过。" in screen and "超过每月 $15.00" in screen
    assert "一次性任务（不占预算）$3.00" in screen
    code, _, screen = eval_cli("cost", "--month", "2026-02")
    assert code == 1 and "数据不足" in screen and "没有任何调用" in screen


def test_a_month_that_is_not_a_month_is_a_usage_error(data_dir: Path) -> None:
    code, out, _ = eval_cli("cost", "--month", "2026-13")
    assert code == 2 and "YYYY-MM" in out
    code, out, _ = eval_cli("cost", "--month", "September")
    assert code == 2


def test_the_default_month_is_this_month_and_the_run_is_listed(data_dir: Path) -> None:
    code, _, screen = eval_cli("cost")
    assert code == 1 and "R-EVAL-007" in screen  # nothing spent yet: not enough to judge
    code, _, listed = eval_cli("runs", "--kind", "cost")
    assert code == 0 and "cost" in listed and "insufficient" in listed
