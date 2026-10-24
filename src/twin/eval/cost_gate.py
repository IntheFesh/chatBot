"""The cost of a month against its ceiling: ``twin eval cost --month YYYY-MM`` (R-EVAL-007).

The requirement: **the month costs at most 15 US dollars** (``twin cost report``); the one-time
batches of R-LLM-014 (the replay of the whole history, picture descriptions, the generation of an
evaluation ...) are shown on a line of their own and **do not count**.  The numbers are the cost
report's own (:func:`twin.ops.cost.build_report`) - the same ledger, the same local month, the same
split by purpose and model, the same cache hit rate - and nothing is counted a second time here.

The ceiling is :data:`MONTHLY_LIMIT_USD`, fixed here and pinned to the SPEC text by a test.  It is
**not** ``budget.monthly_usd``: that is a setting the user may change (and the degradation of the
bot follows it), whereas a gate that moved with its own setting could be passed by editing the
file.  Both are printed, so a budget set above the ceiling is visible.

A month is judged only on what is on record:

* over the ceiling: ``failed`` - even when the month is not over yet, the money is spent;
* no call in the ledger at all: ``insufficient`` - a month in which the bot was not used has not
  shown what it costs;
* under the ceiling but the month still running: ``insufficient`` - "so far" is not a month; the
  report shows what the month would come to at the same pace, as information and not as a verdict;
* under (or exactly at) the ceiling in a month that is over: ``passed``.

The total is compared to the millionth of a dollar, so the float sums of many small calls cannot
turn exactly 15 into 15.000000000000002.  The result is stored as ``eval_runs(kind=cost)``.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from twin.eval.gates import Check, Verdict
from twin.eval.store import EvalStore, RunView
from twin.llm.ledger import LedgerStore, SpendSummary
from twin.ops.cost import CostReport, build_report, render_text, report_to_json
from twin.schedule.service import time_service_for
from twin.services import Services

MONTHLY_LIMIT_USD = Decimal("15.00")  # R-EVAL-007: "月费用 ≤ 15 美元" (pinned to the SPEC text)
PRECISION = Decimal("0.000001")  # a millionth of a dollar: below what the ledger can tell apart
SHOWN = 8  # rows of the by-purpose and by-model tables kept in the stored summary


@dataclass(frozen=True)
class CostEvaluation:
    """A month, its report, the verdict and the run it was stored as."""

    report: CostReport
    verdict: Verdict
    reason: str
    check: Check
    complete: bool
    days_elapsed: int
    days_in_month: int
    projected_usd: float | None
    run: RunView | None = None

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"


def usd_of(value: float) -> Decimal:
    """A dollar amount from the ledger's floats, to the millionth of a dollar."""
    return Decimal(repr(value)).quantize(PRECISION)


def judge_month(total_usd: float, calls: int, *, complete: bool) -> tuple[Verdict, str]:
    """The verdict on a month's daily-account cost and the sentence that says why."""
    spent = usd_of(total_usd)
    if spent > MONTHLY_LIMIT_USD:
        return "failed", f"日常账目 ${spent:.2f}，超过每月 ${MONTHLY_LIMIT_USD:.2f} 的上限"
    if calls == 0:
        return "insufficient", "这个月日常账目里没有任何调用：还不知道它每月花多少"
    if not complete:
        return "insufficient", (
            f"日常账目目前 ${spent:.2f}（上限 ${MONTHLY_LIMIT_USD:.2f}），"
            "但这个月还没结束：月底后再判定"
        )
    return "passed", f"日常账目 ${spent:.2f}，没有超过每月 ${MONTHLY_LIMIT_USD:.2f} 的上限"


def _months_progress(services: Services, month_start: date) -> tuple[bool, int, int]:
    """``(the month is over, local days elapsed including today, days in the month)``."""
    time = time_service_for(services)
    _, end = time.month_bounds_utc(month_start)
    now = services.clock.now_utc()
    in_month = calendar.monthrange(month_start.year, month_start.month)[1]
    if now >= end:
        return True, in_month, in_month
    elapsed = (time.local_date(now) - month_start).days + 1
    return False, max(0, min(elapsed, in_month)), in_month


def _rows(items: tuple[SpendSummary, ...]) -> list[dict[str, Any]]:
    return [
        {
            "key": row.key,
            "calls": row.calls,
            "cost_usd": round(row.cost_usd, 6),
            "cache_hit_ratio": round(row.cache_hit_ratio, 4),
        }
        for row in items[:SHOWN]
    ]


def summary_of(evaluation: CostEvaluation) -> dict[str, Any]:
    """What is stored with the run: numbers and the closed names of purposes and models."""
    report = evaluation.report
    detail = report_to_json(report)
    return {
        **detail,
        "limit_usd": float(MONTHLY_LIMIT_USD),
        "complete": evaluation.complete,
        "days_elapsed": evaluation.days_elapsed,
        "days_in_month": evaluation.days_in_month,
        "projected_usd": evaluation.projected_usd,
        "reason": evaluation.reason,
        "by_purpose": _rows(report.purposes),
        "by_model": _rows(report.models),
        "one_time_by_purpose": _rows(report.one_time_purposes),
    }


def evaluate_cost(services: Services, month_start: date, *, record: bool = True) -> CostEvaluation:
    """Judge the local month that starts on ``month_start`` and (by default) store the run."""
    time = time_service_for(services)
    ledger = LedgerStore(services.db, services.clock, time)
    report = build_report(ledger, time, services.settings.budget, month_start)
    complete, elapsed, in_month = _months_progress(services, month_start)
    total = report.total
    verdict, reason = judge_month(total.cost_usd, total.calls, complete=complete)
    projected = (
        round(total.cost_usd / elapsed * in_month, 6)
        if not complete and elapsed > 0 and total.calls
        else None
    )
    check = Check(
        f"日常账目月费用 ≤ ${MONTHLY_LIMIT_USD:.2f}（一次性批任务不计入）",
        verdict == "passed",
        reason,
        round(total.cost_usd, 6),
        float(MONTHLY_LIMIT_USD),
        total.calls,
    )
    evaluation = CostEvaluation(
        report, verdict, reason, check, complete, elapsed, in_month, projected
    )
    if not record:
        return evaluation
    run = EvalStore(services.db, services.clock).create_run(
        "cost",
        status="done",
        verdict=verdict,
        params={"month": report.month, "zone": report.zone},
        summary=summary_of(evaluation),
    )
    return CostEvaluation(
        report, verdict, reason, check, complete, elapsed, in_month, projected, run
    )


def render_lines(evaluation: CostEvaluation) -> list[str]:
    """The screen: the cost report of the month, then the gate."""
    report = evaluation.report
    lines = [render_text(report), ""]
    lines.append(
        f"预算设置：月 ${report.monthly_budget_usd:.2f}、日 ${report.daily_budget_usd:.2f}"
        f"（门槛固定为 ${MONTHLY_LIMIT_USD:.2f}，不随预算设置变化）"
    )
    if evaluation.projected_usd is not None:
        lines.append(
            f"按目前的速度，这个月（已过 {evaluation.days_elapsed}/{evaluation.days_in_month} 天）"
            f"大约 ${evaluation.projected_usd:.2f}（只是估计，不参与判定）"
        )
    names = {"passed": "通过", "failed": "未通过", "insufficient": "未通过（数据不足）"}
    lines.append(f"R-EVAL-007 成本门槛：{names[evaluation.verdict]}。{evaluation.reason}")
    if evaluation.run is not None:
        lines.append(f"评估记录：{evaluation.run.id}")
    return lines
