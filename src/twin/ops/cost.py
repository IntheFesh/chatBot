"""The cost report: ``twin cost report`` and the monthly e-mail (R-OPS-005).

For one local month of the bot: the **daily** account of ``cost_ledger`` (what counts against the
budget) by day, by purpose and by model, the cache hit rate of the prompts, how much was spent
at peak and at off-peak prices, and the comparison with ``budget.monthly_usd`` and
``budget.daily_usd``.  One-time batches (R-LLM-014) have a line of their own: they are not part
of the budget.  Only prices, counts and the closed names of purposes and models appear, so the
report can be e-mailed.

On the 1st of every month at 09:00 local time the report of the month that just ended goes to
``ops.smtp.to`` (:mod:`twin.ops.scheduler`).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date, timedelta

from twin.config.settings import BudgetConfig
from twin.llm.ledger import LedgerStore, SpendSummary
from twin.schedule.time_service import TimeService

MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class CostReportError(ValueError):
    """The month asked for is not ``YYYY-MM``."""


@dataclass(frozen=True)
class CostReport:
    """One month of spending."""

    month: str
    zone: str
    days: tuple[SpendSummary, ...]
    purposes: tuple[SpendSummary, ...]
    models: tuple[SpendSummary, ...]
    total: SpendSummary
    one_time: SpendSummary
    peak: SpendSummary
    offpeak: SpendSummary
    monthly_budget_usd: float
    daily_budget_usd: float
    days_over_daily: int
    one_time_purposes: tuple[SpendSummary, ...] = ()  # what the one-time batches were for

    @property
    def month_ratio(self) -> float:
        return self.total.cost_usd / self.monthly_budget_usd if self.monthly_budget_usd > 0 else 0.0

    @property
    def peak_share(self) -> float:
        """Share of the spending that was at peak prices."""
        spent = self.peak.cost_usd + self.offpeak.cost_usd
        return self.peak.cost_usd / spent if spent > 0 else 0.0

    @property
    def peak_call_share(self) -> float:
        calls = self.peak.calls + self.offpeak.calls
        return self.peak.calls / calls if calls else 0.0


def parse_month(text: str) -> date:
    """The first day of ``YYYY-MM``."""
    found = MONTH.fullmatch(text.strip())
    if found is None:
        raise CostReportError(f"{text!r} is not a month; use YYYY-MM, for example 2026-09")
    return date(int(found.group(1)), int(found.group(2)), 1)


def previous_month(day: date) -> date:
    """The first day of the month before ``day``'s month."""
    return (day.replace(day=1) - timedelta(days=1)).replace(day=1)


def build_report(
    ledger: LedgerStore, time: TimeService, budget: BudgetConfig, month_start: date
) -> CostReport:
    """The report of the local month that starts on ``month_start``."""
    start, end = time.month_bounds_utc(month_start)
    last = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    days = tuple(ledger.by_day(month_start, last, account="daily"))
    peak = ledger.by_peak(start, end, account="daily")
    return CostReport(
        month=f"{month_start:%Y-%m}",
        zone=time.bot_timezone().key,
        days=days,
        purposes=tuple(sorted(ledger.by_purpose(start, end, account="daily"), key=_by_cost)),
        models=tuple(sorted(ledger.by_model(start, end, account="daily"), key=_by_cost)),
        total=ledger.totals(start, end, account="daily"),
        one_time=ledger.totals(start, end, account="one_time"),
        peak=peak["peak"],
        offpeak=peak["offpeak"],
        monthly_budget_usd=budget.monthly_usd,
        daily_budget_usd=budget.daily_usd,
        days_over_daily=sum(1 for day in days if day.cost_usd > budget.daily_usd > 0),
        one_time_purposes=tuple(
            sorted(ledger.by_purpose(start, end, account="one_time"), key=_by_cost)
        ),
    )


def _by_cost(row: SpendSummary) -> float:
    return -row.cost_usd


def _usd(value: float) -> str:
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


def _line(row: SpendSummary, total: float) -> str:
    share = f"{row.cost_usd / total:.0%}" if total > 0 else "-"
    return f"{row.key:<18} {_usd(row.cost_usd):>10} {share:>5} {row.calls:>7} calls"


def render_text(report: CostReport) -> str:
    """The report as plain text (also the text part of the mail)."""
    total = report.total
    lines = [
        f"费用报告 {report.month}（{report.zone}）",
        "",
        f"本月合计 {_usd(total.cost_usd)} / 月预算 {_usd(report.monthly_budget_usd)}"
        f"（{report.month_ratio:.0%}），{total.calls} 次调用；"
        f"超过日预算 {_usd(report.daily_budget_usd)} 的有 {report.days_over_daily} 天",
        f"缓存命中率 {total.cache_hit_ratio:.0%}（命中 {total.cache_hit_tokens} / "
        f"提示共 {total.cache_hit_tokens + total.cache_miss_tokens} token）",
        f"高峰价 {_usd(report.peak.cost_usd)}（{report.peak.calls} 次），"
        f"非高峰价 {_usd(report.offpeak.cost_usd)}（{report.offpeak.calls} 次），"
        f"高峰占花费 {report.peak_share:.0%}、占调用 {report.peak_call_share:.0%}",
    ]
    if report.one_time.calls:
        spent, calls = _usd(report.one_time.cost_usd), report.one_time.calls
        lines.append(f"一次性任务（不占预算）{spent}，{calls} 次调用")
        lines += [
            f"  一次性 · {_line(row, report.one_time.cost_usd)}" for row in report.one_time_purposes
        ]
    lines += ["", "按用途"]
    lines += [_line(row, total.cost_usd) for row in report.purposes] or ["（没有调用）"]
    lines += ["", "按模型"]
    lines += [_line(row, total.cost_usd) for row in report.models] or ["（没有调用）"]
    lines += ["", "按日"]
    lines += [_line(row, total.cost_usd) for row in report.days] or ["（没有调用）"]
    return "\n".join(lines)


def render_html(report: CostReport) -> str:
    body = "".join(
        f"<p>{html.escape(line)}</p>" for line in render_text(report).splitlines() if line
    )
    return (
        '<!doctype html><html lang="zh"><head><meta charset="utf-8"></head>'
        f"<body>{body}</body></html>"
    )


def report_to_json(report: CostReport) -> dict[str, object]:
    """The report as plain data (``twin cost report --json``)."""

    def rows(items: tuple[SpendSummary, ...]) -> list[dict[str, object]]:
        return [
            {
                "key": r.key,
                "calls": r.calls,
                "cost_usd": round(r.cost_usd, 6),
                "cache_hit_ratio": round(r.cache_hit_ratio, 4),
            }
            for r in items
        ]

    return {
        "month": report.month,
        "zone": report.zone,
        "total_usd": round(report.total.cost_usd, 6),
        "calls": report.total.calls,
        "monthly_budget_usd": report.monthly_budget_usd,
        "month_ratio": round(report.month_ratio, 4),
        "daily_budget_usd": report.daily_budget_usd,
        "days_over_daily": report.days_over_daily,
        "cache_hit_ratio": round(report.total.cache_hit_ratio, 4),
        "peak_usd": round(report.peak.cost_usd, 6),
        "offpeak_usd": round(report.offpeak.cost_usd, 6),
        "peak_share": round(report.peak_share, 4),
        "one_time_usd": round(report.one_time.cost_usd, 6),
        "one_time_calls": report.one_time.calls,
        "one_time_by_purpose": rows(report.one_time_purposes),
        "by_day": rows(report.days),
        "by_purpose": rows(report.purposes),
        "by_model": rows(report.models),
    }
