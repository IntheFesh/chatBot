"""What the evaluation commands print: blind-test statistics, style metrics, memory score, gates.

Every function takes a rich :class:`~rich.console.Console`, so the tests read what a command
shows from a console on a string.  Cells are :class:`~rich.text.Text`, never markup strings.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from twin.eval.blind import LENGTH_ORDER, PERIOD_ORDER, BackendReport, BlindReport, ordered
from twin.eval.gates import GateOutcome
from twin.eval.memory_test import MemorySummary
from twin.eval.samples import LENGTH_LABELS, PERIOD_LABELS
from twin.eval.stats import Rate
from twin.eval.style_metrics import TOLERANCE, MetricResult, StyleReport


def percent(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def interval_text(rate: Rate) -> str:
    found = rate.interval
    return "—" if found is None else f"{found[0] * 100:.1f}%–{found[1] * 100:.1f}%"


def print_blind_report(console: Console, report: BlindReport) -> None:
    """Guess rate with its Wilson interval per backend, the groups, and the tests between them."""
    table = Table(title=f"盲测 {report.run.id}", show_lines=False)
    for column in (
        "后端",
        "有效判断",
        "猜对",
        "猜对率",
        "95% 区间 (Wilson)",
        "跳过",
        "未能生成",
        "待判",
    ):
        table.add_column(column, justify="right" if column != "后端" else "left")
    for entry in report.backends:
        rate = entry.rate
        table.add_row(
            Text(entry.backend),
            Text(str(entry.judged)),
            Text(str(entry.correct)),
            Text(percent(rate.point)),
            Text(interval_text(rate)),
            Text(str(entry.skipped)),
            Text(str(entry.failed)),
            Text(str(entry.waiting)),
        )
    console.print(table)
    for entry in report.backends:
        _print_groups(console, entry)
    for comparison in report.comparisons:
        test = comparison.test
        console.print(
            Text(
                f"{comparison.first} 对 {comparison.second}：两比例检验 z = {test.z:.3f}，"
                f"双侧 p = {test.p_two_sided:.4f}，"
                f"单侧 p（{comparison.first} 猜对率更低）= {test.p_less:.4f}"
            )
        )


def _print_groups(console: Console, entry: BackendReport) -> None:
    if not entry.judged:
        return
    table = Table(title=f"{entry.backend}：按时段与会话长度", show_lines=False)
    table.add_column("分组")
    table.add_column("有效判断", justify="right")
    table.add_column("猜对率", justify="right")
    table.add_column("95% 区间", justify="right")
    groups: list[tuple[str, Rate]] = [
        *((PERIOD_LABELS[k], r) for k, r in ordered(entry.by_period, PERIOD_ORDER)),
        *((f"会话 {LENGTH_LABELS[k]}", r) for k, r in ordered(entry.by_length, LENGTH_ORDER)),
    ]
    for label, rate in groups:
        table.add_row(
            Text(label), Text(str(rate.total)), Text(percent(rate.point)), Text(interval_text(rate))
        )
    console.print(table)


def _deviation(result: MetricResult) -> str:
    if result.status == "n/a":
        return "不可测"
    return "—" if result.deviation is None else f"{result.deviation * 100:+.1f}%"


def _value(metric: MetricResult, value: float | None, proportion: bool) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if proportion else f"{value:.2f}"


_STATUS = {"pass": "通过", "fail": "不通过", "n/a": "不适用", "no_data": "无数据"}
_PROPORTIONS = {"comma_rate", "sticker_share", "emoji_code_rate", "quote_rate"}


def print_style_report(console: Console, report: StyleReport) -> None:
    """Hers against the bot's, the relative deviation and whether it is within +-30 %."""
    where = (
        f"最近 {report.days} 天的真实输出 vs live 画像"
        if report.source == "live"
        else f"回测 {report.run_id}（{report.backend} 后端）vs pre_holdout 画像"
    )
    table = Table(title=f"风格指标：{where}（每项偏差 ±{TOLERANCE:.0%} 内为通过）")
    for column in ("指标", "她（画像）", "机器人", "相对偏差", "结论", "机器人的 95% 区间"):
        table.add_column(column)
    show_real = any(r.real is not None for r in report.results)
    if show_real:
        table.add_column("她的真实回复（同样的上下文）")
    for result in report.results:
        proportion = result.key in _PROPORTIONS
        cells = [
            result.label,
            _value(result, result.reference.value, proportion),
            _value(result, result.measured.value, proportion),
            _deviation(result),
            _STATUS[result.status],
            "—"
            if result.interval is None
            else f"{result.interval[0] * 100:.1f}%–{result.interval[1] * 100:.1f}%",
        ]
        if show_real:
            cells.append(_value(result, result.real.value if result.real else None, proportion))
        table.add_row(*(Text(cell) for cell in cells))
    console.print(table)
    for note in report.notes:
        console.print(Text(f"注：{note}", style="dim"))
    console.print(Text(f"机器人的消息条数：{report.messages}"))
    verdict = "每项都在 ±30% 内：通过" if report.passed else "有指标超出 ±30%：不通过"
    console.print(Text(verdict, style="green" if report.passed else "red"))
    if not report.passed:
        worst = [r for r in report.worst if r.status != "pass"][:3]
        if worst:
            console.print(
                Text("偏差最大的几项：" + "、".join(f"{r.label} {_deviation(r)}" for r in worst))
            )


def print_memory_summary(
    console: Console, summary: MemorySummary, reasons: Sequence[str] = ()
) -> None:
    """The score of a memory run."""
    if not summary.total:
        console.print(Text("记忆测试未通过（样本不足）", style="red"))
        for reason in reasons:
            console.print(Text(f"  {reason}"))
        return
    accuracy = summary.accuracy
    lines = [
        f"题目 {summary.total}：真实记录 {summary.real_items} + 机器人对话 {summary.bot_items}",
        f"已复核 {summary.reviewed}；正确 {summary.correct}、部分正确 {summary.partial}、"
        f"错误 {summary.wrong}、未能出题 {summary.failed}",
        f"得分 {summary.points:g}/{summary.total}"
        + (f" = {accuracy * 100:.1f}%" if accuracy is not None else ""),
    ]
    for line in lines:
        console.print(Text(line))
    names = {"passed": "通过", "failed": "未通过", "insufficient": "未判定（样本不足或没复核完）"}
    console.print(
        Text(
            f"记忆测试：{names[summary.verdict]}",
            style="green" if summary.verdict == "passed" else "red",
        )
    )


def print_gate(console: Console, outcome: GateOutcome) -> None:
    """The criteria of a gate, the verdict and the evidence."""
    source = "（读取已存结果）" if outcome.checked else ""
    if outcome.verdict is None:
        console.print(Text(f"{outcome.milestone}：{outcome.message}{source}"))
        return
    table = Table(title=f"门槛 {outcome.milestone}{source}")
    table.add_column("条件")
    table.add_column("结果")
    table.add_column("数据")
    for check in outcome.verdict.checks:
        table.add_row(
            Text(check.name), Text("✔ 达标" if check.passed else "✘ 未达标"), Text(check.detail)
        )
    console.print(table)
    if outcome.verdict.runs:
        console.print(Text("依据的评估运行：" + "、".join(outcome.verdict.runs)))
    names = {"passed": "通过", "failed": "未通过", "insufficient": "未通过（样本不足）"}
    console.print(
        Text(
            f"{outcome.milestone}：{names.get(outcome.status, outcome.status)}",
            style="green" if outcome.status == "passed" else "red",
        )
    )
