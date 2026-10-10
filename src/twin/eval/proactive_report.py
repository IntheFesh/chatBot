"""What ``twin eval proactive`` and ``twin proactive log`` print (R-EVAL-005, R-PRO-008).

Every function takes a rich :class:`~rich.console.Console`; cells are :class:`~rich.text.Text`.
The log never shows the words of a message unless the caller says so (``show_text``).
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from twin.eval.proactive_audit import Audit
from twin.schedule.proactive.store import LogEntry
from twin.schedule.proactive.types import KIND_LABELS, REASON_LABELS

BAR = "█"
BAR_WIDTH = 24
OUTCOME_LABELS = {
    "sent": "已发",
    "rejected": "被拒",
    "declined": "规划不发",
    "failed": "失败",
    "expired": "过期",
    "dropped": "放弃",
    "opened": "当日开始",
}
STATE_LABELS = {
    "deep_sleep": "深睡",
    "sleep_edge": "睡眠边缘",
    "busy": "忙",
    "free": "空闲",
}


def print_audit(console: Console, audit: Audit) -> None:
    """The audit as tables: the days, the hours the messages went out, and the totals."""
    days = Table(title=f"主动消息审计 {audit.first_day} 至 {audit.last_day}")
    for column in (
        "日期",
        "已监测",
        "条数",
        "范围",
        "深睡",
        "边缘",
        "间隔违规",
        "追发违规",
        "被窗口抑制",
        "结论",
    ):
        days.add_column(column, justify="left" if column in ("日期", "结论") else "right")
    for day in audit.days:
        span = (
            "—"
            if day.low is None
            else (f"{day.low}-{day.high}" if day.enabled is not False else "关")
        )
        verdict = "合规" if day.compliant else "；".join(day.problems)
        if day.compliant and day.excused and day.low is not None and day.sent < day.low:
            verdict = "合规（少于下限，但被窗口/暂停等拦下）"
        days.add_row(
            Text(day.day.isoformat()),
            Text("是" if day.observed else "否"),
            Text(str(day.sent)),
            Text(span),
            Text(str(day.deep_sleep)),
            Text(str(day.edge)),
            Text(str(day.spacing_violations)),
            Text(str(day.chase_violations)),
            Text(str(day.suppressed_window)),
            Text(verdict, style="green" if day.compliant else "red"),
        )
    console.print(days)
    print_hours(console, audit.hours)
    console.print(
        Text(
            f"共发出 {audit.sent} 条；深睡时段 {audit.deep_sleep} 条；"
            f"间隔不足 {audit.spacing_violations} 次；追发超限 {audit.chase_violations} 次；"
            f"被窗口抑制 {audit.suppressed_window} 次；"
            f"任意 7 天内边缘消息最多 {audit.edge_max_week} 条（上限 {audit.edge_weekly_max}）。"
        )
    )
    if audit.ratings:
        mean = audit.rating_mean
        console.print(
            Text(
                f"这段时间 /评分 {len(audit.ratings)} 次，平均 {mean:.2f} 分"
                if mean is not None
                else "这段时间没有 /评分"
            )
        )
    else:
        console.print(Text("这段时间没有 /评分。"))
    console.print(
        Text(
            f"连续观察 {audit.streak}/{len(audit.days)} 天。"
            + ("" if audit.complete else f"还差 {audit.missing_days} 天。"),
        )
    )
    console.print(
        Text(
            "审计结论：" + ("合规" if audit.compliant and audit.complete else "不合规或未满"),
            style="green" if audit.compliant and audit.complete else "red",
        )
    )


def print_hours(console: Console, hours: dict[int, int]) -> None:
    """The local hours the messages went out in, as a text table with bars."""
    table = Table(title="发送时刻分布（当地时间）")
    table.add_column("时段", justify="right")
    table.add_column("条数", justify="right")
    table.add_column("")
    peak = max(hours.values(), default=0)
    for hour in range(24):
        count = hours.get(hour, 0)
        width = round(BAR_WIDTH * count / peak) if peak else 0
        table.add_row(Text(f"{hour:02d}:00"), Text(str(count)), Text(BAR * width))
    console.print(table)


def print_log(console: Console, entries: Sequence[LogEntry], *, show_text: bool = False) -> None:
    """The log as a table, one row per decision; the words only with ``show_text``."""
    table = Table(title="主动消息日志")
    columns = ["当地时间", "类型", "结果", "原因", "她的状态", "追发", "条数", "后端"]
    if show_text:
        columns.append("内容")
    for column in columns:
        table.add_column(column)
    for entry in entries:
        if entry.outcome == "opened":
            continue
        reason = REASON_LABELS.get(entry.reason or "", entry.reason or "")
        row = [
            Text(entry.local_at),
            Text(KIND_LABELS.get(entry.kind, entry.kind)),
            Text(OUTCOME_LABELS.get(entry.outcome, entry.outcome)),
            Text(reason),
            Text(STATE_LABELS.get(entry.her_state or "", entry.her_state or "")),
            Text(str(entry.chase_seq) if entry.outcome == "sent" else ""),
            Text(str(entry.bubbles_sent) if entry.outcome == "sent" else ""),
            Text(entry.backend or ""),
        ]
        if show_text:
            bubbles = (entry.content or {}).get("bubbles")
            row.append(
                Text(" / ".join(str(b) for b in bubbles) if isinstance(bubbles, list) else "")
            )
        table.add_row(*row)
    console.print(table)
