"""Text of ``twin profile show``: the style numbers and the routine overview (R-ACT-006).

The overview is in the local clock time of the place she was in (the time zone the data was
learnt in), not in the bot's current zone.  It always ends with the request to confirm the
inferred sleep time, and a daytime sleep is announced prominently, because that almost always
means ``time.source_timezone`` does not match the export.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.cells import cell_len

from twin.profile.activity_model import (
    DAY_TYPE_LABELS,
    LOCAL_DAY_TYPES,
    ActivityModel,
    BusyWindow,
    SleepProfile,
    SleepWindow,
)
from twin.profile.diffing import summarize
from twin.profile.localtime import format_minute
from twin.profile.overrides import WEEKDAY_NAMES, OverrideView
from twin.profile.snapshot import ProfileMetrics
from twin.profile.store import ProfileVersionView
from twin.profile.values import Rates, Scalar

CONFIDENCE_LABELS = {"high": "高", "low": "低（数据不足，取整体活跃曲线的最长低谷）", "none": "无"}

# The sample of seven days in SPEC section 0 (her | user), shown for orientation only.
REFERENCE_SAMPLE: dict[str, tuple[str, str]] = {
    "text_length": ("5 / 10", "11 / 47"),
    "comma": ("2.9%", "44.5%"),
    "period_end": ("0.2%", "0.4%"),
    "burst": ("2 / 6", "1 / 4"),
    "burst_gap": ("6", "12"),
    "latency": ("17 / 240", "17 / 103"),
    "sticker_share": ("10.5%", "16.3%"),
    "emoji_code": ("3.5%", "6.8%"),
    "quote": ("6.4%", "—"),
    "initiations": ("3.6", "1.7"),
}


def _pad(text: str, width: int) -> str:
    """``text`` padded to ``width`` terminal cells (Chinese characters are two cells wide)."""
    return text + " " * max(0, width - cell_len(text))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _dist_pair(metrics: ProfileMetrics, party: str, name: str, low: float, high: float) -> str:
    dist = metrics.distribution(party, name)
    if dist is None:
        return "—"
    return f"{dist.quantile(low):.0f} / {dist.quantile(high):.0f}"


def _rate(metrics: ProfileMetrics, party: str, name: str, key: str) -> float | None:
    leaf = metrics.leaf(party, name)
    if not isinstance(leaf, Rates) or leaf.n == 0:
        return None
    return leaf.values.get(key, 0.0)


def _per_day(metrics: ProfileMetrics, party: str) -> str:
    leaf = metrics.leaf(party, "initiations_per_day")
    if not isinstance(leaf, Scalar) or leaf.n == 0:
        return "—"
    return f"{leaf.value:.1f}"


def style_table(metrics: ProfileMetrics) -> list[str]:
    """Rows ``metric | her | user | SPEC §0 sample (her | user)``."""
    rows: list[tuple[str, str, str, str]] = []

    def add(label: str, her: str, user: str, key: str) -> None:
        ref = REFERENCE_SAMPLE[key]
        rows.append((label, her, user, f"{ref[0]} | {ref[1]}"))

    def both(fn: Any) -> tuple[str, str]:
        return fn("her"), fn("user")

    add(
        "文字长度 中位 / p90（字）",
        *both(lambda p: _dist_pair(metrics, p, "text_length", 0.5, 0.9)),
        "text_length",
    )
    add(
        "带逗号的文字比例", *both(lambda p: _pct(_rate(metrics, p, "punct_rate", "comma"))), "comma"
    )
    add(
        "以句号结尾比例",
        *both(lambda p: _pct(_rate(metrics, p, "end_rate", "period"))),
        "period_end",
    )
    add(
        "连发条数 中位 / p90",
        *both(lambda p: _dist_pair(metrics, p, "burst_size", 0.5, 0.9)),
        "burst",
    )
    add(
        "连发条间隔 中位（秒）",
        *both(lambda p: _dist_pair(metrics, p, "burst_gap_s", 0.5, 0.5).split(" / ")[0]),
        "burst_gap",
    )
    add(
        "回复延迟 中位 / p90（秒）",
        *both(lambda p: _dist_pair(metrics, p, "reply_latency_s", 0.5, 0.9)),
        "latency",
    )
    add(
        "表情包占全部消息",
        *both(lambda p: _pct(metrics.scalar(p, "sticker_share"))),
        "sticker_share",
    )
    add(
        "带表情代码的文字比例",
        *both(lambda p: _pct(metrics.scalar(p, "emoji_code_rate"))),
        "emoji_code",
    )
    add("引用回复占文字类消息", *both(lambda p: _pct(metrics.scalar(p, "quote_rate"))), "quote")
    add("每天先开口次数", *both(lambda p: _per_day(metrics, p)), "initiations")
    width = max(cell_len(label) for label, *_ in rows)
    header = f"{_pad('指标', width)}  {_pad('她', 10)} {_pad('用户', 10)} SPEC §0 样本（她 | 用户）"
    lines = [header]
    for label, her, user, ref in rows:
        lines.append(f"{_pad(label, width)}  {_pad(her, 10)} {_pad(user, 10)} {ref}")
    return lines


def _window_text(window: SleepWindow, edge: int) -> str:
    core_start, core_end = window.core(edge)
    spread = window.onset.std
    return (
        f"{format_minute(window.onset_min)}–{format_minute(window.wake_min)}"
        f"（核心 {format_minute(core_start)}–{format_minute(core_end)}，"
        f"约 {window.duration_min() / 60:.1f} 小时，入睡时刻波动 ±{spread:.0f} 分钟，"
        f"依据 {window.valid_days} 个夜晚）"
    )


def sleep_lines(profile: SleepProfile, edge: int) -> list[str]:
    if not profile.windows:
        return ["- 睡眠：无法推断"]
    lines: list[str] = []
    for key in ("all", *LOCAL_DAY_TYPES):
        window = profile.windows.get(key)
        if window is None:
            continue
        source = {"days": "", "curve": "（活跃曲线低谷）", "override": "（手动修正）"}[
            window.source
        ]
        lines.append(f"- 睡眠·{DAY_TYPE_LABELS[key]}：{_window_text(window, edge)}{source}")
    return lines


def busy_lines(model: ActivityModel) -> list[str]:
    lines: list[str] = []
    for key in LOCAL_DAY_TYPES:
        for window in model.busy.get(key, ()):
            lines.append(f"- 忙碌·{DAY_TYPE_LABELS[key]}：{_busy_text(window)}")
    for window in model.manual_busy:
        days = "、".join(WEEKDAY_NAMES[d] for d in window.weekdays)
        lines.append(f"- 忙碌·{days}：{window.label()}（手动修正）")
    return lines or ["- 没有发现稳定的忙碌时段"]


def _busy_text(window: BusyWindow) -> str:
    median = window.latency.median() if not window.latency.is_empty else 0.0
    return (
        f"{window.label()}（回复延迟中位 {median / 60:.1f} 分钟，"
        f"是全天的 {window.latency_ratio:.1f} 倍；"
        f"稳定性 {window.stability:.0%}；置信度 {window.confidence:.2f}）"
    )


def routine_overview(model: ActivityModel, overrides: Sequence[OverrideView] = ()) -> list[str]:
    """The routine in local clock time, with warnings and the request to confirm."""
    profile = model.sleep
    days = model.days
    lines = [
        f"学习用的时区：{model.zone or '—'}；有消息的天数 工作日 {days.get('workday', 0)} / "
        f"周末 {days.get('weekend', 0)} / 节假日 {days.get('holiday', 0)}",
        f"推断置信度：{CONFIDENCE_LABELS[profile.confidence]}；"
        f"可用于推断的夜晚 {profile.valid_days} 个",
        *sleep_lines(profile, model.edge_minutes),
        *busy_lines(model),
        f"- 每天先开口 {model.initiations_per_day:.1f} 次",
    ]
    for warning in profile.warnings:
        lines.append(f"!!! 警告：{warning}")
    lines.append("")
    lines.append("请确认：上面推断的睡眠时段与实际相符吗？")
    lines.append(
        "  不对时用 `twin routine add sleep HH:MM HH:MM`"
        "（可加 --days workday/weekend/holiday）修正；"
        "忙碌时段用 `twin routine add busy`，节假日用 `twin routine add holiday`。"
    )
    if overrides:
        lines.append("")
        lines.append("手动修正（优先于推断）：")
        for item in overrides:
            state = "" if item.enabled else "（已停用）"
            lines.append(f"  {item.id}  {item.describe()}{state}")
    return lines


def profile_overview(
    version: ProfileVersionView,
    metrics: ProfileMetrics,
    model: ActivityModel | None,
    overrides: Sequence[OverrideView] = (),
) -> list[str]:
    full = metrics.window_info("full")
    recent = metrics.window_info("recent")
    marker = "（当前生效）" if version.active else ""
    cutoff = version.data_range.get("cutoff")
    lines = [
        f"画像版本 {version.id}{marker}",
        f"范围 {version.scope}；{version.created_at:%Y-%m-%d %H:%M} UTC；原因 {version.reason}；"
        f"上一版 {version.parent_id or '—'}",
        f"数据：{full.get('start')} 至 {full.get('end')}（{full.get('days')} 天；"
        f"她 {full.get('her_messages'):,} 条，用户 {full.get('user_messages'):,} 条）；"
        f"最近窗口 {recent.get('days')} 天（她 {recent.get('her_messages'):,} 条）",
        *([f"本版本只用 {cutoff} 之前的消息（留出集之前）"] if cutoff else []),
        "",
        f"== 风格指标（混合值：最近窗口权重 {metrics.data['config'].get('recency_weight')}）==",
        *style_table(metrics),
        "",
        "== 数字风格规则 ==",
        *([f"- {line}" for line in version.rule_lines] or ["（没有规则：数据不足）"]),
    ]
    if version.changes:
        lines += ["", "== 相对上一版变化超过 10% 的指标 =="]
        lines += [f"- {line}" for line in summarize(version.changes, 20)]
    lines += ["", f"== 作息概览（当地时间，{version.scope}）=="]
    if model is None:
        lines.append("（这个版本没有作息模型：运行 `twin profile rebuild`）")
    else:
        lines += routine_overview(model, overrides)
    return lines
