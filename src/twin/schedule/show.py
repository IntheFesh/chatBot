"""A day plan in words, in the clock times of the place she lives (``twin plan show``).

Everything is stored as instants; this is where they become clock times again.  Only numbers and
clock times are shown - the plan holds no chat text.  The life line of the day is not part of it.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from twin.schedule.plan_model import DailyPlan, Segment, SleepEpisode

WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
DAY_TYPES = {"workday": "工作日", "weekend": "周末", "holiday": "节假日"}
STATES = {"deep_sleep": "深睡", "sleep_edge": "入睡/将醒", "busy": "忙碌", "free": "空闲"}
MEALS = {"breakfast": "早饭", "lunch": "午饭", "dinner": "晚饭"}
SOURCES = {"history": "来自她的活跃峰", "default": "常见饭点"}
SLEEP_SOURCES = {"days": "逐日推断", "curve": "活跃曲线低谷", "override": "手动修正"}
GREETING_REASONS = {
    "ok": "可以发",
    "no_wake_up_planned": "没有规划起床时间",
    "wake_up_already_passed": "起床问候的时间窗已过",
    "already_sent_after_this_wake_up": "起床后已经问候过了",
    "within_min_gap_of_the_last_greeting": "距上一次起床问候不足间隔，今天不再问候",
}
RULES = {
    "gap": "春季跳过的时刻，按跳变前的偏移读，实际晚一个小时",
    "repeated": "秋季重复的时刻，取第一次",
}


def local_text(moment: datetime, zone: ZoneInfo, day: date) -> str:
    """``HH:MM`` on the wall clock, with the date when it is not ``day``."""
    local = moment.astimezone(zone)
    if local.date() == day:
        return f"{local:%H:%M}"
    return f"{local:%m-%d %H:%M}"


def _night(label: str, episode: SleepEpisode | None, zone: ZoneInfo, day: date) -> str:
    if episode is None:
        return f"- {label}：无（没有睡眠数据）"
    minutes = round(episode.duration.total_seconds() / 60)
    notes = [SLEEP_SOURCES.get(episode.source, episode.source)]
    if episode.adopted_from:
        notes.append("接自前一天的计划")
    if episode.clipped:
        notes.append("时区切换时正处于夜里，立即去睡")
    return (
        f"- {label}：{local_text(episode.onset, zone, day)} → {local_text(episode.wake, zone, day)}"
        f"（{minutes // 60}小时{minutes % 60:02d}分；{'，'.join(notes)}）"
    )


def _segment(segment: Segment, zone: ZoneInfo, day: date) -> str:
    extra = f"（{segment.busy.label}）" if segment.busy else ""
    return (
        f"  {local_text(segment.start, zone, day)} – {local_text(segment.end, zone, day)}  "
        f"{STATES[segment.kind]}{extra}"
    )


def render_plan(plan: DailyPlan, *, stored: bool = True, replaced: int = 0) -> list[str]:
    """The lines of ``twin plan show`` for ``plan`` (local time of the plan's zone)."""
    zone = ZoneInfo(plan.timezone)
    day = plan.local_date
    kind = DAY_TYPES.get(plan.day_type, plan.day_type)
    head = f"{day.isoformat()}（{WEEKDAYS[day.weekday()]}，{kind}，{plan.timezone}）"
    covered = (
        f"{local_text(plan.effective_from, zone, day)} 至 {local_text(plan.ends_at, zone, day)}"
    )
    lines = [
        head if stored else f"{head}  [预览：尚未保存，生成第一份计划后种子才固定]",
        f"- 计划 {plan.id or '—'}；原因 {plan.reason}；种子 {plan.seed}；"
        f"生效 {covered}" + (f"；此前替换过 {replaced} 份" if replaced else ""),
        _night("睡眠·早晨", plan.morning, zone, day),
        _night("睡眠·夜晚", plan.night, zone, day),
    ]
    for span in plan.busy:
        latency = (
            f"，回复延迟中位 {span.latency_median_s / 60:.1f} 分钟 / p90 "
            f"{(span.latency_p90_s or 0) / 60:.1f} 分钟"
            if span.latency_median_s is not None
            else ""
        )
        lines.append(f"- 忙碌：{span.label}（{span.source}{latency}；延迟分布 {span.latency_ref}）")
    if not plan.busy:
        lines.append("- 忙碌：无")
    for meal in plan.meals:
        lines.append(
            f"- 饭点：{MEALS[meal.kind]} {local_text(meal.at, zone, day)}"
            f"（{SOURCES[meal.source]}，约 {meal.minutes} 分钟）"
        )
    quota = plan.quota
    if quota.enabled:
        lines.append(
            f"- 主动配额：{quota.total} 次（范围 {quota.minimum}–{quota.maximum}，"
            f"均值目标 {quota.mean_target}，来源 {quota.mean_source}；"
            f"本计划可用 {quota.for_plan} 次）"
        )
    else:
        lines.append("- 主动配额：0 次（主动消息已关闭）")
    greeting = plan.greeting
    window = (
        f"，{local_text(greeting.earliest, zone, day)}–{local_text(greeting.latest, zone, day)}"
        if greeting.allowed and greeting.earliest and greeting.latest
        else ""
    )
    lines.append(
        f"- 起床问候：{'允许' if greeting.allowed else '不发'}{window}"
        f"（{GREETING_REASONS.get(greeting.reason, greeting.reason)}）"
    )
    for item in plan.extra.get("dst", ()):
        lines.append(
            f"- 夏令时：{item['what']} 想排在 {item['asked']}，实际读作 {item['read_as']}"
            f"（{RULES.get(item['rule'], item['rule'])}）"
        )
    for warning in plan.warnings:
        lines.append(f"!!! 提示：{warning}")
    lines.append("- 状态时间线：")
    lines.extend(_segment(segment, zone, day) for segment in plan.segments)
    return lines
