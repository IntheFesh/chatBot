"""The words of the memory block: headings, subject labels, the way a line is written.

Everything the model reads of the memory is worded here, so the block of a live reply and the
block of a training sample (R-TRN-002) cannot differ in wording.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from twin.memory.records import FactRecord, FollowupRecord, SummaryRecord

SECTION_TODAY = "今天要留意的"
SECTION_LIFELINE = "今天的生活线"
SECTION_FACTS = "相关的事"
SECTION_RECENT = "最近几天"
SECTION_EARLIER = "更早的相关日子"
SECTION_ORDER = (SECTION_TODAY, SECTION_LIFELINE, SECTION_FACTS, SECTION_RECENT, SECTION_EARLIER)

SUBJECT_LABELS = {"her": "她", "user": "对方", "both": "你们俩", "other": "别人"}
WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
BOT_FACT_NOTE = "（我自己说过的，没有记录证实）"


def heading(section: str) -> str:
    return f"【{section}】"


def day_label(day: date) -> str:
    return f"{day.month}月{day.day}日{WEEKDAY_NAMES[day.weekday()]}"


def fact_line(fact: FactRecord, offset: int | None) -> str:
    """One line for a fact; ``offset`` is the days to its date when that date is near."""
    label = SUBJECT_LABELS.get(fact.subject, "")
    text = f"{label}：{fact.text}" if label else fact.text
    if offset == 0:
        text += "（就是今天）"
    elif offset == 1:
        text += "（就是明天）"
    elif offset == -1:
        text += "（是昨天）"
    elif offset is not None and 2 <= offset <= 7:
        text += f"（{offset}天后）"
    if fact.source == "bot_invented":
        text += BOT_FACT_NOTE
    return text


def summary_line(summary: SummaryRecord) -> str:
    kind = "和对方的聊天" if summary.scope == "bot" else "聊天记录"
    return f"{day_label(summary.local_date)}（{kind}）：{summary.text}"


def followup_line(followup: FollowupRecord, zone: ZoneInfo) -> str:
    local = followup.due_at.astimezone(zone)
    return f"{followup.text}（{local.month}月{local.day}日 {local:%H:%M}）"
