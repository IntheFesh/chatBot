"""The vocabulary of the proactive messages: kinds, reasons, candidates (R-PRO-003 to R-PRO-005).

``TriggerKind``
    why a message is considered: a follow-up that came due, one of the fixed moments of her day
    (the wake-up greeting, a meal, goodnight), the silence of the conversation, something from her
    day to share, or the edge of her sleep.  :data:`PRIORITY` orders them (R-PRO-004): follow-up,
    then the routine ones, then silence, then sharing; the edge of sleep stands apart (it is only
    considered while she is at the edge of her sleep, where nothing else may go out).
``Reason``
    the closed set of codes that say why a candidate did not go out.  The hard constraints of
    R-PRO-003 are the first ones; the later ones are what the planner and the channel decide.
    The codes are what ``proactive_log.reason`` holds, so ``twin proactive log`` and the audit
    count by them.
``Candidate``
    one message to consider: a row of ``proactive_candidates`` (fixed moments, follow-ups) or one
    drawn by the tick (``id`` is ``None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class TriggerKind(StrEnum):
    FOLLOWUP = "followup"
    GREETING = "greeting"
    MEAL = "meal"
    BEDTIME = "bedtime"
    SILENCE = "silence"
    SHARE = "share"
    EDGE = "edge"


PRIORITY: Mapping[TriggerKind, int] = {
    TriggerKind.FOLLOWUP: 0,
    TriggerKind.GREETING: 1,
    TriggerKind.MEAL: 2,
    TriggerKind.BEDTIME: 2,
    TriggerKind.SILENCE: 3,
    TriggerKind.SHARE: 4,
    TriggerKind.EDGE: 5,
}
ROUTINE_KINDS = frozenset({TriggerKind.GREETING, TriggerKind.MEAL, TriggerKind.BEDTIME})
DRAWN_KINDS = frozenset({TriggerKind.SILENCE, TriggerKind.SHARE, TriggerKind.EDGE})
KIND_LABELS: Mapping[str, str] = {
    TriggerKind.FOLLOWUP: "跟进",
    TriggerKind.GREETING: "起床问候",
    TriggerKind.MEAL: "饭点",
    TriggerKind.BEDTIME: "睡前",
    TriggerKind.SILENCE: "沉默",
    TriggerKind.SHARE: "分享",
    TriggerKind.EDGE: "睡不着/刚醒",
    "day": "当日开始",
}


class Reason(StrEnum):
    """Why a candidate did not go out (``proactive_log.reason``)."""

    # the hard constraints (R-PRO-003)
    DEEP_SLEEP = "deep_sleep"
    SLEEP_EDGE = "sleep_edge"  # at the edge of her sleep only the edge message may go out
    EDGE_WEEKLY = "edge_weekly"  # the weekly allowance of edge messages is used up
    STATE_CHANGED = "state_changed"  # an edge message, but she is no longer at the edge
    NO_PLAN = "no_plan"  # there is no day plan, so nobody knows whether she is asleep
    SPACING = "spacing"
    AWAITING_REPLY = "awaiting_reply"  # the last message is too recent to count as unanswered
    CHASE_LIMIT = "chase_limit"
    PAUSED = "paused"
    DISABLED = "disabled"
    WINDOW_CLOSED = "window_closed"  # the platform window is over (or never began)
    QUOTA_EXHAUSTED = "quota_exhausted"  # no message is left in the platform's count
    CHANNEL_UNAVAILABLE = "channel_unavailable"  # nobody bound, not logged in
    BUDGET = "budget"
    DAILY_MAX = "daily_max"
    QUOTA_SPENT = "quota_spent"  # the day's draw is used up: only follow-ups go out now
    GREETING_GAP = "greeting_gap"  # the 18-hour rule of the wake-up greeting (R-SCH-002)
    # what happens after the constraints
    USER_ACTIVE = "user_active"  # the user wrote while the message was being made
    PLANNER_DECLINED = "planner_declined"
    PLANNER_FAILED = "planner_failed"
    GENERATION_FAILED = "generation_failed"
    SEND_FAILED = "send_failed"
    SUPERSEDED = "superseded"  # the plan changed under it
    FOLLOWUP_CLOSED = "followup_closed"  # the user raised the thing himself: nothing to ask
    # why a candidate is void
    INTERRUPTED = "interrupted"  # the program was off or the machine slept (R-SCH-005)
    TIMEZONE_SWITCH = "timezone_switch"
    WINDOW_OVER = "window_over"  # its own time window passed


WINDOW_REASONS = frozenset({Reason.WINDOW_CLOSED, Reason.QUOTA_EXHAUSTED})
"""The reasons that count as "suppressed by the platform window" (R-PRO-003, R-EVAL-005)."""

OUTSIDE_REASONS = frozenset(
    {
        Reason.WINDOW_CLOSED,
        Reason.QUOTA_EXHAUSTED,
        Reason.CHANNEL_UNAVAILABLE,
        Reason.CHASE_LIMIT,
        Reason.PAUSED,
        Reason.DISABLED,
        Reason.BUDGET,
    }
)
"""Reasons the schedule cannot help: a day short of its range is excused by them (R-EVAL-005).

The platform window and count, the login, a pause or the switch, the budget, and a user who
does not answer (the chase limit stops her from writing again and again)."""

REASON_LABELS: Mapping[str, str] = {
    Reason.DEEP_SLEEP: "她在深睡",
    Reason.SLEEP_EDGE: "她快睡着或刚醒，只允许边缘消息",
    Reason.EDGE_WEEKLY: "本周的边缘消息已用完",
    Reason.STATE_CHANGED: "她已不在睡眠边缘",
    Reason.NO_PLAN: "没有日程",
    Reason.SPACING: "离上一条主动消息太近",
    Reason.AWAITING_REPLY: "上一条主动消息刚发出，还不算没回",
    Reason.CHASE_LIMIT: "已追发到上限",
    Reason.PAUSED: "暂停中",
    Reason.DISABLED: "主动消息已关闭",
    Reason.WINDOW_CLOSED: "被窗口抑制",
    Reason.QUOTA_EXHAUSTED: "平台条数用完",
    Reason.CHANNEL_UNAVAILABLE: "通道不可用",
    Reason.BUDGET: "预算降级，主动已暂停",
    Reason.DAILY_MAX: "今天已到上限",
    Reason.QUOTA_SPENT: "今天的配额用完（只剩跟进型）",
    Reason.GREETING_GAP: "起床问候距上次不足 18 小时",
    Reason.USER_ACTIVE: "用户正在聊天",
    Reason.PLANNER_DECLINED: "规划决定不发",
    Reason.PLANNER_FAILED: "规划失败",
    Reason.GENERATION_FAILED: "内容没有通过检查",
    Reason.SEND_FAILED: "发送失败",
    Reason.SUPERSEDED: "日程已重建",
    Reason.FOLLOWUP_CLOSED: "用户已经提过，不再问",
    Reason.INTERRUPTED: "中断期间过期",
    Reason.TIMEZONE_SWITCH: "切换时区后作废",
    Reason.WINDOW_OVER: "时间窗已过",
}


@dataclass(frozen=True)
class Candidate:
    """One message to consider (see the module description)."""

    kind: TriggerKind
    key: str
    planned_at: datetime
    window_end: datetime
    id: str | None = None
    attempts: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return PRIORITY[self.kind]

    @property
    def followup_id(self) -> str | None:
        found = self.detail.get("followup_id")
        return str(found) if found else None


@dataclass(frozen=True)
class Refusal:
    """A hard constraint that refuses a candidate, with a short note for the log."""

    reason: Reason
    detail: str | None = None
