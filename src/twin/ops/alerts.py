"""Alerts: the hook other rounds call and the service behind it (R-ARCH-003, R-OPS-004).

Round 00 defined the hook (:class:`AlertSink`) that the job worker, the task supervisors and
every component call when something needs attention.  Round 12 puts :class:`AlertService`
behind it - the only code that writes the ``alerts`` table:

* every alert is **recorded**, under one of the twenty categories of :class:`AlertCategory` (the
  names the earlier rounds use are mapped onto them, see :func:`canonical_category`; the ones
  that have no place among the twenty keep their own name and are recorded only, unless they are
  critical);
* an alert that is worth telling the user about is **delivered** by the delivery component
  (:mod:`twin.ops.alert_delivery`): a Windows notification and an e-mail.  At most one notice
  per category goes out per ``ops.alert_cooldown_min`` (an hour); a repeat is stored as
  ``suppressed`` and a worse one (higher severity) is let through;
* when the cause is gone, :meth:`AlertService.recover` closes the open alerts of the category and
  queues **one** "recovered" notice - only if the alert it closes was announced;
* the text that leaves the machine is made from fixed wording and numbers
  (:mod:`twin.ops.alert_text`), never from chat content.

The service only decides and records; sending is somebody else's job, so raising an alert never
waits for a mail server and never fails the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import delete, or_, select, update

from twin.clock import Clock
from twin.ops.logging import get_logger
from twin.storage.db import Database
from twin.storage.models import ALERT_SEVERITIES, Alert

log = get_logger("twin.alerts")

SEVERITY_RANK = {severity: rank for rank, severity in enumerate(ALERT_SEVERITIES)}
ALERT_KEEP_DAYS = 90
STALE_AFTER = timedelta(hours=24)  # a notice that was never delivered is not sent a day later
CLAIM_LEASE = timedelta(minutes=2)  # a delivery attempt that did not report back is retried


class AlertSink(Protocol):
    """Receives operational alerts."""

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None: ...


class AlertCategory(StrEnum):
    """The kinds of alert the user is told about (R-OPS-004)."""

    LOGIN_LOST = "login_lost"
    DEEPSEEK_FAILURE = "deepseek_failure"
    CIRCUIT_OPEN = "circuit_open"
    BUDGET_80 = "budget_80"
    BUDGET_LEVEL_N = "budget_level_n"
    ONE_TIME_OVERRUN = "one_time_overrun"
    STYLE_MODEL_DOWN = "style_model_down"
    STYLE_TOKENIZE_MISMATCH = "style_tokenize_mismatch"
    BACKUP_FAILED = "backup_failed"
    BACKUP_MIRROR_UNAVAILABLE = "backup_mirror_unavailable"
    DISK_LOW = "disk_low"
    QUEUE_BACKLOG = "queue_backlog"
    RETRAIN_SUGGESTED = "retrain_suggested"
    CRISIS_DETECTED = "crisis_detected"
    CHANNEL_WINDOW_UNEXPECTED = "channel_window_unexpected"
    CHANNEL_POLL_STALE = "channel_poll_stale"
    CALENDAR_OUT_OF_RANGE = "calendar_out_of_range"
    SLEEP_TIMEZONE_SUSPECT = "sleep_timezone_suspect"
    PROCESS_RESTARTED = "process_restarted"
    SYSTEM_RESUMED = "system_resumed"


@dataclass(frozen=True)
class AlertSpec:
    """How one category reads and whether it is announced.

    ``notify`` ``None`` means "only when the alert is critical".  ``show_title`` false keeps the
    caller's one-line title out of the outgoing text (the category's wording says it all).
    """

    label: str
    advice: str
    notify: bool | None = True
    show_title: bool = True


SPECS: dict[str, AlertSpec] = {
    "login_lost": AlertSpec(
        "微信登录失效",
        "电脑上会弹出二维码窗口，请用手机微信扫码重新登录（二维码不会通过邮件发送）。",
    ),
    "deepseek_failure": AlertSpec(
        "DeepSeek 连续出错",
        "请检查 API Key、账户余额和网络；机器人在恢复前会用兜底话术回复。",
    ),
    "circuit_open": AlertSpec(
        "DeepSeek 已熔断",
        "连续失败触发了 5 分钟熔断；若反复出现，请检查网络和 DeepSeek 服务状态。",
    ),
    "budget_80": AlertSpec("费用达到预算的 80%", "可以用 twin cost report 查看花在哪里。"),
    "budget_level_n": AlertSpec(
        "费用超出预算，已降级",
        "机器人会按顺序降级（关思考、减例子、停主动消息……），但不会停止回复。",
    ),
    "one_time_overrun": AlertSpec(
        "一次性任务费用超出估算", "该批任务已暂停；检查后用 twin jobs approve 重新批准，或取消。"
    ),
    "style_model_down": AlertSpec(
        "风格模型不可用", "回复已回退到 DeepSeek；模型恢复后会自动切回。"
    ),
    "style_tokenize_mismatch": AlertSpec(
        "风格模型分词核对不一致", "模型与提示词模板对不上，暂不能启用；请重新核对训练时的模板版本。"
    ),
    "backup_failed": AlertSpec(
        "备份失败或已过期", "用 twin backup now 手动备份一次，并查看日志里的原因。"
    ),
    "backup_mirror_unavailable": AlertSpec(
        "异地备份目录不可用", "本地备份不受影响；请检查外接盘或网络位置是否在线。"
    ),
    "disk_low": AlertSpec("磁盘剩余空间不足", "请清理磁盘，数据库、媒体和备份都在数据目录里。"),
    "queue_backlog": AlertSpec(
        "后台任务积压", "用 twin jobs list 查看；可能是 DeepSeek 不可用或某类任务一直失败。"
    ),
    "retrain_suggested": AlertSpec(
        "建议重新训练风格模型", "她的新消息已达到上次训练数据的一成以上。"
    ),
    "crisis_detected": AlertSpec(
        "检测到可能需要关心的信号",
        "机器人已跳出角色并给出求助渠道；请你亲自联系对方。（这里不会出现聊天内容。）",
        show_title=False,
    ),
    "channel_window_unexpected": AlertSpec(
        "微信会话窗口意外失效", "请在手机上给机器人发一条消息，会话就会恢复。"
    ),
    "channel_poll_stale": AlertSpec(
        "微信长轮询中断", "请检查电脑的网络连接；恢复后会自动继续，并通知你已恢复。"
    ),
    "calendar_out_of_range": AlertSpec(
        "节假日日历没有覆盖当前年份", "高峰时段按周一至周五估算；升级 chinese-calendar 依赖即可。"
    ),
    "sleep_timezone_suspect": AlertSpec(
        "作息推断可能受时区影响", "请用 twin profile show 与 twin timezone show 核对作息与时区。"
    ),
    "process_restarted": AlertSpec(
        "机器人进程重启了", "监督进程已自动重启它；若短时间内反复出现，请查看日志。"
    ),
    "system_resumed": AlertSpec(
        "电脑从睡眠中恢复", "机器人睡着期间不会收发消息；请检查电源计划是否允许睡眠。"
    ),
    # categories of the earlier rounds that are not among the twenty
    "integrity_failed": AlertSpec(
        "数据完整性检查发现问题", "请立即用 twin backup now 备份，并查看 twin health 的详情。"
    ),
    "llm_ledger": AlertSpec(
        "DeepSeek 费用没有记上账", "请查看日志；费用统计和预算在恢复前可能偏低。"
    ),
    "channel_send_failed": AlertSpec("微信发送失败", "回复发不出去；请查看 twin channel status。"),
    "channel_unbound": AlertSpec("微信没有绑定用户", "请运行 twin channel login 完成绑定。"),
    "retrieval_index": AlertSpec("检索库需要重建", "请运行 twin retrieval rebuild。"),
    "emergency_contact": AlertSpec("紧急联系人提醒没有发出", "请检查邮件设置。", show_title=False),
    "task_crashed": AlertSpec("后台任务崩溃", "已自动重启；反复出现请查看日志。", notify=None),
    "engine_error": AlertSpec("回复引擎出错", "已自动恢复；反复出现请查看日志。", notify=None),
    "job_failed": AlertSpec(
        "后台任务失败", "用 twin jobs list --status failed 查看。", notify=False
    ),
    "reply_failed": AlertSpec("一次回复没有发出", "已用延迟回复降级。", notify=False),
    "channel_quota_exhausted": AlertSpec("微信条数用完", "等你发下一条消息后恢复。", notify=False),
    "power_resume_failed": AlertSpec("唤醒后的处理失败", "请查看日志。", notify=False),
    "learning_fact_corrections": AlertSpec("有纠正是关于事实的", "请改用 /记住。", notify=False),
    "dpo_ready": AlertSpec("偏好对已足够做 DPO", "可以运行 twin train export-dpo。", notify=False),
    "lifeline_corrected": AlertSpec("今天的生活线被修正过", "仅供参考。", notify=False),
    "channel_probe": AlertSpec("通道探针的进展", "查看 twin channel probe status。", notify=False),
}

# the names the earlier rounds use  ->  the category they belong to
ALIASES: dict[str, str] = {
    "channel.auth_expired": AlertCategory.LOGIN_LOST,
    "channel.poll_failing": AlertCategory.CHANNEL_POLL_STALE,
    "channel.probe": "channel_probe",
    "channel_session_expired": AlertCategory.CHANNEL_WINDOW_UNEXPECTED,
    "llm_auth": AlertCategory.DEEPSEEK_FAILURE,
    "llm_balance": AlertCategory.DEEPSEEK_FAILURE,
    "llm_circuit": AlertCategory.CIRCUIT_OPEN,
    "batch_overrun": AlertCategory.ONE_TIME_OVERRUN,
    "style_fallback": AlertCategory.STYLE_MODEL_DOWN,
    "crisis": AlertCategory.CRISIS_DETECTED,
    "routine_timezone": AlertCategory.SLEEP_TIMEZONE_SUSPECT,
}

# a "recovered" notice of the earlier rounds  ->  the category it closes
RECOVERIES: dict[str, str] = {
    "channel.auth_recovered": AlertCategory.LOGIN_LOST,
    "style_recovered": AlertCategory.STYLE_MODEL_DOWN,
}


def canonical_category(raw: str, detail: dict[str, Any] | None = None) -> str:
    """The category an alert is recorded under (the budget one depends on its detail)."""
    if raw == "budget":
        level = (detail or {}).get("level")
        return AlertCategory.BUDGET_LEVEL_N if level is not None else AlertCategory.BUDGET_80
    return str(ALIASES.get(raw, raw))


def spec_of(category: str) -> AlertSpec:
    """The wording and policy of a category (unknown names are recorded and read as they are)."""
    return SPECS.get(category) or AlertSpec(category, "请查看日志。", notify=None)


def is_announced(category: str, severity: str) -> bool:
    """Whether an alert of this category and severity is told to the user."""
    notify = spec_of(category).notify
    return severity == "critical" if notify is None else notify


@dataclass(frozen=True)
class AlertView:
    """A row of ``alerts`` as the delivery side reads it."""

    id: str
    category: str
    kind: str
    severity: str
    title: str
    detail: dict[str, Any] | None
    created_at: datetime
    toast_state: str
    mail_state: str
    toast_at: datetime | None
    mail_at: datetime | None
    mail_attempts: int
    mail_next_at: datetime | None
    resolved_at: datetime | None
    suppressed: bool
    notified_at: datetime | None = None


def _view(row: Alert) -> AlertView:
    return AlertView(
        id=row.id,
        category=row.category,
        kind=row.kind,
        severity=row.severity,
        title=row.title,
        detail=dict(row.detail) if row.detail else None,
        created_at=row.created_at,
        toast_state=row.toast_state,
        mail_state=row.mail_state,
        toast_at=row.toast_at,
        mail_at=row.mail_at,
        mail_attempts=row.mail_attempts,
        mail_next_at=row.mail_next_at,
        resolved_at=row.resolved_at,
        suppressed=row.suppressed,
        notified_at=row.notified_at,
    )


class AlertService:
    """Records alerts, decides which are announced, closes them again (see the module text)."""

    def __init__(
        self,
        db: Database,
        clock: Clock,
        *,
        cooldown_min: float = 60.0,
        wake: Callable[[], None] | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._cooldown = timedelta(minutes=cooldown_min)
        self._wake = wake

    def set_wake(self, wake: Callable[[], None] | None) -> None:
        """The delivery component asks to be told when something is waiting for it."""
        self._wake = wake

    # ------------------------------------------------------------------- raising

    def raise_alert(
        self,
        category: str,
        title: str,
        *,
        severity: str = "warning",
        detail: dict[str, Any] | None = None,
        dedup_key: str | None = None,
    ) -> None:
        if severity not in ALERT_SEVERITIES:
            raise ValueError(f"unknown alert severity {severity!r}")
        closes = RECOVERIES.get(category)
        if closes is not None:
            self.recover(closes, title, detail=detail)
            return
        canonical = canonical_category(category, detail)
        announce = is_announced(canonical, severity)
        now = self._clock.now_utc()
        suppressed = False
        with self._db.transaction(bump_state=False) as session:
            if announce:
                suppressed = self._held_back(session, canonical, severity, now)
            wanted = announce and not suppressed
            session.add(
                Alert(
                    category=canonical,
                    kind="alert",
                    severity=severity,
                    title=title[:200],
                    detail=detail,
                    dedup_key=dedup_key,
                    suppressed=suppressed,
                    toast_state="pending" if wanted else "none",
                    mail_state="pending" if wanted else "none",
                    created_at=now,
                    updated_at=now,
                )
            )
        log.warning(
            "alert",
            category=canonical,
            severity=severity,
            alert_title=title[:200],
            announced=announce and not suppressed,
        )
        if announce and not suppressed:
            self._poke()

    def _held_back(self, session: Any, category: str, severity: str, now: datetime) -> bool:
        """True if an announcement of this category went out within the cooldown (and this one
        is not worse than it)."""
        since = now - self._cooldown
        last = session.scalars(
            select(Alert)
            .where(
                Alert.category == category,
                Alert.kind == "alert",
                Alert.suppressed.is_(False),
                Alert.resolved_at.is_(None),
                Alert.created_at > since,
                or_(Alert.toast_state != "none", Alert.mail_state != "none"),
            )
            .order_by(Alert.created_at.desc())
            .limit(1)
        ).first()
        if last is None:
            return False
        return SEVERITY_RANK[severity] <= SEVERITY_RANK[last.severity]

    def recover(self, category: str, title: str, *, detail: dict[str, Any] | None = None) -> bool:
        """The cause of ``category`` is gone: close its alerts, announce it once.

        Returns ``True`` if a "recovered" notice was queued (there was an announced alert to
        close).
        """
        canonical = canonical_category(RECOVERIES.get(category, category), detail)
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            open_rows = list(
                session.scalars(
                    select(Alert).where(
                        Alert.category == canonical,
                        Alert.kind == "alert",
                        Alert.resolved_at.is_(None),
                    )
                )
            )
            if not open_rows:
                return False
            announced = any(
                not row.suppressed and (row.toast_state != "none" or row.mail_state != "none")
                for row in open_rows
            )
            for row in open_rows:
                row.resolved_at = now
            if not announced:
                return False
            session.add(
                Alert(
                    category=canonical,
                    kind="recovery",
                    severity="info",
                    title=title[:200],
                    detail=detail,
                    dedup_key=None,
                    suppressed=False,
                    toast_state="pending",
                    mail_state="pending",
                    created_at=now,
                    updated_at=now,
                )
            )
        log.info("alert_recovered", category=canonical)
        self._poke()
        return True

    def _poke(self) -> None:
        wake = self._wake
        if wake is not None:
            try:
                wake()
            except Exception:  # the delivery side being gone must not fail the caller
                log.warning("alert_wake_failed")

    # ------------------------------------------------------------------ delivery

    def claim_due(self, *, limit: int = 20) -> list[AlertView]:
        """Alerts with a channel still to deliver, leased to the caller for two minutes."""
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            rows = list(
                session.scalars(
                    select(Alert)
                    .where(
                        or_(Alert.toast_state == "pending", Alert.mail_state == "pending"),
                        or_(Alert.claimed_at.is_(None), Alert.claimed_at < now - CLAIM_LEASE),
                        or_(
                            Alert.mail_next_at.is_(None),
                            Alert.mail_next_at <= now,
                            Alert.toast_state == "pending",
                        ),
                    )
                    .order_by(Alert.created_at)
                    .limit(limit)
                )
            )
            claimed: list[AlertView] = []
            for row in rows:
                if now - row.created_at > STALE_AFTER:
                    row.toast_state = "failed" if row.toast_state == "pending" else row.toast_state
                    row.mail_state = "failed" if row.mail_state == "pending" else row.mail_state
                    row.mail_error = "expired"
                    continue
                row.claimed_at = now
                claimed.append(_view(row))
            return claimed

    def release(self, alert_id: str) -> None:
        """Give a leased alert back untouched (its mail is not due yet)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(Alert, alert_id)
            if row is not None:
                row.claimed_at = None

    def finish_toast(self, alert_id: str, *, ok: bool | None) -> None:
        """Record the outcome of the notification: sent, failed, or ``None`` (not available)."""
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(Alert, alert_id)
            if row is None:
                return
            row.toast_state = "none" if ok is None else ("sent" if ok else "failed")
            if ok:
                row.toast_at = now
                row.notified_at = row.notified_at or now
            row.claimed_at = None

    def finish_mail(
        self,
        alert_id: str,
        *,
        ok: bool | None,
        error: str | None = None,
        retry_in: timedelta | None = None,
    ) -> None:
        """Record the outcome of the e-mail: sent, not wanted (``None``), or failed.

        A failure with ``retry_in`` stays pending and is tried again then; without it the mail
        is given up.
        """
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(Alert, alert_id)
            if row is None:
                return
            if ok is None:
                row.mail_state = "none"
            elif ok:
                row.mail_state = "sent"
                row.mail_at = now
                row.notified_at = row.notified_at or now
                row.mail_error = None
            else:
                row.mail_attempts += 1
                row.mail_error = (error or "error")[:64]
                if retry_in is None:
                    row.mail_state = "failed"
                else:
                    row.mail_next_at = now + retry_in
            row.claimed_at = None

    # ------------------------------------------------------------------- reading

    def pending_count(self) -> int:
        with self._db.session() as session:
            rows = session.scalars(
                select(Alert.id).where(
                    or_(Alert.toast_state == "pending", Alert.mail_state == "pending")
                )
            )
            return len(list(rows))

    def recent(self, *, limit: int = 20, since: datetime | None = None) -> list[AlertView]:
        stmt = select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc()).limit(limit)
        if since is not None:
            stmt = stmt.where(Alert.created_at >= since)
        with self._db.session() as session:
            return [_view(row) for row in session.scalars(stmt)]

    def open_alerts(self) -> list[AlertView]:
        """Announced alerts whose cause has not been reported gone yet."""
        with self._db.session() as session:
            rows = session.scalars(
                select(Alert)
                .where(
                    Alert.kind == "alert",
                    Alert.resolved_at.is_(None),
                    Alert.suppressed.is_(False),
                    or_(Alert.toast_state != "none", Alert.mail_state != "none"),
                )
                .order_by(Alert.created_at)
            )
            return [_view(row) for row in rows]

    def prune(self, *, keep_days: int = ALERT_KEEP_DAYS) -> int:
        """Delete alerts older than ``keep_days``; returns how many."""
        cutoff = self._clock.now_utc() - timedelta(days=keep_days)
        with self._db.transaction(bump_state=False) as session:
            result = session.execute(delete(Alert).where(Alert.created_at < cutoff))
            return int(getattr(result, "rowcount", 0) or 0)

    def reset_claims(self) -> None:
        """Forget unfinished leases (the delivery component calls it when it starts)."""
        with self._db.transaction(bump_state=False) as session:
            session.execute(
                update(Alert).where(Alert.claimed_at.is_not(None)).values(claimed_at=None)
            )
