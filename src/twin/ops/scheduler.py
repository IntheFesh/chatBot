"""The jobs that run by the clock: backup, monthly cost report, the weekly consistency audit
(R-OPS-005, R-OPS-006, R-EVAL-004).

:class:`OpsScheduler` is a component of the application.  Every 30 seconds it asks, for each
:class:`OpsTask`, "was the latest due moment of its rule served yet?"
(:func:`~twin.ops.recurring.latest_due`), runs the task if not, and remembers the answer in the
``settings`` table (``ops.task.<name>``: when it last succeeded, when it last tried and when it
may try again).  So:

* a task whose moment passed while the computer was off runs once when the application starts
  (the backup: ``catch_up``) - or is skipped (the cost report: it would be a report of a month
  nobody asked about);
* a task that is not done yet is tried again after a pause that depends on how it ended: a
  backup that had to wait for a quiet moment after a minute, one that failed after half an hour;
* nothing runs twice for one moment, also across restarts.

The daily backup **avoids a conversation that is sending**: while the engine is in its sending
state the task asks to be postponed, for up to ``ops.backup_postpone_max_h`` hours after the
moment it was due, then it runs anyway (a backup that never happens is worse).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from twin.app import ComponentHealth, TaskSupervisor
from twin.engine.state_store import ConversationStateStore
from twin.eval.consistency_jobs import CONSISTENCY_RULE, queue_consistency_job
from twin.ops.backup.service import BackupBusyError, BackupError, BackupService
from twin.ops.cost import build_report, previous_month, render_html, render_text
from twin.ops.logging import get_logger
from twin.ops.mail import Mailer, MailError, OutgoingMail
from twin.ops.recurring import Daily, Monthly, Rule, latest_due
from twin.services import Services
from twin.storage.settings_store import get_setting, put_setting

log = get_logger("twin.ops.scheduler")

TICK_S = 30.0
STATE_PREFIX = "ops.task."
SENDING = "SENDING"
COST_REPORT_GIVE_UP = timedelta(days=3)
SUBJECT = "[wechat-twin] 费用月报 {month}"


class TaskResult(StrEnum):
    DONE = "done"
    POSTPONED = "postponed"  # not now, soon
    FAILED = "failed"


@dataclass
class OpsTask:
    """One recurring job: its rule and what to do."""

    name: str
    rule: Rule
    run: Callable[[datetime], Awaitable[TaskResult]]
    catch_up: bool = True
    retry: dict[TaskResult, timedelta] = field(
        default_factory=lambda: {
            TaskResult.POSTPONED: timedelta(minutes=1),
            TaskResult.FAILED: timedelta(minutes=30),
        }
    )


def conversation_is_sending(services: Services) -> bool:
    """True while the engine is sending the bubbles of a reply."""
    return ConversationStateStore(services.db, services.clock).load().state == SENDING


class OpsScheduler:
    """Runs the :class:`OpsTask` list by the clock (see the module description)."""

    name = "ops_scheduler"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        services: Services,
        tasks: Sequence[OpsTask],
        *,
        zone: Callable[[], ZoneInfo],
        tick_s: float = TICK_S,
    ) -> None:
        self._services = services
        self._clock = services.clock
        self._tasks = list(tasks)
        self._zone = zone
        self._tick_s = tick_s
        self._supervisor = TaskSupervisor(self.name, self._clock, services.alerts)

    # -------------------------------------------------------------------- state

    def _load(self, name: str) -> dict[str, str]:
        with self._services.db.session() as session:
            value = get_setting(session, STATE_PREFIX + name)
        return dict(value) if isinstance(value, dict) else {}

    def _save(self, name: str, state: dict[str, str]) -> None:
        with self._services.db.transaction(bump_state=False) as session:
            put_setting(
                session,
                STATE_PREFIX + name,
                state,
                clock=self._clock,
                by="ops",
                record_history=False,
            )

    @staticmethod
    def _when(state: dict[str, str], key: str) -> datetime | None:
        raw = state.get(key)
        return datetime.fromisoformat(raw) if raw else None

    # --------------------------------------------------------------------- a pass

    async def tick(self) -> list[str]:
        """Run every task that is due; returns the names of those that were started."""
        started: list[str] = []
        for task in self._tasks:
            if await self._tick_task(task):
                started.append(task.name)
        return started

    async def _tick_task(self, task: OpsTask) -> bool:
        now = self._clock.now_utc()
        due = latest_due(task.rule, now, self._zone())
        state = await asyncio.to_thread(self._load, task.name)
        done = self._when(state, "done")
        if done is None and not state and not task.catch_up:
            await asyncio.to_thread(self._save, task.name, {"done": due.isoformat()})
            return False  # first sight: the moments before now are not owed
        if done is not None and done >= due:
            return False
        retry_at = self._when(state, "retry_at")
        if retry_at is not None and now < retry_at:
            return False
        result = await task.run(due)
        new = dict(state)
        if result is TaskResult.DONE:
            new = {"done": self._clock.now_utc().isoformat()}
        else:
            new["tried"] = now.isoformat()
            new["retry_at"] = (now + task.retry[result]).isoformat()
        await asyncio.to_thread(self._save, task.name, new)
        log.info("ops_task", task=task.name, result=result.value)
        return True

    # ---------------------------------------------------------------- component

    async def _loop(self) -> None:
        while True:
            await self.tick()
            await self._clock.sleep(self._tick_s)

    async def start(self) -> None:
        self._supervisor.spawn("tick", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()


# ------------------------------------------------------------------ the tasks


def backup_task(
    services: Services,
    backup: BackupService,
    *,
    busy: Callable[[], bool] | None = None,
) -> OpsTask:
    """The daily backup at ``ops.backup_hour_local`` (see the module description)."""
    config = services.settings.ops
    is_busy = busy or (lambda: conversation_is_sending(services))
    postpone_max = timedelta(hours=config.backup_postpone_max_h)

    async def run(due: datetime) -> TaskResult:
        waited = services.clock.now_utc() - due
        if waited < postpone_max and await asyncio.to_thread(is_busy):
            return TaskResult.POSTPONED
        try:
            await asyncio.to_thread(backup.create, "daily")
        except BackupBusyError:
            return TaskResult.POSTPONED
        except BackupError:
            return TaskResult.FAILED
        return TaskResult.DONE

    return OpsTask("backup", Daily(config.backup_hour_local), run)


def cost_report_task(services: Services, mailer: Mailer | None) -> OpsTask:
    """The report of the month that ended, e-mailed on the 1st at 09:00 local time."""
    from twin.schedule.service import time_service_for

    time = time_service_for(services)

    def make() -> tuple[str, str, str]:
        from twin.llm.ledger import LedgerStore

        ledger = LedgerStore(services.db, services.clock, time)
        month = previous_month(time.local_date())
        report = build_report(ledger, time, services.settings.budget, month)
        return SUBJECT.format(month=report.month), render_text(report), render_html(report)

    async def run(due: datetime) -> TaskResult:
        recipient = services.settings.ops.smtp.to
        if mailer is None or not mailer.configured or not recipient:
            log.info("cost_report_not_sent", reason="no_mail_account")
            return TaskResult.DONE
        if services.clock.now_utc() - due > COST_REPORT_GIVE_UP:
            return TaskResult.DONE
        subject, text, page = await asyncio.to_thread(make)
        try:
            await asyncio.to_thread(mailer.send, OutgoingMail(recipient, subject, text, page))
        except MailError as exc:
            log.warning("cost_report_mail_failed", reason=exc.code)
            return TaskResult.FAILED
        return TaskResult.DONE

    return OpsTask(
        "cost_report",
        Monthly(1, 9),
        run,
        catch_up=False,
        retry={
            TaskResult.POSTPONED: timedelta(minutes=5),
            TaskResult.FAILED: timedelta(hours=1),
        },
    )


def consistency_task(services: Services) -> OpsTask:
    """Every Monday at 04:20 on the bot's clock: queue the audit of the bot against itself.

    The audit itself is an off-peak job (:mod:`twin.eval.consistency_jobs`); this only queues it,
    so a week in which the computer was off at that moment is simply not audited (the next audit
    looks back over seven days again).
    """

    async def run(due: datetime) -> TaskResult:
        job_id = await asyncio.to_thread(queue_consistency_job, services)
        log.info("consistency_audit_queued" if job_id else "consistency_audit_already_waiting")
        return TaskResult.DONE

    return OpsTask("consistency_audit", CONSISTENCY_RULE, run, catch_up=False)
