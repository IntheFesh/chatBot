"""The daily backup and the monthly cost mail run by the clock (R-OPS-005, R-OPS-006)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from tests.support.ops import RecordingMailer
from tests.support.waiting import wait_until
from twin.engine.machine import STATE_SENDING
from twin.engine.state_store import ConversationStateStore
from twin.ops.backup.service import BackupBusyError, BackupError, BackupService
from twin.ops.recurring import Daily
from twin.ops.scheduler import (
    SENDING,
    OpsScheduler,
    OpsTask,
    TaskResult,
    backup_task,
    conversation_is_sending,
    cost_report_task,
)
from twin.services import Services
from twin.storage.settings_store import get_setting

CHICAGO = ZoneInfo("America/Chicago")


def at(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


class Counter:
    def __init__(self, *results: TaskResult) -> None:
        self.results = list(results)
        self.runs: list[datetime] = []

    async def __call__(self, due: datetime) -> TaskResult:
        self.runs.append(due)
        return self.results.pop(0) if self.results else TaskResult.DONE


def scheduler(services: Services, *tasks: OpsTask) -> OpsScheduler:
    return OpsScheduler(services, list(tasks), zone=lambda: CHICAGO)


async def test_a_task_runs_once_for_one_moment_also_after_a_restart(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(at(2026, 10, 9, 10))  # 05:00 Chicago: today's 04:00 has passed
    work = Counter()
    task = OpsTask("daily", Daily(4), work)
    assert await scheduler(services, task).tick() == ["daily"]
    assert await scheduler(services, task).tick() == []  # a restarted application: not again
    assert work.runs == [at(2026, 10, 9, 9)]
    clock.set_time(at(2026, 10, 10, 8))  # 03:00 the next morning: not due yet
    assert await scheduler(services, task).tick() == []
    clock.set_time(at(2026, 10, 10, 9, 1))
    assert await scheduler(services, task).tick() == ["daily"]
    assert len(work.runs) == 2


async def test_a_moment_missed_while_the_computer_was_off_is_made_up_once(
    services: Services, clock: ManualClock
) -> None:
    work = Counter()
    task = OpsTask("daily", Daily(4), work)
    clock.set_time(at(2026, 10, 9, 10))
    await scheduler(services, task).tick()
    clock.set_time(at(2026, 10, 12, 20))  # three days later, the application starts again
    assert await scheduler(services, task).tick() == ["daily"]
    assert await scheduler(services, task).tick() == []
    assert len(work.runs) == 2


async def test_a_task_that_does_not_catch_up_waits_for_its_first_moment(
    services: Services, clock: ManualClock
) -> None:
    work = Counter()
    task = OpsTask("monthly", Daily(4), work, catch_up=False)
    clock.set_time(at(2026, 10, 9, 10))
    assert await scheduler(services, task).tick() == []  # the moments before now are not owed
    clock.set_time(at(2026, 10, 10, 9, 1))
    assert await scheduler(services, task).tick() == ["monthly"]


async def test_a_postponed_task_comes_back_after_a_minute_and_a_failed_one_after_half_an_hour(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(at(2026, 10, 9, 10))
    work = Counter(TaskResult.POSTPONED, TaskResult.FAILED, TaskResult.DONE)
    runner = scheduler(services, OpsTask("daily", Daily(4), work))
    assert await runner.tick() == ["daily"]  # postponed
    clock.tick(30)
    assert await runner.tick() == []
    clock.tick(31)
    assert await runner.tick() == ["daily"]  # failed
    clock.tick(29 * 60)
    assert await runner.tick() == []
    clock.tick(61)
    assert await runner.tick() == ["daily"]  # done
    clock.tick(3600)
    assert await runner.tick() == []
    assert len(work.runs) == 3
    with services.db.session() as session:
        state = get_setting(session, "ops.task.daily")
    assert set(state) == {"done"}


async def test_the_component_ticks_with_the_clock(services: Services, clock: ManualClock) -> None:
    clock.set_time(at(2026, 10, 9, 10))
    work = Counter()
    runner = OpsScheduler(
        services, [OpsTask("daily", Daily(4), work)], zone=lambda: CHICAGO, tick_s=30
    )
    await runner.start()
    try:
        await wait_until(lambda: len(work.runs) == 1)
        await clock.advance(30)
    finally:
        await runner.stop()
    assert len(work.runs) == 1 and runner.health().status.value == "ok"


# ----------------------------------------------------------------------- the backup


class FakeBackup:
    """Stands in for :class:`BackupService` (its own tests make real backups)."""

    def __init__(self, *outcomes: Exception | None) -> None:
        self.outcomes = list(outcomes)
        self.kinds: list[str] = []

    def create(self, kind: str = "daily") -> None:
        self.kinds.append(kind)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


async def test_the_backup_waits_for_a_quiet_moment_but_not_forever(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(at(2026, 10, 9, 9, 5))  # five minutes after the due moment
    sending = [True]
    fake = FakeBackup()
    task = backup_task(services, fake, busy=lambda: sending[0])  # type: ignore[arg-type]
    due = at(2026, 10, 9, 9)
    assert await task.run(due) is TaskResult.POSTPONED and fake.kinds == []
    sending[0] = False
    assert await task.run(due) is TaskResult.DONE and fake.kinds == ["daily"]
    sending[0] = True
    clock.set_time(at(2026, 10, 9, 11, 1))  # two hours past the moment (backup_postpone_max_h)
    assert await task.run(due) is TaskResult.DONE and fake.kinds == ["daily", "daily"]


async def test_the_backup_reports_busy_and_failure(services: Services, clock: ManualClock) -> None:
    clock.set_time(at(2026, 10, 9, 9, 5))
    fake = FakeBackup(BackupBusyError("busy"), BackupError("failed"))
    task = backup_task(services, fake, busy=lambda: False)  # type: ignore[arg-type]
    assert await task.run(at(2026, 10, 9, 9)) is TaskResult.POSTPONED
    assert await task.run(at(2026, 10, 9, 9)) is TaskResult.FAILED


def test_the_engine_state_decides_whether_a_reply_is_being_sent(services: Services) -> None:
    assert not conversation_is_sending(services)
    assert SENDING == STATE_SENDING
    ConversationStateStore(services.db, services.clock).transition(STATE_SENDING)
    assert conversation_is_sending(services)


def test_the_backup_task_is_due_at_the_configured_hour(services: Services) -> None:
    services.settings.ops.backup_hour_local = 5
    task = backup_task(services, BackupService.from_services(services))
    assert task.rule == Daily(5) and task.name == "backup"


# ----------------------------------------------------------------------- the report


async def test_the_cost_report_goes_out_on_the_first_at_nine(
    services: Services, clock: ManualClock
) -> None:
    services.settings.ops.smtp.to = "me@example.org"
    mailer = RecordingMailer()
    task = cost_report_task(services, mailer)
    clock.set_time(at(2026, 10, 1, 14, 5))  # 09:05 Chicago on the 1st
    assert await task.run(at(2026, 10, 1, 14)) is TaskResult.DONE
    (mail,) = mailer.sent
    assert mail.subject == "[wechat-twin] 费用月报 2026-09" and mail.to == "me@example.org"
    assert "费用报告 2026-09" in mail.text and mail.html is not None


async def test_the_cost_report_is_retried_when_the_mail_fails_and_given_up_after_days(
    services: Services, clock: ManualClock
) -> None:
    services.settings.ops.smtp.to = "me@example.org"
    mailer = RecordingMailer()
    mailer.errors = ["network"]
    task = cost_report_task(services, mailer)
    clock.set_time(at(2026, 10, 1, 14, 5))
    assert await task.run(at(2026, 10, 1, 14)) is TaskResult.FAILED
    assert await task.run(at(2026, 10, 1, 14)) is TaskResult.DONE and len(mailer.sent) == 1
    mailer.errors = ["network"]
    clock.set_time(at(2026, 10, 5, 14))
    assert await task.run(at(2026, 10, 1, 14)) is TaskResult.DONE  # three days: given up
    assert len(mailer.sent) == 1


async def test_no_mail_account_means_no_report_and_no_retrying(
    services: Services, clock: ManualClock
) -> None:
    for mailer in (None, RecordingMailer(configured=False), RecordingMailer()):
        task = cost_report_task(services, mailer)
        clock.set_time(at(2026, 10, 1, 14, 5))
        assert await task.run(at(2026, 10, 1, 14)) is TaskResult.DONE  # no recipient configured


def test_the_task_list_of_a_running_application_has_the_two_jobs(
    services: Services, clock: ManualClock
) -> None:
    mail = cost_report_task(services, RecordingMailer())
    assert mail.name == "cost_report" and mail.catch_up is False
    assert mail.retry[TaskResult.FAILED] == timedelta(hours=1)
    with pytest.raises(KeyError):
        mail.retry[TaskResult.DONE]
