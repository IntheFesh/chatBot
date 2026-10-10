"""R-EVAL-004: the audit as a weekly, off-peak, offline job.

Once a week the operations scheduler queues the job; the worker runs it only in the cheap hours
(or when its day is up); the audit finds contradictions and tells the user they wait - it changes
no memory and sends no message.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.consistency_world import AuditModel, build_week, the_two_contradictions
from tests.support.deepseek import API
from tests.support.ops import RecordingMailer, RecordingNotifier
from tests.support.policies import AlwaysOffPeak, NeverOffPeak
from twin.app import Application
from twin.eval.consistency_jobs import (
    CONSISTENCY_JOB,
    CONSISTENCY_PRIORITY,
    CONSISTENCY_RULE,
    FORCE_AFTER,
    REVIEW_ALERT,
    audit_is_waiting,
    handle_consistency,
    queue_consistency_job,
)
from twin.eval.store import EvalStore
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.ops.jobs import HANDLER_MODULES, HandlerRegistry, JobQueue, Worker, load_handlers
from twin.ops.jobs import default_registry as handlers
from twin.ops.recurring import Weekly, latest_due
from twin.ops.scheduler import OpsScheduler, consistency_task
from twin.ops.wiring import register_ops
from twin.services import Services

CHICAGO = ZoneInfo("America/Chicago")


@pytest.fixture
def model() -> AuditModel:
    return AuditModel(rules=the_two_contradictions())


@pytest.fixture
def api(model: AuditModel) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=model)
        yield router


def worker(services: Services, policy: AlwaysOffPeak | NeverOffPeak) -> Worker:
    registry = HandlerRegistry()
    registry.register(CONSISTENCY_JOB, handle_consistency)
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=policy,
        alerts=services.alerts,
    )


# ------------------------------------------------------------------------- queueing


def test_queueing_makes_one_off_peak_job_that_is_forced_after_a_day(services: Services) -> None:
    job_id = queue_consistency_job(services)
    assert job_id is not None and audit_is_waiting(services)
    job = JobQueue(services.db, services.clock).get(job_id)
    assert job is not None and job.type == CONSISTENCY_JOB and job.status == "pending"
    assert job.offpeak_only and job.priority == CONSISTENCY_PRIORITY and job.max_attempts == 3
    assert job.deadline == services.clock.now_utc() + FORCE_AFTER
    assert job.payload == {"days": 7} and not job.requires_approval
    assert queue_consistency_job(services) is None  # one waits already
    assert queue_consistency_job(services, days=14) is None
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type=CONSISTENCY_JOB)) == 1


def test_the_window_comes_from_the_setting_or_the_caller(services: Services) -> None:
    services.settings.eval.consistency_days = 10
    first = queue_consistency_job(services)
    assert first is not None
    queue = JobQueue(services.db, services.clock)
    assert queue.get(first).payload == {"days": 10}  # type: ignore[union-attr]
    queue.cancel(first)
    second = queue_consistency_job(services, days=21)
    assert second is not None and queue.get(second).payload == {"days": 21}  # type: ignore[union-attr]


def test_the_handler_is_among_those_the_worker_loads() -> None:
    assert "twin.eval.consistency_jobs" in HANDLER_MODULES
    load_handlers()
    assert handlers.has(CONSISTENCY_JOB)


# --------------------------------------------------------------------------- running


async def test_the_job_audits_and_tells_the_user_that_contradictions_wait(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    queue_consistency_job(services)
    summary = await worker(services, AlwaysOffPeak()).run_until_idle()
    assert summary.done == 1 and summary.failed == 0 and len(model.requests) == 1
    run = EvalStore(services.db, services.clock).list_runs("consistency")[0]
    assert run.status == "running" and run.summary["decisions"]["undecided"] == 2
    alerts = [a for a in services.alerts.recent() if a.category == REVIEW_ALERT]
    assert len(alerts) == 1 and alerts[0].severity == "info"
    assert (
        "2 contradiction" in alerts[0].title and "twin eval consistency --review" in alerts[0].title
    )
    assert alerts[0].detail == {"run_id": run.id, "findings": 2}
    assert alerts[0].toast_state == "none" and alerts[0].mail_state == "none"  # recorded only
    for chat in ("图书馆", "整天都在家", "上海"):
        assert chat not in alerts[0].title and chat not in str(alerts[0].detail)


async def test_a_week_with_no_contradiction_is_closed_without_a_word(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    model.rules = []
    queue_consistency_job(services)
    await worker(services, AlwaysOffPeak()).run_until_idle()
    run = EvalStore(services.db, services.clock).list_runs("consistency")[0]
    assert run.status == "done" and run.verdict == "passed"
    assert [a for a in services.alerts.recent() if a.category == REVIEW_ALERT] == []


async def test_the_job_waits_for_the_cheap_hours_until_its_day_is_up(
    services: Services, clock: ManualClock, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    queue_consistency_job(services)
    runner = worker(services, NeverOffPeak())
    assert (await runner.run_until_idle()).done == 0 and model.requests == []
    clock.tick(FORCE_AFTER.total_seconds() - 60)
    assert (await runner.run_until_idle()).done == 0  # still a minute short
    clock.tick(120)
    assert (await runner.run_until_idle()).done == 1 and len(model.requests) == 1


async def test_a_call_the_budget_or_the_breaker_holds_back_is_tried_again_later(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def budget(*_: object, **__: object) -> None:
        raise BudgetDeniedError("consistency", 4)

    async def circuit(*_: object, **__: object) -> None:
        raise CircuitOpenError(300.0)

    queue_consistency_job(services)
    runner = worker(services, AlwaysOffPeak())
    monkeypatch.setattr("twin.eval.consistency_jobs.run_audit", budget)
    assert (await runner.run_until_idle()).deferred == 1
    queue = JobQueue(services.db, services.clock)
    job = queue.list_jobs(job_type=CONSISTENCY_JOB)[0]
    assert job.status == "pending" and job.attempts == 0
    assert job.run_after == services.clock.now_utc() + timedelta(hours=1)
    clock.tick(3601)
    monkeypatch.setattr("twin.eval.consistency_jobs.run_audit", circuit)
    assert (await runner.run_until_idle()).deferred == 1
    again = queue.list_jobs(job_type=CONSISTENCY_JOB)[0]
    assert again.run_after == services.clock.now_utc() + timedelta(minutes=5)


async def test_a_job_without_the_services_fails_loudly() -> None:
    from twin.ops.jobs import JobContext, JobView

    view = JobView(
        id="j",
        type=CONSISTENCY_JOB,
        payload={},
        priority=1,
        status="running",
        attempts=1,
        max_attempts=3,
        run_after=datetime(2026, 10, 9, tzinfo=UTC),
        offpeak_only=True,
        deadline=None,
        last_error=None,
        batch_id=None,
        estimated_cost_usd=None,
        requires_approval=False,
        approved_at=None,
        approved_usd=None,
        started_at=None,
        finished_at=None,
        created_at=datetime(2026, 10, 9, tzinfo=UTC),
        updated_at=datetime(2026, 10, 9, tzinfo=UTC),
    )
    context = JobContext(job=view, services=None, clock=ManualClock())
    with pytest.raises(RuntimeError, match="services"):
        await handle_consistency(context)


# ------------------------------------------------------------------------- the week


def at(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_the_rule_is_monday_at_twenty_past_four_on_the_bots_clock() -> None:
    assert Weekly(weekday=0, hour=4, minute=20) == CONSISTENCY_RULE
    friday = at(2026, 10, 9, 12)  # 07:00 in Chicago
    assert latest_due(CONSISTENCY_RULE, friday, CHICAGO) == at(2026, 10, 5, 9, 20)
    assert latest_due(CONSISTENCY_RULE, at(2026, 10, 12, 9, 19), CHICAGO) == at(2026, 10, 5, 9, 20)
    assert latest_due(CONSISTENCY_RULE, at(2026, 10, 12, 9, 20), CHICAGO) == at(2026, 10, 12, 9, 20)


async def test_the_scheduler_queues_the_audit_once_a_week(
    services: Services, clock: ManualClock
) -> None:
    runner = OpsScheduler(services, [consistency_task(services)], zone=lambda: CHICAGO)
    queue = JobQueue(services.db, services.clock)
    assert await runner.tick() == []  # first sight: the Mondays before now are not owed
    assert queue.list_jobs(job_type=CONSISTENCY_JOB) == []
    clock.set_time(at(2026, 10, 12, 9, 19))  # a minute before
    assert await runner.tick() == []
    clock.set_time(at(2026, 10, 12, 9, 21))  # 04:21 on Monday
    assert await runner.tick() == ["consistency_audit"]
    assert len(queue.list_jobs(job_type=CONSISTENCY_JOB)) == 1
    clock.set_time(at(2026, 10, 14, 12))
    assert await runner.tick() == []  # once for one Monday
    clock.set_time(at(2026, 10, 19, 9, 30))
    assert await runner.tick() == ["consistency_audit"]
    assert len(queue.list_jobs(job_type=CONSISTENCY_JOB)) == 1  # the last one is still waiting


async def test_the_application_has_the_weekly_task(services: Services) -> None:
    kit = register_ops(
        Application(), services, mailer=RecordingMailer(), notifier=RecordingNotifier()
    )
    names = [task.name for task in kit.scheduler._tasks]
    assert "consistency_audit" in names and {"backup", "cost_report"} <= set(names)


async def test_a_monday_missed_while_the_computer_was_off_is_made_up_once(
    services: Services, clock: ManualClock
) -> None:
    runner = OpsScheduler(services, [consistency_task(services)], zone=lambda: CHICAGO)
    await runner.tick()  # first sight, on the Friday before
    clock.set_time(at(2026, 10, 15, 12))  # the application starts again on Thursday
    assert await runner.tick() == ["consistency_audit"]  # Monday's audit, late
    assert await runner.tick() == []
    assert len(JobQueue(services.db, services.clock).list_jobs(job_type=CONSISTENCY_JOB)) == 1
