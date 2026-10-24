"""The probe runner when something unexpected happens around it (R-CH-009, R-ARCH-004)."""

from __future__ import annotations

import asyncio

import pytest

from tests.support.probe import PlatformSim, Rig, make_rig
from twin.channel.base import (
    BypassGrant,
    BypassRefused,
    BypassRequest,
    ChannelError,
    OutboundResult,
    RecipientNotAllowed,
    SendBypass,
)
from twin.channel.probe.model import (
    ActionKind,
    ActionStatus,
    AttemptPhase,
    PlanStatus,
    ProbePlan,
    StepId,
    StepStatus,
)
from twin.channel.probe.runner import PlanChanged, ProbeRunner
from twin.channel.probe.store import NoProbePlan
from twin.channel.probe.summary import load_channel_probe_summary
from twin.storage.db import Database


class RefuseAll:
    """A policy that never authorises."""

    def authorize(self, request: BypassRequest) -> BypassGrant | None:
        raise BypassRefused("not today")


class RefuseMeasuring:
    """Authorises the announcements but not the measuring sends."""

    def authorize(self, request: BypassRequest) -> BypassGrant | None:
        if request.text is not None and "条数测试" in request.text:
            raise BypassRefused("not this one")
        return None


class RaisingSim(PlatformSim):
    """A platform whose sends raise instead of answering."""

    raises: Exception | None = None

    async def send_text(self, text: str, *, bypass: SendBypass) -> OutboundResult:
        if self.raises is not None:
            raise self.raises
        return await super().send_text(text, bypass=bypass)


async def open_and_announce(rig: Rig) -> None:
    await rig.runner.tick()  # opens the attempt
    await rig.runner.tick()  # announces it


# -------------------------------------------------------------- the announcement


async def test_a_refused_announcement_means_the_plan_changed_under_the_runner(db: Database) -> None:
    rig = make_rig(db)
    rig.runner = ProbeRunner(
        store=rig.store,
        channel=rig.sim,
        policy=RefuseAll(),
        clock=rig.clock,
        alerts=rig.alerts,
        banner=rig.banner,
    )
    await rig.runner.tick()
    with pytest.raises(PlanChanged):
        await rig.runner.tick()
    assert rig.sim.log == []


async def test_nobody_bound_stops_the_plan_with_that_reason(db: Database) -> None:
    rig = make_rig(db)
    sim = RaisingSim(rig.clock, {})
    sim.raises = RecipientNotAllowed("no user is bound")
    rig.runner = ProbeRunner(
        store=rig.store,
        channel=sim,
        policy=rig.policy,
        clock=rig.clock,
        alerts=rig.alerts,
        banner=rig.banner,
    )
    await rig.runner.tick()
    with pytest.raises(PlanChanged):
        await rig.runner.tick()
    plan = rig.store.load()
    assert plan is not None and plan.status is PlanStatus.STOPPED
    assert plan.stop_reason == "no user is bound to the channel"
    with db.session() as session:
        assert load_channel_probe_summary(session) is not None


async def test_an_announcement_that_cannot_be_sent_is_noted_and_the_probe_still_waits(
    db: Database,
) -> None:
    rig = make_rig(db)
    sim = RaisingSim(rig.clock, {})
    sim.raises = ChannelError("something in the channel")
    rig.runner = ProbeRunner(
        store=rig.store,
        channel=sim,
        policy=rig.policy,
        clock=rig.clock,
        alerts=rig.alerts,
        banner=rig.banner,
    )
    await open_and_announce(rig)
    plan = rig.store.load()
    assert plan is not None
    attempt = plan.step(StepId.COUNT).attempts[0]
    assert attempt.phase is AttemptPhase.WAITING
    assert attempt.announce_note == "not sent (ChannelError)"


# --------------------------------------------------------------- the measuring


async def test_a_send_the_policy_refuses_voids_the_attempt_instead_of_being_retried(
    db: Database,
) -> None:
    rig = make_rig(db)
    rig.runner = ProbeRunner(
        store=rig.store,
        channel=rig.sim,
        policy=RefuseMeasuring(),
        clock=rig.clock,
        alerts=rig.alerts,
        banner=rig.banner,
    )
    plan = await rig.tick_until(lambda p: p.step(StepId.COUNT).voided_attempts == 1)
    first = plan.step(StepId.COUNT).attempts[0]
    refused = next(a for a in first.actions if a.send and a.send.outcome == "refused")
    assert refused.send is not None and refused.send.reason == "BypassRefused"
    assert refused.status is ActionStatus.FAILED
    assert "refused" in (first.void_reason or "")
    assert not any(e.text and "条数测试" in e.text for e in rig.sim.log)  # nothing left the machine


# ---------------------------------------------------------------- the loop itself


async def test_the_loop_waits_a_second_after_a_plan_changed_and_keeps_going(db: Database) -> None:
    rig = make_rig(db)
    calls: list[str] = []

    async def tick() -> float:
        calls.append("tick")
        if len(calls) == 1:
            raise PlanChanged("changed")
        if len(calls) == 2:
            raise NoProbePlan("gone")
        if len(calls) == 3:
            return 0.5
        raise asyncio.CancelledError

    rig.runner.tick = tick  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await rig.runner.run_forever()
    assert rig.clock.sleeps == [1.0, 1.0, 0.5]


async def test_without_a_running_plan_the_runner_idles(db: Database) -> None:
    rig = make_rig(db, start=False)
    assert await rig.runner.tick() == 5.0  # no plan at all
    plan = rig.store.create()
    rig.store.finish(PlanStatus.COMPLETED, None)
    assert await rig.runner.tick() == plan.options.idle_poll_s


async def test_a_plan_whose_steps_are_all_done_is_completed_by_the_next_tick(db: Database) -> None:
    rig = make_rig(db)

    def all_done(plan: ProbePlan) -> None:
        for step in plan.steps:
            step.status = StepStatus.SKIPPED

    rig.store.update(all_done)
    await rig.runner.tick()
    plan = rig.store.load()
    assert plan is not None and plan.status is PlanStatus.COMPLETED
    assert any("finished" in title for title, _lines in rig.banner.shown)


# ---------------------------------------------------- stale views of the plan


async def test_decisions_based_on_a_stale_view_of_the_plan_are_dropped(db: Database) -> None:
    rig = make_rig(db)
    await rig.runner.tick()  # an attempt is open
    stale = rig.store.load()
    assert stale is not None
    step = stale.step(StepId.COUNT)
    with pytest.raises(PlanChanged, match="already open"):
        await rig.runner._open_attempt(stale, step)
    rig.store.update(lambda p: setattr(p.step(StepId.COUNT), "status", StepStatus.DONE))
    with pytest.raises(PlanChanged, match="already finished"):
        await rig.runner._open_attempt(stale, step)
    with pytest.raises(PlanChanged, match="no longer exists"):
        ProbeRunner._find(stale, StepId.COUNT, 9)
    attempt = step.attempts[0]
    with pytest.raises(PlanChanged, match="no longer exists"):
        ProbeRunner._find_action(attempt, "send:99")


async def test_nothing_is_sent_for_a_plan_that_was_stopped_after_the_decision(
    db: Database,
) -> None:
    rig = make_rig(db)
    plan = await rig.tick_until(
        lambda p: (
            any(
                a.kind is ActionKind.SEND_TEXT and a.status is ActionStatus.PENDING
                for a in (
                    p.step(StepId.COUNT).attempts[0].actions
                    if p.step(StepId.COUNT).attempts
                    else []
                )
            )
            and p.step(StepId.COUNT).attempts[0].phase is AttemptPhase.RUNNING
        )
    )
    step = plan.step(StepId.COUNT)
    attempt = step.attempts[0]
    action = next(a for a in attempt.actions if a.status is ActionStatus.PENDING)
    rig.store.finish(PlanStatus.STOPPED, "stopped by the user")
    logged = len(rig.sim.log)
    with pytest.raises(PlanChanged, match="stopped"):
        await rig.runner._send(plan, step, attempt, action)
    typing = action
    with pytest.raises(PlanChanged, match="stopped"):
        await rig.runner._typing(plan, step, attempt, typing)
    assert len(rig.sim.log) == logged


async def test_voiding_or_finishing_an_attempt_twice_changes_nothing(db: Database) -> None:
    rig = make_rig(db)
    plan = await rig.tick_until(
        lambda p: (
            bool(p.step(StepId.COUNT).attempts)
            and p.step(StepId.COUNT).attempts[0].phase is AttemptPhase.RUNNING
        )
    )
    step = plan.step(StepId.COUNT)
    attempt = step.attempts[0]
    await rig.runner._void(plan, step, attempt, "first reason")
    await rig.runner._void(plan, step, attempt, "second reason")
    await rig.runner._finish_attempt(plan, step, attempt)
    stored = rig.store.load()
    assert stored is not None
    assert stored.step(StepId.COUNT).attempts[0].void_reason == "first reason"
