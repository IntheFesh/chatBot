"""The random side of her day: what a refused draw does next, the silence she breaks, the edges."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tests.support.proactive_world import World
from twin.channel.base import OutboundKind, OutboundResult
from twin.profile.distribution import EmpiricalDistribution
from twin.schedule.proactive.types import TriggerKind
from twin.schedule.time_service import PlanUnavailableError

FRIDAY = date(2026, 10, 9)


async def test_a_refused_draw_rests_but_the_rest_ends_when_the_constraint_lets_go(
    eager: World,
) -> None:
    eager.user_writes(at=eager.at(10, 0, day=8))  # the window of the platform closes Friday 08:00
    first = await eager.tick_at(eager.at(10, 0))
    assert first.drawn and first.reason == "window_closed", first
    for minute in (5, 10, 15):
        assert (await eager.tick_at(eager.at(10, minute))).candidate is None
    assert eager.reasons() == ["window_closed"]  # the refusal was written once
    eager.user_writes(at=eager.at(10, 20))
    assert (await eager.tick_at(eager.at(10, 25))).skipped == "user_active"
    report = await eager.tick_at(eager.at(10, 35))  # the window is open again: no two hours' rest
    assert report.drawn and report.outcome == "sent", report


async def test_a_drawn_message_the_channel_refused_leaves_the_draw_resting(eager: World) -> None:
    eager.user_writes(at=eager.at(20, 0, day=8))
    eager.channel.fail_with = OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "window_elapsed")
    report = await eager.tick_at(eager.at(10, 0))
    assert report.drawn and report.outcome == "rejected" and report.reason == "window_closed"
    for minute in (5, 10, 15, 20):
        assert (await eager.tick_at(eager.at(10, minute))).candidate is None
    assert len(eager.script.requests) == 1 and eager.sent_rows() == []


async def test_a_drawn_message_dropped_for_the_user_leaves_nothing_behind(eager: World) -> None:
    eager.user_writes(at=eager.at(20, 0, day=8))
    eager.script.on_request = lambda _kind: eager.talk.user_wrote()
    report = await eager.tick_at(eager.at(10, 0))
    assert report.drawn and report.outcome == "dropped" and report.reason == "user_active"
    assert eager.candidates.pending() == [] and eager.channel.out == []
    assert eager.log.entries(outcomes=["dropped"])[0].reason == "user_active"


@pytest.mark.parametrize(
    ("typical", "quiet_for", "kind"),
    [
        (7200.0, 95, "share"),  # two hours is her usual silence: ninety minutes is not enough
        (7200.0, 150, "silence"),
        (600.0, 70, "silence"),  # never shorter than a conversation segment (one hour) ...
        (600.0, 50, "share"),  # ... however short the silences in her history are
    ],
)
async def test_the_silence_that_brings_a_message_is_drawn_from_her_history(
    eager: World, typical: float, quiet_for: int, kind: str
) -> None:
    distribution = EmpiricalDistribution.from_counter({typical: 10}, discrete=True)
    eager.scheduler._silence_source = lambda: distribution  # type: ignore[attr-defined]
    now = eager.at(10, 5)
    eager.user_writes(at=now - timedelta(minutes=quiet_for))
    report = await eager.tick_at(now)
    assert report.drawn and report.candidate is not None
    assert report.candidate.kind.value == kind, report


async def test_at_the_waking_edge_the_planner_is_told_she_has_just_woken(
    eager_night: World,
) -> None:
    world = eager_night
    world.user_writes(at=world.at(20, 0, day=8))
    plan = world.rig.planner.ensure(FRIDAY)
    assert plan.morning is not None
    edges = [(a, b) for kind, a, b in plan.morning.intervals() if kind == "sleep_edge"]
    report = await world.tick_at(edges[1][0] + timedelta(minutes=2))
    assert report.drawn and report.outcome == "sent", report
    assert report.candidate is not None and report.candidate.detail["side"] == "waking"
    assert "她刚醒" in str(world.script.requests[-1]["messages"][-1]["content"])
    assert world.sent_rows()[0].her_state == "sleep_edge"


async def test_without_a_day_plan_nobody_can_tell_if_she_sleeps_so_nothing_is_sent(
    calm: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    calm.user_writes(at=calm.at(20, 0, day=8))
    calm.put(TriggerKind.SHARE, calm.at(10, 0))

    def nobody_knows(_moment: object) -> object:
        raise PlanUnavailableError("no plan")

    monkeypatch.setattr(calm.rig.kit.time, "her_state", nobody_knows)
    report = await calm.tick_at(calm.at(10, 5))
    assert report.outcome == "rejected" and report.reason == "no_plan", report
    assert calm.channel.out == [] and calm.script.requests == []
