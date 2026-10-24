"""Before the bot is ready she writes nothing first (R18-B8, D-872).

The production scheduler (``tests.support.proactive_world``) on a database without her profile,
routine model or described persona card: every candidate is refused ``not_ready``, nothing is
drawn, the day is opened as one she may not write - and once the three exist the same world
writes as before, with no switch flipped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.proactive_world import (
    Always,
    ProactiveScript,
    World,
    opening_curve,
    proactive_model,
)
from tests.support.proactive_world import build_world as build_proactive_world
from tests.support.readiness import install_card, install_profile, install_ready, install_routine
from twin.profile.persona.refresh import refresh_stats
from twin.schedule.plan_builder import QuotaRange
from twin.schedule.proactive.types import TriggerKind
from twin.services import Services


@pytest.fixture
async def unready(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[World]:
    """The ``eager`` world of the draws (every draw succeeds), before anything was imported."""
    built = build_proactive_world(
        services,
        clock,
        script=pro_script,
        model=proactive_model(opening_curve(base=0.0, peaks={})),
        quota=QuotaRange(2, 2),
        rng=Always(),
        ready=False,
    )
    yield built
    await built.aclose()


async def test_r18_b8_a_due_candidate_is_refused_not_ready_and_the_planner_is_not_asked(
    unready: World,
) -> None:
    unready.user_writes(at=unready.at(20, 0, day=8))
    unready.put(TriggerKind.FOLLOWUP, unready.at(10, 0))
    report = await unready.tick_at(unready.at(10, 5))
    assert report.outcome == "rejected" and report.reason == "not_ready", report
    assert unready.channel.out == [] and unready.script.requests == []
    assert unready.reasons() == ["not_ready"]


async def test_r18_b8_nothing_is_drawn_and_the_day_is_opened_as_one_she_may_not_write(
    unready: World,
) -> None:
    unready.user_writes(at=unready.at(20, 0, day=8))
    for minute in range(0, 60, 5):  # an hour of ticks with draws that always succeed
        report = await unready.tick_at(unready.at(11, minute))
        assert report.candidate is None and not report.drawn, report
    assert unready.sent_rows() == [] and unready.script.requests == []
    opened = unready.log.entries(outcomes=["opened"])
    assert len(opened) == 1 and opened[0].enabled is False


async def test_r18_b8_a_card_without_its_description_is_not_enough(unready: World) -> None:
    """The profile and routine exist, the card has statistics only: still ``not_ready`` (D-870);
    with its description she writes."""
    install_profile(unready.services)
    install_routine(unready.services, unready.rig.holder["model"])
    assert refresh_stats(unready.services, "live").status == "created"
    unready.user_writes(at=unready.at(20, 0, day=8))
    unready.put(TriggerKind.SHARE, unready.at(10, 0))
    report = await unready.tick_at(unready.at(10, 5))
    assert report.reason == "not_ready", report
    assert unready.sent_rows() == [] and unready.script.requests == []
    install_card(unready.services)
    report = await unready.tick_at(unready.at(10, 10))
    assert report.outcome == "sent", report


async def test_r18_b8_once_ready_she_writes_as_before(unready: World) -> None:
    """Ready (the three stored): the same world draws and sends; nothing was switched off."""
    unready.user_writes(at=unready.at(20, 0, day=8))
    first = await unready.tick_at(unready.at(10, 0))
    assert first.candidate is None
    install_ready(unready.services, model=unready.rig.holder["model"])
    report = await unready.tick_at(unready.at(10, 30))
    assert report.drawn and report.outcome == "sent", report
    assert len(unready.sent_rows()) == 1
