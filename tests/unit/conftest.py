"""Fixtures of the evaluation tests (round 09b) and of the proactive tests (round 10)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.deepseek import API
from tests.support.embedding import HashingBackend
from tests.support.eval_world import DeepSeekScript, build_eval_world, mock_deepseek, open_kit
from tests.support.export_world import World
from tests.support.proactive_world import (
    Always,
    Never,
    ProactiveScript,
    opening_curve,
    proactive_model,
)
from tests.support.proactive_world import World as ProactiveWorld
from tests.support.proactive_world import build_world as build_proactive_world
from twin.eval.sandbox import SandboxKit
from twin.schedule.plan_builder import QuotaRange
from twin.services import Services


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    """A conversation with everything a reply reads, and a hold-out wide enough to draw from."""
    return build_eval_world(services, embedder)


@pytest.fixture
def script() -> DeepSeekScript:
    """What DeepSeek answers: two bubbles, the second with an emoji code."""
    return DeepSeekScript(lambda body: "好呀\n哈哈[拥抱]")


@pytest.fixture
def api(script: DeepSeekScript) -> Iterator[respx.MockRouter]:
    yield from mock_deepseek(script)


@pytest.fixture
async def kit(world: World, api: respx.MockRouter) -> AsyncIterator[SandboxKit]:
    """A hold-out sandbox that books its calls on the batch ``eval-test-1``."""
    async with open_kit(world) as built:
        yield built


@pytest.fixture
def pro_script() -> ProactiveScript:
    """What DeepSeek answers the planner of the proactive messages."""
    return ProactiveScript()


@pytest.fixture
def pro_api(pro_script: ProactiveScript) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=pro_script)
        yield router


@pytest.fixture
async def pro(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[ProactiveWorld]:
    """The production scheduler over the synthetic world (``tests.support.proactive_world``)."""
    built = build_proactive_world(services, clock, script=pro_script)
    yield built
    await built.aclose()


@pytest.fixture
async def calm(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[ProactiveWorld]:
    """The same world on a day with no fixed moments but the greeting and a draw of six.

    Her history shows no conversation she opened at a mealtime or at goodnight, so nothing but
    the wake-up greeting is laid out: a test puts the candidate it wants and nothing else
    is due.
    """
    built = build_proactive_world(
        services,
        clock,
        script=pro_script,
        model=proactive_model(opening_curve(base=0.0, peaks={})),
        quota=QuotaRange(6, 6),
        rng=Never(),
    )
    yield built
    await built.aclose()


@pytest.fixture
async def eager(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[ProactiveWorld]:
    """A world whose draws always succeed (two messages a day) and who opens nothing at night."""
    built = build_proactive_world(
        services,
        clock,
        script=pro_script,
        model=proactive_model(opening_curve(base=0.0, peaks={})),
        quota=QuotaRange(2, 2),
        rng=Always(),
    )
    yield built
    await built.aclose()


@pytest.fixture
async def eager_night(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[ProactiveWorld]:
    """As ``eager``, but her history has some late-night openings: the edge of sleep can draw."""
    built = build_proactive_world(
        services,
        clock,
        script=pro_script,
        model=proactive_model(opening_curve(base=0.05, peaks={})),
        quota=QuotaRange(2, 2),
        rng=Always(),
    )
    yield built
    await built.aclose()


@pytest.fixture
async def full(
    services: Services, clock: ManualClock, pro_script: ProactiveScript, pro_api: respx.MockRouter
) -> AsyncIterator[ProactiveWorld]:
    """A world whose day always has its greeting, three meals and goodnight; nothing is drawn."""
    built = build_proactive_world(
        services,
        clock,
        script=pro_script,
        model=proactive_model(opening_curve(base=2.0, peaks={})),
        quota=QuotaRange(6, 6),
        rng=Never(),
    )
    yield built
    await built.aclose()
