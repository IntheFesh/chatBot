"""Fixtures of the end-to-end scenarios: a clock that can live through days, and the world."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import respx

from tests.support.embedding import HashingBackend
from tests.support.life_clock import LifeClock
from tests.support.life_world import LifeWorld
from twin.clock import set_active_clock
from twin.services import Services

WorldFactory = Callable[..., Awaitable[LifeWorld]]


@pytest.fixture
def clock() -> LifeClock:
    """The clock of the scenarios (the ``services`` fixture builds the container on it)."""
    life = LifeClock()
    set_active_clock(life)
    return life


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def make_world(
    services: Services,
    clock: LifeClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    tmp_path: Path,
) -> AsyncIterator[WorldFactory]:
    """``await make_world(start=..., ...)``: the world, closed again when the test is over."""
    made: list[LifeWorld] = []

    async def build(start: datetime | None = None, **options: Any) -> LifeWorld:
        workdir = tmp_path / "pictures"
        workdir.mkdir(exist_ok=True)
        world = await LifeWorld.create(
            services, clock, embedder, api, workdir=workdir, start=start, **options
        )
        made.append(world)
        return world

    yield build
    for world in reversed(made):
        await world.close()
