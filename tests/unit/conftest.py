"""Fixtures of the evaluation tests (round 09b): a synthetic world and a scripted DeepSeek."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import respx

from tests.support.embedding import HashingBackend
from tests.support.eval_world import DeepSeekScript, build_eval_world, mock_deepseek, open_kit
from tests.support.export_world import World
from twin.eval.sandbox import SandboxKit
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
