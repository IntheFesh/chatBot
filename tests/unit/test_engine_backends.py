"""The engine asks the backend chooser for every reply and tells it how the reply went (R-SRV-004).

The chooser here is a double; the real selector and the style model are in
``test_engine_wiring_e2e.py``.  What is checked is the engine's side of the arrangement: which
backend goes into the context, what is reported back, and that a chooser that breaks never costs
a reply.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from tests.support.clock import ManualClock
from tests.support.engine_harness import Harness, build_harness, make_draft, run_to_idle
from tests.support.waiting import wait_until
from twin.config.runtime import BACKEND_ACTIVE
from twin.engine.backend_select import BackendChoice
from twin.engine.types import ReplyDraft
from twin.services import Services


@dataclass
class ScriptedChooser:
    """A chooser whose answer the test sets; it keeps what the engine told it."""

    name: str = "style"
    requested: str = "style"
    reason: str | None = None
    choose_error: Exception | None = None
    record_error: Exception | None = None
    chosen: int = 0
    recorded: list[tuple[str, ReplyDraft]] = field(default_factory=list)

    async def choose(self) -> BackendChoice:
        self.chosen += 1
        if self.choose_error is not None:
            raise self.choose_error
        return BackendChoice(self.name, self.requested, self.reason)

    def record(self, backend: str, draft: ReplyDraft) -> None:
        if self.record_error is not None:
            raise self.record_error
        self.recorded.append((backend, draft))


async def talk(harness: Harness, text: str = "在吗") -> None:
    await harness.message(text)
    await run_to_idle(harness.engine, harness.clock)


async def test_the_backend_of_the_reply_is_the_choosers_and_the_outcome_is_reported_back(
    services: Services, clock: ManualClock
) -> None:
    chooser = ScriptedChooser("hybrid", "hybrid")
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        draft = make_draft("好呀", backend="hybrid")
        harness.writer.add(draft)
        await talk(harness)
        assert harness.writer.contexts[0].backend == "hybrid"
        assert chooser.recorded == [("hybrid", draft)]  # the draft as the pipeline made it
    finally:
        await harness.engine.stop()


async def test_a_fallback_choice_is_what_the_pipeline_is_told_to_use(
    services: Services, clock: ManualClock
) -> None:
    chooser = ScriptedChooser("deepseek", "hybrid", "fallback:unhealthy")
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀"))
        await talk(harness)
        assert harness.writer.contexts[0].backend == "deepseek"
        assert [name for name, _ in chooser.recorded] == ["deepseek"]
    finally:
        await harness.engine.stop()


async def test_without_a_chooser_the_setting_decides(
    services: Services, clock: ManualClock
) -> None:
    services.runtime.set(BACKEND_ACTIVE, "hybrid", by="command")
    harness = build_harness(services, clock)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀"))
        await talk(harness)
        assert harness.writer.contexts[0].backend == "hybrid"
    finally:
        await harness.engine.stop()


async def test_a_chooser_that_breaks_means_deepseek_and_the_reply_goes_out(
    services: Services, clock: ManualClock
) -> None:
    chooser = ScriptedChooser(choose_error=RuntimeError("the registry is gone"))
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀"))
        await talk(harness)
        assert harness.writer.contexts[0].backend == "deepseek"
        assert harness.channel.texts == ["好呀"]
    finally:
        await harness.engine.stop()


async def test_a_chooser_that_cannot_take_the_record_does_not_cost_the_reply(
    services: Services, clock: ManualClock
) -> None:
    chooser = ScriptedChooser(record_error=RuntimeError("the books are locked"))
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀"))
        await talk(harness)
        assert harness.channel.texts == ["好呀"] and harness.engine.snapshot().state == "IDLE"
    finally:
        await harness.engine.stop()


async def test_a_generation_the_user_cancelled_is_not_reported_only_the_one_that_was_sent(
    services: Services, clock: ManualClock
) -> None:
    chooser = ScriptedChooser("deepseek", "deepseek")
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        harness.writer.gate = asyncio.Event()
        harness.writer.add(make_draft("第二版"))  # the cancelled call never took its answer
        await harness.message("周末去看电影吧")
        await run_to_idle(harness.engine, clock, until=harness.writer.started.is_set)
        await harness.message("算了我们明天去")  # the first generation is cancelled
        await wait_until(lambda: harness.writer.cancelled == 1)
        harness.writer.gate.set()
        await run_to_idle(harness.engine, clock)
        assert chooser.chosen == 2  # asked before each generation ...
        assert len(chooser.recorded) == 1  # ... told only about the one that came back
        assert harness.channel.texts == ["第二版"]
    finally:
        await harness.engine.stop()


@pytest.mark.parametrize("name", ["deepseek", "style", "hybrid"])
async def test_each_backend_name_reaches_the_context_unchanged(
    services: Services, clock: ManualClock, name: str
) -> None:
    chooser = ScriptedChooser(name, name)
    harness = build_harness(services, clock, backends=chooser)
    await harness.engine.start()
    try:
        harness.writer.add(make_draft("好呀", backend=name))
        await talk(harness)
        assert harness.writer.contexts[0].backend == name
    finally:
        await harness.engine.stop()
