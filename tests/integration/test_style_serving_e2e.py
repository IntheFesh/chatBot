"""Ten turns with a local model behind the terminal channel (R-SRV-002, R-ENG-006, R-SRV-005).

What is real: the terminal channel (``twin chat --local``), the engine, the prompts, the hybrid
backend (DeepSeek plans, the model writes), the backend selector, the HTTP client of the style
model and the server process, which ``LlamaServerManager`` starts with the very command line of
``llama-server``.  The server is the simulated one (``tests/support/llama_server_sim.py``); the
second test runs the same ten turns against a real ``llama-server`` and a real GGUF and is skipped
unless ``TWIN_LLAMA_SERVER`` and ``TWIN_LLAMA_GGUF`` name them.

Both record what a user would feel - the time the model needs for a reply and how often its
output broke a rule - in the test report (``record_property``) and in a JSON file beside the
test's data, where ``docs/SERVING_NOTES.md`` says to look for the numbers of a real run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import statistics
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.engine_harness import run_to_idle
from tests.support.llama_sim import free_port, make_model_file, spec_for
from tests.support.style_models import register_model
from tests.support.waiting import wait_until
from tests.unit.test_engine_wiring_e2e import PLAN, World, assemble, close, stored_card
from twin.clock import SystemClock
from twin.config.runtime import BACKEND_ACTIVE
from twin.engine.types import ReplyDraft
from twin.llm.errors import StyleModelError
from twin.llm.style_client import (
    LlamaCppCompletionClient,
    RenderedPrompt,
    StyleHealth,
    StyleOutput,
    StyleParams,
)
from twin.services import Services
from twin.serving.llamacpp import ServerSpec
from twin.serving.server import LlamaServerManager, ServerState, ServerTimings

pytestmark = pytest.mark.integration

TURNS = 10
QUESTIONS = [
    "周末去看电影吧",
    "你想看哪部",
    "几点比较好",
    "要不要先吃饭",
    "我请你吃火锅",
    "你最近忙不忙",
    "今天下班早吗",
    "那我六点去接你",
    "记得带伞",
    "晚安",
]
TIMINGS = ServerTimings(
    start_timeout_s=600.0,
    backoff_start_s=0.5,
    backoff_max_s=5.0,
    stable_after_s=60.0,
    health_interval_s=1.0,
    unhealthy_limit=5,
    stop_grace_s=10.0,
)


class MeasuredClient:
    """The real HTTP client of the style model; keeps what each call took and how it ended."""

    def __init__(self, inner: LlamaCppCompletionClient) -> None:
        self._inner = inner
        self.outputs: list[StyleOutput] = []
        self.errors = 0

    async def generate(self, prompt: RenderedPrompt, params: StyleParams) -> StyleOutput:
        try:
            output = await self._inner.generate(prompt, params)
        except StyleModelError:
            self.errors += 1
            raise
        self.outputs.append(output)
        return output

    async def health(self) -> StyleHealth:
        return await self._inner.health()

    async def tokenize(self, text: str) -> list[int]:
        return await self._inner.tokenize(text)

    async def aclose(self) -> None:
        await self._inner.aclose()


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.route(host="127.0.0.1").pass_through()  # the model's server is real
        yield router


@contextlib.asynccontextmanager
async def serving(spec: ServerSpec, tmp_path: Path) -> AsyncIterator[MeasuredClient]:
    """The server process under the manager of the application, until the block ends."""
    port = spec.port
    inner = LlamaCppCompletionClient(f"http://127.0.0.1:{port}", clock=SystemClock())
    manager = LlamaServerManager(
        spec,
        clock=SystemClock(),
        client=inner,
        log_path=tmp_path / "logs" / "llama-server.log",
        timings=TIMINGS,
    )
    task = asyncio.create_task(manager.run())
    try:
        await wait_until(
            lambda: manager.state is ServerState.READY, limit_s=TIMINGS.start_timeout_s
        )
        yield MeasuredClient(
            LlamaCppCompletionClient(f"http://127.0.0.1:{port}", clock=SystemClock())
        )
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 30)
        await inner.aclose()


def rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


async def ten_turns(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    client: MeasuredClient,
    tmp_path: Path,
) -> tuple[World, dict[str, Any]]:
    """Ten messages to a chat whose answers come from the hybrid backend; the numbers of it."""
    stored_card(services)
    register_model(services, persona_version="v1")
    world = await assemble(services, clock, embedder, api, monkeypatch, style_client=client)
    drafts: list[tuple[str, ReplyDraft]] = []
    record = world.style.selector.record

    def spy(backend: str, draft: ReplyDraft) -> None:
        drafts.append((backend, draft))
        record(backend, draft)

    monkeypatch.setattr(world.style.selector, "record", spy)
    world.services.runtime.set(BACKEND_ACTIVE, "hybrid", by="command")
    for number, text in enumerate(QUESTIONS[:TURNS], 1):
        world.replies.append(PLAN)  # DeepSeek plans the reply ...
        await world.chat(text, number)
        await run_to_idle(world.engine, world.clock)  # ... the model writes it
    latencies = [output.latency_ms for output in client.outputs]
    broken = [draft for _, draft in drafts if draft.meta.get("attempt_violations")]
    numbers: dict[str, Any] = {
        "turns": TURNS,
        "replies": len(drafts),
        "model_calls": len(client.outputs),
        "model_errors": client.errors,
        "latency_ms_median": round(statistics.median(latencies)) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "violation_rate": rate(len(broken), len(drafts)),
        "answered_by_the_model": sum(1 for _, d in drafts if d.backend == "hybrid"),
    }
    (tmp_path / "serving_report.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    return world, numbers


async def test_ten_turns_through_the_hybrid_backend_with_the_simulated_server(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Any,
) -> None:
    model = make_model_file(tmp_path / "models")
    spec = spec_for(model, free_port(), "--sim-reply", "嗯嗯\n好哒")
    async with serving(spec, tmp_path) as client:
        world, numbers = await ten_turns(
            services, clock, embedder, api, monkeypatch, client, tmp_path
        )
        try:
            for key, value in numbers.items():
                record_property(key, value)
            assert numbers["replies"] == TURNS and numbers["answered_by_the_model"] == TURNS
            assert numbers["model_calls"] == TURNS and numbers["model_errors"] == 0
            assert numbers["violation_rate"] == 0.0
            assert numbers["latency_ms_median"] is not None
            assert numbers["latency_ms_median"] < 5000  # the simulation answers at once
            # DeepSeek only planned: one call per turn, and the words are the model's
            assert world.route.call_count == TURNS
            assert world.bot_lines == ["bot: 嗯嗯", "bot: 好哒"] * TURNS
            out = [r for r in world.rows() if r.direction == "out" and not r.is_command]
            # the first bubble of a reply carries the numbers of the reply, among them its backend
            assert [r.backend for r in out if r.backend] == ["hybrid"] * TURNS
            assert len(out) == 2 * TURNS
            assert world.alerts() == []  # no fallback, nothing to tell anyone
            # every prompt was the rendered ChatML string, ending where the model starts writing
            prompts = [o.prompt_tokens for o in client.outputs]
            assert all(count > 0 for count in prompts)
        finally:
            await close(world)


@pytest.mark.skipif(
    not (os.environ.get("TWIN_LLAMA_SERVER") and os.environ.get("TWIN_LLAMA_GGUF")),
    reason="needs a real llama-server (TWIN_LLAMA_SERVER) and a real GGUF (TWIN_LLAMA_GGUF)",
)
async def test_ten_turns_through_the_hybrid_backend_with_a_real_gguf(
    services: Services,
    clock: ManualClock,
    embedder: HashingBackend,
    api: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Any,
) -> None:
    spec = ServerSpec(
        (os.environ["TWIN_LLAMA_SERVER"],),
        Path(os.environ["TWIN_LLAMA_GGUF"]),
        free_port(),
    )
    async with serving(spec, tmp_path) as client:
        world, numbers = await ten_turns(
            services, clock, embedder, api, monkeypatch, client, tmp_path
        )
        try:
            for key, value in numbers.items():
                record_property(key, value)
            assert numbers["replies"] >= TURNS  # every turn got a reply, from the model or DeepSeek
            assert numbers["model_calls"] >= 1 and numbers["latency_ms_median"] is not None
            assert numbers["violation_rate"] <= 0.5
        finally:
            await close(world)
