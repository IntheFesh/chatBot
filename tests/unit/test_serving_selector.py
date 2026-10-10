"""The way the engine treats a local model: loading, tokens that differ, a crash and the way back.

R-ENG-006 (the fallback and the return), R-LLM-008 (the budget's last level only hands over to a
healthy model that passed the gate), R-SRV-004 (the tokenizer check refuses a model),
R-SRV-002 (a model that is still loading is not a failure).  The first tests use a scripted client
and the manual clock; the last one is the whole chain with the simulated llama-server and the
serving component of the application.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.serving_world import FAST, ServingWorld, build_serving_world
from tests.support.style_models import ScriptedStyleClient, register_model
from tests.support.waiting import wait_until
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK
from twin.engine.backend_select import (
    LOADING,
    TOKENIZER,
    BackendChoice,
    BackendSelector,
)
from twin.engine.style_models import StyleModels, tokenizer_verdict
from twin.engine.style_runtime import ConfiguredStyleClient, StyleRuntime
from twin.llm.runtime import build_llm_runtime
from twin.llm.style_client import StyleHealth
from twin.services import Services
from twin.serving.component import StyleServingComponent
from twin.storage.models import Alert
from twin.training.registry import EVAL_TOKENIZE_CHECK, record_eval


class LoadingClient(ScriptedStyleClient):
    """A server that answers 503 while it reads the model, then is healthy."""

    loading: bool = True

    async def health(self) -> StyleHealth:
        self.health_calls += 1
        if self.loading:
            return StyleHealth(False, "Loading model", 5, loading=True)
        return StyleHealth(True, "ok", 3)


def selector_for(
    services: Services, client: ScriptedStyleClient, *, grace_s: float = 0.0
) -> BackendSelector:
    services.runtime.initialize()
    return BackendSelector(
        runtime=services.runtime,
        models=StyleModels(services.db),
        client=client,
        config=services.settings.backend,
        clock=services.clock,
        alerts=services.alerts,
        loading_grace_s=grace_s,
    )


def categories(services: Services) -> list[str]:
    with services.db.session() as session:
        return list(session.scalars(select(Alert.category).order_by(Alert.created_at, Alert.id)))


# ------------------------------------------------------------------------- loading


async def test_a_model_that_is_loading_is_answered_by_deepseek_without_a_fallback(
    services: Services, clock: ManualClock
) -> None:
    register_model(services, gate_passed=True)
    services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    client = LoadingClient()
    selector = selector_for(services, client, grace_s=180.0)
    choice = await selector.choose()
    assert choice == BackendChoice("deepseek", "style", LOADING)
    assert selector.fallback() is None and categories(services) == []
    clock.tick(100)
    assert (await selector.choose()).reason == LOADING  # still within the time it may take
    assert not selector.available()
    client.loading = False
    clock.tick(31)
    again = await selector.choose()
    assert again.name == "style" and not again.fell_back
    assert categories(services) == []  # nothing happened that anyone needs to hear about


async def test_a_model_that_never_finishes_loading_ends_in_the_fallback_with_the_alert(
    services: Services, clock: ManualClock
) -> None:
    register_model(services, gate_passed=True)
    services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    selector = selector_for(services, LoadingClient(), grace_s=180.0)
    assert (await selector.choose()).reason == LOADING
    clock.tick(181)
    choice = await selector.choose()
    assert choice.name == "deepseek" and choice.fell_back
    record = services.runtime.get(BACKEND_FALLBACK)
    assert record is not None and record["reason"] == "loading"
    assert categories(services) == ["style_model_down"]


async def test_without_a_grace_period_loading_is_a_failure_like_any_other(
    services: Services,
) -> None:
    register_model(services, gate_passed=True)
    services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    selector = selector_for(services, LoadingClient(), grace_s=0.0)  # a remote server
    choice = await selector.choose()
    assert choice.name == "deepseek" and choice.fell_back
    assert categories(services) == ["style_model_down"]


# ------------------------------------------------------------------- the tokenizer


def test_the_recorded_comparison_is_read_from_the_registry_document() -> None:
    assert tokenizer_verdict({}) is None
    assert tokenizer_verdict({EVAL_TOKENIZE_CHECK: {"ok": True}}) is True
    assert tokenizer_verdict({EVAL_TOKENIZE_CHECK: {"ok": False}}) is False
    assert tokenizer_verdict({EVAL_TOKENIZE_CHECK: {"ok": "yes"}}) is None
    assert tokenizer_verdict({EVAL_TOKENIZE_CHECK: "broken"}) is None


async def test_a_model_whose_tokens_differ_is_never_used_however_well_it_is_otherwise(
    services: Services, clock: ManualClock
) -> None:
    model_id = register_model(services, gate_passed=True)
    services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    selector = selector_for(services, ScriptedStyleClient())
    assert (await selector.choose()).name == "style"  # not compared yet: the server decides
    record_eval(services.db, model_id, {EVAL_TOKENIZE_CHECK: {"ok": False, "at": "x"}})
    clock.tick(31)
    probe = await selector.probe()
    assert not probe.usable and probe.code == TOKENIZER and "twin model verify" in probe.detail
    assert not selector.available()
    choice = await selector.choose()
    assert choice.name == "deepseek" and choice.fell_back
    assert categories(services) == ["style_model_down"]
    # a new comparison that passes lets it come back after the usual wait
    record_eval(services.db, model_id, {EVAL_TOKENIZE_CHECK: {"ok": True, "at": "y"}})
    services.settings.backend.recover_after_min = 0
    clock.tick(31)
    assert (await selector.choose()).name == "style"


def test_a_pinned_registry_view_answers_with_its_model_even_if_it_is_not_active(
    services: Services,
) -> None:
    register_model(services, run_id="live", gate_passed=True)
    candidate = register_model(services, run_id="cand", active=False, gate_passed=None)
    assert StyleModels(services.db).active() is not None
    assert StyleModels(services.db).active().id == "live-Q5_K_M"  # type: ignore[union-attr]
    pinned = StyleModels(services.db, pinned=candidate).active()
    assert pinned is not None and pinned.id == candidate and not pinned.passed_gate


# ----------------------------------------------------------- the configured client


async def test_the_hook_of_the_serving_component_speaks_before_the_server_listens(
    services: Services,
) -> None:
    client = ConfiguredStyleClient(services.settings.style_model, services.clock)
    services.settings.style_model.endpoint = "http://127.0.0.1:9"  # nobody there
    services.settings.style_model.model_id = "twin-style"
    client.health_hook = lambda: StyleHealth(False, "starting", 0, loading=True)
    health = await client.health()
    assert health.loading and not health.ok and health.detail == "starting"
    client.health_hook = lambda: None  # the hook has nothing to say: the server is asked
    answer = await client.health()
    assert not answer.ok and not answer.loading
    await client.aclose()


# --------------------------------------------------- the chain with a real server


async def choose_until(
    selector: BackendSelector, name: str, limit_s: float = 30.0
) -> BackendChoice:
    """Ask for the backend until it is ``name`` (real time: the server is a process)."""
    waited = 0.0
    while True:
        choice = await selector.choose()
        if choice.name == name:
            return choice
        if waited > limit_s:
            raise AssertionError(f"still {choice} after {limit_s}s")
        await asyncio.sleep(0.05)
        waited += 0.05


@contextlib.asynccontextmanager
async def chain(
    world: ServingWorld,
) -> AsyncIterator[tuple[StyleRuntime, StyleServingComponent]]:
    llm = build_llm_runtime(world.services)
    style = StyleRuntime.from_services(world.services, llm)
    component = StyleServingComponent(
        world.services,
        program=world.program(),
        tokenizers=world.tokenizers(),
        timings=FAST,
        style=style,
        persist_poll_s=0.05,
        remind_poll_s=3600.0,
        verify_poll_s=0.05,
    )
    await component.start()
    try:
        yield style, component
    finally:
        await component.stop()
        await style.aclose()
        await llm.client.aclose()


async def test_a_server_that_stops_answering_means_deepseek_then_the_way_back(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, backend="style")
    world.services.settings.backend.recover_after_min = 0.02  # 1.2 s instead of ten minutes
    async with chain(world) as (style, component):
        selector = style.selector
        await choose_until(selector, "style")  # the model loaded, health says ok
        assert selector.available()
        world.control.write_text("hang", encoding="utf-8")  # the process is alive, silent
        down = await choose_until(selector, "deepseek")
        assert down.fell_back and down.requested == "style"
        record = world.services.runtime.get(BACKEND_FALLBACK)
        assert record is not None and record["reason"] == "unhealthy"
        assert not selector.available()
        world.control.write_text("ok", encoding="utf-8")
        wait_for_server = choose_until(selector, "style", limit_s=60.0)
        back = await wait_for_server
        assert not back.fell_back
        assert world.services.runtime.get(BACKEND_FALLBACK) is None
        assert selector.available()
        await wait_until(lambda: component._server is not None, limit_s=5)
    with world.services.db.session() as session:
        rows = [(r.category, r.severity) for r in session.scalars(select(Alert))]
    # one fallback with its notice that the model is back, and none from the start-up
    assert rows == [("style_model_down", "warning"), ("style_model_down", "info")]
