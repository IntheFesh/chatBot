"""The style model goes away in the middle of a conversation and comes back (R-SRV-004, R-LLM-008).

The model is served by a made-up ``llama-server`` (a real HTTP server on the loopback, answering
the way ``/health`` and ``/completion`` do), and the application of ``twin run`` talks to it
through its own client and its own backend selector:

* ``/后端 hybrid``: DeepSeek only plans, the style model writes - ``bot_turns`` says so;
* the server dies: the next reply is DeepSeek's, nothing of the trouble is shown to him, the
  selector records why and since when (``backend.fallback``), and the alert ``style_model_down``
  goes through the alert service;
* while it is down the style model is not asked again, only looked at;
* the server comes back: ten healthy minutes later - not before - the selector switches back by
  itself, the record is cleared, a notice goes out, and the next reply is the style model's.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import register_style_model
from tests.support.proactive_world import opening_curve, proactive_model
from tests.support.style_server import StyleServer, llama_defaults
from twin.commands import texts
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK
from twin.services import Services

pytestmark = pytest.mark.integration
ASSISTANT_OPENER = "<|im_start|>assistant\n"
RECOVER_AFTER = timedelta(minutes=10)
DEFAULT_ENDPOINT = "http://127.0.0.1:8081"


@pytest.fixture
def server(api: respx.MockRouter) -> StyleServer:
    """The made-up llama-server at the address the settings give by default."""
    found = StyleServer()
    llama_defaults(found, "好呀")
    found.mount(api, DEFAULT_ENDPOINT)
    return found


def completions(server: StyleServer) -> int:
    return len([r for r in server.requests if r.path == "/completion"])


def set_down(server: StyleServer) -> None:
    server.set("GET", "/health", status=500, body={"error": "gone"})
    server.set("POST", "/completion", status=503, body={"error": "gone"})


async def test_the_style_model_dies_and_comes_back(
    make_world: WorldFactory, server: StyleServer, services: Services
) -> None:
    assert server.url == services.settings.style_model.endpoint == DEFAULT_ENDPOINT

    world = await make_world(
        datetime(2026, 10, 9, 15, 0, tzinfo=UTC),  # 10:00 in Chicago, she is up
        model=proactive_model(opening_curve(base=0.02)),
    )
    register_style_model(services)
    world.watch.add("engine/backend_select.py")  # the monitor that looks at the model every 30 s

    # ---- hybrid: DeepSeek plans, the style model writes -------------------------------------
    await world.say("/后端 hybrid")
    chosen = texts.PREFIX + texts.BACKEND_SET.format(name="hybrid")
    assert world.system_said[-1].text.startswith(chosen)
    assert services.runtime.get(BACKEND_ACTIVE) == "hybrid"
    replies = world.deepseek.calls["reply"]
    await world.say("在吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "好呀"
    assert world.deepseek.calls["reply"] == replies  # DeepSeek did not write it ...
    assert world.deepseek.calls["reply_plan"] == 1  # ... it planned it
    sent = server.last("/completion").json
    assert sent["prompt"].endswith(ASSISTANT_OPENER) and "在吗" in sent["prompt"]
    assert [r.backend for r in world.out_rows()] == ["hybrid"]

    # ---- the server dies ----------------------------------------------------------------------
    set_down(server)
    asked = completions(server)
    await world.run_for(minutes=1)
    await world.say("还在吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "嗯嗯"  # DeepSeek's answer, nothing about the trouble
    assert [r.backend for r in world.out_rows()] == ["hybrid", "deepseek"]
    fallback = services.runtime.get(BACKEND_FALLBACK)
    assert fallback is not None and fallback["requested"] == "hybrid"
    assert fallback["reason"] == "unhealthy" and not fallback["by_budget"]
    assert services.runtime.get(BACKEND_ACTIVE) == "hybrid"  # his choice stays as it was
    assert world.alerts() == [("style_model_down", "warning")]
    await world.say("/状态")
    assert "风格模型不可用，眼下由 deepseek 回复" in world.system_said[-1].text

    # ---- while it is down: looked at, not asked ------------------------------------------------
    went_down = world.now
    asked = completions(server)
    looks = len([r for r in server.requests if r.path == "/health"])
    await world.run_for(minutes=5)
    await world.say("好了吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "嗯嗯"
    assert completions(server) == asked  # the style model was left alone
    assert len([r for r in server.requests if r.path == "/health"]) > looks + 2  # but watched

    # ---- the server is back; ten healthy minutes later so is the style model --------------------
    llama_defaults(server, "回来了")
    back = world.now
    await world.run_until(back + RECOVER_AFTER - timedelta(minutes=2))
    assert services.runtime.get(BACKEND_FALLBACK) is not None  # not yet: it must stay healthy
    await world.run_until(back + RECOVER_AFTER + timedelta(minutes=2))
    assert services.runtime.get(BACKEND_FALLBACK) is None
    assert world.alerts()[-1] == ("style_model_down", "info")  # the notice that it is back
    await world.say("回来了吗")
    await world.run_until_idle()
    assert world.persona_said[-1].text == "回来了"
    assert [r.backend for r in world.out_rows()] == ["hybrid", "deepseek", "deepseek", "hybrid"]
    assert went_down < back

    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert world.deepseek.unexpected == []
