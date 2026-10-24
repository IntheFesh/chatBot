"""The serving component of the application: server, tunnel, reminders, the record (R-SRV-002/003).

The local server is the simulated llama-server (a real child process); the instance is a local SSH
server with a scripted vLLM behind it.  The services run on the real clock because the processes do.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from tests.support.llama_sim import free_port
from tests.support.serving_world import FAST, ServingWorld, build_serving_world
from tests.support.ssh_server import PASSWORD, USER, LocalSSHServer
from tests.support.style_server import running_server, vllm_defaults
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from tests.support.waiting import wait_until
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, TUNNEL_WANTED
from twin.engine.backend_select import LOADING
from twin.engine.style_runtime import ConfiguredStyleClient, StyleRuntime
from twin.llm.runtime import build_llm_runtime
from twin.ops.health import HealthLevel
from twin.services import Services
from twin.serving.component import (
    Desired,
    StyleServingComponent,
    decide,
    format_span,
    reminder_due,
    reminder_text,
)
from twin.serving.evaluation import installed_servers
from twin.serving.server import ServerState
from twin.serving.state import ServingStateStore
from twin.serving.tunnel import TunnelState
from twin.storage.models import Alert
from twin.training.registry import get_model
from twin.training.remote.connection import remember_host


@pytest.fixture
def world(services: Services, tmp_path: Path) -> ServingWorld:
    return build_serving_world(services, tmp_path)


def make_component(world: ServingWorld, *options: str, **kwargs: object) -> StyleServingComponent:
    return StyleServingComponent(
        world.services,
        program=world.program(*options),
        tokenizers=world.tokenizers(),
        timings=FAST,
        persist_poll_s=0.05,
        remind_poll_s=0.1,
        verify_poll_s=0.05,
        **kwargs,  # type: ignore[arg-type]
    )


@contextlib.asynccontextmanager
async def running(component: StyleServingComponent) -> AsyncIterator[StyleServingComponent]:
    await component.start()
    try:
        yield component
    finally:
        await component.stop()


def server_state(component: StyleServingComponent) -> ServerState | None:
    return component._server.state if component._server is not None else None


async def until_ready(component: StyleServingComponent, limit: float = 30) -> None:
    await wait_until(lambda: server_state(component) is ServerState.READY, limit_s=limit)


def record(world: ServingWorld) -> dict[str, object]:
    return ServingStateStore(world.services.db, world.services.clock).read()


def alert_categories(world: ServingWorld) -> list[str]:
    with world.services.db.session() as session:
        return [row.category for row in session.scalars(select(Alert))]


# ------------------------------------------------------------------ what is wanted


def test_a_server_is_wanted_while_the_style_or_hybrid_backend_is_asked_for(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, gate_passed=None, backend="style")
    wanted = decide(world.services)
    assert wanted.server is not None and wanted.server.id == world.model_id and not wanted.tunnel
    world.services.runtime.set(BACKEND_ACTIVE, "hybrid", by="test")
    assert decide(world.services).server is not None
    world.services.runtime.set(BACKEND_ACTIVE, "deepseek", by="test")
    assert decide(world.services) == Desired()  # not passed the gate: nothing keeps it loaded


def test_a_fallback_keeps_the_server_because_the_way_back_needs_it(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, gate_passed=None, backend="deepseek")
    world.services.runtime.set(BACKEND_FALLBACK, {"requested": "style", "reason": "error"}, by="t")
    assert decide(world.services).server is not None


def test_a_model_that_passed_the_gate_stays_loaded_for_budget_level_four(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, gate_passed=True, backend="deepseek")
    assert decide(world.services).server is not None
    world.services.settings.style_model.serve.warm_standby = False
    assert decide(world.services) == Desired()


def test_nothing_is_wanted_without_an_active_model_or_with_a_refused_one(
    services: Services, tmp_path: Path
) -> None:
    none = build_serving_world(services, tmp_path, active=False)
    assert decide(none.services) == Desired()
    refused = build_serving_world(services, tmp_path, run_id="r-2", tokenize_ok=False)
    assert decide(refused.services) == Desired()  # the tokens differ: never started again


def test_a_model_of_another_template_is_never_started(services: Services, tmp_path: Path) -> None:
    world = build_serving_world(services, tmp_path, template_version="qwen3_think@1")
    assert decide(world.services) == Desired()


def test_the_tunnel_is_wanted_in_vllm_mode_only_while_it_is_switched_on(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, run_id="r-lora", quant="lora", kind="adapter")
    world.services.settings.style_model.mode = "vllm_completion"
    assert decide(world.services) == Desired()
    world.services.runtime.set(TUNNEL_WANTED, True, by="test")
    assert decide(world.services) == Desired(tunnel=True)


# --------------------------------------------------------------------- the server


async def test_the_server_starts_is_compared_warmed_up_and_recorded(world: ServingWorld) -> None:
    async with running(make_component(world)) as component:
        await until_ready(component)
        assert installed_servers() is not None  # the evaluation jobs can use the pool
        await component.flush()
        data = record(world)
        server = data["server"]
        assert isinstance(server, dict) and server["state"] == "ready"
        assert server["model"] == world.model_id and server["restarts"] == 0
        warmup = data["warmup"]
        assert isinstance(warmup, dict) and warmup["tokens_per_s"] == 42.5
        assert warmup["measured_by"] == "server" and warmup["first_token_ms"] > 0
        assert component.health().status.value == "ok"
        check = await component.health_check()
        assert check.level is HealthLevel.OK and "ready" in check.detail
    model = get_model(world.services.db, world.model_id)
    assert model.eval["tokenize_check"]["ok"] is True
    assert installed_servers() is None
    after = record(world)
    assert after.get("server") is None and after.get("warmup") is None  # stopped: nothing served


async def test_while_the_model_loads_the_engine_is_not_told_it_is_down(
    world: ServingWorld,
) -> None:
    services = world.services
    llm = build_llm_runtime(services)
    style = StyleRuntime.from_services(services, llm)
    assert isinstance(style.client, ConfiguredStyleClient)
    component = make_component(world, "--sim-load-s", "2.0", style=style)
    async with running(component):
        await wait_until(lambda: server_state(component) is ServerState.STARTING)
        health = await style.client.health()
        assert not health.ok and health.loading  # known from the process, before it listens
        choice = await style.selector.choose()
        assert (
            choice.name == "deepseek" and choice.reason == LOADING and choice.requested == "style"
        )
        assert style.selector.fallback() is None  # no fallback, no alert, no ten-minute wait
        await until_ready(component)
        assert (await style.client.health()).ok
        again = await style.selector.choose()
        assert again.name == "style" and not again.fell_back
    await style.aclose()
    await llm.client.aclose()
    assert alert_categories(world) == []


async def test_a_change_of_the_settings_starts_and_stops_the_server(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path, gate_passed=None, backend="deepseek")
    async with running(make_component(world)) as component:
        assert component._server is None  # not asked for, not passed the gate: nothing loaded
        world.services.runtime.set(BACKEND_ACTIVE, "style", by="test")
        await component.on_state_change(1, 2)
        await until_ready(component)
        world.services.runtime.set(BACKEND_ACTIVE, "deepseek", by="test")
        await component.on_state_change(2, 3)
        assert component._server is None


async def test_a_crashed_server_is_restarted_and_the_record_says_so(world: ServingWorld) -> None:
    async with running(make_component(world)) as component:
        await until_ready(component)
        world.control.write_text("exit", encoding="utf-8")
        await wait_until(
            lambda: component._server is not None and component._server.snapshot().restarts >= 1
        )
        await until_ready(component)
        await component.flush()
        server = record(world)["server"]
        assert isinstance(server, dict) and server["restarts"] == 1 and server["state"] == "ready"


async def test_a_model_whose_tokens_differ_is_refused_stopped_and_kept_stopped(
    world: ServingWorld,
) -> None:
    async with running(make_component(world, "--sim-add-bos", "7")) as component:
        await wait_until(lambda: server_state(component) is ServerState.BLOCKED)
        blocked = component._server.blocked if component._server else None
        assert blocked is not None and blocked.reason == "tokenizer"
        check = await component.health_check()
        assert check.level is HealthLevel.FAIL and check.category == "style_tokenize_mismatch"
        assert alert_categories(world) == ["style_tokenize_mismatch"]
        model = get_model(world.services.db, world.model_id)
        assert model.eval["tokenize_check"]["ok"] is False
        # the next look at the settings does not start it again
        await component.reconcile()
        assert component._server is None
        assert (await component.health_check()).level is HealthLevel.OK


async def test_a_missing_installation_is_reported_and_tried_again(
    services: Services, tmp_path: Path
) -> None:
    world = build_serving_world(services, tmp_path)
    component = StyleServingComponent(
        world.services, tokenizers=world.tokenizers(), timings=FAST, remind_poll_s=0.05
    )
    async with running(component):
        assert component._server is None
        check = await component.health_check()
        assert check.level is HealthLevel.FAIL and check.category == "style_model_down"
        assert "llama.cpp is not installed" in check.detail
        assert component.health().status.value == "degraded"
        await component.flush()
        server = record(world)["server"]
        assert isinstance(server, dict) and server["state"] == "unavailable"


async def test_a_missing_model_file_is_reported(world: ServingWorld) -> None:
    world.model_file.unlink()
    async with running(make_component(world)) as component:
        check = await component.health_check()
        assert check.level is HealthLevel.FAIL and "does not exist" in check.detail


async def test_a_model_file_of_another_size_is_refused(world: ServingWorld) -> None:
    world.model_file.write_bytes(b"short")
    async with running(make_component(world)) as component:
        check = await component.health_check()
        assert check.level is HealthLevel.FAIL and "another size" in check.detail


# ------------------------------------------------------------------ the instance


@contextlib.asynccontextmanager
async def instance(
    services: Services, tmp_path: Path
) -> AsyncIterator[tuple[ServingWorld, LocalSSHServer]]:
    """A rented instance: an SSH server and, behind it, vLLM that tokenizes like the trainer."""
    world = build_serving_world(
        services, tmp_path, run_id="r-lora", quant="lora", kind="adapter", backend="style"
    )
    tokenizer = tiny_qwen_tokenizer()
    with running_server() as vllm:
        vllm_defaults(vllm, model="twin-style")
        vllm.set(
            "POST",
            "/tokenize",
            handler=lambda payload: {
                "count": 0,
                "max_model_len": 4096,
                "tokens": tokenizer.encode(payload["prompt"]),
            },
        )
        async with LocalSSHServer(tmp_path / "ssh") as ssh:
            ssh.state.allow_forwarding = True
            real = world.services
            config = real.settings.style_model
            config.mode = "vllm_completion"
            config.model_id = "twin-style"
            config.tunnel.local_port = free_port()
            config.tunnel.remote_port = vllm.port
            config.tunnel.backoff_start_s = 0.05
            config.tunnel.backoff_max_s = 0.2
            real.settings.autodl.host = "127.0.0.1"
            real.settings.autodl.port = ssh.port
            real.settings.autodl.user = USER
            real.secrets.set("autodl_password", PASSWORD)
            from twin.serving.evaluation import tunnel_known_hosts

            remember_host(tunnel_known_hosts(real), "127.0.0.1", ssh.port, ssh.host_key)
            real.runtime.set(TUNNEL_WANTED, True, by="test")
            yield world, ssh


async def test_the_tunnel_is_kept_up_verified_warmed_up_and_recorded(
    services: Services, tmp_path: Path
) -> None:
    async with instance(services, tmp_path) as (world, _ssh):
        said: list[str] = []

        async def say(text: str) -> None:
            said.append(text)

        component = StyleServingComponent(
            world.services,
            tokenizers=world.tokenizers(),
            say=say,
            persist_poll_s=0.05,
            remind_poll_s=3600.0,  # the test asks for the reminder itself
            verify_poll_s=0.05,
        )
        async with running(component):
            await wait_until(
                lambda: component._tunnel is not None and component._tunnel.state is TunnelState.UP,
                limit_s=30,
            )
            await wait_until(lambda: component._verified is not None, limit_s=30)
            await component.flush()
            data = record(world)
            tunnel = data["tunnel"]
            assert isinstance(tunnel, dict) and tunnel["state"] == "up"
            assert tunnel["reconnects"] == 0 and tunnel["local_port"] > 0
            model = get_model(world.services.db, world.model_id)
            assert model.eval["tokenize_check"]["ok"] is True  # compared over the tunnel
            warmup = data["warmup"]
            assert isinstance(warmup, dict) and warmup["measured_by"] == "client"
            check = await component.health_check()
            assert check.level is HealthLevel.OK and "tunnel up" in check.detail
            # the daily reminder: once, with the hours of the instance
            assert said == []
            world.services.settings.commands.morning_hour = 0
            assert await component.remind_once() is True
            assert "按小时计费" in said[0] and "实例" in said[0]
            assert await component.remind_once() is False  # not twice in a day
            # `twin model tunnel stop` switches it off; the application follows
            world.services.runtime.set(TUNNEL_WANTED, False, by="test")
            await component.on_state_change(1, 2)
            assert component._tunnel is None


async def test_no_reminder_when_it_is_switched_off_or_there_is_no_tunnel(
    services: Services, tmp_path: Path
) -> None:
    async with instance(services, tmp_path) as (world, _ssh):
        said: list[str] = []

        async def say(text: str) -> None:
            said.append(text)

        component = StyleServingComponent(world.services, tokenizers=world.tokenizers(), say=say)
        assert await component.remind_once() is False  # no tunnel yet
        world.services.settings.style_model.tunnel.remind_remote = False
        async with running(component):
            await wait_until(
                lambda: component._tunnel is not None and component._tunnel.state is TunnelState.UP,
                limit_s=30,
            )
            assert await component.remind_once() is False
        assert said == []


async def test_a_tunnel_without_login_data_is_reported_not_raised(
    services: Services, tmp_path: Path
) -> None:
    async with instance(services, tmp_path) as (world, _ssh):
        world.services.settings.autodl.host = None
        component = StyleServingComponent(world.services, tokenizers=world.tokenizers())
        async with running(component):
            check = await component.health_check()
            assert check.level is HealthLevel.FAIL and "autodl.host" in check.detail


# ---------------------------------------------------------------- pure helpers


def test_the_reminder_comes_once_a_day_and_not_before_the_morning_hour() -> None:
    day = datetime(2026, 10, 10, 7, 59, tzinfo=UTC)
    assert not reminder_due(day, 8, None)
    assert reminder_due(day + timedelta(minutes=1), 8, None)
    assert not reminder_due(day + timedelta(hours=3), 8, "2026-10-10")
    assert reminder_due(day + timedelta(days=1, hours=1), 8, "2026-10-10")


def test_the_reminder_says_what_runs_by_the_hour_and_for_how_long() -> None:
    text = reminder_text(3 * 3600 + 20 * 60, None)
    assert "按小时计费" in text and "3 小时 20 分" in text and "/后端 deepseek" in text
    assert "twin model tunnel stop" in text and "控制台关机" in text
    assert "隧道已连上 45 分钟" in reminder_text(None, 45 * 60)
    assert "实例在运行" in reminder_text(None, None)
    assert (
        format_span(timedelta(hours=2)) == "2 小时"
        and format_span(timedelta(minutes=5)) == "5 分钟"
    )


async def test_the_pool_is_installed_while_the_component_runs(world: ServingWorld) -> None:
    assert installed_servers() is None
    async with running(make_component(world)) as component:
        assert installed_servers() is component._pool
    await asyncio.sleep(0)
    assert installed_servers() is None
