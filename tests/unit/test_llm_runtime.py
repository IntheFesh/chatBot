"""Wiring of the LLM layer: services in, client and policies out (R-ARCH-003, R-LLM-007)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import openai
import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, ok, request_json
from twin.assembly import assemble
from twin.channel.local import StreamInput, StreamOutput
from twin.config.runtime import BOT_TIMEZONE
from twin.config.secrets import SecretStoreError
from twin.llm.capabilities import LlmCapabilities, save_capabilities
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import CircuitOpenError
from twin.llm.health import llm_health_of
from twin.llm.images import ImageInput
from twin.llm.pricing import CalendarOffPeakPolicy
from twin.llm.reliability import BreakerState
from twin.llm.runtime import (
    EXTRAS_KEY,
    LlmRuntime,
    activate_offpeak_policy,
    build_llm_runtime,
    close_llm_runtime,
    closing_llm_runtime,
)
from twin.llm.synth_images import draw_png
from twin.llm.tokens import Calibration
from twin.llm.types import ChatMessage, Purpose
from twin.ops.components import build_application
from twin.ops.jobs import DeferredOffPeakPolicy, get_offpeak_policy
from twin.ops.process_model import JobSpec, enqueue_heavy
from twin.ops.state_watch import StateWatcher
from twin.services import Services
from twin.storage.models import Alert


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def test_the_production_policy_replaces_the_default_when_the_application_is_built(
    services: Services,
) -> None:
    assert isinstance(get_offpeak_policy(), DeferredOffPeakPolicy)
    build_application(services)
    policy = get_offpeak_policy()
    assert isinstance(policy, CalendarOffPeakPolicy)
    # Friday 12:00 UTC is peak time, Saturday is not
    assert not policy.allows(datetime(2026, 10, 9, 2, tzinfo=UTC))
    assert policy.allows(datetime(2026, 10, 17, 2, tzinfo=UTC))


def test_activating_the_policy_needs_no_api_key_and_returns_the_prices(services: Services) -> None:
    pricing = activate_offpeak_policy(services)
    assert pricing.has_model("deepseek-flash")
    assert isinstance(get_offpeak_policy(), CalendarOffPeakPolicy)


def test_the_offpeak_discount_can_be_switched_off_in_the_configuration(
    services: Services,
) -> None:
    flat = Services(**{**services.__dict__, "settings": services.settings.model_copy(deep=True)})
    flat.settings.pricing.offpeak_multiplier = 1.0
    activate_offpeak_policy(flat)
    assert get_offpeak_policy().allows(datetime(2026, 10, 9, 2, tzinfo=UTC))  # nothing to wait for


def test_foreground_heavy_commands_use_the_production_policy(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(datetime(2026, 10, 9, 2, tzinfo=UTC))  # peak hours
    enqueue_heavy(services, [JobSpec("unhandled-type", {}, offpeak_only=True)], foreground=True)
    assert isinstance(get_offpeak_policy(), CalendarOffPeakPolicy)


def test_building_the_runtime_does_not_read_the_api_key(services: Services) -> None:
    runtime = build_llm_runtime(services)  # no key stored: still fine
    assert runtime.client.model_for(Purpose.REPLY) == "deepseek-flash"
    assert runtime.budget.current_level() == 0
    assert runtime.pricing.has_model("deepseek-v4-pro")


async def test_the_client_reads_the_key_from_the_credential_store(
    services: Services, api: respx.MockRouter
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    route = api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    await runtime.client.chat([{"role": "user", "content": "你好"}], purpose="reply")
    assert route.calls.last.request.headers["authorization"] == f"Bearer {TEST_KEY}"
    await runtime.client.aclose()


def test_the_time_service_follows_the_bot_time_zone_setting(
    services: Services, clock: ManualClock
) -> None:
    clock.set_time(datetime(2026, 10, 9, 3, 30, tzinfo=UTC))
    runtime = build_llm_runtime(services)
    assert runtime.time_service.local_date().isoformat() == "2026-10-08"  # Chicago
    services.runtime.set(BOT_TIMEZONE, "Asia/Shanghai")
    assert runtime.time_service.local_date().isoformat() == "2026-10-09"


def test_a_year_outside_the_holiday_library_raises_one_alert(
    services: Services, clock: ManualClock
) -> None:
    runtime = build_llm_runtime(services)
    far = datetime(2031, 3, 4, 2, tzinfo=UTC)
    assert runtime.pricing.is_peak(far) and runtime.pricing.is_peak(far)
    with services.db.session() as session:
        alerts = list(session.execute(select(Alert)).scalars())
        assert [a.category for a in alerts] == ["calendar_out_of_range"]
        assert alerts[0].detail == {"year": 2031}


def test_the_token_calibration_survives_a_restart(services: Services) -> None:
    first = build_llm_runtime(services)
    first.estimator.observe([{"role": "user", "content": "你好世界" * 20}], 200)
    with services.db.transaction(bump_state=False) as session:
        first.estimator.save(session, services.clock)
    services.extras.clear()  # the process ends; the next one builds its runtime afresh
    second = build_llm_runtime(services)
    assert second is not first
    assert second.estimator.calibration == first.estimator.calibration != Calibration()


async def test_a_probe_result_changes_what_the_running_client_sends(
    services: Services, api: respx.MockRouter, clock: ManualClock
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    route = api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    image = ImageInput.from_bytes(draw_png(), "low")
    messages = [{"role": "user", "content": "看"}]
    await runtime.client.chat(messages, purpose="caption", images=[image])  # type: ignore[arg-type]
    first = request_json(route.calls.last.request)["messages"][0]["content"][1]["image_url"]
    assert first["detail"] == "low"

    measured = LlmCapabilities(
        detail_supported=False, image_tokens=((4096, 33),), measured_at=clock.now_utc().isoformat()
    )
    with services.db.transaction() as session:
        save_capabilities(session, measured, services.clock)
    runtime.capabilities.invalidate()
    await runtime.client.chat(messages, purpose="caption", images=[image])  # type: ignore[arg-type]
    second = request_json(route.calls.last.request)["messages"][0]["content"][1]["image_url"]
    assert "detail" not in second
    assert runtime.estimator.estimate_image(64, 64) == 33  # the estimator learned the sizes too
    await runtime.client.aclose()


# ------------------------------------------------------- one runtime per process (R17-10)


def test_there_is_one_runtime_per_services_container_however_often_it_is_asked_for(
    services: Services,
) -> None:
    """R17-10 (D-701): the engine, every job and the health check share one client."""
    first = build_llm_runtime(services)
    assert services.extras[EXTRAS_KEY] is first
    assert build_llm_runtime(services) is first
    assert first.client.breaker is build_llm_runtime(services).client.breaker
    other = Services(**{**services.__dict__, "extras": {}})
    assert build_llm_runtime(other) is not first  # another container: its own runtime


def test_threads_asking_at_the_same_time_get_the_same_runtime(services: Services) -> None:
    """The health check builds it from a worker thread while the application starts."""
    start = threading.Barrier(8)
    found: list[LlmRuntime] = []

    def ask() -> None:
        start.wait(10)
        found.append(build_llm_runtime(services))

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(found) == 8 and all(runtime is found[0] for runtime in found)


def test_a_style_status_given_later_is_applied_to_the_shared_budget(services: Services) -> None:
    class Ready:
        def available(self) -> bool:
            return True

    first = build_llm_runtime(services)
    second = build_llm_runtime(services, style=Ready())
    assert second is first
    assert first.budget._style.available()


async def test_calls_of_three_callers_never_exceed_the_configured_concurrency(
    services: Services, api: respx.MockRouter
) -> None:
    """R17-10: every caller used to bring a semaphore of its own, so six calls ran at once."""
    services.secrets.set("deepseek_api_key", TEST_KEY)
    services.settings.deepseek.max_concurrency = 2
    state = {"active": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        for _ in range(8):
            await asyncio.sleep(0)
        state["active"] -= 1
        return ok()

    api.post(API).mock(side_effect=handler)
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]

    async def caller() -> None:  # the engine, a job and the sticker description each ask
        runtime = build_llm_runtime(services)
        await asyncio.gather(*[runtime.client.chat(messages, purpose="reply") for _ in range(2)])

    await asyncio.gather(caller(), caller(), caller())
    assert state["peak"] == 2
    await close_llm_runtime(services)


async def test_a_breaker_opened_in_a_job_is_seen_by_the_engine_and_the_health_check(
    services: Services, api: respx.MockRouter
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    route = api.post(API).mock(return_value=ok())
    job = build_llm_runtime(services)
    job.client.breaker.failure_threshold = 2
    for _ in range(2):  # what two failed attempts of a night job do to the breaker
        job.client.breaker.before_call()
        job.client.breaker.record_failure()
    engine = build_llm_runtime(services)
    assert engine.client.breaker.state is BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        await engine.client.chat([{"role": "user", "content": "x"}], purpose="reply")
    assert route.call_count == 0  # the engine does not even try
    assert llm_health_of(services).snapshot(900).circuit_open
    await close_llm_runtime(services)


def test_a_second_event_loop_on_the_same_services_can_use_the_runtime(
    services: Services, api: respx.MockRouter
) -> None:
    """A command that calls ``asyncio.run`` twice (``twin eval blind`` does) keeps working."""
    services.secrets.set("deepseek_api_key", TEST_KEY)
    api.post(API).mock(return_value=ok(content="一"))
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]

    async def ask() -> str:
        runtime = build_llm_runtime(services)
        return (await runtime.client.chat(messages, purpose="reply")).content

    assert asyncio.run(ask()) == "一"
    assert asyncio.run(ask()) == "一"  # the shared runtime moved to the new loop
    asyncio.run(close_llm_runtime(services))
    assert EXTRAS_KEY not in services.extras
    assert asyncio.run(ask()) == "一"  # and after the close a new one is made
    asyncio.run(close_llm_runtime(services))


async def test_closing_the_runtime_twice_or_before_it_exists_is_harmless(
    services: Services,
) -> None:
    await close_llm_runtime(services)  # nothing was built
    runtime = build_llm_runtime(services)
    await runtime.aclose()
    await close_llm_runtime(services)
    await close_llm_runtime(services)
    assert EXTRAS_KEY not in services.extras


# ------------------------------------------------- a replaced or deleted API key (R17-30)

OTHER_KEY = "synthetic-test-key-0002"


def bump_state(services: Services) -> None:
    """What ``twin secrets set`` does after it stored the secret (a LIGHT command, R-ARCH-006)."""
    with services.db.transaction(bump_state=True):
        pass


def bearer(route: respx.Route) -> str:
    return str(route.calls.last.request.headers["authorization"])


@pytest.fixture
def watched(services: Services) -> StateWatcher:
    return StateWatcher(services.db, services.clock)


async def test_a_replaced_key_is_used_from_the_next_request_on(
    services: Services, api: respx.MockRouter, watched: StateWatcher
) -> None:
    """R17-30 (D-702): no failed request first - the 401 only came when the old key was revoked."""
    services.secrets.set("deepseek_api_key", TEST_KEY)
    route = api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    watched.subscribe(runtime.on_state_change)
    await watched.poll_once()  # the baseline
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]
    await runtime.client.chat(messages, purpose="reply")
    assert bearer(route) == f"Bearer {TEST_KEY}"
    services.secrets.set("deepseek_api_key", OTHER_KEY)
    bump_state(services)
    assert await watched.poll_once()
    await runtime.client.chat(messages, purpose="reply")
    assert bearer(route) == f"Bearer {OTHER_KEY}"
    assert route.call_count == 2  # the old key was never tried again, nothing failed in between
    await close_llm_runtime(services)


async def test_a_state_change_that_leaves_the_key_alone_keeps_the_connection(
    services: Services,
    api: respx.MockRouter,
    watched: StateWatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    api.post(API).mock(return_value=ok())
    made: list[str] = []
    original = DeepSeekClient._make_sdk

    def counting(self: DeepSeekClient, secret: str) -> object:
        made.append(secret)
        return original(self, secret)

    monkeypatch.setattr(DeepSeekClient, "_make_sdk", counting)
    runtime = build_llm_runtime(services)
    watched.subscribe(runtime.on_state_change)
    await watched.poll_once()
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]
    await runtime.client.chat(messages, purpose="reply")
    for _ in range(3):  # other settings change all the time
        bump_state(services)
        assert await watched.poll_once()
    await runtime.client.chat(messages, purpose="reply")
    assert made == [TEST_KEY]
    await close_llm_runtime(services)


async def test_a_request_in_flight_finishes_on_the_old_key_and_the_old_connection_closes_after(
    services: Services, api: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    gate = asyncio.Event()
    started = asyncio.Event()
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if len(seen) == 1:
            started.set()
            await gate.wait()
            return ok(content="old")
        return ok(content="new")

    api.post(API).mock(side_effect=handler)
    closed: list[int] = []
    real_close = openai.AsyncOpenAI.close

    async def counting_close(self: openai.AsyncOpenAI) -> None:
        closed.append(1)
        await real_close(self)

    monkeypatch.setattr(openai.AsyncOpenAI, "close", counting_close)
    runtime = build_llm_runtime(services)
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]
    first = asyncio.create_task(runtime.client.chat(messages, purpose="reply"))
    await started.wait()
    services.secrets.set("deepseek_api_key", OTHER_KEY)
    assert await runtime.client.refresh_key() is True
    second = await runtime.client.chat(messages, purpose="reply")
    assert second.content == "new" and seen[1] == f"Bearer {OTHER_KEY}"
    assert closed == []  # the first request is still on the old connection
    gate.set()
    assert (await first).content == "old" and seen[0] == f"Bearer {TEST_KEY}"
    assert closed == [1]  # the last lease was given back: the retired connection is closed
    await close_llm_runtime(services)
    assert closed == [1, 1]  # and the current one with the runtime


async def test_a_deleted_key_gives_the_clear_error_at_the_next_request(
    services: Services, api: respx.MockRouter, watched: StateWatcher
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    route = api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    watched.subscribe(runtime.on_state_change)
    await watched.poll_once()
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]
    await runtime.client.chat(messages, purpose="reply")
    services.secrets.delete("deepseek_api_key")
    bump_state(services)
    await watched.poll_once()
    with pytest.raises(SecretStoreError, match="twin secrets set deepseek_api_key"):
        await runtime.client.chat(messages, purpose="reply")
    assert route.call_count == 1  # the deleted key was not used again
    services.secrets.set("deepseek_api_key", OTHER_KEY)  # and setting it again works
    await runtime.client.chat(messages, purpose="reply")
    assert bearer(route) == f"Bearer {OTHER_KEY}"
    await close_llm_runtime(services)


async def test_a_client_that_has_not_been_used_does_not_read_the_store(
    services: Services,
) -> None:
    runtime = build_llm_runtime(services)  # no key stored, nothing sent yet
    assert await runtime.client.refresh_key() is False
    runtime.client.reset_key()  # nothing to drop either


async def test_two_looks_at_a_replaced_key_reset_the_connection_once(
    services: Services, api: respx.MockRouter
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    await runtime.client.chat([{"role": "user", "content": "你好"}], purpose="reply")
    services.secrets.set("deepseek_api_key", OTHER_KEY)
    first, second = await asyncio.gather(runtime.client.refresh_key(), runtime.client.refresh_key())
    assert sorted([first, second]) == [False, True]
    await close_llm_runtime(services)


def test_dropping_the_key_after_the_loop_is_over_does_not_need_the_loop(
    services: Services, api: respx.MockRouter
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)

    async def use() -> None:
        await runtime.client.chat([{"role": "user", "content": "你好"}], purpose="reply")

    asyncio.run(use())  # the loop of the connection is closed now
    runtime.client.reset_key()  # nothing to close it on: the pool died with the loop
    asyncio.run(use())  # and the next request builds a new connection on the new loop
    asyncio.run(close_llm_runtime(services))


async def test_neither_the_key_nor_its_hash_is_logged(
    services: Services, api: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    api.post(API).mock(return_value=ok())
    runtime = build_llm_runtime(services)
    messages: list[ChatMessage] = [{"role": "user", "content": "你好"}]
    with caplog.at_level(logging.DEBUG):
        await runtime.client.chat(messages, purpose="reply")
        services.secrets.set("deepseek_api_key", OTHER_KEY)
        await runtime.client.refresh_key()
        await runtime.client.chat(messages, purpose="reply")
    everything = caplog.text + repr([record.__dict__ for record in caplog.records])
    for key in (TEST_KEY, OTHER_KEY):
        assert key not in everything
        assert hashlib.sha256(key.encode()).hexdigest() not in everything
        assert hashlib.sha256(key.encode()).hexdigest()[:16] not in everything
    assert "deepseek_key_changed" in caplog.text
    await close_llm_runtime(services)


def test_the_assembled_application_listens_for_a_new_key(services: Services) -> None:
    """The listener is wired where the application is built (``twin run``)."""
    services.secrets.set("deepseek_api_key", TEST_KEY)
    assembly = assemble(
        services,
        console_input=StreamInput(io.StringIO("")),
        console_output=StreamOutput(io.StringIO()),
    )
    runtime = build_llm_runtime(services)
    assert runtime is assembly.llm
    assert any(
        getattr(listener, "__func__", None) is LlmRuntime.on_state_change
        and listener.__self__ is runtime  # type: ignore[union-attr]
        for listener in assembly.watcher._listeners
    )


async def test_a_one_shot_command_closes_the_runtime_when_it_ends_or_fails(
    services: Services,
) -> None:
    async def work() -> str:
        build_llm_runtime(services)
        return "done"

    assert await closing_llm_runtime(services, work()) == "done"
    assert EXTRAS_KEY not in services.extras

    async def broken() -> None:
        build_llm_runtime(services)
        raise RuntimeError("the command failed")

    with pytest.raises(RuntimeError, match="the command failed"):
        await closing_llm_runtime(services, broken())
    assert EXTRAS_KEY not in services.extras
