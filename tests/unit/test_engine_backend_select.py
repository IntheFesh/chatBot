"""Choosing the backend, falling back to DeepSeek and coming back (R-SRV-004, R-LLM-008, R-SRV-005).

The selector reads the user's choice and the registry, looks at the style model through its
client and writes its decisions as runtime settings (which keep a history) and as alerts.  Time is
the manual clock: the ten minutes of the way back are ``clock.tick``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.style_models import ScriptedStyleClient, register_model
from tests.support.waiting import wait_until
from twin.app import HealthStatus
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK
from twin.engine.backend_select import (
    GATE_NOT_PASSED,
    NO_MODEL,
    NOT_ACTIVE,
    NOT_REGISTERED,
    TEMPLATE,
    UNHEALTHY,
    BackendMonitorComponent,
    BackendSelector,
)
from twin.engine.style_models import StyleModels
from twin.engine.types import PostAction, ReplyDraft, UsageSummary
from twin.llm.budget import BudgetLimits
from twin.services import Services
from twin.storage.models import Alert
from twin.storage.training_models import ModelRegistryEntry


def selector_for(
    services: Services,
    client: ScriptedStyleClient,
    *,
    limits: Callable[[], BudgetLimits] | None = None,
) -> BackendSelector:
    services.runtime.initialize()  # as at the start of the application: every setting has a value
    return BackendSelector(
        runtime=services.runtime,
        models=StyleModels(services.db),
        client=client,
        config=services.settings.backend,
        clock=services.clock,
        alerts=services.alerts,
        limits=limits,
    )


def want(services: Services, name: str) -> None:
    services.runtime.set(BACKEND_ACTIVE, name, by="command")  # type: ignore[arg-type]


def alerts(services: Services) -> list[tuple[str, str]]:
    with services.db.session() as session:
        rows = session.scalars(select(Alert).order_by(Alert.created_at, Alert.id))
        return [(row.category, row.severity) for row in rows]


def draft(
    *,
    backend: str = "style",
    failing: int = 0,
    usable: bool = True,
    actions: tuple[PostAction, ...] = (),
) -> ReplyDraft:
    """A reply as the pipeline reports it; ``failing`` outputs of the backend were refused."""
    meta: dict[str, Any] = {"attempt_violations": [["think_tag"]] * failing} if failing else {}
    return ReplyDraft(
        bubbles=(),
        quote=None,
        no_reply=False,
        needs_fallback=not usable,
        fallback_reason=None if usable else "violations",
        backend=backend,
        thinking=False,
        reasoning=None,
        plan=None,
        cost_usd=0.0,
        usage=UsageSummary(),
        timings_ms={},
        actions=actions,
        violations=(),
        attempts=1 + failing,
        meta=meta,
    )


# ----------------------------------------------------------------------- choosing


async def test_deepseek_is_the_choice_when_deepseek_is_asked_for(services: Services) -> None:
    selector = selector_for(services, ScriptedStyleClient())
    choice = await selector.choose()
    assert (choice.name, choice.requested, choice.reason) == ("deepseek", "deepseek", None)
    assert not choice.fell_back and services.runtime.get(BACKEND_FALLBACK) is None


@pytest.mark.parametrize("name", ["style", "hybrid"])
async def test_a_healthy_active_model_is_used_when_asked_for(services: Services, name: str) -> None:
    register_model(services)
    want(services, name)
    client = ScriptedStyleClient()
    choice = await selector_for(services, client).choose()
    assert choice.name == name and not choice.fell_back and client.health_calls == 1


async def test_the_server_is_looked_at_once_per_interval(
    services: Services, clock: ManualClock
) -> None:
    register_model(services)
    want(services, "style")
    client = ScriptedStyleClient()
    selector = selector_for(services, client)
    for _ in range(3):
        await selector.choose()
    assert client.health_calls == 1
    clock.tick(services.settings.backend.health_check_s + 1)
    await selector.choose()
    assert client.health_calls == 2


# ------------------------------------------------------------------------ falling back


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (lambda s: None, NO_MODEL),
        (lambda s: register_model(s, active=False), NO_MODEL),
        (lambda s: register_model(s, template_version="qwen3_think@1"), TEMPLATE),
    ],
)
async def test_without_a_usable_model_the_replies_go_to_deepseek_and_an_alert_says_so(
    services: Services, setup: Callable[[Services], object], code: str
) -> None:
    setup(services)
    want(services, "style")
    choice = await selector_for(services, ScriptedStyleClient()).choose()
    assert choice.name == "deepseek" and choice.requested == "style" and choice.fell_back
    assert choice.reason == f"fallback:{code}"
    assert alerts(services) == [("style_fallback", "warning")]
    assert services.runtime.get(BACKEND_ACTIVE) == "style"  # the user's choice is untouched


async def test_a_server_that_does_not_answer_is_a_fallback_with_a_full_record(
    services: Services,
) -> None:
    register_model(services)
    want(services, "hybrid")
    client = ScriptedStyleClient(healthy=False, detail="llama-server is still loading")
    selector = selector_for(services, client)
    choice = await selector.choose()
    assert choice.name == "deepseek" and choice.reason == f"fallback:{UNHEALTHY}"
    record = services.runtime.get(BACKEND_FALLBACK)
    assert record is not None and record["requested"] == "hybrid"
    assert record["reason"] == UNHEALTHY and record["by_budget"] is False and record["since"]
    history = services.runtime.history(BACKEND_FALLBACK)  # the audit trail of every switch
    assert [(h.by, h.old, bool(h.new)) for h in history if not h.created] == [("auto", None, True)]
    again = await selector.choose()  # a second reply does not raise a second alert
    assert again.name == "deepseek" and alerts(services) == [("style_fallback", "warning")]


async def test_a_style_server_that_fails_while_answering_is_a_fallback_at_once(
    services: Services,
) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient())
    assert (await selector.choose()).name == "style"
    failed = draft(
        backend="deepseek", actions=(PostAction("backend_error", detail="StyleModelError"),)
    )
    selector.record("style", failed)
    assert (await selector.choose()).reason == "fallback:error"
    assert alerts(services) == [("style_fallback", "warning")]


async def test_a_failing_planner_is_not_the_style_model_failing(services: Services) -> None:
    register_model(services)
    want(services, "hybrid")
    selector = selector_for(services, ScriptedStyleClient())
    await selector.choose()
    planner_down = draft(
        backend="deepseek", actions=(PostAction("backend_error", detail="InvalidRequestError"),)
    )
    selector.record("hybrid", planner_down)
    assert (await selector.choose()).name == "hybrid" and alerts(services) == []


async def test_hard_violations_in_a_row_cause_a_fallback_and_a_good_reply_starts_the_count_over(
    services: Services,
) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient())
    await selector.choose()
    two_bad = draft(backend="deepseek", failing=2)  # both tries of the style model refused
    selector.record("style", two_bad)
    assert (await selector.choose()).name == "style"  # two are not three
    selector.record("style", draft(backend="style"))  # a clean reply of the style model
    selector.record("style", two_bad)
    assert (await selector.choose()).name == "style"  # the count started over
    selector.record("style", two_bad)  # four in a row now
    choice = await selector.choose()
    assert choice.name == "deepseek" and choice.reason == "fallback:violations"
    assert alerts(services) == [("style_fallback", "warning")]


async def test_a_reply_that_the_style_model_got_right_on_its_second_try_resets_the_count(
    services: Services,
) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient())
    await selector.choose()
    selector.record("style", draft(backend="deepseek", failing=2))
    selector.record("style", draft(backend="style", failing=1))  # the second try was clean
    selector.record("style", draft(backend="deepseek", failing=2))
    assert (await selector.choose()).name == "style"


async def test_replies_of_deepseek_do_not_count_against_the_style_model(
    services: Services,
) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient())
    await selector.choose()
    for _ in range(5):
        selector.record("deepseek", draft(backend="deepseek", failing=2))
    assert (await selector.choose()).name == "style"


# ------------------------------------------------------------------ the way back


async def test_ten_healthy_minutes_bring_the_style_model_back_and_say_so(
    services: Services, clock: ManualClock
) -> None:
    register_model(services)
    want(services, "style")
    client = ScriptedStyleClient(healthy=False)
    selector = selector_for(services, client)
    assert (await selector.choose()).name == "deepseek"
    client.healthy = True
    clock.tick(31)
    assert (await selector.choose()).name == "deepseek"  # healthy since now; the clock starts
    clock.tick(9 * 60)
    assert (await selector.choose()).name == "deepseek"  # nine minutes are not ten
    clock.tick(61)
    choice = await selector.choose()
    assert choice.name == "style" and not choice.fell_back
    assert services.runtime.get(BACKEND_FALLBACK) is None
    assert alerts(services) == [("style_fallback", "warning"), ("style_recovered", "info")]
    changes = [(h.by, bool(h.new)) for h in services.runtime.history(BACKEND_FALLBACK)]
    assert changes[-2:] == [("auto", True), ("auto", False)]  # switched away, switched back


async def test_a_break_in_the_health_starts_the_ten_minutes_over(
    services: Services, clock: ManualClock
) -> None:
    register_model(services)
    want(services, "style")
    client = ScriptedStyleClient(healthy=False)
    selector = selector_for(services, client)
    await selector.choose()
    client.healthy = True
    clock.tick(31)
    await selector.choose()
    clock.tick(8 * 60)
    client.healthy = False
    await selector.choose()  # unhealthy again at the next look
    clock.tick(31)
    await selector.choose()
    client.healthy = True
    clock.tick(31)
    await selector.choose()
    clock.tick(9 * 60)
    assert (await selector.choose()).name == "deepseek"  # only nine since the break
    clock.tick(2 * 60)
    assert (await selector.choose()).name == "style"


async def test_a_recovery_time_of_zero_switches_back_at_the_first_healthy_look(
    services: Services, clock: ManualClock
) -> None:
    services.settings.backend.recover_after_min = 0
    register_model(services)
    want(services, "style")
    client = ScriptedStyleClient(healthy=False)
    selector = selector_for(services, client)
    await selector.choose()
    client.healthy = True
    clock.tick(31)
    assert (await selector.choose()).name == "style"


async def test_choosing_deepseek_ends_a_fallback_without_a_notice(services: Services) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient(healthy=False))
    await selector.choose()
    want(services, "deepseek")
    selector.note_user_choice()
    assert services.runtime.get(BACKEND_FALLBACK) is None
    assert (await selector.choose()).name == "deepseek"
    assert alerts(services) == [("style_fallback", "warning")]  # no "recovered" for a choice


# ---------------------------------------------------------- the budget and the gate


async def test_the_last_budget_level_prefers_a_healthy_model_that_passed_the_gate(
    services: Services,
) -> None:
    register_model(services, gate_passed=True)
    holder: list[BackendSelector] = []

    def limits() -> BudgetLimits:
        return BudgetLimits(
            level=4,
            examples_k=2,
            memory_budget_factor=0.25,
            chat_thinking_allowed=False,
            planner_thinking_allowed=False,
            proactive_allowed=False,
            prefer_style_backend=holder[0].available(),
            minimal_context=False,
        )

    selector = selector_for(services, ScriptedStyleClient(), limits=limits)
    holder.append(selector)
    assert not selector.available()  # nobody has looked at the server yet
    choice = await selector.choose()
    assert (choice.name, choice.requested, choice.reason) == ("style", "deepseek", "budget")
    assert selector.available()


async def test_a_model_activated_by_force_does_not_take_part_in_the_budget_degradation(
    services: Services,
) -> None:
    register_model(services, gate_passed=False)
    selector = selector_for(services, ScriptedStyleClient())
    await selector.probe()
    assert not selector.available()  # healthy, but it did not pass the gate (R-SRV-005)
    want(services, "style")  # ... which does not stop the user from choosing it
    assert (await selector.choose()).name == "style"


async def test_a_style_failure_while_the_budget_chose_it_is_a_fallback_that_comes_back_too(
    services: Services, clock: ManualClock
) -> None:
    register_model(services, gate_passed=True)
    holder: list[BackendSelector] = []

    def limits() -> BudgetLimits:
        return BudgetLimits(4, 2, 0.25, False, False, False, holder[0].available(), False)

    selector = selector_for(services, ScriptedStyleClient(), limits=limits)
    holder.append(selector)
    assert (await selector.choose()).name == "style"
    selector.record(
        "style",
        draft(backend="deepseek", actions=(PostAction("backend_error", detail="StyleModelError"),)),
    )
    record = services.runtime.get(BACKEND_FALLBACK)
    assert record is not None and record["by_budget"] is True and record["requested"] == "style"
    assert (await selector.choose()).name == "deepseek"  # not preferred while in fallback
    clock.tick(31)
    await selector.choose()
    clock.tick(11 * 60)
    assert (await selector.choose()).name == "style"  # healthy for ten minutes: back


async def test_the_chat_command_may_switch_only_to_a_model_that_is_fit_to_use(
    services: Services,
) -> None:
    client = ScriptedStyleClient()
    selector = selector_for(services, client)
    assert await selector.verify_switch("deepseek") is None
    refusal = await selector.verify_switch("style")
    assert refusal is not None and refusal.code == NOT_REGISTERED
    register_model(services, run_id="r1", active=False, gate_passed=None)
    refusal = await selector.verify_switch("hybrid")
    assert refusal is not None and refusal.code == NOT_ACTIVE
    register_model(services, run_id="r2", gate_passed=False)
    refusal = await selector.verify_switch("style")
    assert refusal is not None and refusal.code == GATE_NOT_PASSED and "r2" in refusal.detail
    with services.db.transaction(bump_state=False) as session:
        row = session.get(ModelRegistryEntry, "r2-Q5_K_M")
        assert row is not None
        row.gate_passed = True  # the release gate was passed meanwhile
    assert await selector.verify_switch("style") is None
    client.healthy, client.detail = False, "the server is down"
    refusal = await selector.verify_switch("style")
    assert refusal is not None and refusal.code == UNHEALTHY and "down" in refusal.detail
    assert await selector.verify_switch("deepseek") is None  # deepseek is always possible


async def test_the_status_says_what_was_asked_for_and_what_answers(services: Services) -> None:
    register_model(services)
    want(services, "style")
    selector = selector_for(services, ScriptedStyleClient(healthy=False))
    status = await selector.status()
    assert status.registered == 1 and status.model is not None and status.probe is not None
    assert (status.requested, status.effective) == ("style", "style")  # not yet decided
    await selector.choose()
    status = await selector.status()
    assert (status.requested, status.effective) == ("style", "deepseek")
    assert status.fallback is not None and status.fallback["reason"] == UNHEALTHY


# ---------------------------------------------------------------- the monitor


async def test_the_monitor_looks_at_the_model_while_nobody_is_talking(
    services: Services, clock: ManualClock
) -> None:
    register_model(services)
    want(services, "style")
    client = ScriptedStyleClient()
    selector = selector_for(services, client)
    monitor = BackendMonitorComponent(selector, clock, services.alerts, interval_s=30)
    await monitor.start()
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1)
        assert client.health_calls == 1 and monitor.health().status is HealthStatus.OK
        client.healthy = False
        await clock.advance(31)
        await wait_until(lambda: services.runtime.get(BACKEND_FALLBACK) is not None)
        assert alerts(services) == [("style_fallback", "warning")]
        assert monitor.health().status is HealthStatus.DEGRADED
        client.healthy = True
        await clock.advance(31)
        await clock.advance(11 * 60)
        await wait_until(lambda: services.runtime.get(BACKEND_FALLBACK) is None)
        assert monitor.health().status is HealthStatus.OK
    finally:
        await monitor.stop()
