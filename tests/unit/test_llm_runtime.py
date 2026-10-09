"""Wiring of the LLM layer: services in, client and policies out (R-ARCH-003, R-LLM-007)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.deepseek import API, TEST_KEY, ok, request_json
from twin.config.runtime import BOT_TIMEZONE
from twin.llm.capabilities import LlmCapabilities, save_capabilities
from twin.llm.images import ImageInput
from twin.llm.pricing import CalendarOffPeakPolicy
from twin.llm.runtime import activate_offpeak_policy, build_llm_runtime
from twin.llm.synth_images import draw_png
from twin.llm.tokens import Calibration
from twin.llm.types import Purpose
from twin.ops.components import build_application
from twin.ops.jobs import DeferredOffPeakPolicy, get_offpeak_policy
from twin.ops.process_model import JobSpec, enqueue_heavy
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
    second = build_llm_runtime(services)
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
