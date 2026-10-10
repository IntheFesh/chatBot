"""The M0 probe, its stored result and its report (R-LLM-013)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY
from tests.support.simulated_deepseek import SimulatedDeepSeek
from twin.llm.capabilities import LlmCapabilities, load_capabilities
from twin.llm.probe import (
    CHECKS,
    GATE_CHECKS,
    HISTORY_KEY,
    HISTORY_LIMIT,
    PROBE_KEY,
    SCHEMA_VERSION,
    LlmProbe,
    ProbeConfig,
    ProbeReport,
    load_probe_report,
    load_probe_summary,
    save_probe,
)
from twin.llm.probe_report import (
    MEASURED_MARKER,
    PENDING_MARKER,
    render_pending_report,
    render_report,
    write_report,
)
from twin.llm.runtime import build_llm_runtime
from twin.services import Services
from twin.storage.models import CostLedger
from twin.storage.settings_store import get_setting, put_setting

ROOT = Path(__file__).resolve().parents[2]
NO_WAIT = ProbeConfig(cache_wait_s=0.0, cache_retry_wait_s=0.0)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def keyed(services: Services) -> Services:
    services.secrets.set("deepseek_api_key", TEST_KEY)
    return services


async def run_probe(
    services: Services,
    api: respx.MockRouter,
    simulated: SimulatedDeepSeek,
    config: ProbeConfig = NO_WAIT,
) -> ProbeReport:
    api.post(API).mock(side_effect=simulated)
    runtime = build_llm_runtime(services)
    task = asyncio.ensure_future(
        LlmProbe(runtime.client, clock=services.clock, config=config).run()
    )
    try:
        while not task.done():  # the manual clock only moves when advanced: release the waits
            await asyncio.wait({task}, timeout=0.01)
            if not task.done():
                await services.clock.advance(5.0)  # type: ignore[attr-defined]
        return await task
    finally:
        await runtime.client.aclose()


# ---------------------------------------------------------------- a healthy API


async def test_a_healthy_api_passes_m0_and_every_check_is_recorded(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek())
    assert report.m0_passed and report.fatal is None
    assert [c.number for c in report.checks] == [1, 2, 3, 4, 5, 6, 7]
    assert [c.id for c in report.checks] == list(CHECKS.values())
    assert [c.number for c in report.checks if c.gate] == list(GATE_CHECKS)
    assert all(c.ran and c.passed for c in report.checks)

    thinking = report.check("thinking_toggle")
    assert thinking.metrics["thinking_on_reasoning_returned"] is True
    assert thinking.metrics["thinking_off_reasoning_returned"] is False
    assert thinking.metrics["thinking_on_reasoning_tokens"] == 8

    cache = report.check("cache_hit")
    assert cache.metrics["first_cache_hit_tokens"] == 0
    assert cache.metrics["repeat_cache_hit_tokens"][0] > 0
    assert cache.metrics["prompt_tokens"] > 1000  # a genuinely long prefix

    vision = report.check("vision")
    assert (vision.metrics["jpeg_ok"], vision.metrics["png_ok"], vision.metrics["gif_ok"]) == (
        True,
        True,
        True,
    )
    assert vision.metrics["gif_frames"] > 1
    assert all(vision.metrics["answer_names_scene"].values())

    json_check = report.check("json_output")
    assert (
        json_check.metrics["thinking_off_parsed"] == 3 == json_check.metrics["thinking_on_parsed"]
    )

    detail = report.check("detail_param")
    assert detail.metrics["detail_accepted"] is True
    sizes = report.check("image_tokens").metrics["per_image"]
    assert [row["width"] for row in sizes] == [64, 256, 512, 1024, 2048]
    tokens = [row["tokens"] for row in sizes]
    assert tokens == sorted(tokens) and max(tokens) <= 1024
    assert tokens[0] == SimulatedDeepSeek.image_tokens(64, 64)

    cost = report.check("latency_cost")
    assert cost.metrics["requests"] == len(report.requests) > 20
    assert cost.metrics["successful"] == cost.metrics["requests"]
    assert cost.metrics["total_cost_usd"] == pytest.approx(report.total_cost_usd, rel=1e-3)
    assert report.total_cost_usd > 0


async def test_probe_requests_use_only_synthetic_content(
    keyed: Services, api: respx.MockRouter
) -> None:
    simulated = SimulatedDeepSeek()
    await run_probe(keyed, api, simulated)
    texts: set[str] = set()
    for body in simulated.requests:
        assert body["model"] == "deepseek-flash"
        for message in body["messages"]:
            content = message["content"]
            for part in (
                content if isinstance(content, list) else [{"type": "text", "text": content}]
            ):
                if part["type"] == "text":
                    texts.add(part["text"].split("\n", 1)[0])
    assert any(t.startswith("This text is synthetic filler") for t in texts)
    assert all(t.isascii() for t in texts)  # no real conversation, nothing but probe prompts


async def test_every_probe_call_is_on_the_one_time_account_under_the_probe_batch(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek())
    with keyed.db.session() as session:
        rows = list(session.execute(select(CostLedger)).scalars())
        assert len(rows) == len(report.requests)
        assert {(r.purpose, r.account, r.batch_id) for r in rows} == {
            ("probe", "one_time", report.run_id)
        }
        assert sum(r.cost_usd for r in rows) == pytest.approx(report.total_cost_usd, rel=1e-3)
    runtime = build_llm_runtime(keyed)
    assert runtime.budget.status(force=True).daily_spent == 0.0  # never degrades the bot


async def test_the_learned_capabilities_follow_the_measurements(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek())
    caps = report.capabilities
    assert caps.detail_supported and caps.gif_supported and caps.json_in_thinking
    assert caps.measured and len(caps.image_tokens) == 5
    assert caps.image_tokens[0] == (64 * 64, SimulatedDeepSeek.image_tokens(64, 64))


# ------------------------------------------------------------- what it must notice


async def test_a_rejected_gif_is_recorded_but_does_not_fail_m0(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(reject_gif=True))
    vision = report.check("vision")
    assert vision.passed and vision.metrics["gif_ok"] is False and vision.metrics["gif_rejected"]
    assert report.m0_passed
    assert report.capabilities.gif_supported is False
    assert any("GIF" in note for note in vision.notes)


async def test_a_rejected_detail_parameter_is_a_measurement_and_changes_the_client(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(reject_detail=True))
    detail = report.check("detail_param")
    assert detail.metrics["detail_accepted"] is False and detail.passed
    assert report.m0_passed and report.capabilities.detail_supported is False


async def test_unreliable_json_with_thinking_fails_m0_and_is_flagged(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(json_valid_with_thinking=False))
    json_check = report.check("json_output")
    assert not json_check.passed and json_check.metrics["thinking_off_ok"] is True
    assert json_check.metrics["thinking_on_parsed"] == 0
    assert json_check.metrics["thinking_on_parse_failures"] == 3
    assert any("planner" in note for note in json_check.notes)
    assert not report.m0_passed and report.capabilities.json_in_thinking is False


async def test_a_cache_that_never_hits_fails_after_waiting_and_retrying(
    keyed: Services, api: respx.MockRouter
) -> None:
    config = ProbeConfig(cache_wait_s=2.0, cache_retry_wait_s=7.0, cache_retries=2)
    report = await run_probe(keyed, api, SimulatedDeepSeek(cache_works=False), config)
    cache = report.check("cache_hit")
    assert not cache.passed and not report.m0_passed
    assert cache.metrics["repeats"] == 3 and cache.metrics["repeat_cache_hit_tokens"] == [0, 0, 0]
    assert cache.metrics["waited_s"] == pytest.approx(2.0 + 7.0 * 2)
    assert len(cache.requests) == 4
    assert keyed.clock.sleeps[:3] == [2.0, 7.0, 7.0]  # type: ignore[attr-defined]


async def test_the_cache_check_stops_repeating_after_the_first_hit(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek())
    assert len(report.check("cache_hit").requests) == 2


async def test_thinking_that_returns_no_reasoning_fails_the_first_check(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(reasoning_when_thinking=False))
    assert not report.check("thinking_toggle").passed and not report.m0_passed
    stray = await run_probe(keyed, api, SimulatedDeepSeek(reasoning_when_not_thinking=True))
    assert not stray.check("thinking_toggle").passed
    assert stray.check("thinking_toggle").notes


async def test_a_rejected_jpeg_fails_the_vision_check(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(reject_jpeg=True))
    vision = report.check("vision")
    assert not vision.passed and vision.metrics["jpeg_ok"] is False
    assert not report.m0_passed
    assert not report.check("image_tokens").passed  # nothing could be measured with JPEG


async def test_a_bad_key_stops_the_probe_with_a_fatal_message(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(unauthorized=True))
    assert report.fatal is not None and "check 1" in report.fatal
    assert not report.m0_passed
    assert [c.ran for c in report.checks[:6]] == [False] * 6
    assert report.checks[-1].id == "latency_cost"


async def test_running_out_of_balance_midway_keeps_what_was_measured(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(out_of_balance_after=5))
    assert report.fatal is not None and "balance" in report.fatal.lower()
    assert report.check("thinking_toggle").ran and report.check("thinking_toggle").passed
    assert report.check("cache_hit").ran and report.check("cache_hit").passed
    assert not report.check("vision").ran and not report.m0_passed


# ------------------------------------------------------------------ persistence


async def test_the_result_is_stored_for_the_m0_evaluator_and_the_client(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(reject_detail=True))
    with keyed.db.transaction() as session:
        save_probe(session, report, keyed.clock)
    with keyed.db.session() as session:
        stored = get_setting(session, PROBE_KEY)
        assert stored["schema_version"] == SCHEMA_VERSION == 1 and stored["live"] is True
        assert stored["m0_passed"] is True and stored["run_id"] == report.run_id
        assert {c["id"]: c["passed"] for c in stored["checks"] if c["gate"]} == {
            "thinking_toggle": True,
            "cache_hit": True,
            "vision": True,
            "json_output": True,
        }
        loaded = load_probe_report(session)
        assert loaded is not None and loaded.to_json() == report.to_json()
        summary = load_probe_summary(session)
        assert summary is not None and summary.m0_passed
        assert set(summary.gate_checks) == {"thinking_toggle", "cache_hit", "vision", "json_output"}
        assert set(summary.measurements) == {"detail_param", "image_tokens", "latency_cost"}
        assert summary.measurements["detail_param"]["detail_accepted"] is False
        caps = load_capabilities(session)
        assert caps.detail_supported is False and caps.measured
        history = get_setting(session, HISTORY_KEY)
        assert [h["run_id"] for h in history] == [report.run_id]


async def test_history_is_bounded_and_an_aborted_probe_keeps_the_old_capabilities(
    keyed: Services, api: respx.MockRouter
) -> None:
    good = await run_probe(keyed, api, SimulatedDeepSeek())
    for _ in range(HISTORY_LIMIT + 2):
        with keyed.db.transaction() as session:
            save_probe(session, good, keyed.clock)
    aborted = await run_probe(keyed, api, SimulatedDeepSeek(unauthorized=True))
    with keyed.db.transaction() as session:
        save_probe(session, aborted, keyed.clock)
    with keyed.db.session() as session:
        assert len(get_setting(session, HISTORY_KEY)) == HISTORY_LIMIT
        assert load_capabilities(session).measured_at == good.capabilities.measured_at
        summary = load_probe_summary(session)
        assert summary is not None and not summary.m0_passed and summary.fatal


def test_nothing_is_loaded_before_a_probe_or_for_another_schema(services: Services) -> None:
    with services.db.session() as session:
        assert load_probe_report(session) is None and load_probe_summary(session) is None
    with services.db.transaction() as session:
        put_setting(session, PROBE_KEY, {"schema_version": 99}, clock=services.clock)
    with services.db.session() as session:
        assert load_probe_report(session) is None


def test_capabilities_survive_a_json_round_trip_and_bad_data() -> None:
    caps = LlmCapabilities(
        detail_supported=False, image_tokens=((4096, 20), (262144, 300)), measured_at="2026-10-09"
    )
    assert LlmCapabilities.from_json(caps.to_json()) == caps
    assert LlmCapabilities.from_json("junk") == LlmCapabilities()
    assert LlmCapabilities.from_json({"image_tokens": [["a", "b"]]}) == LlmCapabilities()


# --------------------------------------------------------------------- the report


def test_the_shipped_report_is_the_pending_template_until_a_real_probe_ran() -> None:
    text = (ROOT / "docs" / "LLM_REPORT.md").read_text(encoding="utf-8")
    if MEASURED_MARKER in text:
        assert "运行编号" in text  # a real probe result replaced the template
    else:
        assert text == render_pending_report()


def test_the_pending_report_states_no_measurement() -> None:
    text = render_pending_report()
    assert PENDING_MARKER in text and "待实测" in text and MEASURED_MARKER not in text
    for name in CHECKS.values():
        assert f"`{name}`" in text
    assert "twin secrets set deepseek_api_key" in text and "twin llm probe" in text
    assert "官方文档核对" in text and "2026-10-09" in text
    assert "已实测" not in text and "运行编号" not in text


async def test_a_measured_report_shows_verdict_tables_and_warnings(
    keyed: Services, api: respx.MockRouter, tmp_path: Path
) -> None:
    good = await run_probe(keyed, api, SimulatedDeepSeek())
    text = render_report(good)
    assert MEASURED_MARKER in text and PENDING_MARKER not in text and good.run_id in text
    assert "M0 判定：**通过**" in text and "| 2048×2048 |" in text
    assert "官方文档核对" in text
    bad = await run_probe(keyed, api, SimulatedDeepSeek(json_valid_with_thinking=False))
    bad_text = render_report(bad)
    assert "M0 判定：**未通过**" in bad_text and "主动消息规划" in bad_text
    aborted = await run_probe(keyed, api, SimulatedDeepSeek(unauthorized=True))
    assert "探针中途停止" in render_report(aborted)
    target = tmp_path / "out" / "LLM_REPORT.md"
    write_report(target, text)
    assert target.read_text(encoding="utf-8") == text


# ------------------------------------------------------- partial and odd outcomes


async def test_json_that_fails_only_without_thinking_is_noted_separately(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(json_valid_without_thinking=False))
    json_check = report.check("json_output")
    assert not json_check.passed and json_check.metrics["thinking_on_ok"] is True
    assert json_check.metrics["thinking_off_parse_failures"] == 3
    assert any("disabled" in note for note in json_check.notes)
    assert report.capabilities.json_in_thinking is True  # thinking mode itself was fine


async def test_an_api_error_during_json_is_not_mistaken_for_unreliable_json(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(json_server_error=422))
    json_check = report.check("json_output")
    assert not json_check.passed and json_check.metrics["thinking_on_parse_failures"] == 0
    assert json_check.metrics["thinking_on_api_failures"] == 3
    assert report.capabilities.json_in_thinking is True  # no evidence against it


async def test_an_inconclusive_detail_result_leaves_the_client_unchanged(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(detail_server_error=429))
    detail = report.check("detail_param")
    assert detail.metrics["detail_accepted"] is None and not detail.passed
    assert report.m0_passed  # a measurement that could not be taken does not fail M0
    assert report.capabilities.detail_supported is True


async def test_running_out_of_balance_during_the_json_check_stops_the_probe(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek(out_of_balance_after=8))
    assert report.fatal is not None and "check 4" in report.fatal
    assert report.check("vision").passed and not report.check("json_output").ran


async def test_the_stored_summary_names_its_schema_version(
    keyed: Services, api: respx.MockRouter
) -> None:
    report = await run_probe(keyed, api, SimulatedDeepSeek())
    with keyed.db.transaction() as session:
        save_probe(session, report, keyed.clock)
    with keyed.db.session() as session:
        summary = load_probe_summary(session)
    assert summary is not None and summary.schema_version == 1
