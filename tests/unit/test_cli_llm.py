"""``twin llm probe`` and ``twin llm status`` (R-LLM-013, R-LLM-006, R-LLM-008)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from tests.support.deepseek import API, TEST_KEY
from tests.support.simulated_deepseek import SimulatedDeepSeek
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.llm.probe import PROBE_KEY
from twin.llm.probe_report import MEASURED_MARKER
from twin.llm.runtime import build_llm_runtime
from twin.ops.process_model import ExitCode
from twin.services import build_services
from twin.storage.settings_store import get_setting

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def store_key() -> None:
    result = runner.invoke(
        app, ["secrets", "set", "deepseek_api_key", "--stdin"], input=TEST_KEY + "\n"
    )
    assert result.exit_code == 0, result.output


def stored_probe() -> object:
    services = build_services(load_settings(), root=resolve_paths(load_settings()).root)
    try:
        with services.db.session() as session:
            return get_setting(session, PROBE_KEY)
    finally:
        services.close()


# --------------------------------------------------------------------- probe


def test_probe_needs_the_api_key_and_says_how_to_set_it(data_dir: Path) -> None:
    result = runner.invoke(app, ["llm", "probe", "--yes"])
    assert result.exit_code == int(ExitCode.SECRETS)
    assert "twin secrets set deepseek_api_key" in result.output


def test_probe_asks_before_spending_money(data_dir: Path, api: respx.MockRouter) -> None:
    store_key()
    route = api.post(API).mock(side_effect=SimulatedDeepSeek())
    result = runner.invoke(app, ["llm", "probe"], input="n\n")
    assert result.exit_code == 1 and "Run the probe now?" in result.output
    assert route.call_count == 0


def test_a_successful_probe_writes_the_report_and_the_database_record(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    simulated = SimulatedDeepSeek()
    api.post(API).mock(side_effect=simulated)
    report = tmp_path / "out" / "LLM_REPORT.md"
    result = runner.invoke(
        app, ["llm", "probe", "--yes", "--report", str(report), "--cache-wait", "0"]
    )
    assert result.exit_code == 0, result.output
    assert "M0 (DeepSeek) passed" in result.output and "total cost" in result.output
    for name in ("thinking_toggle", "cache_hit", "vision", "json_output", "latency_cost"):
        assert name in result.output
    text = report.read_text(encoding="utf-8")
    assert MEASURED_MARKER in text and "M0 判定：**通过**" in text
    assert TEST_KEY not in text and TEST_KEY not in result.output
    stored = stored_probe()
    assert isinstance(stored, dict) and stored["m0_passed"] is True
    assert all(request["model"] == "deepseek-flash" for request in simulated.requests)


def test_the_default_report_location_is_docs_llm_report_md_under_the_project_root(
    data_dir: Path, api: respx.MockRouter
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek())
    result = runner.invoke(app, ["llm", "probe", "--yes", "--cache-wait", "0"])
    assert result.exit_code == 0, result.output
    root = resolve_paths(load_settings()).root
    assert (root / "docs" / "LLM_REPORT.md").read_text(encoding="utf-8").count(MEASURED_MARKER) == 1


def test_no_report_still_stores_the_result(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek())
    target = tmp_path / "never.md"
    result = runner.invoke(
        app, ["llm", "probe", "--yes", "--no-report", "--report", str(target), "--cache-wait", "0"]
    )
    assert result.exit_code == 0 and not target.exists()
    assert isinstance(stored_probe(), dict)


def test_unreliable_json_with_thinking_stops_with_a_clear_message(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek(json_valid_with_thinking=False))
    report = tmp_path / "r.md"
    result = runner.invoke(
        app, ["llm", "probe", "--yes", "--report", str(report), "--cache-wait", "0"]
    )
    assert result.exit_code == 1
    assert "STOP" in result.output and "planner" in result.output
    assert "M0 判定：**未通过**" in report.read_text(encoding="utf-8")


def test_a_failed_check_makes_the_command_fail(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek(reject_jpeg=True))
    result = runner.invoke(
        app, ["llm", "probe", "--yes", "--report", str(tmp_path / "r.md"), "--cache-wait", "0"]
    )
    assert result.exit_code == 1 and "M0 check failed" in result.output


def test_a_rejected_key_is_reported_as_a_fatal_stop(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek(unauthorized=True))
    result = runner.invoke(
        app, ["llm", "probe", "--yes", "--report", str(tmp_path / "r.md"), "--cache-wait", "0"]
    )
    assert result.exit_code == 1 and "probe stopped at check 1" in result.output


# -------------------------------------------------------------------- status


def test_status_before_any_probe(data_dir: Path) -> None:
    result = runner.invoke(app, ["llm", "status"])
    assert result.exit_code == 0, result.output
    assert "NOT SET" in result.output and "documented defaults" in result.output
    assert "never run" in result.output and "degradation level: 0" in result.output
    assert "chat=deepseek-flash" in result.output


def test_status_after_a_probe_shows_the_learned_capabilities(
    data_dir: Path, api: respx.MockRouter, tmp_path: Path
) -> None:
    store_key()
    api.post(API).mock(side_effect=SimulatedDeepSeek(reject_detail=True))
    probe = runner.invoke(
        app, ["llm", "probe", "--yes", "--report", str(tmp_path / "r.md"), "--cache-wait", "0"]
    )
    assert probe.exit_code == 0, probe.output
    result = runner.invoke(app, ["llm", "status"])
    assert result.exit_code == 0, result.output
    assert "API key (deepseek_api_key): set" in result.output
    assert "detail=False" in result.output and "image_token_samples=5" in result.output
    assert "M0 passed" in result.output and "measured" in result.output
    assert "one-time spending today" in result.output
    assert TEST_KEY not in result.output


def test_status_is_read_only_and_does_not_touch_the_alert_table(data_dir: Path) -> None:
    result = runner.invoke(app, ["llm", "status"])
    assert result.exit_code == 0
    services = build_services(load_settings(), root=resolve_paths(load_settings()).root)
    try:
        from sqlalchemy import func, select

        from twin.storage.models import Alert

        with services.db.session() as session:
            assert session.execute(select(func.count()).select_from(Alert)).scalar_one() == 0
    finally:
        services.close()
    assert os.environ["TWIN_PATHS__DATA_DIR"] == str(data_dir)


# ------------------------------------------------------------- batch approval


def test_jobs_approve_records_the_approval_and_the_overrun_cap(data_dir: Path) -> None:
    services = build_services(load_settings(), root=resolve_paths(load_settings()).root)
    try:
        runtime = build_llm_runtime(services)
        runtime.batches.enqueue("replay-9", "replay", [{"n": 1}], [4.0])
    finally:
        services.close()
    result = runner.invoke(app, ["jobs", "approve", "replay-9", "--yes"])
    assert result.exit_code == 0, result.output
    assert "approved 1 job(s), $4.00" in result.output
    assert "pauses itself if it spends more than $4.80" in result.output
    services = build_services(load_settings(), root=resolve_paths(load_settings()).root)
    try:
        status = build_llm_runtime(services).batches.status("replay-9")
    finally:
        services.close()
    assert status.approved and status.cap_usd == pytest.approx(4.8)
