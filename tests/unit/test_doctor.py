"""``twin doctor`` checks (R-OPS-009 base items), exercised directly."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import twin.ops.doctor as doctor
from tests.support.credentials import BrokenCredentials, MemoryCredentials
from tests.support.network import OfflineTransport
from tests.support.ops import ScriptedRunner, healthy_machine_runner
from twin.config.loader import load_settings
from twin.config.secrets import BackendInfo, SecretStore
from twin.ops.doctor import CheckResult, CheckStatus, DoctorContext, exit_code, run_checks
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.storage import migrate
from twin.storage.keystore import KeyStore

Usage = namedtuple("Usage", "total used free")


def context(tmp_path: Path, **overrides: object) -> DoctorContext:
    settings = load_settings(None, {"paths": {"data_dir": str(tmp_path / "d")}})
    values: dict[str, object] = {
        "settings": settings,
        "secrets": SecretStore(MemoryCredentials()),
        "root": tmp_path,
        "http_transport": OfflineTransport(),
    }
    values.update(overrides)
    return DoctorContext(**values)  # type: ignore[arg-type]


def result_of(check: doctor.DoctorCheck, ctx: DoctorContext) -> CheckResult:
    return check(ctx)


def healthy_network(request: httpx.Request) -> httpx.Response:
    """Every host answers; DeepSeek says that the account has a balance."""
    if request.url.path == doctor.DEEPSEEK_BALANCE_PATH:
        return httpx.Response(
            200,
            json={
                "is_available": True,
                "balance_infos": [{"currency": "USD", "total_balance": "5.00"}],
            },
        )
    return httpx.Response(404)


def test_all_checks_pass_on_a_healthy_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever computer this runs on, the verdict is about the setup that is handed in.

    The Windows checks (scheduled task, power plan) and the GPU check ask the machine; on Windows
    they get a scripted runner instead of the CI computer's own task list and power plan.  The
    holiday check depends on the date, the key and balance checks on a stored key and on the
    network; the date, the key and the network are given too.  Nothing here may be only
    advisory: every check must be OK.  (The Windows branches of the individual checks are
    exercised on every platform in ``test_ops_doctor.py``.)
    """
    monkeypatch.setattr(doctor, "now_utc", lambda: datetime(2024, 3, 1, tzinfo=UTC))
    secrets = SecretStore(MemoryCredentials())
    secrets.set("deepseek_api_key", "synthetic-key-1")
    ctx = context(
        tmp_path,
        secrets=secrets,
        runner=healthy_machine_runner(),
        http_transport=httpx.MockTransport(healthy_network),
    )
    migrate.upgrade(ctx.paths().db_path)  # type: ignore[union-attr]
    results = run_checks(ctx)
    assert {r.name for r in results} >= {
        "python",
        "dependencies",
        "tzdata",
        "config",
        "keyring",
        "db-key",
        "data-dir",
        "database",
        "instances",
        "power",
        "holiday-calendar",
        "llm-config",
        "deepseek-key",
        "ilink-api",
        "ilink-cdn",
        "vector-model",
        "deepseek-net",
        "gpu",
        "scheduled-task",
        "power-plan",
    }
    assert all(r.status is CheckStatus.OK for r in results), [
        r for r in results if r.status is not CheckStatus.OK
    ]
    assert exit_code(results) == 0
    by_name = {r.name: r for r in results}
    # the Windows checks really looked at the scripted machine on Windows, and stood aside elsewhere
    on_windows = sys.platform == "win32"
    expected = "registered, InteractiveToken" if on_windows else "Windows only"
    assert expected in by_name["scheduled-task"].detail
    assert ("sleep is set to never" if on_windows else "Windows only") in (
        by_name["power-plan"].detail
    )


def test_a_machine_that_is_not_set_up_is_reported_by_the_same_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The healthy run above is only meaningful if the same inputs can also fail."""
    monkeypatch.setattr(doctor, "now_utc", lambda: datetime(2040, 1, 1, tzinfo=UTC))
    ctx = context(tmp_path, platform="win32", runner=ScriptedRunner({}))  # no schtasks, no powercfg
    migrate.upgrade(ctx.paths().db_path)  # type: ignore[union-attr]
    warned = {r.name for r in run_checks(ctx) if r.status is CheckStatus.WARN}
    assert {"holiday-calendar", "scheduled-task", "power-plan", "deepseek-key"} <= warned


def test_python_version_is_checked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Version(tuple):  # type: ignore[type-arg]
        pass

    monkeypatch.setattr(sys, "version_info", Version((3, 13, 1, "final", 0)))
    result = doctor.check_python(context(tmp_path))
    assert result.status is CheckStatus.FAIL and "3.13.1" in result.detail
    assert "uv python install 3.12" in result.hint


def test_missing_dependencies_are_listed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor, "REQUIRED_MODULES", ("typer", "no_such_module_xyz"))
    result = doctor.check_dependencies(context(tmp_path))
    assert result.status is CheckStatus.FAIL
    assert "no_such_module_xyz" in result.detail and "typer" not in result.detail
    assert result.hint == "run `uv sync`"


def test_dependency_without_distribution_metadata_still_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        doctor, "REQUIRED_MODULES", ("json",)
    )  # stdlib: importable, no distribution
    assert doctor.check_dependencies(context(tmp_path)).status is CheckStatus.OK


def test_tzdata_failure_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(name: str) -> object:
        raise KeyError(name)

    monkeypatch.setattr(doctor, "ZoneInfo", boom)
    result = doctor.check_tzdata(context(tmp_path))
    assert result.status is CheckStatus.FAIL and "tzdata" in result.hint


def test_config_check_reports_load_errors_and_missing_consent(tmp_path: Path) -> None:
    assert doctor.check_config(DoctorContext(None, "invalid configuration: x")).detail.startswith(
        "invalid"
    )
    assert doctor.check_config(DoctorContext(None)).status is CheckStatus.FAIL
    no_consent = load_settings(None, {"consent": {"confirmed_at": None}})
    result = doctor.check_config(DoctorContext(no_consent))
    assert result.status is CheckStatus.FAIL and "consent" in result.detail
    ok = doctor.check_config(context(tmp_path))
    assert ok.status is CheckStatus.OK and "2026-10-08" in ok.detail


def test_keyring_check_variants(tmp_path: Path) -> None:
    unusable_system = SecretStore(
        BrokenCredentials(), BackendInfo("WinVault", "system", False, "locked")
    )
    on_windows = doctor.check_keyring(context(tmp_path, secrets=unusable_system, platform="win32"))
    assert on_windows.status is CheckStatus.FAIL and "Credential Manager" in on_windows.hint
    on_linux = doctor.check_keyring(context(tmp_path, secrets=unusable_system, platform="linux"))
    assert on_linux.status is CheckStatus.FAIL and "TWIN_SECRETS_DIR" in on_linux.hint

    broken_io = SecretStore(BrokenCredentials(), BackendInfo("X", "system", True, "claims to work"))
    assert (
        "read/write test failed"
        in doctor.check_keyring(context(tmp_path, secrets=broken_io)).detail
    )

    class Forgetful(MemoryCredentials):
        def get_password(self, service_name: str, username: str, /) -> str | None:
            return "something else"

    mismatch = SecretStore(Forgetful(), BackendInfo("Forgetful", "system", True, "ok"))
    assert "read-back mismatch" in doctor.check_keyring(context(tmp_path, secrets=mismatch)).detail

    fallback = SecretStore(
        MemoryCredentials(), BackendInfo("EncryptedFileKeyring", "file", True, "fallback")
    )
    warned = doctor.check_keyring(context(tmp_path, secrets=fallback))
    assert warned.status is CheckStatus.WARN and "encrypted file fallback" in warned.detail

    fine = doctor.check_keyring(context(tmp_path))
    assert fine.status is CheckStatus.OK


def test_keyring_check_builds_the_default_store_when_none_is_given(tmp_path: Path) -> None:
    result = doctor.check_keyring(context(tmp_path, secrets=None))
    assert result.status is CheckStatus.WARN  # the test environment forces the file backend


def test_database_key_check(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    assert doctor.check_database_key(ctx).detail.startswith("no master key yet")
    assert ctx.secrets is not None
    keystore = KeyStore(ctx.secrets)
    ring = keystore.create_initial()
    keystore.add_key(ring)
    keystore.retire(ring, [1])
    assert "master key id 2" in doctor.check_database_key(ctx).detail
    ctx.secrets.delete("db-key-1")
    broken = doctor.check_database_key(ctx)
    assert broken.status is CheckStatus.FAIL and "db-key-1" in broken.detail
    assert doctor.check_database_key(DoctorContext(None)).status is CheckStatus.WARN


def test_data_dir_free_space_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ctx = context(tmp_path)
    gib = 1024**3
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100 * gib, 99 * gib, gib // 2))
    low = doctor.check_data_dir(ctx)
    assert low.status is CheckStatus.FAIL and "0.5 GB free" in low.detail
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100 * gib, 96 * gib, 3 * gib))
    assert doctor.check_data_dir(ctx).status is CheckStatus.WARN
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100 * gib, 10 * gib, 90 * gib))
    ok = doctor.check_data_dir(ctx)
    assert ok.status is CheckStatus.OK and "90 GB free" in ok.detail
    assert doctor.check_data_dir(DoctorContext(None)).status is CheckStatus.WARN


def test_database_check_states(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    paths = ctx.paths()
    assert paths is not None
    assert doctor.check_database(ctx).status is CheckStatus.WARN  # not initialised
    migrate.upgrade(paths.db_path)
    assert doctor.check_database(ctx).status is CheckStatus.OK
    connection = sqlite3.connect(paths.db_path)
    connection.execute("UPDATE alembic_version SET version_num = '9999'")
    connection.commit()
    connection.close()
    ahead = doctor.check_database(ctx)
    assert ahead.status is CheckStatus.FAIL and "newer than this program" in ahead.detail
    assert doctor.check_database(DoctorContext(None)).status is CheckStatus.WARN


def test_instances_check_reports_running_processes(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    paths = ctx.paths()
    assert paths is not None
    assert doctor.check_instances(ctx).detail == "no instance is running"
    lock = InstanceLock(LOCK_RUN, locks_dir=paths.locks_dir)
    assert lock.acquire()
    try:
        assert doctor.check_instances(ctx).detail == "running: run"
    finally:
        lock.release()
    assert doctor.check_instances(DoctorContext(None)).status is CheckStatus.WARN


def test_power_check_describes_the_platform(tmp_path: Path) -> None:
    assert "not needed" in doctor.check_power(context(tmp_path, platform="linux")).detail
    assert (
        "SetThreadExecutionState" in doctor.check_power(context(tmp_path, platform="win32")).detail
    )


def test_a_crashing_check_is_reported_instead_of_aborting_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def exploding(ctx: DoctorContext) -> CheckResult:
        raise RuntimeError("check bug")

    exploding.__name__ = "check_exploding"
    monkeypatch.setattr(doctor, "_CHECKS", [exploding, doctor.check_python])
    results = run_checks(context(tmp_path))
    assert results[0].name == "exploding" and results[0].status is CheckStatus.FAIL
    assert "check bug" in results[0].detail
    assert results[1].name == "python"
    assert exit_code(results) == 1


def test_doctor_check_decorator_registers_new_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_CHECKS", [])

    @doctor.doctor_check
    def check_extra(ctx: DoctorContext) -> CheckResult:
        return CheckResult("extra", CheckStatus.OK, "added by a later round")

    assert run_checks(DoctorContext(None))[0].detail == "added by a later round"


def test_holiday_calendar_check_warns_about_years_the_library_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    monkeypatch.setattr(doctor, "now_utc", lambda: datetime(2026, 10, 9, tzinfo=UTC))
    warn = doctor.check_holiday_calendar(context(tmp_path))
    assert warn.status is CheckStatus.WARN and "2027" in warn.detail
    assert "chinese-calendar" in warn.hint and "extra_peak_dates" in warn.hint
    monkeypatch.setattr(doctor, "now_utc", lambda: datetime(2024, 3, 1, tzinfo=UTC))
    ok = doctor.check_holiday_calendar(context(tmp_path))
    assert ok.status is CheckStatus.OK and "2024, 2025" in ok.detail
    monkeypatch.setattr(doctor, "now_utc", lambda: datetime(2040, 1, 1, tzinfo=UTC))
    assert doctor.check_holiday_calendar(context(tmp_path)).status is CheckStatus.WARN


def test_llm_config_check_validates_models_prices_and_vision(tmp_path: Path) -> None:
    good = doctor.check_llm_config(context(tmp_path))
    assert good.status is CheckStatus.OK and "deepseek-flash" in good.detail
    pro_vision = load_settings(None, {"deepseek": {"vision_model": "deepseek-v4-pro"}})
    bad = doctor.check_llm_config(context(tmp_path, settings=pro_vision))
    assert bad.status is CheckStatus.FAIL and "cannot read images" in bad.detail
    unpriced = load_settings(None, {"deepseek": {"offline_model": "mystery-model"}})
    missing = doctor.check_llm_config(context(tmp_path, settings=unpriced))
    assert missing.status is CheckStatus.FAIL and "mystery-model" in missing.detail
    legacy = load_settings(None, {"deepseek": {"vision_model": "deepseek-v4-flash"}})
    assert doctor.check_llm_config(context(tmp_path, settings=legacy)).status is CheckStatus.OK
    skipped = DoctorContext(settings=None, settings_error="broken")
    assert doctor.check_llm_config(skipped).status is CheckStatus.WARN


def test_deepseek_key_check_reports_presence_and_store_failures(tmp_path: Path) -> None:
    store = SecretStore(MemoryCredentials())
    missing = doctor.check_deepseek_key(context(tmp_path, secrets=store))
    assert missing.status is CheckStatus.WARN and "twin llm probe" in missing.hint
    store.set("deepseek_api_key", "synthetic-key-1")
    present = doctor.check_deepseek_key(context(tmp_path, secrets=store))
    assert present.status is CheckStatus.OK and "synthetic-key-1" not in present.detail
    broken = doctor.check_deepseek_key(context(tmp_path, secrets=SecretStore(BrokenCredentials())))
    assert broken.status is CheckStatus.WARN and "cannot read" in broken.detail


# ------------------------------------------------------- WeChat connectivity


def reachable(status: int = 404) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(status))


def test_ilink_hosts_that_answer_are_reported_as_reachable(tmp_path: Path) -> None:
    ctx = context(tmp_path, http_transport=reachable(404))
    for check in (doctor.check_ilink_api, doctor.check_ilink_cdn):
        result = result_of(check, ctx)
        assert result.status is CheckStatus.OK
        assert "reachable (HTTP 404" in result.detail
    assert "ilinkai.weixin.qq.com" in doctor.check_ilink_api(ctx).detail
    assert "novac2c.cdn.weixin.qq.com" in doctor.check_ilink_cdn(ctx).detail


def test_ilink_hosts_that_cannot_be_reached_warn_with_a_hint(tmp_path: Path) -> None:
    ctx = context(tmp_path)  # the default test transport refuses every connection
    for check in (doctor.check_ilink_api, doctor.check_ilink_cdn):
        result = result_of(check, ctx)
        assert result.status is CheckStatus.WARN
        assert "cannot connect (ConnectError)" in result.detail
        assert "proxy or VPN" in result.hint
    assert exit_code(run_checks(ctx)) == 0  # a network problem never fails the doctor


def test_a_slow_ilink_host_is_reported_as_a_timeout(tmp_path: Path) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    ctx = context(tmp_path, http_transport=httpx.MockTransport(slow))
    result = doctor.check_ilink_api(ctx)
    assert result.status is CheckStatus.WARN and "timed out" in result.detail


def test_the_connectivity_checks_are_skipped_for_the_console_channel(tmp_path: Path) -> None:
    settings = load_settings(
        None, {"paths": {"data_dir": str(tmp_path / "d")}, "channel": {"kind": "console"}}
    )
    ctx = context(tmp_path, settings=settings)
    for check in (doctor.check_ilink_api, doctor.check_ilink_cdn):
        result = result_of(check, ctx)
        assert result.status is CheckStatus.OK and "not needed" in result.detail


def test_the_vector_check_names_missing_libraries_and_notes_a_model_not_yet_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path)
    fresh = doctor.check_vector_model(ctx)
    assert fresh.status is CheckStatus.OK and "downloaded on first use" in fresh.detail

    monkeypatch.setattr(doctor, "VECTOR_MODULES", ("numpy", "no_such_vector_library"))
    missing = doctor.check_vector_model(ctx)
    assert missing.status is CheckStatus.FAIL and "no_such_vector_library" in missing.detail
    assert missing.hint == "run `uv sync`"


def test_the_vector_check_is_happy_once_the_model_is_on_disk(tmp_path: Path) -> None:
    from twin.retrieval.embedder import manifest_path

    ctx = context(tmp_path)
    paths = ctx.paths()
    assert paths is not None
    paths.embeddings_dir.mkdir(parents=True)
    manifest_path(paths.embeddings_dir, ctx.settings.retrieval.model).write_text(  # type: ignore[union-attr]
        '{"revision": "7999e1d3359715c5"}', encoding="utf-8"
    )
    found = doctor.check_vector_model(ctx)
    assert found.status is CheckStatus.OK and "revision 7999e1d3" in found.detail
    without_settings = doctor.check_vector_model(DoctorContext(None))
    assert without_settings.status is CheckStatus.OK
