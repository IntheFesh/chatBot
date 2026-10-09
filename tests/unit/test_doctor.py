"""``twin doctor`` checks (R-OPS-009 base items), exercised directly."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from collections import namedtuple
from pathlib import Path

import pytest

import twin.ops.doctor as doctor
from tests.support.credentials import BrokenCredentials, MemoryCredentials
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
    }
    values.update(overrides)
    return DoctorContext(**values)  # type: ignore[arg-type]


def result_of(check: doctor.DoctorCheck, ctx: DoctorContext) -> CheckResult:
    return check(ctx)


def test_all_checks_pass_on_a_healthy_setup(tmp_path: Path) -> None:
    ctx = context(tmp_path)
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
    }
    assert all(r.status is CheckStatus.OK for r in results), [
        r for r in results if r.status is not CheckStatus.OK
    ]
    assert exit_code(results) == 0


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
