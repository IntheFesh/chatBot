"""Sleep inhibition (R-OPS-002) and UTF-8 console set-up (CLAUDE.md section 7)."""

from __future__ import annotations

import io
import os
import sys

import pytest

from tests.support.win32 import FakeWin32
from twin.ops.console import CP_UTF8, ensure_utf8
from twin.ops.power import (
    NotNeededPowerManager,
    PowerError,
    WindowsPowerManager,
    default_power_manager,
)
from twin.ops.winapi import ES_CONTINUOUS, ES_SYSTEM_REQUIRED


def test_windows_manager_sets_and_restores_the_execution_state() -> None:
    fake = FakeWin32()
    manager = WindowsPowerManager(fake)
    assert not manager.active
    manager.start()
    assert manager.active
    assert fake.execution_states == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    manager.stop()
    assert not manager.active
    assert fake.execution_states == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]
    manager.stop()  # stopping twice is harmless
    assert len(fake.execution_states) == 2


def test_windows_manager_sets_and_clears_on_the_same_thread() -> None:
    import threading

    threads: list[int] = []

    class Recording(FakeWin32):
        def set_thread_execution_state(self, flags: int) -> int:
            threads.append(threading.get_ident())
            return super().set_thread_execution_state(flags)

    manager = WindowsPowerManager(Recording())
    manager.start()
    manager.stop()
    assert len(threads) == 2 and threads[0] == threads[1] != threading.get_ident()


def test_windows_manager_reports_failure_and_double_start() -> None:
    failing = WindowsPowerManager(FakeWin32(state_ok=False))
    with pytest.raises(PowerError, match="failed"):
        failing.start()
    manager = WindowsPowerManager(FakeWin32())
    manager.start()
    try:
        with pytest.raises(PowerError, match="already started"):
            manager.start()
    finally:
        manager.stop()


def test_non_windows_manager_is_a_real_implementation_that_records_why() -> None:
    manager = NotNeededPowerManager("linux")
    assert not manager.started and not manager.active
    assert "not needed on platform 'linux'" in manager.description
    manager.start()
    assert manager.started
    manager.stop()
    assert not manager.started


def test_default_manager_matches_the_platform() -> None:
    assert isinstance(default_power_manager("linux"), NotNeededPowerManager)
    assert isinstance(default_power_manager("darwin"), NotNeededPowerManager)
    assert isinstance(default_power_manager(), NotNeededPowerManager if sys.platform != "win32" else WindowsPowerManager)


@pytest.mark.windows
def test_real_windows_manager_runs() -> None:
    manager = default_power_manager("win32")
    manager.start()
    assert manager.active
    manager.stop()


# ----------------------------------------------------------------- console


def test_ensure_utf8_sets_environment_and_reconfigures_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    ensure_utf8(platform="linux")
    assert os.environ["PYTHONUTF8"] == "1" and os.environ["PYTHONIOENCODING"] == "utf-8"
    sys.stdout.write("你好，微信")
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8") == "你好，微信"
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"


def test_ensure_utf8_sets_the_windows_console_code_page(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeWin32()
    ensure_utf8(platform="win32", win32=fake)
    assert fake.code_pages == [CP_UTF8] == [65001]
    ensure_utf8(platform="linux", win32=fake)
    assert fake.code_pages == [65001]  # only Windows touches the console code page


def test_ensure_utf8_tolerates_streams_that_cannot_be_reconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Plain:
        def write(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(sys, "stdout", Plain())
    monkeypatch.setattr(sys, "stderr", Plain())
    ensure_utf8(platform="linux")  # must not raise
