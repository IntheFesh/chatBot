"""The notice on the screen: a Windows toast, or a panel on the console (R-OPS-004)."""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar

import pytest

from twin.ops import notify as notify_module
from twin.ops.notify import (
    APP_NAME,
    ConsoleNotifier,
    NotifyError,
    WindowsToastNotifier,
    default_notifier,
)


def test_the_console_panel_carries_the_title_and_the_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ConsoleNotifier().notify("wechat-twin：备份失败", "用 twin backup now 手动备份一次")
    captured = capsys.readouterr()
    assert captured.out == ""  # standard output stays free for the commands
    assert "wechat-twin：备份失败" in captured.err and "twin backup now" in captured.err


def test_a_console_that_cannot_be_written_to_is_a_notify_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(self: object, *args: object, **kwargs: object) -> None:
        raise OSError("closed")

    monkeypatch.setattr(notify_module.Console, "print", broken)
    with pytest.raises(NotifyError) as caught:
        ConsoleNotifier().notify("t", "b")
    assert str(caught.value) == "console_unavailable"


def test_windows_gets_the_toast_and_everything_else_the_panel() -> None:
    assert isinstance(default_notifier("win32"), WindowsToastNotifier)
    assert isinstance(default_notifier("linux"), ConsoleNotifier)
    assert isinstance(default_notifier("darwin"), ConsoleNotifier)
    assert isinstance(default_notifier(), ConsoleNotifier) == (sys.platform != "win32")
    assert WindowsToastNotifier.name == "windows_toast" and ConsoleNotifier.name == "console"


class FakeToaster:
    """The shape of ``windows_toasts`` the notifier uses: ``WindowsToaster(app).show_toast``."""

    created: ClassVar[list[str]] = []
    shown: ClassVar[list[Any]] = []

    def __init__(self, application: str) -> None:
        self.created.append(application)

    def show_toast(self, toast: Any) -> None:
        self.shown.append(toast)


class FakeToast:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines


@pytest.fixture
def fake_library(monkeypatch: pytest.MonkeyPatch) -> type[FakeToaster]:
    module = types.ModuleType("windows_toasts")
    module.WindowsToaster = FakeToaster  # type: ignore[attr-defined]
    module.Toast = FakeToast  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windows_toasts", module)
    FakeToaster.created, FakeToaster.shown = [], []
    return FakeToaster


def test_the_toast_is_shown_through_the_library_with_the_title_and_the_text(
    fake_library: type[FakeToaster],
) -> None:
    notifier = WindowsToastNotifier()
    notifier.notify("标题", "正文一")
    notifier.notify("标题二", "正文二")
    assert fake_library.created == [APP_NAME]  # one toaster for all notices
    assert [toast.lines for toast in fake_library.shown] == [
        ["标题", "正文一"],
        ["标题二", "正文二"],
    ]


def test_the_cpp_runtime_is_loaded_before_the_toast_library_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``windows_toasts`` brings an old msvcp140.dll that ``torch`` cannot start with: the system
    runtime has to be in the process first (D-621).  The library is only importable once the
    preload has run, so a notifier that imported it first would fail here."""
    order: list[str] = []
    monkeypatch.setitem(sys.modules, "windows_toasts", None)  # an import now would fail

    def preload() -> tuple[str, ...]:
        order.append("preload")
        module = types.ModuleType("windows_toasts")
        module.WindowsToaster = FakeToaster  # type: ignore[attr-defined]
        module.Toast = FakeToast  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "windows_toasts", module)
        return ()

    monkeypatch.setattr(notify_module, "preload_system_runtime", preload)
    FakeToaster.created, FakeToaster.shown = [], []
    WindowsToastNotifier().notify("t", "b")
    assert order == ["preload"] and len(FakeToaster.shown) == 1


def test_a_library_that_fails_or_is_missing_is_a_notify_error_with_a_code_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "windows_toasts", None)  # not installed
    with pytest.raises(NotifyError) as missing:
        WindowsToastNotifier().notify("t", "b")
    assert str(missing.value) in ("ImportError", "ModuleNotFoundError")

    class Failing(FakeToaster):
        def show_toast(self, toast: Any) -> None:
            raise RuntimeError("the text of the toast must not travel in the error")

    module = types.ModuleType("windows_toasts")
    module.WindowsToaster = Failing  # type: ignore[attr-defined]
    module.Toast = FakeToast  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windows_toasts", module)
    with pytest.raises(NotifyError) as broken:
        WindowsToastNotifier().notify("t", "b")
    assert str(broken.value) == "RuntimeError"
