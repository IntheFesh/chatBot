"""Telling the person at the computer: a Windows notification (R-OPS-004).

:class:`WindowsToastNotifier` shows a toast through the ``windows-toasts`` package (a dependency
on Windows only).  Where there is no Windows notification area, :class:`ConsoleNotifier` writes
the same notice as a conspicuous panel on standard error, so a notice is never silently dropped.
Both are blocking: the delivery component calls them from a worker thread.

A toast needs the user's session (the scheduled task runs with ``InteractiveToken`` for exactly
this reason, R-OPS-001).
"""

from __future__ import annotations

import sys
from typing import Any, Protocol

from rich.console import Console
from rich.panel import Panel

from twin.ops.logging import get_logger

log = get_logger("twin.notify")

APP_NAME = "wechat-twin"


class NotifyError(Exception):
    """The notice could not be shown; the message is a short code, never a text."""


class Notifier(Protocol):
    """Shows a short notice to the person at the computer."""

    name: str

    def notify(self, title: str, body: str) -> None:
        """Show it or raise :class:`NotifyError`."""
        ...


class WindowsToastNotifier:
    """A Windows toast (``windows-toasts``)."""

    name = "windows_toast"

    def __init__(self) -> None:
        self._toaster: Any = None

    def notify(self, title: str, body: str) -> None:  # pragma: win32-only
        try:
            from windows_toasts import Toast, WindowsToaster

            if self._toaster is None:
                self._toaster = WindowsToaster(APP_NAME)
            self._toaster.show_toast(Toast([title, body]))
        except Exception as exc:
            raise NotifyError(type(exc).__name__) from None


class ConsoleNotifier:
    """A panel on standard error (the stand-in where there are no Windows notifications)."""

    name = "console"

    def notify(self, title: str, body: str) -> None:
        try:
            Console(stderr=True, highlight=False).print(
                Panel(body, title=title, border_style="bold yellow")
            )
        except (OSError, ValueError):
            raise NotifyError("console_unavailable") from None


def default_notifier(platform: str | None = None) -> Notifier:
    """The toast on Windows, the console panel elsewhere."""
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return WindowsToastNotifier()
    return ConsoleNotifier()
