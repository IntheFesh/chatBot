"""A lost WeChat login fixes itself as soon as someone scans (R-CH-003, R-OPS-004).

When the server rejects the bot's token (error -14) the channel marks the login as expired and the
alert ``login_lost`` goes out (a notification and an e-mail that says only that the login has to
be renewed - the QR code is **never** in a mail or a notification, it exists on this computer
only).  :class:`LoginRecovery` is the other half, a component of ``twin run``:

1. it notices the expired login (the channel's own stored state, every ``check_s`` seconds);
2. it opens a **window on this computer** with a fresh QR code (:class:`QrWindow`: a small Tk
   window on Windows; the default image viewer where Tk is not available) and, if the server asks
   for the verification number shown on the phone, a box to type it in;
3. it runs the very login of ``twin channel login``
   (:func:`~twin.channel.ilink.flows.login_with_ui`) in a thread of its own - waiting for a
   person never blocks the application;
4. on success the credentials are stored, the poller carries on by itself, and the alert is closed
   with one "recovered" notice (:meth:`~twin.ops.alerts.AlertService.recover`).

If nobody scans (the login code expires several times, ten minutes in all) the window is closed
and the attempt is repeated after ``retry_s`` seconds; the alert stays open and is announced
again by the health monitor.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from twin.app import ComponentHealth, TaskSupervisor
from twin.channel.base import AuthState
from twin.channel.ilink.flows import login_with_ui
from twin.channel.ilink.login import LoginError, LoginResult
from twin.channel.ilink.qr import open_in_viewer, remove_old_pngs, save_png
from twin.channel.ilink.store import IlinkStore
from twin.channel.state import ChannelStateStore
from twin.ops.logging import get_logger
from twin.services import Services

log = get_logger("twin.login_recovery")

CHECK_S = 10.0
RETRY_S = 300.0
CODE_WAIT_S = 600.0


class QrWindow(Protocol):
    """A window on this computer that shows the login QR code and may ask for a number."""

    def show(self, png: Path, title: str) -> None: ...

    def status(self, text: str) -> None: ...

    def ask_code(self, prompt: str) -> str | None:
        """The number the person typed, or ``None`` if there is no way to ask."""
        ...

    def close(self) -> None: ...


class ViewerQrWindow:
    """The default image viewer shows the picture; nothing can be typed (no verification box)."""

    def show(self, png: Path, title: str) -> None:
        opened = open_in_viewer(png)
        log.info("login_qr_shown", viewer=opened, title_chars=len(title))

    def status(self, text: str) -> None:
        log.info("login_status", message=text[:80])

    def ask_code(self, prompt: str) -> str | None:
        return None

    def close(self) -> None:
        return None


class TkQrWindow:  # pragma: win32-only
    """A small always-on-top Tk window in a thread of its own."""

    def __init__(self) -> None:
        self._commands: Any = None
        self._answers: Any = None
        self._thread: threading.Thread | None = None

    def _start(self) -> None:
        import queue

        if self._thread is not None:
            return
        self._commands = queue.Queue()
        self._answers = queue.Queue()
        self._thread = threading.Thread(target=self._main, name="twin-qr-window", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.title("wechat-twin")
        root.attributes("-topmost", True)
        picture = tk.Label(root)
        picture.pack(padx=12, pady=(12, 4))
        caption = tk.Label(root, text="", wraplength=360, justify="center")
        caption.pack(padx=12, pady=(0, 12))
        images: list[Any] = []

        def pump() -> None:
            while not self._commands.empty():
                command, value = self._commands.get_nowait()
                if command == "show":
                    png, title = value
                    image = tk.PhotoImage(file=str(png))
                    images[:] = [image]
                    picture.configure(image=image)
                    caption.configure(text=title)
                    root.deiconify()
                    root.lift()
                elif command == "status":
                    caption.configure(text=value)
                elif command == "ask":
                    self._answers.put(simpledialog.askstring("wechat-twin", value, parent=root))
                elif command == "close":
                    root.destroy()
                    return
            root.after(200, pump)

        root.after(200, pump)
        root.mainloop()

    def show(self, png: Path, title: str) -> None:
        self._start()
        self._commands.put(("show", (png, title)))

    def status(self, text: str) -> None:
        if self._thread is not None:
            self._commands.put(("status", text))

    def ask_code(self, prompt: str) -> str | None:
        import queue

        self._start()
        self._commands.put(("ask", prompt))
        try:
            answer = self._answers.get(timeout=CODE_WAIT_S)
        except queue.Empty:
            return None
        return None if answer is None else str(answer)

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._commands.put(("close", None))
        thread.join(timeout=5.0)
        self._thread = None


def default_window(platform: str | None = None) -> QrWindow:
    """The Tk window on Windows when Tk is there, the image viewer otherwise."""
    import importlib.util
    import sys

    plat = sys.platform if platform is None else platform
    if plat == "win32" and importlib.util.find_spec("tkinter") is not None:  # pragma: win32-only
        return TkQrWindow()
    return ViewerQrWindow()


class WindowLoginUI:
    """The login screens of :class:`~twin.channel.ilink.login.IlinkLogin` on a :class:`QrWindow`."""

    def __init__(self, window: QrWindow, tmp_dir: Path) -> None:
        self._window = window
        self._tmp_dir = tmp_dir

    def show_qr(self, content: str, *, number: int, total: int) -> None:
        remove_old_pngs(self._tmp_dir)
        png = save_png(content, self._tmp_dir)
        self._window.show(png, f"微信登录失效：请用手机微信扫码（第 {number}/{total} 张）")

    def ask_verify_code(self, *, previous_was_wrong: bool) -> str:
        prompt = "请输入手机上显示的数字" + ("（上一次没有被接受）" if previous_was_wrong else "")
        code = self._window.ask_code(prompt)
        if not code:
            raise LoginError("the verification number was not entered")
        return code

    def info(self, message: str) -> None:
        self._window.status(message)

    def cleanup(self) -> None:
        remove_old_pngs(self._tmp_dir)
        self._window.close()


LoginRunner = Callable[[WindowLoginUI], Awaitable[LoginResult]]


class LoginRecovery:
    """Component: shows the QR window when the login has expired (see the module description)."""

    name = "login_recovery"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        services: Services,
        *,
        window: Callable[[], QrWindow] = default_window,
        runner: LoginRunner | None = None,
        check_s: float = CHECK_S,
        retry_s: float = RETRY_S,
    ) -> None:
        self._services = services
        self._clock = services.clock
        self._window = window
        self._runner = runner or self._real_runner
        self._check_s = check_s
        self._retry_s = retry_s
        self._store = IlinkStore(ChannelStateStore(services.db), services.clock)
        self._supervisor = TaskSupervisor(self.name, self._clock, services.alerts)
        self._next_try: float = 0.0
        self.attempts = 0
        self.recoveries = 0

    async def _real_runner(self, ui: WindowLoginUI) -> LoginResult:
        # a thread with its own event loop: waiting for a person must not stop the application
        return await asyncio.to_thread(asyncio.run, login_with_ui(self._services, ui))

    def needed(self) -> bool:
        return self._store.auth_record().state is AuthState.NEEDS_RELOGIN

    async def check_once(self) -> bool:
        """Start a recovery if the login is expired and it is time; True if one succeeded."""
        if self._clock.monotonic() < self._next_try:
            return False
        if not await asyncio.to_thread(self.needed):
            return False
        return await self.recover()

    async def recover(self) -> bool:
        """One attempt: window, scan, credentials, "recovered" notice."""
        self.attempts += 1
        ui = WindowLoginUI(self._window(), self._services.paths.tmp_dir)
        try:
            await self._runner(ui)
        except LoginError as exc:
            log.warning("login_recovery_failed", reason=str(exc)[:80])
            self._next_try = self._clock.monotonic() + self._retry_s
            return False
        finally:
            ui.cleanup()
        self.recoveries += 1
        await asyncio.to_thread(
            self._services.alerts.recover, "login_lost", "WeChat login works again"
        )
        log.info("login_recovered_by_scan")
        return True

    async def _loop(self) -> None:
        while True:
            await self.check_once()
            await self._clock.sleep(self._check_s)

    async def start(self) -> None:
        self._supervisor.spawn("watch", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        return self._supervisor.health()
