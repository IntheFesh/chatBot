"""A recording stand-in for the Windows API (:class:`twin.ops.winapi.Win32`)."""

from __future__ import annotations

import threading

from twin.ops.winapi import ERROR_ALREADY_EXISTS, WM_CLOSE, CtrlHandler, WindowHandler


class FakeWin32:
    """Simulates named mutexes (shared through ``registry``), execution state and console."""

    def __init__(self, registry: dict[str, int] | None = None, *, state_ok: bool = True) -> None:
        self.registry: dict[str, int] = registry if registry is not None else {}
        self._handles: dict[int, str] = {}
        self._next_handle = 100
        self.state_ok = state_ok
        self.execution_states: list[int] = []
        self.code_pages: list[int] = []
        self.ctrl_handlers: list[tuple[CtrlHandler, bool]] = []
        self.mutex_fails = False
        self.window_fails = False
        self.windows: dict[int, WindowHandler] = {}
        self.closed: list[int] = []
        self._loop_ends = threading.Event()
        self._window_ready = threading.Event()

    def create_mutex(self, name: str) -> tuple[int | None, int]:
        if self.mutex_fails:
            return None, 5
        existed = self.registry.get(name, 0) > 0
        self.registry[name] = self.registry.get(name, 0) + 1
        self._next_handle += 1
        self._handles[self._next_handle] = name
        return self._next_handle, ERROR_ALREADY_EXISTS if existed else 0

    def close_handle(self, handle: int) -> None:
        name = self._handles.pop(handle)
        self.registry[name] -= 1

    def set_thread_execution_state(self, flags: int) -> int:
        self.execution_states.append(flags)
        return 0x80000000 if self.state_ok else 0

    def set_console_output_cp(self, codepage: int) -> bool:
        self.code_pages.append(codepage)
        return True

    def set_console_ctrl_handler(self, handler: CtrlHandler, add: bool) -> bool:
        self.ctrl_handlers.append((handler, add))
        return True

    # -- the hidden message window -------------------------------------------------

    def create_message_window(self, on_message: WindowHandler) -> int | None:
        if self.window_fails:
            return None
        hwnd = 5000 + len(self.windows)
        self.windows[hwnd] = on_message
        self._window_ready.set()
        return hwnd

    def run_message_loop(self) -> None:
        """Blocks like the real loop: until ``post_close`` (or a WM_CLOSE message) ends it."""
        self._loop_ends.wait(timeout=30)

    def post_close(self, hwnd: int) -> None:
        self.closed.append(hwnd)
        self.windows.pop(hwnd, None)
        self._loop_ends.set()

    def send_message(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        """Delivers to the window's handler like ``SendMessageW`` does (1 when it handled it)."""
        if message == WM_CLOSE:
            self.post_close(hwnd)
            return 0
        handler = self.windows.get(hwnd)
        return 1 if handler is not None and handler(message, wparam, lparam) else 0

    def wait_for_window(self, timeout: float = 5.0) -> bool:
        return self._window_ready.wait(timeout)
