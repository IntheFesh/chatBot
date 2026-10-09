"""The slice of the Windows API used by the project, behind an injectable protocol.

Windows-only behaviour (named mutexes, ``SetThreadExecutionState``, console code
page, console control events) is written against :class:`Win32` so that the
logic can be exercised on any platform with a test double, while
:class:`RealWin32` (ctypes, defined only on Windows) performs the real calls.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Protocol

ERROR_ALREADY_EXISTS = 183
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

CtrlHandler = Callable[[int], bool]


class Win32(Protocol):
    def create_mutex(self, name: str) -> tuple[int | None, int]:
        """``CreateMutexW`` -> ``(handle or None, GetLastError())``."""
        ...

    def close_handle(self, handle: int) -> None: ...

    def set_thread_execution_state(self, flags: int) -> int:
        """``SetThreadExecutionState``; returns the previous state, 0 on failure."""
        ...

    def set_console_output_cp(self, codepage: int) -> bool: ...

    def set_console_ctrl_handler(self, handler: CtrlHandler, add: bool) -> bool: ...


if sys.platform == "win32":  # pragma: win32-only
    import ctypes
    from ctypes import wintypes

    class RealWin32:
        """ctypes implementation backed by kernel32."""

        def __init__(self) -> None:
            self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            self._k32.CreateMutexW.restype = wintypes.HANDLE
            self._k32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._k32.CloseHandle.restype = wintypes.BOOL
            self._k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
            self._k32.SetThreadExecutionState.restype = wintypes.DWORD
            self._k32.SetConsoleOutputCP.argtypes = [wintypes.UINT]
            self._k32.SetConsoleOutputCP.restype = wintypes.BOOL
            self._handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            self._k32.SetConsoleCtrlHandler.argtypes = [self._handler_type, wintypes.BOOL]
            self._k32.SetConsoleCtrlHandler.restype = wintypes.BOOL
            self._callbacks: dict[CtrlHandler, object] = {}  # keep ctypes callbacks alive

        def create_mutex(self, name: str) -> tuple[int | None, int]:
            ctypes.set_last_error(0)
            handle = self._k32.CreateMutexW(None, False, name)
            error = ctypes.get_last_error()
            return (int(handle) if handle else None), error

        def close_handle(self, handle: int) -> None:
            self._k32.CloseHandle(handle)

        def set_thread_execution_state(self, flags: int) -> int:
            return int(self._k32.SetThreadExecutionState(flags))

        def set_console_output_cp(self, codepage: int) -> bool:
            return bool(self._k32.SetConsoleOutputCP(codepage))

        def set_console_ctrl_handler(self, handler: CtrlHandler, add: bool) -> bool:
            if add:
                callback = self._handler_type(handler)
                self._callbacks[handler] = callback
            else:
                callback = self._callbacks.pop(handler, None)  # type: ignore[assignment]
                if callback is None:
                    return False
            return bool(self._k32.SetConsoleCtrlHandler(callback, add))

    def load_win32() -> Win32:
        """The real Win32 binding."""
        return RealWin32()

else:

    def load_win32() -> Win32:
        """The real Win32 binding; unavailable on other platforms."""
        raise RuntimeError("the Windows API is only available on Windows")
