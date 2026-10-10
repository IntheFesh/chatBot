"""The slice of the Windows API used by the project, behind an injectable protocol.

Windows-only behaviour (named mutexes, ``SetThreadExecutionState``, console code
page, console control events) is written against :class:`Win32` so that the
logic can be exercised on any platform with a test double, while
:class:`RealWin32` (ctypes, defined only on Windows) performs the real calls.
"""

from __future__ import annotations

import itertools
import os
import sys
from collections.abc import Callable
from typing import Any, Protocol

ERROR_ALREADY_EXISTS = 183
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
WM_POWERBROADCAST = 0x0218
PBT_APMSUSPEND = 0x0004
PBT_APMRESUMESUSPEND = 0x0007
PBT_APMRESUMEAUTOMATIC = 0x0012
TRUE = 1

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001

CtrlHandler = Callable[[int], bool]
WindowHandler = Callable[[int, int, int], bool]

_window_numbers = itertools.count(1)


def next_window_class_name() -> str:
    """A window class name no other window of this process has had (D-624).

    A window class belongs to the process, not to the :class:`Win32` object that registered it, and
    stays registered as long as its thread lives.  A name counted per object (``...-1`` for the
    first window of *every* object) is taken again by the next object - a component started a
    second time in one process, a test that simulates a crash by abandoning an application whose
    window thread is still running - and ``RegisterClassW`` then fails with "class already
    exists": no hidden window, no ``WM_POWERBROADCAST``.  The counter is of the process.
    """
    return f"wechat-twin-power-{os.getpid()}-{next(_window_numbers)}"


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

    def create_message_window(self, on_message: WindowHandler) -> int | None:
        """A hidden top-level window on the calling thread; ``None`` if it cannot be made.

        Every message the window receives is passed to ``on_message(message, wparam, lparam)``;
        the handler returns ``True`` for a message it dealt with (the window then answers TRUE).
        It is a top-level window and not a message-only window because broadcast messages such
        as ``WM_POWERBROADCAST`` are not sent to message-only windows.
        """
        ...

    def run_message_loop(self) -> None:
        """Run the message loop of the calling thread until the window has been closed."""
        ...

    def post_close(self, hwnd: int) -> None:
        """Ask the window to close (callable from any thread); its message loop then ends."""
        ...

    def send_message(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        """``SendMessageW``: deliver a message to a window and wait for its handler."""
        ...

    def create_kill_on_close_job(self) -> int | None:
        """A job object whose processes are all ended when its last handle is closed.

        ``None`` if the job could not be made or configured.
        """
        ...

    def current_process_handle(self) -> int:
        """The pseudo handle of this process (usable with :meth:`assign_to_job`)."""
        ...

    def open_process(self, pid: int) -> int | None:
        """A handle of process ``pid`` that may be put in a job; ``None`` if it cannot be opened."""
        ...

    def assign_to_job(self, job: int, process: int) -> bool:
        """``AssignProcessToJobObject``."""
        ...


if sys.platform == "win32":  # pragma: win32-only
    import ctypes
    from ctypes import wintypes

    class _WindowClass(ctypes.Structure):
        """``WNDCLASSW``."""

        _fields_ = (
            ("style", wintypes.UINT),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HANDLE),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HANDLE),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        )

    class _IoCounters(ctypes.Structure):
        """``IO_COUNTERS``."""

        _fields_ = (
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        )

    class _BasicLimits(ctypes.Structure):
        """``JOBOBJECT_BASIC_LIMIT_INFORMATION``."""

        _fields_ = (
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _ExtendedLimits(ctypes.Structure):
        """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``."""

        _fields_ = (
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

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
            self._k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self._k32.CreateJobObjectW.restype = wintypes.HANDLE
            self._k32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            self._k32.SetInformationJobObject.restype = wintypes.BOOL
            self._k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self._k32.AssignProcessToJobObject.restype = wintypes.BOOL
            self._k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self._k32.OpenProcess.restype = wintypes.HANDLE
            self._k32.GetCurrentProcess.argtypes = []
            self._k32.GetCurrentProcess.restype = wintypes.HANDLE
            self._callbacks: dict[object, object] = {}  # keep ctypes callbacks alive
            self._u32: ctypes.WinDLL | None = None
            self._wndproc_type: Any = None
            self._classes: dict[int, str] = {}

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

        # ------------------------------------------------------ hidden message window

        def _user32(self) -> ctypes.WinDLL:
            if self._u32 is None:
                u32 = ctypes.WinDLL("user32", use_last_error=True)
                self._wndproc_type = ctypes.WINFUNCTYPE(
                    ctypes.c_ssize_t,
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                )
                u32.DefWindowProcW.argtypes = [
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                ]
                u32.DefWindowProcW.restype = ctypes.c_ssize_t
                u32.CreateWindowExW.argtypes = [
                    wintypes.DWORD,
                    wintypes.LPCWSTR,
                    wintypes.LPCWSTR,
                    wintypes.DWORD,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    wintypes.HWND,
                    wintypes.HMENU,
                    wintypes.HINSTANCE,
                    wintypes.LPVOID,
                ]
                u32.CreateWindowExW.restype = wintypes.HWND
                u32.DestroyWindow.argtypes = [wintypes.HWND]
                u32.DestroyWindow.restype = wintypes.BOOL
                u32.PostMessageW.argtypes = [
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                ]
                u32.PostMessageW.restype = wintypes.BOOL
                u32.SendMessageW.argtypes = [
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.WPARAM,
                    wintypes.LPARAM,
                ]
                u32.SendMessageW.restype = ctypes.c_ssize_t
                u32.GetMessageW.argtypes = [
                    ctypes.POINTER(wintypes.MSG),
                    wintypes.HWND,
                    wintypes.UINT,
                    wintypes.UINT,
                ]
                u32.GetMessageW.restype = ctypes.c_int
                u32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
                u32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
                u32.DispatchMessageW.restype = ctypes.c_ssize_t
                u32.PostQuitMessage.argtypes = [ctypes.c_int]
                u32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
                u32.UnregisterClassW.restype = wintypes.BOOL
                self._u32 = u32
            return self._u32

        def create_message_window(self, on_message: WindowHandler) -> int | None:
            u32 = self._user32()
            class_name = next_window_class_name()
            instance = self._k32.GetModuleHandleW(None)

            def procedure(hwnd: int, message: int, wparam: int, lparam: int) -> int:
                if message == WM_DESTROY:
                    u32.PostQuitMessage(0)
                    return 0
                try:
                    if on_message(message, wparam, lparam):
                        return TRUE
                except Exception:  # a failing handler must not take the window thread down
                    return 0
                return int(u32.DefWindowProcW(hwnd, message, wparam, lparam))

            callback = self._wndproc_type(procedure)
            self._callbacks[on_message] = callback  # keep the ctypes callback alive
            window_class = _WindowClass()
            window_class.lpfnWndProc = ctypes.cast(callback, ctypes.c_void_p)
            window_class.hInstance = instance
            window_class.lpszClassName = class_name
            register = u32.RegisterClassW
            register.argtypes = [ctypes.POINTER(_WindowClass)]
            register.restype = wintypes.ATOM
            if not register(ctypes.byref(window_class)):
                return None
            hwnd = u32.CreateWindowExW(
                0, class_name, class_name, 0, 0, 0, 0, 0, None, None, instance, None
            )
            if not hwnd:
                u32.UnregisterClassW(class_name, instance)
                return None
            self._classes[int(hwnd)] = class_name
            return int(hwnd)

        def run_message_loop(self) -> None:
            u32 = self._user32()
            message = wintypes.MSG()
            while u32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                u32.TranslateMessage(ctypes.byref(message))
                u32.DispatchMessageW(ctypes.byref(message))
            for hwnd, name in list(self._classes.items()):
                u32.UnregisterClassW(name, self._k32.GetModuleHandleW(None))
                self._classes.pop(hwnd, None)

        def post_close(self, hwnd: int) -> None:
            self._user32().PostMessageW(hwnd, WM_CLOSE, 0, 0)

        def send_message(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
            return int(self._user32().SendMessageW(hwnd, message, wparam, lparam))

        # ------------------------------------------------------------- job objects

        def create_kill_on_close_job(self) -> int | None:
            job = self._k32.CreateJobObjectW(None, None)
            if not job:
                return None
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = self._k32.SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            )
            if not ok:
                self._k32.CloseHandle(job)
                return None
            return int(job)

        def current_process_handle(self) -> int:
            return int(self._k32.GetCurrentProcess())

        def open_process(self, pid: int) -> int | None:
            handle = self._k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            return int(handle) if handle else None

        def assign_to_job(self, job: int, process: int) -> bool:
            return bool(self._k32.AssignProcessToJobObject(job, process))

    def load_win32() -> Win32:
        """The real Win32 binding."""
        return RealWin32()

else:

    def load_win32() -> Win32:
        """The real Win32 binding; unavailable on other platforms."""
        raise RuntimeError("the Windows API is only available on Windows")
