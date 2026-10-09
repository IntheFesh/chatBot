"""A recording stand-in for the Windows API (:class:`twin.ops.winapi.Win32`)."""

from __future__ import annotations

from twin.ops.winapi import ERROR_ALREADY_EXISTS, CtrlHandler


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
