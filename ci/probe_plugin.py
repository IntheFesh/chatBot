"""pytest plugin of the probe: when a test fails with WinError 1114, list the loaded DLLs."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


def loaded_modules() -> list[str]:
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel.GetCurrentProcess()
    modules = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    psapi.EnumProcessModulesEx(process, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3)
    count = needed.value // ctypes.sizeof(wintypes.HMODULE)
    buffer = ctypes.create_unicode_buffer(1024)
    names = []
    for index in range(count):
        psapi.GetModuleFileNameExW(process, modules[index], buffer, 1024)
        names.append(buffer.value)
    return names


def pytest_runtest_logreport(report: object) -> None:
    text = str(getattr(report, "longrepr", ""))
    if getattr(report, "failed", False) and "WinError 1114" in text:
        sys.stderr.write(f"\nPROBE: {getattr(report, 'nodeid', '?')} failed with WinError 1114\n")
        names = loaded_modules()
        sys.stderr.write(f"PROBE: {len(names)} modules loaded\n")
        for name in names:
            lowered = name.lower()
            if "system32" in lowered or "syswow64" in lowered:
                continue
            sys.stderr.write(f"PROBE dll {name}\n")
        heavy = [
            m
            for m in (
                "numpy",
                "scipy",
                "sklearn",
                "pyarrow",
                "lancedb",
                "lance",
                "pandas",
                "torch",
                "onnxruntime",
                "tokenizers",
                "cryptography",
                "PIL",
            )
            if m in sys.modules
        ]
        sys.stderr.write(f"PROBE python modules: {heavy}\n")
        sys.stderr.flush()
