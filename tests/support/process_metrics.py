"""What the operating system says about this process: resident memory and processor time.

Used by the long-run script (``scripts/soak.py``), the runtime measurements
(``scripts/bench_runtime.py``) and the tests of the non-functional requirements (R-NFR-001..003).
One implementation for Linux (``/proc``), macOS (``getrusage``) and Windows (``psapi``), none of
which needs a third-party package.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _windows_counters() -> tuple[int, int]:
    """``(working set, peak working set)`` of this process in bytes (Windows only)."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined]
    # without these declarations ctypes passes the pseudo handle (-1) as a 32-bit int
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    handle = kernel.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    return 0, 0


def _proc_status(field: str) -> int | None:
    """A ``kB`` field of ``/proc/self/status`` in bytes (Linux), else ``None``."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def peak_rss_bytes() -> int:
    """The largest resident set this process has had, in bytes (0 if the system cannot say)."""
    if sys.platform == "win32":
        return _windows_counters()[1]
    peak = _proc_status("VmHWM") if sys.platform.startswith("linux") else None
    if peak is not None:
        return peak
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def current_rss_bytes() -> int:
    """The resident set of this process right now, in bytes (0 if the system cannot say)."""
    if sys.platform == "win32":
        return _windows_counters()[0]
    current = _proc_status("VmRSS") if sys.platform.startswith("linux") else None
    if current is not None:
        return current
    return peak_rss_bytes()  # macOS: only the high-water mark is available without a package


def rss_breakdown_bytes() -> dict[str, int]:
    """``RssAnon`` / ``RssFile`` / ``RssShmem`` of this process in bytes (Linux; else empty).

    Anonymous memory is the heap (Python objects, native allocations); file memory is mapped
    files (libraries, the database and vector files) that the system may drop at any time.
    """
    found: dict[str, int] = {}
    for name in ("RssAnon", "RssFile", "RssShmem"):
        value = _proc_status(name)
        if value is not None:
            found[name] = value
    return found


def cpu_seconds() -> float:
    """User plus system processor time of this process (all threads) so far."""
    return time.process_time()


def megabytes(value: float) -> float:
    return value / (1024.0 * 1024.0)
