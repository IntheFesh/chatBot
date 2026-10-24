"""UTF-8 console set-up (CLAUDE.md section 7)."""

from __future__ import annotations

import os
import sys

from twin.ops.winapi import Win32, load_win32

CP_UTF8 = 65001


def ensure_utf8(*, platform: str | None = None, win32: Win32 | None = None) -> None:
    """Force UTF-8 for this process, its children and the console.

    * ``PYTHONUTF8=1`` so child Python processes inherit UTF-8 mode;
    * ``stdout``/``stderr`` are reconfigured to UTF-8 (undecodable output is
      replaced, never raises);
    * on Windows the console output code page is set to 65001.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        (win32 or load_win32()).set_console_output_cp(CP_UTF8)
