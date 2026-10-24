"""Load the C++ runtime Windows has installed before a package brings an older one (D-621).

``winrt`` (the Windows Runtime binding under ``windows-toasts``) ships its own, plainly named
``MSVCP140.dll`` - version 14.29, from 2021.  Windows resolves a DLL by *name* and reuses a module
of that name that is already in the process, so once a toast has been shown, everything that is
loaded afterwards and needs ``msvcp140.dll`` gets that old copy.  ``torch`` is such a thing: its
``c10.dll`` then fails to initialise (``WinError 1114``) and the embedding model cannot be loaded
for the rest of the process - while ``import torch`` first and the toast afterwards works.  The
first toast of a session (an alert at start-up) decides which way round it is.

Loading the runtime from the system folder by its full path puts the installed, current copy (the
one every Python 3.12 and every ``torch`` needs the Visual C++ Redistributable for) in the process
first; a later request for ``MSVCP140.dll`` - ``winrt``'s included - is then answered by it.  The
other packages that bundle a copy (``numpy``, ``pyarrow``) give it a hashed name and are not
affected.  Nothing is done off Windows.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

from twin.ops.logging import get_logger

log = get_logger("twin.cpp_runtime")

RUNTIME_DLLS: Final = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
)

Loader = Callable[[str], object]


def system_directory(environ: dict[str, str] | None = None) -> Path | None:
    """``%SystemRoot%\\System32``; ``None`` if the variable is not set."""
    env = os.environ if environ is None else environ
    root = env.get("SystemRoot") or env.get("SYSTEMROOT")
    return Path(root) / "System32" if root else None


def _load_library(path: str) -> object:
    if sys.platform != "win32":
        raise OSError("DLLs are loaded on Windows only")
    return ctypes.WinDLL(path)  # pragma: win32-only


def preload_system_runtime(
    *,
    platform: str | None = None,
    directory: Path | None = None,
    loader: Loader | None = None,
) -> tuple[str, ...]:
    """Load the installed C++ runtime DLLs by full path; returns the names that were loaded.

    A DLL that is not there, or does not load, is skipped: a computer without the Visual C++
    Redistributable cannot run ``torch`` anyway, and a toast must still be tried.
    """
    if (sys.platform if platform is None else platform) != "win32":
        return ()
    folder = system_directory() if directory is None else directory
    if folder is None:
        return ()
    load = _load_library if loader is None else loader
    loaded: list[str] = []
    for name in RUNTIME_DLLS:
        path = folder / name
        if not path.is_file():
            continue
        try:
            load(str(path))
        except OSError:
            log.warning("cpp_runtime_not_loaded", dll=name)
            continue
        loaded.append(name)
    return tuple(loaded)
