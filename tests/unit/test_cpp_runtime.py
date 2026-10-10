"""The installed C++ runtime goes into the process before ``winrt`` brings an old one (D-621).

The logic runs everywhere with a recording loader; the Windows tests load the real DLLs in fresh
processes, because what they prove is an *order* of imports.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from twin.ops import cpp_runtime
from twin.ops.cpp_runtime import RUNTIME_DLLS, preload_system_runtime, system_directory

ROOT = Path(__file__).resolve().parents[2]


class RecordingLoader:
    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.paths: list[str] = []
        self._failing = failing

    def __call__(self, path: str) -> object:
        self.paths.append(path)
        if Path(path).name in self._failing:
            raise OSError("cannot load")
        return object()


def folder_with(tmp_path: Path, names: tuple[str, ...]) -> Path:
    folder = tmp_path / "System32"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


def test_the_runtime_is_loaded_from_the_system_folder_by_full_path_in_order(
    tmp_path: Path,
) -> None:
    folder = folder_with(tmp_path, RUNTIME_DLLS)
    loader = RecordingLoader()
    loaded = preload_system_runtime(platform="win32", directory=folder, loader=loader)
    assert loaded == RUNTIME_DLLS
    assert loader.paths == [str(folder / name) for name in RUNTIME_DLLS]
    assert all(Path(path).is_absolute() for path in loader.paths)  # a name could be answered by
    # the old copy that is already in the process; a full path cannot


def test_a_dll_that_is_missing_or_does_not_load_is_skipped(tmp_path: Path) -> None:
    folder = folder_with(tmp_path, ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll"))
    loader = RecordingLoader(failing=frozenset({"msvcp140_1.dll"}))
    loaded = preload_system_runtime(platform="win32", directory=folder, loader=loader)
    assert loaded == ("vcruntime140.dll", "msvcp140.dll")
    assert len(loader.paths) == 3  # the absent ones were never asked for


def test_nothing_is_loaded_off_windows_or_without_a_system_folder(tmp_path: Path) -> None:
    folder = folder_with(tmp_path, RUNTIME_DLLS)
    loader = RecordingLoader()
    assert preload_system_runtime(platform="linux", directory=folder, loader=loader) == ()
    assert preload_system_runtime(platform="darwin", directory=folder, loader=loader) == ()
    assert loader.paths == []
    assert system_directory({}) is None


def test_the_system_folder_comes_from_systemroot() -> None:
    assert system_directory({"SystemRoot": "C:/Windows"}) == Path("C:/Windows") / "System32"
    assert system_directory({"SYSTEMROOT": "D:/Win"}) == Path("D:/Win") / "System32"


def test_without_systemroot_the_default_folder_is_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("SYSTEMROOT", raising=False)
    loader = RecordingLoader()
    assert preload_system_runtime(platform="win32", loader=loader) == ()
    assert loader.paths == []


@pytest.mark.skipif(sys.platform == "win32", reason="the real loader works on Windows")
def test_the_real_loader_refuses_to_load_a_dll_off_windows() -> None:
    with pytest.raises(OSError, match="Windows only"):
        cpp_runtime._load_library("libc.so.6")


# ------------------------------------------------------------- the real Windows


@pytest.mark.windows
def test_the_real_runtime_of_this_computer_is_loaded() -> None:
    loaded = preload_system_runtime()
    assert "msvcp140.dll" in loaded and "vcruntime140.dll" in loaded


RESTORED_ORDER = textwrap.dedent(
    """
    import sys

    from twin.ops.notify import NotifyError, WindowsToastNotifier

    try:  # a computer without a notification area refuses; the library is imported either way
        WindowsToastNotifier().notify("t", "b")
    except NotifyError:
        pass
    assert "winrt" in sys.modules, "the toast library was not loaded: the test proves nothing"
    import torch  # the first import of torch of the process, after the toast library

    print("torch", torch.__version__)
    """
)


@pytest.mark.windows
def test_torch_still_loads_after_the_first_toast_of_the_process() -> None:
    """The failure of Windows shard 4/5: ``winrt``'s MSVCP140.dll 14.29 answered torch's request
    for ``msvcp140.dll`` and ``c10.dll`` died with ``WinError 1114``."""
    done = subprocess.run(
        [sys.executable, "-c", RESTORED_ORDER],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.startswith("torch ")
