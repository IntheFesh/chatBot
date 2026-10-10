"""The local ``llama-server``: where it is, what it needs, how it is started (R-SRV-002).

``scripts/windows/get_llamacpp.ps1`` unpacks a pinned llama.cpp release into
``tools/llama.cpp/<tag>/``.  This module finds the newest such folder (or the executable named in
``style_model.serve.binary``), knows which files a working installation of each build has - the
CUDA builds need ``ggml-cuda.dll`` *and* the matching ``cudart``/``cublas`` runtime next to it, a
folder that has only one of them fails at the first request - and builds the command line.

The command line (:meth:`ServerSpec.argv`)::

    llama-server -m <gguf> --host 127.0.0.1 --port 8081 -c 4096 -ngl 999 --parallel 1
                 --chat-template chatml --no-webui

* it **listens on 127.0.0.1 only**; :func:`loopback_endpoint` refuses any other address of
  ``style_model.endpoint`` before a process is started;
* we use the raw ``/completion`` endpoint with the prompt that ``StylePromptBuilder`` rendered, so
  the server's chat template is never applied; ``--chat-template chatml`` only keeps the server
  from parsing a Jinja template at start-up and, should anything ever call a chat endpoint, makes
  it plain ChatML - the format of the training;
* the flags exist in llama.cpp b11177 (the pinned release) and in the newest one checked
  (b11539): checked in ``tools/server/README.md`` of both tags on 2026-10-10.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

LLAMA_DIR: Final = Path("tools") / "llama.cpp"
LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})
BIND_HOST: Final = "127.0.0.1"

WINDOWS_BASE_FILES: Final = (
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama.dll",
    "ggml.dll",
    "ggml-base.dll",
    "llama-common.dll",
)
CUDA_RUNTIME: Final = {
    "cuda-12.4": ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"),
    "cuda-13.4": ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"),
}
CUDA_BACKEND: Final = "ggml-cuda.dll"
CPU_BACKEND_PATTERN: Final = "ggml-cpu*.dll"
TAG_NUMBER: Final = re.compile(r"(\d+)")


class ServeError(RuntimeError):
    """The server cannot be started: no installation, no model file, a bad address."""


def server_name(platform: str | None = None) -> str:
    """The file name of the server executable on ``platform``."""
    return "llama-server.exe" if (platform or sys.platform) == "win32" else "llama-server"


def loopback_endpoint(endpoint: str) -> tuple[str, int]:
    """``(host, port)`` of an endpoint that is on this computer; anything else is refused."""
    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise ServeError(
            f"style_model.endpoint {endpoint!r} is not on this computer: the local llama-server "
            "listens on 127.0.0.1 only (use style_model.mode vllm_completion for a remote model)"
        )
    if parts.port is None:
        raise ServeError(f"style_model.endpoint {endpoint!r} has no port")
    return BIND_HOST, parts.port


def version_key(name: str) -> tuple[int, str]:
    """Sort key of a release folder (``b11177`` -> 11177): the highest number is the newest."""
    found = TAG_NUMBER.search(name)
    return (int(found.group(1)) if found else -1, name)


def detect_build(directory: Path) -> str:
    """Which build a folder holds: ``cpu``, ``cuda-12.4``, ``cuda-13.4`` or ``cuda`` (a CUDA
    backend whose runtime DLLs do not say which version they belong to - or are not there)."""
    if not (directory / CUDA_BACKEND).is_file():
        return "cpu"
    for kind, runtime in CUDA_RUNTIME.items():
        if any((directory / name).is_file() for name in runtime):
            return kind
    return "cuda"


def missing_files(directory: Path, kind: str) -> list[str]:
    """The files a working Windows installation of ``kind`` lacks (empty: complete)."""
    missing = [name for name in WINDOWS_BASE_FILES if not (directory / name).is_file()]
    if kind == "cpu":
        if not any(directory.glob(CPU_BACKEND_PATTERN)):
            missing.append(CPU_BACKEND_PATTERN)
    elif kind in CUDA_RUNTIME:
        missing.extend(name for name in CUDA_RUNTIME[kind] if not (directory / name).is_file())
    else:  # ggml-cuda.dll without any of the runtime DLLs: the cudart zip was not unpacked
        missing.append("cudart64_*.dll, cublas64_*.dll, cublasLt64_*.dll (the cudart zip)")
    return missing


@dataclass(frozen=True)
class LlamaInstall:
    """A llama.cpp folder."""

    binary: Path
    version: str
    kind: str  # cpu, cuda-12.4, cuda-13.4 or cuda
    missing: tuple[str, ...]  # empty off Windows, where the file list is not known

    @property
    def complete(self) -> bool:
        return not self.missing


def inspect_install(binary: Path, *, platform: str | None = None) -> LlamaInstall:
    """What the folder of ``binary`` holds and, on Windows, whether it is complete."""
    directory = binary.parent
    kind = detect_build(directory)
    lacking: list[str] = []
    if (platform or sys.platform) == "win32":
        lacking = missing_files(directory, kind)
    return LlamaInstall(binary, directory.name, kind, tuple(lacking))


def find_install(
    root: Path, configured: str | None = None, *, platform: str | None = None
) -> LlamaInstall | None:
    """``style_model.serve.binary`` if set, else the newest folder below ``tools/llama.cpp``."""
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        return inspect_install(path, platform=platform) if path.is_file() else None
    base = root / LLAMA_DIR
    if not base.is_dir():
        return None
    name = server_name(platform)
    candidates = sorted(
        (folder for folder in base.iterdir() if (folder / name).is_file()),
        key=lambda folder: version_key(folder.name),
        reverse=True,
    )
    return inspect_install(candidates[0] / name, platform=platform) if candidates else None


@dataclass(frozen=True)
class ServerSpec:
    """Everything that makes up one ``llama-server`` command line."""

    prefix: tuple[str, ...]  # the executable (and, for a script, its interpreter)
    model: Path
    port: int
    context: int = 4096
    gpu_layers: int = 999
    parallel: int = 1
    host: str = BIND_HOST

    def __post_init__(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise ServeError(f"llama-server must listen on this computer only, not {self.host!r}")

    def argv(self) -> list[str]:
        return [
            *self.prefix,
            "-m",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.context),
            "-ngl",
            str(self.gpu_layers),
            "--parallel",
            str(self.parallel),
            "--chat-template",
            "chatml",
            "--no-webui",
        ]
