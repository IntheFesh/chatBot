"""A registered model, a simulated llama-server and services with a real clock (round 14).

The serving component, the evaluation servers and ``twin model serve`` start real child processes
and wait for real HTTP answers, so they run on the real clock; everything else of ``services``
(database, settings, secrets) is the usual test container.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.support.llama_sim import free_port, sim_prefix, write_tokenizer
from tests.support.style_models import register_model
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.clock import SystemClock
from twin.config.runtime import BACKEND_ACTIVE, BackendName
from twin.services import Services
from twin.serving.llamacpp import WINDOWS_BASE_FILES
from twin.serving.runtime import LocalProgram, TokenizerSource
from twin.serving.server import ServerTimings
from twin.storage.training_models import ModelRegistryEntry

MODEL_SIZE = 1000
FAST = ServerTimings(
    start_timeout_s=30.0,
    backoff_start_s=0.05,
    backoff_max_s=0.2,
    stable_after_s=60.0,
    health_interval_s=0.1,
    unhealthy_limit=3,
    stop_grace_s=5.0,
)


@dataclass
class ServingWorld:
    """Services on the real clock, an active model and the program that serves it."""

    services: Services
    model_id: str
    model_file: Path
    tokenizer_json: Path
    control: Path
    port: int
    tmp: Path

    def program(self, *options: str, tokenizer: bool = True) -> LocalProgram:
        """The simulated llama-server (``options`` are its ``--sim-*`` options)."""
        extra = ["--sim-control", str(self.control), *options]
        if tokenizer:
            extra += ["--sim-tokenizer", str(self.tokenizer_json)]
        return LocalProgram(sim_prefix(*extra), None, "simulated llama-server")

    def tokenizers(self) -> TokenizerSource:
        """The tiny tokenizer the simulated server tokenizes with."""
        return TokenizerSource(self.services, loader=tiny_qwen_tokenizer)


def install_wrapper(directory: Path, *options: str, tokenizer_json: Path | None = None) -> Path:
    """An executable ``llama-server`` that runs the simulated one (POSIX): the commands find it
    through ``style_model.serve.binary`` like the real program."""
    directory.mkdir(parents=True, exist_ok=True)
    extra = list(options)
    if tokenizer_json is not None:
        extra += ["--sim-tokenizer", str(tokenizer_json)]
    command = " ".join(shlex.quote(part) for part in sim_prefix(*extra))
    path = directory / "llama-server"
    path.write_text(f'#!/bin/sh\nexec {command} "$@"\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    if sys.platform == "win32":
        # The commands check the folder of the program for the files of a real Windows
        # installation before they look at anything else.  The script cannot run there (the
        # tests that start it are POSIX-only), but the ones that stop earlier, at the model file,
        # must get past that check as they do on Linux.
        for name in (*WINDOWS_BASE_FILES, "ggml-cpu-x64.dll"):
            (directory / name).write_bytes(b"")
    return path


def real_clock(services: Services) -> Services:
    """The same container with the real clock (the processes it starts run in real time)."""
    return dataclasses.replace(services, clock=SystemClock())


def build_serving_world(
    services: Services,
    tmp_path: Path,
    *,
    run_id: str = "r-1",
    quant: str = "Q5_K_M",
    active: bool = True,
    gate_passed: bool | None = True,
    backend: str = "style",
    kind: str = "gguf",
    tokenize_ok: bool | None = None,
    template_version: str | None = None,
    real_sha256: bool = False,
) -> ServingWorld:
    """An active model, its (dummy) file of the registered size, a free port, fast timings."""
    real = real_clock(services)
    port = free_port()
    config = real.settings.style_model
    config.endpoint = f"http://127.0.0.1:{port}"
    config.serve.start_timeout_s = 30.0
    config.serve.backoff_start_s = 0.05
    config.serve.backoff_max_s = 0.2
    config.serve.eval_port = free_port()
    real.settings.backend.health_check_s = 0.1
    options: dict[str, object] = {}
    if template_version is not None:
        options["template_version"] = template_version
    model_id = register_model(
        real,
        run_id=run_id,
        quant=quant,
        kind=kind,
        active=active,
        gate_passed=gate_passed,
        **options,  # type: ignore[arg-type]
    )
    file = real.paths.models_dir / run_id / f"{quant}.gguf"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"G" * MODEL_SIZE)
    if real_sha256:  # the file is checked against the registry the way the commands do it
        with real.db.transaction(bump_state=False) as session:
            row = session.get(ModelRegistryEntry, model_id)
            if row is not None:
                row.sha256 = hashlib.sha256(b"G" * MODEL_SIZE).hexdigest()
    real.runtime.set(BACKEND_ACTIVE, backend, by="test")  # type: ignore[arg-type]
    if tokenize_ok is not None:
        from twin.training.registry import record_eval

        record_eval(
            real.db,
            model_id,
            {"tokenize_check": {"ok": tokenize_ok, "model_sha256": "0" * 64, "at": "x"}},
        )
    tmp = tmp_path / "serving"
    tmp.mkdir(parents=True, exist_ok=True)
    return ServingWorld(real, model_id, file, write_tokenizer(tmp), tmp / "control", port, tmp)


__all__ = [
    "FAST",
    "BackendName",
    "ServingWorld",
    "build_serving_world",
    "install_wrapper",
    "real_clock",
]
