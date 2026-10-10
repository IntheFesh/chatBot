"""Helpers for the tests that run ``llama_server_sim.py`` as a real child process (round 14)."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

from tests.support.tiny_tokenizer import build_tiny_tokenizer
from twin.serving.llamacpp import ServerSpec

SIM = Path(__file__).with_name("llama_server_sim.py")


def free_port() -> int:
    """A port nobody listens on right now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sim_prefix(*extra: str) -> tuple[str, ...]:
    """The program part of a command line that starts the simulated server."""
    return (sys.executable, "-I", str(SIM), *extra)


def make_model_file(directory: Path, name: str = "model-Q5_K_M.gguf") -> Path:
    """A file that stands in for a GGUF (the simulated server never reads it)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


def write_tokenizer(directory: Path, name: str = "tokenizer.json") -> Path:
    """The tiny Qwen-like tokenizer as a ``tokenizer.json`` the simulated server loads."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    build_tiny_tokenizer().save(str(path))
    return path


def spec_for(
    model: Path, port: int, *extra: str, context: int = 4096, gpu_layers: int = 999
) -> ServerSpec:
    """A server spec whose program is the simulated server."""
    return ServerSpec(
        prefix=sim_prefix(*extra), model=model, port=port, context=context, gpu_layers=gpu_layers
    )
