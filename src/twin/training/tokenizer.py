"""The tokenizer the training export counts tokens with (R-TRN-002, R-TRN-011).

LLaMA-Factory cuts a sample at ``cutoff_len`` (2048 tokens).  With ``mask_history`` the part that
is cut is the *end* of the last turn's prompt, which holds the assistant opener
(``<|im_start|>assistant\\n``); the exporter therefore has to make every sample fit by dropping the
oldest context itself, and for that it must count tokens exactly like the trainer does.  All four
GPU profiles train a Qwen3 model, and the 8B, 14B and 32B repositories ship the very same
``tokenizer.json`` (:data:`~twin.training.versions.QWEN3_TOKENIZER_SHA256`), so one file serves
every profile.  It is a file of about 11 MB; no model weights are needed.

:func:`ensure_tokenizer` finds the file in this order: the path the user passed (a
``tokenizer.json`` or a model folder that has one), the copy in the data directory, and finally
a download (Hugging Face at the pinned revision, then ModelScope), verified against the pinned
hash.  A file with another hash is refused, because token counts from another vocabulary would let
samples through that the trainer cuts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
from tokenizers import Tokenizer

from twin.training import versions

CACHE_NAME = "qwen3"
HASH_BLOCK = 1 << 20
DOWNLOAD_TIMEOUT_S = 120.0

Fetch = Callable[[str], bytes]


class TokenizerError(RuntimeError):
    """The tokenizer file is missing, damaged, or not the pinned one."""


class TextTokenizer(Protocol):
    """What the export needs from a tokenizer."""

    def encode(self, text: str) -> list[int]: ...

    def count(self, text: str) -> int: ...


class QwenTokenizer:
    """The Qwen3 tokenizer; ``encode`` is ``tokenizer.encode(text, add_special_tokens=False)``."""

    def __init__(self, tokenizer: Tokenizer, sha256: str) -> None:
        self._tokenizer = tokenizer
        self.sha256 = sha256

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_file(path: Path) -> Path:
    """``path`` itself if it is the file, else the ``tokenizer.json`` inside the folder."""
    if path.is_dir():
        path = path / versions.QWEN3_TOKENIZER_FILE
    if not path.is_file():
        raise TokenizerError(f"{path} is not a tokenizer.json (or a folder that has one)")
    return path


def load_tokenizer(
    path: Path, *, expected_sha256: str | None = versions.QWEN3_TOKENIZER_SHA256
) -> QwenTokenizer:
    """Load a ``tokenizer.json``; with ``expected_sha256`` it must be that exact file."""
    file = tokenizer_file(path)
    digest = file_sha256(file)
    if expected_sha256 is not None and digest != expected_sha256:
        raise TokenizerError(
            f"{file} is not the pinned Qwen3 tokenizer (sha256 {digest[:12]}..., expected "
            f"{expected_sha256[:12]}...); the export counts tokens with exactly that file"
        )
    try:
        return QwenTokenizer(Tokenizer.from_file(str(file)), digest)
    except Exception as exc:
        raise TokenizerError(
            f"{file} cannot be read as a tokenizer ({type(exc).__name__})"
        ) from exc


def http_fetch(url: str) -> bytes:
    """Download ``url`` (following redirects) and return the body."""
    with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def download_tokenizer(
    target: Path,
    *,
    urls: tuple[str, ...] = versions.QWEN3_TOKENIZER_URLS,
    fetch: Fetch = http_fetch,
    expected_sha256: str = versions.QWEN3_TOKENIZER_SHA256,
) -> Path:
    """Download the tokenizer into ``target`` (a folder), verified; tries each URL in turn."""
    problems: list[str] = []
    for url in urls:
        host = httpx.URL(url).host
        try:
            body = fetch(url)
        except (httpx.HTTPError, OSError) as exc:
            problems.append(f"{host}: {type(exc).__name__}")
            continue
        if hashlib.sha256(body).hexdigest() != expected_sha256:
            problems.append(f"{host}: the file is not the pinned tokenizer")
            continue
        target.mkdir(parents=True, exist_ok=True)
        destination = target / versions.QWEN3_TOKENIZER_FILE
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(body)
        partial.replace(destination)
        return destination
    raise TokenizerError(
        "cannot download the Qwen3 tokenizer ("
        + "; ".join(problems)
        + "); download tokenizer.json of Qwen/Qwen3-8B by hand and pass it with --tokenizer"
    )


def ensure_tokenizer(
    cache_dir: Path,
    *,
    explicit: Path | None = None,
    fetch: Fetch = http_fetch,
    expected_sha256: str = versions.QWEN3_TOKENIZER_SHA256,
) -> QwenTokenizer:
    """The pinned tokenizer: the given file, the cached copy, or a verified download."""
    if explicit is not None:
        return load_tokenizer(explicit, expected_sha256=expected_sha256)
    cached = cache_dir / CACHE_NAME / versions.QWEN3_TOKENIZER_FILE
    if cached.is_file():
        try:
            return load_tokenizer(cached, expected_sha256=expected_sha256)
        except TokenizerError:
            cached.unlink()  # damaged or replaced: fetch it again
    downloaded = download_tokenizer(
        cache_dir / CACHE_NAME, fetch=fetch, expected_sha256=expected_sha256
    )
    return load_tokenizer(downloaded, expected_sha256=expected_sha256)
