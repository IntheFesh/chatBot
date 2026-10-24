"""The tokenizer the export counts tokens with: pinned by hash, found or downloaded (R-TRN-011)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import pytest

from tests.support.tiny_tokenizer import build_tiny_tokenizer
from twin.training import versions
from twin.training.tokenizer import (
    CACHE_NAME,
    TokenizerError,
    download_tokenizer,
    ensure_tokenizer,
    file_sha256,
    load_tokenizer,
    tokenizer_file,
)

# read at import: the tests' own fixture clears every TWIN_* variable while a test runs
REAL_TOKENIZER = os.environ.get("TWIN_QWEN_TOKENIZER")


@pytest.fixture
def tiny_file(tmp_path: Path) -> Path:
    path = tmp_path / "source" / "tokenizer.json"
    path.parent.mkdir()
    build_tiny_tokenizer().save(str(path))
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_pin_names_the_one_file_of_the_qwen3_family() -> None:
    assert len(versions.QWEN3_TOKENIZER_SHA256) == 64
    assert versions.QWEN3_TOKENIZER_BYTES > 10_000_000
    assert all(url.startswith("https://") for url in versions.QWEN3_TOKENIZER_URLS)
    assert "Qwen3-8B" in versions.QWEN3_TOKENIZER_URLS[0]  # at a fixed commit, not at "main"
    assert "/main/" not in versions.QWEN3_TOKENIZER_URLS[0]


def test_a_tokenizer_that_is_not_the_pinned_file_is_refused(tiny_file: Path) -> None:
    with pytest.raises(TokenizerError, match="not the pinned Qwen3 tokenizer"):
        load_tokenizer(tiny_file)
    loaded = load_tokenizer(tiny_file, expected_sha256=digest(tiny_file))
    assert loaded.sha256 == digest(tiny_file) and loaded.count("你好") >= 1
    assert loaded.decode(loaded.encode("你好 world")) == "你好 world"
    assert load_tokenizer(tiny_file, expected_sha256=None).count("a") == 1


def test_a_folder_stands_for_the_tokenizer_json_in_it(tiny_file: Path, tmp_path: Path) -> None:
    assert tokenizer_file(tiny_file.parent) == tiny_file
    with pytest.raises(TokenizerError, match=r"not a tokenizer\.json"):
        tokenizer_file(tmp_path / "nowhere")
    (tmp_path / "empty").mkdir()
    with pytest.raises(TokenizerError, match=r"not a tokenizer\.json"):
        tokenizer_file(tmp_path / "empty")


def test_a_damaged_file_is_reported_as_one_and_not_as_a_crash(tmp_path: Path) -> None:
    broken = tmp_path / "tokenizer.json"
    broken.write_text("{ not a tokenizer", encoding="utf-8")
    with pytest.raises(TokenizerError, match="cannot be read"):
        load_tokenizer(broken, expected_sha256=None)


def test_the_download_tries_the_next_address_and_checks_the_hash(
    tiny_file: Path, tmp_path: Path
) -> None:
    body = tiny_file.read_bytes()
    seen: list[str] = []

    def fetch(url: str) -> bytes:
        seen.append(url)
        if "first" in url:
            raise httpx.ConnectError("unreachable")
        if "wrong" in url:
            return b"something else"
        return body

    target = tmp_path / "cache"
    path = download_tokenizer(
        target,
        urls=(
            "https://first.example/t.json",
            "https://wrong.example/t.json",
            "https://ok.example/t",
        ),
        fetch=fetch,
        expected_sha256=digest(tiny_file),
    )
    assert path == target / "tokenizer.json" and path.read_bytes() == body
    assert len(seen) == 3 and not (target / "tokenizer.json.part").exists()


def test_when_no_address_gives_the_file_the_message_says_what_to_do(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        raise httpx.ConnectError("offline")

    with pytest.raises(TokenizerError, match="--tokenizer") as raised:
        download_tokenizer(tmp_path / "cache", fetch=fetch)
    assert "huggingface.co" in str(raised.value) and "modelscope.cn" in str(raised.value)
    assert not (tmp_path / "cache").exists()


def test_the_tokenizer_is_taken_from_the_path_then_the_cache_then_the_network(
    tiny_file: Path, tmp_path: Path
) -> None:
    pin = digest(tiny_file)
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return tiny_file.read_bytes()

    cache = tmp_path / "training" / "tokenizer"
    given = ensure_tokenizer(cache, explicit=tiny_file, fetch=fetch, expected_sha256=pin)
    assert given.sha256 == pin and not calls and not cache.exists()
    ensure_tokenizer(cache, fetch=fetch, expected_sha256=pin)
    assert len(calls) == 1 and (cache / CACHE_NAME / "tokenizer.json").is_file()
    ensure_tokenizer(cache, fetch=fetch, expected_sha256=pin)
    assert len(calls) == 1  # the cached copy is enough
    (cache / CACHE_NAME / "tokenizer.json").write_text("damaged", encoding="utf-8")
    ensure_tokenizer(cache, fetch=fetch, expected_sha256=pin)
    assert len(calls) == 2  # a damaged copy is fetched again


@pytest.mark.skipif(REAL_TOKENIZER is None, reason="TWIN_QWEN_TOKENIZER is not set")
def test_the_real_qwen3_file_is_the_pinned_one() -> None:
    assert REAL_TOKENIZER is not None
    path = tokenizer_file(Path(REAL_TOKENIZER))
    assert file_sha256(path) == versions.QWEN3_TOKENIZER_SHA256
    assert path.stat().st_size == versions.QWEN3_TOKENIZER_BYTES
    tokenizer = load_tokenizer(path)
    ids = tokenizer.encode("<|im_start|>assistant\n好的<|im_end|>\n")
    assert ids[0] == 151644 and 151645 in ids  # the two ChatML tokens of Qwen3
    assert tokenizer.decode(ids) == "<|im_start|>assistant\n好的<|im_end|>\n"
