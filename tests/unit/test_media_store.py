"""Encrypted, streaming media store (R-STO-004)."""

from __future__ import annotations

import hashlib
import io
import os
import random
import struct
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from twin.storage.crypto import KeyRing, UnknownKeyError, generate_key, use_keyring
from twin.storage.media import (
    HEADER_BYTES,
    MediaIntegrityError,
    MediaKind,
    MediaNotFoundError,
    MediaStore,
)


@pytest.fixture
def ring() -> Iterator[KeyRing]:
    key_ring = KeyRing({1: generate_key()}, 1)
    with use_keyring(key_ring):
        yield key_ring


@pytest.fixture
def store(tmp_path: Path, ring: KeyRing) -> MediaStore:
    return MediaStore(tmp_path / "media", tmp_path / "tmp", chunk_size=4096)


def test_put_and_read_round_trip_with_sha256_file_name(store: MediaStore) -> None:
    data = "synthetic image bytes ".encode() * 1000
    stored = store.put(data, MediaKind.IMAGE)
    digest = hashlib.sha256(data).hexdigest()
    assert stored.sha256 == digest and stored.size == len(data) and stored.created
    assert store.path_for(digest) == store.root / f"{digest}.enc"
    assert store.path_for(digest).is_file()
    assert store.read_bytes(digest) == data
    assert store.exists(digest)
    assert list(store.iter_hashes()) == [digest]


def test_stored_file_contains_no_plaintext(store: MediaStore) -> None:
    marker = b"PLAINTEXT-MARKER-0123456789"
    stored = store.put(marker * 500, MediaKind.STICKER)
    assert marker not in store.path_for(stored.sha256).read_bytes()


def test_identical_content_is_deduplicated(store: MediaStore) -> None:
    first = store.put(b"same bytes", MediaKind.IMAGE)
    second = store.put(b"same bytes", MediaKind.IMAGE)
    assert first.sha256 == second.sha256
    assert first.created and not second.created
    assert len(list(store.root.glob("*.enc"))) == 1
    assert not list(store.root.glob(".put-*"))  # no leftover part files


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 8192, 12345])
def test_boundary_sizes_round_trip(store: MediaStore, size: int) -> None:
    data = random.Random(size).randbytes(size)
    stored = store.put(data, MediaKind.FILE)
    assert store.read_bytes(stored.sha256) == data
    assert store.verify(stored.sha256)


def test_put_accepts_streams_and_iterables(store: MediaStore) -> None:
    data = os.urandom(20_000)
    from_stream = store.put(io.BytesIO(data), MediaKind.VIDEO)
    pieces = (data[i : i + 777] for i in range(0, len(data), 777))
    from_iter = store.put(pieces, MediaKind.VIDEO)
    assert from_stream.sha256 == from_iter.sha256 == hashlib.sha256(data).hexdigest()


def test_kind_is_recorded_in_the_header(store: MediaStore) -> None:
    stored = store.put(b"avatar", MediaKind.AVATAR)
    with store.open(stored.sha256) as reader:
        assert reader.kind is MediaKind.AVATAR
        assert reader.size == 6 and reader.key_id == 1


def test_large_file_streams_without_holding_it_in_memory(tmp_path: Path, ring: KeyRing) -> None:
    """> 50 MB of synthetic data, 1 MB chunks, streamed both ways (task C.5)."""
    big = MediaStore(tmp_path / "media", tmp_path / "tmp")  # default 1 MB chunk size
    total = 56 * 1024 * 1024
    rng = random.Random(7)
    digest = hashlib.sha256()

    def generate() -> Iterator[bytes]:
        produced = 0
        while produced < total:
            block = rng.randbytes(1024 * 1024)
            digest.update(block)
            produced += len(block)
            yield block

    stored = big.put(generate(), MediaKind.VIDEO)
    assert stored.size == total and stored.sha256 == digest.hexdigest()
    on_disk = big.path_for(stored.sha256).stat().st_size
    assert on_disk == HEADER_BYTES + total + 16 * (total // (1024 * 1024))
    check = hashlib.sha256()
    count = 0
    with big.open(stored.sha256) as reader:
        for chunk in reader.chunks():
            check.update(chunk)
            count += len(chunk)
    assert count == total and check.hexdigest() == stored.sha256
    assert big.verify(stored.sha256)


def test_partial_reads_return_the_requested_amounts(store: MediaStore) -> None:
    data = bytes(range(256)) * 100
    stored = store.put(data, MediaKind.OTHER)
    with store.open(stored.sha256) as reader:
        assert reader.read(10) == data[:10]
        assert reader.read(5000) == data[10:5010]
        assert reader.read(0) == b""
        assert reader.read() == data[5010:]
        assert reader.read(10) == b""


def test_missing_media_is_reported(store: MediaStore) -> None:
    with pytest.raises(MediaNotFoundError):
        store.open("0" * 64)
    assert not store.verify("0" * 64)
    assert store.delete("0" * 64) is False


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "A" * 64, "g" * 64, "0" * 63])
def test_invalid_hashes_cannot_escape_the_store(store: MediaStore, bad: str) -> None:
    with pytest.raises(ValueError, match="sha256"):
        store.path_for(bad)


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    "where",
    ["body", "tag", "header_len", "header_sha", "header_prefix", "header_key"],
)
def test_tampering_is_detected(store: MediaStore, where: str) -> None:
    stored = store.put(os.urandom(10_000), MediaKind.IMAGE)
    path = store.path_for(stored.sha256)
    offset = {
        "body": HEADER_BYTES + 100,
        "tag": path.stat().st_size - 1,
        "header_len": 4 + 1 + 4 + 4 + 8 + 7,
        "header_sha": 4 + 1 + 4 + 4 + 8 + 8 + 3,
        "header_prefix": 4 + 1 + 4 + 4 + 2,
        "header_key": 4 + 1 + 3,
    }[where]
    flip(path, offset)
    assert not store.verify(stored.sha256)
    with pytest.raises((MediaIntegrityError, UnknownKeyError)), store.open(stored.sha256) as r:
        r.read()


def test_truncation_and_trailing_garbage_are_detected(store: MediaStore) -> None:
    stored = store.put(os.urandom(10_000), MediaKind.IMAGE)
    path = store.path_for(stored.sha256)
    original = path.read_bytes()
    path.write_bytes(original[:-100])
    assert not store.verify(stored.sha256)
    path.write_bytes(original + b"x")
    assert not store.verify(stored.sha256)
    path.write_bytes(original[: HEADER_BYTES - 5])
    assert not store.verify(stored.sha256)
    path.write_bytes(original)
    assert store.verify(stored.sha256)


def test_chunk_reordering_is_detected(store: MediaStore) -> None:
    stored = store.put(os.urandom(4096 * 3), MediaKind.IMAGE)
    path = store.path_for(stored.sha256)
    data = path.read_bytes()
    chunk = 4096 + 16
    first = data[HEADER_BYTES : HEADER_BYTES + chunk]
    second = data[HEADER_BYTES + chunk : HEADER_BYTES + 2 * chunk]
    swapped = data[:HEADER_BYTES] + second + first + data[HEADER_BYTES + 2 * chunk :]
    path.write_bytes(swapped)
    assert not store.verify(stored.sha256)


def test_a_file_stored_under_another_name_is_rejected(store: MediaStore) -> None:
    first = store.put(b"first content", MediaKind.IMAGE)
    second = store.put(b"second content", MediaKind.IMAGE)
    store.path_for(first.sha256).write_bytes(store.path_for(second.sha256).read_bytes())
    assert not store.verify(first.sha256)
    with pytest.raises(MediaIntegrityError, match="file name"):
        store.open(first.sha256)


def test_wrong_header_magic_is_not_a_media_file(store: MediaStore) -> None:
    stored = store.put(b"data", MediaKind.IMAGE)
    path = store.path_for(stored.sha256)
    path.write_bytes(b"XXXX" + path.read_bytes()[4:])
    with pytest.raises(MediaIntegrityError, match="supported"):
        store.open(stored.sha256)
    path.write_bytes(struct.pack(">4sB", b"TWM1", 1) + b"\x00" * 10)
    with pytest.raises(MediaIntegrityError):
        store.open(stored.sha256)


def test_temp_file_decrypts_then_wipes_and_deletes(store: MediaStore) -> None:
    data = b"private photo bytes" * 300
    stored = store.put(data, MediaKind.IMAGE)
    with store.temp_file(stored.sha256, ".jpg") as path:
        assert path.suffix == ".jpg"
        assert path.read_bytes() == data
        assert path.parent == store.tmp_dir
        seen = path
    assert not seen.exists()
    assert not list(store.tmp_dir.iterdir())
    if sys.platform != "win32":
        assert (store.tmp_dir.stat().st_mode & 0o777) == 0o700


def test_temp_file_is_removed_even_when_the_body_raises(store: MediaStore) -> None:
    stored = store.put(b"x" * 100, MediaKind.IMAGE)
    with pytest.raises(RuntimeError), store.temp_file(stored.sha256) as path:
        assert path.exists()
        raise RuntimeError("boom")
    assert not list(store.tmp_dir.iterdir())


def test_wipe_overwrites_content_before_unlinking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from twin.storage import media

    target = tmp_path / "secret.bin"
    target.write_bytes(b"A" * 200_000)
    observed: list[bytes] = []
    real_unlink = Path.unlink

    def spy(self: Path, missing_ok: bool = False) -> None:
        if self == target:
            observed.append(target.read_bytes())
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", spy)
    media._wipe(target, 200_000)
    assert observed == [b"\x00" * 200_000]
    assert not target.exists()


def test_a_failed_put_leaves_no_part_file(store: MediaStore) -> None:
    def broken() -> Iterator[bytes]:
        yield b"x" * 10_000
        raise OSError("disk went away")

    with pytest.raises(OSError, match="disk went away"):
        store.put(broken(), MediaKind.FILE)
    assert not list(store.root.glob(".put-*"))
    assert not list(store.root.glob("*.enc"))


def test_delete_and_listing(store: MediaStore) -> None:
    a = store.put(b"a", MediaKind.IMAGE).sha256
    b = store.put(b"b", MediaKind.IMAGE).sha256
    assert sorted(store.iter_hashes()) == sorted([a, b])
    assert store.delete(a) is True
    assert list(store.iter_hashes()) == [b]


def test_rekey_moves_files_to_the_current_key_keeping_name_and_content(
    tmp_path: Path, ring: KeyRing
) -> None:
    store = MediaStore(tmp_path / "media", tmp_path / "tmp", chunk_size=4096)
    payloads = {store.put(os.urandom(10_000 + i), MediaKind.IMAGE).sha256 for i in range(3)}
    ring.add_key(2, generate_key())
    ring.set_current(2)
    progress: list[tuple[int, int]] = []
    assert store.rekey_all(lambda done, total: progress.append((done, total))) == 3
    assert progress[-1] == (3, 3)
    assert store.rekey_all() == 0  # idempotent
    for sha in payloads:
        assert store.key_id_of(sha) == 2
        assert store.verify(sha)


def test_chunk_size_must_be_reasonable(tmp_path: Path, ring: KeyRing) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        MediaStore(tmp_path / "m", tmp_path / "t", chunk_size=10)
