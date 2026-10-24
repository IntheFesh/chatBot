"""AES-256-GCM sealing and the key ring (R-STO-002, R-STO-003)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from twin.storage.crypto import (
    BLOB_VERSION,
    HEADER_BYTES,
    MIN_BLOB_BYTES,
    DecryptionError,
    KeyRing,
    KeyRingNotConfiguredError,
    SealedBlob,
    UnknownKeyError,
    blob_key_id,
    build_aad,
    generate_key,
    get_keyring,
    set_active_keyring,
    use_keyring,
)

AAD = build_aad("messages", ["01ABC"], "content")


def make_ring() -> KeyRing:
    return KeyRing({1: generate_key()}, 1)


def test_seal_open_round_trip_and_format() -> None:
    ring = make_ring()
    blob = ring.seal("你好, 世界".encode(), AAD)
    assert isinstance(blob, SealedBlob)
    assert blob[0] == BLOB_VERSION
    assert blob.key_id == 1 == blob_key_id(blob)
    assert len(blob) == HEADER_BYTES + 12 + len("你好, 世界".encode()) + 16
    assert ring.open(blob, AAD) == "你好, 世界".encode()


def test_every_seal_uses_a_fresh_nonce() -> None:
    ring = make_ring()
    blobs = {ring.seal(b"same", AAD) for _ in range(200)}
    assert len(blobs) == 200
    nonces = {blob[HEADER_BYTES : HEADER_BYTES + 12] for blob in blobs}
    assert len(nonces) == 200


@pytest.mark.parametrize("position", ["version", "key_id", "nonce", "body", "tag"])
def test_any_modification_of_the_blob_is_detected(position: str) -> None:
    ring = KeyRing({1: generate_key(), 2: generate_key()}, 1)
    blob = bytearray(ring.seal(b"secret payload", AAD))
    index = {
        "version": 0,
        "key_id": HEADER_BYTES - 1,
        "nonce": HEADER_BYTES + 3,
        "body": HEADER_BYTES + 12 + 2,
        "tag": len(blob) - 1,
    }[position]
    blob[index] ^= 0x01
    with pytest.raises(DecryptionError):
        ring.open(bytes(blob), AAD)


@pytest.mark.parametrize(
    "other",
    [
        build_aad("other_table", ["01ABC"], "content"),
        build_aad("messages", ["01ABD"], "content"),
        build_aad("messages", ["01ABC"], "raw"),
        build_aad("messages", ["01ABC", "x"], "content"),
        build_aad("messages", [], "content"),
    ],
)
def test_ciphertext_cannot_move_to_another_table_row_or_column(other: bytes) -> None:
    ring = make_ring()
    blob = ring.seal(b"hello", AAD)
    with pytest.raises(DecryptionError, match="authentication failed"):
        ring.open(blob, other)


def test_aad_is_unambiguous_for_concatenation_tricks() -> None:
    assert build_aad("ab", ["c"], "d") != build_aad("a", ["bc"], "d")
    assert build_aad("a", ["b", "c"], "d") != build_aad("a", ["b"], "cd")


def test_short_and_unsupported_blobs_are_rejected() -> None:
    ring = make_ring()
    with pytest.raises(DecryptionError, match="too short"):
        ring.open(b"\x01" * (MIN_BLOB_BYTES - 1), AAD)
    blob = bytearray(ring.seal(b"x", AAD))
    blob[0] = 99
    with pytest.raises(DecryptionError, match="version"):
        ring.open(bytes(blob), AAD)


def test_unknown_key_id_gives_actionable_error() -> None:
    other = KeyRing({7: generate_key()}, 7)
    blob = other.seal(b"x", AAD)
    ring = make_ring()
    with pytest.raises(UnknownKeyError, match="key id 7") as info:
        ring.open(blob, AAD)
    assert info.value.key_id == 7
    assert "Credential Manager" in str(info.value)


def test_old_ciphertext_still_opens_after_the_current_key_changes() -> None:
    """Retired keys keep decrypting old rows and old backups (R-STO-003)."""
    ring = make_ring()
    old_blob = ring.seal(b"written under key 1", AAD)
    ring.add_key(2, generate_key())
    ring.set_current(2)
    ring.retire(1)
    new_blob = ring.seal(b"written under key 2", AAD)
    assert blob_key_id(new_blob) == 2
    assert ring.open(old_blob, AAD) == b"written under key 1"
    assert ring.open(new_blob, AAD) == b"written under key 2"
    assert ring.retired_ids == frozenset({1})
    assert ring.key_ids == (1, 2)


def test_retired_and_missing_keys_cannot_be_used_for_sealing() -> None:
    ring = KeyRing({1: generate_key(), 2: generate_key()}, 2, retired=[1])
    with pytest.raises(ValueError, match="retired"):
        ring.seal(b"x", AAD, key_id=1)
    with pytest.raises(UnknownKeyError):
        ring.seal(b"x", AAD, key_id=9)
    with pytest.raises(ValueError, match="retired"):
        ring.set_current(1)
    with pytest.raises(UnknownKeyError):
        ring.key_bytes(9)


def test_key_ring_validation() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        KeyRing({1: b"short"}, 1)
    with pytest.raises(ValueError, match="not in the key ring"):
        KeyRing({1: generate_key()}, 2)
    with pytest.raises(ValueError, match="current key cannot be retired"):
        KeyRing({1: generate_key()}, 1, retired=[1])
    with pytest.raises(ValueError, match="invalid key id"):
        KeyRing({0: generate_key()}, 0)
    ring = make_ring()
    with pytest.raises(ValueError, match="current key cannot be retired"):
        ring.retire(1)
    with pytest.raises(UnknownKeyError):
        ring.retire(5)
    with pytest.raises(UnknownKeyError):
        ring.set_current(5)


def test_forget_removes_a_retired_key_only() -> None:
    ring = KeyRing({1: generate_key(), 2: generate_key()}, 2, retired=[1])
    ring.forget(1)
    assert ring.key_ids == (2,)
    with pytest.raises(ValueError, match="cannot be removed"):
        ring.forget(2)


def test_active_ring_management() -> None:
    previous = set_active_keyring(None)
    try:
        with pytest.raises(KeyRingNotConfiguredError):
            get_keyring()
        ring = make_ring()
        with use_keyring(ring):
            assert get_keyring() is ring
        with pytest.raises(KeyRingNotConfiguredError):
            get_keyring()
    finally:
        set_active_keyring(previous)


@given(
    plaintext=st.binary(max_size=2048),
    table=st.text(min_size=1, max_size=20),
    pk=st.lists(st.text(max_size=12), max_size=3),
    column=st.text(min_size=1, max_size=20),
)
def test_round_trip_property(plaintext: bytes, table: str, pk: list[str], column: str) -> None:
    ring = KeyRing({1: bytes(range(32))}, 1)
    aad = build_aad(table, pk, column)
    assert ring.open(ring.seal(plaintext, aad), aad) == plaintext
