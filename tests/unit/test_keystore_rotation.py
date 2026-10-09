"""Key store persistence and database key rotation (R-STO-003)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select, text

from tests.support.clock import ManualClock
from tests.support.credentials import MemoryCredentials
from twin.config.secrets import SecretStore
from twin.storage import migrate
from twin.storage.crypto import KeyRing, blob_key_id, set_active_keyring
from twin.storage.db import Database
from twin.storage.keystore import INDEX_NAME, KeyStore, KeyStoreError, key_secret_name
from twin.storage.media import MediaKind, MediaStore
from twin.storage.models import Alert, ChannelState, Job, Setting
from twin.storage.rotate import (
    PROGRESS_KEY,
    RotationError,
    encrypted_tables,
    rotate_db_key,
)
from twin.storage.settings_store import get_setting


@pytest.fixture
def credentials() -> SecretStore:
    return SecretStore(MemoryCredentials())


@pytest.fixture
def keystore(credentials: SecretStore) -> KeyStore:
    return KeyStore(credentials)


# ----------------------------------------------------------------- key store


def test_first_run_generates_a_256_bit_key_in_the_credential_store(
    keystore: KeyStore, credentials: SecretStore
) -> None:
    assert not keystore.exists()
    ring = keystore.load_or_create(allow_create=True)
    assert ring.current_id == 1 and ring.key_ids == (1,)
    assert len(bytes.fromhex(credentials.require(key_secret_name(1)))) == 32
    again = keystore.load_or_create(allow_create=False)
    assert again.key_bytes(1) == ring.key_bytes(1)


def test_a_missing_key_with_existing_data_is_never_silently_replaced(keystore: KeyStore) -> None:
    with pytest.raises(KeyStoreError, match="already contains data"):
        keystore.load_or_create(allow_create=False)
    assert not keystore.exists()


def test_create_initial_refuses_to_overwrite(keystore: KeyStore) -> None:
    keystore.create_initial()
    with pytest.raises(KeyStoreError, match="refusing to replace"):
        keystore.create_initial()


def test_add_key_and_retire_persist_across_loads(keystore: KeyStore) -> None:
    ring = keystore.create_initial()
    new_id = keystore.add_key(ring)
    assert new_id == 2 and ring.current_id == 2
    keystore.retire(ring, [1])
    reloaded = keystore.load()
    assert reloaded.current_id == 2
    assert reloaded.retired_ids == frozenset({1})
    assert reloaded.key_ids == (1, 2)


def test_missing_key_credential_gives_an_actionable_error(
    keystore: KeyStore, credentials: SecretStore
) -> None:
    keystore.create_initial()
    credentials.delete(key_secret_name(1))
    with pytest.raises(KeyStoreError, match=r"db-key-1.*missing"):
        keystore.load()


def test_corrupted_index_and_key_material_are_reported(
    keystore: KeyStore, credentials: SecretStore
) -> None:
    keystore.create_initial()
    credentials.set(INDEX_NAME, "{not json")
    with pytest.raises(KeyStoreError, match="corrupted"):
        keystore.load()
    credentials.set(INDEX_NAME, '{"v":1,"current":9,"ids":[1],"retired":[]}')
    with pytest.raises(KeyStoreError, match="not listed"):
        keystore.load()
    credentials.set(INDEX_NAME, '{"v":1,"current":1,"ids":[1],"retired":[]}')
    credentials.set(key_secret_name(1), "zz")
    with pytest.raises(KeyStoreError, match="not valid hex"):
        keystore.load()
    credentials.set(key_secret_name(1), "ab" * 16)
    with pytest.raises(KeyStoreError, match="wrong length"):
        keystore.load()


def test_delete_all_removes_every_key_credential(
    keystore: KeyStore, credentials: SecretStore
) -> None:
    ring = keystore.create_initial()
    keystore.add_key(ring)
    assert keystore.delete_all() == 2
    assert not keystore.exists()
    assert not credentials.exists(key_secret_name(1))
    assert keystore.delete_all() == 0


# ------------------------------------------------------------------ rotation


@pytest.fixture
def rotation_env(
    tmp_path: Path, keystore: KeyStore, clock: ManualClock
) -> Iterator[tuple[Database, KeyRing, MediaStore]]:
    ring = keystore.create_initial()
    set_active_keyring(ring)
    path = tmp_path / "rot.db"
    migrate.upgrade(path)
    database = Database(path, clock=clock)
    media = MediaStore(tmp_path / "media", tmp_path / "tmp", chunk_size=2048)
    yield database, ring, media
    database.dispose()


def populate(db: Database, media: MediaStore) -> dict[str, str]:
    with db.transaction() as session:
        for index in range(30):
            session.add(Job(type="demo", payload={"n": index, "text": f"正文{index}"}))
        session.add(Setting(key="alpha", value={"v": 1}, history=[{"old": 0}]))
        session.add(Setting(key="beta", value="bee"))
        session.add(ChannelState(key="cursor", value="abc"))
        session.add(Alert(category="c", title="t", detail={"k": "v"}))
        session.add(Alert(category="c2", title="t2"))
    stored = {
        "small": media.put(b"hello media", MediaKind.IMAGE).sha256,
        "big": media.put(bytes(range(256)) * 40, MediaKind.STICKER).sha256,
    }
    return stored


def all_key_ids(db: Database) -> set[int]:
    ids: set[int] = set()
    with db.session() as session:
        for spec in encrypted_tables():
            for column in spec.columns:
                for (blob,) in session.execute(select(column).where(column.is_not(None))):
                    ids.add(blob_key_id(blob))
    return ids


def test_encrypted_tables_are_discovered_from_the_schema() -> None:
    names = {spec.table.name: {c.name for c in spec.columns} for spec in encrypted_tables()}
    assert names == {
        "alerts": {"detail"},
        "channel_state": {"value"},
        "jobs": {"payload"},
        "settings": {"value", "history"},
    }


def test_rotation_reencrypts_everything_and_retires_the_old_key(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    hashes = populate(db, media)
    assert all_key_ids(db) == {1}
    with db.session() as session:
        updated_before = {r.key: r.updated_at for r in session.scalars(select(Setting))}

    report = rotate_db_key(db, keystore, ring, db.clock, media, batch_size=7)

    assert report.new_key_id == 2 and not report.resumed
    assert report.retired_key_ids == [1]
    # 30 job payloads + alpha value/history + beta value + 1 channel value + 1 alert detail
    assert report.values_reencrypted == 35
    assert report.media_reencrypted == 2
    assert all_key_ids(db) == {2}
    assert ring.current_id == 2 and ring.retired_ids == frozenset({1})
    assert keystore.load().retired_ids == frozenset({1})  # old key is kept, only marked retired
    with db.session() as session:
        assert get_setting(session, PROGRESS_KEY) is None
        jobs = session.scalars(select(Job).order_by(Job.created_at, Job.id)).all()
        assert sorted(j.payload["n"] for j in jobs) == list(range(30))
        assert session.get(Setting, "alpha").value == {"v": 1}  # type: ignore[union-attr]
        assert session.get(Setting, "alpha").history == [{"old": 0}]  # type: ignore[union-attr]
        assert session.get(ChannelState, "cursor").value == "abc"  # type: ignore[union-attr]
        updated_after = {r.key: r.updated_at for r in session.scalars(select(Setting))}
    # rotation is not a user-visible modification
    assert {k: v for k, v in updated_after.items() if k in updated_before} == updated_before
    for sha in hashes.values():
        assert media.key_id_of(sha) == 2
        assert media.verify(sha)


def test_old_ciphertext_and_backups_remain_readable_with_the_retired_key(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    populate(db, media)
    with db.session() as session:
        old_blob = bytes(
            session.execute(text("SELECT value FROM settings WHERE key = 'beta'")).scalar_one()
        )
    from twin.storage.crypto import build_aad

    rotate_db_key(db, keystore, ring, db.clock, media)
    assert blob_key_id(old_blob) == 1
    assert ring.open(old_blob, build_aad("settings", ["beta"], "value")) == b'"bee"'
    reloaded = keystore.load()  # a fresh process also keeps the retired key
    assert reloaded.open(old_blob, build_aad("settings", ["beta"], "value")) == b'"bee"'


def test_an_interrupted_rotation_can_be_resumed(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    populate(db, media)
    batches = 0

    def crash_after_three(table: str, rows: int) -> None:
        nonlocal batches
        batches += 1
        if batches == 3:
            raise KeyboardInterrupt("simulated crash")

    with pytest.raises(KeyboardInterrupt):
        rotate_db_key(db, keystore, ring, db.clock, media, batch_size=5, on_batch=crash_after_three)

    # half-way: new and old keys coexist, everything still reads, progress is saved
    assert all_key_ids(db) == {1, 2}
    with db.session() as session:
        progress = get_setting(session, PROGRESS_KEY)
        assert progress["target_key_id"] == 2
        assert progress["tables"]
        assert len(session.scalars(select(Job)).all()) == 30
    assert keystore.load().current_id == 2
    assert keystore.load().retired_ids == frozenset()

    fresh_ring = keystore.load()  # as after a restart
    set_active_keyring(fresh_ring)
    report = rotate_db_key(db, keystore, fresh_ring, db.clock, media, batch_size=5)

    assert report.resumed and report.new_key_id == 2
    assert all_key_ids(db) == {2}
    assert fresh_ring.retired_ids == frozenset({1})
    assert keystore.load().key_ids == (1, 2)
    with db.session() as session:
        assert get_setting(session, PROGRESS_KEY) is None


def test_rotation_can_be_repeated_and_chains_retired_keys(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    populate(db, media)
    rotate_db_key(db, keystore, ring, db.clock, media)
    second = rotate_db_key(db, keystore, ring, db.clock, media)
    assert second.new_key_id == 3
    assert second.retired_key_ids == [2]
    assert all_key_ids(db) == {3}
    assert ring.retired_ids == frozenset({1, 2})


def test_rotation_on_an_empty_database(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    report = rotate_db_key(db, keystore, ring, db.clock, media)
    assert report.values_reencrypted == 0 and report.media_reencrypted == 0


def test_resuming_with_a_changed_key_index_is_refused(
    rotation_env: tuple[Database, KeyRing, MediaStore], keystore: KeyStore
) -> None:
    db, ring, media = rotation_env
    populate(db, media)
    with pytest.raises(KeyboardInterrupt):
        rotate_db_key(
            db,
            keystore,
            ring,
            db.clock,
            media,
            batch_size=5,
            on_batch=lambda t, r: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    stale = KeyRing({1: ring.key_bytes(1), 2: ring.key_bytes(2)}, 1)
    with pytest.raises(RotationError, match="unfinished rotation"):
        rotate_db_key(db, keystore, stale, db.clock, media)


def test_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        rotate_db_key(None, None, None, None, batch_size=0)  # type: ignore[arg-type]
