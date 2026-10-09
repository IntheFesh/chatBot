"""Database key rotation (R-STO-003): ``twin secrets rotate-db-key``.

Procedure (EXCLUSIVE command: the application must not be running):

1. create a new key and make it current, or resume an unfinished rotation;
2. walk every table that has encrypted columns in primary-key order, in small
   write transactions; every value not yet sealed with the new key is decrypted
   with the key named in its header and sealed again with the new key (same
   ``(table, pk, column)`` associated data); progress is saved in the ``settings``
   table in the *same* transaction as each batch;
3. re-encrypt the media files the same way;
4. verify that nothing still uses an old key, then mark the old keys *retired*
   (kept in the credential store: old backups and any missed value still need them)
   and drop the progress record.

Correctness never depends on the saved progress: each value is inspected by its
own ``key_id``, so an interrupted run can simply be started again.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Column, MetaData, Table, select, tuple_, update

from twin.clock import Clock
from twin.storage.crypto import KeyRing, blob_key_id, build_aad
from twin.storage.db import Database
from twin.storage.keystore import KeyStore
from twin.storage.media import MediaStore
from twin.storage.models import Base
from twin.storage.settings_store import delete_setting, get_setting, put_setting
from twin.storage.types import EncryptedJSON, EncryptedText

PROGRESS_KEY = "rotation.progress"
DEFAULT_BATCH = 200

ProgressCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class EncryptedTable:
    table: Table
    pk: tuple[Column[Any], ...]
    columns: tuple[Column[Any], ...]


@dataclass
class RotationReport:
    new_key_id: int
    resumed: bool
    values_reencrypted: int = 0
    media_reencrypted: int = 0
    retired_key_ids: list[int] = field(default_factory=list)


class RotationError(RuntimeError):
    """The rotation could not be completed safely."""


def encrypted_tables(metadata: MetaData | None = None) -> list[EncryptedTable]:
    """Every table with at least one encrypted column, discovered from the schema."""
    found: list[EncryptedTable] = []
    for table in (metadata or Base.metadata).sorted_tables:
        columns = tuple(
            col for col in table.columns if isinstance(col.type, EncryptedText | EncryptedJSON)
        )
        if columns:
            found.append(EncryptedTable(table, tuple(table.primary_key.columns), columns))
    return found


def _scan_batch(
    session: Any, spec: EncryptedTable, last_pk: list[Any] | None, batch_size: int
) -> list[Any]:
    stmt = select(*spec.pk, *spec.columns).order_by(*spec.pk).limit(batch_size)
    if last_pk is not None:
        stmt = stmt.where(tuple_(*spec.pk) > tuple_(*last_pk))
    return list(session.execute(stmt).all())


def _reseal_batch(
    session: Any, spec: EncryptedTable, rows: list[Any], ring: KeyRing, target: int
) -> int:
    changed = 0
    pk_count = len(spec.pk)
    for row in rows:
        pk_values = tuple(row[:pk_count])
        updates: dict[str, Any] = {}
        for offset, column in enumerate(spec.columns):
            blob = row[pk_count + offset]
            if blob is None or blob_key_id(blob) == target:
                continue
            aad = build_aad(spec.table.name, pk_values, column.name)
            updates[column.key] = ring.seal(ring.open(blob, aad), aad)
        if not updates:
            continue
        stmt = update(spec.table).values(**updates)
        if "updated_at" in spec.table.c:  # rotation is not a user-visible modification
            stmt = stmt.values(updated_at=spec.table.c.updated_at)
        for pk_col, value in zip(spec.pk, pk_values, strict=True):
            stmt = stmt.where(pk_col == value)
        session.execute(stmt)
        changed += len(updates)
    return changed


def _remaining(db: Database, ring: KeyRing, target: int) -> int:
    count = 0
    with db.session() as session:
        for spec in encrypted_tables():
            for column in spec.columns:
                for (blob,) in session.execute(select(column).where(column.is_not(None))):
                    if blob_key_id(blob) != target:
                        count += 1
    return count


def rotate_db_key(
    db: Database,
    keystore: KeyStore,
    ring: KeyRing,
    clock: Clock,
    media: MediaStore | None = None,
    *,
    batch_size: int = DEFAULT_BATCH,
    on_batch: ProgressCallback | None = None,
) -> RotationReport:
    """Rotate the database encryption key; safe to call again after an interruption."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    with db.session() as session:
        saved = get_setting(session, PROGRESS_KEY)
    resumed = saved is not None
    if resumed:
        target = int(saved["target_key_id"])
        if ring.current_id != target:
            raise RotationError(
                f"an unfinished rotation targets key {target} but the current key is "
                f"{ring.current_id}; the key index was changed by hand"
            )
        progress: dict[str, Any] = saved
    else:
        target = keystore.add_key(ring)
        progress = {
            "target_key_id": target,
            "started_at": clock.now_utc().isoformat(),
            "tables": {},
            "media_done": False,
        }
        with db.transaction(bump_state=False) as session:
            put_setting(
                session, PROGRESS_KEY, progress, clock=clock, by="rotation", record_history=False
            )

    report = RotationReport(new_key_id=target, resumed=resumed)

    for spec in encrypted_tables():
        name = str(spec.table.name)
        last_pk: list[Any] | None = progress["tables"].get(name)
        while True:
            with db.transaction(bump_state=False) as session:
                rows = _scan_batch(session, spec, last_pk, batch_size)
                if not rows:
                    break
                report.values_reencrypted += _reseal_batch(session, spec, rows, ring, target)
                last_pk = list(rows[-1][: len(spec.pk)])
                progress["tables"][name] = last_pk
                put_setting(
                    session,
                    PROGRESS_KEY,
                    progress,
                    clock=clock,
                    by="rotation",
                    record_history=False,
                )
            if on_batch is not None:
                on_batch(name, len(rows))

    if media is not None and not progress.get("media_done"):
        report.media_reencrypted = media.rekey_all()
        progress["media_done"] = True
        with db.transaction(bump_state=False) as session:
            put_setting(
                session, PROGRESS_KEY, progress, clock=clock, by="rotation", record_history=False
            )

    leftover = _remaining(db, ring, target)
    if leftover:
        raise RotationError(f"{leftover} encrypted values are still on an old key; run it again")

    old = [key_id for key_id in ring.key_ids if key_id != target and key_id not in ring.retired_ids]
    keystore.retire(ring, old)
    report.retired_key_ids = old
    with db.transaction(bump_state=False) as session:
        delete_setting(session, PROGRESS_KEY)
    return report
