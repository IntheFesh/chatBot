"""Writing a batch of normalised messages to the database (R-IMP-004, R-IMP-005).

The hot path of the import.  It works on SQLAlchemy Core with ``executemany`` and seals the
encrypted columns itself (``build_aad`` + the key ring), which is what the model
descriptors do per attribute; this is the same format, only without one ORM object per row.

Idempotence (R-IMP-005): a message is identified by the exported ``id``.

* unknown id                      -> inserted;
* known id, same content          -> counted as a duplicate;
* known id, different content     -> a *conflict*: the version from the export with the
  newer ``exportedAt`` wins (updated, or the stored one is kept when it is newer).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import orjson
from sqlalchemy import Table, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from twin.ingest.normalize import MediaRequest, NormalizedMessage
from twin.storage.chat_models import MediaAsset, Message, Sticker, StickerUse
from twin.storage.crypto import KeyRing, SealedBlob, build_aad

MESSAGES = cast(Table, Message.__table__)
STICKERS = cast(Table, Sticker.__table__)
STICKER_USES = cast(Table, StickerUse.__table__)
MEDIA_ASSETS = cast(Table, MediaAsset.__table__)
INT64_LIMIT = 2**63
IN_CHUNK = 900


class ConversationMismatchError(RuntimeError):
    """A message id already belongs to another conversation."""


def _clamp_ints(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clamp_ints(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clamp_ints(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= INT64_LIMIT:
        return str(value)
    return value


def dumps_json(value: Any) -> bytes:
    """Canonical JSON bytes (the codec of ``sealed_json``); oversize integers become text."""
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    except TypeError:
        return orjson.dumps(_clamp_ints(value), option=orjson.OPT_SORT_KEYS)


class RowSealer:
    """Seals column values of rows that are written with Core statements."""

    def __init__(self, ring: KeyRing) -> None:
        self._ring = ring

    def text(self, table: str, pk: str, column: str, value: str | None) -> SealedBlob | None:
        if value is None:
            return None
        return self._ring.seal(value.encode("utf-8"), build_aad(table, [pk], column))

    def json(self, table: str, pk: str, column: str, value: Any) -> SealedBlob | None:
        if value is None:
            return None
        return self._ring.seal(dumps_json(value), build_aad(table, [pk], column))

    def open_json(self, table: str, pk: str, column: str, blob: bytes) -> Any:
        return orjson.loads(self._ring.open(blob, build_aad(table, [pk], column)))


def asset_id(*parts: str) -> str:
    """Deterministic primary key of a media asset (32 hex characters)."""
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


@dataclass
class BatchResult:
    inserted: int = 0
    duplicates: int = 0
    conflict_updated: int = 0
    conflict_kept: int = 0
    stickers: int = 0
    assets: int = 0
    id_clashes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchContext:
    conversation_id: str
    export_id: str
    exported_at: datetime | None
    now: datetime
    sealer: RowSealer
    missing_message_ids: frozenset[str] = frozenset()


def _message_row(item: NormalizedMessage, ctx: BatchContext) -> dict[str, Any]:
    pk = item.id
    sealer = ctx.sealer
    return {
        "id": pk,
        "conversation_id": ctx.conversation_id,
        "create_time_utc": item.create_time,
        "sort_seq": item.sort_seq,
        "local_id": item.local_id,
        "server_id": item.server_id,
        "is_sent": item.is_sent,
        "kind": item.kind,
        "render_type": item.render_type[:48] if item.render_type else None,
        "text": sealer.text("messages", pk, "text", item.text),
        "raw": sealer.json("messages", pk, "raw", item.raw),
        "sticker_md5": item.sticker_md5,
        "media_sha256": None,
        "quote": sealer.json("messages", pk, "quote", item.quote),
        "call_status": item.call_status,
        "call_duration_s": item.call_duration_s,
        "voice_seconds": item.voice_seconds,
        "has_transcript": item.has_transcript,
        "source_export_id": ctx.export_id,
        "source_exported_at": ctx.exported_at,
        "created_at": ctx.now,
        "updated_at": ctx.now,
    }


def _chunks[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_newer(candidate: datetime | None, stored: datetime | None) -> bool:
    """True if an export stamped ``candidate`` should replace one stamped ``stored``."""
    if stored is None:
        return True
    if candidate is None:
        return False
    return candidate >= stored


def persist_messages(
    session: Session, items: list[NormalizedMessage], ctx: BatchContext
) -> BatchResult:
    """Insert, skip or update ``items`` (see the module docstring); returns the counts."""
    result = BatchResult()
    if not items:
        return result
    unique: dict[str, NormalizedMessage] = {}
    for item in items:
        if item.id in unique:
            result.duplicates += 1  # the same id twice in one file: the later one counts
        unique[item.id] = item
    ids = list(unique)

    known: dict[str, tuple[str, str, datetime | None]] = {}
    for chunk in _chunks(ids, IN_CHUNK):
        rows = session.execute(
            select(
                Message.id,
                Message.conversation_id,
                Message.source_export_id,
                Message.source_exported_at,
            ).where(Message.id.in_(chunk))
        )
        for row in rows:
            known[row.id] = (row.conversation_id, row.source_export_id, row.source_exported_at)

    to_insert: list[NormalizedMessage] = []
    to_compare: list[NormalizedMessage] = []
    for message_id, item in unique.items():
        stored = known.get(message_id)
        if stored is None:
            to_insert.append(item)
            continue
        conversation_id, export_id, exported_at = stored
        if conversation_id != ctx.conversation_id:
            result.id_clashes.append(message_id)
        elif export_id == ctx.export_id and exported_at == ctx.exported_at:
            result.duplicates += 1  # the very same export, read again
        else:
            to_compare.append(item)
    if result.id_clashes:
        raise ConversationMismatchError(
            f"{len(result.id_clashes)} message id(s) already belong to another conversation"
        )

    replaced = _resolve_conflicts(session, to_compare, ctx, result)
    insert_rows = [_message_row(item, ctx) for item in to_insert]
    if insert_rows:
        session.execute(MESSAGES.insert(), insert_rows)
    result.inserted = len(insert_rows)

    new_stickers: dict[str, dict[str, Any]] = {}
    sticker_uses: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    to_write = to_insert + replaced
    for item in to_write:
        _collect_stickers(item, ctx, new_stickers, sticker_uses)
        _collect_assets(item, ctx, assets)

    if new_stickers:
        _upsert_stickers(session, list(new_stickers.values()))
        result.stickers = len(new_stickers)
    if sticker_uses:
        _upsert_sticker_uses(session, sticker_uses)
    if assets:
        upsert_assets(session, assets)
        result.assets = len(assets)
    return result


def _resolve_conflicts(
    session: Session,
    candidates: list[NormalizedMessage],
    ctx: BatchContext,
    result: BatchResult,
) -> list[NormalizedMessage]:
    """Compare against the stored version; update the ones the newer export replaces."""
    replaced: list[NormalizedMessage] = []
    if not candidates:
        return replaced
    by_id = {item.id: item for item in candidates}
    for chunk in _chunks(list(by_id), IN_CHUNK):
        rows = session.execute(
            select(Message.id, Message.raw_ct, Message.source_exported_at).where(
                Message.id.in_(chunk)
            )
        )
        for row in rows:
            item = by_id[row.id]
            stored_raw = ctx.sealer.open_json("messages", row.id, "raw", row.raw_ct)
            if stored_raw == item.raw or stored_raw == orjson.loads(dumps_json(item.raw)):
                result.duplicates += 1
            elif _is_newer(ctx.exported_at, row.source_exported_at):
                result.conflict_updated += 1
                replaced.append(item)
            else:
                result.conflict_kept += 1
    for item in replaced:
        values = _message_row(item, ctx)
        for column in ("id", "created_at", "media_sha256"):
            values.pop(column)  # keep the identity, the creation time and any imported media
        session.execute(update(MESSAGES).where(MESSAGES.c.id == item.id).values(**values))
    return replaced


# -------------------------------------------------------------- stickers, assets


def _collect_stickers(
    item: NormalizedMessage,
    ctx: BatchContext,
    stickers: dict[str, dict[str, Any]],
    uses: list[dict[str, Any]],
) -> None:
    md5 = item.sticker_md5
    if md5 is None:
        return
    entry = stickers.get(md5)
    if entry is None:
        entry = {
            "md5": md5,
            "url": None,
            "source_path": None,
            "status": "pending",
            "reason": None,
            "sha256": None,
            "mime": None,
            "width": None,
            "height": None,
            "size_bytes": None,
            "attempts": 0,
            "last_attempt_at": None,
            "her_uses": 0,
            "user_uses": 0,
            "first_used_at": None,
            "last_used_at": None,
            "created_at": ctx.now,
            "updated_at": ctx.now,
        }
        stickers[md5] = entry
    if entry["url"] is None and item.sticker_url:
        entry["url"] = ctx.sealer.text("stickers", md5, "url", item.sticker_url)
    if entry["source_path"] is None:
        for request in item.media:
            if request.kind == "sticker" and request.path:
                entry["source_path"] = ctx.sealer.text("stickers", md5, "source_path", request.path)
                break
    uses.append(
        {
            "message_id": item.id,
            "sticker_md5": md5,
            "conversation_id": ctx.conversation_id,
            "by_her": not item.is_sent,
            "used_at": item.create_time,
            "created_at": ctx.now,
            "updated_at": ctx.now,
        }
    )


def _upsert_stickers(session: Session, rows: list[dict[str, Any]]) -> None:
    stmt = sqlite_insert(STICKERS)
    stmt = stmt.on_conflict_do_update(
        index_elements=["md5"],
        set_={
            "url": func.coalesce(STICKERS.c.url, stmt.excluded.url),
            "source_path": func.coalesce(STICKERS.c.source_path, stmt.excluded.source_path),
            "updated_at": stmt.excluded.updated_at,
        },
    )
    session.execute(stmt, rows)


def _upsert_sticker_uses(session: Session, rows: list[dict[str, Any]]) -> None:
    stmt = sqlite_insert(STICKER_USES)
    stmt = stmt.on_conflict_do_update(
        index_elements=["message_id"],
        set_={
            "sticker_md5": stmt.excluded.sticker_md5,
            "by_her": stmt.excluded.by_her,
            "used_at": stmt.excluded.used_at,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    session.execute(stmt, rows)


def asset_row(
    ctx: BatchContext,
    *,
    key: str,
    message_id: str | None,
    kind: str,
    request: MediaRequest | None,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    identifier = asset_id(ctx.conversation_id, key, kind)
    path = request.path if request is not None else None
    return {
        "id": identifier,
        "conversation_id": ctx.conversation_id,
        "message_id": message_id,
        "kind": kind,
        "status": status,
        "reason": reason,
        "orig_md5": (request.md5 or None) if request is not None else None,
        "file_id": (request.file_id or None) if request is not None else None,
        "source_path": ctx.sealer.text("media_assets", identifier, "source_path", path),
        "sha256": None,
        "mime": None,
        "width": None,
        "height": None,
        "size_bytes": None,
        "caption": None,
        "caption_at": None,
        "created_at": ctx.now,
        "updated_at": ctx.now,
    }


def _collect_assets(item: NormalizedMessage, ctx: BatchContext, rows: list[dict[str, Any]]) -> None:
    listed_missing = item.id in ctx.missing_message_ids
    wanted: set[str] = set()
    for request in item.media:
        if request.kind in ("skip", "sticker", "avatar"):
            continue
        key = f"{item.id}#{request.index}"
        wanted.add(request.kind)
        if request.path:
            rows.append(
                asset_row(
                    ctx,
                    key=key,
                    message_id=item.id,
                    kind=request.kind,
                    request=request,
                    status="pending",
                    reason=None,
                )
            )
        else:
            rows.append(
                asset_row(
                    ctx,
                    key=key,
                    message_id=item.id,
                    kind=request.kind,
                    request=request,
                    status="missing",
                    reason="listed_missing" if listed_missing else "no_path",
                )
            )
    # an image or video without any usable file is recorded as missing, so counts are complete
    expected = {"image": "image", "video": "video_cover"}.get(item.kind)
    if expected is not None and expected not in wanted:
        rows.append(
            asset_row(
                ctx,
                key=f"{item.id}#none",
                message_id=item.id,
                kind=expected,
                request=None,
                status="missing",
                reason="listed_missing" if listed_missing else "not_exported",
            )
        )


def upsert_assets(session: Session, rows: list[dict[str, Any]]) -> None:
    stmt = sqlite_insert(MEDIA_ASSETS)
    stmt = stmt.on_conflict_do_update(
        index_elements=["id"],
        set_={
            "status": stmt.excluded.status,
            "reason": stmt.excluded.reason,
            "orig_md5": stmt.excluded.orig_md5,
            "file_id": stmt.excluded.file_id,
            "source_path": stmt.excluded.source_path,
            "updated_at": stmt.excluded.updated_at,
        },
        where=MEDIA_ASSETS.c.status != "available",
    )
    session.execute(stmt, rows)
