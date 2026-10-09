"""Importing the media files of the target conversation (R-IMP-008).

Runs after the messages are in: for every ``media_assets`` row that is still ``pending`` the
file named by ``offlineMedia[].path`` is streamed through an integrity / MD5 check into the
encrypted media store.  A file that does not exist (or is listed in ``report.json`` as
missing) makes the row ``missing``.  Stickers use local files from ``media/emojis`` when
there are any; the rest wait for the download job.  Only the avatars of the two people in
the target conversation are imported.

Each step is idempotent and resumable: a row leaves ``pending`` only when its outcome is
saved, so a run that is interrupted in this phase simply picks up what is left.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

from PIL import Image, UnidentifiedImageError
from sqlalchemy import ColumnElement, ScalarSelect, Table, func, select, update

from twin.clock import Clock
from twin.ingest.integrity import (
    FileVerdict,
    IntegrityEntry,
    IntegrityIndex,
    StreamHasher,
    check_stream,
    normalize_key,
    stream_algorithms,
    verify_file,
)
from twin.ingest.layout import EMOJI_MEDIA_DIR, ExportLayout
from twin.ingest.normalize import MediaRequest
from twin.ingest.paths import long_path, nfc, resolve_in_root, split_relative
from twin.ingest.persist import (
    MESSAGES,
    BatchContext,
    RowSealer,
    asset_id,
    asset_row,
    upsert_assets,
)
from twin.ingest.stats import RunStats
from twin.llm.errors import UnsupportedImageError
from twin.llm.images import sniff_mime
from twin.stickers.library import MAX_STICKER_BYTES, store_sticker_file
from twin.storage.chat_models import Conversation, MediaAsset, Sticker, StickerUse
from twin.storage.db import Database
from twin.storage.media import MediaKind, MediaStore

ASSET_GROUP = 50
STICKER_GROUP = 100
_STORE_KINDS = {
    "image": MediaKind.IMAGE,
    "video_cover": MediaKind.IMAGE,
    "voice": MediaKind.VOICE,
    "avatar": MediaKind.AVATAR,
}
_IMAGE_KINDS = frozenset({"image", "video_cover", "avatar"})


@dataclass(frozen=True)
class AssetWork:
    id: str
    kind: str
    message_id: str | None
    path: str | None
    orig_md5: str | None


@dataclass(frozen=True)
class AssetOutcome:
    work: AssetWork
    status: str
    reason: str | None = None
    sha256: str | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    size: int | None = None


class MediaImporter:
    """Imports the media of one conversation from an export."""

    def __init__(
        self,
        *,
        db: Database,
        media: MediaStore,
        clock: Clock,
        layout: ExportLayout,
        integrity: IntegrityIndex,
        stats: RunStats,
        conversation_id: str,
        sealer: RowSealer,
        stop: threading.Event,
        on_progress: Callable[[], None] | None = None,
    ) -> None:
        self._db = db
        self._media = media
        self._clock = clock
        self._layout = layout
        self._integrity = integrity
        self._stats = stats
        self._conversation_id = conversation_id
        self._sealer = sealer
        self._stop = stop
        self._on_progress = on_progress

    # ------------------------------------------------------------------ avatars

    def plan_avatars(self, her_path: str | None, user_path: str | None) -> None:
        """Add pending rows for the two avatars (existing rows are left alone)."""
        ctx = BatchContext(
            conversation_id=self._conversation_id,
            export_id="",
            exported_at=None,
            now=self._clock.now_utc(),
            sealer=self._sealer,
        )
        rows = []
        for role, path in (("her", her_path), ("user", user_path)):
            if not path:
                continue
            request = MediaRequest("avatar", path, None, None, 0)
            rows.append(
                asset_row(
                    ctx,
                    key=f"avatar:{role}",
                    message_id=None,
                    kind="avatar",
                    request=request,
                    status="pending",
                    reason=None,
                )
            )
        if rows:
            with self._db.transaction(bump_state=False) as session:
                upsert_assets(session, rows)

    def finish_avatars(self) -> None:
        """Record the stored avatar files on the conversation (her side first)."""
        with self._db.transaction(bump_state=False) as session:
            conversation = session.get(Conversation, self._conversation_id)
            if conversation is None:
                return
            for role in ("her", "user"):
                row = session.get(
                    MediaAsset, asset_id(self._conversation_id, f"avatar:{role}", "avatar")
                )
                if row is not None and row.status == "available" and row.sha256:
                    if role == "her":
                        conversation.her_avatar_sha256 = row.sha256
                    else:
                        conversation.user_avatar_sha256 = row.sha256

    # ------------------------------------------------------------------- assets

    def run_assets(self) -> bool:
        """Import every pending asset; returns ``False`` if stopped before the end."""
        while True:
            work = self._pending_assets(ASSET_GROUP)
            if not work:
                return True
            outcomes = [self._import_asset(item) for item in work]
            self._save_outcomes(outcomes)
            if self._on_progress is not None:
                self._on_progress()
            if self._stop.is_set():
                return False

    def _pending_assets(self, limit: int) -> list[AssetWork]:
        with self._db.session() as session:
            rows = session.scalars(
                select(MediaAsset)
                .where(
                    MediaAsset.conversation_id == self._conversation_id,
                    MediaAsset.status == "pending",
                )
                .order_by(MediaAsset.id)
                .limit(limit)
            ).all()
            return [AssetWork(r.id, r.kind, r.message_id, r.source_path, r.orig_md5) for r in rows]

    def _import_asset(self, work: AssetWork) -> AssetOutcome:
        if not work.path:
            return AssetOutcome(work, "missing", "no_path")
        if split_relative(work.path) is None:
            return AssetOutcome(work, "missing", "unsafe_path")
        source = resolve_in_root(self._layout.root, work.path)
        if source is None:
            return AssetOutcome(work, "missing", "file_not_found")
        entry = self._integrity.lookup(work.path)
        try:
            return self._store_file(work, source, entry)
        except OSError:
            return AssetOutcome(work, "missing", "unreadable")

    def _store_file(
        self, work: AssetWork, source: Path, entry: IntegrityEntry | None
    ) -> AssetOutcome:
        mime = None
        width = height = None
        if work.kind in _IMAGE_KINDS:
            mime, width, height = _image_facts(source)
        algorithms = stream_algorithms(entry, "md5")
        with long_path(source).open("rb") as handle:
            hasher = StreamHasher(handle, algorithms)
            stored = self._media.put(cast(BinaryIO, hasher), _STORE_KINDS[work.kind])
        verdict = check_stream(entry, hasher)
        if verdict.verdict is FileVerdict.FAILED:
            if stored.created:
                self._media.delete(stored.sha256)
            self._stats.media["integrity_failed"] += 1
            return AssetOutcome(work, "corrupt", "integrity_failed")
        if verdict.verdict is FileVerdict.OK:
            self._stats.media["integrity_verified"] += 1
        elif self._integrity.entries:
            self._stats.media["integrity_unlisted"] += 1
        reason = None
        if work.orig_md5 and hasher.hexdigest("md5") != work.orig_md5.lower():
            reason = "md5_mismatch"
            self._stats.media["md5_mismatch"] += 1
        return AssetOutcome(
            work,
            "available",
            reason,
            sha256=stored.sha256,
            mime=mime or "application/octet-stream",
            width=width,
            height=height,
            size=stored.size,
        )

    def _save_outcomes(self, outcomes: list[AssetOutcome]) -> None:
        now = self._clock.now_utc()
        table = cast(Table, MediaAsset.__table__)
        with self._db.transaction(bump_state=False) as session:
            for outcome in outcomes:
                session.execute(
                    update(table)
                    .where(table.c.id == outcome.work.id)
                    .values(
                        status=outcome.status,
                        reason=outcome.reason,
                        sha256=outcome.sha256,
                        mime=outcome.mime,
                        width=outcome.width,
                        height=outcome.height,
                        size_bytes=outcome.size,
                        updated_at=now,
                    )
                )
                work = outcome.work
                if outcome.status == "available" and work.message_id and outcome.sha256:
                    stmt = update(MESSAGES).where(MESSAGES.c.id == work.message_id)
                    if work.kind != "image":  # the full picture outranks cover and voice file
                        stmt = stmt.where(MESSAGES.c.media_sha256.is_(None))
                    session.execute(stmt.values(media_sha256=outcome.sha256))

    # ----------------------------------------------------------------- stickers

    def run_stickers(self) -> bool:
        """Store local sticker files; returns ``False`` if stopped before the end."""
        index = self._emoji_index()
        after = ""
        while True:
            with self._db.session() as session:
                md5s = list(
                    session.scalars(
                        select(Sticker.md5)
                        .where(Sticker.status == "pending", Sticker.md5 > after)
                        .order_by(Sticker.md5)
                        .limit(STICKER_GROUP)
                    )
                )
            if not md5s:
                break
            self._store_stickers(md5s, index)
            after = md5s[-1]
            if self._on_progress is not None:
                self._on_progress()
            if self._stop.is_set():
                return False
        self.recount_sticker_uses()
        return True

    def _emoji_index(self) -> dict[str, Path]:
        """Files of ``media/emojis`` by their lower-case stem (the MD5 in the usual naming)."""
        found: dict[str, Path] = {}
        folder = self._layout.root / EMOJI_MEDIA_DIR
        if not long_path(folder).is_dir():
            return found
        for path in folder.iterdir():
            if long_path(path).is_file():
                found.setdefault(nfc(path.stem).lower(), path)
        return found

    def _store_stickers(self, md5s: list[str], index: dict[str, Path]) -> None:
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            for md5 in md5s:
                sticker = session.get(Sticker, md5)
                if sticker is None or sticker.status != "pending":
                    continue
                source = self._sticker_source(sticker, index)
                if source is None:
                    if sticker.url is None:  # nothing to read and nothing to download from
                        sticker.status, sticker.reason = "unavailable", "no_url"
                    continue  # otherwise the download job takes it from here
                path, relative = source
                self._stats.stickers["local_files"] += 1
                if entry_failed(self._integrity, relative, path):
                    sticker.status, sticker.reason = "unavailable", "integrity_failed"
                    self._stats.media["integrity_failed"] += 1
                    continue
                try:
                    if long_path(path).stat().st_size > MAX_STICKER_BYTES:
                        sticker.status, sticker.reason = "unavailable", "too_large"
                        continue
                    data = long_path(path).read_bytes()
                except OSError:
                    continue  # unreadable now; the download job will try the URL
                store_sticker_file(sticker, data, self._media, now)

    def _sticker_source(self, sticker: Sticker, index: dict[str, Path]) -> tuple[Path, str] | None:
        hint = sticker.source_path
        if hint:
            found = resolve_in_root(self._layout.root, hint)
            if found is not None:
                return found, hint
        local = index.get(sticker.md5)
        if local is not None:
            return local, normalize_key(f"{EMOJI_MEDIA_DIR}/{local.name}")
        return None

    def recount_sticker_uses(self) -> None:
        """Recompute use counts and first / last use of every sticker from ``sticker_uses``."""
        her = _uses_subquery(StickerUse.by_her.is_(True))
        user = _uses_subquery(StickerUse.by_her.is_(False))
        first = (
            select(func.min(StickerUse.used_at))
            .where(StickerUse.sticker_md5 == Sticker.md5)
            .scalar_subquery()
        )
        last = (
            select(func.max(StickerUse.used_at))
            .where(StickerUse.sticker_md5 == Sticker.md5)
            .scalar_subquery()
        )
        with self._db.transaction(bump_state=False) as session:
            session.execute(
                update(Sticker).values(
                    her_uses=her,
                    user_uses=user,
                    first_used_at=first,
                    last_used_at=last,
                    updated_at=Sticker.updated_at,
                )
            )


def _uses_subquery(condition: ColumnElement[bool]) -> ScalarSelect[int]:
    return (
        select(func.count())
        .select_from(StickerUse)
        .where(StickerUse.sticker_md5 == Sticker.md5, condition)
        .scalar_subquery()
    )


def entry_failed(index: IntegrityIndex, relative: str, path: Path) -> bool:
    return verify_file(index, relative, path).verdict is FileVerdict.FAILED


def _image_facts(source: Path) -> tuple[str | None, int | None, int | None]:
    """``(mime, width, height)`` of a picture file; unknown parts are ``None``."""
    mime: str | None = None
    try:
        with long_path(source).open("rb") as handle:
            mime = sniff_mime(handle.read(32))
    except (UnsupportedImageError, OSError):
        mime = None
    width = height = None
    try:
        with Image.open(long_path(source)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        pass
    return mime, width, height
