"""The sticker library: stored files, their status and the outbound allow-list (R-IMP-008).

A sticker is identified by the MD5 the export names (``emojiMd5``).  Its picture is stored
encrypted in the media store; the ``stickers`` row records the stored file's SHA-256 and a
status:

``pending``       known, file not obtained yet;
``available``     file stored and its MD5 equals the exported ``emojiMd5``;
``md5_mismatch``  file stored but its MD5 differs (kept for recognition, never sent);
``unavailable``   no usable file (reason recorded; ``twin stickers download --retry-failed``).

Only ``available`` stickers may be sent to the chat (R-SAFE-006): the channel asks
:func:`sticker_file_allowed` whether the bytes it is about to upload are such a sticker.

**The first frame of a GIF (R-STK-007, D-696, D-697).**  Whether a GIF moves on the phone is the
finding of the M0 probe; if it does not, the user may let the bot send the still first frame
(``stickers.gif_send_mode: first_frame``).  So that the choice costs nothing to make, every GIF
that is stored also gets its first frame as a PNG - derived with Pillow, stored encrypted like
every picture, its digest in ``stickers.first_frame_sha256``.  A GIF stored before that existed
gets its PNG from :func:`backfill_first_frames` (``twin stickers first-frames``) or, at the
latest, from :func:`ensure_first_frame` when the bot is about to send it in that mode.  The
PNG is on the allow-list of the channel next to the original and is nothing else.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, UnidentifiedImageError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from twin.llm.errors import UnsupportedImageError
from twin.llm.images import sniff_mime
from twin.ops.logging import get_logger
from twin.storage.chat_models import Sticker
from twin.storage.crypto import DecryptionError
from twin.storage.db import Database
from twin.storage.media import MediaError, MediaKind, MediaStore
from twin.storage.state import bump_state_version

log = get_logger("twin.stickers.library")

MAX_STICKER_BYTES = 16 * 1024 * 1024
GIF_MIME = "image/gif"
FIRST_FRAME_MIME = "image/png"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StickerImage:
    """What is known about the bytes of a sticker file."""

    mime: str
    width: int | None
    height: int | None
    size: int
    md5: str
    sha256: str


def describe_sticker_bytes(data: bytes) -> StickerImage:
    """Identify a picture by its file header (GIF, PNG, JPEG, WebP); raises if it is none."""
    mime = sniff_mime(data)  # UnsupportedImageError for anything else
    width: int | None = None
    height: int | None = None
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError):
        pass  # the header is right but Pillow cannot read the size; the file is still kept
    return StickerImage(
        mime=mime,
        width=width,
        height=height,
        size=len(data),
        md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
        sha256=hashlib.sha256(data).hexdigest(),
    )


class FirstFrameError(Exception):
    """The still first frame of a sticker cannot be made (the reason is safe to log)."""


def first_frame_png(data: bytes) -> bytes:
    """The first frame of the GIF ``data`` as PNG bytes (transparency kept).

    Raises :class:`FirstFrameError` if Pillow cannot read the picture - a file whose header says
    GIF but whose frames are damaged, or one too large to open safely.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(0)
            frame = image.convert("RGBA")
        out = io.BytesIO()
        frame.save(out, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError, EOFError, Image.DecompressionBombError):
        raise FirstFrameError("the picture could not be read as a GIF") from None
    return out.getvalue()


def attach_first_frame(sticker: Sticker, data: bytes, media: MediaStore) -> str:
    """Make and store the first-frame PNG of the GIF ``data`` of ``sticker``; returns its digest.

    The caller holds the session the row belongs to.  Raises :class:`FirstFrameError`.
    """
    stored = media.put(first_frame_png(data), MediaKind.STICKER)
    sticker.first_frame_sha256 = stored.sha256
    return stored.sha256


def store_sticker_file(sticker: Sticker, data: bytes, media: MediaStore, now: datetime) -> str:
    """Store ``data`` as the file of ``sticker`` and set its status; returns the new status.

    A file that is not a GIF/PNG/JPEG/WebP picture makes the sticker ``unavailable``
    (reason ``not_an_image``); a picture with a different MD5 is kept as ``md5_mismatch``.
    """
    sticker.attempts += 1
    sticker.last_attempt_at = now
    if len(data) > MAX_STICKER_BYTES:
        sticker.status, sticker.reason = "unavailable", "too_large"
        return sticker.status
    try:
        described = describe_sticker_bytes(data)
    except UnsupportedImageError:
        sticker.status, sticker.reason = "unavailable", "not_an_image"
        return sticker.status
    media.put(data, MediaKind.STICKER)
    sticker.sha256 = described.sha256
    sticker.mime = described.mime
    sticker.width = described.width
    sticker.height = described.height
    sticker.size_bytes = described.size
    sticker.first_frame_sha256 = None
    if described.md5 == sticker.md5:
        sticker.status, sticker.reason = "available", None
        if described.mime == GIF_MIME:
            try:
                attach_first_frame(sticker, data, media)
            except FirstFrameError as exc:  # the GIF stays; the still is made again on demand
                log.warning("first_frame_not_made", reason=str(exc))
    else:
        sticker.status, sticker.reason = "md5_mismatch", "md5_mismatch"
    return sticker.status


def ensure_first_frame(db: Database, media: MediaStore, md5: str) -> str:
    """The digest of the first-frame PNG of the library GIF ``md5``, made now if it is missing.

    Blocking (database, media store, Pillow).  The row is changed in a transaction that moves
    the state version, so the running application's label book hears of the new picture.  Raises
    :class:`FirstFrameError` for a sticker that is not an ``available`` GIF of the library or
    whose first frame cannot be made - the caller must not fall back to the animated file.
    """
    with db.transaction(bump_state=False) as session:
        row = session.get(Sticker, md5)
        if row is None or row.status != "available" or row.mime != GIF_MIME or not row.sha256:
            raise FirstFrameError("this sticker is not a usable GIF of the library")
        if row.first_frame_sha256 and media.exists(row.first_frame_sha256):
            return row.first_frame_sha256
        try:
            data = media.read_bytes(row.sha256)
        except (MediaError, DecryptionError, OSError):
            raise FirstFrameError("the file of the GIF cannot be read") from None
        made = attach_first_frame(row, data, media)
        bump_state_version(session, db.clock)  # only a new picture is news to the application
        return made


@dataclass(frozen=True)
class FirstFrameBackfill:
    """What :func:`backfill_first_frames` did."""

    made: int
    present: int
    failed: int


def backfill_first_frames(db: Database, media: MediaStore) -> FirstFrameBackfill:
    """Give every available GIF of the library its first-frame PNG (R-IMP-011, R17-38).

    A sticker that already has a PNG that is still in the media store is left alone, so running
    it again does nothing.  Each sticker is its own short transaction; one that cannot be
    decoded is counted and skipped.
    """
    with db.session() as session:
        candidates = list(
            session.scalars(
                select(Sticker.md5)
                .where(Sticker.status == "available", Sticker.mime == GIF_MIME)
                .order_by(Sticker.md5)
            )
        )
    made = present = failed = 0
    for md5 in candidates:
        with db.session() as session:
            known = session.scalar(select(Sticker.first_frame_sha256).where(Sticker.md5 == md5))
        if known and media.exists(known):
            present += 1
            continue
        try:
            ensure_first_frame(db, media, md5)
        except FirstFrameError:
            failed += 1
        else:
            made += 1
    return FirstFrameBackfill(made, present, failed)


def sticker_file_allowed(session: Session, sha256: str) -> bool:
    """True if ``sha256`` is the stored file of an ``available`` sticker (R-SAFE-006).

    The check is on the SHA-256 of the file's *content*, the same hash the media store names
    its files by, so the channel can verify the exact bytes it is about to upload.  The still
    first frame of a GIF sticker (``first_frame_sha256``) is accepted like the GIF itself.
    """
    if not _SHA256.fullmatch(sha256):
        return False
    row = session.execute(
        select(Sticker.md5)
        .where(
            or_(Sticker.sha256 == sha256, Sticker.first_frame_sha256 == sha256),
            Sticker.status == "available",
        )
        .limit(1)
    ).first()
    return row is not None


class StickerLibraryAllowList:
    """The channel's :class:`~twin.channel.policy.StickerAllowList`, backed by the sticker table.

    Called from a worker thread right before an upload, so it opens its own short session.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def allowed(self, sha256: str) -> bool:
        with self._db.session() as session:
            return sticker_file_allowed(session, sha256)
