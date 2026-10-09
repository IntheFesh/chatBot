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
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import datetime

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from twin.llm.errors import UnsupportedImageError
from twin.llm.images import sniff_mime
from twin.storage.chat_models import Sticker
from twin.storage.media import MediaKind, MediaStore

MAX_STICKER_BYTES = 16 * 1024 * 1024
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
    if described.md5 == sticker.md5:
        sticker.status, sticker.reason = "available", None
    else:
        sticker.status, sticker.reason = "md5_mismatch", "md5_mismatch"
    return sticker.status


def sticker_file_allowed(session: Session, sha256: str) -> bool:
    """True if ``sha256`` is the stored file of an ``available`` sticker (R-SAFE-006).

    The check is on the SHA-256 of the file's *content*, the same hash the media store names
    its files by, so the channel can verify the exact bytes it is about to upload.
    """
    if not _SHA256.fullmatch(sha256):
        return False
    row = session.execute(
        select(Sticker.md5).where(Sticker.sha256 == sha256, Sticker.status == "available").limit(1)
    ).first()
    return row is not None
