"""GIF pictures for the tests of the first frame (R-STK-007, R17-38); one flat colour a frame."""

from __future__ import annotations

import hashlib
import io

from PIL import Image
from sqlalchemy.orm import Session

from twin.services import Services
from twin.stickers.library import store_sticker_file
from twin.storage.chat_models import Sticker

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def animated_gif(
    colors: tuple[tuple[int, int, int], ...] = (RED, GREEN, BLUE), size: tuple[int, int] = (12, 8)
) -> bytes:
    """A GIF that moves: one flat colour per frame (the first is the one a still would show)."""
    frames = [Image.new("RGB", size, colour) for colour in colors]
    buffer = io.BytesIO()
    frames[0].save(buffer, "GIF", save_all=True, append_images=frames[1:], duration=80, loop=0)
    return buffer.getvalue()


def md5_of(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def store_as_sticker(
    services: Services,
    data: bytes,
    *,
    her_uses: int = 3,
    tags: tuple[str, ...] | None = None,
    session: Session | None = None,
) -> str:
    """Put ``data`` in the library as a stored sticker (status as the file decides); its MD5."""
    md5 = md5_of(data)
    now = services.clock.now_utc()

    def put(open_session: Session) -> None:
        row = Sticker(md5=md5, status="pending", attempts=0, her_uses=her_uses, user_uses=0)
        open_session.add(row)
        open_session.flush()
        store_sticker_file(row, data, services.media, now)
        if tags is not None:
            row.tags = list(tags)

    if session is not None:
        put(session)
    else:
        with services.db.transaction(bump_state=False) as opened:
            put(opened)
    return md5
