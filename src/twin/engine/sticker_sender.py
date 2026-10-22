"""The one way the engine sends a picture: a sticker of her library (R-SAFE-006, R-STK-007).

:meth:`StickerSender.send_sticker` takes a library sticker
(:class:`~twin.stickers.catalog.StickerRecord`, called ``Sticker`` here) - nothing else.  There
is no parameter for a path or for bytes, so a photo, a video frame or a voice message of the
media store cannot be handed to it; the bytes it sends are read from the media store under the
digest the library recorded for that sticker, and the channel checks once more that the digest
is on its allow list (``MediaNotAllowed`` otherwise).

**GIF and PNG (R-STK-007, D-696).**  Pictures go out as they are, and so does a GIF - unless the
setting ``stickers.gif_send_mode`` is ``first_frame``: then a GIF goes out as the still PNG of its
first frame, which the library derived when it stored the file (``first_frame_sha256``; made now,
and registered on the allow list, if the GIF predates it).  Whether a GIF moves on the phone is the
M0 probe's finding, and ``twin channel probe report`` suggests the mode.  The animated file is
**never** sent in place of a first frame that cannot be made: the sticker is not sent at all
(``MediaNotAllowed``, which the bubble sender treats like any sticker that cannot be sent).  The
setting is read at every send, so a change takes effect at the next sticker.

This is the only module of the engine that calls ``send_image`` of a channel; a scan test keeps it
that way.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from twin.channel.base import Channel, MediaNotAllowed, OutboundResult
from twin.stickers.catalog import StickerRecord
from twin.stickers.library import FIRST_FRAME_MIME, GIF_MIME, FirstFrameError, ensure_first_frame
from twin.storage.media import MediaStore

if TYPE_CHECKING:
    from twin.services import Services

Sticker = StickerRecord  # the library's sticker, under the name R-SAFE-006 gives it

AS_IS = "as_is"
FIRST_FRAME = "first_frame"
NOT_SENT = (
    "the still first frame of this GIF is not available ({why}); the animated file is not sent "
    "in its place"
)


class StickerSender:
    """Sends library stickers through a channel."""

    def __init__(
        self,
        channel: Channel,
        media: MediaStore,
        *,
        gif_send_mode: Callable[[], str] | None = None,
        make_first_frame: Callable[[str], str] | None = None,
    ) -> None:
        """``gif_send_mode`` says how a GIF goes out (``as_is`` by default); ``make_first_frame``
        makes the first-frame PNG of the library GIF with the given MD5 and returns its digest."""
        self._channel = channel
        self._media = media
        self._mode = gif_send_mode or (lambda: AS_IS)
        self._make = make_first_frame

    @classmethod
    def from_services(cls, channel: Channel, services: Services) -> StickerSender:
        """The sender of the running application: the configured mode, the library's frames."""
        settings = services.settings

        def make(md5: str) -> str:
            return ensure_first_frame(services.db, services.media, md5)

        return cls(
            channel,
            services.media,
            gif_send_mode=lambda: settings.stickers.gif_send_mode,
            make_first_frame=make,
        )

    async def send_sticker(self, sticker: Sticker) -> OutboundResult:
        """Send ``sticker`` to the bound user.

        Raises :class:`~twin.channel.base.MediaNotAllowed` for a sticker that is not usable
        (not downloaded, a picture that does not match its MD5, switched off) and whatever the
        channel raises for a recipient or a capability it refuses.
        """
        if not sticker.available or sticker.disabled or not sticker.sha256 or not sticker.mime:
            raise MediaNotAllowed("this sticker has no usable picture in the library")
        data, mime = await asyncio.to_thread(self._picture, sticker, sticker.sha256, sticker.mime)
        return await self._channel.send_image(data, mime)

    def _picture(self, sticker: Sticker, sha256: str, mime: str) -> tuple[bytes, str]:
        """The bytes and the type to send: the file, or the still first frame of a GIF."""
        if mime != GIF_MIME or self._mode() != FIRST_FRAME:
            return self._media.read_bytes(sha256), mime
        still = sticker.first_frame_sha256
        if not still or not self._media.exists(still):
            if self._make is None:
                raise MediaNotAllowed(NOT_SENT.format(why="no way to make it"))
            try:
                still = self._make(sticker.md5)
            except FirstFrameError as exc:
                raise MediaNotAllowed(NOT_SENT.format(why=str(exc))) from None
        return self._media.read_bytes(still), FIRST_FRAME_MIME
