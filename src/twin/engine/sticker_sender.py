"""The one way the engine sends a picture: a sticker of her library (R-SAFE-006, R-STK-007).

:meth:`StickerSender.send_sticker` takes a library sticker
(:class:`~twin.stickers.catalog.StickerRecord`, called ``Sticker`` here) - nothing else.  There
is no parameter for a path or for bytes, so a photo, a video frame or a voice message of the
media store cannot be handed to it; the bytes it sends are read from the media store under the
digest the library recorded for that sticker, and the channel checks once more that the digest
is on its allow list (``MediaNotAllowed`` otherwise).  GIF and PNG go out as they are
(R-STK-007; whether a GIF moves is the M0 probe's finding, not a decision here).

This is the only module of the engine that calls ``send_image`` of a channel; a scan test keeps it
that way.
"""

from __future__ import annotations

import asyncio

from twin.channel.base import Channel, MediaNotAllowed, OutboundResult
from twin.stickers.catalog import StickerRecord
from twin.storage.media import MediaStore

Sticker = StickerRecord  # the library's sticker, under the name R-SAFE-006 gives it


class StickerSender:
    """Sends library stickers through a channel."""

    def __init__(self, channel: Channel, media: MediaStore) -> None:
        self._channel = channel
        self._media = media

    async def send_sticker(self, sticker: Sticker) -> OutboundResult:
        """Send ``sticker`` to the bound user.

        Raises :class:`~twin.channel.base.MediaNotAllowed` for a sticker that is not usable
        (not downloaded, a picture that does not match its MD5, switched off) and whatever the
        channel raises for a recipient or a capability it refuses.
        """
        if not sticker.available or sticker.disabled or not sticker.sha256 or not sticker.mime:
            raise MediaNotAllowed("this sticker has no usable picture in the library")
        data = await asyncio.to_thread(self._media.read_bytes, sticker.sha256)
        return await self._channel.send_image(data, sticker.mime)
