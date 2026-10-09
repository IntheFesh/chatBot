"""How a message of the user reads in the conversation (R-ENG-013, R-STK-006, R-SAFE-006).

The model reads text, so every kind of message the user can send becomes a stable line of text,
made once when the message arrives and stored as it is in ``bot_turns`` (the same line then
appears in every later prompt, which keeps the cache prefix unchanged, R-LLM-010):

=============  =====================================================================
text           the words; a quote reply is preceded by ``[引用:<被引用的内容>]``
picture        ``[图片：<描述>]`` - described now by the vision model within
               ``ingest.caption_wait_timeout_s``; ``[图片]`` if that does not work out
voice          ``[语音：<转写>]`` from the transcript WeChat made, ``[语音，未转写]`` without
video          ``[视频：<封面描述>]`` from the cover picture, ``[视频]`` without a cover
file           ``[文件：<文件名>]``
sticker        ``[表情包：<描述>（情绪：<标签>）]`` from the library or the vision model
anything else  ``[其他消息]``
=============  =====================================================================

The bracket forms are the event-text templates of :mod:`twin.ingest.events` - the same text the
real history uses - so the model sees one notation everywhere.  The renderer never raises for a
picture it cannot describe: the reply goes on without the description.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from twin.channel.base import InboundMessage, MediaRef, MessageKind, QuoteInfo
from twin.engine.types import InboundItem
from twin.ingest.captions import CaptionService
from twin.ingest.events import Kind, render_event_text
from twin.llm.deepseek import DeepSeekClient
from twin.ops.logging import get_logger
from twin.stickers.incoming import describe_incoming_sticker
from twin.storage.media import MediaKind

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.inbound")

KIND_NAMES = {
    MessageKind.TEXT: "text",
    MessageKind.IMAGE: "image",
    MessageKind.VOICE: "voice",
    MessageKind.VIDEO: "video",
    MessageKind.FILE: "file",
    MessageKind.UNKNOWN: "unknown",
}


@dataclass(frozen=True)
class _EventView:
    """A message as :func:`~twin.ingest.events.render_event_text` reads it."""

    kind: str
    text: str | None = None
    is_sent: bool = False
    call_status: str | None = None
    call_duration_s: int | None = None
    voice_seconds: int | None = None
    has_transcript: bool = False


def quote_line(quote: QuoteInfo | None) -> str | None:
    """``[引用:<内容>]`` for a quote reply that carries something to show, else ``None``."""
    if quote is None:
        return None
    shown = " ".join((quote.text or quote.title or "").split())
    return f"[引用:{shown}]" if shown else None


def media_record(message: InboundMessage) -> dict[str, Any] | None:
    """What ``bot_turns`` keeps of the attachment, the quote and the flags (sealed there)."""
    record: dict[str, Any] = {}
    if message.media_ref is not None:
        record["media_ref"] = message.media_ref.to_dict()
    if message.quote is not None:
        record["quote"] = message.quote.to_dict()
    if message.flags:
        record["flags"] = sorted(message.flags)
    return record or None


class InboundRenderer:
    """Turns :class:`~twin.channel.base.InboundMessage` objects into :class:`InboundItem` lines."""

    def __init__(
        self,
        services: Services,
        *,
        captions: CaptionService | None = None,
        client: DeepSeekClient | None = None,
    ) -> None:
        self._services = services
        self._captions = captions or CaptionService(services, client)
        self._client = client

    async def aclose(self) -> None:
        await self._captions.aclose()

    async def render(self, message: InboundMessage) -> InboundItem:
        """The stable line for ``message`` (see the module description)."""
        text = await self._text(message)
        quote = quote_line(message.quote)
        if quote is not None and message.kind is MessageKind.TEXT:
            text = f"{quote}\n{text}" if text else quote
        kind = KIND_NAMES.get(message.kind, "unknown")
        ref = message.media_ref
        if message.kind is MessageKind.IMAGE and ref is not None and ref.kind is MediaKind.STICKER:
            kind = "sticker"
        return InboundItem(message.id, message.at, kind, text, media_record(message))

    async def render_all(self, messages: Sequence[InboundMessage]) -> list[InboundItem]:
        """Several messages at once (their descriptions are fetched side by side), in order."""
        return list(await asyncio.gather(*(self.render(message) for message in messages)))

    # -------------------------------------------------------------------------- kinds

    async def _text(self, message: InboundMessage) -> str:
        kind = message.kind
        if kind is MessageKind.TEXT:
            return (message.text or "").strip()
        ref = message.media_ref
        if kind is MessageKind.IMAGE:
            if ref is not None and ref.kind is MediaKind.STICKER:
                return await self._sticker(ref)
            return self._event(Kind.IMAGE, await self._describe(ref))
        if kind is MessageKind.VIDEO:
            return self._event(Kind.VIDEO, await self._describe(ref))
        if kind is MessageKind.VOICE:
            transcript = (message.text or "").strip()
            return self._event(Kind.VOICE, None, text=transcript, has_transcript=bool(transcript))
        if kind is MessageKind.FILE:
            name = (message.text or (ref.file_name if ref else None) or "").strip()
            return self._event(Kind.FILE, None, text=name)
        return self._event(Kind.UNKNOWN, None)

    @staticmethod
    def _event(
        kind: Kind, caption: str | None, *, text: str | None = None, has_transcript: bool = False
    ) -> str:
        view = _EventView(kind.value, text, has_transcript=has_transcript)
        return render_event_text(view, caption=caption) or ""

    async def _describe(self, ref: MediaRef | None) -> str | None:
        """The description of a picture (or a video cover) in the media store."""
        if ref is None or ref.kind not in (MediaKind.IMAGE, MediaKind.STICKER):
            return None
        return await self._captions.describe_stored(ref.sha256)

    async def _sticker(self, ref: MediaRef) -> str:
        data = await asyncio.to_thread(self._services.media.read_bytes, ref.sha256)
        found = await describe_incoming_sticker(self._services, image=data, client=self._client)
        return found.text
