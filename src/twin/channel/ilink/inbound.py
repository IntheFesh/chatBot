"""Inbound parsing: a wire message becomes ``InboundMessage`` objects (R-CH-005).

* text: the words;
* image: downloaded from the CDN, decrypted and stored in the encrypted media store;
* voice: the transcription WeChat's cloud already made (``None`` and a flag when absent);
* video: the cover picture (``thumb_media``), the video itself is not fetched;
* file: the file name (the file is not fetched);
* a quote: the quoted text or picture, looked up by server id when only the id was sent;
* anything else: ``unknown`` with the item type number.

A media problem never loses the message: the message is delivered without its media and a
flag says what went wrong.  Only type numbers and reasons are recorded, never content.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from twin.channel.base import (
    FLAG_MEDIA_UNAVAILABLE,
    FLAG_UNKNOWN_ITEM_TYPE,
    FLAG_VIDEO_NO_COVER,
    FLAG_VOICE_UNTRANSCRIBED,
    InboundMessage,
    MediaRef,
    MessageKind,
    QuoteInfo,
)
from twin.channel.ilink.media import (
    CdnClient,
    MediaCryptoError,
    MediaTransferError,
    aes_ecb_decrypt,
    download_url,
    parse_hex_key,
    parse_media_key,
    sniff_image_mime,
)
from twin.channel.ilink.store import IlinkStore
from twin.channel.ilink.wire import (
    ITEM_FILE,
    ITEM_IMAGE,
    ITEM_TEXT,
    ITEM_TOOL_RESULT,
    ITEM_TOOL_START,
    ITEM_VIDEO,
    ITEM_VOICE,
    CdnMedia,
    PartialText,
    RefMsg,
    WireItem,
    WireMessage,
    as_text,
)
from twin.clock import Clock, from_epoch
from twin.ops.logging import get_logger
from twin.storage.media import MediaKind, MediaStore

log = get_logger("twin.channel.ilink.inbound")


class MediaFetchError(Exception):
    """Fetching or decrypting one media file failed; ``reason`` is a short code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Converted:
    """The result of converting one wire message."""

    messages: list[InboundMessage] = field(default_factory=list)
    item_types: list[int] = field(default_factory=list)
    failures: Counter[str] = field(default_factory=Counter)
    quote_entries: list[tuple[str, str]] = field(default_factory=list)


def message_time(wire: WireMessage, received_at: datetime) -> datetime:
    """``create_time_ms`` as UTC; values that look like seconds are accepted too."""
    value = wire.create_time_ms
    if value is None or value <= 0:
        return received_at
    millis = value if value > 1e12 else value * 1000
    return from_epoch(millis / 1000)


def _nth(haystack: str, needle: str, n: int, start: int = 0) -> int:
    position = start - 1
    for _ in range(max(n, 0) + 1):
        position = haystack.find(needle, position + 1)
        if position < 0:
            return -1
    return position


def extract_partial(full: str, partial: PartialText) -> str:
    """The selected part of a quoted text (protocol document section 6.4).

    ``startindex``/``endindex`` count occurrences (from 0).  The end index may mean "the n-th
    occurrence in the whole text" or "the n-th after the start"; both readings are candidates
    and ``quotemd5`` picks the right one.  Without a hash the first reading is used; when
    nothing fits, the whole quoted text is returned.
    """
    start, end = partial.start, partial.end
    if not start or not end:
        return full
    begin = _nth(full, start, partial.startindex or 0)
    if begin < 0:
        return full
    candidates: list[str] = []
    for finish in (
        _nth(full, end, partial.endindex or 0),
        _nth(full, end, partial.endindex or 0, start=begin),
    ):
        if finish >= begin:
            candidate = full[begin : finish + len(end)]
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        return full
    if partial.quotemd5:
        for candidate in candidates:
            digest = hashlib.md5(candidate.encode("utf-8"), usedforsecurity=False).hexdigest()
            if digest == partial.quotemd5.lower():
                return candidate
        return full
    return candidates[0]


class InboundConverter:
    """Converts wire messages, downloading media through the CDN client."""

    def __init__(self, cdn: CdnClient, media: MediaStore, store: IlinkStore, clock: Clock) -> None:
        self._cdn = cdn
        self._media = media
        self._store = store
        self._clock = clock

    async def convert(self, wire: WireMessage, message_id: str) -> Converted:
        received_at = self._clock.now_utc()
        at = message_time(wire, received_at)
        items = [
            item for item in wire.items() if item.type not in (ITEM_TOOL_START, ITEM_TOOL_RESULT)
        ]
        result = Converted(item_types=[item.type for item in wire.items() if item.type is not None])
        quote: QuoteInfo | None = None
        quote_index: int | None = None
        for index, item in enumerate(items):
            if item.ref_msg is not None:
                quote = await self._resolve_quote(item.ref_msg, result.failures)
                quote_index = index
                break
        built: list[tuple[int, InboundMessage]] = []
        for index, item in enumerate(items):
            suffix = message_id if len(items) == 1 else f"{message_id}#{index}"
            message = await self._convert_item(item, suffix, at, result)
            if message is not None:
                built.append((index, message))
        if quote is not None and built:
            target = next((pos for pos, (i, _) in enumerate(built) if i == quote_index), 0)
            built[target] = (built[target][0], _with_quote(built[target][1], quote))
        result.messages = [message for _, message in built]
        for message in result.messages:
            if message.text and message.kind in (MessageKind.TEXT, MessageKind.VOICE):
                result.quote_entries.append((message.id, message.text))
        for item in items:
            item_id = as_text(item.msg_id)
            text = _item_text(item)
            if item_id and text:
                result.quote_entries.append((item_id, text))
        return result

    # ----------------------------------------------------------- one item

    async def _convert_item(
        self, item: WireItem, message_id: str, at: datetime, result: Converted
    ) -> InboundMessage | None:
        kind = item.type
        if kind == ITEM_TEXT:
            text = item.text_item.text if item.text_item else None
            if not text:
                return None
            return InboundMessage(message_id, at, MessageKind.TEXT, text=text, item_type=kind)
        if kind == ITEM_IMAGE:
            image = item.image_item
            ref, flags = None, frozenset[str]()
            if image is not None and image.media is not None:
                ref = await self._try_media(
                    image.media, image.aeskey, MediaKind.IMAGE, "image", result.failures
                )
            if ref is None:
                flags = frozenset({FLAG_MEDIA_UNAVAILABLE})
            return InboundMessage(
                message_id, at, MessageKind.IMAGE, media_ref=ref, flags=flags, item_type=kind
            )
        if kind == ITEM_VOICE:
            transcript = item.voice_item.text if item.voice_item else None
            if transcript:
                return InboundMessage(
                    message_id, at, MessageKind.VOICE, text=transcript, item_type=kind
                )
            return InboundMessage(
                message_id,
                at,
                MessageKind.VOICE,
                flags=frozenset({FLAG_VOICE_UNTRANSCRIBED}),
                item_type=kind,
            )
        if kind == ITEM_FILE:
            name = item.file_item.file_name if item.file_item else None
            return InboundMessage(message_id, at, MessageKind.FILE, text=name, item_type=kind)
        if kind == ITEM_VIDEO:
            cover_media = item.video_item.thumb_media if item.video_item else None
            ref = None
            if cover_media is not None:
                ref = await self._try_media(
                    cover_media, None, MediaKind.IMAGE, "video_cover", result.failures
                )
            flags = frozenset() if ref is not None else frozenset({FLAG_VIDEO_NO_COVER})
            return InboundMessage(
                message_id, at, MessageKind.VIDEO, media_ref=ref, flags=flags, item_type=kind
            )
        result.failures["unknown_item_type"] += 1
        log.debug("unknown_item_type", item_type=kind)
        return InboundMessage(
            message_id,
            at,
            MessageKind.UNKNOWN,
            flags=frozenset({FLAG_UNKNOWN_ITEM_TYPE}),
            item_type=kind,
        )

    # -------------------------------------------------------------- media

    async def _try_media(
        self,
        media: CdnMedia,
        hex_key: str | None,
        kind: MediaKind,
        label: str,
        failures: Counter[str],
    ) -> MediaRef | None:
        try:
            return await self._fetch(media, hex_key, kind)
        except MediaFetchError as exc:
            failures[f"{label}.{exc.reason}"] += 1
            log.warning("inbound_media_failed", media=label, reason=exc.reason)
            return None

    async def _fetch(self, media: CdnMedia, hex_key: str | None, kind: MediaKind) -> MediaRef:
        url = download_url(media)
        if url is None:
            raise MediaFetchError("no_download_address")
        key = _first_key(hex_key, media.aes_key)
        try:
            data = await self._cdn.download(url)
        except MediaTransferError as exc:
            reason = f"download_http_{exc.status}" if exc.status else "download_failed"
            raise MediaFetchError(reason) from None
        if key is not None:
            try:
                data = aes_ecb_decrypt(key, data)
            except MediaCryptoError:
                raise MediaFetchError("decrypt_failed") from None
        stored = await asyncio.to_thread(self._media.put, data, kind)
        return MediaRef(stored.sha256, kind, stored.size, sniff_image_mime(data))

    # -------------------------------------------------------------- quote

    async def _resolve_quote(self, ref: RefMsg, failures: Counter[str]) -> QuoteInfo:
        svr_id = as_text(ref.svr_id)
        text: str | None = None
        media_ref: MediaRef | None = None
        quoted = ref.message_item
        if quoted is not None:
            text = _item_text(quoted)
            if quoted.type == ITEM_IMAGE and quoted.image_item and quoted.image_item.media:
                media_ref = await self._try_media(
                    quoted.image_item.media,
                    quoted.image_item.aeskey,
                    MediaKind.IMAGE,
                    "quoted_image",
                    failures,
                )
            elif quoted.type == ITEM_VIDEO and quoted.video_item and quoted.video_item.thumb_media:
                media_ref = await self._try_media(
                    quoted.video_item.thumb_media, None, MediaKind.IMAGE, "quoted_cover", failures
                )
        elif svr_id:
            text = await asyncio.to_thread(self._store.quote_text, svr_id)
        if text and ref.partial_text is not None:
            text = extract_partial(text, ref.partial_text)
        return QuoteInfo(
            text=text,
            title=ref.title,
            svr_id=svr_id,
            media_ref=media_ref,
            resolved=text is not None or media_ref is not None,
        )


def _first_key(hex_key: str | None, media_key: str | None) -> bytes | None:
    """``image_item.aeskey`` first, then ``media.aes_key``; none at all means plaintext."""
    for value, parser in ((hex_key, parse_hex_key), (media_key, parse_media_key)):
        if value:
            try:
                return parser(value)
            except MediaCryptoError:
                continue
    if hex_key or media_key:
        raise MediaFetchError("bad_key")
    return None


def _item_text(item: WireItem) -> str | None:
    if item.type == ITEM_TEXT and item.text_item:
        return item.text_item.text or None
    if item.type == ITEM_VOICE and item.voice_item:
        return item.voice_item.text or None
    if item.type == ITEM_FILE and item.file_item:
        return item.file_item.file_name or None
    return None


def _with_quote(message: InboundMessage, quote: QuoteInfo) -> InboundMessage:
    return InboundMessage(
        id=message.id,
        at=message.at,
        kind=message.kind,
        text=message.text,
        media_ref=message.media_ref,
        quote=quote,
        flags=message.flags,
        item_type=message.item_type,
    )
