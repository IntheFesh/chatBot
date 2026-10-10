"""Recognising a sticker the user sent (R-STK-006).

:func:`describe_incoming_sticker` answers "what did the user just send?" for the reply path:

1. an MD5 that is in the library with a description is answered from the library, with no API
   call at all (her stickers and every sticker either of them sent in the imported history);
2. otherwise the vision model describes the picture - from the bytes the channel downloaded, or
   from the stored file if the library has one - within ``stickers.describe_timeout_s`` (15 s);
3. the result is cached in the library, marked as coming from the user (``origin =
   "incoming"``), so the next time the same sticker needs no call;
4. if the picture is missing, the model answers wrongly or too slowly, or the API fails, the
   answer is the plain ``[表情包]`` and nothing is cached: the reply goes on without a
   description rather than waiting.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import LlmError
from twin.llm.images import DETAIL_STICKER, ImageInput
from twin.llm.runtime import build_llm_runtime
from twin.llm.types import DAILY, LedgerTag
from twin.ops.logging import get_logger
from twin.services import Services
from twin.stickers.catalog import ORIGIN_INCOMING, StickerCatalog, StickerRecord
from twin.stickers.library import store_sticker_file
from twin.stickers.tagging import StickerTagger, Vision
from twin.storage.chat_models import Sticker

log = get_logger("twin.stickers.incoming")

FALLBACK_TEXT = "[表情包]"


@dataclass(frozen=True)
class StickerDescription:
    """What the reply path learns about a sticker the user sent."""

    md5: str | None
    tags: tuple[str, ...]
    description: str | None
    source: str  # "library", "vision" or "none"

    @property
    def known(self) -> bool:
        return self.source != "none"

    @property
    def text(self) -> str:
        """The line for the prompt: ``[表情包：<描述>（情绪：<标签>）]``, or ``[表情包]``."""
        if not self.known or not self.description:
            return FALLBACK_TEXT
        tags = "、".join(self.tags)
        return (
            f"[表情包：{self.description}（情绪：{tags}）]"
            if tags
            else f"[表情包：{self.description}]"
        )


NOTHING = StickerDescription(None, (), None, "none")


def _from_record(record: StickerRecord, source: str) -> StickerDescription:
    return StickerDescription(record.md5, record.tags, record.description, source)


def _cache(
    services: Services, catalog: StickerCatalog, md5: str, data: bytes | None, seen: Vision
) -> StickerRecord:
    """Store the description; a sticker the library does not know yet is added first."""
    now = services.clock.now_utc()
    if catalog.get(md5) is None:
        with services.db.transaction(bump_state=False) as session:
            row = Sticker(
                md5=md5,
                status="pending",
                attempts=0,
                her_uses=0,
                user_uses=0,
                context_uses=0,
                disabled=False,
                origin=ORIGIN_INCOMING,
                created_at=now,
                updated_at=now,
            )
            if data is not None:
                store_sticker_file(row, data, services.media, now)
            else:
                row.status = "unavailable"
                row.reason = "no_file"
            session.add(row)
    return catalog.save_vision(md5, seen.tags, seen.description, seen.use_cases, at=now)


async def describe_incoming_sticker(
    services: Services,
    *,
    md5: str | None = None,
    image: bytes | None = None,
    client: DeepSeekClient | None = None,
    tag: LedgerTag = DAILY,
) -> StickerDescription:
    """Describe a sticker by its MD5 (library) or its picture (vision); see the module text."""
    key = md5.strip().lower() if md5 else None
    if key is None and image is not None:
        key = hashlib.md5(image, usedforsecurity=False).hexdigest()
    if key is None:
        return NOTHING
    catalog = StickerCatalog(services)
    record = await asyncio.to_thread(catalog.get, key)
    if record is not None and record.description and record.tags:
        return _from_record(record, "library")
    source = (
        ImageInput.from_bytes(image, detail=DETAIL_STICKER)
        if image is not None
        else (
            ImageInput.from_media(services.media, record.sha256, detail=DETAIL_STICKER)
            if record is not None and record.sha256 and record.available
            else None
        )
    )
    if source is None:
        return StickerDescription(key, record.tags if record else (), None, "none")
    owned = client is None
    runtime = build_llm_runtime(services) if client is None else None
    active = client if client is not None else (runtime.client if runtime else None)
    if active is None:
        return NOTHING
    tagger = StickerTagger(services, active, catalog=catalog)
    timeout = services.settings.stickers.describe_timeout_s
    try:
        seen = await asyncio.wait_for(tagger.see(source, tag), timeout)
        cached = await asyncio.to_thread(_cache, services, catalog, key, image, seen)
    except (TimeoutError, LlmError, ValueError, OSError) as exc:
        log.warning("incoming_sticker_not_described", reason=type(exc).__name__)
        return StickerDescription(key, (), None, "none")
    finally:
        if owned:
            await active.aclose()
    return _from_record(cached, "vision")
