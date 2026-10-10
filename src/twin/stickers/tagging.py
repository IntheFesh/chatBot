"""Tagging stickers: what the picture shows, and what she used it for (R-STK-003).

**Vision.**  The vision model (``deepseek.vision_model``) is shown the sticker with ``detail:
low`` (the client leaves the parameter out if the M0 probe found it unsupported) and answers in
JSON: up to three tags from the closed vocabulary, one sentence describing the picture and one
describing when it is used.  A tag outside the vocabulary makes the reply invalid; the client
sends the error back once and asks again (R-LLM-003).  The description and the use cases are
redacted (R-LLM-009) before they are stored.

**Context.**  For a sticker she used at least ``stickers.context_min_uses`` (3) times *before
the hold-out cutoff*, up to ``stickers.context_max_samples`` (5) of those uses - spread over the
period - are shown to DeepSeek with the few messages before and after each, and the model says
what she used it for.  The evidence of use outweighs the picture when the two are merged
(:func:`twin.stickers.tags.merge_tags`).  Nothing at or after the cutoff is read: not the uses,
not the messages around them (R-TRN-013).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, select

from twin.ingest.corpus import messages_between
from twin.ingest.transcript import StickerTagOf, TranscriptLine, transcript_lines
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import ImageError
from twin.llm.images import DETAIL_STICKER, ImageInput
from twin.llm.redaction import ConsistentRedactor, redact
from twin.llm.types import DAILY, LedgerTag, Purpose
from twin.ops.logging import get_logger
from twin.profile.holdout import HoldoutError, holdout_cutoff
from twin.profile.prompt_templates import STICKER_CONTEXT, STICKER_TAG, PromptText, TemplateStore
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, StickerRecord
from twin.stickers.tags import MAX_TAGS, TagVocabulary
from twin.storage.chat_models import StickerUse

log = get_logger("twin.stickers.tagging")

TEXT_CHARS = 200
CONTEXT_BEFORE = 4
CONTEXT_AFTER = 2
SEARCH_RADIUS_S = 3600
VISION_COMPLETION_TOKENS = 160
CONTEXT_COMPLETION_TOKENS = 120
MARKER = "【她发了这个表情包】"


class _TagReply(BaseModel):
    """The tags of a reply: 1 to 3, all from the vocabulary of the schema subclass."""

    model_config = ConfigDict(extra="ignore")

    vocabulary: ClassVar[tuple[str, ...]] = ()
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(tag.strip() for tag in value if tag.strip()))
        if not cleaned:
            raise ValueError("tags must name at least one tag")
        unknown = [tag for tag in cleaned if tag not in cls.vocabulary]
        if unknown:
            raise ValueError(
                f"tag {unknown[0]!r} is not allowed; use only: {'、'.join(cls.vocabulary)}"
            )
        return cleaned[:MAX_TAGS]


class VisionReply(_TagReply):
    """``{tags, description, use_cases}``."""

    description: str
    use_cases: str = ""

    @field_validator("description")
    @classmethod
    def _need_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty")
        return value


class ContextReply(_TagReply):
    """``{tags, meaning}``."""

    meaning: str = ""


def reply_model[M: _TagReply](base: type[M], vocabulary: TagVocabulary) -> type[M]:
    """``base`` with its tag vocabulary set (the schema the client validates against)."""
    return cast(type[M], type(base.__name__, (base,), {"vocabulary": vocabulary.tags}))


def one_line(text: str, limit: int = TEXT_CHARS) -> str:
    """One line, redacted, shortened."""
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1] + "…"
    return redact(collapsed).text


@dataclass(frozen=True)
class Vision:
    tags: list[str]
    description: str
    use_cases: str


# ------------------------------------------------------------------------ uses


def her_uses_before(services: Services, md5: str, cutoff: datetime) -> list[tuple[str, datetime]]:
    """``(message id, time)`` of her uses of a sticker before ``cutoff``, oldest first."""
    stmt = (
        select(StickerUse.message_id, StickerUse.used_at)
        .where(
            StickerUse.sticker_md5 == md5,
            StickerUse.by_her.is_(True),
            StickerUse.used_at < cutoff,
        )
        .order_by(StickerUse.used_at, StickerUse.message_id)
    )
    with services.db.session() as session:
        return [(row[0], row[1]) for row in session.execute(stmt)]


def her_use_counts_before(services: Services, cutoff: datetime, minimum: int) -> dict[str, int]:
    """Stickers she used at least ``minimum`` times before ``cutoff``, with the counts."""
    stmt = (
        select(StickerUse.sticker_md5, func.count())
        .where(StickerUse.by_her.is_(True), StickerUse.used_at < cutoff)
        .group_by(StickerUse.sticker_md5)
        .having(func.count() >= minimum)
    )
    with services.db.session() as session:
        return {row[0]: int(row[1]) for row in session.execute(stmt)}


def context_due(minimum: int, record: StickerRecord, cutoff: datetime, uses: int) -> bool:
    """Is a correction due: enough uses, and none yet for this cutoff or her uses doubled."""
    if uses < minimum or not record.available:
        return False
    if record.context_tagged_at is None or record.context_cutoff_at != cutoff:
        return True
    return uses >= 2 * max(1, record.context_uses)


def spread(count: int, wanted: int) -> list[int]:
    """``wanted`` indexes spread evenly over ``count`` items (all of them if fewer)."""
    if count <= wanted:
        return list(range(count))
    if wanted == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})


def context_of_use(
    services: Services,
    message_id: str,
    moment: datetime,
    cutoff: datetime,
    *,
    tag_of: StickerTagOf | None,
) -> list[TranscriptLine]:
    """The messages around one use: a few before, the sticker (marked), a few after.

    Only messages before ``cutoff`` are read.  ``system`` notices have no line.
    """
    radius = timedelta(seconds=SEARCH_RADIUS_S)
    with services.db.session() as session:
        rows = list(
            session.scalars(messages_between(moment - radius, moment + radius, before=cutoff))
        )
        rows = [row for row in rows if row.kind != "system"]
        ids = [row.id for row in rows]
        if message_id not in ids:
            return []
        position = ids.index(message_id)
        window = rows[max(0, position - CONTEXT_BEFORE) : position + 1 + CONTEXT_AFTER]
        lines = transcript_lines(window, sticker_tag_of=tag_of)
    return [
        TranscriptLine(
            line.message_id, line.her, MARKER if line.message_id == message_id else line.text
        )
        for line in lines
    ]


def uses_text(blocks: list[list[TranscriptLine]], redactor: ConsistentRedactor) -> str:
    """The surroundings of several uses as prompt text (redacted)."""
    out: list[str] = []
    for number, lines in enumerate(blocks, start=1):
        out.append(f"场合 {number}：")
        for line in lines:
            shown = line.text if line.text == MARKER else redactor.redact_text(line.text)
            out.append(TranscriptLine(line.message_id, line.her, shown).render())
        out.append("")
    return "\n".join(out).strip()


# ----------------------------------------------------------------------- tagger


class StickerTagger:
    """Tags stickers with the vision model and the context correction."""

    def __init__(
        self,
        services: Services,
        client: DeepSeekClient,
        *,
        catalog: StickerCatalog | None = None,
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._catalog = catalog or StickerCatalog(services)
        self._templates = templates or TemplateStore(services.db, services.clock)

    @property
    def catalog(self) -> StickerCatalog:
        return self._catalog

    def template(self, name: str) -> PromptText:
        return self._templates.active(name)

    def _vocabulary_text(self) -> str:
        return "、".join(self._catalog.vocabulary.tags)

    # ----------------------------------------------------------------- vision

    async def see(self, image: ImageInput, tag: LedgerTag = DAILY) -> Vision:
        """Ask the vision model what a picture shows (does not store anything)."""
        schema = reply_model(VisionReply, self._catalog.vocabulary)
        messages = self.template(STICKER_TAG).render(vocabulary=self._vocabulary_text())
        result = await self._client.chat_json(
            messages,
            schema,
            purpose=Purpose.STICKER_TAG,
            images=[image],
            tag=tag,
            temperature=0.2,
            max_tokens=VISION_COMPLETION_TOKENS * 4,
        )
        reply = result.value
        return Vision(reply.tags, one_line(reply.description), one_line(reply.use_cases))

    def image_of(self, record: StickerRecord) -> ImageInput:
        if not record.sha256:
            raise ImageError(f"sticker {record.md5} has no stored file")
        return ImageInput.from_media(self._services.media, record.sha256, detail=DETAIL_STICKER)

    async def tag_sticker(self, md5: str, tag: LedgerTag = DAILY) -> bool:
        """Tag one sticker by its picture; ``False`` if it was already tagged or has no file."""
        record = await asyncio.to_thread(self._catalog.require, md5)
        if record.vision_tags or not record.available:
            return False
        seen = await self.see(self.image_of(record), tag)
        await asyncio.to_thread(
            self._catalog.save_vision,
            md5,
            seen.tags,
            seen.description,
            seen.use_cases,
            at=self._services.clock.now_utc(),
        )
        return True

    # ---------------------------------------------------------------- context

    async def correct_with_context(self, md5: str, tag: LedgerTag = DAILY) -> bool:
        """Judge from her earlier uses what the sticker means to her; ``False`` if not due."""
        try:
            cutoff = await asyncio.to_thread(holdout_cutoff, self._services)
        except HoldoutError:
            return False
        config = self._services.settings.stickers
        record = await asyncio.to_thread(self._catalog.require, md5)
        uses = await asyncio.to_thread(her_uses_before, self._services, md5, cutoff)
        if not context_due(config.context_min_uses, record, cutoff, len(uses)):
            return False
        chosen = [uses[i] for i in spread(len(uses), config.context_max_samples)]
        tag_of = await asyncio.to_thread(self._catalog.tag_lookup)
        blocks = [
            await asyncio.to_thread(
                context_of_use, self._services, message_id, moment, cutoff, tag_of=tag_of
            )
            for message_id, moment in chosen
        ]
        blocks = [block for block in blocks if block]
        if not blocks:
            return False
        schema = reply_model(ContextReply, self._catalog.vocabulary)
        messages = self.template(STICKER_CONTEXT).render(
            vocabulary=self._vocabulary_text(), uses=uses_text(blocks, ConsistentRedactor())
        )
        result = await self._client.chat_json(
            messages,
            schema,
            purpose=Purpose.STICKER_TAG,
            images=[self.image_of(record)],
            tag=tag,
            temperature=0.2,
            max_tokens=CONTEXT_COMPLETION_TOKENS * 4,
        )
        reply = result.value
        await asyncio.to_thread(
            self._catalog.save_context,
            md5,
            reply.tags,
            one_line(reply.meaning),
            uses=len(uses),
            cutoff=cutoff,
            at=self._services.clock.now_utc(),
        )
        return True
