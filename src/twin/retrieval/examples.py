"""Examples: what a retrieved window looks like, and how it is shown to a model (R-RET-005).

An :class:`Example` is the structured form handed to the prompt builder (round 09), the
evaluation sandbox (round 09b) and the training export (round 13): the turns of the context
(who spoke, what) and her real reply block, one :class:`ExampleLine` per message.

**Event lines (R-SAFE-006).**  Every reply line carries ``reproducible`` (the third-round
:func:`~twin.ingest.events.is_reproducible`).  A line the bot could not have written - a
picture, a voice message, a call, a transfer - is not a line to imitate: :func:`render_example`
shows it as a note, "（此处她发了：<event text>）", in the place where she sent it.  Pictures
carry their description when there is one; a historic picture without a description reads
``[图片]`` and a description job has been queued (``get_caption(wait=False)``, R-IMP-012), so
building examples never waits for the network.

:func:`render_example` is the one place that turns an example into prompt text; the online
prompt (round 09) and the evaluation sandbox (round 09b) call it with the same arguments, so a
model sees examples the same way in both.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from twin.ingest.captions import CaptionService
from twin.ingest.events import Kind, is_reproducible, render_event_text
from twin.ops.logging import get_logger
from twin.retrieval.records import MessageData, load_messages
from twin.retrieval.texts import STICKER_TEXT, one_line
from twin.retrieval.windows import WindowRecord
from twin.storage.chat_models import MediaAsset

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.retrieval.examples")

ID_CHUNK = 800
QUOTE_CHARS = 40
DAY_TYPE_NAMES = {"workday": "工作日", "weekend": "周末", "holiday": "节假日"}

StickerLabeler = Callable[[str], str | None]


@dataclass(frozen=True)
class ExampleLine:
    """One message of an example."""

    text: str  # what was written; for a line that is not reproducible, its event text
    kind: str  # the message kind (text, sticker, quote, image, call, ...)
    reproducible: bool  # False: the bot could not have sent this line (R-SAFE-006)
    quoted: str | None = None  # the text a quote reply quotes


@dataclass(frozen=True)
class ExampleTurn:
    """A merged turn of the context: one speaker, one or more lines."""

    her: bool
    lines: tuple[ExampleLine, ...]


@dataclass(frozen=True)
class Example:
    """A window as the model sees it: the context before her reply, and her reply block."""

    window_id: str
    reply_at: datetime
    local_slot: int
    day_type: str
    context: tuple[ExampleTurn, ...]
    reply: tuple[ExampleLine, ...]
    similarity: float = 0.0
    score: float = 0.0

    @property
    def clock(self) -> str:
        minutes = self.local_slot * 15
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @property
    def reproducible_lines(self) -> tuple[ExampleLine, ...]:
        return tuple(line for line in self.reply if line.reproducible)


@dataclass(frozen=True)
class ExampleLabels:
    """The words around an example in a prompt (round 09 may override them)."""

    user: str = "对方"
    her: str = "她"
    reply: str = "她的回复"
    event_note: str = "（此处她发了：{event}）"
    heading: str = "例子{number}（{clock} 左右，{day_type}）"


DEFAULT_LABELS = ExampleLabels()


def render_example(
    example: Example, *, number: int | None = None, labels: ExampleLabels = DEFAULT_LABELS
) -> str:
    """The prompt text of one example (R-RET-005, R-SAFE-006).

    Context lines are ``<who>：<text>`` (event texts as they are); the reply lines follow under
    the reply label, one per line, with every line that is not reproducible replaced by the
    note "（此处她发了：…）" so it is context, not something to imitate.
    """
    day_type = DAY_TYPE_NAMES.get(example.day_type, example.day_type)
    heading = labels.heading.format(
        number=f" {number}" if number is not None else "", clock=example.clock, day_type=day_type
    )
    out = [heading]
    for turn in example.context:
        who = labels.her if turn.her else labels.user
        for line in turn.lines:
            if line.quoted:
                out.append(f"{who}：[引用:{line.quoted}]")
            out.append(f"{who}：{line.text}")
    out.append(f"{labels.reply}：")
    for line in example.reply:
        if not line.reproducible:
            out.append(labels.event_note.format(event=line.text))
            continue
        if line.quoted:
            out.append(f"[引用:{line.quoted}]")
        out.append(line.text)
    return "\n".join(out)


def render_examples(examples: Sequence[Example], *, labels: ExampleLabels = DEFAULT_LABELS) -> str:
    """Several examples, numbered from 1 and separated by a blank line."""
    return "\n\n".join(
        render_example(example, number=index, labels=labels)
        for index, example in enumerate(examples, start=1)
    )


# ---------------------------------------------------------------------- building


def quoted_text(message: MessageData) -> str | None:
    """The short text a quote reply refers to."""
    if message.kind != Kind.QUOTE.value:
        return None
    quote = message.quote or {}
    source = quote.get("quoteContent") or quote.get("quoteTitle")
    if not isinstance(source, str):
        return None
    snippet = one_line(source)
    if not snippet:
        return None
    return snippet if len(snippet) <= QUOTE_CHARS else snippet[: QUOTE_CHARS - 1] + "…"


@dataclass
class ExampleBuilder:
    """Turns window rows into :class:`Example` objects (messages, descriptions, labels)."""

    services: Services
    captions: CaptionService | None = None
    sticker_label: StickerLabeler | None = None
    _owned: bool = field(default=False, init=False)

    def _captions(self) -> CaptionService:
        if self.captions is None:
            self.captions = CaptionService(self.services)
            self._owned = True
        return self.captions

    async def aclose(self) -> None:
        if self._owned and self.captions is not None:
            await self.captions.aclose()
            self.captions = None
            self._owned = False

    # -------------------------------------------------------------- data access

    def _load(self, ids: Sequence[str]) -> tuple[dict[str, MessageData], dict[str, str]]:
        """Messages by id and the picture asset of each picture message."""
        wanted = sorted(set(ids))
        assets: dict[str, str] = {}
        with self.services.db.session() as session:
            found = load_messages(session, wanted)
            for start in range(0, len(wanted), ID_CHUNK):
                rows = session.execute(
                    select(MediaAsset.message_id, MediaAsset.id).where(
                        MediaAsset.kind == "image",
                        MediaAsset.message_id.in_(wanted[start : start + ID_CHUNK]),
                    )
                )
                for message_id, asset_id in rows:
                    assets.setdefault(str(message_id), str(asset_id))
        return found, assets

    async def _descriptions(self, assets: dict[str, str]) -> dict[str, str]:
        """Stored picture descriptions; a missing one queues a job and is left out (R-IMP-012)."""
        if not assets:
            return {}
        service = self._captions()
        order = sorted(assets)
        results = await asyncio.gather(
            *(service.get_caption(assets[message_id], wait=False) for message_id in order)
        )
        return {mid: text for mid, text in zip(order, results, strict=True) if text}

    # ------------------------------------------------------------------ lines

    def _line(self, message: MessageData, descriptions: dict[str, str]) -> ExampleLine:
        kind = message.kind
        if is_reproducible(message):
            if kind == Kind.STICKER.value:
                label = None
                if self.sticker_label and message.sticker_md5:
                    label = self.sticker_label(message.sticker_md5)
                text = f"[表情包:{label}]" if label else STICKER_TEXT
                return ExampleLine(text, kind, True)
            return ExampleLine(message.text or "", kind, True, quoted_text(message))
        event = render_event_text(message, caption=descriptions.get(message.id))
        return ExampleLine(event or "", kind, False)

    # ------------------------------------------------------------------ build

    async def build(
        self, windows: Sequence[WindowRecord], scores: dict[str, tuple[float, float]]
    ) -> list[Example]:
        """Examples for ``windows`` in the given order; ``scores[id] = (similarity, score)``.

        A window whose reply messages are gone or are not hers is left out (and logged), so a
        damaged row can never put the user's words or the bot's into an example.
        """
        ids = [
            message_id
            for window in windows
            for message_id in (
                *window.reply_block_ids,
                *(i for turn in window.context_block_ids for i in turn),
            )
        ]
        found, assets = await asyncio.to_thread(self._load, ids)
        descriptions = await self._descriptions(
            {mid: asset for mid, asset in assets.items() if mid in found}
        )
        examples: list[Example] = []
        for window in windows:
            reply_rows = [found[i] for i in window.reply_block_ids if i in found]
            if any(row.is_sent for row in reply_rows):
                log.error("window_reply_not_hers", window=window.id)
                continue
            if not reply_rows:
                log.warning("window_reply_missing", window=window.id)
                continue
            turns = []
            for turn_ids in window.context_block_ids:
                members = [found[i] for i in turn_ids if i in found]
                if members:
                    turns.append(
                        ExampleTurn(
                            members[0].her,
                            tuple(self._line(m, descriptions) for m in members),
                        )
                    )
            similarity, score = scores.get(window.id, (0.0, 0.0))
            examples.append(
                Example(
                    window_id=window.id,
                    reply_at=window.reply_at_utc,
                    local_slot=window.local_slot,
                    day_type=window.day_type,
                    context=tuple(turns),
                    reply=tuple(self._line(m, descriptions) for m in reply_rows),
                    similarity=similarity,
                    score=score,
                )
            )
        return examples
