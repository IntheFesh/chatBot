"""Her reply blocks and the conversation before each of them, for the training export (R-TRN-002).

A **reply block** is one of her bursts (the unit of :mod:`twin.retrieval.windows`); its
**context** is the merged turns of both sides that precede it in the same conversation segment,
at most eight (the same number the live prompt keeps).  The blocks are found with the same
:class:`~twin.retrieval.windows.WindowAssembler` the retrieval library uses, so the training set
and the example library cut the conversation at the same places, and the messages are read only
through :mod:`twin.ingest.corpus` (R-STO-007: ``bot_turns`` is a different table the exporter
never opens).

**Only her real messages (R-TRN-004).**  :func:`block_from_rows` is the one door a block enters
through.  Its reply must consist of rows of the ``messages`` table that she sent: any other object
- a row of ``bot_turns``, a preference pair - is a :class:`TypeError`, a message of the user in
the reply is a :class:`~twin.retrieval.records.NotHerMessageError`.  The context may hold the
user's messages (they are what she answered) but again only ``messages`` rows.

**Nothing from the future (R-TRN-013).**  A block is refused when a context message is later
than the first message of the reply (:class:`LeakError`); the assembler cannot produce one, and
the check says so loudly if that ever changes.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from twin.engine.style_prompt import STYLE_CONTEXT_TURNS
from twin.ingest.corpus import conversation_skeleton, messages_by_ids
from twin.retrieval.records import MessageData, require_her_message, require_message
from twin.retrieval.windows import WindowAssembler, WindowDraft, window_id_of
from twin.storage.chat_models import Message

if TYPE_CHECKING:
    from twin.services import Services

ID_CHUNK = 800
SKELETON_ROWS = 5000
CACHE_ROWS = 30_000


class LeakError(RuntimeError):
    """A context message is later than the reply it belongs to."""


@dataclass(frozen=True)
class BlockRef:
    """A reply block and its context as message ids (no text)."""

    sample_id: str
    conversation_id: str
    reply_ids: tuple[str, ...]
    context_ids: tuple[tuple[str, ...], ...]
    reply_at: datetime
    reproducible: int

    @classmethod
    def of(cls, draft: WindowDraft) -> BlockRef:
        return cls(
            sample_id=window_id_of(draft.reply_ids[0]),
            conversation_id=draft.conversation_id,
            reply_ids=draft.reply_ids,
            context_ids=draft.context_ids,
            reply_at=draft.reply_at,
            reproducible=draft.reproducible,
        )

    @property
    def message_ids(self) -> tuple[str, ...]:
        return (*self.reply_ids, *(i for turn in self.context_ids for i in turn))


@dataclass(frozen=True)
class LoadedMessage:
    """A message of the conversation with the moment it was sent."""

    data: MessageData
    at: datetime

    @classmethod
    def from_row(cls, row: object) -> LoadedMessage:
        message = require_message(row)
        return cls(MessageData.from_row(message), message.create_time_utc.astimezone(UTC))


@dataclass(frozen=True)
class LoadedBlock:
    """A reply block with the text of its messages and of the context before it."""

    sample_id: str
    reply: tuple[LoadedMessage, ...]
    context: tuple[tuple[LoadedMessage, ...], ...]

    @property
    def at(self) -> datetime:
        """The moment of the first message of the reply: ``t`` of every as-of read."""
        return self.reply[0].at

    @classmethod
    def create(
        cls,
        sample_id: str,
        reply: Sequence[LoadedMessage],
        context: Sequence[Sequence[LoadedMessage]],
    ) -> LoadedBlock:
        if not reply:
            raise ValueError("a reply block needs at least one message")
        if any(message.data.is_sent for message in reply):
            raise ValueError("a message sent by the user cannot be part of a training target")
        moment = reply[0].at
        for turn in context:
            for message in turn:
                if message.at > moment:
                    raise LeakError("a message after the reply block was found in its context")
        return cls(sample_id, tuple(reply), tuple(tuple(turn) for turn in context if turn))


def block_from_rows(
    reply: Sequence[Message], context: Sequence[Sequence[Message]] = ()
) -> LoadedBlock:
    """A block from rows of the ``messages`` table (R-TRN-004; the entry of the SFT export).

    ``reply`` must be her messages; an object of any other type (a ``bot_turns`` row, a
    preference pair, ...) raises :class:`TypeError`, a message of the user raises
    :class:`~twin.retrieval.records.NotHerMessageError`.
    """
    her = [require_her_message(row) for row in reply]
    others = [[require_message(row) for row in turn] for turn in context]
    if not her:
        raise ValueError("a reply block needs at least one message")
    return LoadedBlock.create(
        window_id_of(her[0].id),
        [LoadedMessage.from_row(row) for row in her],
        [[LoadedMessage.from_row(row) for row in turn] for turn in others],
    )


# --------------------------------------------------------------------------- reading


def iter_block_refs(
    services: Services, *, since: datetime | None = None, until: datetime | None = None
) -> Iterator[BlockRef]:
    """Her reply blocks in time order (``since <= first message < until``) with their context."""
    profile = services.settings.profile
    assembler = WindowAssembler(
        profile.burst_gap_s, profile.segment_gap_min * 60.0, STYLE_CONTEXT_TURNS
    )

    def wanted(draft: WindowDraft) -> bool:
        return (since is None or draft.reply_at >= since) and (
            until is None or draft.reply_at < until
        )

    with services.db.session() as session:
        stream = session.scalars(conversation_skeleton().execution_options(yield_per=SKELETON_ROWS))
        for message in stream:
            draft = assembler.add(message)
            if draft is not None and wanted(draft):
                yield BlockRef.of(draft)
    last = assembler.finish()
    if last is not None and wanted(last):
        yield BlockRef.of(last)


class MessageLoader:
    """Loads the text of messages by id, keeping the most recent ones (contexts overlap)."""

    def __init__(self, services: Services, *, capacity: int = CACHE_ROWS) -> None:
        self._services = services
        self._capacity = capacity
        self._cache: OrderedDict[str, LoadedMessage] = OrderedDict()

    def _fetch(self, ids: Sequence[str]) -> dict[str, LoadedMessage]:
        fetched: dict[str, LoadedMessage] = {}
        with self._services.db.session() as session:
            for start in range(0, len(ids), ID_CHUNK):
                for row in session.scalars(messages_by_ids(ids[start : start + ID_CHUNK])):
                    fetched[row.id] = LoadedMessage.from_row(row)
        self._cache.update(fetched)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return fetched

    def load(self, refs: Iterable[BlockRef]) -> list[LoadedBlock]:
        """The blocks of ``refs`` with their messages (one database pass for the whole batch)."""
        batch = list(refs)
        wanted = {i for ref in batch for i in ref.message_ids}
        held = {i: self._cache[i] for i in wanted if i in self._cache}
        held.update(self._fetch(sorted(wanted - held.keys())))
        for i in wanted & self._cache.keys():
            self._cache.move_to_end(i)
        blocks: list[LoadedBlock] = []
        for ref in batch:
            if not all(i in held for i in ref.message_ids):
                raise LookupError("a message of a reply block is not in the messages table")
            blocks.append(
                LoadedBlock.create(
                    ref.sample_id,
                    [held[i] for i in ref.reply_ids],
                    [[held[i] for i in turn] for turn in ref.context_ids],
                )
            )
        return blocks
