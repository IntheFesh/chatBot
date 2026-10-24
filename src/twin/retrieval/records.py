"""Her real messages as the retrieval library handles them (R-RET-004, R-STO-007).

Two things live here:

* the **guards** that keep everything but real rows of the ``messages`` table out of the library:
  :func:`require_message` (a :class:`TypeError` for any other object - the bot's own turns of
  round 09 are such objects) and :func:`require_her_message` (a reply member must be hers);
* :class:`MessageData`, a plain copy of the columns the library needs.  A row read in a
  database session is expired when the session ends, so text is copied out while it is open.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from twin.ingest.corpus import messages_by_ids
from twin.storage.chat_models import Message

ID_CHUNK = 800


class NotHerMessageError(ValueError):
    """A reply member is not one of her real messages (R-RET-004)."""


def require_message(candidate: object) -> Message:
    """The ``messages`` row itself, or :class:`TypeError` (anything else cannot enter)."""
    if not isinstance(candidate, Message):
        raise TypeError(
            "only rows of the messages table can enter the retrieval library, "
            f"not {type(candidate).__name__}"
        )
    return candidate


def require_her_message(candidate: object) -> Message:
    """A real message of hers (``is_sent`` false), or :class:`NotHerMessageError`."""
    message = require_message(candidate)
    if message.is_sent:
        raise NotHerMessageError("a message sent by the user cannot be the reply of a window")
    return message


@dataclass(frozen=True, slots=True)
class MessageData:
    """The columns of a message that rendering and encoding read (decrypted)."""

    id: str
    kind: str
    is_sent: bool
    text: str | None
    quote: dict[str, Any] | None
    sticker_md5: str | None
    call_status: str | None
    call_duration_s: int | None
    voice_seconds: int | None
    has_transcript: bool

    @classmethod
    def from_row(cls, row: object) -> MessageData:
        """Copy a ``messages`` row (any other object is a :class:`TypeError`)."""
        message = require_message(row)
        return cls(
            id=message.id,
            kind=message.kind,
            is_sent=message.is_sent,
            text=message.text,
            quote=message.quote,
            sticker_md5=message.sticker_md5,
            call_status=message.call_status,
            call_duration_s=message.call_duration_s,
            voice_seconds=message.voice_seconds,
            has_transcript=message.has_transcript,
        )

    @property
    def her(self) -> bool:
        return not self.is_sent


def load_messages(session: Session, ids: Sequence[str]) -> dict[str, MessageData]:
    """The messages with these ids (ids that are not rows of ``messages`` are simply absent)."""
    wanted = sorted(set(ids))
    found: dict[str, MessageData] = {}
    for start in range(0, len(wanted), ID_CHUNK):
        for row in session.scalars(messages_by_ids(wanted[start : start + ID_CHUNK])):
            found[row.id] = MessageData.from_row(row)
    return found
