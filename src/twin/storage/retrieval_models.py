"""The example windows of the retrieval library (round 05; R-RET-001, R-STO-006).

``example_windows`` holds, for each of her reply blocks, *where* the example is: the ids of the
messages of the reply and of the (up to six) merged turns before it, when she replied (UTC),
the local 15-minute slot and day type of that moment, and whether the window belongs to the
held-out evaluation set.  No text is stored here or in the vector index: the words are looked up
by id in ``messages`` (where they are sealed) when an example is rendered.

``embed_version`` names the encoding (model, dimension, weights hash, text scheme) the window's
vector in the LanceDB table was made with; ``NULL`` means "no vector yet" (held-out, no context,
or waiting to be encoded).  ``signature`` is a hash of the id lists, so a re-import that changes
which messages make up a window is noticed and only that window is encoded again (R-RET-006).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, CheckConstraint, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from twin.storage.models import Base, TimestampMixin
from twin.storage.types import UTCDateTime

DAY_TYPES = ("workday", "weekend", "holiday")
WINDOW_ID_LENGTH = 24


class ExampleWindow(TimestampMixin, Base):
    """One of her reply blocks with the conversation just before it."""

    __tablename__ = "example_windows"

    id: Mapped[str] = mapped_column(String(WINDOW_ID_LENGTH), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("conversations.id"), nullable=False
    )
    reply_block_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    context_block_ids: Mapped[list[list[str]]] = mapped_column(JSON, nullable=False)
    context_turns: Mapped[int] = mapped_column(Integer, nullable=False)
    reply_reproducible: Mapped[int] = mapped_column(Integer, nullable=False)
    reply_at_utc: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    local_slot: Mapped[int] = mapped_column(Integer, nullable=False)
    day_type: Mapped[str] = mapped_column(String(8), nullable=False)
    holdout: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    signature: Mapped[str] = mapped_column(String(40), nullable=False)
    embed_version: Mapped[str | None] = mapped_column(String(200), nullable=True)

    __table_args__ = (
        CheckConstraint("local_slot >= 0 AND local_slot < 96", name="local_slot"),
        CheckConstraint(
            "day_type IN (" + ", ".join(repr(v) for v in DAY_TYPES) + ")", name="day_type"
        ),
        Index("ix_example_windows_reply_at_utc", "reply_at_utc"),
        Index("ix_example_windows_holdout_embed_version", "holdout", "embed_version"),
    )
