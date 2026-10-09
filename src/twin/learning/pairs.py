"""Preference pairs: the user's wording against the bot's reply (R-LRN-002, R-LRN-004).

``/不像 <正确说法>`` stores ``(prompt_sample, chosen, rejected)``.  The **prompt sample** is the
structured form of the situation - the system segment and the conversation turns - made by the
functions that make the style model's prompt (``StylePromptBuilder.compose``: the turns come out
of :func:`~twin.engine.style_prompt.normalize_context`, so they open with the user and alternate,
like the samples of the training set).  It is **not** a rendered prompt: a rendered string would be
wrapped in the chat template a second time when LLaMA-Factory trains on it (R-TRN-011), so
:class:`PromptSample` refuses any text that holds a template marker.

``rejected`` is the bot's own text.  It may be a negative example in a DPO export and nowhere else:
this module is imported by the commands, the learning and :mod:`twin.training.dpo_export`, and by
no code that builds a style sample, a retrieval library or an SFT set (R-LRN-004; a test reads the
imports of those packages).

The system segment is a record of the moment (and of the persona card and template it was made
with); the DPO export makes it again for the versions an adapter is locked to when they differ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from twin.clock import Clock, ensure_aware
from twin.engine.turns import TurnRecord
from twin.storage.db import Database
from twin.storage.learning_models import PreferencePair
from twin.training import lf_template
from twin.training.lf_template import Turn

SOURCE_USER_CORRECTION = "user_correction"
SAMPLE_SCHEMA = 1
MAX_CORRECTION_CHARS = 400
NO_PERSONA = "none"


class PairError(ValueError):
    """A preference pair or its prompt sample is not a valid one."""


def clean_wording(text: str) -> str:
    """The way the user says it: lines trimmed, blank lines dropped, no template markers.

    A multi-line wording is her burst of messages (one line each), as in the training targets.
    """
    lines = []
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        for token in (*lf_template.CONTROL_TOKENS, *lf_template.LF_SLOTS):
            line = line.replace(token, "")
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def reply_text(rows: Sequence[TurnRecord]) -> str:
    """A reply of the bot as one text: a bubble per line, a sticker as ``[表情包:<标签>]``."""
    lines: list[str] = []
    for row in rows:
        if row.kind == "sticker":
            tag = " ".join(row.text.split()).replace("]", "")
            lines.append(f"[表情包:{tag}]" if tag else "[表情包]")
        else:
            line = clean_wording(row.text)
            if line:
                lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptSample:
    """The situation of a reply as a structured sample (see the module description).

    ``turns`` open with the user, alternate and end with the user's turn that was answered; the
    texts hold no ChatML marker and no LLaMA-Factory slot.  ``at`` is the moment of the reply;
    ``prelude`` are her turns that opened the conversation part and went into the system segment
    (kept so the segment can be made again for another persona card or template).
    """

    system: str
    turns: tuple[Turn, ...]
    at: datetime | None = None
    prelude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            lf_template.check_alternation(self.turns)
            lf_template.check_content(self.system)
            for turn in self.turns:
                if not turn.content.strip():
                    raise lf_template.TemplateError("a turn of a sample has no text")
                lf_template.check_content(turn.content)
            for opening in self.prelude:
                lf_template.check_content(opening)
        except lf_template.TemplateError as exc:
            raise PairError(f"not a structured prompt sample: {exc}") from None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SAMPLE_SCHEMA,
            "system": self.system,
            "turns": [{"role": turn.role, "content": turn.content} for turn in self.turns],
            "at": self.at.isoformat() if self.at is not None else None,
            "prelude": list(self.prelude),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PromptSample:
        if data.get("schema") != SAMPLE_SCHEMA:
            raise PairError("the prompt sample is of an unknown schema")
        system, raw_turns = data.get("system"), data.get("turns")
        if not isinstance(system, str) or not isinstance(raw_turns, list):
            raise PairError("the prompt sample needs a system segment and a list of turns")
        turns: list[Turn] = []
        for item in raw_turns:
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            if role not in ("user", "assistant") or not isinstance(content, str):
                raise PairError("a turn of the prompt sample is malformed")
            turns.append(Turn(role, content))
        stamp = data.get("at")
        opening = data.get("prelude") or []
        if not isinstance(opening, list) or not all(isinstance(item, str) for item in opening):
            raise PairError("the prelude of the prompt sample is malformed")
        return cls(
            system,
            tuple(turns),
            datetime.fromisoformat(stamp) if stamp else None,
            tuple(opening),
        )


@dataclass(frozen=True)
class PairRecord:
    """One row of ``preference_pairs``, decrypted."""

    id: str
    feedback_id: str | None
    reply_id: str
    sample: PromptSample
    chosen: str
    rejected: str
    source: str
    template_version: str
    persona_version: str
    created_at: datetime


def _record(row: PreferencePair) -> PairRecord:
    return PairRecord(
        id=row.id,
        feedback_id=row.feedback_id,
        reply_id=row.reply_id,
        sample=PromptSample.from_json(row.prompt_sample),
        chosen=row.chosen,
        rejected=row.rejected,
        source=row.source,
        template_version=row.template_version,
        persona_version=row.persona_version,
        created_at=ensure_aware(row.created_at),
    )


class PreferencePairStore:
    """Writes and reads ``preference_pairs``."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def add(
        self,
        *,
        reply_id: str,
        sample: PromptSample,
        chosen: str,
        rejected: str,
        template_version: str,
        persona_version: str,
        feedback_id: str | None = None,
    ) -> tuple[PairRecord, bool]:
        """Store a pair; the same wording for the same reply is stored once (``created`` False)."""
        wording = clean_wording(chosen)[:MAX_CORRECTION_CHARS]
        bad = clean_wording(rejected)
        if not wording or not bad:
            raise PairError("a preference pair needs both a preferred and a rejected text")
        if wording == bad:
            raise PairError("the preferred wording is the reply that was rejected")
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            for existing in session.scalars(
                select(PreferencePair).where(PreferencePair.reply_id == reply_id)
            ):
                if existing.chosen == wording:
                    return _record(existing), False
            row = PreferencePair(
                feedback_id=feedback_id,
                reply_id=reply_id,
                source=SOURCE_USER_CORRECTION,
                template_version=template_version,
                persona_version=persona_version,
                created_at=now,
                updated_at=now,
            )
            row.prompt_sample = sample.to_json()
            row.chosen = wording
            row.rejected = bad
            session.add(row)
            session.flush()
            return _record(row), True

    def all(self) -> list[PairRecord]:
        """Every pair, oldest first."""
        with self._db.session() as session:
            rows = session.scalars(
                select(PreferencePair).order_by(PreferencePair.created_at, PreferencePair.id)
            )
            return [_record(row) for row in rows]

    def for_reply(self, reply_id: str) -> list[PairRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(PreferencePair)
                .where(PreferencePair.reply_id == reply_id)
                .order_by(PreferencePair.created_at, PreferencePair.id)
            )
            return [_record(row) for row in rows]

    def count(self) -> int:
        with self._db.session() as session:
            return int(session.scalar(select(func.count()).select_from(PreferencePair)) or 0)


def dpo_hint(pairs: int, minimum: int) -> str | None:
    """The line that says there are enough pairs for DPO (R-TRN-012), or ``None``."""
    if pairs < minimum:
        return None
    return f"偏好对已有 {pairs} 对（至少 {minimum} 对），可以做 DPO：先 twin train export-dpo"
