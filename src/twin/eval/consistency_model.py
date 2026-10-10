"""What the consistency audit works with: the evidence, the model's answer, the checks (R-EVAL-004).

Once a week DeepSeek is given what the bot has said about "her" lately - the life line of the last
days, her own words about herself in the replies, and the facts that concern her, each with its
source and the time it became known - and asked which of them cannot be true together.  This module
is the part that needs no model and no database:

* :class:`Evidence` is one numbered record handed over (``L3`` a life line entry, ``R12`` a turn of
  the bot's replies, ``F5`` a fact), :class:`EvidencePack` all of them;
* :class:`ConsistencyOut` is the JSON the model must answer with.  It is checked twice: by the
  schema (types, enums, lengths; a reply that does not fit is sent back once by the client,
  R-LLM-003) and then by :func:`validate_output`, which holds the answer to the pack - a
  contradiction that names a record nobody was shown, the same record twice, or a pair already
  reported is dropped and counted, never stored;
* a :class:`Finding` is a contradiction that survived the checks, with a fingerprint of the two
  records it is about (so that what the user decided is remembered);
* :func:`judge_audit` is the rule of the requirement: at most one *obvious* contradiction a week,
  confirmed by the user.

The text of a statement is never taken from the model's wording alone: a quotation is kept only if
it really occurs in the record it names, otherwise the record's own text is shown.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from twin.clock import ensure_aware
from twin.memory.api import SOURCE_PRIORITY
from twin.memory.conflict import normalised

Verdict = Literal["passed", "failed", "insufficient"]
Severity = Literal["obvious", "minor"]
EvidenceKind = Literal["lifeline", "reply", "fact"]

MAX_OBVIOUS_PER_WEEK = 1  # R-EVAL-004: "明显矛盾 ≤ 1 次/周" (pinned to the SPEC text by a test)
WEEK_DAYS = 7
MAX_FINDINGS = 12  # the model is asked for at most this many; more in an answer are cut
QUOTE_CHARS = 300
_REF = re.compile(r"^([LRF])(\d{1,4})$")
KIND_OF_PREFIX: dict[str, EvidenceKind] = {"L": "lifeline", "R": "reply", "F": "fact"}
KIND_LABELS = {"lifeline": "生活安排", "reply": "她说过的话", "fact": "记忆里的事实"}

DROPPED_UNKNOWN_REF = "unknown_ref"
DROPPED_SAME_RECORD = "same_record"
DROPPED_REPEAT = "repeat"
DROPPED_OVER_LIMIT = "over_limit"


class ConsistencyError(RuntimeError):
    """The audit cannot be made or its result cannot be used."""


# ------------------------------------------------------------------------- the evidence


@dataclass(frozen=True)
class Evidence:
    """One numbered record the model is shown."""

    ref: str
    kind: EvidenceKind
    item_id: str  # life line entry id, id of the first bubble of the turn, fact id
    at: datetime  # when it was said or happens (a fact: when it became known)
    text: str
    source: str | None = None  # fact: real_record | user_said | bot_invented | user_command
    known_at: datetime | None = None  # fact: when it became known; entry: when it was written
    number: int | None = None  # the number of a fact, which is how the user names it
    message_ids: tuple[str, ...] = ()  # reply: every bubble of the turn

    @property
    def bot_made(self) -> bool:
        return is_bot_made(self)


@dataclass(frozen=True)
class EvidencePack:
    """Everything one audit was given, in the order it was numbered."""

    window_start: datetime
    window_end: datetime
    zone: str
    days: int
    items: tuple[Evidence, ...] = ()
    reply_turns_cut: int = 0  # older turns left out because of the size limit

    def get(self, ref: str) -> Evidence | None:
        return next((item for item in self.items if item.ref == ref), None)

    def of(self, kind: EvidenceKind) -> tuple[Evidence, ...]:
        return tuple(item for item in self.items if item.kind == kind)

    @property
    def empty(self) -> bool:
        """Nothing was said and nothing was planned: there is no one to contradict."""
        return not self.of("lifeline") and not self.of("reply")

    @property
    def chars(self) -> int:
        return sum(len(item.text) for item in self.items)


def ref_kind(ref: str) -> EvidenceKind | None:
    """``lifeline`` for ``L3`` and so on; ``None`` for anything that is not a record number."""
    found = _REF.match(ref)
    return KIND_OF_PREFIX[found.group(1)] if found else None


# ------------------------------------------------------------------------ the answer


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class StatementOut(_Strict):
    ref: str = Field(pattern=_REF.pattern)
    quote: str | None = Field(default=None, max_length=QUOTE_CHARS)

    @field_validator("quote")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None


class ContradictionOut(_Strict):
    time: str = Field(min_length=1, max_length=40)
    first: StatementOut
    second: StatementOut
    related: list[str] = Field(default_factory=list, max_length=8)
    severity: Severity
    reason: str = Field(min_length=1, max_length=200)
    keep: str | None = Field(default=None, max_length=8)
    rewrite: str | None = Field(default=None, max_length=120)

    @field_validator("keep", "rewrite")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return value or None


class ConsistencyOut(_Strict):
    """The JSON the audit prompt asks for."""

    contradictions: list[ContradictionOut] = Field(default_factory=list, max_length=40)


# ------------------------------------------------------------------------ who may be wrong


class Record(Protocol):
    """What :func:`rank_of` and :func:`losers_of` need to know about a record or a statement."""

    @property
    def ref(self) -> str: ...

    @property
    def kind(self) -> EvidenceKind: ...

    @property
    def source(self) -> str | None: ...


def is_bot_made(record: Record) -> bool:
    """What the bot made up or said: a life line entry, a reply, a fact it invented.

    These are the only records an audit may suggest changing; what comes from the real chat, from
    the user's words or from his ``/记住`` is never touched (R-MEM-004: a lower source does not
    override a higher one).
    """
    return record.kind != "fact" or record.source == "bot_invented"


def rank_of(record: Record) -> int:
    """The rank of the source of a record (R-MEM-004); the bot's own words rank lowest."""
    if record.kind == "fact":
        return SOURCE_PRIORITY.get(record.source or "", SOURCE_PRIORITY["bot_invented"])
    return SOURCE_PRIORITY["bot_invented"]


def losers_of[R: Record](first: R, second: R, keep: str | None) -> tuple[R, ...]:
    """The side or sides that give way when two records contradict each other.

    The side with the lower source gives way (R-MEM-004).  Between equals the model's ``keep``
    decides; with no answer, what the bot already *said* stays (a sent reply cannot be changed)
    and the other side gives way; if that does not decide either, both are candidates and the
    user chooses.
    """
    low, high = rank_of(first), rank_of(second)
    if low != high:
        return (first,) if low < high else (second,)
    if keep == first.ref:
        return (second,)
    if keep == second.ref:
        return (first,)
    changeable = [side for side in (first, second) if side.kind != "reply"]
    return (changeable[0],) if len(changeable) == 1 else (first, second)


# --------------------------------------------------------------------- the checked result


@dataclass(frozen=True)
class Statement:
    """One side of a contradiction, with what the user needs to recognise it."""

    ref: str
    kind: EvidenceKind
    item_id: str
    text: str
    source: str | None
    at: datetime
    known_at: datetime | None = None
    number: int | None = None
    message_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "item_id": self.item_id,
            "text": self.text,
            "source": self.source,
            "at": self.at.isoformat(),
            "known_at": self.known_at.isoformat() if self.known_at else None,
            "number": self.number,
            "message_ids": list(self.message_ids),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Statement:
        known = data.get("known_at")
        return cls(
            ref=str(data["ref"]),
            kind=data["kind"],
            item_id=str(data["item_id"]),
            text=str(data["text"]),
            source=data.get("source"),
            at=ensure_aware(datetime.fromisoformat(str(data["at"]))),
            known_at=ensure_aware(datetime.fromisoformat(str(known))) if known else None,
            number=data.get("number"),
            message_ids=tuple(str(i) for i in data.get("message_ids", [])),
        )


@dataclass(frozen=True)
class Finding:
    """A contradiction that passed the checks (not yet shown to the user)."""

    time_text: str
    at: datetime
    first: Statement
    second: Statement
    related: tuple[Statement, ...]
    severity: Severity
    reason: str
    keep: str | None
    rewrite: str | None
    fingerprint: str

    @property
    def statements(self) -> tuple[Statement, ...]:
        return (self.first, self.second, *self.related)

    def to_payload(self) -> dict[str, Any]:
        """The sealed document stored with the finding."""
        return {
            "time": self.time_text,
            "first": self.first.to_json(),
            "second": self.second.to_json(),
            "related": [item.to_json() for item in self.related],
            "reason": self.reason,
            "keep": self.keep,
            "rewrite": self.rewrite,
        }


def fingerprint_of(first: Statement | Evidence, second: Statement | Evidence) -> str:
    """Identifies the pair of records, whichever is named first (a decision is remembered by it)."""
    parts = sorted(f"{item.kind}:{item.item_id}" for item in (first, second))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def statement_of(item: Evidence, quote: str | None = None) -> Statement:
    """The statement made of a record; the model's quotation is kept only if it is in the record."""
    quoted = quote.strip() if quote else ""
    genuine = bool(normalised(quoted)) and normalised(quoted) in normalised(item.text)
    return Statement(
        ref=item.ref,
        kind=item.kind,
        item_id=item.item_id,
        text=(quoted if genuine else item.text)[: 2 * QUOTE_CHARS],
        source=item.source,
        at=item.at,
        known_at=item.known_at,
        number=item.number,
        message_ids=item.message_ids,
    )


@dataclass
class Validation:
    """The contradictions that survived and how many did not (and why)."""

    findings: list[Finding] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())


def validate_output(out: ConsistencyOut, pack: EvidencePack) -> Validation:
    """Hold the model's answer to the pack: only records that were shown, no repeats.

    A contradiction whose ``first`` or ``second`` names a record that is not in the pack, or
    names the same record twice, or repeats a pair already reported, is dropped.  An unknown
    ``related`` number, a ``keep`` that is neither side and a ``rewrite`` for a record the audit
    may not change only lose that one detail.
    """
    result = Validation()
    seen: set[str] = set()
    for number, item in enumerate(out.contradictions):
        if number >= MAX_FINDINGS:
            result.dropped[DROPPED_OVER_LIMIT] += 1
            continue
        first, second = pack.get(item.first.ref), pack.get(item.second.ref)
        if first is None or second is None:
            result.dropped[DROPPED_UNKNOWN_REF] += 1
            continue
        if first.item_id == second.item_id and first.kind == second.kind:
            result.dropped[DROPPED_SAME_RECORD] += 1
            continue
        fingerprint = fingerprint_of(first, second)
        if fingerprint in seen:
            result.dropped[DROPPED_REPEAT] += 1
            continue
        seen.add(fingerprint)
        sides = {first.ref, second.ref}
        extra = [
            found
            for ref in dict.fromkeys(item.related)
            if ref not in sides and (found := pack.get(ref)) is not None
        ]
        keep = item.keep if item.keep in sides else None
        losers = losers_of(first, second, keep)
        rewritable = len(losers) == 1 and losers[0].kind != "reply" and is_bot_made(losers[0])
        left = statement_of(first, item.first.quote)
        right = statement_of(second, item.second.quote)
        result.findings.append(
            Finding(
                time_text=item.time,
                at=max(first.at, second.at),
                first=left,
                second=right,
                related=tuple(statement_of(found) for found in extra),
                severity=item.severity,
                reason=item.reason,
                keep=keep,
                rewrite=item.rewrite if rewritable else None,
                fingerprint=fingerprint,
            )
        )
    return result


# ---------------------------------------------------------------------------- the rule


def allowed_obvious(days: int) -> Fraction:
    """How many obvious contradictions a window of ``days`` days may hold: one a week."""
    return Fraction(MAX_OBVIOUS_PER_WEEK * days, WEEK_DAYS)


def judge_audit(*, days: int, confirmed_obvious: int, undecided: int) -> tuple[Verdict, str]:
    """The verdict of R-EVAL-004 and the sentence that says why.

    A window shorter than a week cannot show a weekly rate, and an audit whose contradictions the
    user has not all decided is not finished: both are "not enough yet".
    """
    if days < WEEK_DAYS:
        return "insufficient", f"审阅范围只有 {days} 天，不足一周，不能判断“每周明显矛盾 ≤ 1 次”"
    if undecided:
        return "insufficient", f"还有 {undecided} 条矛盾没有确认：twin eval consistency --review"
    limit = allowed_obvious(days)
    if confirmed_obvious > limit:
        return (
            "failed",
            f"确认的明显矛盾 {confirmed_obvious} 次，超过每周 {MAX_OBVIOUS_PER_WEEK} 次的上限"
            f"（{days} 天内最多 {float(limit):.3g} 次）",
        )
    return (
        "passed",
        f"确认的明显矛盾 {confirmed_obvious} 次（{days} 天内最多 {float(limit):.3g} 次）",
    )
