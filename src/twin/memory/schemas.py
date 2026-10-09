"""The JSON the memory prompts ask DeepSeek for, validated with pydantic (R-LLM-003).

A reply that does not fit is sent back to the model once with the list of what is wrong; a
second failure is a failed task.  Fitting the schema is the first check only: the extractor then
drops what cannot be trusted (evidence lines that do not exist, dates that do not parse).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Subject = Literal["her", "user", "both", "other"]
Category = Literal[
    "life", "preference", "plan", "anniversary", "nickname", "relation", "work_study", "other"
]
Recurrence = Literal["none", "yearly", "monthly"]
Relation = Literal["same", "update", "conflict", "unrelated"]

SUMMARY_MAX_CHARS = 300
_CLOCK = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class LifelineHint(_Lenient):
    """What the bot said she did, when the fact describes a stretch of her day."""

    activity: str = Field(min_length=1, max_length=120)
    place: str | None = Field(default=None, max_length=60)
    mood: str | None = Field(default=None, max_length=40)
    start: str | None = None
    end: str | None = None

    @field_validator("start", "end")
    @classmethod
    def _clock(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not _CLOCK.match(value):
            return None  # a time that cannot be read is left out, not a reason to drop the fact
        hours, _, minutes = value.partition(":")
        return f"{int(hours):02d}:{minutes}"


class ExtractedFact(_Lenient):
    speaker: Literal["her", "user", "bot"] | None = None
    subject: Subject
    category: Category = "other"
    text: str = Field(min_length=2, max_length=400)
    importance: int = Field(default=3, ge=1, le=5)
    evidence: list[int] = Field(min_length=1)
    event_date: str | None = None
    event_phrase: str | None = None
    recurrence: Recurrence = "none"
    confidence: float = Field(default=0.8, ge=0, le=1)
    valid_from: str | None = None
    valid_to: str | None = None
    lifeline: LifelineHint | None = None


class ExtractedFollowup(_Lenient):
    text: str = Field(min_length=2, max_length=300)
    due: str = ""
    due_local: str | None = None
    window_minutes: int | None = Field(default=None, ge=0, le=100_000)
    evidence: list[int] = Field(min_length=1)


class ClosedFollowup(_Lenient):
    ref: str = Field(min_length=1, max_length=20)
    reason: Literal["done", "cancelled"] = "done"
    evidence: list[int] = Field(default_factory=list)


class ExtractionOut(_Lenient):
    facts: list[ExtractedFact] = Field(default_factory=list)
    followups: list[ExtractedFollowup] = Field(default_factory=list)
    closed_followups: list[ClosedFollowup] = Field(default_factory=list)


class Verdict(_Lenient):
    candidate: str = Field(min_length=1, max_length=20)
    relation: Relation


class Judgement(_Lenient):
    new: str = Field(min_length=1, max_length=20)
    verdicts: list[Verdict] = Field(default_factory=list)


class ConflictOut(_Lenient):
    judgements: list[Judgement] = Field(default_factory=list)


class SummaryOut(_Lenient):
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
