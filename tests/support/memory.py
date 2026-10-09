"""Helpers for the memory tests: build a memory, put facts, summaries and follow-ups into it.

Everything here is synthetic.  The dates are in March 2026, when Chicago is on standard time
until the 8th (UTC-6) and on daylight time after (UTC-5): tests that need an exact instant say
so in UTC.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from tests.support.deepseek import error, ok, request_json
from twin.clock import Clock
from twin.ingest.times import SourceTime
from twin.memory.extract import FactDraft
from twin.memory.localdate import MemoryClock
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord
from twin.memory.store import NewEvent, NewFact, NewFollowup
from twin.retrieval.embedder import EmbeddingService
from twin.schedule.time_service import ConfiguredTimeService
from twin.services import Services
from twin.storage.memory_models import BOT_CONVERSATION_SOURCES

CHICAGO = "America/Chicago"


def utc(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_memory(services: Services, embedder: EmbeddingService | None = None) -> Memory:
    return Memory(services, embedder=embedder)


def add_fact(
    memory: Memory,
    text: str,
    known_at: datetime,
    *,
    source: str = "real_record",
    subject: str = "her",
    category: str = "life",
    importance: int = 3,
    confidence: float = 0.8,
    event_date: date | None = None,
    recurrence: str = "none",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    status: str = "active",
    superseded_by: str | None = None,
    superseded_at: datetime | None = None,
    evidence: dict[str, object] | None = None,
    embed: bool = True,
) -> FactRecord:
    record = memory.store.add_fact(
        NewFact(
            subject=subject,
            category=category,
            text=text,
            source=source,
            known_at=known_at,
            confidence=confidence,
            importance=importance,
            valid_from=valid_from,
            valid_to=valid_to,
            event_date=event_date,
            recurrence=recurrence,
            evidence=evidence if evidence is not None else {"kind": "messages", "ids": ["m1"]},
            status=status,
            superseded_by=superseded_by,
            superseded_at=superseded_at,
        )
    )
    if source in BOT_CONVERSATION_SOURCES:  # what the writer does: the bot's conversation exists
        memory.store.mark_bot_online(known_at)
    if embed and status == "active":
        memory.sync_vectors(facts=[record])
    else:
        memory.refresh()
    return record


def add_summary(
    memory: Memory,
    scope: str,
    day: date,
    text: str,
    *,
    zone: str = CHICAGO,
    embed: bool = True,
) -> SummaryRecord:
    if scope == "real":
        start, end = memory.clock.real_bounds(day)
    else:
        start, end = memory.clock.bot_bounds(day)
    record = memory.store.add_summary(
        scope=scope,
        local_date=day,
        timezone=zone,
        utc_start=start,
        utc_end=end,
        text=text,
        input_hash=None,
        template_version=None,
    )
    if embed:
        memory.sync_vectors(summaries=[record])
    else:
        memory.refresh()
    return record


def add_followup(
    memory: Memory,
    text: str,
    due_at: datetime,
    created_at: datetime,
    *,
    window_minutes: int = 240,
    origin: str = "real_record",
    close_at: datetime | None = None,
    status: str = "done",
) -> FollowupRecord:
    record = memory.store.add_followup(
        NewFollowup(
            text=text,
            due_at=due_at,
            window_minutes=window_minutes,
            created_at=created_at,
            origin=origin,
        )
    )
    if close_at is not None:
        closed = memory.store.close_followup(record.id, status=status, at=close_at, reason="test")
        assert closed is not None
        record = closed
    memory.refresh()
    return record


def add_event(
    memory: Memory,
    day: date,
    activity: str,
    *,
    start: str | None = None,
    end: str | None = None,
    source: str = "plan",
    created_at: datetime | None = None,
) -> LifelineRecord:
    record = memory.store.add_event(
        NewEvent(
            local_date=day,
            timezone=CHICAGO,
            activity=activity,
            source=source,
            start_local=start,
            end_local=end,
        ),
        at=created_at,
    )
    memory.refresh()
    return record


def days_after(moment: datetime, days: float) -> datetime:
    return moment + timedelta(days=days)


def memory_clock(clock: Clock, zone: str = CHICAGO) -> MemoryClock:
    """A memory calendar with her zone and the bot's zone both set to ``zone``."""
    return MemoryClock(SourceTime(zone), ConfiguredTimeService(clock, lambda: zone))


def fact_record(**fields: object) -> FactRecord:
    """A fact record with sensible defaults; override what the test is about."""
    base: dict[str, object] = {
        "id": "F1",
        "rev": 1,
        "number": 1,
        "subject": "her",
        "category": "life",
        "text": "她喜欢吃火锅",
        "source": "real_record",
        "status": "active",
        "confidence": 0.8,
        "importance": 3,
        "known_at": utc(2026, 3, 1),
        "valid_from": None,
        "valid_to": None,
        "event_date": None,
        "recurrence": "none",
        "superseded_by": None,
        "superseded_at": None,
        "rejected_by": None,
        "evidence": None,
        "embedding_id": None,
        "embed_version": None,
        "created_at": utc(2026, 3, 1),
        "updated_at": utc(2026, 3, 1),
    }
    base.update(fields)
    return FactRecord(**base)  # type: ignore[arg-type]


def summary_record(**fields: object) -> SummaryRecord:
    base: dict[str, object] = {
        "id": "S1",
        "rev": 1,
        "scope": "real",
        "local_date": date(2026, 3, 1),
        "timezone": CHICAGO,
        "utc_start": utc(2026, 3, 1, 6),
        "utc_end": utc(2026, 3, 2, 6),
        "text": "两个人聊了考试",
        "version": 1,
        "is_current": True,
        "embedding_id": None,
        "embed_version": None,
        "input_hash": None,
        "created_at": utc(2026, 3, 2),
        "updated_at": utc(2026, 3, 2),
    }
    base.update(fields)
    return SummaryRecord(**base)  # type: ignore[arg-type]


def followup_record(**fields: object) -> FollowupRecord:
    base: dict[str, object] = {
        "id": "U1",
        "text": "她明天考试",
        "due_at": utc(2026, 3, 5, 21),
        "window_minutes": 240,
        "source_turn_id": None,
        "status": "open",
        "created_at": utc(2026, 3, 4),
        "closed_at": None,
        "close_reason": None,
        "origin": "real_record",
        "fact_id": None,
    }
    base.update(fields)
    return FollowupRecord(**base)  # type: ignore[arg-type]


def fact_draft(text: str = "她喜欢吃火锅", **fields: object) -> FactDraft:
    """A fact draft as the extractor would hand it over."""
    base: dict[str, object] = {
        "subject": "her",
        "category": "life",
        "text": text,
        "importance": 3,
        "confidence": 0.8,
        "source": "real_record",
        "known_at": utc(2026, 3, 1),
        "evidence_kind": "messages",
        "evidence_refs": ("m1",),
    }
    base.update(fields)
    return FactDraft(**base)  # type: ignore[arg-type]


LINE = re.compile(r"^(\d+)\. \[", re.MULTILINE)
NEW_ITEM = re.compile(r"^新信息 n(\d+)（[^）]*）：(.*)$", re.MULTILINE)
OLD_ITEM = re.compile(r"^  旧条目 ([cl]\d+)（[^）]*）：(.*)$", re.MULTILINE)


def line_number(dialogue: str, marker: str) -> int | None:
    """The number of the first dialogue line that contains ``marker`` (1-based)."""
    for line in dialogue.splitlines():
        match = re.match(r"^(\d+)\. \[", line)
        if match and marker in line:
            return int(match.group(1))
    return None


@dataclass
class FactRule:
    """When a dialogue contains ``marker``, the model answers with this fact."""

    marker: str
    fields: dict[str, Any]
    also: tuple[str, ...] = ()  # more markers whose lines are cited as evidence too


@dataclass
class FollowupRule:
    marker: str
    fields: dict[str, Any]


@dataclass
class CloseRule:
    marker: str  # a line of the dialogue that shows the follow-up is over
    followup_text: str  # the open follow-up it closes (matched by text in the prompt)
    reason: str = "done"


@dataclass
class ScriptedMemoryModel:
    """A respx side effect that answers the memory prompts from rules the test sets.

    It reads the prompt it is given (the numbered dialogue, the pairs of a conflict question,
    the date of a summary) and answers like the real model would, in the real JSON shape.
    Everything it was asked is in :attr:`requests`; :attr:`calls` counts by kind.
    """

    facts: list[FactRule] = field(default_factory=list)
    followups: list[FollowupRule] = field(default_factory=list)
    closes: list[CloseRule] = field(default_factory=list)
    relations: dict[tuple[str, str], str] = field(default_factory=dict)  # (new, old) markers
    summaries: dict[str, str] = field(default_factory=dict)  # date -> text
    default_summary: str = "这一天他们聊了日常。"
    prompt_tokens: int = 200
    completion_tokens: int = 80
    fail_from_call: int | None = None  # a 400 from this call on (1-based)
    on_request: Callable[[dict[str, Any]], None] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    calls: Counter[str] = field(default_factory=Counter)

    def relate(self, new_marker: str, old_marker: str, relation: str) -> None:
        self.relations[(new_marker, old_marker)] = relation

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        self.requests.append(body)
        if self.on_request is not None:
            self.on_request(body)
        if self.fail_from_call is not None and len(self.requests) >= self.fail_from_call:
            return error(400, "scripted failure")
        messages = body["messages"]
        system, user = messages[0]["content"], messages[1]["content"]
        if "记忆整理员" in system:
            kind, reply = "extract", self._extraction(user)
        elif "记忆管理员" in system:
            kind, reply = "conflict", self._conflict(user)
        elif "摘要员" in system and "合并成" in system:
            kind, reply = "summary_merge", {"summary": "合并后的一天摘要。"}
        elif "摘要员" in system:
            kind, reply = "summary", self._summary(user)
        else:
            raise AssertionError("the model was asked something the memory never asks")
        self.calls[kind] += 1
        return ok(
            content=json.dumps(reply, ensure_ascii=False),
            prompt=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            hit=0,
        )

    # -- the answers ------------------------------------------------------------

    def _extraction(self, prompt: str) -> dict[str, Any]:
        dialogue = prompt
        facts: list[dict[str, Any]] = []
        for rule in self.facts:
            number = line_number(dialogue, rule.marker)
            if number is None:
                continue
            evidence = [number]
            for extra in rule.also:
                more = line_number(dialogue, extra)
                if more is not None:
                    evidence.append(more)
            facts.append({**rule.fields, "evidence": evidence})
        follows: list[dict[str, Any]] = []
        for follow in self.followups:
            number = line_number(dialogue, follow.marker)
            if number is not None:
                follows.append({**follow.fields, "evidence": [number]})
        closed: list[dict[str, Any]] = []
        for close in self.closes:
            number = line_number(dialogue, close.marker)
            ref = re.search(rf"^F(\d+)：{re.escape(close.followup_text)}", prompt, re.MULTILINE)
            if number is not None and ref:
                closed.append(
                    {"ref": f"F{ref.group(1)}", "reason": close.reason, "evidence": [number]}
                )
        return {"facts": facts, "followups": follows, "closed_followups": closed}

    def _conflict(self, prompt: str) -> dict[str, Any]:
        judgements = []
        blocks = re.split(r"\n\n(?=新信息)", prompt.strip())
        for block in blocks:
            new = NEW_ITEM.search(block)
            if new is None:
                continue
            verdicts = []
            for key, text in OLD_ITEM.findall(block):
                for (new_marker, old_marker), relation in self.relations.items():
                    if new_marker in new.group(2) and old_marker in text:
                        verdicts.append({"candidate": key, "relation": relation})
            judgements.append({"new": f"n{new.group(1)}", "verdicts": verdicts})
        return {"judgements": judgements}

    def _summary(self, prompt: str) -> dict[str, Any]:
        match = re.search(r"日期：(\d{4}-\d{2}-\d{2})", prompt)
        day = match.group(1) if match else ""
        return {"summary": self.summaries.get(day, self.default_summary)}
