"""Turning a conversation into facts and follow-ups: the fact extractor (R-MEM-006, R-MEM-007).

:class:`FactExtractor` shows DeepSeek a stretch of conversation (real messages, or the bot's own
conversation) and gets back, as JSON checked against :mod:`twin.memory.schemas`, the facts worth
keeping and the things worth asking about later.  What it returns is a list of *drafts*; storing
them, with the conflict rules, is :mod:`twin.memory.writer`.

What the extractor itself guarantees, whatever the model says:

* **evidence** - every fact and follow-up cites lines of the dialogue; a citation of a line that
  does not exist drops the item.  The fact's ``known_at`` is the time of the *latest* line it
  cites (R-MEM-010), so a fact is never known earlier than the messages that prove it.
* **source** (R-MEM-004) - real messages give ``real_record``; in the bot's conversation what the
  user said gives ``user_said`` and what the bot said about *herself* gives ``bot_invented``
  (what the bot says about the user or about third parties is only an echo and is dropped); the
  user's ``/记住`` gives ``user_command``.
* **time** - "明天下午三点考试" is read against the time of the evidence message *in the zone of
  the conversation* (hers for real records, the bot's for its own conversation): the model is
  told that moment, and the date it names is checked with :func:`twin.memory.timeparse.parse_when`,
  whose answer wins whenever it can read the phrase.  The same words mean different instants in
  Chicago and in Beijing.
* **size** - a long conversation is cut into chunks of ``memory.replay_chunk_lines`` lines, each a
  call of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import LedgerTag, Purpose
from twin.memory.localdate import MemoryClock
from twin.memory.records import FollowupRecord
from twin.memory.render import WEEKDAY_NAMES
from twin.memory.schemas import (
    ClosedFollowup,
    ExtractedFact,
    ExtractedFollowup,
    ExtractionOut,
    LifelineHint,
)
from twin.memory.timeparse import parse_iso_date, parse_local_iso, parse_when
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import MEMORY_EXTRACT, MEMORY_EXTRACT_BOT, TemplateStore
from twin.services import Services

log = get_logger("twin.memory.extract")

Mode = Literal["real", "bot", "command"]
Speaker = Literal["her", "user", "bot"]

MAX_FACTS_PER_CALL = 12  # facts asked for in one extraction call
MAX_FACT_CHARS = 200
MAX_FOLLOWUP_CHARS = 160
DEFAULT_WINDOW_MINUTES = 240
DATE_ONLY_WINDOW_MINUTES = 720
MIN_WINDOW_MINUTES = 30
MAX_WINDOW_MINUTES = 2880
DEFAULT_DUE_CLOCK = time(9, 0)
MAX_OPEN_FOLLOWUPS_SHOWN = 10
SPEAKER_LABELS = {"her": "她", "bot": "她", "user": "对方"}
SOURCE_OF_MODE = {"real": "real_record", "bot": "user_said", "command": "user_command"}
EVIDENCE_KIND = {"real": "messages", "bot": "bot_turns", "command": "command"}
FOLLOWUP_ORIGIN = {"real": "real_record", "bot": "bot_session", "command": "user_command"}


def _stamp(moment: datetime) -> str:
    """``2026-03-05 21:10（周四）``: the local time the model reads the dialogue against."""
    return f"{moment:%Y-%m-%d %H:%M}（{WEEKDAY_NAMES[moment.weekday()]}）"


@dataclass(frozen=True)
class DialogueLine:
    """One line of a conversation: where it comes from, who said it, what, and when."""

    ref: str  # id of the message, or of the bot turn
    speaker: Speaker
    text: str
    at: datetime


@dataclass(frozen=True)
class LifelineDraft:
    activity: str
    place: str | None = None
    mood: str | None = None
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class FactDraft:
    """A fact found in a conversation, not stored yet."""

    subject: str
    category: str
    text: str
    importance: int
    confidence: float
    source: str
    known_at: datetime
    evidence_kind: str
    evidence_refs: tuple[str, ...]
    event_date: date | None = None
    recurrence: str = "none"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    lifeline: LifelineDraft | None = None

    def evidence(self) -> dict[str, object]:
        return {"kind": self.evidence_kind, "ids": list(self.evidence_refs)}


@dataclass(frozen=True)
class FollowupDraft:
    text: str
    due_at: datetime
    window_minutes: int
    created_at: datetime  # when the commitment became known (the latest evidence line)
    origin: str
    evidence_kind: str
    evidence_refs: tuple[str, ...]
    source_turn_id: str | None = None

    def evidence(self) -> dict[str, object]:
        return {"kind": self.evidence_kind, "ids": list(self.evidence_refs)}


@dataclass(frozen=True)
class ClosedDraft:
    followup_id: str
    reason: str
    at: datetime


@dataclass
class Extraction:
    """Everything one extraction found."""

    facts: list[FactDraft] = field(default_factory=list)
    followups: list[FollowupDraft] = field(default_factory=list)
    closed: list[ClosedDraft] = field(default_factory=list)
    dropped: int = 0  # items the model returned that could not be used
    calls: int = 0
    cost_usd: float = 0.0

    def merge(self, other: Extraction) -> None:
        self.facts.extend(other.facts)
        self.followups.extend(other.followups)
        self.closed.extend(other.closed)
        self.dropped += other.dropped
        self.calls += other.calls
        self.cost_usd += other.cost_usd


class FactExtractor:
    """Extracts facts and follow-ups from dialogue lines (see the module description)."""

    def __init__(
        self,
        services: Services,
        client: DeepSeekClient,
        clock: MemoryClock,
        *,
        templates: TemplateStore | None = None,
    ) -> None:
        self._services = services
        self._client = client
        self._clock = clock
        self._templates = templates or TemplateStore(services.db, services.clock)

    @property
    def enabled(self) -> bool:
        """``memory.fact_extraction``: with it off nothing is sent to DeepSeek for facts."""
        return self._services.settings.memory.fact_extraction

    # ------------------------------------------------------------------ extraction

    async def extract(
        self,
        lines: Sequence[DialogueLine],
        *,
        mode: Mode,
        tag: LedgerTag,
        open_followups: Sequence[FollowupRecord] = (),
    ) -> Extraction:
        """Facts, follow-ups and closed follow-ups of ``lines`` (oldest first)."""
        result = Extraction()
        if not lines or not self.enabled:
            return result
        size = self._services.settings.memory.replay_chunk_lines
        still_open = list(open_followups)
        for start in range(0, len(lines), size):
            chunk = lines[start : start + size]
            part = await self._extract_chunk(chunk, mode, tag, still_open)
            closed_ids = {c.followup_id for c in part.closed}
            still_open = [f for f in still_open if f.id not in closed_ids]
            result.merge(part)
        return result

    async def _extract_chunk(
        self,
        lines: Sequence[DialogueLine],
        mode: Mode,
        tag: LedgerTag,
        open_followups: Sequence[FollowupRecord],
    ) -> Extraction:
        zone = self._zone_of(mode, lines[0].at)
        shown = list(open_followups)[:MAX_OPEN_FOLLOWUPS_SHOWN]
        template = self._templates.active(MEMORY_EXTRACT if mode == "real" else MEMORY_EXTRACT_BOT)
        messages = template.render(
            max_facts=MAX_FACTS_PER_CALL,
            zone=zone.key,
            reference=_stamp(lines[0].at.astimezone(zone)),
            open_followups=self._render_followups(shown, zone),
            count=len(lines),
            dialogue=self._render_dialogue(lines, zone),
        )
        reply = await self._client.chat_json(
            messages, ExtractionOut, purpose=Purpose.EXTRACT, tag=tag
        )
        out = self.interpret(reply.value, lines, mode, shown)
        out.calls = reply.attempts
        out.cost_usd = reply.total_cost_usd
        return out

    def _zone_of(self, mode: Mode, moment: datetime) -> ZoneInfo:
        return self._clock.real_zone(moment) if mode == "real" else self._clock.bot_zone()

    @staticmethod
    def _render_dialogue(lines: Sequence[DialogueLine], zone: ZoneInfo) -> str:
        return "\n".join(
            f"{number}. [{line.at.astimezone(zone):%m-%d %H:%M}] "
            f"{SPEAKER_LABELS[line.speaker]}：{line.text}"
            for number, line in enumerate(lines, start=1)
        )

    @staticmethod
    def _render_followups(followups: Sequence[FollowupRecord], zone: ZoneInfo) -> str:
        if not followups:
            return "（没有）"
        return "\n".join(
            f"F{number}：{f.text}（预定 {f.due_at.astimezone(zone):%Y-%m-%d %H:%M}）"
            for number, f in enumerate(followups, start=1)
        )

    # ------------------------------------------------------------- interpretation

    def interpret(
        self,
        out: ExtractionOut,
        lines: Sequence[DialogueLine],
        mode: Mode,
        shown_followups: Sequence[FollowupRecord] = (),
    ) -> Extraction:
        """Check the model's answer against the dialogue and turn it into drafts.

        Everything that is wrong with one item - a citation of a missing line, a speaker the
        mode does not allow, a time that cannot be read - drops that item and counts it in
        ``dropped``; the rest of the answer is kept.
        """
        result = Extraction()
        for item in out.facts:
            draft = self._fact(item, lines, mode)
            if draft is None:
                result.dropped += 1
            else:
                result.facts.append(draft)
        for follow in out.followups:
            followup = self._followup(follow, lines, mode)
            if followup is None:
                result.dropped += 1
            else:
                result.followups.append(followup)
        for closing in out.closed_followups:
            closed = self._closed(closing, lines, shown_followups)
            if closed is None:
                result.dropped += 1
            else:
                result.closed.append(closed)
        if result.dropped:
            log.info("extraction_items_dropped", dropped=result.dropped, kept=len(result.facts))
        return result

    @staticmethod
    def _cited(evidence: Sequence[int], lines: Sequence[DialogueLine]) -> list[DialogueLine] | None:
        """The lines an item cites; ``None`` if it cites a line that does not exist."""
        cited: list[DialogueLine] = []
        for number in dict.fromkeys(evidence):
            if not 1 <= number <= len(lines):
                return None
            cited.append(lines[number - 1])
        return cited or None

    def _fact(
        self, item: ExtractedFact, lines: Sequence[DialogueLine], mode: Mode
    ) -> FactDraft | None:
        cited = self._cited(item.evidence, lines)
        if cited is None:
            return None
        source = SOURCE_OF_MODE[mode]
        if mode == "bot":
            if item.speaker == "bot":
                if item.subject not in ("her", "both"):
                    return None  # what the bot says about the user or others is only an echo
                source = "bot_invented"
            elif item.speaker != "user":
                return None
        text = " ".join(item.text.split())[:MAX_FACT_CHARS]
        if len(text) < 2:
            return None
        known_at = max(line.at for line in cited)
        zone = self._zone_of(mode, known_at)
        event_date = self._event_date(item, known_at, zone)
        recurrence = item.recurrence if event_date is not None else "none"
        hint = item.lifeline if source == "bot_invented" else None
        return FactDraft(
            subject=item.subject,
            category=item.category,
            text=text,
            importance=item.importance,
            confidence=1.0 if mode == "command" else item.confidence,
            source=source,
            known_at=known_at,
            evidence_kind=EVIDENCE_KIND[mode],
            evidence_refs=tuple(line.ref for line in cited),
            event_date=event_date,
            recurrence=recurrence,
            valid_from=self._bound(item.valid_from, zone, end=False),
            valid_to=self._bound(item.valid_to, zone, end=True),
            lifeline=self._lifeline(hint),
        )

    @staticmethod
    def _lifeline(hint: LifelineHint | None) -> LifelineDraft | None:
        if hint is None:
            return None
        return LifelineDraft(hint.activity, hint.place, hint.mood, hint.start, hint.end)

    @staticmethod
    def _event_date(item: ExtractedFact, known_at: datetime, zone: ZoneInfo) -> date | None:
        """The day the fact names: the phrase read against the evidence time, else the ISO date."""
        reference = known_at.astimezone(zone)
        if item.event_phrase:
            when = parse_when(item.event_phrase, reference)
            if when is not None and when.day_given:
                return when.day
        return parse_iso_date(item.event_date) if item.event_date else None

    @staticmethod
    def _bound(text: str | None, zone: ZoneInfo, *, end: bool) -> datetime | None:
        day = parse_iso_date(text) if text else None
        if day is None:
            return None
        if end:
            day += timedelta(days=1)
        moment = datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)
        return moment - timedelta(microseconds=1) if end else moment

    def _followup(
        self, item: ExtractedFollowup, lines: Sequence[DialogueLine], mode: Mode
    ) -> FollowupDraft | None:
        cited = self._cited(item.evidence, lines)
        if cited is None:
            return None
        created = max(line.at for line in cited)
        zone = self._zone_of(mode, created)
        reference = created.astimezone(zone)
        due: datetime | None = None
        date_only = False
        when = parse_when(item.due, reference) if item.due else None
        if when is not None and when.day_given:
            due = when.at(zone, default=DEFAULT_DUE_CLOCK)
            date_only = not when.has_time
        elif item.due_local:
            parsed = parse_local_iso(item.due_local, zone)
            if parsed is not None:
                due = parsed
                date_only = len(item.due_local.strip()) <= len("YYYY-MM-DD")
                if date_only:
                    due = datetime.combine(parsed.date(), DEFAULT_DUE_CLOCK, tzinfo=zone)
        if due is None or due.astimezone(UTC) < created:
            return None  # no readable time, or a time that was already past when it was said
        window = item.window_minutes if item.window_minutes is not None else DEFAULT_WINDOW_MINUTES
        if date_only:
            window = max(window, DATE_ONLY_WINDOW_MINUTES)
        window = min(MAX_WINDOW_MINUTES, max(MIN_WINDOW_MINUTES, window))
        return FollowupDraft(
            text=" ".join(item.text.split())[:MAX_FOLLOWUP_CHARS],
            due_at=due.astimezone(UTC),
            window_minutes=window,
            created_at=created,
            origin=FOLLOWUP_ORIGIN[mode],
            evidence_kind=EVIDENCE_KIND[mode],
            evidence_refs=tuple(line.ref for line in cited),
            source_turn_id=cited[-1].ref if mode == "bot" else None,
        )

    @staticmethod
    def _closed(
        item: ClosedFollowup,
        lines: Sequence[DialogueLine],
        shown: Sequence[FollowupRecord],
    ) -> ClosedDraft | None:
        ref = item.ref.strip().upper().removeprefix("F")
        if not ref.isdigit() or not 1 <= int(ref) <= len(shown):
            return None
        cited = FactExtractor._cited(item.evidence, lines) if item.evidence else None
        at = max(line.at for line in cited) if cited else lines[-1].at
        return ClosedDraft(shown[int(ref) - 1].id, item.reason, at)
