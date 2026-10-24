"""Collecting the evidence of one audit (R-EVAL-004): the last days of the life line, the bot's
words about herself, the facts that concern her.

All of it is read through the doors the memory offers - :func:`~twin.memory.api.memory_view` (the
memory as it stands at the moment of the audit: what is invalidated, replaced or not yet known is
not in it) and the reader of the bot's conversation (:func:`~twin.memory.recent.bot_turn_reader`,
which leaves out commands and thrown-away replies) - and never as a query of the tables.  Nothing
is written.

* **life line** - the entries that still count (not invalidated) of the local days from the start
  of the window to today, in time order, numbered ``L1``..;
* **replies** - the turns (consecutive bubbles) the bot sent in the window, newest first until
  ``eval.consistency_reply_chars`` characters are reached, then numbered in time order ``R1``..
  (the user's own messages are not handed over: the audit is about what *she* says);
* **facts** - the facts about her (``her`` or ``both``) that are visible now: those that something
  in the window is close to (the memory's own search by meaning and keyword; without the
  embedding model the keyword index alone answers), the ones of the highest importance (where she
  lives, studies, works: what an invention contradicts) and the ones the bot invented since the
  window began (the likeliest to clash with what it says next); at most
  ``eval.consistency_facts``, numbered ``F1``..

The bot's replies are read here for *evaluation only*: they are shown to the model and, if the user
confirms a contradiction, quoted in the finding.  They go nowhere else - not into the style samples,
the retrieval library or the training set (CLAUDE.md rule 7).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from twin.eval.consistency_model import Evidence, EvidencePack
from twin.memory.api import CORE_IMPORTANCE, Memory, memory_view, minutes_of
from twin.memory.asof import WEEKDAY_NAMES
from twin.memory.conflict import SOURCE_NAMES
from twin.memory.recent import Turn, bot_turn_reader, merge_turns
from twin.memory.records import FactRecord, LifelineRecord
from twin.memory.view import MemoryView
from twin.retrieval.embedder import EmbedderError
from twin.schedule.service import time_service_for
from twin.services import Services

MAX_TURN_CHARS = 400  # one turn of the replies is cut to this
MIN_RELATED_SCORE = 0.2  # the share of a record's keyword weight a fact must have
RELATED_PER_RECORD = 3
MAX_QUERIES = 120  # facts are searched for this many records at most: the life line, then replies
CORE_BONUS = 0.1
RECENT_BOT_BONUS = 0.15  # an invention of the window outranks an equally close older fact
HER_SUBJECTS = ("her", "both")
END_SLACK = timedelta(seconds=1)  # ``messages_between`` excludes its end; the audit runs "now"


def local_stamp(moment: datetime, zone: ZoneInfo) -> str:
    """``MM-DD HH:MM`` on the bot's wall clock."""
    return f"{moment.astimezone(zone):%m-%d %H:%M}"


def one_line(text: str) -> str:
    """The bubbles of a turn on one line, separated by a slash."""
    return " / ".join(line.strip() for line in text.splitlines() if line.strip())


def lifeline_text(entry: LifelineRecord) -> str:
    """The date, the weekday and the entry as the prompt shows it."""
    detail = f"；{entry.detail}" if entry.detail and entry.detail != entry.activity else ""
    day = f"{entry.local_date:%m-%d}（{WEEKDAY_NAMES[entry.local_date.weekday()]}）"
    return f"{day} {entry.line()}{detail}"


def lifeline_moment(entry: LifelineRecord, zone: ZoneInfo) -> datetime:
    """When the entry begins (the start of its day when it has no readable start time)."""
    minute = minutes_of(entry.start_local)
    clock = time(0, 0) if minute is None else time(minute // 60, minute % 60)
    return datetime.combine(entry.local_date, clock, tzinfo=zone).astimezone(UTC)


def _lifeline_items(
    entries: Sequence[LifelineRecord], zone: ZoneInfo, first_day: date, last_day: date
) -> list[Evidence]:
    inside = [e for e in entries if first_day <= e.local_date <= last_day]
    inside.sort(key=lambda e: (e.local_date, minutes_of(e.start_local) or 0, e.created_at, e.id))
    return [
        Evidence(
            ref=f"L{number}",
            kind="lifeline",
            item_id=entry.id,
            at=lifeline_moment(entry, zone),
            text=lifeline_text(entry),
            source=entry.source,
            known_at=entry.created_at,
        )
        for number, entry in enumerate(inside, start=1)
    ]


def _what_the_bot_said(services: Services, start: datetime, end: datetime) -> list[Turn]:
    reader = bot_turn_reader(services)
    if reader is None:
        return []
    messages = reader.messages_between(start, end + END_SLACK)
    return [turn for turn in merge_turns(messages) if turn.role == "bot"]


def _reply_items(
    turns: Sequence[Turn], zone: ZoneInfo, max_chars: int
) -> tuple[list[Evidence], int]:
    """The newest turns that fit in ``max_chars``, oldest first, and how many were left out."""
    kept: list[Turn] = []
    used = 0
    for turn in reversed(turns):
        size = min(len(turn.text), MAX_TURN_CHARS)
        if kept and used + size > max_chars:
            break
        kept.append(turn)
        used += size
    kept.reverse()
    items = [
        Evidence(
            ref=f"R{number}",
            kind="reply",
            item_id=turn.id,
            at=turn.at,
            text=one_line(turn.text)[:MAX_TURN_CHARS],
            message_ids=turn.message_ids,
        )
        for number, turn in enumerate(kept, start=1)
    ]
    return items, len(turns) - len(kept)


def _related_facts(memory: Memory, view: MemoryView, queries: Sequence[str]) -> dict[str, float]:
    """``{fact id: best closeness}`` of the visible facts close to any of the queries.

    The memory's own search (by meaning and by keyword); if the embedding model cannot be loaded
    the keyword index alone answers, so the audit does not depend on it.
    """
    found: dict[str, float] = {}
    for query in queries[:MAX_QUERIES]:
        try:
            scores = {
                hit.fact.id: hit.relevance for hit in view.search_facts(query, RELATED_PER_RECORD)
            }
        except EmbedderError:
            index = memory.corpus.fact_index
            scores = {
                hit.doc_id: hit.score
                for hit in index.search(query, RELATED_PER_RECORD, allowed=view.fact_visible)
                if hit.score >= MIN_RELATED_SCORE
            }
        for fact_id, score in scores.items():
            if score > found.get(fact_id, 0.0):
                found[fact_id] = score
    return found


def _fact_items(
    memory: Memory,
    view: MemoryView,
    queries: Sequence[str],
    *,
    since: datetime,
    limit: int,
) -> list[Evidence]:
    """The facts about her worth setting against the rest, best first, at most ``limit``.

    A fact is in if something in the window is close to it, if the bot invented it since the
    window began (the likeliest to clash with what it says next) or if it is one of the facts of
    the highest importance (where she lives, studies, works: the ones an invention contradicts).
    """
    if limit <= 0:
        return []
    visible = {fact.id: fact for fact in view.facts() if fact.subject in HER_SUBJECTS}
    fresh = [f for f in visible.values() if f.source == "bot_invented" and f.known_at >= since]
    related = _related_facts(memory, view, [*(f.text for f in fresh), *queries])
    scored: list[tuple[float, FactRecord]] = []
    for fact in visible.values():
        core = fact.importance >= CORE_IMPORTANCE
        if fact.id not in related and fact not in fresh and not core:
            continue
        score = related.get(fact.id, 0.0)
        score += RECENT_BOT_BONUS if fact in fresh else 0.0
        score += CORE_BONUS if core else 0.0
        scored.append((score + fact.importance / 100, fact))
    scored.sort(key=lambda pair: (-pair[0], pair[1].number))
    chosen = sorted((fact for _, fact in scored[:limit]), key=lambda f: (f.known_at, f.number))
    return [
        Evidence(
            ref=f"F{number}",
            kind="fact",
            item_id=fact.id,
            at=fact.known_at,
            text=fact.text,
            source=fact.source,
            known_at=fact.known_at,
            number=fact.number,
        )
        for number, fact in enumerate(chosen, start=1)
    ]


def build_pack(
    services: Services,
    *,
    days: int,
    now: datetime | None = None,
    memory: Memory | None = None,
) -> EvidencePack:
    """The evidence of the last ``days`` days up to ``now`` (see the module description)."""
    moment = now if now is not None else services.clock.now_utc()
    time_service = time_service_for(services)
    zone = time_service.bot_timezone()
    start = moment - timedelta(days=days)
    config = services.settings.eval
    held = memory or Memory(services)
    view = memory_view(services, moment, memory=held)
    lifeline = _lifeline_items(
        view.lifeline(), zone, time_service.local_date(start), time_service.local_date(moment)
    )
    replies, cut = _reply_items(
        _what_the_bot_said(services, start, moment), zone, config.consistency_reply_chars
    )
    # what the facts are held against: the life line first, then the newest replies
    queries = [item.text for item in (*lifeline, *reversed(replies))]
    facts = _fact_items(held, view, queries, since=start, limit=config.consistency_facts)
    return EvidencePack(
        window_start=start,
        window_end=moment,
        zone=zone.key,
        days=days,
        items=(*lifeline, *replies, *facts),
        reply_turns_cut=cut,
    )


# ------------------------------------------------------------------- what the model reads


def _numbered(lines: Sequence[str]) -> str:
    return "\n".join(lines) if lines else "（没有）"


def render_pack(pack: EvidencePack) -> dict[str, str]:
    """The fields of the ``consistency_audit`` template: window, life line, replies, facts."""
    zone = ZoneInfo(pack.zone)
    first, last = pack.window_start.astimezone(zone), pack.window_end.astimezone(zone)
    lifeline = [f"{item.ref} {item.text}" for item in pack.of("lifeline")]
    replies = [f"{item.ref} {local_stamp(item.at, zone)} {item.text}" for item in pack.of("reply")]
    facts = [
        f"{item.ref}（来源：{SOURCE_NAMES.get(item.source or '', '未知')}；"
        f"{(item.known_at or item.at).astimezone(zone):%Y-%m-%d} 知道）{item.text}"
        for item in pack.of("fact")
    ]
    return {
        "window": f"{first:%Y-%m-%d %H:%M} 至 {last:%Y-%m-%d %H:%M}",
        "days": str(pack.days),
        "lifeline": _numbered(lifeline),
        "replies": _numbered(replies),
        "facts": _numbered(facts),
    }
