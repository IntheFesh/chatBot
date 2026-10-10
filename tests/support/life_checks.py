"""What every end-to-end scenario promises, written once (R-ARCH-004, R-SCH-*, R-PRO-003, rule 7).

A scenario tells a story and asserts what is special to it.  The promises that hold for **any**
story of the application are asserted here, and a scenario calls the ones that fit it:

* :func:`assert_clean_screen` - nothing a user sees looks like an error, a stack trace or an
  internal name (R-ENG-010, R-SAFE-005);
* :func:`assert_never_in_deep_sleep` - she says nothing while the plan has her in deep sleep
  (R-SCOPE-006, R-ENG-003, R-PRO-003);
* :func:`assert_screen_matches_records` - what the terminal showed is what ``bot_turns`` holds,
  bubble for bubble, once (R-ENG-011; "no bubble lost, none twice");
* :func:`assert_proactive_rules` - the audit of ``twin eval proactive`` over the days: deep sleep,
  spacing, chase, range (R-PRO-003, R-EVAL-005), the production audit itself;
* :func:`assert_within_quota` - the messages between two messages of the user fit the platform's
  count (R-CH-008, R-ENG-009);
* :func:`snapshot_isolation` / :func:`assert_bot_text_not_in_her_data` /
  :func:`assert_bot_text_stays_out` - nothing the bot said, or the user said to it, reached the
  real messages, her profile or the retrieval library (CLAUDE.md rule 7, R-STO-007, R-RET-004,
  R-LRN-004); the first two are cheap and close every story, the last one is heavy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from sqlalchemy import select

from tests.support.embedding import HashingBackend
from tests.support.life_world import LifeWorld, Said
from twin.eval.proactive_audit import Audit, audit_days
from twin.ingest.corpus import her_messages
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.indexer import run_index
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.schedule.proactive.store import RatingStore
from twin.schedule.time_service import PlanUnavailableError
from twin.storage.chat_models import Message
from twin.storage.retrieval_models import ExampleWindow

# words that only an error or an internal name contains, never her
LEAKS = (
    "Traceback",
    "Exception",
    "Error",
    "error",
    "HTTP",
    "deepseek",
    "DeepSeek",
    "api key",
    "token",
    "sqlite",
    "SQL",
    "None",
    "null",
    "{",
    "}",
    "错误",
    "异常",
    "失败",
    "超时",
    "报错",
)
SYSTEM_LEAKS = ("Traceback", "Exception", "Error:", "sqlite", "SQL", 'File "')


def assert_clean_screen(world: LifeWorld) -> None:
    """No message looks like an error: hers carry no internal word, the system's no stack trace."""
    for item in world.said:
        words = SYSTEM_LEAKS if item.kind == "system" else LEAKS
        found = [word for word in words if word in item.text]
        assert not found, f"{world.local_text(item.at)}: {item.text!r} contains {found}"


def her_kind_at(world: LifeWorld, item: Said) -> str:
    try:
        return str(world.kit.time.her_state(item.at).kind)
    except PlanUnavailableError:
        return "unknown"


def assert_never_in_deep_sleep(world: LifeWorld, items: list[Said] | None = None) -> None:
    """She says nothing, a reply or a message of her own, while the plan has her in deep sleep."""
    for item in items if items is not None else world.persona_said:
        state = her_kind_at(world, item)
        assert state != "deep_sleep", (
            f"{world.local_text(item.at)}: she spoke in deep sleep: {item.text!r}"
        )


def assert_screen_matches_records(world: LifeWorld) -> None:
    """The terminal and ``bot_turns`` agree: every bubble once, in order, none missing."""
    rows = world.out_rows()
    screen = list(world.persona_said)
    assert len(rows) == len(screen), (
        f"{len(screen)} bubbles on the screen, {len(rows)} in bot_turns: "
        f"{[i.text for i in screen]} / {[r.text for r in rows]}"
    )
    for row, item in zip(rows, screen, strict=True):
        if row.kind == "text":
            assert row.text == item.text, (row.text, item.text)
        else:
            assert item.kind in ("sticker", "image"), (row.kind, item.kind)
        assert abs((row.at - item.at).total_seconds()) < 2.0, (row.at, item.at)
    commands = [row for row in world.rows() if row.direction == "out" and row.is_command]
    assert len(commands) == len(world.system_said)


def assert_proactive_rules(world: LifeWorld, first: date, last: date) -> Audit:
    """The production audit of the proactive messages over completed days: no violation."""
    audit = audit_days(
        world.proactive_log(),
        RatingStore(world.services.db, world.services.clock),
        world.services.settings.proactive,
        first_day=first,
        last_day=last,
        now=world.now,
    )
    assert audit.deep_sleep == 0, audit.deep_sleep
    for day in audit.days:
        assert day.spacing_violations == 0, day
        assert day.chase_violations == 0, day
        assert day.sent <= (day.high or 0), day
    return audit


def assert_within_quota(world: LifeWorld) -> None:
    """Between two messages of the user, no more than the platform's count goes out (R-CH-008)."""
    quota = world.services.settings.channel.outbound_quota_safe
    inbound = [row.at for row in world.in_rows()]
    outbound = [item.at for item in world.said]
    bounds = [*inbound, world.now + timedelta(days=1)]
    for opening, closing in pairwise(bounds):
        sent = [at for at in outbound if opening <= at < closing]
        assert len(sent) <= quota, f"{len(sent)} messages after {opening}: more than {quota}"


# ------------------------------------------------------------------ rule 7: nothing leaks in


@dataclass(frozen=True)
class IsolationSnapshot:
    """What the real data looked like before the story began."""

    message_ids: frozenset[str]
    window_ids: frozenset[str]


def snapshot_isolation(world: LifeWorld) -> IsolationSnapshot:
    with world.services.db.session() as session:
        return IsolationSnapshot(
            frozenset(session.scalars(select(Message.id))),
            frozenset(session.scalars(select(ExampleWindow.id))),
        )


def conversation_texts(world: LifeWorld) -> set[str]:
    """Everything said in the bot's conversation, both sides, commands included."""
    return {
        row.text.strip()
        for row in world.rows()
        if len(row.text.strip()) >= 2 and not row.text.startswith("[")
    }


def assert_bot_text_not_in_her_data(
    world: LifeWorld, before: IsolationSnapshot | None = None
) -> None:
    """The cheap form, for any story: the real messages and the library are as they were.

    The real messages are the same rows (a line of the conversation with the bot that became a
    message would be one more; her short answers repeat her own words, so the text alone says
    nothing), and the example library has no window it did not have.  ``before`` is what the world
    held when the story began (``world.real_data``, taken by the ``make_world`` fixture) unless the
    story changed the real data itself (an import) and took its own.  The heavy form below also
    rebuilds the profile and asks the retriever; the long stories do that.
    """
    before = before or world.real_data
    assert before is not None, "the world was not built by the make_world fixture"
    with world.services.db.session() as session:
        now_ids = frozenset(session.scalars(select(Message.id)))
        windows = frozenset(session.scalars(select(ExampleWindow.id)))
    assert now_ids == before.message_ids, "the conversation with the bot wrote real messages"
    assert windows == before.window_ids, "the conversation with the bot entered the library"


async def assert_bot_text_stays_out(
    world: LifeWorld, before: IsolationSnapshot, embedder: HashingBackend
) -> None:
    """Nothing of the conversation with the bot is in her messages, her profile or her library."""
    spoken = conversation_texts(world)
    assert spoken, "the story said nothing"
    services = world.services
    with services.db.session() as session:
        assert frozenset(session.scalars(select(Message.id))) == before.message_ids
        hers = [row.text or "" for row in session.scalars(her_messages())]
    assert hers and not [text for text in hers if text.strip() in spoken]
    rebuild(services, "all", force=True)  # her profile, from what is stored now
    profile = load_profile(services, "live")
    assert profile is not None
    phrases = json.dumps(profile.phrases() or {}, ensure_ascii=False)
    leaked = [text for text in spoken if text in phrases]
    assert not leaked, f"in her phrases: {leaked}"
    run_index(services)
    with services.db.session() as session:
        windows = frozenset(session.scalars(select(ExampleWindow.id)))
        assert windows == before.window_ids
    retriever = ExampleRetriever(services, EmbeddingService(embedder))
    for text in sorted(spoken)[:6]:
        asked = [QueryTurn(False, text), QueryTurn(True, text)]
        examples = await retriever.query(ExampleQuery(asked, 720, "workday", None, 8))
        shown = " ".join(
            line.text
            for example in examples
            for line in (*(item for turn in example.context for item in turn.lines), *example.reply)
        )
        assert not [other for other in spoken if other in shown], text
