"""Whatever order the replay meets the days in, the memory of a moment holds only what was said
before it (R-MEM-004, R-MEM-010, iron rule 13; round 18 item C1, D-780).

Replay jobs of neighbouring weeks run side by side and older history can be imported after newer
history, so a later day is often written first.  The writer must then never let a fact's wording
become visible before the time it was first said: a merge that lowered the known time of a later,
more specific fact to that of an earlier, vaguer one did exactly that.

The property test drives the writer's decision-and-apply step directly with every relation the
judge could give (``same``, ``update``, ``conflict``, ``unrelated``) between every pair, every
source, exact repeats and every processing order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.support.memory import fact_draft, make_memory, utc
from twin.memory.conflict import FactCandidate, Recall, normalised
from twin.memory.extract import FactDraft
from twin.memory.memory import Memory
from twin.memory.view import MemoryView
from twin.memory.writer import MemoryWriter
from twin.services import Services

TEXTS = (
    "她下个月要考雅思",
    "她3月15日下午考雅思口语",
    "她的雅思口语改到了3月20日",
    "她不考雅思了",
)
SOURCES = ("real_record", "user_said", "bot_invented", "user_command")
RELATIONS = ("same", "update", "conflict", "unrelated")
START = utc(2026, 2, 1, 3)


@dataclass(frozen=True)
class Said:
    """One statement of the history: what, when (hours after ``START``) and by which source."""

    text: int
    hours: int
    source: int


statements = st.lists(
    st.builds(
        Said,
        text=st.integers(0, len(TEXTS) - 1),
        hours=st.integers(0, 24 * 40),
        source=st.integers(0, len(SOURCES) - 1),
    ),
    min_size=2,
    max_size=6,
)


def _draft(said: Said, index: int) -> FactDraft:
    return fact_draft(
        TEXTS[said.text],
        category="preference",
        source=SOURCES[said.source],
        known_at=START + timedelta(hours=said.hours),
        evidence_refs=(f"m{index}",),
    )


def _replay(
    memory: Memory,
    drafts: list[FactDraft],
    order: list[int],
    relation_of: dict[tuple[int, int], str],
) -> None:
    """Write the drafts in ``order``; the judge answers ``relation_of[(new, old origin)]``."""
    writer = MemoryWriter(memory, None)
    origin: dict[str, int] = {}
    for index in order:
        memory.refresh()
        current = sorted(
            (f for f in memory.corpus.facts.values() if f.current), key=lambda f: f.number
        )
        candidates = tuple(
            FactCandidate(f"c{number}", fact) for number, fact in enumerate(current, start=1)
        )
        mapping = {c.key: relation_of[(index, origin[c.fact.id])] for c in candidates}
        writer._apply_all([(drafts[index], Recall(candidates))], [mapping])
        for fact in memory.store.all_facts():  # the one fact this draft added, if any
            origin.setdefault(fact.id, index)


def _moments(drafts: list[FactDraft]) -> list[datetime]:
    found: set[datetime] = set()
    for draft in drafts:
        for shift in (timedelta(0), timedelta(seconds=1), timedelta(hours=1), timedelta(days=3)):
            found.add(draft.known_at + shift)
    return sorted(found)


def _leaks(memory: Memory, drafts: list[FactDraft]) -> list[tuple[datetime, str]]:
    """Facts visible at a moment whose words nobody had said before that moment."""
    leaks: list[tuple[datetime, str]] = []
    for moment in _moments(drafts):
        said = {normalised(d.text) for d in drafts if d.known_at < moment}
        for fact in MemoryView(memory, moment).facts():
            if normalised(fact.text) not in said:
                leaks.append((moment, fact.text))
    return leaks


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(data=st.data(), history=statements)
def test_no_processing_order_shows_words_before_they_were_said(
    services: Services, data: st.DataObject, history: list[Said]
) -> None:
    """R18-C1 (D-780): the hypothesis variant over orders, relations, sources and repeats."""
    memory = make_memory(services)
    memory.store.delete_facts([f.id for f in memory.store.all_facts()])
    drafts = [_draft(said, index) for index, said in enumerate(history)]
    order = data.draw(st.permutations(range(len(drafts))), label="order")
    relation_of = {
        (new, old): data.draw(st.sampled_from(RELATIONS), label=f"relation {new}->{old}")
        for new in range(len(drafts))
        for old in range(len(drafts))
    }
    _replay(memory, drafts, list(order), relation_of)
    assert _leaks(memory, drafts) == []


def test_the_later_wording_stays_in_the_future_in_either_order(services: Services) -> None:
    """R18-C1 (D-780): the audit's case, both orders; every pair is judged "same"."""
    vague = fact_draft("她下个月要考雅思", known_at=utc(2026, 2, 20, 21))
    exact = fact_draft("她3月15日下午考雅思口语", known_at=utc(2026, 3, 10, 21))
    for order in ([0, 1], [1, 0]):
        memory = make_memory(services)
        memory.store.delete_facts([f.id for f in memory.store.all_facts()])
        relations = {(a, b): "same" for a in range(2) for b in range(2)}
        _replay(memory, [vague, exact], order, relations)
        assert [f.text for f in MemoryView(memory, utc(2026, 3, 1)).facts()] == [vague.text]
        assert _leaks(memory, [vague, exact]) == []
