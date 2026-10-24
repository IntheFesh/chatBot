"""R-EVAL-004: the corrections a confirmed contradiction suggests - proposed, never imposed.

A proposal changes nothing; the memory changes only when a fix is applied, and only what the bot
made up can change.  The tests build findings from the synthetic week and look at the memory
before and after.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
import respx

from tests.support.consistency_world import (
    HOME_ALL_DAY,
    LIBRARY,
    SHANGHAI,
    AuditModel,
    Week,
    build_week,
    the_two_contradictions,
)
from tests.support.deepseek import API
from tests.support.embedding import HashingBackend
from tests.support.memory import add_event, add_fact, add_followup
from twin.eval.consistency_audit import run_audit
from twin.eval.consistency_fixes import (
    INVALIDATE_FACT,
    INVALIDATE_LIFELINE,
    MAX_PROPOSALS,
    REWRITE_FACT,
    REWRITE_LIFELINE,
    FixApplier,
    propose_fixes,
)
from twin.eval.consistency_model import Finding, Statement, fingerprint_of
from twin.eval.consistency_store import ConsistencyStore, FindingView, NewFinding, NewFix
from twin.eval.isolation import changes, snapshot
from twin.eval.store import EvalStore
from twin.llm.runtime import build_llm_runtime
from twin.memory.api import memory_view
from twin.memory.lifeline import LifelineStore
from twin.services import Services
from twin.storage.memory_models import Followup, LifelineEvent

pytestmark = pytest.mark.usefixtures("embedder")


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=AuditModel(rules=the_two_contradictions()))
        yield router


async def audited(services: Services) -> tuple[Week, ConsistencyStore, list[FindingView]]:
    week = build_week(services)
    runtime = build_llm_runtime(services)
    try:
        outcome = await run_audit(services, runtime.client, days=7)
    finally:
        await runtime.client.aclose()
    store = ConsistencyStore(services.db, services.clock)
    return week, store, store.findings(outcome.run.id)


NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)


def statement(
    kind: str,
    item_id: str,
    text: str,
    source: str | None = None,
    message_ids: tuple[str, ...] = (),
) -> Statement:
    return Statement(
        ref=f"{kind[0].upper()}{item_id[-2:]}",
        kind=kind,  # type: ignore[arg-type]
        item_id=item_id,
        text=text,
        source=source,
        at=NOW,
        known_at=NOW,
        message_ids=message_ids,
    )


def stored_finding(
    services: Services,
    first: Statement,
    second: Statement,
    *,
    keep: str | None = None,
    rewrite: str | None = None,
) -> FindingView:
    run = EvalStore(services.db, services.clock).create_run("consistency", mode="live")
    found = Finding(
        time_text="10-06",
        at=first.at,
        first=first,
        second=second,
        related=(),
        severity="obvious",
        reason="两处说法对不上",
        keep=keep,
        rewrite=rewrite,
        fingerprint=fingerprint_of(first, second),
    )
    store = ConsistencyStore(services.db, services.clock)
    return store.add_findings(run.id, [NewFinding(found)])[0]


# ------------------------------------------------------------------------ proposing


async def test_the_life_line_entry_gives_way_to_what_was_already_said(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, findings = await audited(services)
    fixes = propose_fixes(findings[0], week.memory)  # the library against "at home all day"
    assert [(f.action, f.target_id) for f in fixes] == [(INVALIDATE_LIFELINE, week.library.id)]
    assert fixes[0].old_text == week.library.line() and fixes[0].new_text is None
    assert HOME_ALL_DAY not in fixes[0].old_text and LIBRARY in fixes[0].old_text


async def test_a_fact_the_bot_invented_gives_way_to_the_real_one_with_the_rewrite(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, findings = await audited(services)
    fixes = propose_fixes(findings[1], week.memory)
    assert [(f.action, f.target_id) for f in fixes] == [(REWRITE_FACT, week.invented_fact.id)]
    assert fixes[0].old_text == SHANGHAI and fixes[0].new_text == "她在北京工作"
    assert week.real_fact.id not in {f.target_id for f in fixes}  # the real fact is never a target


async def test_proposing_changes_nothing(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    before = snapshot(services.db)
    for finding in findings:
        propose_fixes(finding, week.memory)
    assert changes(before, snapshot(services.db)).tables == frozenset()
    assert store.run_fixes(findings[0].run_id) == []


async def test_nothing_that_is_not_the_bots_is_ever_proposed(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, _ = await audited(services)
    real = statement("fact", week.real_fact.id, week.real_fact.text, "real_record")
    said = statement("fact", week.user_fact.id, week.user_fact.text, "user_said")
    command = add_fact(
        week.memory, "她不吃香菜", week.real_fact.known_at, source="user_command", embed=False
    )
    ordered = statement("fact", command.id, command.text, "user_command")
    assert (
        propose_fixes(stored_finding(services, real, said), week.memory) == []
    )  # the user's words give way
    # both are not the bot's: whichever gives way, nothing of the bot's is touched
    assert propose_fixes(stored_finding(services, ordered, real), week.memory) == []


async def test_a_sent_reply_cannot_be_changed_but_what_was_taken_from_it_can(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, findings = await audited(services)
    kept = findings[0]
    reply = kept.second
    library = kept.first
    finding = stored_finding(services, library, reply, keep=library.ref)  # the reply is wrong
    fixes = propose_fixes(finding, week.memory)
    assert [(f.action, f.target_id) for f in fixes] == [(INVALIDATE_FACT, week.invented_fact.id)]
    assert fixes[0].new_text is None


async def test_when_the_model_cannot_say_which_is_right_both_sides_are_offered(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, _ = await audited(services)
    one = statement("lifeline", week.library.id, week.library.line())
    two = statement("lifeline", week.meeting.id, week.meeting.line())
    fixes = propose_fixes(stored_finding(services, one, two), week.memory)
    assert {f.target_id for f in fixes} == {week.library.id, week.meeting.id}
    assert {f.action for f in fixes} == {INVALIDATE_LIFELINE}  # no rewrite for an undecided loser
    chosen = propose_fixes(stored_finding(services, one, two, keep=one.ref), week.memory)
    assert [f.target_id for f in chosen] == [week.meeting.id]


async def test_a_record_that_is_already_gone_is_not_proposed(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, findings = await audited(services)
    week.memory.store.invalidate_event(week.library.id, by=None, at=services.clock.now_utc())
    week.memory.store.update_fact(week.invented_fact.id, status="rejected")
    assert propose_fixes(findings[0], week.memory) == []
    assert propose_fixes(findings[1], week.memory) == []


async def test_there_are_at_most_a_few_proposals_for_one_contradiction(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, _, findings = await audited(services)
    for number in range(5):  # five facts the bot took from the same reply
        add_fact(
            week.memory,
            f"她今天做了第{number}件事",
            services.clock.now_utc(),
            source="bot_invented",
            evidence={"kind": "bot_turns", "ids": [week.home_reply_id]},
            embed=False,
        )
    kept = findings[0]
    finding = stored_finding(services, kept.first, kept.second, keep=kept.first.ref)
    assert len(propose_fixes(finding, week.memory)) == MAX_PROPOSALS


# ----------------------------------------------------------------------- applying


async def test_a_fix_does_nothing_until_it_is_applied_and_then_exactly_what_it_says(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    fix = store.add_fixes(findings[0].id, propose_fixes(findings[0], week.memory))[0]
    assert fix.status == "proposed" and fix.applied_at is None
    assert [e.id for e in LifelineStore(week.memory).day(week.library.local_date)] == [
        week.library.id,
        week.meeting.id,
    ]
    outcome = FixApplier(week.memory, store).apply(fix.id)
    assert outcome.status == "applied" and outcome.fix.applied_at is not None
    assert [e.id for e in LifelineStore(week.memory).day(week.library.local_date)] == [
        week.meeting.id
    ]
    entry = week.memory.store.event(week.library.id)
    assert entry is not None and entry.status == "invalidated" and entry.invalidated_at is not None
    assert store.fix(fix.id).status == "applied"


async def test_invalidating_a_fact_hides_it_and_what_was_made_from_it(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    memory = week.memory
    linked = add_event(
        memory, week.library.local_date, "搬家后第一天上班", start="08:00", end="09:00"
    )
    with services.db.transaction(bump_state=False) as session:
        row = session.get(LifelineEvent, linked.id)
        assert row is not None
        row.fact_id = week.invented_fact.id
    followup = add_followup(
        memory,
        "问她新工作怎么样",
        services.clock.now_utc() + timedelta(days=1),
        services.clock.now_utc(),
        origin="bot_session",
    )
    with services.db.transaction(bump_state=False) as session:
        owner = session.get(Followup, followup.id)
        assert owner is not None
        owner.fact_id = week.invented_fact.id
    fix = store.add_fixes(
        findings[1].id, [NewFix(INVALIDATE_FACT, week.invented_fact.id, SHANGHAI, None, "x")]
    )[0]
    outcome = FixApplier(memory, store).apply(fix.id)
    assert outcome.status == "applied"
    gone = memory.store.fact(week.invented_fact.id)
    assert gone is not None and gone.status == "rejected" and not gone.current
    assert week.invented_fact.id not in {
        f.id for f in memory_view(services, services.clock.now_utc() + timedelta(seconds=1)).facts()
    }
    event = memory.store.event(linked.id)
    assert event is not None and event.status == "invalidated" and event.invalidated_by == gone.id
    closed = memory.store.followup(followup.id)
    assert closed is not None and closed.status == "cancelled"
    assert memory.store.fact(week.real_fact.id) == week.real_fact  # the real fact is as it was


async def test_rewriting_a_fact_puts_a_new_one_in_its_place(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    memory = week.memory
    fix = store.add_fixes(findings[1].id, propose_fixes(findings[1], memory))[0]
    assert fix.action == REWRITE_FACT
    outcome = FixApplier(memory, store).apply(fix.id)
    assert outcome.status == "applied" and outcome.fix.new_id is not None
    new = memory.store.fact(outcome.fix.new_id)
    old = memory.store.fact(week.invented_fact.id)
    assert new is not None and old is not None
    assert (new.text, new.source, new.subject, new.status) == (
        "她在北京工作",
        "bot_invented",
        "her",
        "active",
    )
    assert new.known_at == services.clock.now_utc() and new.embedding_id is not None  # encoded
    assert old.superseded_by == new.id and old.superseded_at == services.clock.now_utc()
    assert new.evidence is not None and new.evidence["replaces"] == old.id
    assert new.evidence["ids"] == (old.evidence or {})["ids"]  # thrown away with the reply (/重来)
    later = memory_view(services, services.clock.now_utc() + timedelta(seconds=1))
    texts = {f.text for f in later.facts()}
    assert "她在北京工作" in texts and SHANGHAI not in texts and week.real_fact.text in texts


async def test_rewriting_a_life_line_entry_keeps_its_place_in_the_day(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    kept = findings[0]
    seen = stored_finding(
        services, kept.first, kept.second, keep=kept.second.ref, rewrite="在家休息"
    )
    fix = store.add_fixes(seen.id, propose_fixes(seen, week.memory))[0]
    assert fix.action == REWRITE_LIFELINE
    outcome = FixApplier(week.memory, store).apply(fix.id)
    assert outcome.status == "applied" and outcome.fix.new_id is not None
    new = week.memory.store.event(outcome.fix.new_id)
    assert new is not None
    assert (new.activity, new.start_local, new.end_local, new.source) == (
        "在家休息",
        "09:00",
        "11:00",
        "improvised",
    )
    assert new.local_date == week.library.local_date and new.active
    old = week.memory.store.event(week.library.id)
    assert old is not None and old.status == "invalidated"


async def test_declining_leaves_the_memory_as_it_is(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    fix = store.add_fixes(findings[0].id, propose_fixes(findings[0], week.memory))[0]
    before = snapshot(services.db)
    outcome = FixApplier(week.memory, store).decline(fix.id)
    assert outcome.status == "declined" and store.fix(fix.id).status == "declined"
    moved = changes(before, snapshot(services.db)).tables
    assert moved == {"consistency_fixes"}
    assert FixApplier(week.memory, store).apply(fix.id).status == "declined"  # it stays declined


async def test_a_record_that_changed_since_the_proposal_is_left_alone(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    memory = week.memory
    fixes = store.add_fixes(findings[1].id, propose_fixes(findings[1], memory))
    memory.store.update_fact(week.invented_fact.id, text="她刚搬到苏州工作")
    outcome = FixApplier(memory, store).apply(fixes[0].id)
    assert outcome.status == "stale" and store.fix(fixes[0].id).status == "stale"
    fact = memory.store.fact(week.invented_fact.id)
    assert fact is not None and fact.text == "她刚搬到苏州工作" and fact.superseded_by is None
    assert not [f for f in memory.store.all_facts() if f.text == "她在北京工作"]


async def test_the_real_facts_cannot_be_reached_by_a_forged_fix(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    forged = store.add_fixes(
        findings[1].id, [NewFix(INVALIDATE_FACT, week.real_fact.id, week.real_fact.text, None, "x")]
    )[0]
    outcome = FixApplier(week.memory, store).apply(forged.id)
    assert outcome.status == "stale"  # only a fact whose source is bot_invented can be changed
    assert week.memory.store.fact(week.real_fact.id) == week.real_fact


async def test_a_fix_is_applied_once(
    services: Services, embedder: HashingBackend, api: respx.MockRouter
) -> None:
    week, store, findings = await audited(services)
    fix = store.add_fixes(findings[1].id, propose_fixes(findings[1], week.memory))[0]
    applier = FixApplier(week.memory, store)
    first = applier.apply(fix.id)
    again = applier.apply(fix.id)
    assert first.status == again.status == "applied" and again.fix.new_id == first.fix.new_id
    rewritten = [f for f in week.memory.store.all_facts() if f.text == "她在北京工作"]
    assert len(rewritten) == 1
