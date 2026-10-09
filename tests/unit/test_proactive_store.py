"""The tables of the proactive messages and their repositories (R-STO-006, R-PRO-008)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.support.clock import ManualClock
from twin.schedule.events import CandidatesExpired
from twin.schedule.proactive.store import (
    CandidateStore,
    NewCandidate,
    NewLog,
    ProactiveLogStore,
    RatingStore,
    average,
)
from twin.schedule.proactive.types import Reason, TriggerKind
from twin.services import Services
from twin.storage import migrate

T0 = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)
DAY = date(2026, 10, 9)
ZONE = "America/Chicago"
CHICAGO = ZoneInfo(ZONE)


def new_candidate(
    key: str = "greeting", *, at: datetime = T0, end_after: int = 30, **extra: object
) -> NewCandidate:
    return NewCandidate(
        local_date=DAY,
        timezone=ZONE,
        key=key,
        kind=TriggerKind(extra.pop("kind", "greeting")),  # type: ignore[arg-type]
        priority=1,
        planned_at=at,
        window_end=at + timedelta(minutes=end_after),
        plan_id="plan1",
        detail=extra,  # type: ignore[arg-type]
    )


def log_row(outcome: str = "sent", *, at: datetime = T0, **extra: object) -> NewLog:
    return NewLog(
        at=at,
        candidate_at=at,
        local_date=DAY,
        local_at=f"{at.astimezone(CHICAGO):%Y-%m-%d %H:%M}",
        timezone=ZONE,
        kind=str(extra.pop("kind", "greeting")),
        outcome=outcome,
        **extra,  # type: ignore[arg-type]
    )


def test_a_slot_of_the_day_is_made_once_per_zone(services: Services, clock: ManualClock) -> None:
    store = CandidateStore(services.db, clock)
    first = store.add(new_candidate(meal="lunch"))
    assert first is not None and first.status == "pending" and first.detail == {"meal": "lunch"}
    assert store.add(new_candidate()) is None  # the same key on the same day and zone
    elsewhere = new_candidate()
    assert (
        store.add(NewCandidate(**{**elsewhere.__dict__, "timezone": "Asia/Shanghai"})) is not None
    )
    assert store.has_key("greeting")
    assert not store.has_key("followup:x")
    assert [row.key for row in store.for_day(DAY, ZONE)] == ["greeting"]


def test_pending_candidates_are_listed_by_their_time_and_marked_when_done(
    services: Services, clock: ManualClock
) -> None:
    store = CandidateStore(services.db, clock)
    late = store.add(new_candidate("meal:dinner", at=T0 + timedelta(hours=5), kind="meal"))
    early = store.add(new_candidate("greeting", at=T0))
    assert late is not None and early is not None
    assert [row.key for row in store.pending()] == ["greeting", "meal:dinner"]
    store.mark(early.id, "sent", at=T0 + timedelta(minutes=3))
    assert [row.key for row in store.pending()] == ["meal:dinner"]
    done = store.get(early.id)
    assert (
        done is not None and done.status == "sent" and done.status_at == T0 + timedelta(minutes=3)
    )
    store.note_reason(late.id, Reason.SPACING)
    again = store.get(late.id)
    assert again is not None and again.last_reason == "spacing"
    store.reschedule(late.id, T0 + timedelta(hours=6))
    moved = store.get(late.id)
    assert moved is not None and moved.attempts == 1 and moved.last_reason is None
    assert moved.planned_at == T0 + timedelta(hours=6)
    assert store.get("nothing") is None
    store.mark("nothing", "sent", at=T0)  # no such row: nothing happens
    store.note_reason("nothing", Reason.SPACING)
    store.reschedule("nothing", T0)


def test_candidates_whose_window_passed_expire(services: Services, clock: ManualClock) -> None:
    store = CandidateStore(services.db, clock)
    store.add(new_candidate("greeting", at=T0, end_after=30))
    store.add(new_candidate("meal:lunch", at=T0 + timedelta(hours=2), kind="meal"))
    expired = store.expire(before=T0 + timedelta(minutes=31), at=T0 + timedelta(minutes=31))
    assert [row.key for row in expired] == ["greeting"]
    assert [row.key for row in store.pending()] == ["meal:lunch"]
    assert store.expire(before=T0 + timedelta(minutes=31), at=T0) == []


def test_a_schedule_event_voids_the_candidates_it_covers(
    services: Services, clock: ManualClock
) -> None:
    store = CandidateStore(services.db, clock)
    store.add(new_candidate("greeting", at=T0))
    store.add(new_candidate("meal:lunch", at=T0 + timedelta(hours=3), kind="meal"))
    cutoff = T0 + timedelta(hours=1)
    event = CandidatesExpired(at=cutoff, cutoff=cutoff, reason="startup")
    voided = store.expire_covered(event)
    assert [row.key for row in voided] == ["greeting"]  # planned before the cutoff
    assert [row.key for row in store.pending()] == ["meal:lunch"]
    everything = CandidatesExpired(
        at=cutoff, cutoff=cutoff, reason="timezone_switch", include_future=True
    )
    assert [row.key for row in store.expire_covered(everything)] == ["meal:lunch"]
    assert store.pending() == []


def test_the_log_keeps_every_decision_and_seals_the_words(
    services: Services, clock: ManualClock
) -> None:
    store = ProactiveLogStore(services.db, clock)
    entry = store.add(
        log_row(
            reply_id="r1",
            her_state="free",
            bubbles_sent=1,
            chase_seq=0,
            quota_total=3,
            range_min=1,
            range_max=6,
            enabled=True,
            plan_reason="想问问他吃饭了没",
            content={"bubbles": ["吃饭了吗"]},
            backend="deepseek",
            cost_usd=0.002,
        )
    )
    assert entry.outcome == "sent" and entry.plan_reason == "想问问他吃饭了没"
    plain = store.entries(first_day=DAY, last_day=DAY)[0]
    assert plain.plan_reason is None and plain.content is None  # no words unless asked
    full = store.entries(first_day=DAY, last_day=DAY, with_text=True)[0]
    assert full.content == {"bubbles": ["吃饭了吗"]} and full.plan_reason == "想问问他吃饭了没"
    connection = sqlite3.connect(services.paths.db_path)
    try:
        raw = connection.execute("SELECT plan_reason, content FROM proactive_log").fetchone()
    finally:
        connection.close()
    assert "吃饭了吗".encode() not in raw[1] and "想问问".encode() not in raw[0]


def test_a_row_written_at_the_first_bubble_is_completed_by_the_last(
    services: Services, clock: ManualClock
) -> None:
    store = ProactiveLogStore(services.db, clock)
    entry = store.add(log_row(bubbles_sent=1, reply_id="r1"))
    store.update(entry.id, bubbles_sent=3, result={"sent": 3}, content={"bubbles": ["a", "b", "c"]})
    done = store.get(entry.id)
    assert done is not None and done.bubbles_sent == 3 and done.result == {"sent": 3}
    assert done.content == {"bubbles": ["a", "b", "c"]}
    store.update("nothing", bubbles_sent=9)  # no such row: nothing happens
    assert store.get("nothing") is None
    store.update(entry.id, reply_id="r2", backend="hybrid", cost_usd=0.0042)
    again = store.get(entry.id)
    assert again is not None and (again.reply_id, again.backend) == ("r2", "hybrid")
    assert again.cost_usd == 0.0042 and again.bubbles_sent == 3  # what was not named stays


def test_the_counts_the_scheduler_needs(services: Services, clock: ManualClock) -> None:
    store = ProactiveLogStore(services.db, clock)
    store.add(log_row("sent", at=T0, her_state="free"))
    store.add(log_row("sent", at=T0 + timedelta(hours=2), her_state="sleep_edge", kind="edge"))
    store.add(log_row("rejected", at=T0 + timedelta(hours=3), reason="spacing"))
    store.add(log_row("opened", at=T0 - timedelta(hours=1), kind="day"))
    assert store.count_sent_on(DAY) == 2
    assert store.count_sent_since(T0) == 2
    assert store.count_sent_since(T0 + timedelta(hours=1)) == 1
    assert store.count_sent_since(T0, state="sleep_edge") == 1
    last = store.last_sent()
    assert last is not None and last.kind == "edge"
    assert [row.at for row in store.sent_since(T0)] == [T0, T0 + timedelta(hours=2)]
    assert store.last_sent_before(T0 + timedelta(hours=1)).at == T0  # type: ignore[union-attr]
    assert store.last_sent_before(T0) is None
    opened = store.opened(DAY, ZONE)
    assert opened is not None and opened.outcome == "opened"
    assert store.opened(DAY, "Asia/Shanghai") is None
    refused = store.last_rejection("greeting", T0)
    assert refused is not None and refused.reason == "spacing"
    assert store.last_rejection("meal", T0) is None
    assert [r.outcome for r in store.entries(outcomes=["sent"])] == ["sent", "sent"]
    assert len(store.entries(since=T0 + timedelta(hours=1), limit=1)) == 1


def test_ratings_are_scores_of_one_to_five_with_a_sealed_note(
    services: Services, clock: ManualClock
) -> None:
    store = RatingStore(services.db, clock)
    one = store.add(4, "晚安发得很自然", at=T0, local_date=DAY)
    store.add(5, None, at=T0 + timedelta(days=1), local_date=DAY + timedelta(days=1))
    assert one.score == 4 and one.note == "晚安发得很自然"
    with pytest.raises(ValueError):
        store.add(6, None, at=T0, local_date=DAY)
    with pytest.raises(ValueError):
        store.add(0, None, at=T0, local_date=DAY)
    both = store.between(T0, T0 + timedelta(days=2))
    assert [r.score for r in both] == [4, 5] and average(both) == 4.5
    assert store.between(T0 + timedelta(days=3), T0 + timedelta(days=4)) == []
    assert average([]) is None
    assert [r.score for r in store.recent(1)] == [5]
    connection = sqlite3.connect(services.paths.db_path)
    try:
        note = connection.execute("SELECT note FROM ratings WHERE score = 4").fetchone()[0]
    finally:
        connection.close()
    assert "晚安".encode() not in note


# ----------------------------------------------------------------------- the migration


def test_round_10_migration_adds_the_tables_the_columns_and_the_audit_kind(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0012_eval_tables")
    connection = sqlite3.connect(path)
    try:
        before = {
            n for (n,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    migrate.upgrade(path, "0013_proactive_tables")
    assert "0013_proactive_tables" in migrate.revision_history()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        after = {
            n for (n,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert after - before == {"proactive_candidates", "proactive_log", "ratings"}
        columns = {row[1] for row in connection.execute("PRAGMA table_info(lifeline_events)")}
        assert {"shared_at", "shared_reply_id"} <= columns
        stamps = "'2026-03-08', '2026-03-08'"
        connection.execute(
            "INSERT INTO eval_runs (id, kind, status, mode, milestone, verdict, backends, "
            f"batch_ids, params, summary, created_at, updated_at) VALUES ('a', 'proactive_audit', "
            f"'done', NULL, NULL, NULL, '[]', '[]', '{{}}', '{{}}', {stamps})"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO eval_runs (id, kind, status, mode, milestone, verdict, backends, "
                f"batch_ids, params, summary, created_at, updated_at) VALUES ('b', 'essay', "
                f"'done', NULL, NULL, NULL, '[]', '[]', '{{}}', '{{}}', {stamps})"
            )
        with pytest.raises(sqlite3.IntegrityError):  # a score is 1 to 5
            connection.execute(
                f"INSERT INTO ratings (id, at, local_date, score, source, created_at, updated_at) "
                f"VALUES ('r', {stamps.split(',')[0]}, '2026-03-08', 6, 'command', {stamps})"
            )
    finally:
        connection.close()


def test_the_migration_keeps_the_evaluation_items_of_the_runs_it_rebuilds(tmp_path: Path) -> None:
    """The ``eval_runs`` table is made again; its children must not go with the old one."""
    path = tmp_path / "m.db"
    migrate.upgrade(path, "0012_eval_tables")
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO eval_runs (id, kind, status, mode, milestone, verdict, backends, "
            "batch_ids, params, summary, created_at, updated_at) VALUES ('r1', 'blind', "
            "'planned', 'holdout', NULL, NULL, '[]', '[]', '{}', '{}', '2026-03-08', '2026-03-08')"
        )
        connection.execute(
            "INSERT INTO eval_items (id, run_id, seq, sample_key, backend, at, status, outcome, "
            "auto_outcome, cost_usd, payload, created_at, updated_at) VALUES ('i1', 'r1', 0, 'k', "
            "'deepseek', '2026-03-08', 'pending', NULL, NULL, 0, x'0102', '2026-03-08', "
            "'2026-03-08')"
        )
        connection.commit()
    finally:
        connection.close()
    migrate.upgrade(path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT id, kind FROM eval_runs").fetchall() == [("r1", "blind")]
        assert connection.execute("SELECT id, run_id, payload FROM eval_items").fetchall() == [
            ("i1", "r1", b"\x01\x02")
        ]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='eval_items'"
        ).fetchone()[0]
        assert "REFERENCES eval_runs" in sql  # the children still point at the new table
    finally:
        connection.close()
    migrate.downgrade(path, "0012_eval_tables")
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT id FROM eval_items").fetchall() == [("i1",)]
        columns = {row[1] for row in connection.execute("PRAGMA table_info(lifeline_events)")}
        assert not {"shared_at", "shared_reply_id"} & columns
    finally:
        connection.close()
