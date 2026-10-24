"""The single hold-out split point (R-RET-003, R-TRN-013) and the scan that keeps it single."""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.support.synth_chat import ChatSpec, MessageWriter, append_texts, build_chat
from twin.ingest.corpus import conversation_messages, conversation_timeline, her_messages
from twin.ingest.events import REPRODUCIBLE_KINDS
from twin.ops.jobs import JobQueue
from twin.profile.api import load_profile
from twin.profile.builder import rebuild
from twin.profile.holdout import (
    HOLDOUT_KEY,
    HoldoutError,
    compute_holdout,
    get_holdout,
    holdout_cutoff,
    on_holdout_change,
    resplit_holdout,
)
from twin.profile.queue import PROFILE_JOB, queue_profile_rebuild
from twin.services import Services
from twin.storage.db import ReadOnlyViolationError, WritePolicy, use_write_policy
from twin.storage.settings_store import get_history

ROOT = Path(__file__).resolve().parents[2]
BASE = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def write_her_blocks(
    services: Services, count: int, *, spacing_h: float = 3.0, pair: bool = False, start: int = 0
) -> list[datetime]:
    """``count`` blocks of her messages ``spacing_h`` apart; returns each block's start."""
    writer = MessageWriter(services)
    starts = []
    for i in range(start, start + count):
        at = BASE + timedelta(hours=spacing_h * i)
        starts.append(at)
        writer.add(at, True, "text", f"消息{i}")
        if pair:
            writer.add(at + timedelta(seconds=30), True, "text", f"又一条{i}")
    writer.store(append=start > 0)
    return starts


def independent_cutoff(services: Services, ratio: float = 0.1) -> tuple[datetime, int, int]:
    """The definition written out again, from the stored rows, for comparison."""
    with services.db.session() as session:
        rows = [
            (m.create_time_utc, not m.is_sent, m.kind)
            for m in session.scalars(conversation_messages())
            if m.kind != "system"
        ]
    blocks: list[tuple[datetime, bool]] = []
    previous: tuple[datetime, bool] | None = None
    for moment, her, kind in rows:
        joins = (
            previous is not None
            and previous[1] == her
            and (moment - previous[0]).total_seconds() <= 120
        )
        if joins:
            if her and kind in REPRODUCIBLE_KINDS:
                blocks[-1] = (blocks[-1][0], True)
        else:
            blocks.append((moment, her and kind in REPRODUCIBLE_KINDS))
            blocks[-1] = (blocks[-1][0], blocks[-1][1])
            if not her:
                blocks[-1] = (blocks[-1][0], False)
        previous = (moment, her)
    mine = [start for start, ok in blocks if ok]
    held = min(len(mine) - 1, max(1, int(len(mine) * ratio + 0.5)))
    return mine[len(mine) - held], len(mine), held


def test_the_cutoff_is_the_start_of_the_last_tenth_of_her_reply_blocks(
    services: Services,
) -> None:
    build_chat(services, ChatSpec(days=30))
    expected, blocks, held = independent_cutoff(services)
    cutoff = holdout_cutoff(services)
    assert cutoff == expected
    stored = get_holdout(services)
    assert stored is not None
    assert (stored.blocks, stored.held_out_blocks, stored.ratio) == (blocks, held, 0.1)
    assert held == pytest.approx(0.1 * blocks, abs=1)
    with services.db.session() as session:
        moments = [m.create_time_utc for m in session.scalars(her_messages())]
    later = [m for m in moments if m >= cutoff]
    share = len(later) / len(moments)
    assert later and 0.08 < share < 0.14  # roughly a tenth is held out


def test_twenty_blocks_hold_out_the_last_two(services: Services) -> None:
    starts = write_her_blocks(services, 20, pair=True)
    assert holdout_cutoff(services) == starts[18]
    stored = get_holdout(services)
    assert stored is not None and (stored.blocks, stored.held_out_blocks) == (20, 2)


def test_blocks_that_the_bot_could_not_have_written_are_not_counted(services: Services) -> None:
    starts = write_her_blocks(services, 20)
    writer = MessageWriter(services)
    for i in range(5):  # pictures, calls and a system notice after the last block
        at = BASE + timedelta(hours=3 * (20 + i))
        writer.add(at, True, "image")
        writer.add(at + timedelta(hours=1), True, "call")
    writer.store(append=True)
    assert holdout_cutoff(services) == starts[18]


def test_the_ratio_comes_from_the_configuration(services: Services) -> None:
    starts = write_her_blocks(services, 20)
    services.settings.retrieval.holdout_ratio = 0.25
    assert compute_holdout(services).cutoff == starts[15]
    services.settings.retrieval.holdout_ratio = 0.01
    assert compute_holdout(services).cutoff == starts[19]  # at least one block is held out


def test_the_cutoff_is_stored_and_does_not_move_when_data_is_imported(
    services: Services,
) -> None:
    starts = write_her_blocks(services, 20)
    first = holdout_cutoff(services)
    assert first == starts[18]
    write_her_blocks(services, 30, start=20)  # a later import
    assert holdout_cutoff(services) == first
    stored = get_holdout(services)
    assert stored is not None and stored.blocks == 20
    assert compute_holdout(services).blocks == 50  # the data changed, the cutoff did not


def test_reading_the_stored_hold_out_needs_no_write_access(services: Services) -> None:
    write_her_blocks(services, 12)
    assert get_holdout(services) is None
    with use_write_policy(WritePolicy(read_only=True)):
        assert get_holdout(services) is None
        with pytest.raises(ReadOnlyViolationError):
            holdout_cutoff(services)  # computing it for the first time is a write
    cutoff = holdout_cutoff(services)
    with use_write_policy(WritePolicy(read_only=True)):
        stored = get_holdout(services)
        assert stored is not None and stored.cutoff == cutoff


def test_a_resplit_moves_the_cutoff_and_queues_the_pre_holdout_rebuild(
    services: Services,
) -> None:
    write_her_blocks(services, 20)
    old = holdout_cutoff(services)
    write_her_blocks(services, 30, start=20)
    result = resplit_holdout(services)
    assert result.previous is not None and result.previous.cutoff == old
    assert result.current.cutoff > old and holdout_cutoff(services) == result.current.cutoff
    assert any("not comparable" in note for note in result.notes)
    assert any(note.startswith("profile:") and "queued" in note for note in result.notes)
    queue = JobQueue(services.db, services.clock)
    (job,) = queue.list_jobs(status="pending", job_type=PROFILE_JOB)
    assert job.payload["scope"] == "pre_holdout" and job.payload["reason"] == "resplit"
    again = resplit_holdout(services)
    assert any("already queued" in note for note in again.notes)
    with services.db.session() as session:
        assert len(get_history(session, HOLDOUT_KEY)) >= 2


def test_dependants_can_register_for_a_resplit(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.profile import holdout

    monkeypatch.setattr(holdout, "_listeners", dict(holdout._listeners))
    seen: list[tuple[bool, datetime]] = []

    @on_holdout_change("retrieval-test")
    def listener(services: Services, previous: object, current: holdout.Holdout) -> str:
        seen.append((previous is None, current.cutoff))
        return "index rebuild queued"

    write_her_blocks(services, 20)
    result = resplit_holdout(services)
    assert seen == [(True, result.current.cutoff)]
    assert "retrieval-test: index rebuild queued" in result.notes


def test_the_pre_holdout_profile_follows_a_resplit(services: Services) -> None:
    build_chat(services, ChatSpec(days=30))
    rebuild(services, "all", reason="test")
    first = load_profile(services, "pre_holdout")
    assert first is not None
    last = max(_her_times(services))
    append_texts(
        services,
        [(last + timedelta(days=1, minutes=10 * i), True, f"新{i}") for i in range(120)],
    )
    resplit_holdout(services)
    rebuild(services, "pre_holdout", reason="resplit")
    second = load_profile(services, "pre_holdout")
    assert second is not None and second.version.id != first.version.id
    assert second.version.data_range["cutoff"] == holdout_cutoff(services).isoformat()
    assert second.version.data_range["cutoff"] != first.version.data_range["cutoff"]


def _her_times(services: Services) -> list[datetime]:
    with services.db.session() as session:
        return [m.create_time_utc for m in session.scalars(her_messages())]


def test_too_little_data_is_reported(services: Services) -> None:
    with pytest.raises(HoldoutError, match="at least 2"):
        holdout_cutoff(services)
    write_her_blocks(services, 1)
    with pytest.raises(HoldoutError, match="found 1"):
        holdout_cutoff(services)
    assert get_holdout(services) is None
    report = rebuild(services, "pre_holdout", reason="test")
    assert report.results[0].status == "skipped" and "at least 2" in report.results[0].note


def test_the_queue_does_not_repeat_identical_requests(services: Services) -> None:
    first = queue_profile_rebuild(services, scope="live")
    second = queue_profile_rebuild(services, scope="live")
    other = queue_profile_rebuild(services, scope="all")
    assert not first.already_queued and second.already_queued and second.job_id == first.job_id
    assert not other.already_queued and other.job_id != first.job_id
    with pytest.raises(ValueError, match="scope"):
        queue_profile_rebuild(services, scope="nope")


def test_the_corpus_reads_both_sides_for_statistics_and_her_alone_for_style(
    services: Services,
) -> None:
    build_chat(services, ChatSpec(days=5))
    with services.db.session() as session:
        both = [
            (m.create_time_utc, m.is_sent, m.conversation_id)
            for m in session.scalars(conversation_messages())
        ]
        hers = [(m.create_time_utc, m.is_sent) for m in session.scalars(her_messages())]
        timeline = list(session.execute(conversation_timeline()))
    assert len(both) == len(timeline) > len(hers) > 0
    assert {sent for _, sent, _ in both} == {True, False} and not any(sent for _, sent in hers)
    times = [moment for moment, _, _ in both]
    assert times == sorted(times) and [row[0] for row in timeline] == times
    assert all(row[1] is sent for row, (_, sent, _) in zip(timeline, both, strict=True))
    one = both[0][2]
    with services.db.session() as session:
        assert len(list(session.scalars(conversation_messages(one)))) == len(both)
        assert list(session.scalars(conversation_messages("elsewhere"))) == []
        assert list(session.execute(conversation_timeline("elsewhere"))) == []


# ----------------------------------------------------------- the one implementation

SCANNED = [
    *sorted((ROOT / "src" / "twin").rglob("*.py")),
    *sorted((ROOT / "scripts").rglob("*.py")),
    *sorted((ROOT / "training").rglob("*.py")),
]
ALLOWED_HOLDOUT_MODULE = ROOT / "src" / "twin" / "profile" / "holdout.py"
SETTINGS_MODULE = ROOT / "src" / "twin" / "config" / "settings.py"


def holdout_ratio_uses(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        name = {ast.Attribute: "attr", ast.Name: "id", ast.keyword: "arg"}.get(type(node))
        if name is not None and getattr(node, name) == "holdout_ratio":
            lines.append(getattr(node, "lineno", None) or node.value.lineno)  # type: ignore[attr-defined]
    return lines


def ratio_split_lines(tree: ast.AST) -> list[int]:
    """``len(x) * 0.9``-style arithmetic: a ratio cut of a sequence."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult | ast.Div):
            continue
        sides = (node.left, node.right)
        has_len = any(
            isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id == "len"
            for s in sides
        )
        has_fraction = any(
            isinstance(s, ast.Constant) and isinstance(s.value, float) and (0.05 <= s.value <= 0.95)
            for s in sides
        )
        if has_len and has_fraction:
            lines.append(node.lineno)
    return lines


def test_nothing_but_the_holdout_module_reads_the_ratio_or_cuts_by_ratio() -> None:
    offenders: list[str] = []
    for path in SCANNED:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT)
        if path not in (ALLOWED_HOLDOUT_MODULE, SETTINGS_MODULE):
            offenders += [f"{rel}:{n} reads holdout_ratio" for n in holdout_ratio_uses(tree)]
        if path != ALLOWED_HOLDOUT_MODULE:
            offenders += [
                f"{rel}:{n} cuts a sequence by a fraction" for n in ratio_split_lines(tree)
            ]
    assert offenders == []


def test_there_is_exactly_one_holdout_cutoff_function() -> None:
    definitions = [
        path.relative_to(ROOT).as_posix()
        for path in SCANNED
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "holdout_cutoff"
    ]
    assert definitions == ["src/twin/profile/holdout.py"]


def test_the_scan_recognises_a_second_implementation() -> None:
    sample = ast.parse("cut = int(len(rows) * 0.9)\nratio = settings.retrieval.holdout_ratio\n")
    assert ratio_split_lines(sample) == [1] and holdout_ratio_uses(sample) == [2]
    clean = ast.parse("half = len(rows) // 2\nprice = cost * 0.9\n")
    assert ratio_split_lines(clean) == [] and holdout_ratio_uses(clean) == []
