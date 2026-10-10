"""R-EVAL-004: one audit - collect, ask DeepSeek, check the answer, store the findings.

DeepSeek is a scripted ``respx`` route that answers in the real JSON shape; the week is synthetic.
What matters here is what the audit does with the answer, what it refuses, what it writes and what
it must never touch.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
import respx
from sqlalchemy import select

from tests.support.consistency_world import (
    HOME_ALL_DAY,
    LIBRARY,
    SHANGHAI,
    AuditModel,
    build_week,
    say,
    the_two_contradictions,
)
from tests.support.deepseek import API
from twin.engine.turns import BotTurnStore
from twin.eval.consistency_audit import (
    AUDIT_WRITES,
    AuditOutcome,
    refresh_run,
    run_audit,
)
from twin.eval.consistency_model import DROPPED_UNKNOWN_REF
from twin.eval.consistency_store import ConsistencyStore, NewFix
from twin.eval.isolation import IsolationViolation, changes, isolated_writes, snapshot
from twin.eval.store import EvalStore
from twin.llm.errors import BudgetDeniedError, CircuitOpenError, StructuredOutputError
from twin.llm.runtime import build_llm_runtime
from twin.profile.prompt_templates import TemplateStore
from twin.services import Services
from twin.storage.models import CostLedger

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"


@pytest.fixture
def model() -> AuditModel:
    return AuditModel(rules=the_two_contradictions())


@pytest.fixture
def api(model: AuditModel) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=model)
        yield router


async def audit(services: Services, days: int | None = None) -> AuditOutcome:
    runtime = build_llm_runtime(services)
    try:
        return await run_audit(services, runtime.client, days=days)
    finally:
        await runtime.client.aclose()


def ledger_rows(services: Services) -> list[tuple[str, str]]:
    with services.db.session() as session:
        return [(row.purpose, row.account) for row in session.scalars(select(CostLedger))]


# ------------------------------------------------------------------------- the findings


async def test_the_planted_contradictions_are_found_and_wait_for_the_user(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    outcome = await audit(services)
    assert outcome.called and outcome.findings == 2 and outcome.inherited == 0
    assert outcome.dropped == {} and outcome.attempts == 1 and outcome.cost_usd > 0
    run = outcome.run
    assert run.kind == "consistency" and run.status == "running" and run.verdict is None
    assert run.mode == "live" and run.backends == ("deepseek",)
    assert run.params["days"] == 7 and run.params["template"] == "consistency_audit@1"
    assert run.summary["evidence"]["lifeline"] == 3 and run.summary["evidence"]["replies"] == 3
    assert run.summary["decisions"]["undecided"] == 2
    store = ConsistencyStore(services.db, services.clock)
    found = store.findings(run.id)
    assert [f.status for f in found] == ["proposed", "proposed"]
    assert [f.model_severity for f in found] == ["obvious", "minor"]
    first = found[0]
    assert LIBRARY in first.first.text and HOME_ALL_DAY in first.second.text
    assert first.first.kind == "lifeline" and first.second.kind == "reply"
    assert first.second.message_ids and first.time_text == "10-06 下午"
    assert SHANGHAI in found[1].second.text and found[1].second.source == "bot_invented"
    assert found[1].rewrite == "她在北京工作"  # the invented fact gives way to the real one
    assert len(model.requests) == 1


async def test_the_call_is_booked_as_a_consistency_call_of_the_daily_account(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    await audit(services)
    assert ledger_rows(services) == [("consistency", "daily")]
    body = model.requests[0]
    assert body["thinking"] == {"type": "enabled"}  # contradictions take reasoning
    assert body["response_format"] == {"type": "json_object"}


async def test_the_prompt_is_the_numbered_records_and_nothing_the_user_wrote(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    await audit(services)
    system, prompt = (str(m["content"]) for m in model.requests[0]["messages"][:2])
    assert "一致性审查员" in system and "contradictions" in system
    assert "审查范围：2026-10-02" in prompt and "最近 7 天" in prompt
    lines = prompt.splitlines()
    assert any(line.startswith("L1 10-06（周二） 09:00-11:00") for line in lines)
    assert any(line.startswith("R2 10-06 21:30 今天整天都在家躺着") for line in lines)
    assert any(
        line.startswith("F") and "（来源：真实聊天记录；2026-09-01 知道）" in line for line in lines
    )
    for typed in ("今天干嘛了", "你今天忙吗", "周末有安排吗"):
        assert typed not in prompt  # his messages are not handed over
    assert "她在芝加哥" not in prompt and "银行" not in prompt  # a fact about him


async def test_what_leaves_the_machine_is_redacted(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    number = "138" + "12345678"  # built from parts: the privacy scan reads this file too
    say(
        BotTurnStore(services.db, services.clock),
        services.clock.now_utc() - timedelta(hours=2),
        "你的号码是多少",
        f"我的手机号是{number}，邮箱 her@example.com",
    )
    await audit(services)
    wire = json.dumps(model.requests[0], ensure_ascii=False)
    assert number not in wire and "her@example.com" not in wire
    assert "[手机号]" in wire and "[邮箱]" in wire


# ------------------------------------------------------------ malformed answers


async def test_a_malformed_answer_is_sent_back_once_and_the_repair_is_used(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    good = json.dumps(
        {
            "contradictions": [
                {
                    "time": "10-06",
                    "first": {"ref": "L1", "quote": None},
                    "second": {"ref": "R2", "quote": "整天都在家躺着"},
                    "severity": "minor",
                    "reason": "时间对不上",
                }
            ]
        },
        ensure_ascii=False,
    )
    model.replies = ["这是一段不是 JSON 的话", good]
    outcome = await audit(services)
    assert outcome.attempts == 2 and outcome.findings == 1 and len(model.requests) == 2
    repair = model.requests[1]["messages"]
    assert repair[-1]["role"] == "user" and "could not be used" in str(repair[-1]["content"])
    assert len(ledger_rows(services)) == 2  # both calls were paid for and are on record
    finding = ConsistencyStore(services.db, services.clock).findings(outcome.run.id)[0]
    assert finding.second.text == "整天都在家躺着"  # a quotation that is in the record is kept


async def test_two_malformed_answers_end_the_audit_with_a_failed_run(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    model.replies = ["不是 JSON", '{"contradictions": "没有"}']
    with pytest.raises(StructuredOutputError):
        await audit(services)
    store = EvalStore(services.db, services.clock)
    run = store.latest_run("consistency")
    assert run is not None and run.status == "failed" and run.verdict is None
    assert run.summary["error"] == "StructuredOutputError"
    assert ConsistencyStore(services.db, services.clock).findings(run.id) == []


async def test_what_does_not_match_the_evidence_is_dropped_and_counted(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    answer = json.dumps(
        {
            "contradictions": [
                {  # R99 was never shown
                    "time": "x",
                    "first": {"ref": "L1"},
                    "second": {"ref": "R99"},
                    "severity": "obvious",
                    "reason": "凭空捏造",
                },
                {
                    "time": "10-06",
                    "first": {"ref": "L1"},
                    "second": {"ref": "R2"},
                    "severity": "obvious",
                    "reason": "真的",
                },
            ]
        },
        ensure_ascii=False,
    )
    model.replies = [answer]
    outcome = await audit(services)
    assert outcome.findings == 1 and outcome.dropped == {DROPPED_UNKNOWN_REF: 1}
    assert outcome.run.summary["findings"]["dropped"] == {DROPPED_UNKNOWN_REF: 1}
    assert outcome.run.summary["findings"]["reported"] == 2


# --------------------------------------------------------------- nothing to audit


async def test_a_week_with_nothing_in_it_asks_nobody_and_is_not_a_pass(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    outcome = await audit(services)
    assert not outcome.called and outcome.findings == 0 and model.requests == []
    run = outcome.run
    assert run.status == "done" and run.verdict == "insufficient"
    assert run.summary["reason"] == "empty_window" and ledger_rows(services) == []
    again = refresh_run(
        EvalStore(services.db, services.clock),
        ConsistencyStore(services.db, services.clock),
        run.id,
    )
    assert again.verdict == "insufficient"  # recounting an empty audit does not turn it into a pass


async def test_a_model_that_finds_nothing_leaves_a_passed_week(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    model.rules = []
    outcome = await audit(services)
    assert outcome.called and outcome.findings == 0
    assert outcome.run.status == "done" and outcome.run.verdict == "passed"
    assert outcome.run.summary["decisions"]["undecided"] == 0


# --------------------------------------------------------------------- held back


class _HeldBack:
    """A client whose calls the budget or the breaker holds back."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def chat_json(self, *args: object, **kwargs: object) -> object:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [BudgetDeniedError("consistency", 4), CircuitOpenError(300.0)],
    ids=["budget", "circuit"],
)
async def test_a_call_that_is_held_back_cancels_the_run_and_lets_the_caller_wait(
    services: Services, error: Exception
) -> None:
    build_week(services)
    with pytest.raises(type(error)):
        await run_audit(services, _HeldBack(error), days=7)  # type: ignore[arg-type]
    run = EvalStore(services.db, services.clock).list_runs("consistency")[0]
    assert run.status == "cancelled" and run.summary["held_back"] == type(error).__name__


async def test_an_api_failure_leaves_a_failed_run(services: Services) -> None:
    build_week(services)
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=AuditModel(failure=400))
        with pytest.raises(Exception, match=r"400|invalid|Invalid|scripted"):
            await audit(services)
    run = EvalStore(services.db, services.clock).list_runs("consistency")[0]
    assert run.status == "failed" and run.summary["error"]


# ----------------------------------------------------------- what is remembered


async def test_a_decision_about_a_pair_is_carried_into_the_next_audit(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    first = await audit(services)
    store = ConsistencyStore(services.db, services.clock)
    one, two = store.findings(first.run.id)
    store.decide(one.id, status="confirmed", severity="obvious")
    store.decide(two.id, status="rejected")
    second = await audit(services)
    assert second.findings == 0 and second.inherited == 2
    again = store.findings(second.run.id)
    assert [(f.status, f.severity, f.inherited_from) for f in again] == [
        ("confirmed", "obvious", one.id),
        ("rejected", None, two.id),
    ]
    assert second.run.status == "done" and second.run.verdict == "passed"  # one obvious: allowed


async def test_the_run_is_closed_with_the_verdict_of_the_rule_once_everything_is_decided(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    outcome = await audit(services)
    estore, store = (
        EvalStore(services.db, services.clock),
        ConsistencyStore(services.db, services.clock),
    )
    one, two = store.findings(outcome.run.id)
    store.decide(one.id, status="confirmed", severity="obvious")
    waiting = refresh_run(estore, store, outcome.run.id)
    assert waiting.status == "running" and waiting.verdict is None
    assert waiting.summary["decisions"] == {
        "confirmed_obvious": 1,
        "confirmed_minor": 0,
        "rejected": 0,
        "undecided": 1,
    }
    store.decide(two.id, status="confirmed", severity="obvious")
    closed = refresh_run(estore, store, outcome.run.id)
    assert closed.status == "done" and closed.verdict == "failed"  # two obvious ones in a week
    assert closed.finished_at is not None and "2" in closed.summary["verdict_reason"]


# --------------------------------------------------------------------- the fence


async def test_the_audit_writes_only_what_an_audit_may_write(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    week = build_week(services)
    # the first use of a prompt installs the template files: that is done before the fence goes up
    TemplateStore(services.db, services.clock).sync()
    before = snapshot(services.db)
    await audit(services)
    moved = changes(before, snapshot(services.db))
    assert moved.outside(AUDIT_WRITES) == []
    assert {"eval_runs", "consistency_findings", "cost_ledger"} <= moved.tables
    for untouched in ("facts", "lifeline_events", "followups", "bot_turns", "messages", "alerts"):
        assert untouched not in moved.tables, untouched
    assert week.memory.store.fact(week.invented_fact.id) == week.invented_fact


async def test_a_write_to_the_memory_is_refused_inside_the_fence(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    week = build_week(services)
    found = ConsistencyStore(services.db, services.clock).findings((await audit(services)).run.id)
    with isolated_writes(AUDIT_WRITES), pytest.raises(IsolationViolation, match="facts"):
        week.memory.store.update_fact(week.invented_fact.id, status="rejected")
    with isolated_writes(AUDIT_WRITES), pytest.raises(IsolationViolation, match="lifeline"):
        week.memory.store.invalidate_event(week.library.id, by=None, at=services.clock.now_utc())
    fix = NewFix("invalidate_fact", week.invented_fact.id, SHANGHAI, None, "x")
    with (
        isolated_writes(AUDIT_WRITES),
        pytest.raises(IsolationViolation, match="consistency_fixes"),
    ):
        ConsistencyStore(services.db, services.clock).add_fixes(found[0].id, [fix])
    assert week.memory.store.fact(week.invented_fact.id) == week.invented_fact  # nothing happened
    assert ConsistencyStore(services.db, services.clock).add_fixes(
        found[0].id, [fix]
    )  # outside: ok


async def test_what_the_bot_said_does_not_reach_samples_retrieval_or_training(
    services: Services, api: respx.MockRouter, model: AuditModel
) -> None:
    build_week(services)
    before = snapshot(services.db)
    await audit(services)
    moved = changes(before, snapshot(services.db)).tables
    for table in (
        "messages",
        "example_windows",
        "profile_versions",
        "persona_cards",
        "training_runs",
        "dataset_versions",
        "model_registry",
        "preference_pairs",
        "stickers",
    ):
        assert table not in moved, table


def modules_of(package: str) -> Iterator[tuple[str, ast.AST]]:
    for path in sorted((SRC / package).rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def imports_of(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_no_style_retrieval_or_training_code_imports_the_audit() -> None:
    offenders = []
    for package in ("profile", "retrieval", "training", "ingest", "stickers"):
        for name, tree in modules_of(package):
            if any(item.startswith("twin.eval.consistency") for item in imports_of(tree)):
                offenders.append(name)
    assert offenders == []


def test_the_audit_never_reaches_for_a_channel() -> None:
    """It sends nothing: no module of it imports a channel, the engine's sender or the state."""
    forbidden = ("twin.channel", "twin.engine.component", "twin.engine.machine", "twin.app")
    offenders = []
    for name, tree in modules_of("eval"):
        if "consistency" not in name:
            continue
        for item in imports_of(tree):
            if item.startswith(forbidden):
                offenders.append(f"{name}: {item}")
    assert offenders == []
