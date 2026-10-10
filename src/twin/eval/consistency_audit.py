"""One consistency audit: collect, ask DeepSeek, check, store (R-EVAL-004).

:func:`run_audit` is what both the weekly job and ``twin eval consistency`` call.  It

1. builds the evidence (:mod:`twin.eval.consistency_pack`) - the last ``eval.consistency_days``
   days of the life line, the bot's words about herself and the facts that concern her;
2. asks DeepSeek for the contradictions (purpose ``consistency``, the ``consistency_audit``
   template, a reply validated against :class:`~twin.eval.consistency_model.ConsistencyOut`; a
   reply that does not fit is sent back once, a second failure ends the audit).  The client
   passes every outgoing message through :mod:`twin.llm.redaction` (R-LLM-009);
3. holds the answer to the evidence (:func:`~twin.eval.consistency_model.validate_output`) and
   stores what survives as findings, each waiting for the user - a pair he has already decided
   about keeps his decision;
4. records the audit as ``eval_runs(kind=consistency)``.

**Nothing of the live memory is touched and nothing is sent.**  Everything after the evidence is
collected happens inside a write fence (:data:`AUDIT_WRITES`, the sandbox's mechanism of
:mod:`twin.eval.isolation`): the run, its findings, the ledger row of the call and the alerts are
all the audit may write; a write to ``facts`` or ``lifeline_events`` is not among them, and no
message is sent to anyone.  The memory changes only later, one proposal at a time, when the user
says yes (:mod:`twin.eval.consistency_fixes`).

The run stays ``running`` while findings wait for a decision; :func:`refresh_run` closes it - with
the verdict of :func:`~twin.eval.consistency_model.judge_audit` - when none is left.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from twin.eval.consistency_model import (
    ConsistencyOut,
    EvidencePack,
    allowed_obvious,
    judge_audit,
    validate_output,
)
from twin.eval.consistency_pack import build_pack, render_pack
from twin.eval.consistency_store import ConsistencyStore, FindingView, NewFinding
from twin.eval.isolation import (
    ALLOWED_SETTING_KEYS,
    ALLOWED_TABLES,
    WriteScope,
    isolated_writes,
)
from twin.eval.store import EvalStore, RunView
from twin.llm.budget import LEVEL_KEY, NOTIFIED_KEY
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.types import DAILY, JsonResult, Purpose
from twin.memory.api import Memory
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import CONSISTENCY_AUDIT, TemplateStore
from twin.services import Services

log = get_logger("twin.eval.consistency")

# what an audit may write: the evaluation's own tables, the findings, and the settings the
# budget keeps while a call is paid for (a daily-account call updates the level it is under)
AUDIT_WRITES = WriteScope(
    tables=ALLOWED_TABLES | {"consistency_findings"},
    setting_keys=ALLOWED_SETTING_KEYS | {NOTIFIED_KEY, LEVEL_KEY},
)
EMPTY_WINDOW = "empty_window"


@dataclass(frozen=True)
class AuditOutcome:
    """What an audit did (counts only; the findings are in the store)."""

    run: RunView
    called: bool  # DeepSeek was asked (an empty window is not)
    findings: int  # waiting for a decision
    inherited: int  # carried over from an earlier decision about the same pair
    dropped: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    attempts: int = 0


def _counts(findings: list[FindingView]) -> dict[str, int]:
    return {
        "confirmed_obvious": sum(
            f.status == "confirmed" and f.severity == "obvious" for f in findings
        ),
        "confirmed_minor": sum(f.status == "confirmed" and f.severity == "minor" for f in findings),
        "rejected": sum(f.status == "rejected" for f in findings),
        "undecided": sum(f.status == "proposed" for f in findings),
    }


def refresh_run(estore: EvalStore, cstore: ConsistencyStore, run_id: str) -> RunView:
    """Bring the summary of a run up to date and close it when no finding is left undecided."""
    run = estore.get_run(run_id)
    if run.summary.get("reason") == EMPTY_WINDOW:
        return run  # nothing was audited: there is nothing to recount
    findings = cstore.findings(run_id)
    decisions = _counts(findings)
    fixes = cstore.run_fixes(run_id)
    days = int(run.params.get("days", 0))
    verdict, why = judge_audit(
        days=days,
        confirmed_obvious=decisions["confirmed_obvious"],
        undecided=decisions["undecided"],
    )
    summary: dict[str, Any] = {
        "decisions": decisions,
        "fixes": {
            state: sum(fix.status == state for fix in fixes)
            for state in ("proposed", "applied", "declined", "stale")
        },
        "allowed_obvious": float(allowed_obvious(days)),
        "verdict_reason": why,
    }
    if decisions["undecided"]:
        return estore.update_run(run_id, status="running", summary=summary)
    return estore.update_run(run_id, status="done", verdict=verdict, summary=summary)


async def run_audit(
    services: Services,
    client: DeepSeekClient,
    *,
    days: int | None = None,
    memory: Memory | None = None,
) -> AuditOutcome:
    """Audit the last ``days`` days (default ``eval.consistency_days``); see the module text.

    Raises what the model layer raises (a budget that does not allow the call, an open circuit, a
    reply that cannot be used); the run is then left as ``cancelled`` (the call was held back) or
    ``failed`` and the job queue decides about trying again.
    """
    window = days if days is not None else services.settings.eval.consistency_days
    estore = EvalStore(services.db, services.clock)
    cstore = ConsistencyStore(services.db, services.clock)
    # the template is installed on first use, which is a write the fence would refuse
    template = await asyncio.to_thread(
        TemplateStore(services.db, services.clock).active, CONSISTENCY_AUDIT
    )
    pack = await asyncio.to_thread(build_pack, services, days=window, memory=memory)
    decided = await asyncio.to_thread(cstore.decisions)
    params: dict[str, Any] = {
        "days": window,
        "window_start": pack.window_start.isoformat(),
        "window_end": pack.window_end.isoformat(),
        "zone": pack.zone,
        "template": template.ref,
    }
    evidence = {
        "lifeline": len(pack.of("lifeline")),
        "replies": len(pack.of("reply")),
        "facts": len(pack.of("fact")),
        "chars": pack.chars,
        "replies_cut": pack.reply_turns_cut,
    }
    with isolated_writes(AUDIT_WRITES):
        run = await asyncio.to_thread(
            estore.create_run,
            "consistency",
            mode="live",
            backends=["deepseek"],
            params=params,
            status="running",
            summary={"evidence": evidence},
        )
        if pack.empty:
            return await asyncio.to_thread(_empty_window, estore, run)
        try:
            answer = await client.chat_json(
                template.render(**render_pack(pack)),
                ConsistencyOut,
                purpose=Purpose.CONSISTENCY,
                thinking=True,
                tag=DAILY,
            )
        except (BudgetDeniedError, CircuitOpenError) as exc:
            await asyncio.to_thread(
                estore.update_run,
                run.id,
                status="cancelled",
                summary={"held_back": type(exc).__name__},
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(
                estore.update_run, run.id, status="failed", summary={"error": type(exc).__name__}
            )
            raise
        return await asyncio.to_thread(_store_answer, estore, cstore, run, pack, answer, decided)


def _empty_window(estore: EvalStore, run: RunView) -> AuditOutcome:
    """Nothing was planned and nothing was said: there is no one to contradict, and no call.

    That is not a pass either - a week without anything to audit has not shown that the bot is
    consistent - so the run is closed as ``insufficient``.
    """
    closed = estore.update_run(
        run.id,
        status="done",
        verdict="insufficient",
        summary={
            "reason": EMPTY_WINDOW,
            "verdict_reason": "这段时间既没有生活安排也没有她说过的话：没有可审阅的内容",
            "findings": {"reported": 0, "stored": 0, "dropped": {}, "inherited": 0},
            "decisions": {
                "confirmed_obvious": 0,
                "confirmed_minor": 0,
                "rejected": 0,
                "undecided": 0,
            },
        },
    )
    return AuditOutcome(closed, called=False, findings=0, inherited=0)


def _store_answer(
    estore: EvalStore,
    cstore: ConsistencyStore,
    run: RunView,
    pack: EvidencePack,
    answer: JsonResult[ConsistencyOut],
    decided: dict[str, FindingView],
) -> AuditOutcome:
    out = answer.value
    checked = validate_output(out, pack)
    items: list[NewFinding] = []
    inherited = 0
    for found in checked.findings:
        earlier = decided.get(found.fingerprint)
        if earlier is None:
            items.append(NewFinding(found))
        else:
            inherited += 1
            items.append(NewFinding(found, earlier.status, earlier.severity, earlier.id))
    cstore.add_findings(run.id, items)
    waiting = sum(item.status == "proposed" for item in items)
    estore.update_run(
        run.id,
        summary={
            "model": {
                "calls": answer.attempts,
                "cost_usd": round(answer.total_cost_usd, 6),
                "model": answer.chat.model,
            },
            "findings": {
                "reported": len(out.contradictions),
                "stored": len(items),
                "dropped": dict(checked.dropped),
                "inherited": inherited,
            },
        },
    )
    closed = refresh_run(estore, cstore, run.id)
    log.info(
        "consistency_audited",
        run_id=run.id,
        days=pack.days,
        reported=len(out.contradictions),
        waiting=waiting,
        dropped=checked.dropped_total,
        cost_usd=round(answer.total_cost_usd, 6),
    )
    return AuditOutcome(
        closed,
        called=True,
        findings=waiting,
        inherited=inherited,
        dropped=dict(checked.dropped),
        cost_usd=answer.total_cost_usd,
        attempts=answer.attempts,
    )
