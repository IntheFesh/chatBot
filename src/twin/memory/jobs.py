"""The memory's jobs: replay, daily summary, extraction from the bot's conversation.

``memory_replay``
    replays some local days of the real history (:mod:`twin.memory.replay`).  Queued as a
    one-time batch waiting for ``twin jobs approve`` (R-LLM-014); its model calls are recorded
    on the ``one_time`` account of the batch, so a paused batch (spend above 120 % of the
    estimate) makes the next call raise :class:`~twin.llm.onetime.BatchPausedError`, a
    :class:`~twin.ops.jobs.JobDeferred`: the job goes back to the queue.
``memory_summary``
    summarises one local day of one scope (R-MEM-002).  The scheduler (round 08) queues it with
    :func:`queue_daily_summary` before she "wakes", off peak.  The ``bot`` scope reads the bot's
    conversation through the reader round 09 registers (:mod:`twin.memory.recent`) and does
    nothing while there is none.
``memory_extract``
    extracts facts and follow-ups from a stretch of the bot's conversation (R-MEM-007).  Round 09
    calls :func:`queue_bot_extraction` once a conversation has been quiet for
    ``memory.quiet_minutes``; the turns travel in the job, so the job needs no table of its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.runtime import build_llm_runtime
from twin.llm.types import DAILY, LedgerTag
from twin.memory.dayload import day_lines
from twin.memory.extract import DialogueLine, FactExtractor
from twin.memory.followups import FollowupStore
from twin.memory.localdate import BOT, REAL
from twin.memory.memory import Memory
from twin.memory.recent import BotMessage, bot_turn_reader
from twin.memory.replay import REPLAY_JOB, MemoryReplayer
from twin.memory.summarize import DailySummarizer
from twin.memory.writer import MemoryWriter
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import TemplateStore
from twin.services import Services

log = get_logger("twin.memory.jobs")

SUMMARY_JOB = "memory_summary"
EXTRACT_JOB = "memory_extract"
SUMMARY_PRIORITY = 70
EXTRACT_PRIORITY = 75
OPEN_FOLLOWUPS_SHOWN = 10
MAX_TURNS_PER_JOB = 400


# ----------------------------------------------------------------------- queueing


def queue_daily_summary(
    services: Services,
    day: date,
    scopes: Sequence[str] = (REAL, BOT),
    *,
    force: bool = False,
    deadline: datetime | None = None,
) -> list[str]:
    """Queue the summary of ``day`` for each scope (off peak); returns the new job ids.

    A scope whose summary of that day is already waiting in the queue is not queued twice.
    """
    queue = JobQueue(services.db, services.clock)
    waiting = {
        (str(job.payload.get("date")), str(job.payload.get("scope")))
        for status in ("pending", "running")
        for job in queue.list_jobs(status=status, job_type=SUMMARY_JOB, limit=10_000)
    }
    ids: list[str] = []
    for scope in scopes:
        if scope not in (REAL, BOT):
            raise ValueError(f"unknown summary scope {scope!r}")
        if (day.isoformat(), scope) in waiting:
            continue
        ids.append(
            queue.enqueue(
                SUMMARY_JOB,
                {"date": day.isoformat(), "scope": scope, "force": force},
                priority=SUMMARY_PRIORITY,
                offpeak_only=True,
                deadline=deadline,
            )
        )
    return ids


def queue_bot_extraction(services: Services, turns: Sequence[BotMessage]) -> str | None:
    """Queue the extraction of facts and follow-ups from a stretch of the bot's conversation.

    ``turns`` are the messages of the conversation since the last extraction, oldest first
    (round 09 calls this after ``memory.quiet_minutes`` of silence).  Returns the job id, or
    ``None`` when fact extraction is switched off or there is nothing to extract from.
    """
    if not services.settings.memory.fact_extraction or not turns:
        return None
    payload = {
        "turns": [
            {"id": t.id, "role": t.role, "text": t.text, "at": t.at.isoformat()}
            for t in turns[-MAX_TURNS_PER_JOB:]
        ]
    }
    queue = JobQueue(services.db, services.clock)
    return queue.enqueue(EXTRACT_JOB, payload, priority=EXTRACT_PRIORITY, max_attempts=3)


def is_quiet(last_message_at: datetime, now: datetime, quiet_minutes: int) -> bool:
    """True once a conversation has been silent for ``quiet_minutes`` (R-MEM-007)."""
    return now - last_message_at >= timedelta(minutes=quiet_minutes)


# ---------------------------------------------------------------------- handlers


def _services_of(ctx: JobContext) -> Services:
    if ctx.services is None:
        raise RuntimeError("the memory jobs need the services container")
    return ctx.services


@job_handler(REPLAY_JOB)
async def handle_memory_replay(ctx: JobContext) -> None:
    services = _services_of(ctx)
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    tag = LedgerTag("one_time", str(batch_id)) if batch_id else DAILY
    days = [date.fromisoformat(str(d)) for d in payload.get("dates", [])]
    runtime = build_llm_runtime(services)
    replayer = MemoryReplayer(Memory(services), runtime.client)
    try:
        run = await replayer.replay_days(
            days,
            tag,
            force=bool(payload.get("force", False)),
            batch_id=str(batch_id) if batch_id else None,
        )
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    log.info(
        "memory_replayed",
        days=len(days),
        replayed=run.replayed,
        skipped=run.skipped,
        facts=run.facts,
        failed=len(run.failures),
    )
    if run.failures:
        raise RuntimeError(f"{len(run.failures)} day(s) could not be replayed yet; the job retries")


@job_handler(SUMMARY_JOB)
async def handle_memory_summary(ctx: JobContext) -> None:
    services = _services_of(ctx)
    payload = ctx.job.payload
    day = date.fromisoformat(str(payload["date"]))
    scope = str(payload["scope"])
    memory = Memory(services)
    reader = bot_turn_reader(services)
    if scope == BOT and reader is None:
        log.info("memory_summary_skipped", reason="no_bot_conversation", day=day.isoformat())
        return
    lines = await asyncio.to_thread(day_lines, services, memory.clock, scope, day, reader)
    runtime = build_llm_runtime(services)
    summarizer = DailySummarizer(memory, runtime.client)
    try:
        result = await summarizer.summarize(
            scope, day, lines, DAILY, force=bool(payload.get("force", False))
        )
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    log.info(
        "memory_summary_job",
        scope=scope,
        day=day.isoformat(),
        written=result.record is not None and result.skipped is None,
        skipped=result.skipped,
    )


@job_handler(EXTRACT_JOB)
async def handle_memory_extract(ctx: JobContext) -> None:
    services = _services_of(ctx)
    turns = [
        BotMessage(
            str(t["id"]),
            "bot" if t["role"] == "bot" else "user",
            str(t["text"]),
            datetime.fromisoformat(str(t["at"])),
        )
        for t in ctx.job.payload.get("turns", [])
    ]
    if not turns or not services.settings.memory.fact_extraction:
        return
    memory = Memory(services)
    memory.note_bot_activity(turns[0].at)
    lines = [DialogueLine(t.id, t.role, " ".join(t.text.split()), t.at) for t in turns]
    runtime = build_llm_runtime(services)
    templates = TemplateStore(services.db, services.clock)
    extractor = FactExtractor(services, runtime.client, memory.clock, templates=templates)
    writer = MemoryWriter(memory, runtime.client, templates=templates)
    try:
        open_followups = await asyncio.to_thread(
            lambda: FollowupStore(memory).open()[:OPEN_FOLLOWUPS_SHOWN]
        )
        extraction = await extractor.extract(
            lines, mode="bot", tag=DAILY, open_followups=open_followups
        )
        report = await writer.write(extraction, DAILY)
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    log.info(
        "memory_extracted",
        facts=report.facts_added,
        merged=report.merged,
        rejected=report.rejected,
        followups=report.followups_added,
        closed=report.followups_closed,
        improvised=report.improvised_events,
    )
