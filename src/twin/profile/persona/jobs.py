"""Planning and running the persona jobs (R-LLM-014, R-ARCH-003, R-IMP-011).

``persona_generate``
    writes the automatic description of one scope (Map-Reduce over a stratified sample).  It is
    a *one-time batch* job: :func:`plan_generation` prices it with the token estimator, queues it
    waiting for ``twin jobs approve <batch>`` and only after the approval does a worker run it;
    its model calls are recorded on the ``one_time`` account of that batch.  The estimate is an
    upper bound computed from the settings alone (every segment at its longest), so planning
    needs neither the messages nor the API key.
``persona_refresh``
    rewrites only the statistics section (and copies the hand-written style lines to the
    pre-holdout card).  It costs nothing, but it must see the profile that the same import
    recomputed, so it hands itself back while a ``profile_rebuild`` job is still waiting.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from twin.ingest.transcript import MESSAGE_CHAR_CAP
from twin.llm.errors import BudgetDeniedError, CircuitOpenError
from twin.llm.onetime import BatchItem
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import LedgerTag
from twin.ops.jobs import JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona.generate import (
    MAX_ITEMS,
    GenerationResult,
    description_block,
    generate_digest,
)
from twin.profile.persona.refresh import (
    count_her_messages,
    refresh_all,
    write_description,
)
from twin.profile.persona.sampling import SampleRequest, build_sample
from twin.profile.prompt_templates import PERSONA_MAP, PERSONA_REDUCE, TemplateStore
from twin.profile.queue import PROFILE_JOB
from twin.services import Services
from twin.storage.ids import new_id

log = get_logger("twin.profile.persona")

PERSONA_JOB = "persona_generate"
REFRESH_JOB = "persona_refresh"
GENERATE_PRIORITY = 80
REFRESH_PRIORITY = 65
MAP_COMPLETION_TOKENS = 1500
REDUCE_COMPLETION_TOKENS = 2000
PROFILE_WAIT_S = 60.0


@dataclass(frozen=True)
class PlannedScope:
    scope: str
    job_id: str
    estimated_usd: float


@dataclass
class PersonaPlan:
    """What :func:`plan_generation` queued."""

    batch_id: str | None
    scopes: list[PlannedScope] = field(default_factory=list)
    already_queued: list[str] = field(default_factory=list)

    @property
    def estimated_usd(self) -> float:
        return sum(item.estimated_usd for item in self.scopes)


def _queued_scopes(queue: JobQueue) -> set[str]:
    found: set[str] = set()
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=PERSONA_JOB, limit=1000):
            found.add(str(job.payload.get("scope")))
    return found


def estimate_items(services: Services, runtime: LlmRuntime) -> list[BatchItem]:
    """The model calls of one scope, each at its upper bound (the segments at full length)."""
    config = services.settings.persona
    model = services.settings.deepseek.offline_model
    templates = TemplateStore(services.db, services.clock)
    map_template = templates.active(PERSONA_MAP)
    reduce_template = templates.active(PERSONA_REDUCE)
    estimator = runtime.estimator
    per_segment = estimator.estimate_text(
        "字" * (config.segment_max_messages * (MESSAGE_CHAR_CAP + 6))
    )
    batches = math.ceil(config.sample_segments / config.batch_segments)
    map_base = estimator.estimate_messages(map_template.render(count=0, segments=""))
    items = [
        BatchItem(
            model,
            map_base + min(config.batch_segments, config.sample_segments) * per_segment,
            MAP_COMPLETION_TOKENS,
        )
        for _ in range(batches)
    ]
    if batches > 1:
        reduce_base = estimator.estimate_messages(
            reduce_template.render(count=0, items="", max_items=MAX_ITEMS)
        )
        items.append(
            BatchItem(
                model, reduce_base + batches * MAP_COMPLETION_TOKENS, REDUCE_COMPLETION_TOKENS
            )
        )
    return items


def seed_for(batch_id: str, scope: str) -> int:
    digest = hashlib.sha256(f"{batch_id}/{scope}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def plan_generation(
    services: Services,
    scopes: list[str],
    *,
    reason: str,
    runtime: LlmRuntime | None = None,
) -> PersonaPlan:
    """Queue the generation of the description for ``scopes`` as one one-time batch."""
    if "pre_holdout" in scopes:
        holdout_cutoff(services)  # raises HoldoutError when there is too little data to split
    llm = runtime or build_llm_runtime(services)
    queue = JobQueue(services.db, services.clock)
    waiting = _queued_scopes(queue)
    plan = PersonaPlan(None)
    wanted: list[str] = []
    for scope in scopes:
        (plan.already_queued if scope in waiting else wanted).append(scope)
    if not wanted:
        return plan
    items = estimate_items(services, llm)
    per_scope = llm.batches.estimate(items).total_usd
    batch_id = f"persona-{services.clock.now_utc():%Y%m%d%H%M%S}-{new_id()[-4:].lower()}"
    payloads = [
        {"scope": scope, "batch_id": batch_id, "seed": seed_for(batch_id, scope), "reason": reason}
        for scope in wanted
    ]
    ids = llm.batches.enqueue(
        batch_id,
        PERSONA_JOB,
        payloads,
        [per_scope] * len(payloads),
        priority=GENERATE_PRIORITY,
        offpeak_only=True,
    )
    plan.batch_id = batch_id
    plan.scopes = [PlannedScope(s, i, per_scope) for s, i in zip(wanted, ids, strict=True)]
    return plan


# ------------------------------------------------------------------- generating


def provenance_of(
    result: GenerationResult,
    sample_labels: dict[str, Any],
    statements: list[dict[str, Any]],
    *,
    seed: int,
    scope: str,
    cutoff: datetime | None,
    dropped: int,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "seed": seed,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "segments": sample_labels,
        "statements": statements,
        "dropped_without_evidence": dropped,
        "map_calls": result.map_calls,
    }


async def generate_scope(
    services: Services,
    runtime: LlmRuntime,
    scope: str,
    *,
    seed: int,
    tag: LedgerTag,
) -> str:
    """Sample, run the Map-Reduce and write the new card version; returns its id."""
    config = services.settings.persona
    templates = TemplateStore(services.db, services.clock)
    map_template = templates.active(PERSONA_MAP)
    reduce_template = templates.active(PERSONA_REDUCE)
    request = SampleRequest(scope, seed, config.sample_segments, config.segment_max_messages)
    sample = await asyncio.to_thread(build_sample, services, request)
    result = await generate_digest(
        runtime.client,
        sample,
        map_template=map_template,
        reduce_template=reduce_template,
        batch_size=config.batch_segments,
        tag=tag,
    )
    body, statements = description_block(result.digest)
    segments = {
        segment.label: {
            "start": segment.info.start.isoformat(),
            "end": segment.info.end.isoformat(),
            "messages": list(segment.message_ids),
        }
        for segment in sample.segments
    }
    her_messages = await asyncio.to_thread(count_her_messages, services, scope)
    card = await asyncio.to_thread(
        write_description,
        services,
        scope,
        body,
        provenance=provenance_of(
            result,
            segments,
            statements,
            seed=seed,
            scope=scope,
            cutoff=sample.cutoff,
            dropped=result.dropped,
        ),
        template_version=f"{map_template.ref},{reduce_template.ref}",
        her_messages=her_messages,
        at=services.clock.now_utc(),
    )
    log.info(
        "persona_generated",
        scope=scope,
        version=card.number,
        statements=len(statements),
        dropped=result.dropped,
        cost_usd=round(result.cost_usd, 4),
    )
    return card.id


@job_handler(PERSONA_JOB)
async def handle_persona_generate(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("persona_generate needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    tag = LedgerTag("one_time", str(batch_id)) if batch_id else LedgerTag()
    runtime = build_llm_runtime(services)
    try:
        await generate_scope(
            services, runtime, str(payload["scope"]), seed=int(payload.get("seed", 0)), tag=tag
        )
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()


# ---------------------------------------------------------------------- refreshing


def profile_rebuild_waiting(services: Services) -> bool:
    queue = JobQueue(services.db, services.clock)
    return any(
        queue.list_jobs(status=status, job_type=PROFILE_JOB, limit=1)
        for status in ("pending", "running")
    )


def queue_refresh(services: Services, *, scope: str = "all", reason: str = "manual") -> str:
    """Queue a statistics refresh unless an identical one is already waiting; returns the job id."""
    queue = JobQueue(services.db, services.clock)
    for job in queue.list_jobs(status="pending", job_type=REFRESH_JOB, limit=200):
        if job.payload.get("scope") == scope:
            return job.id
    return queue.enqueue(
        REFRESH_JOB, {"scope": scope, "reason": reason}, priority=REFRESH_PRIORITY, max_attempts=3
    )


@job_handler(REFRESH_JOB)
async def handle_persona_refresh(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("persona_refresh needs the services container")
    if await asyncio.to_thread(profile_rebuild_waiting, services):
        raise JobDeferred("the profile is being recomputed", retry_in_s=PROFILE_WAIT_S)
    results = await asyncio.to_thread(
        refresh_all, services, str(ctx.job.payload.get("scope", "all"))
    )
    for result in results:
        log.info("persona_refreshed", scope=result.scope, status=result.status)
