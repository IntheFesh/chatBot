"""The ``training_export`` job: ``twin train export`` runs as a heavy task (R-ARCH-006, R-LLM-014).

The command only queues the job.  The job

1. finds the pinned tokenizer (:mod:`twin.training.tokenizer`),
2. makes the first pass over her reply blocks and learns which plans are still missing
   (:meth:`~twin.training.export.TrainingSetExporter.run`),
3. when plans are missing, registers them, prices the batches and queues them waiting for
   ``twin jobs approve <batch>`` (R-LLM-014), and ends - the same command run again after the
   plans are written completes the export,
4. otherwise writes the dataset directory and records the version in ``dataset_versions``.

How it ended is kept in the setting ``training.export.state`` (counts, versions, batch ids; never
text), which ``twin train export-status`` shows.  A job that fails because of the data (too few
samples, a finding of the desensitising scan) is not retried: nothing would change.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from twin.llm.runtime import build_llm_runtime
from twin.ops.jobs import JobContext, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.storage.settings_store import get_setting, put_setting
from twin.training.export import ExportError, ExportOptions, ExportResult, TrainingSetExporter
from twin.training.plans import PlanQueueResult, PlanStore, queue_plan_batches
from twin.training.runs import ensure_dataset_version
from twin.training.tokenizer import Fetch, TokenizerError, ensure_tokenizer, http_fetch

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.training.export_job")

EXPORT_JOB = "training_export"
EXPORT_PRIORITY = 80
STATE_KEY = "training.export.state"


@dataclass(frozen=True)
class ExportRequest:
    """The arguments of ``twin train export`` (the payload of the job; no content)."""

    since: datetime | None = None
    until: datetime | None = None
    out_dir: Path | None = None
    tokenizer: Path | None = None
    plan_ratio: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "out_dir": str(self.out_dir) if self.out_dir else None,
            "tokenizer": str(self.tokenizer) if self.tokenizer else None,
            "plan_ratio": self.plan_ratio,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExportRequest:
        def moment(value: object) -> datetime | None:
            return datetime.fromisoformat(str(value)) if value else None

        ratio = payload.get("plan_ratio")
        return cls(
            since=moment(payload.get("since")),
            until=moment(payload.get("until")),
            out_dir=Path(str(payload["out_dir"])) if payload.get("out_dir") else None,
            tokenizer=Path(str(payload["tokenizer"])) if payload.get("tokenizer") else None,
            plan_ratio=float(ratio) if ratio is not None else None,
        )


@dataclass(frozen=True)
class QueuedExport:
    job_id: str
    already_queued: bool


def local_range(
    services: Services, first: date | None, last: date | None
) -> tuple[datetime | None, datetime | None]:
    """``[start of first, end of last)`` as UTC moments, for local calendar days (inclusive)."""
    from twin.schedule.service import time_service_for

    time = time_service_for(services)
    since = time.day_bounds_utc(first)[0] if first else None
    until = time.day_bounds_utc(last + timedelta(days=1))[0] if last else None
    return since, until


def queue_export(services: Services, request: ExportRequest) -> QueuedExport:
    """Queue the export unless one is already waiting or running."""
    queue = JobQueue(services.db, services.clock)
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=EXPORT_JOB, limit=10):
            return QueuedExport(job.id, True)
    job_id = queue.enqueue(
        EXPORT_JOB, request.to_payload(), priority=EXPORT_PRIORITY, max_attempts=1
    )
    return QueuedExport(job_id, False)


# --------------------------------------------------------------------------- the state


def save_state(services: Services, state: dict[str, Any]) -> None:
    stamped = {"at": services.clock.now_utc().isoformat(), **state}
    with services.db.transaction(bump_state=False) as session:
        put_setting(session, STATE_KEY, stamped, clock=services.clock, record_history=False)


def read_state(services: Services) -> dict[str, Any] | None:
    with services.db.session() as session:
        value = get_setting(session, STATE_KEY, None)
    return dict(value) if isinstance(value, dict) else None


# --------------------------------------------------------------------------- the work


def run_export(
    services: Services, request: ExportRequest, *, fetch: Fetch = http_fetch
) -> ExportResult:
    """Everything the job does that does not need the event loop (call it from a thread)."""
    tokenizer = ensure_tokenizer(
        services.paths.data_dir / "training" / "tokenizer", explicit=request.tokenizer, fetch=fetch
    )
    exporter = TrainingSetExporter(
        services,
        tokenizer,
        ExportOptions(
            since=request.since,
            until=request.until,
            out_dir=request.out_dir,
            plan_ratio=request.plan_ratio,
        ),
    )
    result = exporter.run()
    if result.dataset is not None:
        directory = result.dataset.path
        try:
            relative = directory.resolve().relative_to(services.paths.data_dir.resolve()).as_posix()
        except ValueError:
            relative = str(directory.resolve())
        ensure_dataset_version(services.db, result.dataset, relative)
    return result


async def queue_missing_plans(services: Services, result: ExportResult) -> PlanQueueResult:
    """Register the plans an export lacks and queue them as priced batches."""
    store = PlanStore(services.db)
    store.register(result.missing_plans)
    wanted = sorted({*result.missing_plans, *store.pending_ids()})
    runtime = build_llm_runtime(services)
    try:
        return await asyncio.to_thread(queue_plan_batches, services, runtime, wanted)
    finally:
        await runtime.client.aclose()


def summary_of(result: ExportResult) -> dict[str, Any]:
    dataset = result.dataset
    if dataset is None:
        return {}
    counts = dataset.meta.counts
    return {
        "dataset_version": dataset.meta.dataset_version,
        "directory": str(dataset.path),
        "train": counts.train,
        "val": counts.val,
        "test": counts.test,
    }


@job_handler(EXPORT_JOB)
async def handle_training_export(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("training_export needs the services container")
    request = ExportRequest.from_payload(ctx.job.payload)
    try:
        result = await asyncio.to_thread(run_export, services, request)
    except (ExportError, TokenizerError) as exc:
        save_state(services, {"state": "failed", "message": str(exc)})
        log.warning("training_export_failed", error=type(exc).__name__)
        raise
    if result.state == "waiting_for_plans":
        queued = await queue_missing_plans(services, result)
        save_state(
            services,
            {
                "state": "waiting_for_plans",
                "selected": result.plan_selected,
                "missing": len(result.missing_plans),
                "queued_samples": queued.samples,
                "already_queued": queued.already_queued,
                "batches": list(queued.batch_ids),
                "estimated_usd": round(queued.estimated_usd, 4),
            },
        )
        log.info(
            "training_export_waits_for_plans",
            missing=len(result.missing_plans),
            batches=len(queued.batch_ids),
        )
        return
    save_state(services, {"state": "done", **summary_of(result), "stats": result.stats})
    log.info(
        "training_export_done", **{k: v for k, v in summary_of(result).items() if k != "directory"}
    )
