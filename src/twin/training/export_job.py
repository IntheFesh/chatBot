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

The ``training_export_dpo`` job is ``twin train export-dpo`` as a heavy task (round 17 item 29,
D-713): the command only queues it, the application runs
:func:`~twin.training.dpo_export.export_dpo` in a thread, and how it ended is kept in the setting
``training.export_dpo.state`` (the same kind of record: counts and versions, never text), which
``twin train export-status`` shows as well.  The SFT job above never calls it; the only link to the
preference pairs is that one function of the module that is allowed to read them.
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
from twin.training.dataset_dir import DatasetError
from twin.training.dpo_export import DpoExportResult, export_dpo
from twin.training.export import ExportError, ExportOptions, ExportResult, TrainingSetExporter
from twin.training.plans import PlanQueueResult, PlanStore, queue_plan_batches
from twin.training.registry import RegistryError
from twin.training.runs import RunError, ensure_dataset_version
from twin.training.tokenizer import Fetch, TokenizerError, ensure_tokenizer, http_fetch

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.training.export_job")

EXPORT_JOB = "training_export"
EXPORT_PRIORITY = 80
STATE_KEY = "training.export.state"
EXPORT_DPO_JOB = "training_export_dpo"
EXPORT_DPO_PRIORITY = 80
DPO_STATE_KEY = "training.export_dpo.state"


@dataclass(frozen=True)
class ExportRequest:
    """The arguments of ``twin train export`` (the payload of the job; no content)."""

    since: datetime | None = None
    until: datetime | None = None
    out_dir: Path | None = None
    tokenizer: Path | None = None
    plan_ratio: float | None = None
    allow_incomplete: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
            "out_dir": str(self.out_dir) if self.out_dir else None,
            "tokenizer": str(self.tokenizer) if self.tokenizer else None,
            "plan_ratio": self.plan_ratio,
            "allow_incomplete": self.allow_incomplete,
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
            allow_incomplete=bool(payload.get("allow_incomplete", False)),
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


@dataclass(frozen=True)
class DpoRequest:
    """The arguments of ``twin train export-dpo`` (the payload of the job; paths, no content).

    The paths are absolute: the application that runs the job has another working directory than
    the terminal the command was typed in.
    """

    dataset: Path | None = None
    out_dir: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset": str(self.dataset) if self.dataset else None,
            "out_dir": str(self.out_dir) if self.out_dir else None,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DpoRequest:
        return cls(
            dataset=Path(str(payload["dataset"])) if payload.get("dataset") else None,
            out_dir=Path(str(payload["out_dir"])) if payload.get("out_dir") else None,
        )


def queue_export_dpo(services: Services, request: DpoRequest) -> QueuedExport:
    """Queue the DPO export unless one is already waiting or running."""
    queue = JobQueue(services.db, services.clock)
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=EXPORT_DPO_JOB, limit=10):
            return QueuedExport(job.id, True)
    job_id = queue.enqueue(
        EXPORT_DPO_JOB, request.to_payload(), priority=EXPORT_DPO_PRIORITY, max_attempts=1
    )
    return QueuedExport(job_id, False)


# --------------------------------------------------------------------------- the state


def save_state(services: Services, state: dict[str, Any], *, key: str = STATE_KEY) -> None:
    stamped = {"at": services.clock.now_utc().isoformat(), **state}
    with services.db.transaction(bump_state=False) as session:
        put_setting(session, key, stamped, clock=services.clock, record_history=False)


def read_state(services: Services, *, key: str = STATE_KEY) -> dict[str, Any] | None:
    with services.db.session() as session:
        value = get_setting(session, key, None)
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
            allow_incomplete=request.allow_incomplete,
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
    return await asyncio.to_thread(queue_plan_batches, services, runtime, wanted)


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


def dpo_summary(result: DpoExportResult) -> dict[str, Any]:
    """What ``export-status`` shows about a DPO export: counts and versions, never text."""
    meta = result.dataset.meta
    return {
        "dataset_version": meta.dataset_version,
        "directory": str(result.dataset.path),
        "persona_version": meta.persona_version,
        "template_version": meta.template_version,
        "pairs": result.pairs,
        "trained": result.trained,
        "held_out": result.held_out,
        "after_cutoff": result.after_cutoff,
        "cutoff": result.cutoff.isoformat() if result.cutoff is not None else None,
        "regenerated": result.regenerated,
        "skipped": dict(result.skipped),
        "minimum": result.minimum,
        "enough": result.enough,
    }


@job_handler(EXPORT_DPO_JOB)
async def handle_training_export_dpo(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("training_export_dpo needs the services container")
    request = DpoRequest.from_payload(ctx.job.payload)
    try:
        result = await asyncio.to_thread(
            export_dpo, services, dataset=request.dataset, out_dir=request.out_dir
        )
    except (ExportError, DatasetError, RegistryError, RunError) as exc:
        save_state(services, {"state": "failed", "message": str(exc)}, key=DPO_STATE_KEY)
        log.warning("training_export_dpo_failed", error=type(exc).__name__)
        raise
    summary = dpo_summary(result)
    save_state(services, {"state": "done", **summary}, key=DPO_STATE_KEY)
    log.info(
        "training_export_dpo_done",
        **{k: v for k, v in summary.items() if k in ("dataset_version", "pairs", "regenerated")},
    )


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
