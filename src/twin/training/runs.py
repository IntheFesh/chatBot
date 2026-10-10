"""Bookkeeping of the training runs and the dataset versions (R-TRN-010, R-PRIV-003).

``training_runs`` has one row per run on a rented machine.  Every step of the remote procedure
(upload, setup, train, dpo, eval, export, download, cleanup) records when it started and ended,
its exit code and whether it succeeded, under ``steps``; the columns the spec lists (GPU model,
peak memory, best validation loss, artifact hashes, the time of the cleanup) are filled in as the
numbers become known.  A run that has put data on an instance and has no ``cleaned_at`` is
*uncleaned*: ``uncleaned_runs`` lists them, so ``/状态`` and ``twin train remote status`` can keep
reminding the user to run the cleanup and release the instance.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from sqlalchemy import select

from twin.storage.db import Database
from twin.storage.training_models import DatasetVersion, TrainingRun
from twin.training import lf_template
from twin.training.dataset_dir import DatasetDir
from twin.training.profiles import (
    CUTOFF_LEN,
    LEARNING_RATE,
    TrainingProfile,
    epochs_for,
)

STEP_STATUS: Final = {
    "upload": "uploaded",
    "setup": "set_up",
    "decrypt": "set_up",
    "verify": "set_up",
    "train": "trained",
    "dpo": "dpo_done",
    "eval": "evaluated",
    "export": "exported",
    "download": "downloaded",
    "cleanup": "cleaned",
}


class RunError(ValueError):
    """A run or dataset version does not exist, or is not what the caller thinks."""


@dataclass(frozen=True)
class RunView:
    id: str
    profile: str
    dataset_version: str
    status: str
    steps: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None
    gpu_model: str | None
    peak_vram_mib: int | None
    best_val_loss: float | None
    best_checkpoint: str | None
    bundle_sha256: str | None
    artifacts: dict[str, Any]
    cleaned_at: datetime | None
    last_error: str | None
    hyperparameters: dict[str, Any]

    @property
    def needs_cleanup(self) -> bool:
        uploaded = bool(self.steps.get("upload", {}).get("ok"))
        return uploaded and self.cleaned_at is None


def new_run_id(profile: str, now: datetime) -> str:
    return f"r{now.astimezone(UTC):%Y%m%d-%H%M%S}-{profile}-{secrets.token_hex(2)}"


def hyperparameters(
    profile: TrainingProfile, train_samples: int, *, dpo_min_pairs: int = 200
) -> dict[str, Any]:
    """The settings of the run that are not in the YAML files' file names (for the run record)."""
    return {
        "base_model": profile.base_model,
        "method": profile.method,
        "lora_rank": profile.lora_rank,
        "lora_alpha": profile.lora_alpha,
        "quantization_bit": profile.quantization_bit,
        "per_device_train_batch_size": profile.per_device_train_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "learning_rate": LEARNING_RATE,
        "epochs": epochs_for(train_samples),
        "cutoff_len": CUTOFF_LEN,
        "template": lf_template.LF_TEMPLATE_NAME,
        "llamafactory_version": lf_template.LLAMAFACTORY_VERSION,
        "train_samples": train_samples,
        "dpo_min_pairs": dpo_min_pairs,
    }


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _parse(moment: str) -> datetime:
    parsed = datetime.fromisoformat(moment)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def ensure_dataset_version(db: Database, dataset: DatasetDir, directory: str) -> str:
    """Record an exported dataset (once); returns its version id.

    A version id names exactly one content: recording the same id with other file hashes fails.
    """
    meta = dataset.meta
    with db.transaction() as session:
        row = session.get(DatasetVersion, meta.dataset_version)
        if row is not None:
            if row.files != meta.files:
                raise RunError(
                    f"dataset version {meta.dataset_version} exists with different files; "
                    "export a new version instead of changing one"
                )
            return row.id
        session.add(
            DatasetVersion(
                id=meta.dataset_version,
                scope=meta.scope,
                directory=directory,
                holdout_cutoff=_parse(meta.holdout_cutoff),
                range_from=_parse(meta.range_from) if meta.range_from else None,
                range_to=_parse(meta.range_to) if meta.range_to else None,
                train_count=meta.counts.train,
                val_count=meta.counts.val,
                test_count=meta.counts.test,
                dpo_count=meta.counts.dpo,
                plan_ratio=meta.plan_ratio,
                persona_version=meta.persona_version,
                profile_version=meta.profile_version,
                template_version=meta.template_version,
                files=dict(meta.files),
                stats=dict(meta.stats),
            )
        )
    return meta.dataset_version


def _view(row: TrainingRun) -> RunView:
    return RunView(
        id=row.id,
        profile=row.profile,
        dataset_version=row.dataset_version,
        status=row.status,
        steps=dict(row.steps),
        started_at=row.started_at,
        finished_at=row.finished_at,
        gpu_model=row.gpu_model,
        peak_vram_mib=row.peak_vram_mib,
        best_val_loss=row.best_val_loss,
        best_checkpoint=row.best_checkpoint,
        bundle_sha256=row.bundle_sha256,
        artifacts=dict(row.artifacts),
        cleaned_at=row.cleaned_at,
        last_error=row.last_error,
        hyperparameters=dict(row.hyperparameters),
    )


class RunStore:
    """Reads and writes ``training_runs``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        run_id: str,
        profile: str,
        dataset_version: str,
        bundle_sha256: str,
        bundle_size: int,
        parameters: Mapping[str, Any],
    ) -> RunView:
        with self._db.transaction() as session:
            row = TrainingRun(
                id=run_id,
                profile=profile,
                dataset_version=dataset_version,
                status="created",
                bundle_sha256=bundle_sha256,
                bundle_size=bundle_size,
                hyperparameters=dict(parameters),
                steps={},
                artifacts={},
            )
            session.add(row)
            session.flush()
            return _view(row)

    def get(self, run_id: str) -> RunView:
        with self._db.session() as session:
            row = session.get(TrainingRun, run_id)
            if row is None:
                raise RunError(f"no training run {run_id!r}; see `twin train remote status`")
            return _view(row)

    def latest(self) -> RunView | None:
        with self._db.session() as session:
            row = session.scalars(
                select(TrainingRun).order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
            ).first()
            return _view(row) if row else None

    def all(self) -> list[RunView]:
        with self._db.session() as session:
            rows = session.scalars(
                select(TrainingRun).order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc())
            ).all()
            return [_view(row) for row in rows]

    def begin_step(
        self, run_id: str, step: str, now: datetime, *, keep_offset: bool = False
    ) -> None:
        """Open a step; ``keep_offset`` carries the log offset of an earlier attempt over."""
        with self._db.transaction() as session:
            row = self._row(session, run_id)
            steps = dict(row.steps)
            offset = int(steps.get(step, {}).get("log_offset", 0)) if keep_offset else 0
            steps[step] = {
                "ok": False,
                "started_at": _iso(now),
                "finished_at": None,
                "exit_code": None,
                "log_offset": offset,
            }
            row.steps = steps
            if row.started_at is None:
                row.started_at = now
            row.last_error = None

    def end_step(
        self,
        run_id: str,
        step: str,
        now: datetime,
        *,
        exit_code: int | None,
        error: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        """Close a step.  ``fields`` are columns learned on the way (GPU, loss, hashes, ...)."""
        ok = exit_code == 0 and error is None
        with self._db.transaction() as session:
            row = self._row(session, run_id)
            steps = dict(row.steps)
            record = dict(steps.get(step, {}))
            record.update({"ok": ok, "finished_at": _iso(now), "exit_code": exit_code})
            steps[step] = record
            row.steps = steps
            for name, value in (fields or {}).items():
                setattr(row, name, value)
            if ok:
                row.status = STEP_STATUS[step]
                row.finished_at = now
                if step == "cleanup":
                    row.cleaned_at = now
            else:
                row.status = "failed"
                row.last_error = error or f"{step} exited with code {exit_code}"

    def set_step_fields(self, run_id: str, step: str, **fields: Any) -> None:
        """Add values (such as the log offset) to the record of a step."""
        with self._db.transaction() as session:
            row = self._row(session, run_id)
            steps = dict(row.steps)
            steps[step] = {**steps.get(step, {}), **fields}
            row.steps = steps

    def update(self, run_id: str, **fields: Any) -> None:
        with self._db.transaction() as session:
            row = self._row(session, run_id)
            for name, value in fields.items():
                setattr(row, name, value)

    def uncleaned(self) -> list[RunView]:
        return [run for run in self.all() if run.needs_cleanup]

    @staticmethod
    def _row(session: Any, run_id: str) -> TrainingRun:
        row = session.get(TrainingRun, run_id)
        if row is None:
            raise RunError(f"no training run {run_id!r}")
        return row  # type: ignore[no-any-return]


def uncleaned_runs(db: Database) -> list[RunView]:
    """Runs that put her messages on a rented machine and have not been cleaned up (R-PRIV-003)."""
    return RunStore(db).uncleaned()
