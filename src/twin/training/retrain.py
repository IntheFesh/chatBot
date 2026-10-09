"""When to train again (R-TRN-012, R-IMP-011).

A style model is trained on the messages that existed when its training set was exported.  The
more of her messages arrive afterwards, the more of how she writes now it has never seen; at
``training.retrain_new_ratio`` (10 %) new messages the program says so: an alert
``retrain_suggested`` and a line in ``/状态``.  Nothing is trained by itself (a run costs money and
a rented machine), and a new model is trained from the base model on all the data, never on top of
the old adapter.

*The baseline* is the last training run whose ``train`` step finished: the dataset it was
trained on recorded how many of her messages its time range covered
(``dataset_versions.stats["her_messages_covered"]``).  *The ratio* is the messages that are in
the database now and were not covered, divided by the covered ones.  Before the first training
there is nothing to compare with and nothing is suggested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from sqlalchemy import select

from twin.ingest.corpus import her_message_count
from twin.storage.db import Database
from twin.storage.models import Alert
from twin.storage.training_models import DatasetVersion
from twin.training.runs import RunStore, RunView

if TYPE_CHECKING:
    from twin.services import Services

RETRAIN_ALERT: Final = "retrain_suggested"
COVERED_KEY: Final = "her_messages_covered"

NO_TRAINING: Final = "no_training"
NO_BASELINE: Final = "no_baseline"
BELOW_THRESHOLD: Final = "below_threshold"
SUGGESTED: Final = "suggested"


@dataclass(frozen=True)
class RetrainStatus:
    """Where the program stands against its last training (see the module description)."""

    reason: str  # no_training | no_baseline | below_threshold | suggested
    threshold: float
    current: int
    covered: int | None = None
    run_id: str | None = None
    dataset_version: str | None = None

    @property
    def suggested(self) -> bool:
        return self.reason == SUGGESTED

    @property
    def new_messages(self) -> int | None:
        return None if self.covered is None else max(0, self.current - self.covered)

    @property
    def ratio(self) -> float | None:
        if self.covered is None or self.covered <= 0:
            return None
        return (self.new_messages or 0) / self.covered

    def describe(self) -> str:
        """One line for the terminal and the import report (counts only)."""
        if self.reason == NO_TRAINING:
            return "no style model has been trained yet"
        if self.reason == NO_BASELINE or self.ratio is None:
            return f"the last training ({self.run_id}) did not record how many messages it covered"
        text = (
            f"{self.new_messages:,} of her messages are new since the last training "
            f"({self.ratio:.1%} of the {self.covered:,} it covered)"
        )
        if self.suggested:
            return f"{text}: retraining is suggested (threshold {self.threshold:.0%})"
        return f"{text}: below the {self.threshold:.0%} threshold"


def last_trained(db: Database) -> RunView | None:
    """The newest run whose training step finished."""
    for run in RunStore(db).all():
        if bool(run.steps.get("train", {}).get("ok")):
            return run
    return None


def covered_messages(db: Database, dataset_version: str) -> int | None:
    """How many of her messages the training set of ``dataset_version`` covered, if recorded."""
    with db.session() as session:
        row = session.get(DatasetVersion, dataset_version)
        if row is None:
            return None
        value = row.stats.get(COVERED_KEY)
        return int(value) if isinstance(value, int | float) and value > 0 else None


def her_message_total(db: Database) -> int:
    """Her messages in the database now (system notices are not messages)."""
    with db.session() as session:
        return int(session.scalar(her_message_count()) or 0)


def retrain_status(db: Database, threshold: float) -> RetrainStatus:
    """Compare the messages now with the ones the last training covered."""
    current = her_message_total(db)
    run = last_trained(db)
    if run is None:
        return RetrainStatus(NO_TRAINING, threshold, current)
    covered = covered_messages(db, run.dataset_version)
    if covered is None:
        return RetrainStatus(
            NO_BASELINE, threshold, current, run_id=run.id, dataset_version=run.dataset_version
        )
    status = RetrainStatus(
        BELOW_THRESHOLD,
        threshold,
        current,
        covered=covered,
        run_id=run.id,
        dataset_version=run.dataset_version,
    )
    ratio = status.ratio or 0.0
    if ratio >= threshold:
        return RetrainStatus(
            SUGGESTED,
            threshold,
            current,
            covered=covered,
            run_id=run.id,
            dataset_version=run.dataset_version,
        )
    return status


def status_text(status: RetrainStatus) -> str | None:
    """The line for ``/状态``; ``None`` (shown as "暂无") unless retraining is suggested."""
    if not status.suggested or status.ratio is None:
        return None
    return (
        f"她的新消息已占上次训练数据的 {status.ratio:.0%}（≥ {status.threshold:.0%}），"
        "建议重新训练风格模型（twin train export）"
    )


def retrain_status_text(db: Database, threshold: float) -> str | None:
    """The reminder of ``/状态`` (R-TRN-012): see :func:`status_text`."""
    return status_text(retrain_status(db, threshold))


def alert_open(db: Database, key: str) -> bool:
    with db.session() as session:
        return (
            session.scalar(
                select(Alert.id).where(
                    Alert.category == RETRAIN_ALERT,
                    Alert.dedup_key == key,
                    Alert.resolved_at.is_(None),
                )
            )
            is not None
        )


def check_retrain(services: Services) -> RetrainStatus:
    """Compute the status and raise ``retrain_suggested`` once per training run when it is due."""
    status = retrain_status(services.db, services.settings.training.retrain_new_ratio)
    if status.suggested and status.run_id is not None:
        key = f"retrain:{status.run_id}"
        if not alert_open(services.db, key):
            services.alerts.raise_alert(
                RETRAIN_ALERT,
                f"她的新消息已达到上次训练数据的 {status.ratio or 0:.0%}，建议重新训练风格模型",
                severity="info",
                detail={
                    "new_messages": status.new_messages,
                    "covered": status.covered,
                    "ratio": round(status.ratio or 0.0, 4),
                    "run_id": status.run_id,
                    "dataset_version": status.dataset_version,
                },
                dedup_key=key,
            )
    return status
