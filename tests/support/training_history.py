"""A training run and the dataset it used, as the records of round 13 hold them (tests only)."""

from __future__ import annotations

from datetime import UTC, datetime

from twin.services import Services
from twin.storage.training_models import DatasetVersion
from twin.training.runs import RunStore

MOMENT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def record_dataset(services: Services, dataset: str = "ds-1", *, covered: int | None = 100) -> None:
    """A row of ``dataset_versions``; ``covered`` is the number of her messages it covered."""
    stats = {} if covered is None else {"her_messages_covered": covered}
    with services.db.transaction() as session:
        session.add(
            DatasetVersion(
                id=dataset,
                scope="pre_holdout",
                directory=f"training/datasets/{dataset}",
                holdout_cutoff=MOMENT,
                train_count=10,
                val_count=1,
                test_count=1,
                dpo_count=0,
                plan_ratio=0.3,
                persona_version="v1",
                profile_version="p1",
                template_version="qwen3_nothink@llamafactory-0.9.5",
                files={"sft_train.jsonl": "0" * 64},
                stats=stats,
            )
        )


def record_run(
    services: Services,
    run_id: str = "r1",
    dataset: str = "ds-1",
    *,
    trained: bool = True,
) -> None:
    """A training run; ``trained`` says whether its ``train`` step finished."""
    store = RunStore(services.db)
    store.create(
        run_id=run_id,
        profile="5090-8b",
        dataset_version=dataset,
        bundle_sha256="0" * 64,
        bundle_size=1,
        parameters={},
    )
    store.begin_step(run_id, "train", MOMENT)
    store.end_step(run_id, "train", MOMENT, exit_code=0 if trained else 1)


def record_training(
    services: Services,
    run_id: str = "r1",
    dataset: str = "ds-1",
    *,
    covered: int | None = 100,
    trained: bool = True,
) -> None:
    record_dataset(services, dataset, covered=covered)
    record_run(services, run_id, dataset, trained=trained)
