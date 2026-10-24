"""When her new messages make a new training worthwhile (R-TRN-012, R-IMP-011).

Counts of messages only; the conversation is synthetic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.synth_chat import MessageWriter
from tests.support.training_history import MOMENT, record_training
from twin.ingest.hooks import HookContext, load_hooks
from twin.ops.process_model import iter_commands
from twin.services import Services
from twin.storage.models import Alert
from twin.training import hook as retrain_hook
from twin.training.retrain import (
    BELOW_THRESHOLD,
    NO_BASELINE,
    NO_TRAINING,
    RETRAIN_ALERT,
    SUGGESTED,
    check_retrain,
    last_trained,
    retrain_status,
    retrain_status_text,
    status_text,
)

START = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def add_her(services: Services, count: int, *, system: int = 0, first: int = 0) -> None:
    """``count`` messages of hers (and ``system`` notices, and one of the user's per ten)."""
    writer = MessageWriter(services)
    for number in range(count):
        writer.add(START + timedelta(minutes=first + number), True, "text", f"她的第{number}句")
        if number % 10 == 0:
            writer.add(START + timedelta(minutes=first + number, seconds=30), False, "text", "你好")
    for number in range(system):
        writer.add(START + timedelta(days=1, minutes=number), True, "system", "拍了拍")
    writer.store(append=first > 0)


def alerts(services: Services) -> list[Alert]:
    with services.db.session() as session:
        found = list(session.scalars(select(Alert).where(Alert.category == RETRAIN_ALERT)))
        for alert in found:
            session.expunge(alert)
        return found


# ---------------------------------------------------------------------------- the status


def test_before_the_first_training_there_is_nothing_to_compare_with(services: Services) -> None:
    add_her(services, 50)
    status = retrain_status(services.db, 0.10)
    assert status.reason == NO_TRAINING and not status.suggested and status.ratio is None
    assert status.current == 50  # the user's messages and system notices are not hers
    assert status_text(status) is None and "no style model" in status.describe()


def test_the_new_share_is_the_messages_not_covered_over_the_ones_covered(
    services: Services,
) -> None:
    add_her(services, 109, system=5)
    record_training(services, covered=100)
    status = retrain_status(services.db, 0.10)
    assert status.reason == BELOW_THRESHOLD and status.new_messages == 9 and status.current == 109
    assert status.ratio == pytest.approx(0.09) and not status.suggested
    assert "below the 10% threshold" in status.describe()
    add_her(services, 1, first=500)
    again = retrain_status(services.db, 0.10)
    assert again.reason == SUGGESTED and again.suggested and again.ratio == pytest.approx(0.10)
    assert "retraining is suggested" in again.describe()


def test_the_threshold_is_the_setting(services: Services) -> None:
    add_her(services, 103)
    record_training(services, covered=100)
    assert not retrain_status(services.db, 0.10).suggested
    assert retrain_status(services.db, 0.03).suggested
    assert services.settings.training.retrain_new_ratio == 0.10


async def test_only_a_training_that_finished_counts_and_the_newest_one_is_the_baseline(
    services: Services, clock: ManualClock
) -> None:
    add_her(services, 130)
    record_training(services, "r-old", "ds-old", covered=50)
    await clock.advance(86400)
    record_training(services, "r-good", "ds-good", covered=120)
    await clock.advance(86400)
    record_training(services, "r-broken", "ds-broken", covered=10, trained=False)
    run = last_trained(services.db)
    assert run is not None and run.id == "r-good"
    status = retrain_status(services.db, 0.10)
    assert status.run_id == "r-good" and status.covered == 120 and not status.suggested


def test_a_dataset_that_did_not_record_its_message_count_gives_no_verdict(
    services: Services,
) -> None:
    add_her(services, 500)
    record_training(services, covered=None)
    status = retrain_status(services.db, 0.10)
    assert status.reason == NO_BASELINE and not status.suggested
    assert "did not record" in status.describe()
    check_retrain(services)
    assert alerts(services) == []


# ------------------------------------------------------------------------------ the alert


def test_the_alert_is_raised_once_per_training_with_counts_only(services: Services) -> None:
    add_her(services, 140)
    record_training(services, "r1", covered=100)
    first = check_retrain(services)
    assert first.suggested
    (alert,) = alerts(services)
    assert alert.severity == "info" and alert.dedup_key == "retrain:r1"
    assert "40%" in alert.title
    assert alert.detail == {
        "new_messages": 40,
        "covered": 100,
        "ratio": 0.4,
        "run_id": "r1",
        "dataset_version": "ds-1",
    }
    check_retrain(services)
    assert len(alerts(services)) == 1  # not again for the same training
    with services.db.transaction() as session:
        row = session.get(Alert, alert.id)
        assert row is not None
        row.resolved_at = MOMENT
    check_retrain(services)
    assert len(alerts(services)) == 2  # a resolved reminder may come back
    record_training(services, "r2", "ds-2", covered=135)
    check_retrain(services)  # the new training covers 135: 5 new messages, below the threshold
    assert len(alerts(services)) == 2
    add_her(services, 20, first=900)
    check_retrain(services)
    assert {a.dedup_key for a in alerts(services)} == {"retrain:r1", "retrain:r2"}


def test_nothing_is_raised_below_the_threshold_or_before_a_training(services: Services) -> None:
    add_her(services, 105)
    check_retrain(services)
    record_training(services, covered=100)
    status = check_retrain(services)
    assert not status.suggested and alerts(services) == []


# ---------------------------------------------------------------------------- /状态 text


def test_the_reminder_for_the_status_command_is_text_only_when_it_is_due(
    services: Services,
) -> None:
    add_her(services, 108)
    record_training(services, covered=100)
    assert retrain_status_text(services.db, 0.10) is None
    add_her(services, 12, first=700)
    text = retrain_status_text(services.db, 0.10)
    assert text is not None
    assert "20%" in text and "≥ 10%" in text and "twin train export" in text


# ------------------------------------------------------------------------------- the hook


def context(services: Services) -> HookContext:
    return HookContext(services, "run-1", "conv", "exp", inserted=3, changed=0, first_import=False)


def test_the_hook_is_registered_last_with_a_backfill_command_that_exists() -> None:
    from twin.cli import app

    registry = load_hooks()
    hook = next(h for h in registry.hooks() if h.name == "retrain")
    assert hook.backfill_command == "train retrain-check"
    assert hook.backfill_command in {name for name, _ in iter_commands(app)}
    assert registry.names().index("retrain") > registry.names().index("memory_replay")
    assert hook.run is retrain_hook.check_retraining


def test_the_hook_skips_before_a_training_and_reports_the_share_after_one(
    services: Services,
) -> None:
    add_her(services, 40)
    skipped = retrain_hook.check_retraining(context(services))
    assert skipped.status == "skipped" and "no style model" in skipped.detail
    record_training(services, covered=30)
    done = retrain_hook.check_retraining(context(services))
    assert done.status == "done" and "retraining is suggested" in done.detail
    assert len(alerts(services)) == 1
    assert "她的第" not in done.detail  # counts, never text
