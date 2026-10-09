"""An import killed in the middle continues to the very same result (R-IMP-006).

The importing process is really killed (``os._exit``: no clean-up, no ``finally``, like a power
cut) after a number of committed batches; a second process then resumes the run.  The outcome
must equal that of one uninterrupted import of the same export.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from sqlalchemy import func, select

from tests.support.clock import ManualClock
from tests.support.ingest import (
    make_export,
    message_snapshot,
    run_import,
    second_services,
    start_import,
)
from twin.config.secrets import SecretStore
from twin.ingest.runs import get_run
from twin.services import Services
from twin.storage.chat_models import MediaAsset, Message, Sticker, StickerUse

pytestmark = pytest.mark.integration

VICTIM = textwrap.dedent(
    """
    import os
    import sys

    from twin.config.loader import load_settings, resolve_paths
    from twin.ingest.importer import ImportRunner
    from twin.services import build_services

    run_id, die_after, batch_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root)

    def die(event):
        if event.batch_number == die_after:
            os._exit(9)  # the process disappears here: nothing is flushed or rolled back

    ImportRunner(services, batch_size=batch_size, on_batch=die).run(run_id)
    print("finished without dying")
    """
)


def count(services: Services, model: type) -> int:
    with services.db.session() as session:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)


def test_a_killed_import_resumes_to_exactly_the_uninterrupted_result(
    services: Services, tmp_path: Path, clock: ManualClock, secret_store: SecretStore
) -> None:
    export = make_export(tmp_path, target_messages=500, other_messages=20)
    run_id = start_import(services, export)
    env = {**os.environ, "PYTHONUTF8": "1", "TWIN_PATHS__DATA_DIR": str(services.paths.data_dir)}
    victim = subprocess.run(
        [sys.executable, "-c", VICTIM, run_id, "3", "60"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert victim.returncode == 9, victim.stderr
    assert "finished" not in victim.stdout

    interrupted = get_run(services.db, run_id)
    assert interrupted is not None
    assert (interrupted.status, interrupted.processed) == ("running", 180)
    assert count(services, Message) == 180  # the committed batches, nothing more, nothing less

    outcome = run_import(services, export, run_id=run_id, batch_size=60)
    assert outcome.status == "done" and outcome.run.processed == 500
    assert (outcome.run.inserted, outcome.run.duplicates) == (500, 0)

    reference = second_services(tmp_path, clock, secret_store)
    try:
        run_import(reference, export, batch_size=60)
        assert message_snapshot(reference) == message_snapshot(services)
        for model in (StickerUse, Sticker, MediaAsset):
            assert count(reference, model) == count(services, model)
    finally:
        reference.close()


def test_two_kills_in_a_row_still_end_in_the_same_place(services: Services, tmp_path: Path) -> None:
    export = make_export(tmp_path, target_messages=300)
    run_id = start_import(services, export)
    env = {**os.environ, "PYTHONUTF8": "1", "TWIN_PATHS__DATA_DIR": str(services.paths.data_dir)}
    for die_after in ("1", "2"):
        victim = subprocess.run(
            [sys.executable, "-c", VICTIM, run_id, die_after, "50"],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
            check=False,
        )
        assert victim.returncode == 9, victim.stderr
    # the first process committed 50 messages; the second continued after them and committed
    # two batches (100 more) before it was killed in turn
    assert count(services, Message) == 150
    outcome = run_import(services, export, run_id=run_id, batch_size=50)
    assert outcome.status == "done" and count(services, Message) == 300
