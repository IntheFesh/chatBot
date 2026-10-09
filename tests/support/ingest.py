"""Helpers for the import tests: build an export, prepare and run an import."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select

from tests.fixtures.synth_export import SynthExport, SynthOptions, build_export
from tests.support.clock import ManualClock
from twin.config.loader import load_settings
from twin.config.secrets import SecretStore
from twin.ingest.importer import BatchEvent, ImportRunner, RunOutcome, prepare_import
from twin.services import Services, build_services
from twin.storage import migrate
from twin.storage.chat_models import Message


def make_export(root: Path, **options: Any) -> SynthExport:
    """A synthetic export below ``root / 'export'``."""
    return build_export(root / "export", SynthOptions(**options))


def start_import(
    services: Services,
    export: SynthExport | Path,
    *,
    target: str | None = None,
) -> str:
    """Prepare the run for ``export`` and return its id."""
    root = export if isinstance(export, Path) else export.root
    username = target or (export.target_username if isinstance(export, SynthExport) else "")
    return prepare_import(services, root, target_username=username).run_id


def run_import(
    services: Services,
    export: SynthExport | Path,
    *,
    batch_size: int | None = None,
    stop: threading.Event | None = None,
    on_batch: Callable[[BatchEvent], None] | None = None,
    target: str | None = None,
    run_id: str | None = None,
) -> RunOutcome:
    """Prepare (unless ``run_id`` is given) and run an import in this thread."""
    identifier = run_id or start_import(services, export, target=target)
    runner = ImportRunner(services, batch_size=batch_size, on_batch=on_batch)
    return runner.run(identifier, stop)


def message_snapshot(services: Services) -> dict[str, tuple[object, ...]]:
    """Every stored message as plain values (decrypted), keyed by id, for comparisons."""
    with services.db.session() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
        return {
            row.id: (
                row.create_time_utc,
                row.sort_seq,
                row.local_id,
                row.server_id,
                row.is_sent,
                row.kind,
                row.render_type,
                row.text,
                row.raw,
                row.sticker_md5,
                row.media_sha256,
                row.quote,
                row.call_status,
                row.call_duration_s,
                row.voice_seconds,
                row.has_transcript,
                row.source_export_id,
                row.source_exported_at,
            )
            for row in rows
        }


def second_services(tmp_path: Path, clock: ManualClock, secret_store: SecretStore) -> Services:
    """A second, independent database (same key) for comparing two ways of importing."""
    home = Path(os.environ["TWIN_HOME"])
    settings = load_settings(None, {"paths": {"data_dir": str(tmp_path / "data_b")}})
    migrate.upgrade(Path(settings.paths.data_dir) / "twin.db")
    return build_services(settings, root=home, secrets=secret_store, clock=clock)
