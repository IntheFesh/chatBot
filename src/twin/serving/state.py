"""What the application knows about the style-model server, kept where other processes can read it.

The running application owns the ``llama-server`` process or the SSH tunnel; ``/状态``, ``twin
model tunnel status`` and ``twin model serve`` run in other places and must not guess.  The
component writes a small record into the ``settings`` table (key ``serving.state``) whenever
something changes, and everybody reads it from there:

``server``   the local server: ``state``, ``model``, ``pid``, ``restarts``, ``since``, ``detail``
``warmup``   the last warm-up: first-token latency, tokens per second, when, who measured
``tunnel``   the tunnel: ``state``, ``up_since``, ``reconnects``, ``detail``, ``instance_uptime_s``
``process``  the process that wrote the record (a record of a process that is gone is stale)
``updated``  when (UTC)

The record holds states, numbers and ids - never a message - and it is written without waking the
application (``bump_state=False``): it is output, not a request.
"""

from __future__ import annotations

from typing import Any, Final

from twin.clock import Clock
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

STATE_KEY: Final = "serving.state"
REMINDER_KEY: Final = "serving.remote_reminder"


class ServingStateStore:
    """Reads and writes the record (a few short transactions; call from a thread)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    def read(self) -> dict[str, Any]:
        with self._db.session() as session:
            value = get_setting(session, STATE_KEY, None)
        return dict(value) if isinstance(value, dict) else {}

    def update(self, sections: dict[str, Any]) -> dict[str, Any]:
        """Replace the given sections (``None`` deletes one) and stamp the record."""
        with self._db.transaction(bump_state=False) as session:
            current = get_setting(session, STATE_KEY, None)
            record = dict(current) if isinstance(current, dict) else {}
            for name, value in sections.items():
                if value is None:
                    record.pop(name, None)
                else:
                    record[name] = value
            record["updated"] = self._clock.now_utc().isoformat()
            put_setting(
                session,
                STATE_KEY,
                record,
                clock=self._clock,
                by="serving",
                record_history=False,
            )
        return record

    def reminded_on(self) -> str | None:
        """The local date (``YYYY-MM-DD``) of the last reminder about the rented instance."""
        with self._db.session() as session:
            value = get_setting(session, REMINDER_KEY, None)
        return str(value) if isinstance(value, str) else None

    def mark_reminded(self, day: str) -> None:
        with self._db.transaction(bump_state=False) as session:
            put_setting(
                session, REMINDER_KEY, day, clock=self._clock, by="serving", record_history=False
            )
