"""Low-level typed key/value access to the ``settings`` table (R-CFG-003).

Values are JSON, sealed per row.  Every change can append a history record
``{"at", "by", "old", "new"}`` (the history column is sealed as well, so old
and new values are never in plaintext on disk).  Typed, validated access to the
user-facing settings is in :mod:`twin.config.runtime`; this module is shared
plumbing, also used for internal keys such as ``state_version``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.storage.models import Setting

MAX_HISTORY = 100
_MISSING = object()


def get_setting(session: Session, key: str, default: Any = None) -> Any:
    """Return the stored value for ``key`` or ``default``."""
    row = session.get(Setting, key)
    if row is None:
        return default
    return row.value


def has_setting(session: Session, key: str) -> bool:
    return session.get(Setting, key) is not None


def put_setting(
    session: Session,
    key: str,
    value: Any,
    *,
    clock: Clock,
    by: str = "system",
    record_history: bool = True,
) -> bool:
    """Insert or update ``key``.  Returns ``True`` if the stored value changed."""
    row = session.get(Setting, key)
    if row is None:
        row = Setting(key=key, value=value)
        if record_history:
            row.history = [_record(clock, by, _MISSING, value)]
        session.add(row)
        session.flush()
        return True
    old = row.value
    if old == value:
        return False
    row.value = value
    if record_history:
        history = list(row.history or [])
        history.append(_record(clock, by, old, value))
        row.history = history[-MAX_HISTORY:]
    session.flush()
    return True


def delete_setting(session: Session, key: str) -> bool:
    row = session.get(Setting, key)
    if row is None:
        return False
    session.delete(row)
    session.flush()
    return True


def get_history(session: Session, key: str) -> list[dict[str, Any]]:
    """Change history for ``key`` (oldest first)."""
    row = session.get(Setting, key)
    if row is None:
        return []
    history = row.history
    return [dict(item) for item in history] if history else []


def _record(clock: Clock, by: str, old: Any, new: Any) -> dict[str, Any]:
    return {
        "at": clock.now_utc().isoformat(),
        "by": by,
        "old": None if old is _MISSING else old,
        "new": new,
        "created": old is _MISSING,
    }
