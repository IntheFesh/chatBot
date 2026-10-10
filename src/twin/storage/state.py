"""``settings.state_version``: the cross-process change counter (R-ARCH-006.2).

LIGHT CLI commands increment it in the same transaction as their writes; the
running application polls it every two seconds
(:class:`twin.ops.state_watch.StateWatcher`) and, when it changes, tells its
components to drop cached state.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.storage.settings_store import get_setting, put_setting

STATE_VERSION_KEY = "state_version"


def read_state_version(session: Session) -> int:
    """Current state version (0 if never bumped)."""
    value = get_setting(session, STATE_VERSION_KEY, 0)
    return int(value)


def bump_state_version(session: Session, clock: Clock) -> int:
    """Increment the state version inside the caller's write transaction."""
    new = read_state_version(session) + 1
    put_setting(
        session, STATE_VERSION_KEY, new, clock=clock, by="state_version", record_history=False
    )
    return new
