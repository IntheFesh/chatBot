"""Key/value access to the encrypted ``channel_state`` table (R-CH-003, R-CH-004, R-CH-007).

Every value is sealed JSON (AES-256-GCM, bound to the table, key and column), so login
credentials, the poll cursor, the context token and the unread-message inbox never reach the
disk in plaintext.  Reads use a read session; writes go through
:meth:`ChannelStateStore.transaction`, which gives the caller several keys in one atomic write
(the poller advances the cursor, the seen-id list and the inbox together, which is what makes
"restart loses nothing and repeats nothing" true).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from twin.storage.db import Database
from twin.storage.models import ChannelState


class StateTx:
    """Typed key/value operations inside one write transaction."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, key: str, default: Any = None) -> Any:
        row = self._session.get(ChannelState, key)
        return default if row is None else row.value

    def put(self, key: str, value: Any) -> bool:
        """Store ``value``; returns ``False`` (and writes nothing) if it is unchanged."""
        row = self._session.get(ChannelState, key)
        if row is None:
            self._session.add(ChannelState(key=key, value=value))
            self._session.flush()
            return True
        if row.value == value:
            return False
        row.value = value
        self._session.flush()
        return True

    def delete(self, key: str) -> bool:
        row = self._session.get(ChannelState, key)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


class ChannelStateStore:
    """The ``channel_state`` table as a small key/value store."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str, default: Any = None) -> Any:
        with self._db.session() as session:
            row = session.get(ChannelState, key)
            return default if row is None else row.value

    def put(self, key: str, value: Any) -> bool:
        with self.transaction() as tx:
            return tx.put(key, value)

    def delete(self, key: str) -> bool:
        with self.transaction() as tx:
            return tx.delete(key)

    @contextmanager
    def transaction(self) -> Iterator[StateTx]:
        """One atomic write; the process model decides whether it bumps the state version."""
        with self._db.transaction() as session:
            yield StateTx(session)
