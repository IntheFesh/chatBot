"""Exactly one target conversation in the database (R-IMP-003, R17-41, D-695).

Her data - the profile, the persona card, the example library, the sticker habits, the training
set - is read from the messages of *the* conversation, with no conversation id in the query:
the user name is an encrypted column, so "the target" cannot be picked out in SQL.  A second
conversation in the database would silently become part of "her" (a friend's, a colleague's
messages in her style statistics and in the training set).  So the import refuses a second
conversation (:data:`ONE_TARGET`), and ``twin doctor`` and ``twin import status`` say so when a
database that already has two (from before this rule) is found.  Nothing is deleted
automatically: which one is hers is for the user to say (``twin purge`` removes the imported
data so that the right export can be imported again).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import func, select

from twin.storage.chat_models import Conversation
from twin.storage.db import Database

ONE_TARGET = (
    "a different conversation was already imported: this database holds exactly one target "
    "conversation, so another one would be mixed into her data (R-IMP-003); to use another "
    "conversation, first remove the imported data (see `twin purge`)"
)

MANY_FOUND = (
    'the database holds {count} conversations, not one: everything computed from "her" '
    "messages (profile, persona card, examples, training set) mixes them"
)
MANY_HINT = (
    "decide which export is hers, remove the imported data (`twin purge`) and import that "
    "one again; nothing is deleted automatically"
)


def count_conversations(db: Database) -> int:
    """How many conversations the database holds."""
    with db.session() as session:
        return int(session.scalar(select(func.count()).select_from(Conversation)) or 0)


def count_conversations_in_file(db_path: Path) -> int | None:
    """The same, read straight from the database file (``None`` if there is nothing to read).

    ``twin doctor`` has no keys and no services: it counts rows, which needs no decryption.
    """
    if not db_path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute("SELECT COUNT(*) FROM conversations").fetchone()
    except sqlite3.Error:  # not migrated yet, or not a database
        return None
    finally:
        connection.close()
    return int(row[0]) if row else None
