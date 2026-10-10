"""The memory in memory: decrypted records and the keyword index (R-MEM-008).

Facts and summaries are few, so the retrieval works on a snapshot held in the process: the
records of :mod:`twin.memory.records`, the inverted indexes of :mod:`twin.memory.keywords`, and
the small lists the date logic needs.  It is built from the decrypted rows when the process first
asks and then kept up to date by :meth:`MemoryCorpus.refresh`, which looks at a cheap fingerprint
of the tables (row counts, revisions and newest id), and only when that changed compares the
revision of every row and reloads the rows that are new or changed.  A change made by
another process (``twin memory forget`` while the application runs) is therefore picked up at the
next read, and nothing is read back from the database when nothing changed.

The corpus holds everything, also what is in the future of some moment; it never decides what
may be seen.  That is :mod:`twin.memory.visible`, applied by whoever reads the corpus.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import date

from twin.memory.keywords import KeywordIndex
from twin.memory.records import FactRecord, FollowupRecord, LifelineRecord, SummaryRecord
from twin.memory.store import MemoryStore, StoreSignature
from twin.memory.visible import BotEra

CORE_IMPORTANCE = 5


class MemoryCorpus:
    """A refreshable snapshot of the memory tables (see the module description)."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._signature: StoreSignature | None = None
        self.facts: dict[str, FactRecord] = {}
        self.summaries: dict[str, SummaryRecord] = {}
        self.followups: dict[str, FollowupRecord] = {}
        self.events: dict[str, LifelineRecord] = {}
        self.fact_index = KeywordIndex()
        self.summary_index = KeywordIndex()
        self._summary_by_day: dict[tuple[str, date], SummaryRecord] = {}
        self._dated: dict[str, FactRecord] = {}
        self._core: dict[str, FactRecord] = {}

    @property
    def store(self) -> MemoryStore:
        return self._store

    # ---------------------------------------------------------------- refreshing

    def refresh(self) -> bool:
        """Bring the snapshot up to date; returns ``True`` if anything was reloaded."""
        with self._lock:
            signature = self._store.signature()
            if signature == self._signature:
                return False
            self._sync_facts()
            self._sync_summaries()
            self._sync_followups()
            self._sync_events()
            self._signature = signature
            return True

    @staticmethod
    def _changed(stamps: dict[str, int], known: dict[str, int]) -> list[str]:
        """Ids that are new or whose revision moved."""
        return [key for key, stamp in stamps.items() if known.get(key) != stamp]

    def _sync_facts(self) -> None:
        stamps = self._store.fact_stamps()
        for gone in [key for key in self.facts if key not in stamps]:
            del self.facts[gone]
            self._dated.pop(gone, None)
            self._core.pop(gone, None)
            self.fact_index.remove(gone)
        known = {key: record.rev for key, record in self.facts.items()}
        for record in self._store.facts(self._changed(stamps, known)):
            self.facts[record.id] = record
            self.fact_index.add(record.id, record.text)
            if record.event_date is not None:
                self._dated[record.id] = record
            else:
                self._dated.pop(record.id, None)
            if record.importance >= CORE_IMPORTANCE:
                self._core[record.id] = record
            else:
                self._core.pop(record.id, None)

    def _sync_summaries(self) -> None:
        stamps = self._store.summary_stamps()
        for gone in [key for key in self.summaries if key not in stamps]:
            del self.summaries[gone]
            self.summary_index.remove(gone)
        known = {key: record.rev for key, record in self.summaries.items()}
        for record in self._store.summaries(self._changed(stamps, known)):
            self.summaries[record.id] = record
            self.summary_index.add(record.id, record.text)
        self._summary_by_day = {(s.scope, s.local_date): s for s in self.summaries.values()}

    def _sync_followups(self) -> None:
        stamps = self._store.followup_stamps()
        for gone in [key for key in self.followups if key not in stamps]:
            del self.followups[gone]
        known = {key: record.rev for key, record in self.followups.items()}
        for record in self._store.followups_by_ids(self._changed(stamps, known)):
            self.followups[record.id] = record

    def _sync_events(self) -> None:
        stamps = self._store.event_stamps()
        for gone in [key for key in self.events if key not in stamps]:
            del self.events[gone]
        known = {key: record.rev for key, record in self.events.items()}
        for record in self._store.events_by_ids(self._changed(stamps, known)):
            self.events[record.id] = record

    # ------------------------------------------------------------------ reading

    def era(self) -> BotEra:
        return BotEra(self._store.bot_online_at())

    def summary_of(self, scope: str, day: date) -> SummaryRecord | None:
        return self._summary_by_day.get((scope, day))

    def core_facts(self) -> Iterable[FactRecord]:
        """Facts of the highest importance (names people call each other, big events)."""
        return list(self._core.values())

    def dated_facts(self) -> Iterable[FactRecord]:
        """Facts that name a calendar day (birthdays, anniversaries, exam days)."""
        return list(self._dated.values())
