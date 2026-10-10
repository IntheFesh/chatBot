"""Reading and writing ``consistency_findings`` and ``consistency_fixes`` (R-EVAL-004).

:class:`ConsistencyStore` is the only code that touches the two tables.  A finding is stored when
the audit has checked the model's answer (state ``proposed``); the user's decision moves it to
``confirmed`` - as an ``obvious`` or a ``minor`` contradiction - or ``rejected`` and is written the
moment it is made, so a review that stops anywhere goes on from the finding it had reached.  The
fixes of a confirmed contradiction are stored as proposals; only ``applied`` means that the live
memory was changed.  Nothing is ever deleted (the rows go with their run).

The statements and the wording of a fix are text and live in the sealed ``payload``; the plain
columns carry the fingerprint of the two records, the time and the states.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from twin.clock import Clock, ensure_aware
from twin.eval.consistency_model import Finding, Statement
from twin.storage.db import Database
from twin.storage.eval_models import ConsistencyFinding, ConsistencyFix

DECIDED = ("confirmed", "rejected")


class ConsistencyStoreError(LookupError):
    """A finding or a fix that was asked for does not exist."""


@dataclass(frozen=True)
class NewFinding:
    """A finding to store; ``status`` is not ``proposed`` for a decision carried over."""

    finding: Finding
    status: str = "proposed"
    severity: str | None = None
    inherited_from: str | None = None


@dataclass(frozen=True)
class FindingView:
    """A row of ``consistency_findings`` with its payload decrypted."""

    id: str
    run_id: str
    seq: int
    fingerprint: str
    model_severity: str
    status: str
    severity: str | None
    at: datetime
    inherited_from: str | None
    decided_at: datetime | None
    payload: dict[str, Any]

    @property
    def decided(self) -> bool:
        return self.status in DECIDED

    @property
    def first(self) -> Statement:
        return Statement.from_json(self.payload["first"])

    @property
    def second(self) -> Statement:
        return Statement.from_json(self.payload["second"])

    @property
    def related(self) -> tuple[Statement, ...]:
        return tuple(Statement.from_json(item) for item in self.payload.get("related", []))

    @property
    def time_text(self) -> str:
        return str(self.payload.get("time", ""))

    @property
    def reason(self) -> str:
        return str(self.payload.get("reason", ""))

    @property
    def keep(self) -> str | None:
        keep = self.payload.get("keep")
        return str(keep) if keep else None

    @property
    def rewrite(self) -> str | None:
        rewrite = self.payload.get("rewrite")
        return str(rewrite) if rewrite else None


@dataclass(frozen=True)
class NewFix:
    """A correction to propose: what to do to which record, and the old and new wording."""

    action: str
    target_id: str
    old_text: str
    new_text: str | None
    reason: str


@dataclass(frozen=True)
class FixView:
    """A row of ``consistency_fixes`` with its payload decrypted."""

    id: str
    finding_id: str
    seq: int
    action: str
    target_id: str
    status: str
    new_id: str | None
    decided_at: datetime | None
    applied_at: datetime | None
    payload: dict[str, Any]

    @property
    def old_text(self) -> str:
        return str(self.payload.get("old_text", ""))

    @property
    def new_text(self) -> str | None:
        text = self.payload.get("new_text")
        return str(text) if text else None

    @property
    def reason(self) -> str:
        return str(self.payload.get("reason", ""))


def _finding_view(row: ConsistencyFinding) -> FindingView:
    return FindingView(
        id=row.id,
        run_id=row.run_id,
        seq=row.seq,
        fingerprint=row.fingerprint,
        model_severity=row.model_severity,
        status=row.status,
        severity=row.severity,
        at=row.at,
        inherited_from=row.inherited_from,
        decided_at=row.decided_at,
        payload=dict(row.payload),
    )


def _fix_view(row: ConsistencyFix) -> FixView:
    return FixView(
        id=row.id,
        finding_id=row.finding_id,
        seq=row.seq,
        action=row.action,
        target_id=row.target_id,
        status=row.status,
        new_id=row.new_id,
        decided_at=row.decided_at,
        applied_at=row.applied_at,
        payload=dict(row.payload),
    )


class ConsistencyStore:
    """The two tables of the consistency audit (see the module description)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # ---------------------------------------------------------------------- findings

    def add_findings(self, run_id: str, items: Sequence[NewFinding]) -> list[FindingView]:
        """Store findings numbered after the last one of the run (decisions carried over keep
        their state, severity and the finding they were taken from)."""
        added: list[FindingView] = []
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            last = session.scalar(
                select(func.max(ConsistencyFinding.seq)).where(ConsistencyFinding.run_id == run_id)
            )
            seq = 0 if last is None else int(last) + 1
            for item in items:
                found = item.finding
                row = ConsistencyFinding(
                    run_id=run_id,
                    seq=seq,
                    fingerprint=found.fingerprint,
                    model_severity=found.severity,
                    status=item.status,
                    severity=item.severity,
                    at=ensure_aware(found.at),
                    inherited_from=item.inherited_from,
                    decided_at=now if item.status in DECIDED else None,
                    payload=found.to_payload(),
                )
                session.add(row)
                session.flush()
                added.append(_finding_view(row))
                seq += 1
        return added

    def finding(self, finding_id: str) -> FindingView:
        with self._db.session() as session:
            row = session.get(ConsistencyFinding, finding_id)
            if row is None:
                raise ConsistencyStoreError(f"there is no finding {finding_id!r}")
            return _finding_view(row)

    def findings(self, run_id: str, *, status: str | None = None) -> list[FindingView]:
        """The findings of a run in order, optionally those in one state."""
        stmt = (
            select(ConsistencyFinding)
            .where(ConsistencyFinding.run_id == run_id)
            .order_by(ConsistencyFinding.seq)
        )
        if status is not None:
            stmt = stmt.where(ConsistencyFinding.status == status)
        with self._db.session() as session:
            return [_finding_view(row) for row in session.scalars(stmt)]

    def decide(self, finding_id: str, *, status: str, severity: str | None = None) -> FindingView:
        """Record a decision: ``confirmed`` as ``obvious`` or ``minor``, or ``rejected``."""
        if status not in DECIDED:
            raise ValueError(f"a decision is one of {DECIDED}, not {status!r}")
        if (status == "confirmed") != (severity is not None):
            raise ValueError("a confirmed finding has a severity and a rejected one has none")
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ConsistencyFinding, finding_id)
            if row is None:
                raise ConsistencyStoreError(f"there is no finding {finding_id!r}")
            row.status = status
            row.severity = severity
            row.decided_at = self._clock.now_utc()
            session.flush()
            return _finding_view(row)

    def decisions(self) -> dict[str, FindingView]:
        """The user's decision about each pair of records he was asked about, by fingerprint.

        If a pair was decided more than once the latest decision stands.
        """
        stmt = (
            select(ConsistencyFinding)
            .where(ConsistencyFinding.status.in_(DECIDED))
            .order_by(ConsistencyFinding.decided_at, ConsistencyFinding.id)
        )
        with self._db.session() as session:
            return {row.fingerprint: _finding_view(row) for row in session.scalars(stmt)}

    # -------------------------------------------------------------------------- fixes

    def add_fixes(self, finding_id: str, items: Sequence[NewFix]) -> list[FixView]:
        added: list[FixView] = []
        with self._db.transaction(bump_state=False) as session:
            if session.get(ConsistencyFinding, finding_id) is None:
                raise ConsistencyStoreError(f"there is no finding {finding_id!r}")
            last = session.scalar(
                select(func.max(ConsistencyFix.seq)).where(ConsistencyFix.finding_id == finding_id)
            )
            seq = 0 if last is None else int(last) + 1
            for item in items:
                row = ConsistencyFix(
                    finding_id=finding_id,
                    seq=seq,
                    action=item.action,
                    target_id=item.target_id,
                    status="proposed",
                    payload={
                        "old_text": item.old_text,
                        "new_text": item.new_text,
                        "reason": item.reason,
                    },
                )
                session.add(row)
                session.flush()
                added.append(_fix_view(row))
                seq += 1
        return added

    def fix(self, fix_id: str) -> FixView:
        with self._db.session() as session:
            row = session.get(ConsistencyFix, fix_id)
            if row is None:
                raise ConsistencyStoreError(f"there is no fix {fix_id!r}")
            return _fix_view(row)

    def fixes(self, finding_id: str, *, status: str | None = None) -> list[FixView]:
        stmt = (
            select(ConsistencyFix)
            .where(ConsistencyFix.finding_id == finding_id)
            .order_by(ConsistencyFix.seq)
        )
        if status is not None:
            stmt = stmt.where(ConsistencyFix.status == status)
        with self._db.session() as session:
            return [_fix_view(row) for row in session.scalars(stmt)]

    def run_fixes(self, run_id: str, *, status: str | None = None) -> list[FixView]:
        """The fixes of every finding of a run, in the order of the findings."""
        stmt = (
            select(ConsistencyFix)
            .join(ConsistencyFinding, ConsistencyFinding.id == ConsistencyFix.finding_id)
            .where(ConsistencyFinding.run_id == run_id)
            .order_by(ConsistencyFinding.seq, ConsistencyFix.seq)
        )
        if status is not None:
            stmt = stmt.where(ConsistencyFix.status == status)
        with self._db.session() as session:
            return [_fix_view(row) for row in session.scalars(stmt)]

    def set_fix(self, fix_id: str, *, status: str, new_id: str | None = None) -> FixView:
        """Record what became of a fix: ``applied`` (with the id of the record it made, if any),
        ``declined`` by the user, or ``stale`` (the record is not what it was when proposed)."""
        if status not in ("applied", "declined", "stale"):
            raise ValueError(f"a fix is applied, declined or stale, not {status!r}")
        now = self._clock.now_utc()
        with self._db.transaction(bump_state=False) as session:
            row = session.get(ConsistencyFix, fix_id)
            if row is None:
                raise ConsistencyStoreError(f"there is no fix {fix_id!r}")
            row.status = status
            row.new_id = new_id
            row.decided_at = now
            row.applied_at = now if status == "applied" else None
            session.flush()
            return _fix_view(row)
