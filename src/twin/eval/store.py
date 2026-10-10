"""Reading and writing ``eval_runs`` and ``eval_items`` (R-STO-006, R-EVAL-001, R-EVAL-010).

:class:`EvalStore` is the only code that touches the two tables.  A run is created with its
parameters, the items are added when the run is drawn, and every later step - the generation by a
batch job, a judgement of the user - updates one item in its own short transaction, so a run that
is stopped anywhere continues from the item it had reached.  Nothing is ever deleted.

The sealed ``payload`` of an item is a JSON document that the steps fill in turn; the store merges
the keys it is given and never reads their meaning.  The plain columns carry what is counted:
status, outcome, score, the strata of a blind item.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from twin.clock import Clock, ensure_aware
from twin.storage.db import Database
from twin.storage.eval_models import EvalItem, EvalRun

RESOLVE_LIMIT = 200


class EvalStoreError(LookupError):
    """A run or an item that was asked for does not exist, or the id is ambiguous."""


@dataclass(frozen=True)
class RunView:
    """A row of ``eval_runs``."""

    id: str
    kind: str
    status: str
    mode: str | None
    milestone: str | None
    verdict: str | None
    backends: tuple[str, ...]
    batch_ids: tuple[str, ...]
    params: dict[str, Any]
    summary: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None

    @property
    def finished(self) -> bool:
        return self.status == "done"


@dataclass(frozen=True)
class ItemView:
    """A row of ``eval_items`` with its payload decrypted."""

    id: str
    run_id: str
    seq: int
    sample_key: str
    backend: str
    source: str | None
    period: str | None
    length_bin: str | None
    at: datetime
    status: str
    left_is_bot: bool | None
    outcome: str | None
    auto_outcome: str | None
    score: float | None
    cost_usd: float
    judged_at: datetime | None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """A judgement that counts: the user decided (a skipped pair does not)."""
        return self.status == "judged" and self.outcome is not None


@dataclass(frozen=True)
class NewItem:
    """An item to add to a run."""

    sample_key: str
    backend: str
    at: datetime
    payload: Mapping[str, Any]
    source: str | None = None
    period: str | None = None
    length_bin: str | None = None
    left_is_bot: bool | None = None


def _run_view(row: EvalRun) -> RunView:
    return RunView(
        id=row.id,
        kind=row.kind,
        status=row.status,
        mode=row.mode,
        milestone=row.milestone,
        verdict=row.verdict,
        backends=tuple(row.backends or ()),
        batch_ids=tuple(row.batch_ids or ()),
        params=dict(row.params or {}),
        summary=dict(row.summary or {}),
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def _item_view(row: EvalItem, *, with_payload: bool = True) -> ItemView:
    return ItemView(
        id=row.id,
        run_id=row.run_id,
        seq=row.seq,
        sample_key=row.sample_key,
        backend=row.backend,
        source=row.source,
        period=row.period,
        length_bin=row.length_bin,
        at=row.at,
        status=row.status,
        left_is_bot=row.left_is_bot,
        outcome=row.outcome,
        auto_outcome=row.auto_outcome,
        score=row.score,
        cost_usd=row.cost_usd,
        judged_at=row.judged_at,
        payload=dict(row.payload) if with_payload else {},
    )


class EvalStore:
    """The evaluation tables (see the module description)."""

    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # ------------------------------------------------------------------- runs

    def create_run(
        self,
        kind: str,
        *,
        mode: str | None = None,
        backends: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
        milestone: str | None = None,
        status: str = "planned",
        verdict: str | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> RunView:
        finished = self._clock.now_utc() if status == "done" else None
        with self._db.transaction(bump_state=False) as session:
            row = EvalRun(
                kind=kind,
                status=status,
                mode=mode,
                milestone=milestone,
                verdict=verdict,
                backends=list(backends),
                batch_ids=[],
                params=dict(params or {}),
                summary=dict(summary or {}),
                finished_at=finished,
            )
            session.add(row)
            session.flush()
            return _run_view(row)

    def get_run(self, run_id: str) -> RunView:
        """The run with this id, or the only run whose id starts with it."""
        with self._db.session() as session:
            row = session.get(EvalRun, run_id)
            if row is None:
                found = session.scalars(
                    select(EvalRun).where(EvalRun.id.like(f"{run_id}%")).limit(2)
                ).all()
                if len(found) > 1:
                    raise EvalStoreError(f"{run_id!r} is the start of more than one run")
                row = found[0] if found else None
            if row is None:
                raise EvalStoreError(f"there is no evaluation run {run_id!r}")
            return _run_view(row)

    def latest_run(
        self,
        kind: str,
        *,
        milestone: str | None = None,
        status: str | None = None,
        backend: str | None = None,
    ) -> RunView | None:
        """The newest run of ``kind`` (optionally of one milestone, status or with a backend)."""
        for run in self.list_runs(kind, limit=RESOLVE_LIMIT):
            if milestone is not None and run.milestone != milestone:
                continue
            if status is not None and run.status != status:
                continue
            if backend is not None and backend not in run.backends:
                continue
            return run
        return None

    def list_runs(self, kind: str | None = None, *, limit: int = 20) -> list[RunView]:
        """Runs, newest first."""
        stmt = select(EvalRun).order_by(EvalRun.created_at.desc(), EvalRun.id.desc()).limit(limit)
        if kind is not None:
            stmt = stmt.where(EvalRun.kind == kind)
        with self._db.session() as session:
            return [_run_view(row) for row in session.scalars(stmt)]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        verdict: str | None = None,
        batch_ids: Sequence[str] | None = None,
        params: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> RunView:
        """Change a run; ``params`` and ``summary`` are merged key by key."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(EvalRun, run_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation run {run_id!r}")
            if status is not None:
                row.status = status
                row.finished_at = self._clock.now_utc() if status == "done" else None
            if verdict is not None:
                row.verdict = verdict
            if batch_ids is not None:
                row.batch_ids = list(batch_ids)
            if params is not None:
                row.params = {**(row.params or {}), **params}
            if summary is not None:
                row.summary = {**(row.summary or {}), **summary}
            session.flush()
            return _run_view(row)

    # ------------------------------------------------------------------ items

    def add_items(self, run_id: str, items: Sequence[NewItem], *, start: int | None = None) -> int:
        """Add items to a run, numbered after the last one; returns how many were added."""
        with self._db.transaction(bump_state=False) as session:
            if session.get(EvalRun, run_id) is None:
                raise EvalStoreError(f"there is no evaluation run {run_id!r}")
            if start is None:
                last = session.scalars(
                    select(EvalItem.seq)
                    .where(EvalItem.run_id == run_id)
                    .order_by(EvalItem.seq.desc())
                    .limit(1)
                ).first()
                start = 0 if last is None else last + 1
            for offset, item in enumerate(items):
                session.add(
                    EvalItem(
                        run_id=run_id,
                        seq=start + offset,
                        sample_key=item.sample_key,
                        backend=item.backend,
                        source=item.source,
                        period=item.period,
                        length_bin=item.length_bin,
                        at=ensure_aware(item.at),
                        status="pending",
                        left_is_bot=item.left_is_bot,
                        cost_usd=0.0,
                        payload=dict(item.payload),
                    )
                )
        return len(items)

    def items(
        self,
        run_id: str,
        *,
        status: str | Sequence[str] | None = None,
        backend: str | None = None,
        with_payload: bool = True,
    ) -> list[ItemView]:
        """The items of a run in order, optionally of some statuses or of one backend."""
        stmt = select(EvalItem).where(EvalItem.run_id == run_id).order_by(EvalItem.seq)
        if isinstance(status, str):
            stmt = stmt.where(EvalItem.status == status)
        elif status is not None:
            stmt = stmt.where(EvalItem.status.in_(list(status)))
        if backend is not None:
            stmt = stmt.where(EvalItem.backend == backend)
        with self._db.session() as session:
            return [_item_view(row, with_payload=with_payload) for row in session.scalars(stmt)]

    def item(self, item_id: str) -> ItemView:
        with self._db.session() as session:
            row = session.get(EvalItem, item_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation item {item_id!r}")
            return _item_view(row)

    def counts(self, run_id: str) -> Counter[str]:
        """How many items of the run are in each status."""
        with self._db.session() as session:
            statuses = session.scalars(select(EvalItem.status).where(EvalItem.run_id == run_id))
            return Counter(statuses)

    def save_generated(
        self,
        item_id: str,
        payload: Mapping[str, Any],
        *,
        cost_usd: float,
        status: str = "generated",
    ) -> ItemView:
        """Store the bot's side of an item (merged into the payload) and the cost it took."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(EvalItem, item_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation item {item_id!r}")
            row.payload = {**dict(row.payload), **payload}
            row.status = status
            row.cost_usd = row.cost_usd + cost_usd
            session.flush()
            return _item_view(row)

    def save_auto(
        self, item_id: str, payload: Mapping[str, Any], *, auto_outcome: str | None, cost_usd: float
    ) -> ItemView:
        """Store the answer and the automatic verdict of a memory question."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(EvalItem, item_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation item {item_id!r}")
            row.payload = {**dict(row.payload), **payload}
            row.status = "generated"
            row.auto_outcome = auto_outcome
            row.cost_usd = row.cost_usd + cost_usd
            session.flush()
            return _item_view(row)

    def judge(
        self,
        item_id: str,
        outcome: str | None,
        *,
        score: float | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> ItemView:
        """Record the user's decision: ``outcome`` ``None`` skips the item (it does not count)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(EvalItem, item_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation item {item_id!r}")
            row.outcome = outcome
            row.score = score if outcome is not None else None
            row.status = "judged" if outcome is not None else "skipped"
            row.judged_at = self._clock.now_utc()
            if payload:
                row.payload = {**dict(row.payload), **payload}
            session.flush()
            return _item_view(row)

    def mark_failed(self, item_id: str, reason: str) -> ItemView:
        """An item whose bot reply could not be produced (the reason is a code, not text)."""
        with self._db.transaction(bump_state=False) as session:
            row = session.get(EvalItem, item_id)
            if row is None:
                raise EvalStoreError(f"there is no evaluation item {item_id!r}")
            row.status = "failed"
            row.payload = {**dict(row.payload), "failure": reason}
            session.flush()
            return _item_view(row)

    def used_sample_keys(self, kind: str, *, except_run: str | None = None) -> set[str]:
        """Samples that earlier runs of ``kind`` already used (the user saw them, or will)."""
        stmt = (
            select(EvalItem.sample_key)
            .join(EvalRun, EvalRun.id == EvalItem.run_id)
            .where(EvalRun.kind == kind)
            .where((EvalRun.status != "cancelled") | EvalItem.outcome.is_not(None))
        )
        if except_run is not None:
            stmt = stmt.where(EvalRun.id != except_run)
        with self._db.session() as session:
            return set(session.scalars(stmt))
