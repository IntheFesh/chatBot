"""Which backups are kept (R-OPS-006) and which keys they still need (R-STO-003).

Policy, computed from the backup records and nothing else:

* **daily**: the newest ``ops.backup_keep_daily`` (14) days that have a backup;
* **weekly**: the newest ``ops.backup_keep_weekly`` (8) backups made on a Sunday (local date);
  the two sets overlap for the last two weeks, a backup is kept if it is in either;
* **pre-restore** backups (made by ``twin backup restore`` before it replaces anything) are not
  part of the rotation: the newest three are kept, the rest go.

A day has one backup: when there are two (a manual one after the daily one) the file is the
same name and the newer replaced the older.  Everything else is deleted - and only when it is
deleted can a key become unneeded: :func:`releasable_keys` lists the retired database keys that
no kept backup, and no live data, still depends on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

KEEP_PRE_RESTORE = 3
SUNDAY = 6


@dataclass(frozen=True)
class Candidate:
    """What the policy needs to know about one backup."""

    id: str
    kind: str
    local_date: str
    created_at: str  # sortable text (ISO UTC)


@dataclass(frozen=True)
class RetentionPlan:
    keep: frozenset[str]
    drop: frozenset[str]


def plan_retention(
    candidates: Iterable[Candidate], *, keep_daily: int, keep_weekly: int
) -> RetentionPlan:
    """Decide which of ``candidates`` (usable backups) stay."""
    rotating = [c for c in candidates if c.kind in ("daily", "manual")]
    restores = sorted(
        (c for c in candidates if c.kind == "pre_restore"),
        key=lambda c: c.created_at,
        reverse=True,
    )
    newest_per_day: dict[str, Candidate] = {}
    for candidate in sorted(rotating, key=lambda c: c.created_at):
        newest_per_day[candidate.local_date] = candidate
    days = sorted(newest_per_day, reverse=True)
    sundays = [d for d in days if date.fromisoformat(d).weekday() == SUNDAY]
    kept_days = set(days[: max(keep_daily, 0)]) | set(sundays[: max(keep_weekly, 0)])
    keep = {newest_per_day[d].id for d in kept_days}
    keep |= {c.id for c in restores[:KEEP_PRE_RESTORE]}
    every = {c.id for c in rotating} | {c.id for c in restores}
    return RetentionPlan(frozenset(keep), frozenset(every - keep))


def releasable_keys(
    retired: Iterable[int], kept_backup_keys: Iterable[int], live_keys: Iterable[int]
) -> list[int]:
    """Retired keys that nothing needs any more: no kept backup and no live value uses them."""
    needed = set(kept_backup_keys) | set(live_keys)
    return sorted(key for key in set(retired) if key not in needed)
