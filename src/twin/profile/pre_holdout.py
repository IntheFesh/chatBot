"""Is the pre-holdout derived data built for the cut-off in force? (R-TRN-013; round 18, D-882).

A re-split (``twin retrieval resplit``, or ``twin train export`` by itself, D-881) moves the
cut-off; the data derived from "everything before the cut-off" is then out of date until it is
rebuilt.  :func:`pre_holdout_state` answers, from version records and the job queue only (no
message is read), what the training export must know before it reads anything (iron rule 13):

* **profile and routine** - the cut-off their active pre-holdout versions were computed for
  (``data_range["cutoff"]``);
* **persona card** - the cut-off of the profile its statistics rules come from
  (``profile_version_id``; a card without one was written by hand and is taken as it is) and the
  cut-off its ``[自动-描述]`` was generated for (``provenance["cutoff"]``; a description without one
  was written by hand and is taken as it is);
* **waiting** - the jobs that rebuild this data and are still to run: a ``profile_rebuild`` of the
  pre-holdout scope (or ``all``), a ``persona_refresh`` of it (or ``all``), and a ``sticker_tag``
  job with context corrections (the corrections of the old cut-off were dropped at the re-split;
  the new ones change the sticker labels of the targets, so the plans of the samples come after
  them).  A job still waiting for ``twin jobs approve`` is not counted: it may never run;
* **describing** - the batches of a pre-holdout description job that is planned but not done.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from twin.clock import ensure_aware
from twin.ops.jobs import JobQueue, JobView
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.profile.queue import PROFILE_JOB
from twin.profile.store import VersionStore

if TYPE_CHECKING:
    from twin.services import Services

SCOPE: Final = "pre_holdout"
REBUILT_SCOPES: Final = frozenset({SCOPE, "all"})
REFRESH_JOB: Final = "persona_refresh"  # twin.profile.persona.jobs (not imported: no LLM here)
DESCRIBE_JOB: Final = "persona_generate"
STICKER_JOB: Final = "sticker_tag"  # twin.stickers.tag_jobs
WAITING: Final = ("pending", "running")
JOB_LIMIT: Final = 10_000

Description = Literal["current", "stale", "missing"]


def cutoff_of(record: dict[str, Any] | None) -> datetime | None:
    """The ``cutoff`` a version record names, as an aware moment (``None`` when it names none)."""
    raw = (record or {}).get("cutoff")
    if not raw:
        return None
    try:
        return ensure_aware(datetime.fromisoformat(str(raw)))
    except ValueError:
        return None


@dataclass(frozen=True)
class PreHoldoutState:
    """Where the pre-holdout derived data stands against the cut-off (see the module text)."""

    cutoff: datetime
    profile_cutoff: datetime | None  # None: no pre-holdout profile, or it names no cut-off
    routine_cutoff: datetime | None
    card_cutoff: datetime | None  # the cut-off of the profile behind the card's statistics
    card_by_hand: bool  # the card names no profile: written by hand, taken as it is
    description: Description
    description_cutoff: datetime | None
    waiting: tuple[str, ...] = ()  # job types still to run (the free rebuild)
    describing: tuple[str, ...] = ()  # batches of a planned pre-holdout description

    @property
    def profile_current(self) -> bool:
        return self.profile_cutoff == self.cutoff and self.routine_cutoff == self.cutoff

    @property
    def card_current(self) -> bool:
        return self.card_by_hand or self.card_cutoff == self.cutoff

    def stale(self) -> list[str]:
        """What is built for another cut-off with no job left to rebuild it, with the command."""
        if self.waiting:
            return []
        found = []
        if not self.profile_current:
            found.append(
                "the pre-holdout profile and routine were computed for the cut-off "
                f"{_moment(self.profile_cutoff)} / {_moment(self.routine_cutoff)}, the cut-off is "
                f"{_moment(self.cutoff)}: run `twin profile rebuild`"
            )
        elif not self.card_current:
            found.append(
                "the statistics rules of the pre-holdout persona card come from the profile of "
                f"the cut-off {_moment(self.card_cutoff)}: "
                "run `twin persona regenerate --scope pre_holdout --stats-only`"
            )
        return found


def _moment(value: datetime | None) -> str:
    return f"{value:%Y-%m-%d %H:%M} UTC" if value is not None else "(none)"


def _waiting(queue: JobQueue, job_type: str) -> list[JobView]:
    found = []
    for status in WAITING:
        for job in queue.list_jobs(status=status, job_type=job_type, limit=JOB_LIMIT):
            if job.requires_approval and job.approved_at is None:
                continue  # it may never be approved: not waited for
            found.append(job)
    return found


def rebuild_jobs_waiting(services: Services) -> tuple[str, ...]:
    """The job types that still rebuild pre-holdout data (see the module text)."""
    queue = JobQueue(services.db, services.clock)
    found = []
    for job_type in (PROFILE_JOB, REFRESH_JOB):
        if any(job.payload.get("scope") in REBUILT_SCOPES for job in _waiting(queue, job_type)):
            found.append(job_type)
    if any(job.payload.get("context") for job in _waiting(queue, STICKER_JOB)):
        found.append(STICKER_JOB)
    return tuple(found)


def description_batches(services: Services) -> tuple[str, ...]:
    """Batches of a pre-holdout description that is planned and not done (approved or not)."""
    queue = JobQueue(services.db, services.clock)
    found: list[str] = []
    for status in WAITING:
        for job in queue.list_jobs(status=status, job_type=DESCRIBE_JOB, limit=JOB_LIMIT):
            if job.payload.get("scope") == SCOPE and job.batch_id and job.batch_id not in found:
                found.append(job.batch_id)
    return tuple(found)


def pre_holdout_state(services: Services) -> PreHoldoutState:
    """Where the pre-holdout profile, routine and card stand against the cut-off in force."""
    cutoff = holdout_cutoff(services)
    versions = VersionStore(services.db, services.clock)
    profile = versions.active_profile(SCOPE)
    routine = versions.active_activity(SCOPE)
    store = PersonaStore(services.db, services.clock)
    card = store.active(SCOPE)
    card_cutoff: datetime | None = None
    by_hand = card is None or card.profile_version_id is None
    if card is not None and card.profile_version_id is not None:
        behind = versions.get_profile(card.profile_version_id)
        card_cutoff = cutoff_of(behind.data_range) if behind is not None else None
    description: Description = "missing"
    described: datetime | None = None
    if card is not None and compose.description_lines(card.sections):
        described = cutoff_of(store.provenance(card.id))
        description = "stale" if described is not None and described != cutoff else "current"
    return PreHoldoutState(
        cutoff=cutoff,
        profile_cutoff=cutoff_of(profile.data_range) if profile is not None else None,
        routine_cutoff=cutoff_of(routine.data_range) if routine is not None else None,
        card_cutoff=card_cutoff,
        card_by_hand=by_hand,
        description=description,
        description_cutoff=described,
        waiting=rebuild_jobs_waiting(services),
        describing=description_batches(services),
    )


__all__ = [
    "PreHoldoutState",
    "cutoff_of",
    "description_batches",
    "pre_holdout_state",
    "rebuild_jobs_waiting",
]
