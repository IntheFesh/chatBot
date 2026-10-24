"""What later rounds import: the profile and the routine in their two scopes (R-TRN-013).

``live`` is for the running bot; ``pre_holdout`` (data before :func:`holdout_cutoff`) is for
the training export and the evaluation sandbox, so that nothing they read has seen the held-out
period.  Every function reads the *active* version of the scope; ``None`` means nothing has
been computed yet (run ``twin profile rebuild``).

::

    from twin.profile.api import holdout_cutoff, load_profile, load_activity_model

    cutoff = holdout_cutoff(services)                       # the one split point
    profile = load_profile(services, "pre_holdout")
    burst = profile.metrics.distribution("her", "burst_size")        # sampleable
    rules = profile.version.summary_rules                      # numeric style rules
    model = load_activity_model(services, "pre_holdout")
    state = model.typical_state(time(13, 30), "workday")       # deep_sleep|sleep_edge|busy|free
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from twin.profile.activity_model import ActivityModel
from twin.profile.holdout import (
    Holdout,
    HoldoutError,
    ResplitResult,
    get_holdout,
    holdout_cutoff,
    on_holdout_change,
    resplit_holdout,
)
from twin.profile.snapshot import ProfileMetrics
from twin.profile.store import ProfileVersionView, VersionStore, load_activity

if TYPE_CHECKING:
    from twin.services import Services

__all__ = [
    "Holdout",
    "HoldoutError",
    "ProfileSnapshot",
    "ResplitResult",
    "get_holdout",
    "holdout_cutoff",
    "load_activity_model",
    "load_profile",
    "on_holdout_change",
    "resplit_holdout",
]


@dataclass(frozen=True)
class ProfileSnapshot:
    """An active profile version with typed access to its metrics."""

    version: ProfileVersionView
    metrics: ProfileMetrics
    _store: VersionStore

    def phrases(self) -> dict[str, Any] | None:
        """Frequent sentences, n-grams and address candidates (real text; local use only)."""
        return self._store.phrases(self.version.id)


def load_profile(services: Services, scope: str = "live") -> ProfileSnapshot | None:
    store = VersionStore(services.db, services.clock)
    version = store.active_profile(scope)
    if version is None:
        return None
    return ProfileSnapshot(version, store.metrics(version.id), store)


def load_activity_model(
    services: Services, scope: str = "live", *, apply_overrides: bool = True
) -> ActivityModel | None:
    """The active activity model, with the manual routine corrections applied by default."""
    return load_activity(services.db, services.clock, scope, apply_overrides=apply_overrides)
