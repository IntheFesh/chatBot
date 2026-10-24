"""Ready data: the smallest live profile, routine model and persona card (R18-B8, D-870).

The bot answers and writes first only once the three exist (:mod:`twin.engine.readiness`).  Most
tests of the engine and the scheduler are about something else and used to run on an empty
database; their supports (:func:`tests.support.engine_harness.build_harness`,
:func:`tests.support.proactive_world.build_world`, :class:`tests.support.life_world.LifeWorld`
and the fixtures that call ``build_engine``) put these rows in, so that the production check finds
the bot ready - there is no switch that turns the check off.  The tests of the not-ready path
simply leave them out.

Everything here is synthetic and neutral: the profile has no measured metric (the engine falls
back to its reference numbers exactly as with no profile), the routine model has no sleep, no busy
window and flat curves (the day plan has no sleep and default meals - what D-231 does with a model
that infers no sleep window), and the card describes nobody real.
"""

from __future__ import annotations

import hashlib
import json

from twin.ingest.runs import create_run
from twin.profile.activity_model import CURVE_KEYS, ActivityModel, SleepProfile
from twin.profile.distribution import BucketedDistribution
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.profile.snapshot import assemble_metrics
from twin.profile.store import ACTIVITY_ACTIVE, PROFILE_ACTIVE, VersionStore
from twin.services import Services
from twin.storage.chat_models import ImportRun
from twin.storage.profile_models import ActivityModelVersion, ProfileVersion
from twin.storage.settings_store import put_setting

LIVE = "live"
DESCRIPTION = "### 风格\n- 说话简短，常用语气词（合成）\n\n### 基本情况\n- 一位合成的测试人物\n"


def neutral_model(zone: str = "America/Chicago") -> ActivityModel:
    """A routine model that knows nothing: no sleep, no busy window, flat curves, no latencies."""
    flat = tuple([1.0] * 96)
    curves = dict.fromkeys(CURVE_KEYS, flat)
    empty = BucketedDistribution.from_counters({}, sizes=(1, 4), min_samples=20)
    return ActivityModel(
        scope=LIVE,
        zone=zone,
        edge_minutes=30,
        days={"all": 0, "workday": 0, "weekend": 0, "holiday": 0},
        rate=curves,
        rate_raw=curves,
        initiation=curves,
        initiation_raw=curves,
        latency=empty,
        latency_workday=empty,
        sleep=SleepProfile({}, "none", 0, ()),
        busy={},
        her_messages=0,
        initiations_per_day=0.0,
    )


def described_card() -> str:
    """A live card with statistics rules and a (synthetic) description of her."""
    return (
        compose.stats_block(["单条通常很短（合成）"])
        + compose.auto_block(DESCRIPTION)
        + compose.manual_block([], None)
    )


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def install_profile(services: Services) -> str:
    """An active live profile with no measured metric."""
    metrics = assemble_metrics(
        scope=LIVE,
        config={},
        weight=0.6,
        full={"her": {}, "user": {}},
        recent=None,
        full_info={"start": None, "end": None, "days": 0, "her_messages": 0, "user_messages": 0},
        recent_info={"start": None, "end": None, "days": 0, "her_messages": 0, "user_messages": 0},
    )
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=True) as session:
        row = ProfileVersion(
            scope=LIVE,
            parent_id=None,
            reason="test-ready",
            input_hash=_digest(metrics),
            her_messages=0,
            data_range={"scope": LIVE},
            metrics=metrics,
            phrases=None,
            summary_rules="",
            diff=[],
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        put_setting(session, PROFILE_ACTIVE + LIVE, row.id, clock=services.clock, by="test")
        return row.id


def install_routine(services: Services, model: ActivityModel | None = None) -> str:
    """``model`` (a neutral one by default) as the active live routine model."""
    chosen = model or neutral_model()
    store = VersionStore(services.db, services.clock)
    profile = store.active_profile(LIVE)
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=True) as session:
        row = ActivityModelVersion(
            scope=LIVE,
            parent_id=None,
            profile_version_id=profile.id if profile else None,
            reason="test-ready",
            input_hash=_digest(chosen.to_json()),
            her_messages=chosen.her_messages,
            data_range={},
            model=chosen.to_json(),
            diff=[],
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        put_setting(session, ACTIVITY_ACTIVE + LIVE, row.id, clock=services.clock, by="test")
        return row.id


def install_card(services: Services) -> str:
    """A described live persona card (a new version on top of whatever is there)."""
    return (
        PersonaStore(services.db, services.clock)
        .add_version(LIVE, described_card(), reason="generate")
        .id
    )


def record_import(
    services: Services, *, status: str = "done", processed: int = 0, total: int | None = None
) -> str:
    """A row of ``import_runs`` as an import leaves it (what ``twin import status`` reads)."""
    run_id = create_run(
        services.db,
        services.clock,
        source_dir="synthetic-export",
        target_username="synthetic-target",
        export_id=None,
        fingerprint=_digest([status, processed, total]),
        total=total,
    )
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        row = session.get(ImportRun, run_id)
        assert row is not None
        row.status = status
        row.phase = "done" if status == "done" else "messages"
        row.processed = processed
        row.started_at = now
        row.finished_at = now if status in ("done", "failed") else None
        row.updated_at = now
    return run_id


def install_ready(services: Services, *, model: ActivityModel | None = None) -> None:
    """Whatever of the three is missing, so that the production check finds the bot ready.

    ``model`` is the routine model to store when none is there (a neutral one by default).
    """
    store = VersionStore(services.db, services.clock)
    if store.active_profile(LIVE) is None:
        install_profile(services)
    if store.active_activity(LIVE) is None:
        install_routine(services, model)
    card = PersonaStore(services.db, services.clock).active(LIVE)
    if card is None or not compose.description_lines(card.sections):
        install_card(services)
