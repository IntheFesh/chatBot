"""Recomputing the style profile and the activity model (R-PROF-001 to 005, R-ACT-001 to 006).

One pass over the stored messages feeds, per scope, a full-window and a recent-window metric
collector, the activity collector and the phrase counters (:class:`_ScopePass`):

``live``
    all messages; the recent window is the ``profile.recent_days`` days that end at the newest
    message;
``pre_holdout``
    only messages before :func:`~twin.profile.holdout.holdout_cutoff`; the recent window is the
    ``profile.recent_days`` days that end at the cutoff (R-TRN-013).

The result of a scope is written as one ``profile_versions`` row and one ``activity_models``
row (linked), made the active versions of the scope.  If neither differs from the newest stored
version nothing is written (``unchanged``) unless ``force`` is given.  Message text only passes
through the counters; the stored metrics contain numbers and closed vocabularies, the frequent
sentences and n-grams go to the separately sealed ``phrases`` column.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import orjson
from sqlalchemy import func, select

from twin.config.lists import load_word_list, locate_list_file
from twin.config.settings import Settings
from twin.ingest.corpus import conversation_messages
from twin.ingest.times import SourceTime
from twin.profile.activity_collect import ActivityCollector
from twin.profile.activity_infer import build_activity_model
from twin.profile.activity_model import ActivityModel
from twin.profile.circular import signed_diff
from twin.profile.diffing import Change, diff_metrics
from twin.profile.holdout import HoldoutError, holdout_cutoff
from twin.profile.localtime import LocalClock, format_minute
from twin.profile.metrics import TEXT_KINDS, WindowCollector
from twin.profile.overrides import RoutineOverrides
from twin.profile.phrases import PhraseCollector
from twin.profile.rules import Rule, generate_rules, load_rules
from twin.profile.snapshot import ProfileMetrics, assemble_metrics
from twin.profile.store import ACTIVITY_ACTIVE, PROFILE_ACTIVE, VersionStore
from twin.profile.units import Rec
from twin.schedule.daytype import DayTypeCalendar
from twin.storage.chat_models import Message
from twin.storage.profile_models import ActivityModelVersion, ProfileVersion
from twin.storage.settings_store import put_setting

if TYPE_CHECKING:
    from twin.services import Services

ScopeName = Literal["live", "pre_holdout"]
SCOPE_ORDER: tuple[ScopeName, ...] = ("live", "pre_holdout")
SECONDS_PER_DAY = 86_400.0
ACTIVITY_SHIFT_MINUTES = 30.0


@dataclass
class BuildContext:
    """Everything a build needs besides the messages."""

    settings: Settings
    local_clock: LocalClock
    calendar: DayTypeCalendar
    known_codes: frozenset[str]
    rules: list[Rule]

    @classmethod
    def create(cls, services: Services) -> BuildContext:
        settings = services.settings
        root = services.paths.root
        codes = load_word_list(locate_list_file(root, settings.profile.emoji_codes_file))
        rules = load_rules(locate_list_file(root, settings.profile.rules_file))
        holidays = RoutineOverrides(services.db, services.clock).holiday_ranges()
        return cls(
            settings=settings,
            local_clock=LocalClock(SourceTime.from_config(settings.time)),
            calendar=DayTypeCalendar(
                zone_countries=settings.safety.timezone_country, holiday_ranges=holidays
            ),
            known_codes=frozenset(codes),
            rules=rules,
        )


@dataclass
class ScopeResult:
    scope: str
    status: Literal["created", "unchanged", "skipped"]
    note: str = ""
    profile_version_id: str | None = None
    activity_version_id: str | None = None
    her_messages: int = 0
    changes: tuple[Change, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class BuildReport:
    results: list[ScopeResult] = field(default_factory=list)

    def result(self, scope: str) -> ScopeResult | None:
        return next((r for r in self.results if r.scope == scope), None)


# ------------------------------------------------------------------- reading


def iter_records(services: Services, context: BuildContext) -> Iterator[Rec]:
    """The messages of both sides in time order, as the statistics see them."""
    stmt = conversation_messages().execution_options(yield_per=5000)
    clock, calendar = context.local_clock, context.calendar
    with services.db.session() as session:
        for row in session.scalars(stmt):
            kind = row.kind
            if kind == "system":
                continue
            moment = row.create_time_utc
            stamp = clock.stamp(moment)
            text = row.text if kind in TEXT_KINDS else None
            yield Rec(
                row.id,
                moment.timestamp(),
                not row.is_sent,
                kind,
                text,
                row.sticker_md5,
                stamp,
                calendar.day_type(stamp.day, stamp.zone),
            )


def newest_message_time(services: Services) -> datetime | None:
    with services.db.session() as session:
        found = session.scalar(
            select(func.max(Message.create_time_utc)).where(Message.kind != "system")
        )
    return found if isinstance(found, datetime) else None


# ---------------------------------------------------------------- one scope


class _ScopePass:
    """The collectors of one scope."""

    def __init__(
        self, scope: ScopeName, context: BuildContext, end_ts: float | None, recent_start: float
    ) -> None:
        profile = context.settings.profile
        gap, segment = float(profile.burst_gap_s), profile.segment_gap_min * 60.0
        self.scope = scope
        self.end_ts = end_ts
        self.recent_start = recent_start
        self.full = WindowCollector(gap, segment, context.known_codes)
        self.recent = WindowCollector(gap, segment, context.known_codes)
        self.activity = ActivityCollector()
        self.phrases = {True: PhraseCollector(), False: PhraseCollector()}

    def feed(self, rec: Rec) -> None:
        if self.end_ts is not None and rec.ts >= self.end_ts:
            return
        step = self.full.feed(rec)
        self.activity.feed(rec, step)
        if rec.ts >= self.recent_start:
            self.recent.feed(rec)
        if rec.text:
            self.phrases[rec.her].feed(rec.text)


def _canonical(data: Any) -> bytes:
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS)


def _digest(*parts: bytes) -> str:
    sha = hashlib.sha256()
    for part in parts:
        sha.update(part)
        sha.update(b"\x00")
    return sha.hexdigest()


def activity_changes(old: ActivityModel, new: ActivityModel) -> list[str]:
    """Short sentences about what moved between two activity models."""
    lines: list[str] = []
    for key, window in new.sleep.windows.items():
        before = old.sleep.windows.get(key)
        if before is None:
            lines.append(
                f"睡眠（{key}）新增：{format_minute(window.onset_min)}–{format_minute(window.wake_min)}"
            )
            continue
        for label, a, b in (
            ("入睡", before.onset_min, window.onset_min),
            ("起床", before.wake_min, window.wake_min),
        ):
            shift = signed_diff(b, a)
            if abs(shift) >= ACTIVITY_SHIFT_MINUTES:
                moved = f"{format_minute(a)} → {format_minute(b)}（{shift:+.0f} 分钟）"
                lines.append(f"{label}时间（{key}）{moved}")
    if old.initiations_per_day and abs(new.initiations_per_day - old.initiations_per_day) > (
        0.1 * old.initiations_per_day
    ):
        lines.append(f"每天先开口 {old.initiations_per_day:.1f} → {new.initiations_per_day:.1f} 次")
    before_busy = sorted((k, w.label()) for k, ws in old.busy.items() for w in ws)
    after_busy = sorted((k, w.label()) for k, ws in new.busy.items() for w in ws)
    if before_busy != after_busy:
        lines.append("忙碌时段：" + ("、".join(f"{k} {lab}" for k, lab in after_busy) or "无"))
    return lines


@dataclass
class _Built:
    scope: ScopeName
    metrics: dict[str, Any]
    rules_text: str
    phrases: dict[str, Any]
    activity: ActivityModel
    data_range: dict[str, Any]
    her_messages: int


def _finish_scope(scope_pass: _ScopePass, context: BuildContext, cutoff: datetime | None) -> _Built:
    profile = context.settings.profile
    full_leaves = scope_pass.full.leaves()
    recent_leaves = scope_pass.recent.leaves()
    metrics = assemble_metrics(
        scope=scope_pass.scope,
        config={
            "burst_gap_s": profile.burst_gap_s,
            "segment_gap_min": profile.segment_gap_min,
            "recent_days": profile.recent_days,
            "recency_weight": profile.recency_weight,
            "source_timezone": context.settings.time.source_timezone,
        },
        weight=profile.recency_weight,
        full=full_leaves,
        recent=recent_leaves,
        full_info=scope_pass.full.info(),
        recent_info=scope_pass.recent.info(),
    )
    rule_lines = generate_rules(ProfileMetrics(metrics), context.rules)
    phrases = {
        "her": scope_pass.phrases[True].result(),
        "user": scope_pass.phrases[False].result(),
    }
    raw = scope_pass.activity.finish()
    activity = build_activity_model(
        raw, context.settings.activity, context.calendar, scope_pass.scope
    )
    her_messages = int(scope_pass.full.sides[True].messages)
    data_range = {
        "scope": scope_pass.scope,
        "full": metrics["windows"]["full"],
        "recent": metrics["windows"]["recent"],
        "cutoff": cutoff.isoformat() if cutoff is not None else None,
        "recent_days": profile.recent_days,
        "source_timezone": context.settings.time.source_timezone,
        "source_timezone_ranges": [
            {"from": r.from_.isoformat(), "to": r.to.isoformat(), "tz": r.tz}
            for r in context.settings.time.source_timezone_ranges
        ],
    }
    return _Built(
        scope_pass.scope,
        metrics,
        "\n".join(rule_lines),
        phrases,
        activity,
        data_range,
        her_messages,
    )


def _persist(services: Services, built: _Built, *, reason: str, force: bool) -> ScopeResult:
    store = VersionStore(services.db, services.clock)
    scope = built.scope
    profile_hash = _digest(_canonical(built.metrics), built.rules_text.encode("utf-8"))
    activity_json = built.activity.to_json()
    activity_hash = _digest(_canonical(activity_json))
    latest = store.latest_profile(scope)
    latest_activity = store.latest_activity(scope)
    if (
        not force
        and latest is not None
        and latest_activity is not None
        and latest.input_hash == profile_hash
        and latest_activity.input_hash == activity_hash
    ):
        with services.db.transaction(bump_state=True) as session:
            put_setting(session, PROFILE_ACTIVE + scope, latest.id, clock=services.clock, by=reason)
            put_setting(
                session,
                ACTIVITY_ACTIVE + scope,
                latest_activity.id,
                clock=services.clock,
                by=reason,
            )
        return ScopeResult(
            scope,
            "unchanged",
            "no change since the newest version",
            latest.id,
            latest_activity.id,
            built.her_messages,
            (),
            built.activity.sleep.warnings,
        )
    parent = store.active_profile(scope)
    parent_activity = store.active_activity(scope)
    changes: list[Change] = []
    activity_lines: list[str] = []
    if parent is not None:
        changes = diff_metrics(store.metrics(parent.id), ProfileMetrics(built.metrics))
    if parent_activity is not None:
        activity_lines = activity_changes(store.activity_model(parent_activity.id), built.activity)
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=True) as session:
        version = ProfileVersion(
            scope=scope,
            parent_id=parent.id if parent else None,
            reason=reason,
            input_hash=profile_hash,
            her_messages=built.her_messages,
            data_range=built.data_range,
            metrics=built.metrics,
            phrases=built.phrases,
            summary_rules=built.rules_text,
            diff=[c.to_json() for c in changes],
            created_at=now,
            updated_at=now,
        )
        session.add(version)
        session.flush()
        model_row = ActivityModelVersion(
            scope=scope,
            parent_id=parent_activity.id if parent_activity else None,
            profile_version_id=version.id,
            reason=reason,
            input_hash=activity_hash,
            her_messages=built.her_messages,
            data_range=built.data_range,
            model=activity_json,
            diff=activity_lines,
            created_at=now,
            updated_at=now,
        )
        session.add(model_row)
        session.flush()
        put_setting(session, PROFILE_ACTIVE + scope, version.id, clock=services.clock, by=reason)
        put_setting(session, ACTIVITY_ACTIVE + scope, model_row.id, clock=services.clock, by=reason)
        version_id, model_id = version.id, model_row.id
    return ScopeResult(
        scope,
        "created",
        "",
        version_id,
        model_id,
        built.her_messages,
        tuple(changes),
        built.activity.sleep.warnings,
    )


# --------------------------------------------------------------------- entry


def scopes_for(scope: str) -> tuple[ScopeName, ...]:
    if scope == "all":
        return SCOPE_ORDER
    for name in SCOPE_ORDER:
        if scope == name:
            return (name,)
    raise ValueError(f"scope must be live, pre_holdout or all, not {scope!r}")


def rebuild(
    services: Services,
    scope: str = "all",
    *,
    reason: str = "manual",
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    """Recompute the profile and the activity model of ``scope`` and store new versions."""
    report = BuildReport()
    wanted = scopes_for(scope)
    newest = newest_message_time(services)
    if newest is None:
        for name in wanted:
            report.results.append(ScopeResult(name, "skipped", "no messages imported yet"))
        return report
    context = BuildContext.create(services)
    recent_seconds = context.settings.profile.recent_days * SECONDS_PER_DAY
    passes: list[_ScopePass] = []
    cutoff: datetime | None = None
    for name in wanted:
        if name == "live":
            passes.append(_ScopePass("live", context, None, newest.timestamp() - recent_seconds))
            continue
        try:
            cutoff = holdout_cutoff(services)
        except HoldoutError as exc:
            report.results.append(ScopeResult(name, "skipped", str(exc)))
            continue
        passes.append(
            _ScopePass(name, context, cutoff.timestamp(), cutoff.timestamp() - recent_seconds)
        )
    if not passes:
        return report
    if progress:
        progress("reading messages")
    for rec in iter_records(services, context):
        for item in passes:
            item.feed(rec)
    for item in passes:
        if progress:
            progress(f"computing {item.scope}")
        if item.full.sides[True].messages == 0:
            report.results.append(
                ScopeResult(item.scope, "skipped", "no messages from her in this scope")
            )
            continue
        built = _finish_scope(item, context, cutoff if item.scope == "pre_holdout" else None)
        report.results.append(_persist(services, built, reason=reason, force=force))
    return report
