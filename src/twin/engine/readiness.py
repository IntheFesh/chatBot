"""Is the bot ready to talk as her?  The one test, for the engine and the scheduler (R18-B8).

The bot speaks only from what it knows of her.  It is **ready** when the three things it speaks
from all exist in the live scope (D-870):

==========  ========================================================================================
画像        the style profile: a ``profile_versions`` row of the live scope (``twin profile
            rebuild``, queued by the import)
作息模型    the routine model: an ``activity_model_versions`` row of the live scope (computed with
            the profile).  A model that could infer no sleep window still counts: the day plan then
            has no sleep and warns, as D-231 says
人设卡      the live persona card **with its description** (``[自动-描述]``, R-PERS-001): the first
            version holds the statistics rules only and says nothing of her - the same reading as
            ``twin train export`` (:func:`~twin.profile.persona.refresh.has_description`, D-817)
==========  ========================================================================================

Before that (D-871) the engine stores what the user writes and does not answer it; it says in one
system message what is missing and what stands in the way - the import and its progress, the
recomputation, the description batch that waits for ``twin jobs approve <batch>`` - at most once
in six hours for the same set of missing things (:class:`NoticeBook`, kept in the ``settings``
table so that a restart does not say it again).  The proactive scheduler refuses every message
with ``not_ready`` (D-872).  No clock time is made up and the reference routine of SPEC §0 is not
used: without her own routine she simply does not talk yet.

:func:`check_readiness` is the test; :func:`explain` the lines of "what stands in the way";
:class:`ReadinessGate` is what the engine holds (the test, the text of the notice and its book).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from twin.ingest.runs import RunView, latest_run
from twin.ops.jobs import JobQueue, JobView
from twin.ops.logging import get_logger
from twin.profile.persona.jobs import PERSONA_JOB
from twin.profile.persona.refresh import MIN_DESCRIBED_MESSAGES, description_due, has_description
from twin.profile.queue import PROFILE_JOB
from twin.profile.store import VersionStore
from twin.storage.settings_store import get_setting, put_setting

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.readiness")

PROFILE, ROUTINE, PERSONA = "profile", "routine", "persona"
PARTS = (PROFILE, ROUTINE, PERSONA)
PART_LABELS = {PROFILE: "画像", ROUTINE: "作息模型", PERSONA: "人设卡"}
LIVE = "live"
NOTICE_KEY = "engine.readiness_notice"
NOTICE_EVERY = timedelta(hours=6)  # the same missing things are told at most this often (D-871)
JOBS_LOOKED_AT = 50  # the newest jobs of a type that are looked through for the explanation
ACTIVE_RUNS = ("queued", "running")
STOPPED_RUNS = ("interrupted", "failed")

NOT_READY = "未就绪：缺 {missing}"
READY = "已就绪"
NOTICE_HEAD = (
    "还没准备好，先不回复：缺 {missing}。你的消息都存下了，准备好以后按她的作息回复"
    "（她在睡就等她醒）。"
)
IMPORT_NONE = "导入：还没有导入聊天记录，在电脑上运行 `twin import <导出目录>`"
IMPORT_ACTIVE = "导入：进行中，已处理 {processed} / {total}{percent}；`twin import status` 看进度"
IMPORT_STOPPED = "导入：{status}（已处理 {processed} / {total}），`twin import --resume` 接着导"
IMPORT_DONE = "导入：已完成"
IMPORT_STATUS_WORDS = {"interrupted": "中断了", "failed": "失败了"}
PROFILE_QUEUED = "画像与作息：正在重算（排队中，应用运行时自动执行）"
PROFILE_FAILED = (
    "画像与作息：重算失败，`twin jobs list --type profile_rebuild` 查看，"
    "`twin jobs retry --failed` 重试"
)
PROFILE_AFTER_IMPORT = "画像与作息：导入完成后自动计算"
PROFILE_MISSING = "画像与作息：还没有算，运行 `twin profile rebuild`"
PERSONA_APPROVE = "人设卡：描述生成等你批准，`twin jobs approve {batch}`（估算 ${usd:.2f}）"
PERSONA_APPROVED = "人设卡：描述已批准，在 DeepSeek 非高峰时段生成（应用运行时自动执行）"
PERSONA_RUNNING = "人设卡：描述正在生成"
PERSONA_FAILED = (
    "人设卡：描述生成失败，`twin jobs list --type persona_generate` 查看，"
    "`twin jobs retry --failed` 重试"
)
PERSONA_TOO_FEW = (
    "人设卡：她的消息只有 {count} 条，不够生成描述（至少 {least} 条），先导入更多聊天记录"
)
PERSONA_AFTER_PROFILE = (
    "人设卡：画像算好后生成描述（`twin persona regenerate` 给出估算，批准后执行）"
)
PERSONA_MISSING = (
    "人设卡：还没有描述，运行 `twin persona regenerate`，再 `twin jobs approve <批次号>`"
)


@dataclass(frozen=True)
class Readiness:
    """What is missing for the bot to talk as her (``missing`` empty: ready)."""

    missing: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def key(self) -> str:
        """The kind of the notice: the set of missing things (D-871 "同类")."""
        return ",".join(part for part in PARTS if part in self.missing)

    @property
    def label(self) -> str:
        """``画像、作息模型、人设卡``: the missing things in words."""
        return "、".join(PART_LABELS[part] for part in PARTS if part in self.missing)

    def status_line(self) -> str:
        """``已就绪`` or ``未就绪：缺 …`` (``/状态``, ``twin health``)."""
        return READY if self.ready else NOT_READY.format(missing=self.label)


def check_readiness(services: Services) -> Readiness:
    """The test of D-870: the live profile, routine model and described persona card exist."""
    store = VersionStore(services.db, services.clock)
    missing: list[str] = []
    if store.active_profile(LIVE) is None:
        missing.append(PROFILE)
    if store.active_activity(LIVE) is None:
        missing.append(ROUTINE)
    if not has_description(services, LIVE):
        missing.append(PERSONA)
    return Readiness(tuple(missing))


# ----------------------------------------------------------------------- the explanation


def _count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "?"


def _import_line(run: RunView | None) -> str:
    if run is None:
        return IMPORT_NONE
    if run.status in ACTIVE_RUNS:
        percent = f"（{100.0 * run.processed / run.total:.0f}%）" if run.total else ""
        return IMPORT_ACTIVE.format(
            processed=_count(run.processed), total=_count(run.total), percent=percent
        )
    if run.status in STOPPED_RUNS:
        return IMPORT_STOPPED.format(
            status=IMPORT_STATUS_WORDS[run.status],
            processed=_count(run.processed),
            total=_count(run.total),
        )
    return IMPORT_DONE


def _jobs(queue: JobQueue, job_type: str) -> list[JobView]:
    return queue.list_jobs(job_type=job_type, limit=JOBS_LOOKED_AT)


def _profile_line(queue: JobQueue, run: RunView | None) -> str:
    jobs = _jobs(queue, PROFILE_JOB)
    if any(job.status in ("pending", "running") for job in jobs):
        return PROFILE_QUEUED
    if jobs and jobs[0].status == "failed":
        return PROFILE_FAILED
    if run is None or run.status != "done":
        return PROFILE_AFTER_IMPORT
    return PROFILE_MISSING


def _waiting_for_approval(job: JobView) -> bool:
    return job.status == "pending" and job.requires_approval and job.approved_at is None


def _persona_line(services: Services, queue: JobQueue, readiness: Readiness) -> str:
    jobs = [job for job in _jobs(queue, PERSONA_JOB) if _scope_of(job) == LIVE]
    waiting = [job for job in jobs if _waiting_for_approval(job)]
    if waiting:
        batch = waiting[0].batch_id or "<批次号>"
        usd = sum(
            job.estimated_cost_usd or 0.0
            for job in _jobs(queue, PERSONA_JOB)
            if job.batch_id == waiting[0].batch_id and _waiting_for_approval(job)
        )
        return PERSONA_APPROVE.format(batch=batch, usd=usd)
    if any(job.status == "running" for job in jobs):
        return PERSONA_RUNNING
    if any(job.status == "pending" for job in jobs):
        return PERSONA_APPROVED
    if jobs and jobs[0].status == "failed":
        return PERSONA_FAILED
    due = description_due(services, LIVE, split=False)
    if due.her_messages < MIN_DESCRIBED_MESSAGES:
        return PERSONA_TOO_FEW.format(count=due.her_messages, least=MIN_DESCRIBED_MESSAGES)
    if PROFILE in readiness.missing:
        return PERSONA_AFTER_PROFILE
    return PERSONA_MISSING


def _scope_of(job: JobView) -> str | None:
    payload: Any = job.payload
    return str(payload.get("scope")) if isinstance(payload, dict) else None


def explain(services: Services, readiness: Readiness) -> list[str]:
    """What stands between now and ready, one line each, with the command that moves it on.

    The import (and its progress), then the recomputation of the profile and routine when one of
    them is missing, then the description of the persona card (the batch that waits for
    ``twin jobs approve``) when it is missing.  Numbers and commands only - never a word of a
    message.
    """
    if readiness.ready:
        return []
    queue = JobQueue(services.db, services.clock)
    run = latest_run(services.db)
    lines = [_import_line(run)]
    if PROFILE in readiness.missing or ROUTINE in readiness.missing:
        lines.append(_profile_line(queue, run))
    if PERSONA in readiness.missing:
        lines.append(_persona_line(services, queue, readiness))
    return lines


def status_lines(services: Services) -> list[str]:
    """``/状态``: ``已就绪``, or ``未就绪：缺 …`` with what stands in the way, one line each."""
    readiness = check_readiness(services)
    if readiness.ready:
        return [readiness.status_line()]
    head = f"{readiness.status_line()}（先不回复、不主动）"
    return [head, *(f"- {line}" for line in explain(services, readiness))]


def notice_text(readiness: Readiness, lines: Sequence[str]) -> str:
    """The system message (without the ``⚙️`` prefix, which the engine adds)."""
    head = NOTICE_HEAD.format(missing=readiness.label)
    return "\n".join([head, *(f"- {line}" for line in lines)])


# ------------------------------------------------------------------------- the book


class NoticeBook:
    """When each kind of notice was last delivered (``settings``: ``engine.readiness_notice``)."""

    def __init__(self, services: Services) -> None:
        self._services = services

    def _read(self) -> dict[str, str]:
        with self._services.db.session() as session:
            value = get_setting(session, NOTICE_KEY, {})
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    def last(self, key: str) -> datetime | None:
        raw = self._read().get(key)
        return datetime.fromisoformat(raw) if raw else None

    def due(self, readiness: Readiness, now: datetime, last_inbound: datetime | None) -> bool:
        """Is a notice of this kind due?  Never told; or he wrote six hours or more after it was.

        The notice answers his messages: one that comes within six hours of the last notice of
        the same kind gets none, and the hours that pass bring none by themselves.
        """
        told = self.last(readiness.key)
        if told is None:
            return True
        if now - told < NOTICE_EVERY:
            return False
        return last_inbound is not None and last_inbound - told >= NOTICE_EVERY

    def record(self, readiness: Readiness, at: datetime) -> None:
        book = self._read()
        book[readiness.key] = at.isoformat()
        with self._services.db.transaction(bump_state=False) as session:
            put_setting(
                session,
                NOTICE_KEY,
                book,
                clock=self._services.clock,
                by="engine",
                record_history=False,
            )


class ReadinessGate:
    """What the engine asks (:class:`~twin.engine.machine.ReadinessPort`): the test of D-870,
    the notice of D-871 and its book.  Every method is blocking (the engine runs it in a thread).
    """

    def __init__(self, services: Services) -> None:
        self._services = services
        self.book = NoticeBook(services)

    def check(self) -> Readiness:
        return check_readiness(self._services)

    def notice(
        self, readiness: Readiness, now: datetime, last_inbound: datetime | None
    ) -> str | None:
        """The text of the notice if one is due now (``None``: ready, or told recently)."""
        if readiness.ready or not self.book.due(readiness, now, last_inbound):
            return None
        try:
            lines = explain(self._services, readiness)
        except Exception as exc:  # the notice says what is missing even if the details fail
            log.warning("readiness_explain_failed", error=type(exc).__name__)
            lines = []
        return notice_text(readiness, lines)

    def noted(self, readiness: Readiness, at: datetime) -> None:
        """The notice was delivered at ``at``."""
        self.book.record(readiness, at)
