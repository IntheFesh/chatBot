"""``/状态``: one message that says how the bot is doing (R-CMD-002, R-SRV-003, R-ENG-002).

The report is a list of sections, each read from its own source.  A source that does not exist
yet - the proactive scheduler of round 10 - is not invented:
its line says it is not enabled or that there is nothing (``暂无``).  A source that raises costs
its own line only (``暂无``); the rest of the report is still sent, and the failure is logged by
type.

The sections, in order: time zone and local time; what she is doing and until when; the backend
and the thinking mode; proactive messages; the platform window and the messages that may still
be sent; the cost of today with the cache hit rate; the budget level; the style model; the last
three alerts; reminders (training data left on a rented machine, the retraining reminder).
Rounds that add something to show (the suggested quiet window of step 4, say) put a callable in
:attr:`StatusSources.extra`; it may be a plain function or a coroutine function.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select

from twin.channel.base import SessionState
from twin.clock import Clock
from twin.commands import texts
from twin.config.runtime import SHOW_THINKING, THINKING_CHAT, RuntimeSettings
from twin.config.settings import Settings
from twin.engine.backend_select import STYLE_BACKENDS, BackendSelector
from twin.llm.budget import BudgetManager
from twin.llm.ledger import LedgerStore
from twin.ops.logging import get_logger
from twin.schedule.time_service import PlanUnavailableError, TimeService
from twin.serving.state import ServingStateStore
from twin.storage.db import Database
from twin.storage.models import Alert
from twin.training.retrain import retrain_status_text
from twin.training.runs import uncleaned_runs

log = get_logger("twin.commands.status")

ALERTS_SHOWN = 3
WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass(frozen=True)
class ProactiveStatus:
    """What the proactive scheduler (round 10) reports: today's range and how many went out."""

    low: int
    high: int
    enabled: bool
    sent_today: int


@dataclass
class StatusSources:
    """Where ``/状态`` reads from (see the module description); ``None`` means "not there yet"."""

    runtime: RuntimeSettings
    settings: Settings
    time: TimeService
    selector: BackendSelector
    db: Database
    clock: Clock
    ledger: LedgerStore | None = None
    budget: BudgetManager | None = None
    session_state: Callable[[], SessionState | None] | None = None
    proactive: Callable[[], ProactiveStatus | None] | None = None
    retrain: Callable[[], str | None] | None = None
    uncleaned: Callable[[], int] | None = None
    extra: Sequence[Callable[[], str | Awaitable[str | None] | None]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.uncleaned is None:
            db = self.db
            self.uncleaned = lambda: len(uncleaned_runs(db))
        if self.retrain is None:
            db, threshold = self.db, self.settings.training.retrain_new_ratio
            self.retrain = lambda: retrain_status_text(db, threshold)


def format_span(span: timedelta) -> str:
    """``span`` as "2 小时 5 分" (minutes at least)."""
    minutes = max(0, round(span.total_seconds() / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分" if minutes else f"{hours} 小时"
    return f"{minutes} 分钟"


Section = Callable[[], Awaitable[list[str]]]


class StatusReport:
    """Builds the ``/状态`` text from the sources (see the module description)."""

    def __init__(self, sources: StatusSources) -> None:
        self._s = sources

    async def render(self) -> str:
        sections: list[tuple[str, Section]] = [
            ("时区", self._time),
            ("她此刻", self._her),
            ("后端", self._backend),
            ("思考模式", self._thinking),
            ("主动消息", self._proactive),
            ("平台窗口", self._window),
            ("今日费用", self._cost),
            ("预算级别", self._budget),
            ("风格模型", self._style),
            ("最近告警", self._alerts),
            ("提醒", self._reminders),
        ]
        lines = [texts.STATUS_HEADER]
        for label, section in sections:
            try:
                lines.extend(await section())
            except Exception as exc:  # one broken source costs its own line, not the report
                log.warning("status_section_failed", section=label, error=type(exc).__name__)
                lines.append(f"{label}：{texts.STATUS_NONE}")
        for extra in self._s.extra:
            try:
                found = extra()
                line = await found if inspect.isawaitable(found) else found
            except Exception as exc:
                log.warning("status_section_failed", section="extra", error=type(exc).__name__)
                continue
            if line:
                lines.append(line)
        return "\n".join(lines)

    # ------------------------------------------------------------------------- sections

    async def _time(self) -> list[str]:
        zone = self._s.time.bot_timezone()
        now = self._s.time.now_local()
        local = f"{now:%Y-%m-%d} {WEEKDAYS[now.weekday()]} {now:%H:%M}"
        return [texts.STATUS_TIME.format(zone=zone.key, local=local)]

    async def _her(self) -> list[str]:
        try:
            state = self._s.time.her_state()
        except PlanUnavailableError:
            return [f"她此刻：{texts.STATUS_NONE}（今天还没有日程）"]
        zone = self._s.time.bot_timezone()
        until = state.until.astimezone(zone)
        left = state.until - self._s.time.now_utc()
        return [
            texts.STATUS_HER.format(
                state=texts.STATUS_HER_STATES.get(str(state.kind), str(state.kind)),
                until=f"{until:%H:%M}",
                left=format_span(left),
            )
        ]

    async def _backend(self) -> list[str]:
        status = await self._s.selector.status()
        fallback = status.fallback
        if status.requested in STYLE_BACKENDS and fallback is not None:
            return [
                texts.STATUS_BACKEND_FALLBACK.format(
                    requested=status.requested, reason=fallback.get("reason", "")
                )
            ]
        budget = self._s.budget
        if (
            status.requested not in STYLE_BACKENDS
            and budget is not None
            and budget.limits().prefer_style_backend
        ):
            return [texts.STATUS_BACKEND_BUDGET]
        return [texts.STATUS_BACKEND.format(backend=status.requested)]

    async def _thinking(self) -> list[str]:
        mode = texts.THINK_MODES[str(self._s.runtime.get(THINKING_CHAT))]
        show = "开" if self._s.runtime.get(SHOW_THINKING) else "关"
        return [texts.STATUS_THINKING.format(mode=mode, show=show)]

    async def _proactive(self) -> list[str]:
        found = self._s.proactive() if self._s.proactive is not None else None
        if found is None:
            return [texts.STATUS_PROACTIVE_OFF]
        return [
            texts.STATUS_PROACTIVE.format(
                low=found.low,
                high=found.high,
                on="开" if found.enabled else "关",
                sent=found.sent_today,
            )
        ]

    async def _window(self) -> list[str]:
        state = self._s.session_state() if self._s.session_state is not None else None
        if state is None:
            return [f"平台窗口：{texts.STATUS_NONE}"]
        if state.expired:
            return [texts.STATUS_WINDOW_EXPIRED]
        left = state.window_remaining
        if left is None or left <= timedelta(0):
            return [f"平台窗口：{texts.STATUS_NONE}（还没有收到你的消息，或窗口已过）"]
        return [
            texts.STATUS_WINDOW.format(
                left=format_span(left),
                quota=state.remaining_quota,
                proactive=state.remaining_quota,
            )
        ]

    async def _cost(self) -> list[str]:
        ledger, budget = self._s.ledger, self._s.budget
        day = self._s.time.local_date()
        if budget is not None:
            status = budget.status()
            spent, limit = status.daily_spent, status.daily_budget
        elif ledger is not None:
            spent, limit = ledger.spent_on_day(day), self._s.settings.budget.daily_usd
        else:
            return [f"今日费用：{texts.STATUS_NONE}"]
        if ledger is None:
            return [f"今日费用：${spent:.4f} / ${limit:.2f}"]
        start, end = self._s.time.day_bounds_utc(day)
        totals = ledger.totals(start, end)
        if totals.calls == 0:
            return [texts.STATUS_COST_NO_CALLS.format(spent=spent, budget=limit)]
        cache = f"{totals.cache_hit_ratio:.0%}"
        return [texts.STATUS_COST.format(spent=spent, budget=limit, cache=cache)]

    async def _budget(self) -> list[str]:
        if self._s.budget is None:
            return [f"预算级别：{texts.STATUS_NONE}"]
        level = self._s.budget.current_level()
        meaning = texts.STATUS_BUDGET_LEVELS[min(level, len(texts.STATUS_BUDGET_LEVELS) - 1)]
        return [texts.STATUS_BUDGET.format(level=level, meaning=meaning)]

    async def _style(self) -> list[str]:
        status = await self._s.selector.status()
        if status.registered == 0:
            return [texts.STATUS_STYLE_NONE]
        model = status.model
        if model is None:
            return [texts.STATUS_STYLE_NOT_ACTIVE.format(count=status.registered)]
        probe = status.probe
        health = (
            texts.STATUS_HEALTH_OK
            if probe is not None and probe.usable
            else texts.STATUS_HEALTH_BAD.format(detail=probe.detail if probe else texts.STATUS_NONE)
        )
        gate = texts.STATUS_GATE_PASSED if model.passed_gate else texts.STATUS_GATE_FORCED
        lines = [texts.STATUS_STYLE_ACTIVE.format(label=model.label, gate=gate, health=health)]
        lines.extend(await self._serving())
        return lines

    async def _serving(self) -> list[str]:
        """What the application knows about the model's server or tunnel (round 14)."""
        record = await asyncio.to_thread(ServingStateStore(self._s.db, self._s.clock).read)
        lines: list[str] = []
        server = record.get("server")
        if isinstance(server, dict) and server.get("state") != "ready":
            state = texts.STATUS_SERVE_STATES.get(
                str(server.get("state")), str(server.get("state"))
            )
            detail = f"（{server['detail']}）" if server.get("detail") else ""
            count = int(server.get("restarts") or 0)
            restarts = texts.STATUS_SERVE_RESTARTS.format(count=count) if count else ""
            lines.append(texts.STATUS_SERVE.format(state=state, detail=detail, restarts=restarts))
        warmup = record.get("warmup")
        if isinstance(warmup, dict) and warmup.get("first_token_ms") is not None:
            speed = warmup.get("tokens_per_s")
            zone = self._s.time.bot_timezone()
            when = datetime.fromisoformat(str(warmup["at"])).astimezone(zone)
            lines.append(
                texts.STATUS_SPEED.format(
                    first=warmup["first_token_ms"],
                    tps=f"{speed:.1f}" if isinstance(speed, int | float) else "?",
                    when=f"{when:%m-%d %H:%M}",
                )
            )
        if self._s.settings.style_model.mode == "vllm_completion":
            lines.extend(self._remote_lines(record.get("tunnel")))
        return lines

    def _remote_lines(self, tunnel: object) -> list[str]:
        """The tunnel and the reminder that the rented instance is billed by the hour."""
        if not isinstance(tunnel, dict) or tunnel.get("state") != "up":
            lines = [texts.STATUS_REMOTE_HOURLY]
            if isinstance(tunnel, dict):
                state = texts.STATUS_TUNNEL_STATES.get(str(tunnel.get("state")), "?")
                detail = f"（{tunnel['detail']}）" if tunnel.get("detail") else ""
                count = int(tunnel.get("reconnects") or 0)
                lines.insert(0, texts.STATUS_TUNNEL.format(state=state, detail=detail, count=count))
            return lines
        uptime = tunnel.get("instance_uptime_s")
        if isinstance(uptime, int | float):
            span = timedelta(seconds=float(uptime))
        else:
            up_since = datetime.fromisoformat(str(tunnel["up_since"]))
            span = self._s.clock.now_utc() - up_since
        return [
            texts.STATUS_TUNNEL.format(
                state=texts.STATUS_TUNNEL_STATES["up"],
                detail="",
                count=int(tunnel.get("reconnects") or 0),
            ),
            texts.STATUS_REMOTE_RUNNING.format(span=format_span(span)),
        ]

    async def _alerts(self) -> list[str]:
        with self._s.db.session() as session:
            rows = list(
                session.scalars(
                    select(Alert)
                    .order_by(Alert.created_at.desc(), Alert.id.desc())
                    .limit(ALERTS_SHOWN)
                )
            )
            found = [(row.created_at, row.title) for row in rows]
        if not found:
            return [texts.STATUS_NO_ALERTS]
        zone = self._s.time.bot_timezone()
        lines = [texts.STATUS_ALERTS_HEADER]
        for created, title in found:
            when = created.astimezone(zone)
            lines.append(texts.STATUS_ALERT.format(when=f"{when:%m-%d %H:%M}", title=title))
        return lines

    async def _reminders(self) -> list[str]:
        lines: list[str] = []
        count = self._s.uncleaned() if self._s.uncleaned is not None else 0
        if count:
            lines.append(texts.STATUS_UNCLEANED.format(count=count))
        retrain = self._s.retrain() if self._s.retrain is not None else None
        lines.append(texts.STATUS_RETRAIN.format(text=retrain or texts.STATUS_NONE))
        return lines
