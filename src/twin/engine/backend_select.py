"""Which backend answers: the setting, the style model's health and the way back (R-SRV-004).

The user's choice is the runtime setting ``backend.active`` (``deepseek``, ``style`` or ``hybrid``;
``/后端`` writes it).  :class:`BackendSelector` turns it into the backend of the next reply:

* ``deepseek`` is always available.  Only when the budget has reached its last level (R-LLM-008)
  and an activated model that **passed the release gate** is healthy does the selector pick
  ``style`` instead, because the cheap local model is the way to go on without spending;
* ``style`` / ``hybrid`` need an active model in ``model_registry`` whose template this program
  can render, and a server that answers its health check.  Otherwise - or when the server fails
  while answering, or when its output is judged a hard violation ``backend.fallback_violations``
  times in a row (3) - the selector **falls back** to DeepSeek: it writes the setting
  ``backend.fallback`` (why and since when), raises the alert ``style_fallback`` and the replies
  go to DeepSeek.  The user's choice, ``backend.active``, stays as it was;
* once the model has been healthy for ``backend.recover_after_min`` minutes (10) without a break,
  the selector **switches back** by itself, clears ``backend.fallback`` and raises the notice
  ``style_recovered``.

Every change is on record: the settings keep a history of each write (who, when, old value, new
value - ``by`` says ``auto`` for what the selector did and ``command`` for ``/后端``), and the
selector logs one audit line per switch with the closed reason code, never a text.

How the engine uses it (round 09 step 3 wires it)::

    choice = await selector.choose()                  # before each reply
    draft = await pipeline.run(replace(context, backend=choice.name), data)
    selector.record(choice.name, draft)               # after it: failures and violations
    ...
    await selector.tick()                             # also every ``health_check_s`` (the monitor)

:meth:`BackendSelector.available` is the :class:`~twin.llm.budget.StyleBackendStatus` of the budget
manager (round 01's interface, :meth:`BudgetManager.set_style_status`): true for a gate-passed,
healthy model.  A model that was activated with ``--force`` does not count there (R-SRV-005).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from twin.app import ComponentHealth, HealthStatus, TaskSupervisor
from twin.clock import Clock
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, RuntimeSettings
from twin.config.settings import BackendConfig
from twin.engine.style_models import ActiveStyleModel, StyleModels
from twin.engine.types import ReplyDraft
from twin.llm.budget import BudgetLimits
from twin.llm.style_client import StyleHealth, StyleModelClient
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.training import lf_template

log = get_logger("twin.engine.backend_select")
audit = get_logger("twin.backend.audit")

STYLE_BACKENDS = ("style", "hybrid")
FALLBACK_CATEGORY = "style_fallback"
RECOVERED_CATEGORY = "style_recovered"
STYLE_ERROR_TYPES = frozenset({"StyleModelError"})
PRIMARY_ATTEMPTS = 2  # the pipeline asks the chosen backend twice before it turns to DeepSeek
STALE_AFTER_CHECKS = 3  # a health answer older than this many check intervals is not trusted
MIN_STALE_S = 60.0

# the closed codes of a probe and of a refused switch
NO_MODEL = "no_model"
TEMPLATE = "template"
UNHEALTHY = "unhealthy"
NOT_REGISTERED = "not_registered"
NOT_ACTIVE = "not_active"
GATE_NOT_PASSED = "gate_not_passed"


@dataclass(frozen=True)
class Probe:
    """The answer to "may the style model be used now?" (a registry look and a health call)."""

    usable: bool
    code: str  # "ok" or one of the codes above
    detail: str
    at: datetime
    model: ActiveStyleModel | None = None
    latency_ms: int = 0


@dataclass(frozen=True)
class BackendChoice:
    """The backend of the next reply, and why it is not the one that was asked for."""

    name: str
    requested: str
    reason: str | None = None

    @property
    def fell_back(self) -> bool:
        return self.name != self.requested


@dataclass(frozen=True)
class SwitchCheck:
    """Why ``/后端 style|hybrid`` is refused: a closed ``code`` and a short ``detail``."""

    code: str
    detail: str = ""


@dataclass(frozen=True)
class StyleStatus:
    """What ``/状态`` shows about the style model."""

    registered: int
    model: ActiveStyleModel | None
    probe: Probe | None
    fallback: Mapping[str, Any] | None
    requested: str
    effective: str


class BackendSelector:
    """Chooses the backend and runs the fallback and the way back (see the module description)."""

    def __init__(
        self,
        *,
        runtime: RuntimeSettings,
        models: StyleModels,
        client: StyleModelClient,
        config: BackendConfig,
        clock: Clock,
        alerts: AlertSink,
        limits: Callable[[], BudgetLimits] | None = None,
    ) -> None:
        self._runtime = runtime
        self._models = models
        self._client = client
        self._config = config
        self._clock = clock
        self._alerts = alerts
        self._limits = limits
        self._probe: Probe | None = None
        self._probed_at = 0.0
        self._healthy_since: float | None = None
        self._violations = 0
        self._transition = asyncio.Lock()

    # ------------------------------------------------------------------ the settings

    def requested(self) -> str:
        """What the user asked for (``backend.active``)."""
        return str(self._runtime.get(BACKEND_ACTIVE))

    def fallback(self) -> Mapping[str, Any] | None:
        """Why the style model is not in use (``backend.fallback``), ``None`` when it is."""
        return self._runtime.get(BACKEND_FALLBACK)

    # ------------------------------------------------------------------------ probing

    async def probe(self, *, force: bool = False) -> Probe:
        """Look at the registry and the server (cached for ``health_check_s`` unless ``force``)."""
        now = self._clock.monotonic()
        cached = self._probe
        if cached is not None and not force and now - self._probed_at < self._config.health_check_s:
            return cached
        found = await self._check()
        self._probe, self._probed_at = found, self._clock.monotonic()
        return found

    async def _check(self) -> Probe:
        now = self._clock.now_utc()
        model = await asyncio.to_thread(self._models.active)
        if model is None:
            return Probe(False, NO_MODEL, "no style model is active", now)
        if model.versions.template_version != lf_template.TEMPLATE_VERSION:
            detail = f"the model is bound to {model.versions.template_version}"
            return Probe(False, TEMPLATE, detail, now, model)
        try:
            health: StyleHealth = await self._client.health()
        except Exception as exc:  # a health check must never raise; if it does, the model is down
            return Probe(False, UNHEALTHY, type(exc).__name__, now, model)
        code = "ok" if health.ok else UNHEALTHY
        return Probe(health.ok, code, health.detail, now, model, health.latency_ms)

    def available(self) -> bool:
        """:class:`~twin.llm.budget.StyleBackendStatus`: a gate-passed model that is healthy now."""
        probe = self._probe
        if probe is None or not probe.usable or probe.model is None:
            return False
        stale = max(MIN_STALE_S, STALE_AFTER_CHECKS * self._config.health_check_s)
        if self._clock.monotonic() - self._probed_at > stale:
            return False
        return probe.model.passed_gate and self.fallback() is None

    # ------------------------------------------------------------------- transitions

    async def tick(self) -> None:
        """One look at the style model; enters the fallback or switches back as needed."""
        probe = await self.probe()
        async with self._transition:
            requested = self.requested()
            fallback = self.fallback()
            by_budget = fallback is not None and bool(fallback.get("by_budget"))
            if requested not in STYLE_BACKENDS and not by_budget:
                if fallback is not None:
                    self._clear("deepseek_chosen", notify=False)
                return
            if fallback is None:
                if not probe.usable:
                    self._enter_fallback(requested, probe.code, probe.detail)
                return
            if not probe.usable:
                self._healthy_since = None
                return
            now = self._clock.monotonic()
            if self._healthy_since is None:
                self._healthy_since = now
            if now - self._healthy_since >= self._config.recover_after_min * 60:
                self._clear("recovered", notify=True)

    def _enter_fallback(self, backend: str, code: str, detail: str) -> None:
        """Stop using the style model: ``backend`` is the one that was in use or asked for."""
        self._healthy_since = None
        self._violations = 0
        by_budget = self.requested() not in STYLE_BACKENDS  # the budget, not the user, chose it
        requested = backend if backend in STYLE_BACKENDS else "style"
        record = {
            "requested": requested,
            "reason": code,
            "since": self._clock.now_utc().isoformat(),
            "by_budget": by_budget,
        }
        self._runtime.set(BACKEND_FALLBACK, record, by="auto")
        audit.warning("backend_fallback", requested=requested, reason=code, by_budget=by_budget)
        self._alerts.raise_alert(
            FALLBACK_CATEGORY,
            f"The style model cannot be used ({code}); replies go to the DeepSeek backend",
            severity="warning",
            detail={"requested": requested, "reason": code, "detail": detail[:200]},
            dedup_key=FALLBACK_CATEGORY,
        )

    def _clear(self, reason: str, *, notify: bool) -> None:
        previous = self.fallback()
        self._runtime.set(BACKEND_FALLBACK, None, by="auto" if notify else "command")
        self._healthy_since = None
        self._violations = 0
        audit.info("backend_restored", reason=reason)
        if notify and previous is not None:
            self._alerts.raise_alert(
                RECOVERED_CATEGORY,
                "The style model is healthy again; replies are back on the "
                f"{previous.get('requested', 'style')} backend",
                severity="info",
                detail={"requested": previous.get("requested"), "reason": reason},
                dedup_key=RECOVERED_CATEGORY,
            )

    def note_user_choice(self) -> None:
        """``/后端`` changed ``backend.active``: a fallback that was in force no longer applies."""
        if self.fallback() is not None:
            self._clear("user_choice", notify=False)
        self._violations = 0

    # --------------------------------------------------------------------- choosing

    async def choose(self) -> BackendChoice:
        """The backend of the next reply (looks at the style model first, if it is due)."""
        await self.tick()
        requested = self.requested()
        if requested not in STYLE_BACKENDS:
            if self._limits is not None and self._limits().prefer_style_backend:
                return BackendChoice("style", requested, "budget")
            return BackendChoice(requested, requested)
        fallback = self.fallback()
        if fallback is not None:
            return BackendChoice("deepseek", requested, f"fallback:{fallback.get('reason')}")
        return BackendChoice(requested, requested)

    # ------------------------------------------------------------------- the outcome

    def record(self, backend: str, draft: ReplyDraft) -> None:
        """What became of a reply that was asked of ``backend`` (see the module description).

        A failure of the style server while answering means it is down: fall back at once.
        Otherwise the hard violations of the style model's outputs are counted - the two tries the
        pipeline gives the chosen backend - and reset by a reply the style model got right.
        """
        if backend not in STYLE_BACKENDS or self.fallback() is not None:
            return
        failed = any(
            action.step == "backend_error" and action.detail in STYLE_ERROR_TYPES
            for action in draft.actions
        )
        if failed:
            self._enter_fallback(backend, "error", "the style model failed while answering")
            return
        succeeded = draft.usable and draft.backend in STYLE_BACKENDS
        attempts = draft.meta.get("attempt_violations") or []
        if succeeded:
            self._violations = 0
            return
        self._violations += min(len(attempts), PRIMARY_ATTEMPTS)
        if self._violations >= self._config.fallback_violations:
            self._enter_fallback(backend, "violations", f"{self._violations} violations in a row")

    # ------------------------------------------------------------- the command's side

    async def verify_switch(self, name: str) -> SwitchCheck | None:
        """May ``backend.active`` be set to ``name`` from the chat?  ``None``: yes.

        ``deepseek`` always.  ``style`` and ``hybrid`` need a registered model, an active one, one
        that passed the release gate (forcing an unqualified model is done on the computer with
        ``twin model activate --force``) and a server that is healthy right now.
        """
        if name not in STYLE_BACKENDS:
            return None
        if await asyncio.to_thread(self._models.registered) == 0:
            return SwitchCheck(NOT_REGISTERED, "no style model is registered")
        model = await asyncio.to_thread(self._models.active)
        if model is None:
            return SwitchCheck(NOT_ACTIVE, "no style model is active")
        if not model.passed_gate:
            return SwitchCheck(GATE_NOT_PASSED, model.label)
        probe = await self.probe(force=True)
        if not probe.usable:
            return SwitchCheck(probe.code, probe.detail)
        return None

    async def status(self) -> StyleStatus:
        """The numbers of ``/状态`` (looks at the server if the last look is old)."""
        probe = await self.probe()
        registered = await asyncio.to_thread(self._models.registered)
        model = await asyncio.to_thread(
            self._models.active
        )  # the registry fresh, the health cached
        requested = self.requested()
        fallback = self.fallback()
        effective = "deepseek" if requested in STYLE_BACKENDS and fallback else requested
        return StyleStatus(registered, model, probe, fallback, requested, effective)


class BackendMonitorComponent:
    """Looks at the style model every ``backend.health_check_s`` (a component of the application).

    Without it the selector looks when a reply is due; with it a model that dies is noticed - and
    one that recovers is switched back to - while nobody is talking.
    """

    name = "backend_monitor"
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        selector: BackendSelector,
        clock: Clock,
        alerts: AlertSink | None,
        *,
        interval_s: float,
    ) -> None:
        self._selector = selector
        self._clock = clock
        self._interval_s = interval_s
        self._supervisor = TaskSupervisor(self.name, clock, alerts)

    async def _loop(self) -> None:
        while True:
            await self._selector.tick()
            await self._clock.sleep(self._interval_s)

    async def start(self) -> None:
        self._supervisor.spawn("watch", self._loop, restart_on_exit=True)

    async def stop(self) -> None:
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        base = self._supervisor.health()
        if base.status is not HealthStatus.OK:
            return base
        fallback = self._selector.fallback()
        if fallback is not None:
            return ComponentHealth(
                HealthStatus.DEGRADED, f"style model in fallback ({fallback.get('reason')})"
            )
        return base
