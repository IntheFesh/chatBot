"""The serving component of the application: the model's server, the tunnel, the reminders.

``twin run`` registers :class:`StyleServingComponent` (:func:`register_serving`).  It follows the
settings - at start and whenever ``settings.state_version`` changes (``twin model activate``,
``/后端``, a fall-back) - and keeps exactly what is wanted running (R-ARCH-006.2, R-SRV-002/003):

**Local model** (``style_model.mode = llamacpp_completion``).  The ``llama-server`` of the active
GGUF runs while the style or hybrid backend is asked for, while a fall-back to DeepSeek is in
force (it must be there to switch back to), and - with ``style_model.serve.warm_standby`` - while a
model that passed the release gate is active, because the last budget level hands over to it
(R-LLM-008).  A model whose last tokenizer comparison failed, or that is bound to a template this
program cannot render, is not started.  The server is the managed process of
:mod:`twin.serving.server`; once it is up the comparison of R-TRN-011.4 runs and one warm-up
request measures the first-token latency and the speed for ``/状态``.  While it loads, the client
of the engine reports "loading" instead of "down", so the replies go to DeepSeek without a
fall-back (:mod:`twin.engine.backend_select`).

**Remote model** (``vllm_completion``).  While the setting ``style.tunnel_wanted`` is on
(``twin model tunnel start``, set by the activation of an adapter) the SSH tunnel to the instance
is kept up; each time it connects, and vLLM answers behind it, the comparison and the warm-up
run.  Once a day, from ``commands.morning_hour`` on, the user gets a system message that the
instance is billed by the hour and running, with how long it has been up
(``style_model.tunnel.remind_remote``).

**Evaluation.**  The pool of :mod:`twin.serving.evaluation` is installed for the evaluation jobs of
the application's worker.

What the component knows is written to the ``serving.state`` record (:mod:`twin.serving.state`);
``/状态`` and ``twin model tunnel status`` read it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from twin.app import Application, ComponentHealth, HealthStatus, TaskSupervisor
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, TUNNEL_WANTED
from twin.engine.backend_select import STYLE_BACKENDS
from twin.engine.style_models import StyleModels
from twin.engine.style_runtime import ConfiguredStyleClient, StyleRuntime
from twin.llm.style_client import StyleHealth, StyleModelClient
from twin.ops.health import HealthCheck, HealthLevel
from twin.ops.jobobject import ProcessJob
from twin.ops.logging import get_logger
from twin.ops.state_watch import StateWatcher
from twin.serving.evaluation import EvaluationServers, install_servers, tunnel_known_hosts
from twin.serving.llamacpp import ServeError
from twin.serving.runtime import (
    LocalProgram,
    LocalServer,
    TokenizerSource,
    Warmup,
    build_local_server,
    client_for,
    compare_tokens,
    model_path,
    primary_port,
    resolve_program,
    verify_model_file,
    warm_up,
)
from twin.serving.server import (
    LlamaServerManager,
    ServerSnapshot,
    ServerState,
    ServerTimings,
)
from twin.serving.state import ServingStateStore
from twin.serving.tunnel import (
    Connector,
    TunnelManager,
    TunnelSnapshot,
    TunnelState,
    TunnelTimings,
)
from twin.training import lf_template
from twin.training.registry import ModelView, RegistryError, get_model
from twin.training.remote.connection import RemoteError, target_from_settings

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.serving.component")

COMPONENT_NAME: Final = "style_serving"
PERSIST_POLL_S: Final = 1.0
REMIND_POLL_S: Final = 60.0
VERIFY_POLL_S: Final = 20.0

Say = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class Desired:
    """What should be running now."""

    server: ModelView | None = None
    tunnel: bool = False


def decide(services: Services) -> Desired:
    """Read the registry and the settings: which server and tunnel are wanted (a blocking read)."""
    config = services.settings.style_model
    models = StyleModels(services.db, mode=config.mode)
    active = models.active()
    if active is None:
        return Desired()
    if active.versions.template_version != lf_template.TEMPLATE_VERSION:
        return Desired()
    if config.mode == "vllm_completion":
        return Desired(tunnel=bool(services.runtime.get(TUNNEL_WANTED)))
    if active.tokenizer_ok is False:
        return Desired()
    requested = services.runtime.get(BACKEND_ACTIVE)
    fallback = services.runtime.get(BACKEND_FALLBACK)
    standby = config.serve.warm_standby and active.passed_gate
    if requested in STYLE_BACKENDS or fallback is not None or standby:
        return Desired(server=get_model(services.db, active.id))
    return Desired()


def format_span(span: timedelta) -> str:
    """``span`` as "3 小时 20 分" (or "45 分钟")."""
    minutes = max(0, round(span.total_seconds() / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分" if minutes else f"{hours} 小时"
    return f"{minutes} 分钟"


def reminder_due(now: datetime, morning_hour: int, reminded_on: str | None) -> bool:
    """Once a day, and not before ``commands.morning_hour`` of the bot's local day."""
    return now.hour >= morning_hour and reminded_on != now.date().isoformat()


def reminder_text(uptime_s: float | None, tunnel_s: float | None) -> str:
    """The daily system message about the rented instance (numbers only)."""
    if uptime_s is not None:
        running = f"实例已运行 {format_span(timedelta(seconds=uptime_s))}"
    elif tunnel_s is not None:
        running = f"隧道已连上 {format_span(timedelta(seconds=tunnel_s))}"
    else:
        running = "实例在运行"
    return (
        f"远程风格模型按小时计费，实例运行中（{running}）。"
        "不用了就先发 /后端 deepseek，再在电脑上运行 twin model tunnel stop，"
        "并到 AutoDL 控制台关机。"
    )


class StyleServingComponent:
    """Runs what the settings ask for (see the module description)."""

    name = COMPONENT_NAME
    depends_on: Sequence[str] = ()

    def __init__(
        self,
        services: Services,
        *,
        style: StyleRuntime | None = None,
        say: Say | None = None,
        tokenizers: TokenizerSource | None = None,
        job: ProcessJob | None = None,
        connect: Connector | None = None,
        timings: ServerTimings | None = None,
        program: LocalProgram | None = None,
        persist_poll_s: float = PERSIST_POLL_S,
        remind_poll_s: float = REMIND_POLL_S,
        verify_poll_s: float = VERIFY_POLL_S,
    ) -> None:
        self._services = services
        self._style = style
        self._program = program
        self._persist_poll_s = persist_poll_s
        self._remind_poll_s = remind_poll_s
        self._verify_poll_s = verify_poll_s
        self._say = say
        self._tokenizers = tokenizers or TokenizerSource(services)
        self._job = job
        self._connect = connect
        self._timings = timings
        self._store = ServingStateStore(services.db, services.clock)
        self._supervisor = TaskSupervisor(self.name, services.clock, services.alerts)
        self._pool = EvaluationServers(
            services, tokenizers=self._tokenizers, job=job, connect=connect
        )
        self._lock = asyncio.Lock()
        self._dirty = asyncio.Event()
        self._uptime_s: float | None = None
        self._server: LlamaServerManager | None = None
        self._server_model: ModelView | None = None
        self._tunnel: TunnelManager | None = None
        self._tunnel_model_id: str | None = None
        self._verified: tuple[str, str] | None = None
        self._local: LocalServer | None = None
        self._warmup: Warmup | None = None
        self._unavailable: str | None = None
        self._runs = 0

    def set_say(self, say: Say) -> None:
        """Where the daily reminder is said (the engine exists after this component)."""
        self._say = say

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        install_servers(self._pool)
        self._supervisor.spawn("persist", self._persist_loop, restart_on_exit=True)
        self._supervisor.spawn("remind", self._remind_loop, restart_on_exit=True)
        self._supervisor.spawn("verify", self._verify_loop, restart_on_exit=True)
        await self.reconcile()

    async def stop(self) -> None:
        install_servers(None)
        async with self._lock:
            await self._stop_server()
            await self._stop_tunnel()
        await self._pool.aclose()
        await asyncio.to_thread(self._flush_final)
        await self._supervisor.stop()

    def health(self) -> ComponentHealth:
        base = self._supervisor.health()
        if base.status is not HealthStatus.OK:
            return base
        if self._unavailable is not None:
            return ComponentHealth(HealthStatus.DEGRADED, self._unavailable)
        if self._server is not None and self._server.state in (
            ServerState.BACKOFF,
            ServerState.BLOCKED,
        ):
            return ComponentHealth(HealthStatus.DEGRADED, f"llama-server {self._server.state}")
        if self._tunnel is not None and self._tunnel.state in (
            TunnelState.BACKOFF,
            TunnelState.BLOCKED,
        ):
            return ComponentHealth(HealthStatus.DEGRADED, f"tunnel {self._tunnel.state}")
        return base

    async def health_check(self) -> HealthCheck:
        """The check ``twin health`` shows for the process (registered with the health monitor)."""
        if self._unavailable is not None:
            return HealthCheck(
                "style_serving", HealthLevel.FAIL, self._unavailable, category="style_model_down"
            )
        server = self._server
        if server is not None:
            snapshot = server.snapshot()
            if server.state is ServerState.BLOCKED:
                blocked = server.blocked
                tokenizer = blocked is not None and blocked.reason == "tokenizer"
                return HealthCheck(
                    "style_serving",
                    HealthLevel.FAIL,
                    f"llama-server is blocked: {snapshot.detail}",
                    category="style_tokenize_mismatch" if tokenizer else "style_model_down",
                )
            if server.state is ServerState.BACKOFF:
                return HealthCheck(
                    "style_serving",
                    HealthLevel.WARN,
                    f"llama-server is restarting ({snapshot.detail})",
                    value=float(snapshot.restarts),
                )
            return HealthCheck(
                "style_serving",
                HealthLevel.OK,
                f"llama-server {server.state}, {snapshot.restarts} restart(s)",
                value=float(snapshot.restarts),
            )
        tunnel = self._tunnel
        if tunnel is not None:
            snap = tunnel.snapshot()
            if tunnel.state in (TunnelState.BLOCKED, TunnelState.BACKOFF):
                level = (
                    HealthLevel.FAIL if tunnel.state is TunnelState.BLOCKED else HealthLevel.WARN
                )
                return HealthCheck(
                    "style_serving",
                    level,
                    f"tunnel {tunnel.state}: {snap.detail}",
                    category="style_model_down" if level is HealthLevel.FAIL else None,
                )
            return HealthCheck("style_serving", HealthLevel.OK, f"tunnel {tunnel.state}")
        return HealthCheck("style_serving", HealthLevel.OK, "nothing to serve at the moment")

    # --------------------------------------------------------------- reconcile

    async def on_state_change(self, old: int, new: int) -> None:
        """A setting changed in another process: bring the running parts in line."""
        await self.reconcile()

    async def reconcile(self) -> None:
        """Start, replace or stop the server and the tunnel to match the settings."""
        async with self._lock:
            desired = await asyncio.to_thread(decide, self._services)
            await self._reconcile_server(desired.server)
            await self._reconcile_tunnel(desired.tunnel)
        self._mark_dirty()

    async def _reconcile_server(self, wanted: ModelView | None) -> None:
        current = self._server_model
        if wanted is None:
            if self._server is not None:
                log.info("server_not_wanted")
            self._unavailable = None
            await self._stop_server()
            return
        if current is not None and current.id == wanted.id and current.sha256 == wanted.sha256:
            blocked = self._server.blocked if self._server is not None else None
            if (
                self._server is not None
                and blocked is not None
                and blocked.reason == "start_failed"
            ):
                self._server.unblock()  # the program or the file may be there now
            return
        await self._stop_server()
        await self._start_server(wanted)

    async def _start_server(self, model: ModelView) -> None:
        services = self._services
        config = services.settings.style_model
        try:
            program = self._program or resolve_program(services)
            port = primary_port(config)
            path = model_path(services, model)
            await asyncio.to_thread(verify_model_file, model, path, full=False)
        except ServeError as exc:
            self._unavailable = str(exc)
            log.warning("server_unavailable", reason=str(exc)[:200])
            return
        self._unavailable = None
        local = build_local_server(
            services,
            model,
            program,
            port=port,
            tokenizers=self._tokenizers,
            job=self._job,
            timings=self._timings,
            on_change=self._server_changed,
        )
        manager = local.manager
        self._local, self._server, self._server_model = local, manager, model
        self._runs += 1
        self._hook_health(manager)
        self._supervisor.spawn(f"server-{self._runs}", manager.run)

    def _hook_health(self, manager: LlamaServerManager | None) -> None:
        """While the process loads the model the engine's client says "loading", not "down"."""
        client = self._style.client if self._style is not None else None
        if not isinstance(client, ConfiguredStyleClient):
            return
        if manager is None:
            client.health_hook = None
            return

        def loading() -> StyleHealth | None:
            # STOPPED is the moment between spawning the task and its first step; the grace period
            # of the selector (``serve.start_timeout_s``) bounds a process that never gets going
            if manager.state in (ServerState.STARTING, ServerState.STOPPED):
                return StyleHealth(False, "model is still loading", loading=True)
            return None

        client.health_hook = loading

    async def _stop_server(self) -> None:
        manager, self._server, self._server_model = self._server, None, None
        self._local = None
        self._hook_health(None)
        if manager is not None:
            await manager.stop()

    async def _reconcile_tunnel(self, wanted: bool) -> None:
        if not wanted:
            await self._stop_tunnel()
            return
        if self._tunnel is not None:
            return
        services = self._services
        config = services.settings.style_model
        try:
            target = target_from_settings(
                services.settings.autodl, services.secrets, tunnel_known_hosts(services)
            )
        except RemoteError as exc:
            self._unavailable = str(exc)
            log.warning("tunnel_unavailable", reason=str(exc)[:200])
            return
        self._unavailable = None
        probe = client_for(config, services.clock)
        tunnel = TunnelManager(
            target,
            local_port=config.tunnel.local_port,
            remote_port=config.tunnel.remote_port,
            clock=services.clock,
            timings=TunnelTimings(config.tunnel.backoff_start_s, config.tunnel.backoff_max_s),
            connect=self._connect,
            local_health=probe.health,
            on_change=self._tunnel_changed,
        )
        self._tunnel = tunnel
        self._verified = None
        self._runs += 1
        self._supervisor.spawn(f"tunnel-{self._runs}", tunnel.run)

    async def _stop_tunnel(self) -> None:
        tunnel, self._tunnel = self._tunnel, None
        self._verified = None
        self._warmup = None
        self._uptime_s = None
        if tunnel is not None:
            await tunnel.stop()

    # -------------------------------------------------------------- the record

    def _server_changed(self, snapshot: ServerSnapshot) -> None:
        self._mark_dirty()

    def _tunnel_changed(self, snapshot: TunnelSnapshot) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty.set()

    def _sections(self) -> dict[str, Any]:
        server = self._server
        model = self._server_model
        sections: dict[str, Any] = {}
        if server is not None and model is not None:
            snap = server.snapshot()
            sections["server"] = {
                "state": snap.state.value,
                "model": model.id,
                "pid": snap.pid,
                "restarts": snap.restarts,
                "since": snap.started_at.isoformat() if snap.started_at else None,
                "ready_at": snap.ready_at.isoformat() if snap.ready_at else None,
                "detail": snap.detail,
            }
            warm = self._local.warmup if self._local is not None else self._warmup
            sections["warmup"] = warm.to_json() if warm is not None else None
        else:
            sections["server"] = (
                {"state": "unavailable", "detail": self._unavailable} if self._unavailable else None
            )
            sections["warmup"] = self._warmup.to_json() if self._warmup is not None else None
        tunnel = self._tunnel
        if tunnel is not None:
            snap_t = tunnel.snapshot()
            sections["tunnel"] = {
                "state": snap_t.state.value,
                "up_since": snap_t.up_since.isoformat() if snap_t.up_since else None,
                "reconnects": snap_t.reconnects,
                "detail": snap_t.detail,
                "local_port": snap_t.local_port,
                "instance_uptime_s": self._uptime_s,
            }
        else:
            sections["tunnel"] = None
        sections["process"] = {"pid": os.getpid()}
        return sections

    def _flush_final(self) -> None:
        """The application is stopping: nothing is served any more."""
        self._store.update({"server": None, "warmup": None, "tunnel": None, "process": None})

    async def flush(self) -> None:
        """Write the record now (it is also written, at most once a second, after a change)."""
        await asyncio.to_thread(self._store.update, self._sections())

    async def _persist_loop(self) -> None:
        """Write the record when something changed (at most once a second)."""
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            await asyncio.to_thread(self._store.update, self._sections())
            await self._services.clock.sleep(self._persist_poll_s)

    # ------------------------------------------------------------ remote checks

    async def _verify_loop(self) -> None:
        """Compare the tokens and warm up each time the tunnel is connected to a working vLLM."""
        services = self._services
        while True:
            await services.clock.sleep(self._verify_poll_s)
            tunnel = self._tunnel
            if tunnel is None or tunnel.state is not TunnelState.UP:
                continue
            snap = tunnel.snapshot()
            marker = (
                snap.up_since.isoformat() if snap.up_since else "",
                str(snap.reconnects),
            )
            if self._verified == marker:
                await self._refresh_uptime(tunnel)
                continue
            client = client_for(services.settings.style_model, services.clock)
            try:
                if not (await client.health()).ok:
                    continue  # the tunnel is up but vLLM is not (yet): try again
                model = await self._active_model()
                if model is not None:
                    await self._verify_remote(model, client)
                self._warmup = await warm_up(client, services.clock)
                self._verified = marker
            finally:
                await client.aclose()
            await self._refresh_uptime(tunnel)
            self._mark_dirty()

    async def _active_model(self) -> ModelView | None:
        active = await asyncio.to_thread(
            StyleModels(self._services.db, mode=self._services.settings.style_model.mode).active
        )
        if active is None:
            return None
        try:
            return await asyncio.to_thread(get_model, self._services.db, active.id)
        except RegistryError:
            return None

    async def _verify_remote(self, model: ModelView, client: StyleModelClient) -> None:
        report = await compare_tokens(self._services, model, client, self._tokenizers)
        if not report.ok:
            log.warning("remote_tokenize_mismatch", model=model.id)

    async def _refresh_uptime(self, tunnel: TunnelManager) -> None:
        uptime = await tunnel.instance_uptime()
        if uptime is not None:
            self._uptime_s = uptime
            self._mark_dirty()

    # --------------------------------------------------------------- reminder

    async def remind_once(self) -> bool:
        """Say, once a day, that the rented instance runs by the hour (``True`` if said)."""
        services = self._services
        tunnel = self._tunnel
        if (
            self._say is None
            or tunnel is None
            or tunnel.state is not TunnelState.UP
            or not services.settings.style_model.tunnel.remind_remote
        ):
            return False
        from twin.schedule.service import time_service_for

        now = time_service_for(services).now_local()
        reminded = await asyncio.to_thread(self._store.reminded_on)
        if not reminder_due(now, services.settings.commands.morning_hour, reminded):
            return False
        today = now.date().isoformat()
        uptime = await tunnel.instance_uptime()
        snap = tunnel.snapshot()
        since = (
            (services.clock.now_utc() - snap.up_since).total_seconds() if snap.up_since else None
        )
        await self._say(reminder_text(uptime, since))
        await asyncio.to_thread(self._store.mark_reminded, today)
        return True

    async def _remind_loop(self) -> None:
        """Once a minute: the daily reminder, and a new try for what could not be started."""
        while True:
            with contextlib.suppress(Exception):
                await self.remind_once()
            if self._unavailable is not None:
                with contextlib.suppress(Exception):
                    await self.reconcile()
            await self._services.clock.sleep(self._remind_poll_s)


def register_serving(
    application: Application,
    services: Services,
    *,
    style: StyleRuntime | None,
    say: Say | None,
    watcher: StateWatcher | None = None,
    job: ProcessJob | None = None,
) -> StyleServingComponent:
    """Add the component to ``application`` and let it follow the settings."""
    component = StyleServingComponent(services, style=style, say=say, job=job)
    application.register(component)
    if watcher is not None:
        watcher.subscribe(component.on_state_change)
    return component
