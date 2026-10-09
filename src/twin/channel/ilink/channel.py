"""``IlinkChannel``: the WeChat ClawBot channel (R-CH-002 .. R-CH-008).

It ties together the parts in this package: the long poller that fills a durable inbox, the
sender with its recipient, media and window checks, the login state and the persistent state
in ``channel_state``.  ``incoming()`` hands out the inbox one message at a time; a message is
removed from the inbox when the consumer asks for the next one, so a crash in between delivers
it again after the restart (at least once, never lost).

The channel can also be built without polling (``poll=False``) for commands that only send,
such as ``twin channel send-test``; those must not announce start/stop to the server or compete
with a running application for the poll cursor.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from twin.app import ComponentHealth, HealthStatus, TaskSupervisor
from twin.channel.base import (
    AuthState,
    CapabilityNotSupported,
    Channel,
    ChannelCapabilities,
    InboundMessage,
    OutboundResult,
    QuoteTarget,
    SendBypass,
    SessionState,
)
from twin.channel.binding import RecipientGuard
from twin.channel.console import AlertBanner, StderrBanner
from twin.channel.ilink.auth import AuthGuard
from twin.channel.ilink.http import TIMEOUT_QUICK_S, IlinkError, IlinkHttp
from twin.channel.ilink.inbound import InboundConverter
from twin.channel.ilink.media import CdnClient
from twin.channel.ilink.outbound import IlinkSender
from twin.channel.ilink.poller import IlinkPoller, PollOutcome, system_jitter
from twin.channel.ilink.store import IlinkStore
from twin.channel.ilink.wire import TEXT_CHUNK_LIMIT
from twin.channel.policy import (
    CompositeMediaPolicy,
    OutboundMediaPolicy,
    ProbeImageManifest,
    StickerAllowList,
)
from twin.channel.probe.summary import load_channel_probe_summary, measured_capabilities
from twin.channel.state import ChannelStateStore
from twin.channel.window import SessionWindow
from twin.clock import Clock
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.stickers.library import StickerLibraryAllowList
from twin.storage.db import Database
from twin.storage.media import MediaStore

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.channel.ilink")

NOT_SUPPORTED_QUOTE = (
    "the iLink protocol has no way to send a quote (docs/ILINK_PROTOCOL.md section 10.1)"
)


class IlinkChannel(Channel):
    """The WeChat channel for the one bound user."""

    name = "ilink_channel"

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        media: MediaStore,
        alerts: AlertSink,
        window_h: float,
        quota: int,
        media_policy: OutboundMediaPolicy | None = None,
        stickers: StickerAllowList | None = None,
        http_client: httpx.AsyncClient | None = None,
        banner: AlertBanner | None = None,
        jitter: Callable[[], float] = system_jitter,
        poll: bool = True,
        measured_window_h: float | None = None,
        measured_quota: int | None = None,
        gif_animated: bool | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._media = media
        self._alerts = alerts
        self._window_h = window_h
        self._quota = quota
        self._poll = poll
        self._jitter = jitter
        self._measured = (measured_window_h, measured_quota, gif_animated)
        self._banner = banner or StderrBanner()
        self.state = ChannelStateStore(db)
        self.store = IlinkStore(self.state, clock)
        # Only stickers from the library and the probe's own test pictures may leave (R-SAFE-006).
        self._media_policy: OutboundMediaPolicy = media_policy or CompositeMediaPolicy(
            [*([stickers] if stickers is not None else []), ProbeImageManifest(self.state)]
        )
        self._http = IlinkHttp(http_client)
        self.guard = RecipientGuard(self._bound_user_id)
        self._auth_guard = AuthGuard(self.store, alerts, self._banner)
        self._cdn = CdnClient(self._http, clock)
        self._sender = IlinkSender(
            http=self._http,
            cdn=self._cdn,
            store=self.store,
            guard=self._auth_guard,
            clock=clock,
            recipients=self.guard,
            media_policy=self._media_policy,
            window_h=window_h,
            quota=quota,
        )
        self._inbox_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._poller = IlinkPoller(
            http=self._http,
            store=self.store,
            converter=InboundConverter(self._cdn, media, self.store, clock),
            guard=self._auth_guard,
            alerts=alerts,
            clock=clock,
            jitter=jitter,
            on_inbox=self._inbox_event.set,
        )
        self._supervisor = TaskSupervisor(self.name, clock, alerts)
        self._started = False

    @property
    def media_policy(self) -> OutboundMediaPolicy:
        """What may leave as a picture: library stickers and the probe's own test pictures."""
        return self._media_policy

    @classmethod
    def from_services(
        cls,
        services: Services,
        *,
        media_policy: OutboundMediaPolicy | None = None,
        stickers: StickerAllowList | None = None,
        poll: bool = True,
        banner: AlertBanner | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> IlinkChannel:
        """Build the channel from the configuration (``channel.*`` safe window and quota).

        What the M0 probe measured (window, message count, whether a GIF moves) is read from the
        database and shown by :meth:`capabilities`; it never changes the configured thresholds.
        """
        config = services.settings.channel
        with services.db.session() as session:
            window_h, quota, gif = measured_capabilities(load_channel_probe_summary(session))
        return cls(
            db=services.db,
            clock=services.clock,
            media=services.media,
            alerts=services.alerts,
            window_h=config.proactive_window_safe_h,
            quota=config.outbound_quota_safe,
            media_policy=media_policy,
            stickers=stickers if stickers is not None else StickerLibraryAllowList(services.db),
            banner=banner,
            http_client=http_client,
            poll=poll,
            measured_window_h=window_h,
            measured_quota=quota,
            gif_animated=gif,
        )

    # ----------------------------------------------------------- lifecycle

    def _bound_user_id(self) -> str | None:
        bound = self.store.bound_user()
        return bound.user_id if bound else None

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        if self._poll:
            await self._notify("msg/notifystart")
            self._supervisor.spawn(
                "poll", lambda: self._poller.run(self._stop_event), restart_on_exit=True
            )

    async def stop(self) -> None:
        if not self._started:
            await self._http.aclose()
            return
        self._started = False
        self._stop_event.set()
        self._inbox_event.set()
        await self._supervisor.stop()
        await self._sender.aclose()
        if self._poll:
            await self._notify("msg/notifystop")
        await self._http.aclose()

    async def _notify(self, endpoint: str) -> None:
        """Tell the server the bot is online/offline; failures are only logged."""
        credentials = await asyncio.to_thread(self.store.credentials)
        if credentials is None:
            return
        try:
            await self._http.post(
                endpoint,
                {},
                base_url=credentials.api_base_url,
                token=credentials.bot_token,
                timeout_s=TIMEOUT_QUICK_S,
            )
        except IlinkError as exc:
            log.warning("lifecycle_notice_failed", endpoint=endpoint, error=str(exc))

    def health(self) -> ComponentHealth:
        supervised = self._supervisor.health()
        if supervised.status is not HealthStatus.OK or not self._poll:
            return supervised
        state = self.store.auth_record().state
        if state is AuthState.NEEDS_RELOGIN:
            return ComponentHealth(
                HealthStatus.DEGRADED, "the WeChat login expired: run `twin channel login --force`"
            )
        if state is AuthState.NOT_LOGGED_IN:
            return ComponentHealth(HealthStatus.DEGRADED, "not logged in: run `twin channel login`")
        return supervised

    async def poll_once(self) -> PollOutcome:
        """One ``getupdates`` round without the background loop (diagnostics and tests)."""
        return await self._poller.poll_once()

    # ------------------------------------------------------------- inbound

    async def incoming(self) -> AsyncIterator[InboundMessage]:
        """Inbox messages, oldest first; the consumer asking for the next one acknowledges."""
        while True:
            self._inbox_event.clear()
            pending = await asyncio.to_thread(self.store.inbox)
            if pending:
                message = pending[0]
                yield message
                await asyncio.to_thread(self.store.ack, message.id)
                continue
            if self._stop_event.is_set():
                return
            await self._inbox_event.wait()

    # ------------------------------------------------------------ outbound

    async def send_text(
        self,
        text: str,
        quote: QuoteTarget | None = None,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        if quote is not None:
            raise CapabilityNotSupported(NOT_SUPPORTED_QUOTE)
        return await self._sender.send_text(text, recipient=recipient, bypass=bypass)

    async def send_image(
        self,
        data: bytes | Path,
        mime: str,
        *,
        recipient: str | None = None,
        bypass: SendBypass | None = None,
    ) -> OutboundResult:
        return await self._sender.send_image(data, mime, recipient=recipient, bypass=bypass)

    async def send_typing(self, active: bool, *, recipient: str | None = None) -> None:
        await self._sender.send_typing(active, recipient=recipient)

    # -------------------------------------------------------- introspection

    def capabilities(self) -> ChannelCapabilities:
        window_h, quota, gif = self._measured
        return ChannelCapabilities(
            supports_quote=False,
            supports_typing=True,
            gif_animated=gif,
            proactive_window_h=window_h,
            outbound_quota=quota,
            max_text_chars=TEXT_CHUNK_LIMIT,
        )

    def window(self) -> SessionWindow:
        """The conversation window as stored right now (a snapshot, not a live object)."""
        return SessionWindow(
            window_h=self._window_h, quota=self._quota, state=self.store.window_state()
        )

    def session_state(self) -> SessionState:
        record = self.store.auth_record()
        window = self.window()
        now = self._clock.now_utc()
        return SessionState(
            auth=record.state,
            bound=self.store.bound_user() is not None,
            last_inbound_at=window.last_inbound_at,
            outbound_since_inbound=window.outbound_since_inbound,
            expired=window.expired,
            remaining_quota=window.remaining_quota(),
            window_remaining=window.window_remaining(now),
            has_context_token=self.store.context_token() is not None,
        )
