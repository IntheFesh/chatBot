"""Long polling (R-CH-004; protocol document sections 5.3, 10.4, 10.5).

One iteration asks ``getupdates`` for news.  The server holds the request until something
happens; a read timeout is the normal "nothing new" answer and is not an error.  What
arrives is handled in this order, so that a crash at any point loses and repeats nothing:

1. drop what is not for us: echoes of the bot's own messages, messages from anyone but the
   bound user, and ids already seen;
2. convert what is left (downloading, decrypting and storing media);
3. in one database transaction: add the ids to the seen list, put the messages in the inbox,
   store the context token and advance the cursor;
4. only then wake the consumer.

Network and server errors are retried with a delay of 1 s doubling to 60 s with jitter, back
to the start after one success.  Error -14 ends polling until the user logs in again.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from twin.channel.base import AuthState
from twin.channel.ilink.auth import AuthGuard
from twin.channel.ilink.http import (
    TIMEOUT_DEFAULT_POLL_S,
    TIMEOUT_POLL_MARGIN_S,
    IlinkHttp,
    IlinkHttpError,
    IlinkProtocolError,
    IlinkTransportError,
)
from twin.channel.ilink.inbound import InboundConverter, message_time
from twin.channel.ilink.store import (
    BatchCommit,
    ContextToken,
    Credentials,
    IlinkStore,
    PendingBinding,
)
from twin.channel.ilink.wire import MESSAGE_BOT, WireMessage
from twin.clock import Clock
from twin.llm.redaction import redact_text
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger

log = get_logger("twin.channel.ilink.poller")

BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
MIN_POLL_INTERVAL_S = 1.0
LOGIN_WATCH_INTERVAL_S = 5.0
RECOVERY_PROBE_INTERVAL = timedelta(hours=1)
FAILURES_BEFORE_ALERT = 3
POLL_ALERT_CATEGORY = "channel.poll_failing"


class PollOutcome(StrEnum):
    MESSAGES = "messages"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    AUTH_EXPIRED = "auth_expired"
    NOT_LOGGED_IN = "not_logged_in"
    NEEDS_RELOGIN = "needs_relogin"


def backoff_delay(failures: int, jitter: float) -> float:
    """Seconds to wait after ``failures`` errors in a row: 1, 2, 4 ... capped at 60, jittered."""
    if failures < 1:
        raise ValueError("failures counts errors in a row and starts at 1")
    base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2.0 ** min(failures - 1, 16))
    return min(BACKOFF_CAP_S, base * jitter)


def system_jitter() -> float:
    """A random factor between 0.5 and 1.5."""
    return float(0.5 + random.SystemRandom().random())


class IlinkPoller:
    """Polls for the bound user's messages and keeps the inbox, cursor and window current."""

    def __init__(
        self,
        *,
        http: IlinkHttp,
        store: IlinkStore,
        converter: InboundConverter,
        guard: AuthGuard,
        alerts: AlertSink,
        clock: Clock,
        jitter: Callable[[], float] = system_jitter,
        on_inbox: Callable[[], None] | None = None,
    ) -> None:
        self._http = http
        self._store = store
        self._converter = converter
        self._guard = guard
        self._alerts = alerts
        self._clock = clock
        self._jitter = jitter
        self._on_inbox = on_inbox
        self._poll_timeout_s = TIMEOUT_DEFAULT_POLL_S
        self._last_probe: float | None = None

    # ------------------------------------------------------------ the loop

    async def run(self, stop: asyncio.Event) -> None:
        """Poll until ``stop`` is set."""
        failures = 0
        while not stop.is_set():
            started = self._clock.monotonic()
            outcome = await self.poll_once()
            elapsed = self._clock.monotonic() - started
            if outcome is PollOutcome.MESSAGES:
                failures = 0
            elif outcome is PollOutcome.ERROR:
                failures += 1
                await self._clock.sleep(backoff_delay(failures, self._jitter()))
            elif outcome in (PollOutcome.NOT_LOGGED_IN, PollOutcome.NEEDS_RELOGIN):
                await self._clock.sleep(LOGIN_WATCH_INTERVAL_S)
            elif outcome in (PollOutcome.EMPTY, PollOutcome.TIMEOUT):
                failures = 0
                if elapsed < MIN_POLL_INTERVAL_S:  # a server that answers at once: do not spin
                    await self._clock.sleep(MIN_POLL_INTERVAL_S - elapsed)
            elif outcome is PollOutcome.AUTH_EXPIRED:
                failures = 0
                await self._clock.sleep(LOGIN_WATCH_INTERVAL_S)

    # ------------------------------------------------------------- one poll

    async def poll_once(self) -> PollOutcome:
        credentials = await asyncio.to_thread(self._store.credentials)
        if credentials is None:
            return PollOutcome.NOT_LOGGED_IN
        auth = await asyncio.to_thread(self._store.auth_record)
        probing = False
        if auth.state is AuthState.NEEDS_RELOGIN:
            if not self._probe_due(auth.since):
                return PollOutcome.NEEDS_RELOGIN
            probing = True
            self._last_probe = self._clock.monotonic()
        cursor = await asyncio.to_thread(self._store.cursor)
        try:
            response = await self._http.post(
                "getupdates",
                {"get_updates_buf": cursor},
                base_url=credentials.api_base_url,
                token=credentials.bot_token,
                timeout_s=self._poll_timeout_s + TIMEOUT_POLL_MARGIN_S,
            )
        except IlinkTransportError as exc:
            if exc.timeout and exc.request_sent:  # the server had nothing to say
                if probing:  # it accepted the old token without answering -14
                    await self._guard.on_recovered()
                # a poll that waited out its time is a working connection (R-OPS-003)
                await asyncio.to_thread(self._store.record_poll_success)
                return PollOutcome.TIMEOUT
            return await self._failed("network", None, exc.reason)
        except IlinkHttpError as exc:
            return await self._failed(f"http_{exc.status}", exc.status, None)
        except IlinkProtocolError as exc:
            return await self._failed("bad_response", None, str(exc))
        if response.auth_expired:
            if not probing:
                await self._guard.on_expired(
                    source="getupdates", code=response.error_code, errmsg=response.errmsg
                )
            return PollOutcome.AUTH_EXPIRED
        code = response.error_code
        if code is not None:
            return await self._failed("api", code, response.errmsg)
        if probing:
            await self._guard.on_recovered()
        hint = response.data.get("longpolling_timeout_ms")
        if isinstance(hint, int | float) and not isinstance(hint, bool) and hint > 0:
            self._poll_timeout_s = float(hint) / 1000.0
        return await self._handle(response.data, credentials, cursor)

    def _probe_due(self, since: datetime | None) -> bool:
        if self._last_probe is None:
            if since is None:
                return True
            return self._clock.now_utc() - since >= RECOVERY_PROBE_INTERVAL
        return self._clock.monotonic() - self._last_probe >= RECOVERY_PROBE_INTERVAL.total_seconds()

    async def _failed(self, kind: str, code: int | None, detail: str | None) -> PollOutcome:
        clean = redact_text(detail)[:200] if detail else None
        count = await asyncio.to_thread(self._store.record_poll_failure, kind, code, clean)
        log.warning("poll_failed", kind=kind, code=code, in_a_row=count)
        if count == FAILURES_BEFORE_ALERT:
            await asyncio.to_thread(
                self._alerts.raise_alert,
                POLL_ALERT_CATEGORY,
                "WeChat polling keeps failing; check the network connection",
                severity="warning",
                detail={"kind": kind, "code": code},
                dedup_key=POLL_ALERT_CATEGORY,
            )
        return PollOutcome.ERROR

    # ---------------------------------------------------------- the batch

    async def _handle(
        self, data: dict[str, Any], credentials: Credentials, cursor: str
    ) -> PollOutcome:
        raw_messages = data.get("msgs")
        messages = raw_messages if isinstance(raw_messages, list) else []
        new_cursor = data.get("get_updates_buf")
        batch = BatchCommit(
            new_cursor=new_cursor
            if isinstance(new_cursor, str) and new_cursor and new_cursor != cursor
            else None
        )
        if messages:
            await self._collect(messages, credentials, batch)
        await self._commit(batch, bool(messages))
        return PollOutcome.MESSAGES if messages else PollOutcome.EMPTY

    async def _commit(self, batch: BatchCommit, has_messages: bool) -> None:
        """Steps 3 and 4: commit the batch, then wake the consumer - never one without the other.

        The commit runs on a worker thread and lands even if the task that waits for it is
        cancelled (the wake-up of the machine restarts the poll, ``IlinkChannel.reconnect``); the
        wake-up call after it would then never run, and a message in the inbox that nobody is told
        about waits for the next message or the next start (D-625).  So the two run as one unit
        that a cancellation lets finish before it passes on.
        """
        unit = asyncio.ensure_future(self._commit_and_wake(batch, has_messages))
        try:
            await asyncio.shield(unit)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await unit
            raise

    async def _commit_and_wake(self, batch: BatchCommit, has_messages: bool) -> None:
        if has_messages or batch.new_cursor:
            await asyncio.to_thread(self._store.commit_batch, batch)
        await asyncio.to_thread(self._store.record_poll_success)
        if batch.deliver and self._on_inbox is not None:
            self._on_inbox()

    async def _collect(
        self, messages: list[Any], credentials: Credentials, batch: BatchCommit
    ) -> None:
        seen = set(await asyncio.to_thread(self._store.seen_ids))
        bound = await asyncio.to_thread(self._store.bound_user)
        now = self._clock.now_utc()
        failures: Counter[str] = batch.failures
        for raw in messages:
            try:
                wire = WireMessage.model_validate(raw)
            except ValueError:
                failures["invalid_message"] += 1
                continue
            message_id = wire.dedup_id()
            if message_id in seen:
                continue
            seen.add(message_id)
            batch.seen_ids.append(message_id)
            if wire.message_type == MESSAGE_BOT:
                failures["bot_echo_skipped"] += 1
                continue
            sender = wire.from_user_id or ""
            if bound is None:
                self._note_candidate(batch, wire, sender, credentials, now)
                continue
            if sender != bound.user_id:
                failures["other_sender_dropped"] += 1
                log.info("dropped_message_from_other_sender")
                continue
            try:
                converted = await self._converter.convert(wire, message_id)
            except Exception as exc:  # one bad message must not stop the batch
                failures["convert_error"] += 1
                log.error("convert_failed", error_type=type(exc).__name__)
                self._note_inbound(batch, wire, now)
                continue
            batch.deliver.extend(converted.messages)
            batch.item_types.extend(converted.item_types)
            failures.update(converted.failures)
            batch.quote_entries.extend(converted.quote_entries)
            self._note_inbound(batch, wire, now)

    def _note_inbound(self, batch: BatchCommit, wire: WireMessage, now: datetime) -> None:
        """Any message from the bound user renews the context token and restarts the window."""
        if wire.context_token:
            batch.context_token = ContextToken(wire.context_token, now)
        moment = min(message_time(wire, now), now)
        if batch.last_inbound_at is None or moment > batch.last_inbound_at:
            batch.last_inbound_at = moment

    def _note_candidate(
        self,
        batch: BatchCommit,
        wire: WireMessage,
        sender: str,
        credentials: Credentials,
        now: datetime,
    ) -> None:
        """Nobody is bound yet: remember who wrote first, without processing the message."""
        batch.failures["unbound_message_not_processed"] += 1
        if batch.candidate is not None or not sender:
            return
        expected = credentials.ilink_user_id
        batch.candidate = PendingBinding(
            user_id=sender,
            seen_at=now,
            context_token=wire.context_token,
            matches_expected=None if not expected else sender == expected,
        )
