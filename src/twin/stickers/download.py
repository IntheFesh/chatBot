"""Downloading stickers whose files the export does not contain (R-IMP-008).

Every ``stickers`` row that is ``pending`` and has an URL is fetched with ``httpx``:

* at most ``ingest.sticker_download.concurrency`` (4) downloads at a time and at most
  ``per_second`` (4) requests started per second, shared by all workers;
* a failed attempt (timeout, network error, HTTP 408/429/5xx) is retried
  ``retries`` (3) times with exponential back-off; other HTTP errors are final;
* the picture type comes from the file header (GIF, PNG, JPEG, WebP), not from the URL or
  the server's ``Content-Type``; its MD5 must equal ``emojiMd5`` - a different MD5 is kept
  and marked ``md5_mismatch``;
* a sticker that cannot be fetched becomes ``unavailable`` with a short reason (the import is
  never blocked); ``--retry-failed`` tries those again.

The state of every sticker is saved as soon as its download ends, so an interrupted job
continues with the stickers that are still ``pending``.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select, update

from twin.clock import Clock
from twin.config.settings import StickerDownloadConfig
from twin.ops.jobs import JobContext, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.services import Services
from twin.stickers.library import MAX_STICKER_BYTES, store_sticker_file
from twin.storage.chat_models import Sticker

log = get_logger("twin.stickers.download")

STICKER_JOB = "sticker_download"
BACKOFF_START_S = 1.0
BACKOFF_FACTOR = 2.0
PAGE = 64
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class DownloadError(Exception):
    """One download attempt failed; ``reason`` is the short code stored on the sticker."""

    def __init__(self, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class RateLimiter:
    """Spaces the starts of requests at least ``1 / per_second`` apart (shared by workers)."""

    def __init__(self, per_second: float, clock: Clock) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self._interval = 1.0 / per_second
        self._clock = clock
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
            delay = start - now
        if delay > 0:
            await self._clock.sleep(delay)


@dataclass
class DownloadStats:
    attempted: int = 0
    available: int = 0
    md5_mismatch: int = 0
    unavailable: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    stopped: bool = False

    def summary(self) -> str:
        parts = [
            f"attempted {self.attempted}",
            f"available {self.available}",
            f"md5 mismatch {self.md5_mismatch}",
            f"unavailable {self.unavailable}",
        ]
        if self.reasons:
            parts.append(
                "reasons " + ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
            )
        return "; ".join(parts)


def _check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise DownloadError("bad_url", retryable=False)


class StickerDownloader:
    """Downloads the pending stickers of a database."""

    def __init__(
        self,
        services: Services,
        *,
        config: StickerDownloadConfig | None = None,
        clock: Clock | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._services = services
        self._config = config or services.settings.ingest.sticker_download
        self._clock = clock or services.clock
        self._http = http
        self._limiter = RateLimiter(self._config.per_second, self._clock)

    # -------------------------------------------------------------- one request

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> bytes:
        _check_url(url)
        delay = BACKOFF_START_S
        last = DownloadError("no_attempt", retryable=False)
        for attempt in range(self._config.retries + 1):
            if attempt:
                await self._clock.sleep(delay)
                delay *= BACKOFF_FACTOR
            await self._limiter.wait()
            try:
                return await self._once(client, url)
            except DownloadError as exc:
                last = exc
                if not exc.retryable:
                    break
        raise last

    async def _once(self, client: httpx.AsyncClient, url: str) -> bytes:
        try:
            async with client.stream("GET", url, timeout=self._config.timeout_s) as response:
                status = response.status_code
                if status in RETRYABLE_STATUS or status >= 500:
                    raise DownloadError(f"http_{status}", retryable=True)
                if status != 200:
                    raise DownloadError(f"http_{status}", retryable=False)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data += chunk
                    if len(data) > MAX_STICKER_BYTES:
                        raise DownloadError("too_large", retryable=False)
                return bytes(data)
        except httpx.TimeoutException:
            raise DownloadError("timeout", retryable=True) from None
        except httpx.InvalidURL:
            raise DownloadError("bad_url", retryable=False) from None
        except httpx.HTTPError:
            raise DownloadError("network_error", retryable=True) from None

    # --------------------------------------------------------------- database

    def _pending(self, after: str) -> list[tuple[str, str]]:
        with self._services.db.session() as session:
            rows = session.scalars(
                select(Sticker)
                .where(
                    Sticker.status == "pending", Sticker.url_ct.is_not(None), Sticker.md5 > after
                )
                .order_by(Sticker.md5)
                .limit(PAGE)
            ).all()
            return [(row.md5, row.url or "") for row in rows]

    def _reset_failed(self) -> int:
        with self._services.db.transaction(bump_state=False) as session:
            result = session.execute(
                update(Sticker)
                .where(Sticker.status == "unavailable", Sticker.url_ct.is_not(None))
                .values(status="pending", reason=None, attempts=0)
            )
            return int(result.rowcount or 0)  # type: ignore[attr-defined]

    def _record_success(self, md5: str, data: bytes) -> str:
        now = self._clock.now_utc()
        with self._services.db.transaction(bump_state=False) as session:
            sticker = session.get(Sticker, md5)
            if sticker is None:
                return "unavailable"
            return store_sticker_file(sticker, data, self._services.media, now)

    def _record_failure(self, md5: str, reason: str) -> None:
        now = self._clock.now_utc()
        with self._services.db.transaction(bump_state=False) as session:
            sticker = session.get(Sticker, md5)
            if sticker is not None:
                sticker.status = "unavailable"
                sticker.reason = reason
                sticker.attempts += 1
                sticker.last_attempt_at = now

    # --------------------------------------------------------------------- run

    async def run(
        self, *, retry_failed: bool = False, stop: asyncio.Event | None = None
    ) -> DownloadStats:
        """Download every pending sticker (and, with ``retry_failed``, the failed ones)."""
        stats = DownloadStats()
        if retry_failed:
            await asyncio.to_thread(self._reset_failed)
        own_client = self._http is None
        client = self._http or httpx.AsyncClient(follow_redirects=True)
        semaphore = asyncio.Semaphore(self._config.concurrency)

        async def work(md5: str, url: str) -> None:
            async with semaphore:
                if stop is not None and stop.is_set():
                    return
                stats.attempted += 1
                try:
                    data = await self._fetch(client, url)
                except DownloadError as exc:
                    await asyncio.to_thread(self._record_failure, md5, exc.reason)
                    stats.unavailable += 1
                    stats.reasons[exc.reason] += 1
                    return
                status = await asyncio.to_thread(self._record_success, md5, data)
                if status == "available":
                    stats.available += 1
                elif status == "md5_mismatch":
                    stats.md5_mismatch += 1
                else:
                    stats.unavailable += 1
                    stats.reasons["not_an_image"] += 1

        try:
            after = ""
            while True:
                page = await asyncio.to_thread(self._pending, after)
                if not page:
                    break
                await asyncio.gather(*(work(md5, url) for md5, url in page))
                after = page[-1][0]
                if stop is not None and stop.is_set():
                    stats.stopped = True
                    break
        finally:
            if own_client:
                await client.aclose()
        log.info(
            "sticker_download_finished",
            attempted=stats.attempted,
            available=stats.available,
            unavailable=stats.unavailable,
        )
        return stats


@dataclass(frozen=True)
class QueuedDownload:
    job_id: str | None
    pending: int
    already_queued: bool


def count_pending(services: Services, *, retry_failed: bool = False) -> int:
    """Stickers a download job would work on."""
    statuses = ("pending", "unavailable") if retry_failed else ("pending",)
    with services.db.session() as session:
        rows = session.scalars(
            select(Sticker.md5).where(Sticker.status.in_(statuses), Sticker.url_ct.is_not(None))
        ).all()
    return len(rows)


def queue_sticker_download(services: Services, *, retry_failed: bool = False) -> QueuedDownload:
    """Queue one ``sticker_download`` job unless there is nothing to do or one is waiting."""
    pending = count_pending(services, retry_failed=retry_failed)
    queue = JobQueue(services.db, services.clock)
    waiting: list[Any] = [
        *queue.list_jobs(status="pending", job_type=STICKER_JOB),
        *queue.list_jobs(status="running", job_type=STICKER_JOB),
    ]
    if waiting:
        return QueuedDownload(waiting[0].id, pending, already_queued=True)
    if pending == 0:
        return QueuedDownload(None, 0, already_queued=False)
    job_id = queue.enqueue(
        STICKER_JOB, {"retry_failed": retry_failed}, priority=150, max_attempts=3
    )
    return QueuedDownload(job_id, pending, already_queued=False)


@job_handler(STICKER_JOB)
async def handle_sticker_download(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("sticker_download needs the services container")
    downloader = StickerDownloader(services)
    await downloader.run(retry_failed=bool(ctx.job.payload.get("retry_failed", False)))
