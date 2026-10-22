"""Sticker library and downloads (R-IMP-008, R-SAFE-006)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import random
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import pytest
import respx
from PIL import Image
from sqlalchemy import select

from tests.fixtures.synth_export import make_image_bytes
from tests.support.clock import InstantClock
from tests.support.gifs import animated_gif
from twin.config.settings import StickerDownloadConfig
from twin.llm.images import sniff_mime
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.jobs import HandlerRegistry, JobQueue
from twin.services import Services
from twin.stickers import download as download_module
from twin.stickers import library as library_module
from twin.stickers.download import (
    STICKER_JOB,
    RateLimiter,
    StickerDownloader,
    count_pending,
    handle_sticker_download,
    queue_sticker_download,
)
from twin.stickers.library import (
    FirstFrameError,
    backfill_first_frames,
    describe_sticker_bytes,
    ensure_first_frame,
    first_frame_png,
    sticker_file_allowed,
    store_sticker_file,
)
from twin.storage.chat_models import Sticker

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def md5_of(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def picture(seed: int, fmt: str = "PNG") -> bytes:
    return make_image_bytes(random.Random(seed), fmt)


def add_sticker(
    services: Services,
    md5: str,
    url: str | None,
    status: str = "pending",
    reason: str | None = None,
) -> None:
    with services.db.transaction() as session:
        session.add(
            Sticker(
                md5=md5, url=url, status=status, reason=reason, attempts=0, her_uses=0, user_uses=0
            )
        )


@dataclass(frozen=True)
class StickerView:
    status: str
    reason: str | None
    attempts: int
    sha256: str | None
    mime: str | None
    url: str | None


def sticker(services: Services, md5: str) -> StickerView:
    with services.db.session() as session:
        row = session.get(Sticker, md5)
        assert row is not None
        return StickerView(row.status, row.reason, row.attempts, row.sha256, row.mime, row.url)


@pytest.fixture
def web() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def config(**changes: float) -> StickerDownloadConfig:
    values: dict[str, float] = {"concurrency": 4, "per_second": 4, "retries": 3, "timeout_s": 30}
    values.update(changes)
    return StickerDownloadConfig(**values)  # type: ignore[arg-type]


def downloader(services: Services, clock: InstantClock, **changes: float) -> StickerDownloader:
    return StickerDownloader(services, config=config(**changes), clock=clock)


# --------------------------------------------------------------------- library


def test_sticker_bytes_are_identified_by_header_not_by_name() -> None:
    for fmt, mime in (("GIF", "image/gif"), ("PNG", "image/png"), ("JPEG", "image/jpeg")):
        data = picture(1, fmt)
        described = describe_sticker_bytes(data)
        assert described.mime == mime and described.size == len(data)
        assert (
            described.md5 == md5_of(data) and described.sha256 == hashlib.sha256(data).hexdigest()
        )
        assert described.width and described.height
    with pytest.raises(Exception, match="header"):
        describe_sticker_bytes(b"<html>not a picture</html>")


def test_a_matching_file_makes_the_sticker_available_and_sendable(services: Services) -> None:
    data = picture(2)
    add_sticker(services, md5_of(data), "https://s.example.test/a")
    with services.db.transaction() as session:
        row = session.get(Sticker, md5_of(data))
        assert row is not None
        assert store_sticker_file(row, data, services.media, NOW) == "available"
    row = sticker(services, md5_of(data))
    assert (row.status, row.reason, row.attempts) == ("available", None, 1)
    assert row.sha256 == hashlib.sha256(data).hexdigest() and services.media.exists(row.sha256)
    with services.db.session() as session:
        assert sticker_file_allowed(session, row.sha256)


def test_only_available_stickers_are_allowed_out(services: Services) -> None:
    good, other = picture(3), picture(4)
    add_sticker(services, md5_of(good), None)
    add_sticker(services, "f" * 32, None)  # another sticker; the file below has a different MD5
    with services.db.transaction() as session:
        store_sticker_file(session.get(Sticker, md5_of(good)), good, services.media, NOW)  # type: ignore[arg-type]
        store_sticker_file(session.get(Sticker, "f" * 32), other, services.media, NOW)  # type: ignore[arg-type]
    mismatch = sticker(services, "f" * 32)
    assert mismatch.status == "md5_mismatch" and mismatch.sha256
    with services.db.session() as session:
        assert sticker_file_allowed(session, hashlib.sha256(good).hexdigest())
        assert not sticker_file_allowed(session, mismatch.sha256)
        assert not sticker_file_allowed(session, hashlib.sha256(b"never stored").hexdigest())


def test_pending_and_unavailable_stickers_are_never_allowed(services: Services) -> None:
    for status in ("pending", "unavailable", "md5_mismatch"):
        digest = hashlib.sha256(status.encode()).hexdigest()
        with services.db.transaction() as session:
            session.add(
                Sticker(
                    md5=hashlib.md5(status.encode(), usedforsecurity=False).hexdigest(),
                    status=status,
                    sha256=digest,
                    attempts=0,
                    her_uses=0,
                    user_uses=0,
                )
            )
        with services.db.session() as session:
            assert not sticker_file_allowed(session, digest)


def test_a_file_that_is_not_a_picture_or_too_large_is_unavailable(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_sticker(services, "a" * 32, None)
    with services.db.transaction() as session:
        row = session.get(Sticker, "a" * 32)
        assert row is not None
        assert store_sticker_file(row, b"plain text", services.media, NOW) == "unavailable"
        assert row.reason == "not_an_image" and row.sha256 is None
        monkeypatch.setattr(library_module, "MAX_STICKER_BYTES", 10)
        assert store_sticker_file(row, picture(6), services.media, NOW) == "unavailable"
        assert row.reason == "too_large"


# -------------------------------------------------------------------- downloads


async def test_a_sticker_is_downloaded_checked_and_stored(
    services: Services, web: respx.MockRouter
) -> None:
    data = picture(10, "GIF")
    url = "https://stickers.example.test/one"
    add_sticker(services, md5_of(data), url)
    route = web.get(url).respond(200, content=data, headers={"content-type": "text/plain"})
    stats = await downloader(services, InstantClock()).run()
    assert route.call_count == 1
    assert (stats.attempted, stats.available, stats.unavailable) == (1, 1, 0)
    row = sticker(services, md5_of(data))
    assert row.status == "available" and row.mime == "image/gif"  # from the header, not the server
    assert row.sha256 and services.media.read_bytes(row.sha256) == data
    with services.db.session() as session:
        assert sticker_file_allowed(session, row.sha256)


async def test_a_different_md5_is_kept_and_marked(
    services: Services, web: respx.MockRouter
) -> None:
    announced = md5_of(b"the announced file")
    add_sticker(services, announced, "https://stickers.example.test/m")
    served = picture(11)
    web.get("https://stickers.example.test/m").respond(200, content=served)
    stats = await downloader(services, InstantClock()).run()
    assert stats.md5_mismatch == 1 and stats.available == 0
    row = sticker(services, announced)
    assert row.status == "md5_mismatch" and row.sha256 == hashlib.sha256(served).hexdigest()
    assert services.media.exists(row.sha256)
    with services.db.session() as session:
        assert not sticker_file_allowed(session, row.sha256)


async def test_a_missing_sticker_is_unavailable_without_retries(
    services: Services, web: respx.MockRouter
) -> None:
    add_sticker(services, "b" * 32, "https://stickers.example.test/gone")
    route = web.get("https://stickers.example.test/gone").respond(404)
    clock = InstantClock()
    stats = await downloader(services, clock).run()
    assert route.call_count == 1  # a 404 is final
    assert stats.unavailable == 1 and stats.reasons == {"http_404": 1}
    row = sticker(services, "b" * 32)
    assert (row.status, row.reason, row.attempts) == ("unavailable", "http_404", 1)
    assert 1.0 not in clock.sleeps  # no back-off was needed


async def test_server_errors_are_retried_three_times_with_growing_pauses(
    services: Services, web: respx.MockRouter
) -> None:
    add_sticker(services, "c" * 32, "https://stickers.example.test/flaky")
    route = web.get("https://stickers.example.test/flaky").respond(503)
    clock = InstantClock()
    stats = await downloader(services, clock).run()
    assert route.call_count == 4  # the first try and three retries
    assert stats.reasons == {"http_503": 1}
    backoff = [s for s in clock.sleeps if s >= 1.0]
    assert backoff == [1.0, 2.0, 4.0]
    assert sticker(services, "c" * 32).reason == "http_503"


async def test_a_flaky_server_that_recovers_gives_the_sticker(
    services: Services, web: respx.MockRouter
) -> None:
    data = picture(12)
    add_sticker(services, md5_of(data), "https://stickers.example.test/recover")
    route = web.get("https://stickers.example.test/recover")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(429),
        httpx.Response(200, content=data),
    ]
    stats = await downloader(services, InstantClock()).run()
    assert route.call_count == 3 and stats.available == 1


async def test_timeouts_and_network_errors_are_retried_and_recorded(
    services: Services, web: respx.MockRouter
) -> None:
    add_sticker(services, "d" * 32, "https://stickers.example.test/slow")
    add_sticker(services, "e" * 32, "https://stickers.example.test/down")
    slow = web.get("https://stickers.example.test/slow")
    slow.side_effect = httpx.ReadTimeout("too slow")
    down = web.get("https://stickers.example.test/down")
    down.side_effect = httpx.ConnectError("refused")
    stats = await downloader(services, InstantClock()).run()
    assert slow.call_count == 4 and down.call_count == 4
    assert stats.reasons == {"timeout": 1, "network_error": 1}


async def test_the_request_timeout_comes_from_the_configuration(
    services: Services, web: respx.MockRouter
) -> None:
    data = picture(13)
    add_sticker(services, md5_of(data), "https://stickers.example.test/t")
    route = web.get("https://stickers.example.test/t").respond(200, content=data)
    await downloader(services, InstantClock(), timeout_s=7).run()
    assert route.calls.last.request.extensions["timeout"]["read"] == 7


async def test_something_that_is_not_a_picture_is_unavailable(
    services: Services, web: respx.MockRouter
) -> None:
    add_sticker(services, "9" * 32, "https://stickers.example.test/html")
    web.get("https://stickers.example.test/html").respond(200, content=b"<html>login</html>")
    stats = await downloader(services, InstantClock()).run()
    assert stats.reasons == {"not_an_image": 1}
    assert sticker(services, "9" * 32).reason == "not_an_image"


async def test_odd_urls_and_oversized_answers_are_final(
    services: Services, web: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_sticker(services, "7" * 32, "ftp://stickers.example.test/x")
    add_sticker(services, "8" * 32, "https://stickers.example.test/big")
    web.get("https://stickers.example.test/big").respond(200, content=b"x" * 5000)
    monkeypatch.setattr(download_module, "MAX_STICKER_BYTES", 1000)
    stats = await downloader(services, InstantClock()).run()
    assert stats.reasons == {"bad_url": 1, "too_large": 1}


async def test_at_most_four_downloads_run_at_once(
    services: Services, web: respx.MockRouter
) -> None:
    items = [picture(100 + n) for n in range(10)]
    for data in items:
        add_sticker(services, md5_of(data), f"https://stickers.example.test/{md5_of(data)}")
    inside = 0
    peak = 0
    four_arrived = asyncio.Event()

    def handler_for(data: bytes):  # type: ignore[no-untyped-def]
        async def respond(request: httpx.Request) -> httpx.Response:
            nonlocal inside, peak
            inside += 1
            peak = max(peak, inside)
            if inside >= 4:
                four_arrived.set()
            await asyncio.wait_for(four_arrived.wait(), 5)
            for _ in range(3):
                await asyncio.sleep(0)
            inside -= 1
            return httpx.Response(200, content=data)

        return respond

    for data in items:
        web.get(f"https://stickers.example.test/{md5_of(data)}").mock(side_effect=handler_for(data))
    stats = await downloader(services, InstantClock()).run()
    assert stats.available == 10
    assert peak == 4  # the configured concurrency, reached and never exceeded


async def test_the_limiter_spaces_request_starts_by_the_configured_rate() -> None:
    clock = InstantClock()
    limiter = RateLimiter(4, clock)
    starts: list[float] = []

    async def start() -> None:
        await limiter.wait()
        starts.append(clock.monotonic())

    await asyncio.gather(*(start() for _ in range(9)))
    ordered = sorted(starts)
    gaps = [later - earlier for earlier, later in pairwise(ordered)]
    assert len(gaps) == 8 and all(gap >= 0.25 - 1e-9 for gap in gaps)
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0, clock)


async def test_downloads_wait_for_the_rate_limit(services: Services, web: respx.MockRouter) -> None:
    clock = InstantClock()
    for n in range(5):
        data = picture(200 + n)
        add_sticker(services, md5_of(data), f"https://stickers.example.test/{md5_of(data)}")
        web.get(f"https://stickers.example.test/{md5_of(data)}").respond(200, content=data)
    stats = await downloader(services, clock, per_second=4).run()
    assert stats.available == 5
    assert len([pause for pause in clock.sleeps if pause > 0]) >= 4  # all but the first waited


async def test_an_interrupted_download_continues_with_what_is_left(
    services: Services, web: respx.MockRouter
) -> None:
    items = [picture(300 + n) for n in range(6)]
    routes = {}
    for data in items:
        add_sticker(services, md5_of(data), f"https://stickers.example.test/{md5_of(data)}")
        routes[md5_of(data)] = web.get(f"https://stickers.example.test/{md5_of(data)}").respond(
            200, content=data
        )
    stop = asyncio.Event()
    stop.set()  # asked to stop before the first page is processed
    first = await downloader(services, InstantClock()).run(stop=stop)
    assert first.stopped and first.available == 0
    second = await downloader(services, InstantClock(), concurrency=1).run()
    assert second.available == 6
    third = await downloader(services, InstantClock()).run()
    assert third.attempted == 0  # nothing is fetched twice
    assert all(route.call_count == 1 for route in routes.values())


async def test_failed_stickers_are_tried_again_only_on_request(
    services: Services, web: respx.MockRouter
) -> None:
    data = picture(400)
    add_sticker(services, md5_of(data), "https://stickers.example.test/later")
    route = web.get("https://stickers.example.test/later")
    route.side_effect = [httpx.Response(404), httpx.Response(200, content=data)]
    clock = InstantClock()
    await downloader(services, clock).run()
    assert sticker(services, md5_of(data)).status == "unavailable"
    again = await downloader(services, clock).run()
    assert again.attempted == 0 and route.call_count == 1  # not retried by default
    retried = await downloader(services, clock).run(retry_failed=True)
    assert retried.available == 1 and route.call_count == 2
    assert sticker(services, md5_of(data)).status == "available"


async def test_stickers_without_a_url_or_already_done_are_left_alone(
    services: Services, web: respx.MockRouter
) -> None:
    add_sticker(services, "1" * 32, None)
    add_sticker(services, "2" * 32, "https://stickers.example.test/done", status="available")
    stats = await downloader(services, InstantClock()).run(retry_failed=True)
    assert stats.attempted == 0 and not web.calls


# ------------------------------------------------------------------------ queue


def test_one_download_job_is_queued_for_the_pending_stickers(services: Services) -> None:
    assert queue_sticker_download(services).job_id is None  # nothing pending
    add_sticker(services, "3" * 32, "https://stickers.example.test/q")
    add_sticker(services, "4" * 32, None)  # no URL: cannot be downloaded
    add_sticker(services, "5" * 32, "https://stickers.example.test/x", status="unavailable")
    assert count_pending(services) == 1 and count_pending(services, retry_failed=True) == 2
    first = queue_sticker_download(services)
    assert first.job_id and first.pending == 1 and not first.already_queued
    second = queue_sticker_download(services)
    assert second.already_queued and second.job_id == first.job_id
    jobs = JobQueue(services.db, services.clock).list_jobs(job_type=STICKER_JOB)
    assert len(jobs) == 1 and jobs[0].payload == {"retry_failed": False}


async def test_the_job_handler_downloads_through_the_worker(
    services: Services, web: respx.MockRouter
) -> None:
    data = picture(500)
    add_sticker(services, md5_of(data), "https://stickers.example.test/job")
    web.get("https://stickers.example.test/job").respond(200, content=data)
    queue_sticker_download(services)
    registry = HandlerRegistry()
    registry.register(STICKER_JOB, handle_sticker_download)
    summary = await run_jobs_until_idle(services, registry, tick_seconds=0.01)
    assert summary.done == 1 and summary.failed == 0
    assert sticker(services, md5_of(data)).status == "available"


def test_sticker_rows_are_listed_by_status(services: Services) -> None:
    add_sticker(services, "6" * 32, "https://stickers.example.test/s")
    with services.db.session() as session:
        statuses = session.scalars(select(Sticker.status)).all()
    assert statuses == ["pending"]


# ----------------------------------- R17-39: the numbers of the download are in the import report


@pytest.fixture
def instant(services: Services) -> InstantClock:
    """The downloads pace themselves (a request every quarter second, back-off): no real waiting."""
    clock = InstantClock(services.clock.now_utc())
    services.clock = clock  # type: ignore[assignment]
    return clock


def finished_import(services: Services, tmp_path: Path, **options: object) -> tuple[Any, Any]:
    from tests.support.ingest import make_export, run_import

    export = make_export(
        tmp_path,
        target_messages=150,
        other_conversations=0,
        sticker_kinds=5,
        local_sticker_ratio=0.0,
        **options,
    )
    outcome = run_import(services, export)
    assert outcome.report_path is not None
    return export, outcome


def report_text(outcome: Any) -> str:
    return Path(str(outcome.report_path)).read_text(encoding="utf-8")


def report_file(outcome: Any) -> Path:
    return Path(str(outcome.report_path))


async def run_download_job(services: Services) -> None:
    registry = HandlerRegistry()
    registry.register(STICKER_JOB, handle_sticker_download)
    summary = await run_jobs_until_idle(services, registry, tick_seconds=0.01)
    assert summary.failed == 0


async def test_the_download_job_writes_its_numbers_into_the_import_report(
    services: Services, instant: InstantClock, web: respx.MockRouter, tmp_path: Path
) -> None:
    """R17-39: the report used to say "see the output of that command" - which nobody keeps."""
    export, outcome = finished_import(services, tmp_path)
    before = report_text(outcome)
    assert "下载统计见该命令的输出" not in before and "还没有下载过" in before
    infos = list(export.stickers.values())
    for good in (infos[0], infos[3], infos[4]):
        web.get(good.url).respond(200, content=good.data)
    web.get(infos[1].url).respond(404)
    web.get(infos[2].url).respond(200, content=picture(900, "GIF"))  # not the picture it names
    await run_download_job(services)
    after = report_text(outcome)
    section = after.split("## 表情包")[1].split("\n## ")[0]
    assert "最近一次下载" in section
    assert "尝试 5" in section and "可用 3" in section
    assert "MD5 不符 1" in section and "不可用 1" in section and "http_404 × 1" in section
    assert "| available | 3 |" in section and "| md5_mismatch | 1 |" in section
    assert "还没有下载过" not in after and after.count("## 表情包") == 1
    assert after.count("## 风格变化") == 1 and "## 导入后钩子" in after  # the rest is untouched
    assert after.split("## 表情包")[0] == before.split("## 表情包")[0]


async def test_the_summary_survives_in_the_settings_and_a_later_job_replaces_it(
    services: Services, instant: InstantClock, web: respx.MockRouter, tmp_path: Path
) -> None:
    from twin.stickers.download_summary import load_download_summary

    export, outcome = finished_import(services, tmp_path)
    assert load_download_summary(services) is None
    infos = list(export.stickers.values())
    web.get(infos[0].url).respond(200, content=infos[0].data)
    for other in infos[1:]:
        web.get(other.url).respond(503)
    await run_download_job(services)
    first = load_download_summary(services)
    assert first is not None and (first.attempted, first.available, first.unavailable) == (5, 1, 4)
    assert first.reasons == {"http_503": 4} and not first.stopped and first.at.tzinfo is not None
    # the user fixes the network and downloads the failed ones again
    for other in infos[1:]:
        web.get(other.url).respond(200, content=other.data)
    queue_sticker_download(services, retry_failed=True)
    registry = HandlerRegistry()
    registry.register(STICKER_JOB, handle_sticker_download)
    await run_jobs_until_idle(services, registry, tick_seconds=0.01)
    second = load_download_summary(services)
    assert second is not None and (second.attempted, second.available) == (4, 4)
    text = report_text(outcome)
    assert "尝试 4" in text and "http_503" not in text.split("## 表情包")[1].split("\n## ")[0]


async def test_a_download_with_no_import_report_to_update_is_not_an_error(
    services: Services, instant: InstantClock, web: respx.MockRouter
) -> None:
    data = picture(700)
    add_sticker(services, md5_of(data), "https://stickers.example.test/lonely")
    web.get("https://stickers.example.test/lonely").respond(200, content=data)
    queue_sticker_download(services)
    await run_download_job(services)  # no import ever ran: there is no report
    assert sticker(services, md5_of(data)).status == "available"


async def test_a_report_that_was_deleted_is_not_rewritten_from_nothing(
    services: Services, instant: InstantClock, web: respx.MockRouter, tmp_path: Path
) -> None:
    export, outcome = finished_import(services, tmp_path)
    report_file(outcome).unlink()
    for info in export.stickers.values():
        web.get(info.url).respond(200, content=info.data)
    await run_download_job(services)
    assert not report_file(outcome).exists()


# ------------------------------- R17-38: the still first frame of a GIF, stored beside it


class RowFacts(NamedTuple):
    status: str
    mime: str | None
    sha256: str | None
    first_frame_sha256: str | None


def sticker_row(services: Services, md5: str) -> RowFacts:
    with services.db.session() as session:
        row = session.get(Sticker, md5)
        assert row is not None
        return RowFacts(row.status, row.mime, row.sha256, row.first_frame_sha256)


def add_stored_gif(services: Services, data: bytes, *, md5: str | None = None) -> str:
    """A library sticker whose file is ``data``; returns its MD5 (the real one by default)."""
    key = md5 or md5_of(data)
    add_sticker(services, key, "https://s.example.test/g")
    with services.db.transaction() as session:
        row = session.get(Sticker, key)
        assert row is not None
        store_sticker_file(row, data, services.media, NOW)
    return key


def test_a_gif_is_stored_with_the_still_first_frame_as_a_png(services: Services) -> None:
    data = animated_gif()
    md5 = add_stored_gif(services, data)
    row = sticker_row(services, md5)
    assert row.status == "available" and row.mime == "image/gif"
    assert row.first_frame_sha256 and row.first_frame_sha256 != row.sha256
    still = services.media.read_bytes(row.first_frame_sha256)
    assert sniff_mime(still) == "image/png"
    with Image.open(io.BytesIO(still)) as frame:
        assert frame.size == (12, 8) and getattr(frame, "n_frames", 1) == 1
        assert frame.convert("RGB").getpixel((0, 0)) == (255, 0, 0)  # the first frame, not a later
    with Image.open(io.BytesIO(data)) as moving:
        assert moving.n_frames == 3  # and the original is the moving picture, untouched
    assert services.media.read_bytes(row.sha256) == data


def test_a_picture_that_is_not_a_gif_has_no_first_frame(services: Services) -> None:
    png, jpeg = picture(11, "PNG"), picture(12, "JPEG")
    for data in (png, jpeg):
        assert sticker_row(services, add_stored_gif(services, data)).first_frame_sha256 is None


def test_a_gif_whose_frames_cannot_be_read_is_kept_but_has_no_first_frame(
    services: Services,
) -> None:
    broken = b"GIF89a" + bytes(range(40))  # the header says GIF, nothing after it is a picture
    md5 = add_stored_gif(services, broken)
    row = sticker_row(services, md5)
    assert row.status == "available" and row.first_frame_sha256 is None
    with pytest.raises(FirstFrameError, match="could not be read"):
        first_frame_png(broken)
    with pytest.raises(FirstFrameError, match="could not be read"):
        ensure_first_frame(services.db, services.media, md5)


def test_the_allow_list_accepts_the_gif_and_its_first_frame_and_nothing_else(
    services: Services,
) -> None:
    md5 = add_stored_gif(services, animated_gif())
    row = sticker_row(services, md5)
    assert row.sha256 and row.first_frame_sha256
    with services.db.session() as session:
        assert sticker_file_allowed(session, row.sha256)
        assert sticker_file_allowed(session, row.first_frame_sha256)
        for other in (hashlib.sha256(picture(99)).hexdigest(), "f" * 64, "nope"):
            assert not sticker_file_allowed(session, other)


def test_the_first_frame_of_a_sticker_that_may_not_be_sent_is_not_allowed_out_either(
    services: Services,
) -> None:
    md5 = add_stored_gif(services, animated_gif())
    row = sticker_row(services, md5)
    assert row.first_frame_sha256
    with services.db.transaction() as session:
        stored = session.get(Sticker, md5)
        assert stored is not None
        stored.status = "md5_mismatch"  # the file is kept for recognition, never sent
    with services.db.session() as session:
        assert not sticker_file_allowed(session, row.first_frame_sha256)


def test_the_first_frame_is_made_again_on_demand_and_only_for_a_gif_of_the_library(
    services: Services,
) -> None:
    md5 = add_stored_gif(services, animated_gif())
    row = sticker_row(services, md5)
    first = row.first_frame_sha256
    assert first
    assert ensure_first_frame(services.db, services.media, md5) == first  # nothing to do
    services.media.delete(first)  # the PNG is gone from the store (or the row predates it)
    with services.db.transaction() as session:
        session.get(Sticker, md5).first_frame_sha256 = None  # type: ignore[union-attr]
    again = ensure_first_frame(services.db, services.media, md5)
    assert again == first and services.media.exists(first)  # the same picture, by its digest
    with pytest.raises(FirstFrameError, match="not a usable GIF"):
        ensure_first_frame(services.db, services.media, "0" * 32)
    png = add_stored_gif(services, picture(21, "PNG"))
    with pytest.raises(FirstFrameError, match="not a usable GIF"):
        ensure_first_frame(services.db, services.media, png)


def test_making_the_first_frame_tells_the_running_application(services: Services) -> None:
    from twin.storage.state import current_state_version

    md5 = add_stored_gif(services, animated_gif())
    with services.db.transaction(bump_state=False) as session:
        session.get(Sticker, md5).first_frame_sha256 = None  # type: ignore[union-attr]
    before = current_state_version(services.db)
    ensure_first_frame(services.db, services.media, md5)
    assert current_state_version(services.db) > before
    settled = current_state_version(services.db)
    ensure_first_frame(services.db, services.media, md5)  # already there: nothing is news
    assert current_state_version(services.db) == settled


def test_the_backfill_gives_every_available_gif_its_first_frame_once(services: Services) -> None:
    old = [
        add_stored_gif(services, animated_gif(((n * 40, 0, 0), (0, n * 40, 0))), md5=None)
        for n in range(1, 4)
    ]
    still = add_stored_gif(services, picture(31, "PNG"))
    broken = add_stored_gif(services, b"GIF89a" + bytes(range(30)))
    with services.db.transaction() as session:  # stored before the column existed
        for md5 in old:
            session.get(Sticker, md5).first_frame_sha256 = None  # type: ignore[union-attr]
    first = backfill_first_frames(services.db, services.media)
    assert (first.made, first.present, first.failed) == (3, 0, 1)
    assert all(sticker_row(services, md5).first_frame_sha256 for md5 in old)
    assert sticker_row(services, still).first_frame_sha256 is None
    assert sticker_row(services, broken).first_frame_sha256 is None
    second = backfill_first_frames(services.db, services.media)
    assert (second.made, second.present, second.failed) == (0, 3, 1)  # again: nothing to do


# ------------------------------- the edges of the new code (a gone file, a damaged summary, ...)


def test_the_first_frame_cannot_be_made_when_the_file_of_the_gif_is_gone(
    services: Services,
) -> None:
    md5 = add_stored_gif(services, animated_gif())
    row = sticker_row(services, md5)
    assert row.sha256 and row.first_frame_sha256
    services.media.delete(row.sha256)
    services.media.delete(row.first_frame_sha256)
    with services.db.transaction() as session:
        session.get(Sticker, md5).first_frame_sha256 = None  # type: ignore[union-attr]
    with pytest.raises(FirstFrameError, match="file of the GIF cannot be read"):
        ensure_first_frame(services.db, services.media, md5)


def test_the_summary_of_a_download_says_why_and_when_it_was_stopped(services: Services) -> None:
    from twin.stickers.download_summary import (
        DownloadSummary,
        load_download_summary,
        save_download_summary,
    )

    summary = DownloadSummary(
        at=NOW,
        attempted=3,
        available=1,
        md5_mismatch=0,
        unavailable=2,
        reasons={"timeout": 2},
        stopped=True,
    )
    line = summary.describe()
    assert "尝试 3，可用 1，MD5 不符 0，不可用 2" in line and "原因：timeout × 2" in line
    assert "被中途停止" in line and "2026-10-09 12:00 UTC" in line
    save_download_summary(services, summary)
    assert load_download_summary(services) == summary


def test_a_damaged_summary_in_the_settings_is_not_trusted(services: Services) -> None:
    from twin.stickers.download_summary import SUMMARY_KEY, load_download_summary
    from twin.storage.settings_store import put_setting

    for damaged in ({"at": "not a date"}, {"attempted": 3}, {"at": NOW.isoformat(), "reasons": 5}):
        with services.db.transaction(bump_state=False) as session:
            put_setting(session, SUMMARY_KEY, damaged, clock=services.clock, record_history=False)
        assert load_download_summary(services) is None
    with services.db.transaction(bump_state=False) as session:
        put_setting(session, SUMMARY_KEY, "text", clock=services.clock, record_history=False)
    assert load_download_summary(services) is None


def test_a_picture_too_big_to_be_a_sticker_is_never_read_for_its_md5(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.stickers.incoming import recognise_library_sticker
    from twin.stickers.library import MAX_STICKER_BYTES

    def never(self: object, sha256: str) -> bytes:
        raise AssertionError("a big photo must not be read to look it up")

    monkeypatch.setattr(type(services.media), "read_bytes", never)
    assert recognise_library_sticker(services, "ab" * 32, MAX_STICKER_BYTES + 1) is None
