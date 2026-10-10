"""Is the iLink API reachable from here: the probe of ``twin doctor`` (R-CH-001, R-NFR-005)."""

from __future__ import annotations

import httpx

from tests.support.clock import ManualClock
from twin.channel.ilink.connectivity import PROBE_TIMEOUT_S, probe_api, probe_cdn, probe_endpoint
from twin.channel.ilink.wire import CDN_BASE, DEFAULT_API_BASE


def test_any_answer_counts_as_reachable_and_the_time_it_took_is_the_clocks(
    clock: ManualClock,
) -> None:
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        clock.tick(0.25)  # the round trip, on the injected clock
        return httpx.Response(404, text="not found")

    probe = probe_endpoint("https://ilink.example.test/", transport=httpx.MockTransport(answer))
    assert probe.reachable and probe.url == "https://ilink.example.test/"
    assert probe.detail == "reachable (HTTP 404, 250 ms)"
    assert seen == ["https://ilink.example.test/"]  # one request, nothing sent with it


def test_a_timeout_and_a_refused_connection_are_not_reachable() -> None:
    def too_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    slow = probe_endpoint("https://x.example.test/", transport=httpx.MockTransport(too_slow))
    assert not slow.reachable and slow.detail == f"timed out after {PROBE_TIMEOUT_S:.0f} s"
    down = probe_endpoint("https://x.example.test/", transport=httpx.MockTransport(refused))
    assert not down.reachable and down.detail == "cannot connect (ConnectError)"


def test_the_api_and_the_cdn_are_probed_at_their_own_addresses() -> None:
    urls: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200)

    transport = httpx.MockTransport(answer)
    assert probe_api(transport).reachable and probe_cdn(transport).reachable
    assert urls == [DEFAULT_API_BASE + "/", CDN_BASE + "/"]
