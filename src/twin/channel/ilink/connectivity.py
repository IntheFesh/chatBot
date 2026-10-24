"""Reachability of the iLink API and CDN hosts (``twin doctor``, protocol document item 15).

A reachable host answers *something* (even an error page): DNS, TCP and TLS work.  The check
sends no credentials and no data; it exists because the machine may sit outside China, where
these hosts are not guaranteed to be reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from twin.channel.ilink.wire import CDN_BASE, DEFAULT_API_BASE
from twin.clock import get_clock

PROBE_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class EndpointProbe:
    url: str
    reachable: bool
    detail: str


def probe_endpoint(
    url: str, *, transport: httpx.BaseTransport | None = None, timeout_s: float = PROBE_TIMEOUT_S
) -> EndpointProbe:
    """GET ``url`` once.  Any HTTP answer counts as reachable."""
    clock = get_clock()
    started = clock.monotonic()
    try:
        with httpx.Client(transport=transport, timeout=timeout_s, follow_redirects=False) as client:
            response = client.get(url)
    except httpx.TimeoutException:
        return EndpointProbe(url, False, f"timed out after {timeout_s:.0f} s")
    except httpx.HTTPError as exc:
        return EndpointProbe(url, False, f"cannot connect ({type(exc).__name__})")
    elapsed_ms = int((clock.monotonic() - started) * 1000)
    return EndpointProbe(url, True, f"reachable (HTTP {response.status_code}, {elapsed_ms} ms)")


def probe_api(transport: httpx.BaseTransport | None = None) -> EndpointProbe:
    return probe_endpoint(DEFAULT_API_BASE + "/", transport=transport)


def probe_cdn(transport: httpx.BaseTransport | None = None) -> EndpointProbe:
    return probe_endpoint(CDN_BASE + "/", transport=transport)
