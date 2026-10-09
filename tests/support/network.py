"""A transport that never reaches the network (keeps unrelated tests offline)."""

from __future__ import annotations

import httpx


class OfflineTransport(httpx.BaseTransport):
    """Every request fails to connect, as on a machine without internet access."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("the test network is offline", request=request)
