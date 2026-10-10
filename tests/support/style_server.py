"""A local HTTP server that answers like llama-server and vLLM, for the style client tests.

It listens on a random port of 127.0.0.1 in a background thread (standard library only), records
every request, and answers each route from a table that tests can change.  The default answers
have the same JSON structure as the real servers; a test replaces one route to produce a
timeout, a 5xx or a malformed body.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import respx

Body = dict[str, Any] | list[Any] | bytes | None


@dataclass
class Route:
    status: int = 200
    body: Body = None
    delay_s: float = 0.0
    handler: Callable[[dict[str, Any]], Body] | None = None


@dataclass(frozen=True)
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    json: dict[str, Any]


@dataclass
class StyleServer:
    port: int = 0
    routes: dict[tuple[str, str], Route] = field(default_factory=dict)
    requests: list[Recorded] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def set(self, method: str, path: str, **kwargs: Any) -> Route:
        route = Route(**kwargs)
        self.routes[(method, path)] = route
        return route

    def paths(self) -> list[str]:
        return [request.path for request in self.requests]

    def last(self, path: str) -> Recorded:
        return next(r for r in reversed(self.requests) if r.path == path)

    def answer(
        self, method: str, path: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[int, bytes, float]:
        """Record a request and answer it from the route table: ``(status, body, delay_s)``."""
        self.requests.append(Recorded(method, path, headers, payload))
        route = self.routes.get((method, path))
        if route is None:
            return 404, b"", 0.0
        body = route.handler(payload) if route.handler else route.body
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        return route.status, data, route.delay_s

    def mount(self, router: respx.MockRouter, url: str) -> None:
        """Answer the requests for ``url`` inside the process, through a ``respx`` router.

        The same routes and the same record as the real server on a socket, without the socket:
        a test that steps a virtual clock needs the answer to be there when the request is made,
        not a thread's moment later.
        """
        self.port = int(httpx.URL(url).port or 80)
        base = url.rstrip("/")

        def handler(request: httpx.Request) -> httpx.Response:
            raw = request.content
            payload = json.loads(raw) if raw else {}
            headers = {k.lower(): v for k, v in request.headers.items()}
            status, data, _ = self.answer(request.method, request.url.path, headers, payload)
            kind = {"Content-Type": "application/json"}
            return httpx.Response(status, content=data, headers=kind)

        router.route(url__startswith=base).mock(side_effect=handler)

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                payload = json.loads(raw) if raw else {}
                headers = {k.lower(): v for k, v in self.headers.items()}
                status, data, delay = owner.answer(method, self.path, headers, payload)
                if delay:
                    time.sleep(delay)
                self.send_response(status)
                if status == 404 and not data:
                    self.end_headers()
                    return
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(data)

            def do_GET(self) -> None:
                self._serve("GET")

            def do_POST(self) -> None:
                self._serve("POST")

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        self._server = server
        self.port = int(server.server_address[1])
        self._thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def llama_defaults(server: StyleServer, reply: str = "好呀") -> None:
    """Routes with the structure of llama-server's answers."""
    server.set("GET", "/health", body={"status": "ok"})
    server.set(
        "POST",
        "/completion",
        body={
            "content": reply,
            "stop": True,
            "stop_type": "word",
            "stopping_word": "<|im_end|>",
            "tokens_evaluated": 42,
            "tokens_predicted": 3,
            "truncated": False,
        },
    )
    server.set(
        "POST",
        "/tokenize",
        handler=lambda payload: {"tokens": [ord(c) for c in payload["content"]]},
    )


def vllm_defaults(server: StyleServer, model: str = "lora-a", reply: str = "好呀") -> None:
    """Routes with the structure of vLLM's answers."""
    server.set("GET", "/health", body=b"")
    server.set(
        "GET", "/v1/models", body={"object": "list", "data": [{"id": model}, {"id": "base"}]}
    )
    server.set(
        "POST",
        "/v1/completions",
        body={
            "id": "cmpl-1",
            "object": "text_completion",
            "model": model,
            "choices": [{"index": 0, "text": reply, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44},
        },
    )
    server.set(
        "POST",
        "/tokenize",
        handler=lambda payload: {
            "count": len(payload["prompt"]),
            "max_model_len": 4096,
            "tokens": [ord(c) for c in payload["prompt"]],
        },
    )


@contextmanager
def running_server() -> Iterator[StyleServer]:
    server = StyleServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()
