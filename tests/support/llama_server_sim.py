"""A stand-in for ``llama-server`` that the tests start as a real child process (round 14).

Run as ``python -I llama_server_sim.py <the arguments of llama-server> [--sim-...]``.  It takes
the same flags the program builds for the real server (``-m``, ``--host``, ``--port``, ``-c``,
``-ngl``, ``--parallel``, ``--chat-template``, ``--no-webui``) and speaks the same HTTP
(``GET /health``, ``GET /props``, ``POST /tokenize``, ``POST /completion``), so the tests drive the
real process management: start, wait for the model to load, crash, hang, restart, stop.

``--sim-*`` options steer it:

``--sim-load-s N``           ``/health`` answers 503 for N seconds, as while a model loads
``--sim-control FILE``       read every 50 ms: ``exit`` ends the process with code 7 (a crash),
                             ``hang`` stops answering requests (a stuck server), ``ok`` resumes
``--sim-tokenizer FILE``     ``tokenizer.json`` used by ``/tokenize`` (``tokenizers`` package)
``--sim-add-bos ID``         ``/tokenize`` with ``add_special=true`` puts token ID in front (a
                             model whose GGUF asks for a BOS)
``--sim-split-special``      ``/tokenize`` does not parse control tokens: they fall apart
``--sim-reply TEXT``         the text of ``/completion``
``--sim-tps N``              the ``predicted_per_second`` of the timings
``--sim-requests FILE``      append one JSON line per request (method, path, body)
``--sim-argv FILE``          write the command line it was started with

The file only uses the standard library, except ``tokenizers`` for ``/tokenize``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
HANG = threading.Event()


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="llama-server-sim", allow_abbrev=False)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-c", "--ctx-size", type=int, default=0)
    parser.add_argument("-ngl", "--gpu-layers", default="auto")
    parser.add_argument("--parallel", "-np", type=int, default=-1)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--no-webui", action="store_true")
    parser.add_argument("--sim-load-s", type=float, default=0.0)
    parser.add_argument("--sim-control", default=None)
    parser.add_argument("--sim-tokenizer", default=None)
    parser.add_argument("--sim-add-bos", type=int, default=None)
    parser.add_argument("--sim-split-special", action="store_true")
    parser.add_argument("--sim-reply", default="好呀")
    parser.add_argument("--sim-tps", type=float, default=42.5)
    parser.add_argument("--sim-requests", default=None)
    parser.add_argument("--sim-argv", default=None)
    return parser.parse_args(argv)


def load_tokenizer(path: str | None) -> Any:
    if path is None:
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def tokens_of(options: argparse.Namespace, tokenizer: Any, body: dict[str, Any]) -> list[int]:
    text = str(body.get("content", ""))
    if tokenizer is None:
        ids = [ord(char) for char in text]
    elif options.sim_split_special:
        # control tokens are plain text: encode the pieces around and inside them separately
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            text = text.replace(token, token.replace("|", "│"))
        ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    else:
        ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if body.get("add_special") and options.sim_add_bos is not None:
        ids = [options.sim_add_bos, *ids]
    return ids


def make_handler(options: argparse.Namespace, tokenizer: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _record(self, method: str, body: dict[str, Any]) -> None:
            if options.sim_requests:
                line = json.dumps({"method": method, "path": self.path, "body": body})
                with Path(options.sim_requests).open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")

        def _read(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw) if raw else {}

        def _wait_if_hung(self) -> None:
            while HANG.is_set():
                time.sleep(0.05)

        def do_GET(self) -> None:
            self._wait_if_hung()
            self._record("GET", {})
            if self.path == "/health":
                if time.monotonic() - STARTED < options.sim_load_s:
                    self._send(503, {"error": {"code": 503, "message": "Loading model"}})
                else:
                    self._send(200, {"status": "ok"})
            elif self.path == "/props":
                self._send(
                    200,
                    {
                        "model_path": options.model,
                        "total_slots": max(1, options.parallel),
                        "build_info": "b0-sim",
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            self._wait_if_hung()
            body = self._read()
            self._record("POST", body)
            if self.path == "/tokenize":
                self._send(200, {"tokens": tokens_of(options, tokenizer, body)})
            elif self.path == "/completion":
                n = max(1, min(int(body.get("n_predict", 16)), 64))
                predicted = min(n, max(1, len(options.sim_reply)))
                self._send(
                    200,
                    {
                        "content": options.sim_reply,
                        "stop": True,
                        "stop_type": "word",
                        "stopping_word": "<|im_end|>",
                        "tokens_evaluated": len(str(body.get("prompt", ""))),
                        "tokens_predicted": predicted,
                        "truncated": False,
                        "timings": {
                            "prompt_n": len(str(body.get("prompt", ""))),
                            "prompt_ms": 12.5,
                            "predicted_n": predicted,
                            "predicted_ms": predicted * 1000.0 / options.sim_tps,
                            "predicted_per_second": options.sim_tps,
                        },
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def watch_control(path: str) -> None:
    control = Path(path)
    while True:
        time.sleep(0.05)
        try:
            command = control.read_text(encoding="utf-8").strip() if control.exists() else ""
        except OSError:
            continue
        if command == "exit":
            control.unlink(missing_ok=True)  # one shot: the next process must not die again
            sys.stdout.flush()
            os._exit(7)
        if command == "hang":
            HANG.set()
        elif command == "ok":
            HANG.clear()


def main(argv: list[str]) -> None:
    options = parse(argv)
    if options.sim_argv:
        Path(options.sim_argv).write_text(json.dumps(argv), encoding="utf-8")
    tokenizer = load_tokenizer(options.sim_tokenizer)
    if options.sim_control:
        threading.Thread(target=watch_control, args=(options.sim_control,), daemon=True).start()
    server = ThreadingHTTPServer((options.host, options.port), make_handler(options, tokenizer))
    server.daemon_threads = True
    print(f"sim llama-server listening on {options.host}:{options.port}", flush=True)
    server.serve_forever(poll_interval=0.05)


if __name__ == "__main__":
    main(sys.argv[1:])
