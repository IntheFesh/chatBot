"""A synthetic world for the evaluation tests (round 09b).

``build_eval_world`` is the world of the training-export tests (a conversation with a profile and
a routine in both scopes, a past and a live persona card with different markers, tagged stickers,
two facts) with what a reply needs on top: the retrieval library of her real replies, the
DeepSeek key, and a hold-out wide enough (a quarter of her reply blocks) to draw a test from.
Nothing here is a real conversation.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import respx
from sqlalchemy import select

from tests.fixtures.synth_export import make_image_bytes
from tests.support.deepseek import API, TEST_KEY, completion, error
from tests.support.embedding import HashingBackend
from tests.support.export_world import World, build_world
from tests.support.memory import add_fact
from tests.support.policies import AlwaysOffPeak
from twin.eval.blind import EVAL_GENERATE_JOB, handle_eval_generate, request_from
from twin.eval.memory_test import (
    EVAL_MEMORY_JOB,
    JUDGE_SYSTEM,
    QUESTION_SYSTEM,
    handle_eval_memory,
)
from twin.eval.samples import DrawnSample, SampleDrawer
from twin.eval.sandbox import SandboxKit, SandboxMode, SandboxRequest, build_sandbox
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.retrieval.indexer import run_index
from twin.services import Services
from twin.stickers.library import store_sticker_file
from twin.storage.chat_models import Sticker


def build_eval_world(
    services: Services, embedder: HashingBackend, *, days: int = 30, ratio: float = 0.25
) -> World:
    """The export world plus the retrieval library, the key and a wide hold-out."""
    services.settings.retrieval.holdout_ratio = ratio
    world = build_world(services, embedder, days=days)
    run_index(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return world


@dataclass
class DeepSeekScript:
    """A DeepSeek endpoint for ``respx`` that answers by a function of the request.

    ``reply`` gets the parsed request body and returns the text of the answer; every request is
    kept in :attr:`requests` (bodies only, as sent).
    """

    reply: Callable[[dict[str, Any]], str]
    prompt_tokens: int = 400
    completion_tokens: int = 12
    requests: list[dict[str, Any]] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        try:
            text = self.reply(body)
        except Exception as exc:  # a bug of the test script: fail fast, never retry for ever
            self.errors.append(exc)
            return error(400, f"the test script failed: {type(exc).__name__}")
        return httpx.Response(
            200,
            json=completion(
                text,
                prompt=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                request_id=f"req-{len(self.requests)}",
            ),
        )

    @property
    def system_texts(self) -> list[str]:
        return [str(r["messages"][0]["content"]) for r in self.requests]

    @property
    def last_texts(self) -> list[str]:
        return [str(r["messages"][-1]["content"]) for r in self.requests]


def mock_deepseek(script: DeepSeekScript) -> Iterator[respx.MockRouter]:
    """A respx router that sends the DeepSeek chat endpoint to ``script`` (use as a fixture)."""
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=script)
        yield router


def make_stickers_available(services: Services) -> list[str]:
    """Give every sticker of the synthetic conversation a real picture and mark it available.

    The synthetic stickers have made-up MD5s, so the picture's own MD5 differs; the status is set
    by hand after the file is stored.  Returns the MD5s (the library keys).
    """
    found: list[str] = []
    with services.db.transaction(bump_state=False) as session:
        for number, row in enumerate(session.scalars(select(Sticker).order_by(Sticker.md5)), 1):
            data = make_image_bytes(random.Random(number), "PNG")
            store_sticker_file(row, data, services.media, services.clock.now_utc())
            row.status, row.reason = "available", None
            found.append(row.md5)
    return found


@asynccontextmanager
async def open_kit(
    world: World, mode: SandboxMode = SandboxMode.HOLDOUT, batch_id: str | None = "eval-test-1"
) -> AsyncIterator[SandboxKit]:
    """A sandbox kit of the world, closed when the block ends."""
    built = build_sandbox(world.services, mode=mode, batch_id=batch_id)
    try:
        yield built
    finally:
        await built.aclose()


def sample_of(world: World, count: int = 3, seed: int = 5) -> list[DrawnSample]:
    """Contexts drawn from the hold-out of the world."""
    return SampleDrawer(world.services).draw(count, seed=seed).samples


def request_for(sample: DrawnSample, backend: str = "deepseek") -> SandboxRequest:
    """The round of a drawn sample as the sandbox is asked to answer it."""
    return request_from({**sample.payload(), "seed": 7}, sample.at, backend, "off")


def eval_worker(services: Services) -> Worker:
    """A job worker that runs the evaluation jobs (generation and memory questions)."""
    registry = HandlerRegistry()
    registry.register(EVAL_GENERATE_JOB, handle_eval_generate)
    registry.register(EVAL_MEMORY_JOB, handle_eval_memory)
    return Worker(
        JobQueue(services.db, services.clock),
        registry,
        services.clock,
        services=services,
        offpeak=AlwaysOffPeak(),
        alerts=services.alerts,
    )


KNOWN = datetime(2026, 9, 3, 12, tzinfo=UTC)  # when the facts of the memory tests became known


def marker(n: int) -> str:
    return f"那把编号{n:02d}的蓝色钥匙"


def add_memory_facts(world: World, *, real: int = 12, bot: int = 12) -> None:
    for n in range(real):
        add_fact(world.memory, f"第{n}号旧记忆：{marker(n)}", KNOWN, source="real_record")
    for n in range(bot):
        source = "user_said" if n % 2 else "bot_invented"
        add_fact(world.memory, f"第{n + 50}号新记忆：{marker(n + 50)}", KNOWN, source=source)


def memory_script(*, leak_first: bool = False) -> DeepSeekScript:
    """DeepSeek for the three jobs: the question, the bot's answer and the verdict.

    A fact reads ``<name>：<what>``.  The question names the fact (so the live memory finds it),
    the key point is the ``what``, the bot answers with it when its memory block has the fact,
    and the judge says correct when the answer holds the key point.
    """
    asked: dict[str, int] = {}

    def reply(body: dict[str, object]) -> str:
        messages = body["messages"]
        assert isinstance(messages, list)
        system, last = str(messages[0]["content"]), str(messages[-1]["content"])
        if system.startswith(QUESTION_SYSTEM[:12]):
            fact = re.search(r"事实（[^）]*）：(.+?)：(.+)", last)
            assert fact, last
            name, key = fact.group(1), fact.group(2).split("\n")[0].strip()
            asked[name] = asked.get(name, 0) + 1
            question = f"{name}是什么情况"
            if leak_first and asked[name] == 1:
                question += key  # the answer is in the question: it has to be asked again
            return json.dumps({"question": question, "key_points": [key]}, ensure_ascii=False)
        if system.startswith(JUDGE_SYSTEM[:12]):
            key = re.search(r"要点：(.+)", last)
            assert key
            answered = last.split("机器人的回答：", 1)[1]
            verdict = "correct" if key.group(1).strip() in answered else "wrong"
            return json.dumps({"verdict": verdict, "reason": "按要点判"}, ensure_ascii=False)
        asked_about = re.search(r"对方这一轮说的话：\n(.+?)是什么情况", last)
        assert asked_about, last
        known = re.search(
            re.escape(asked_about.group(1)) + r"：([^\n]+)", last.split("对方这一轮说的话")[0]
        )
        return known.group(1).strip() if known else "想不起来了"  # from the memory block

    return DeepSeekScript(reply, prompt_tokens=60, completion_tokens=20)


def serve(script: DeepSeekScript) -> respx.MockRouter:
    router = respx.mock(assert_all_called=False)
    router.start()
    router.post(API).mock(side_effect=script)
    return router
