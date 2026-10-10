"""Booking the calls of a stretch of code as evaluation on a batch (R-EVAL-009, R-LLM-014)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator

import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY, ok
from twin.llm.onetime import BatchPausedError
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.scope import call_scope, current_call_scope
from twin.llm.types import ChatMessage, LedgerTag, Purpose
from twin.services import Services
from twin.storage.models import CostLedger
from twin.storage.settings_store import put_setting

HELLO: list[ChatMessage] = [{"role": "user", "content": "你好"}]
BATCH = LedgerTag("one_time", "eval-scope-1")


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(return_value=ok(content="好"))
        yield router


@pytest.fixture
async def runtime(services: Services) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    services.settings.deepseek.offline_model = "deepseek-v4-pro"  # not the chat model
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


def rows(services: Services) -> list[tuple[str, str, str | None, str]]:
    with services.db.session() as session:
        found = session.scalars(select(CostLedger).order_by(CostLedger.at, CostLedger.id)).all()
        return [(r.purpose, r.account, r.batch_id, r.model) for r in found]


async def test_calls_in_the_scope_are_booked_as_evaluation_with_the_model_of_a_reply(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    assert current_call_scope() is None
    with call_scope(Purpose.EVAL, BATCH) as scope:
        assert current_call_scope() is scope
        await runtime.client.chat(HELLO, purpose=Purpose.REPLY)
        await runtime.client.chat(HELLO, purpose=Purpose.PLAN)
    assert current_call_scope() is None
    await runtime.client.chat(HELLO, purpose=Purpose.REPLY)  # outside: an ordinary reply
    await runtime.client.chat(HELLO, purpose=Purpose.EVAL)  # an evaluation call by itself
    assert rows(services) == [
        ("eval", "one_time", "eval-scope-1", "deepseek-flash"),
        ("eval", "one_time", "eval-scope-1", "deepseek-flash"),
        ("reply", "daily", None, "deepseek-flash"),
        ("eval", "daily", None, "deepseek-v4-pro"),
    ]
    sent = [json.loads(call.request.content)["model"] for call in api.calls]
    assert sent == ["deepseek-flash", "deepseek-flash", "deepseek-flash", "deepseek-v4-pro"]


async def test_the_scope_belongs_to_the_task_that_opened_it(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    inside_open = asyncio.Event()
    outside_done = asyncio.Event()

    async def inside() -> None:
        with call_scope(Purpose.EVAL, BATCH):
            inside_open.set()
            await outside_done.wait()  # the other task makes its call while this scope is open
            await runtime.client.chat(HELLO, purpose=Purpose.REPLY)

    async def outside() -> None:
        await inside_open.wait()
        assert current_call_scope() is None
        await runtime.client.chat(HELLO, purpose=Purpose.REPLY)
        outside_done.set()

    await asyncio.gather(inside(), outside())
    assert sorted(r[:3] for r in rows(services)) == [
        ("eval", "one_time", "eval-scope-1"),
        ("reply", "daily", None),
    ]


async def test_a_paused_batch_refuses_the_calls_made_inside_its_scope(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    with services.db.transaction(bump_state=False) as session:
        put_setting(
            session,
            "onetime.batch.eval-scope-1",
            {"estimated_usd": 1.0, "paused_at": "2026-10-09T00:00:00+00:00"},
            clock=services.clock,
        )
    with call_scope(Purpose.EVAL, BATCH), pytest.raises(BatchPausedError):
        await runtime.client.chat(HELLO, purpose=Purpose.REPLY)
    assert api.calls.call_count == 0 and rows(services) == []


def test_scopes_nest_and_restore_the_outer_one() -> None:
    other = LedgerTag("one_time", "eval-scope-2")
    with call_scope(Purpose.EVAL, BATCH) as outer:
        with call_scope(Purpose.PLAN, other) as inner:
            assert current_call_scope() is inner and inner.tag == other
        assert current_call_scope() is outer
    assert current_call_scope() is None
