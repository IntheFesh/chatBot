"""R18-D6 (D-890): how long a call somebody is waiting for can take on the client (R-LLM-005).

User decision 【实时回复最坏耗时】: a live call - the reply (with the hybrid planning step and
the second judgement of a crisis), a proactive message, describing the picture the user has just
sent - is not retried by the client when it times out (the engine tries the round again later,
R-ENG-010), and its short retries of 429/5xx end within thirty seconds.  Background work keeps
the offline rules (four retries with back-off).  The client's worst case for a live call is
therefore one attempt's deadline plus thirty seconds.

Before the fix a reply that never answered was tried five times, 65 seconds each, with back-off
pauses in between: more than five minutes before the engine heard of it.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.support.clock import ManualClock
from tests.support.deepseek import (
    API,
    TEST_KEY,
    error,
    ok,
    on_clock,
    request_json,
    slow,
    timing_out,
)
from tests.support.waiting import wait_until
from twin.llm.deepseek import DEADLINE_SLACK_S
from twin.llm.errors import RetriesExhaustedError
from twin.llm.reliability import LIVE_RETRY_WINDOW_S, FailureKind, RetryPolicy
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.llm.types import LIVE_PURPOSES, ChatMessage, Purpose, is_live
from twin.services import Services

NON_THINKING_S = 60.0  # deepseek.timeout_s.non_thinking (the default settings)
THINKING_S = 180.0  # deepseek.timeout_s.thinking
QUESTION: list[ChatMessage] = [{"role": "user", "content": "在吗"}]


def user(text: str) -> ChatMessage:
    return {"role": "user", "content": text}


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def llm(services: Services) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


def deadline(thinking: bool = False) -> float:
    """The hard limit of one attempt: the configured timeout plus the client's slack."""
    return (THINKING_S if thinking else NON_THINKING_S) + DEADLINE_SLACK_S


# ------------------------------------------------------------------ which calls are live


def test_the_reply_and_the_proactive_message_are_live_and_background_work_is_not() -> None:
    """R18-D6: ``plan`` is the life line's purpose - background work - and no longer live."""
    assert {Purpose.REPLY, Purpose.PROACTIVE} == LIVE_PURPOSES
    assert is_live(Purpose.REPLY) and is_live(Purpose.PROACTIVE)
    for purpose in (Purpose.PLAN, Purpose.CAPTION, Purpose.STICKER_TAG, Purpose.SUMMARY):
        assert not is_live(purpose)
    # the picture the user has just sent: the caller says so
    assert is_live(Purpose.CAPTION, True) and is_live(Purpose.STICKER_TAG, True)
    assert not is_live(Purpose.REPLY, False)


# --------------------------------------------------------------------- the client


@pytest.mark.parametrize("purpose", ["reply", "proactive"])
async def test_a_live_call_that_times_out_is_not_retried_by_the_client(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock, purpose: str
) -> None:
    """R18-D6: one attempt, then the error goes to the engine (which retries per R-ENG-010)."""
    route = api.post(API).mock(side_effect=[timing_out(clock, NON_THINKING_S), ok(content="嗯")])
    start = clock.monotonic()
    with pytest.raises(RetriesExhaustedError, match="timed out") as info:
        await on_clock(clock, llm.client.chat(QUESTION, purpose=purpose))
    assert route.call_count == 1 and info.value.attempts == 1
    assert clock.monotonic() - start == NON_THINKING_S  # no back-off pause after it


async def test_a_picture_the_user_just_sent_is_a_live_call_when_the_caller_says_so(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    route = api.post(API).mock(side_effect=[timing_out(clock, NON_THINKING_S), ok(content="猫")])
    with pytest.raises(RetriesExhaustedError):
        await on_clock(clock, llm.client.chat(QUESTION, purpose="caption", live=True))
    assert route.call_count == 1


@pytest.mark.parametrize("purpose", ["summary", "plan", "caption", "sticker_tag"])
async def test_an_offline_call_that_times_out_is_still_retried_four_times(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock, purpose: str
) -> None:
    """R18-D6: background work - the life line (``plan``) too - keeps the offline rules."""
    route = api.post(API).mock(side_effect=[timing_out(clock, NON_THINKING_S)] * 5)
    with pytest.raises(RetriesExhaustedError) as info:
        await on_clock(clock, llm.client.chat(QUESTION, purpose=purpose))
    assert route.call_count == 5 and info.value.attempts == 5


async def test_a_live_call_still_gets_its_short_retries_of_overload(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    route = api.post(API).mock(
        side_effect=[error(503), error(429), httpx.ConnectError("reset"), ok(content="来了")]
    )
    result = await on_clock(clock, llm.client.chat(QUESTION, purpose="reply"))
    assert result.content == "来了" and result.attempts == 4 and route.call_count == 4


async def test_a_retry_after_that_does_not_fit_the_window_ends_a_live_call(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    """Twenty seconds asked for twice do not fit thirty: the second 429 is the last."""
    route = api.post(API).mock(side_effect=[error(429, **{"retry-after": "20"})] * 5)
    start = clock.monotonic()
    with pytest.raises(RetriesExhaustedError) as info:
        await on_clock(clock, llm.client.chat(QUESTION, purpose="reply"))
    assert route.call_count == 2 and info.value.status == 429
    assert clock.monotonic() - start == 20


SCENARIOS = {
    "never answers": lambda clock: [timing_out(clock, NON_THINKING_S)] * 5,
    "overloaded at once": lambda clock: [error(503)] * 5,
    "overloaded after 25 s": lambda clock: [slow(clock, 25, error(503))] * 5,
    "overloaded after 59 s": lambda clock: [slow(clock, 59, error(500))] * 5,
    "rate limited for 20 s": lambda clock: [error(429, **{"retry-after": "20"})] * 5,
    "overloaded, then silent": lambda clock: [error(503)] + [timing_out(clock, NON_THINKING_S)] * 4,
    "cut off after 20 s": lambda clock: (
        [slow(clock, 20, ok(finish="insufficient_system_resource"))] * 5
    ),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_the_worst_case_of_a_live_call_is_one_deadline_plus_thirty_seconds(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock, scenario: str
) -> None:
    """R18-D6: the assertion the user asked for, for each way the service can let us down."""
    api.post(API).mock(side_effect=SCENARIOS[scenario](clock))
    start = clock.monotonic()
    with pytest.raises(RetriesExhaustedError):
        await on_clock(clock, llm.client.chat(QUESTION, purpose="reply"))
    assert clock.monotonic() - start <= deadline() + LIVE_RETRY_WINDOW_S


async def test_a_thinking_reply_that_never_answers_ends_after_its_own_timeout(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    route = api.post(API).mock(side_effect=[timing_out(clock, THINKING_S)] * 5)
    start = clock.monotonic()
    with pytest.raises(RetriesExhaustedError):
        await on_clock(clock, llm.client.chat(QUESTION, purpose="reply", thinking=True))
    assert route.call_count == 1
    assert clock.monotonic() - start <= deadline(thinking=True) + LIVE_RETRY_WINDOW_S


async def test_the_json_re_ask_of_a_live_call_shares_its_window(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    """R18-D6: an answer that came after the window is not asked again (R-LLM-003 re-ask).

    Otherwise a slow invalid answer and a slow second one would take two deadlines.
    """
    from pydantic import BaseModel

    from twin.llm.errors import StructuredOutputError

    class Verdict(BaseModel):
        ok: bool

    route = api.post(API).mock(
        side_effect=[slow(clock, 40, ok(content="不是 JSON")), ok(content='{"ok": true}')]
    )
    with pytest.raises(StructuredOutputError):
        await on_clock(clock, llm.client.chat_json(QUESTION, Verdict, purpose="reply"))
    assert route.call_count == 1
    # inside the window the re-ask still happens, as R-LLM-003 says
    route.reset()
    route.mock(side_effect=[slow(clock, 5, ok(content="不是 JSON")), ok(content='{"ok": true}')])
    answer = await on_clock(clock, llm.client.chat_json(QUESTION, Verdict, purpose="reply"))
    assert answer.value.ok and answer.attempts == 2 and route.call_count == 2


async def test_the_life_line_waits_behind_the_slot_kept_for_the_reply(
    services: Services, api: respx.MockRouter
) -> None:
    """R18-D6 (D-890): the life line (``plan``) is background work, so it no longer takes the slot
    that is kept free for somebody who is waiting (D-701 had counted ``plan`` as live)."""
    services.settings.deepseek.max_concurrency = 2  # one slot for background work, one kept
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    runtime = build_llm_runtime(services)
    gate = asyncio.Event()
    seen: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        said = str(request_json(request)["messages"][-1]["content"])
        seen.append(said)
        if said.startswith("life"):
            await gate.wait()
        return ok(content=said)

    api.post(API).mock(side_effect=answer)
    lines = [
        asyncio.create_task(runtime.client.chat([user(f"life {i}")], purpose="plan"))
        for i in range(2)
    ]
    try:
        await wait_until(lambda: len(seen) == 1)
        for _ in range(20):
            await asyncio.sleep(0)
        assert len(seen) == 1  # the second life-line call waits for the background slot
        reply = await asyncio.wait_for(runtime.client.chat([user("live")], purpose="reply"), 5)
        assert reply.content == "live" and not any(task.done() for task in lines)
    finally:
        gate.set()
        await asyncio.gather(*lines)
        await runtime.client.aclose()
    assert len(seen) == 3


async def test_inside_an_evaluation_scope_nothing_is_live(
    services: Services, llm: LlmRuntime, api: respx.MockRouter, clock: ManualClock
) -> None:
    """The evaluation runs the reply pipeline as a batch: nobody waits, the offline rules hold."""
    from twin.llm.scope import call_scope
    from twin.llm.types import LedgerTag

    route = api.post(API).mock(side_effect=[timing_out(clock, NON_THINKING_S), ok(content="嗯")])
    with call_scope(Purpose.EVAL, LedgerTag("one_time", "eval-r18")):
        result = await on_clock(clock, llm.client.chat(QUESTION, purpose="reply", live=True))
    assert result.content == "嗯" and route.call_count == 2


# ------------------------------------------------------------------- the policy itself


KINDS = [FailureKind.RATE_LIMIT, FailureKind.SERVER, FailureKind.CONNECTION, FailureKind.TIMEOUT]


@settings(max_examples=300, deadline=None)
@given(
    attempts=st.lists(
        st.tuples(
            st.sampled_from(KINDS),
            st.floats(min_value=0.0, max_value=deadline()),
            st.one_of(st.none(), st.floats(min_value=0.0, max_value=60.0)),
        ),
        min_size=1,
        max_size=8,
    ),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_no_sequence_of_failures_takes_a_live_call_past_the_bound(
    attempts: list[tuple[FailureKind, float, float | None]], seed: int
) -> None:
    """R18-D6, any order of failures and latencies: the client gives up within the bound.

    Each attempt takes at most one deadline (``asyncio.timeout``); the policy decides whether a
    retry may start, and how long to wait before it.
    """
    policy, rng = RetryPolicy(), random.Random(seed)
    elapsed = 0.0
    for number, (kind, took, retry_after) in enumerate(attempts, start=1):
        elapsed += took
        wait = policy.next_wait(
            number, kind, rng, live=True, elapsed_s=elapsed, retry_after_s=retry_after
        )
        if wait is None:
            break
        assert kind is not FailureKind.TIMEOUT  # a live timeout is never retried
        elapsed += wait
        assert elapsed <= LIVE_RETRY_WINDOW_S  # a retry starts inside the window
    assert elapsed <= deadline() + LIVE_RETRY_WINDOW_S


def test_the_offline_policy_is_the_old_one() -> None:
    policy, rng = RetryPolicy(), random.Random(3)
    waits = [policy.next_wait(n, FailureKind.TIMEOUT, rng, elapsed_s=500.0) for n in range(1, 6)]
    assert all(wait is not None for wait in waits[:4]) and waits[4] is None
