"""Her real openings, found by the time of day, for the proactive messages (R-PRO-006)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from tests.support.embedding import H, HashingBackend, U, day, write_dialogue
from twin.profile.holdout import get_holdout
from twin.retrieval.indexer import run_index
from twin.retrieval.openers import OpenerExamples
from twin.services import Services
from twin.storage.retrieval_models import ExampleWindow

MORNING = ["早啊刚醒", "起床啦今天好冷", "早上好呀", "醒了醒了", "睡醒了好困"]
NIGHT = ["晚安啦先睡了", "困死了要睡觉", "洗完澡准备睡了", "今天好累想睡", "我去睡觉了"]
REPLIES = ["嗯嗯是的", "好呀一起去", "还没呢在写", "刚吃完饭"]
WORKDAY = "workday"


def configure(services: Services, backend: HashingBackend) -> None:
    services.settings.time.source_timezone = "UTC"
    services.settings.retrieval.model = backend.info.model


@pytest.fixture
def lib(services: Services, embedder: HashingBackend) -> Services:
    """Forty days: she opens at 07:30 (morning) or 21:30 (night), and also answers him at noon."""
    configure(services, embedder)
    episodes = []
    for n in range(40):
        episodes.append((day(n, 7, 30), [H(MORNING[n % 5]), U("早"), H("今天有课吗")]))
        episodes.append((day(n, 12, 0), [U(f"问题{n}号：你在干嘛"), H(REPLIES[n % 4])]))
        episodes.append((day(n, 21, 30), [H(NIGHT[n % 5]), U("晚安")]))
    write_dialogue(services, episodes)
    run_index(services)
    return services


def opener(services: Services, seed: int = 1) -> OpenerExamples:
    return OpenerExamples(services, rng=random.Random(seed))


def replies(found: list[object]) -> list[str]:
    return ["".join(line.text for line in e.reply) for e in found]  # type: ignore[attr-defined]


def test_the_library_has_windows_without_context_for_the_openings(lib: Services) -> None:
    with lib.db.session() as session:
        rows = list(session.scalars(select(ExampleWindow).where(ExampleWindow.context_turns == 0)))
        with_context = list(
            session.scalars(select(ExampleWindow).where(ExampleWindow.context_turns > 0))
        )
    assert len(rows) >= 40 and with_context, "openings have no context, answers do"


def test_only_openings_are_taken_never_answers(lib: Services) -> None:
    ranked = opener(lib).rank(local_minute=12 * 60, day_type=WORKDAY, k=8, now=day(60), before=None)
    assert ranked
    assert all(record.context_turns == 0 for record, _ in ranked)


async def test_the_examples_are_the_openings_of_the_time_of_day(lib: Services) -> None:
    now = day(60)
    morning = await opener(lib).query(local_minute=7 * 60 + 30, day_type=WORKDAY, now=now, k=4)
    night = await opener(lib).query(local_minute=21 * 60 + 30, day_type=WORKDAY, now=now, k=4)
    assert morning and night
    assert set(replies(morning)) <= set(MORNING) and set(replies(night)) <= set(NIGHT)
    assert all(e.context == () for e in morning + night)  # nobody spoke before her


async def test_no_two_examples_say_nearly_the_same(lib: Services) -> None:
    lib.settings.retrieval.dedup_similarity = 0.8
    found = await opener(lib).query(local_minute=7 * 60 + 30, day_type=WORKDAY, now=day(60), k=5)
    texts = replies(found)
    assert len(texts) == len(set(texts)) and len(texts) >= 3


async def test_the_draw_among_the_best_varies_with_the_generator(lib: Services) -> None:
    seen: set[tuple[str, ...]] = set()
    for seed in range(12):
        found = await opener(lib, seed).query(
            local_minute=7 * 60 + 30, day_type=WORKDAY, now=day(60), k=2
        )
        seen.add(tuple(sorted(replies(found))))
    assert len(seen) > 1  # two messages at the same hour do not get the same examples every time


async def test_a_before_keeps_the_examples_to_what_happened_earlier(lib: Services) -> None:
    cutoff = day(10)
    ranked = opener(lib).rank(
        local_minute=7 * 60 + 30, day_type=WORKDAY, k=6, now=day(60), before=cutoff
    )
    assert ranked and all(record.reply_at_utc < cutoff for record, _ in ranked)


async def test_the_held_out_period_is_never_used(lib: Services) -> None:
    holdout = get_holdout(lib)
    assert holdout is not None
    future = holdout.cutoff + timedelta(days=200)
    ranked = opener(lib).rank(
        local_minute=7 * 60 + 30, day_type=WORKDAY, k=40, now=future, before=future
    )
    assert ranked and all(record.reply_at_utc < holdout.cutoff for record, _ in ranked)
    assert all(not record.holdout for record, _ in ranked)


async def test_nothing_is_asked_for_nothing(lib: Services) -> None:
    now: datetime = day(60)
    assert opener(lib).rank(local_minute=450, day_type=WORKDAY, k=0, now=now) == []
    assert await opener(lib).query(local_minute=450, day_type=WORKDAY, now=now, k=0) == []


async def test_an_empty_library_has_no_openings(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    found = await opener(services).query(local_minute=450, day_type=WORKDAY, now=day(60), k=3)
    assert found == []


async def test_the_default_number_is_the_configured_one(lib: Services) -> None:
    lib.settings.engine.examples_k = 2
    found = await opener(lib).query(local_minute=450, day_type=WORKDAY, now=day(60))
    assert len(found) == 2


async def test_the_examples_builder_can_be_closed(lib: Services) -> None:
    await opener(lib).aclose()
