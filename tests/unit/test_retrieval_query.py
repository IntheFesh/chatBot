"""Querying the library: ranking, MMR, time of day, recency, ``before`` (R-RET-005, R-TRN-013)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from tests.support.embedding import H, HashingBackend, U, day, write_dialogue
from twin.profile.holdout import get_holdout
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.examples import Example, render_example
from twin.retrieval.indexer import run_index, window_table
from twin.retrieval.query import (
    ExampleQuery,
    ExampleRetriever,
    QueryTurn,
    Ranked,
    mmr_select,
    recency_bonus,
    reply_signature,
    time_bonus,
)
from twin.retrieval.records import MessageData
from twin.retrieval.vector_store import IndexMismatchError
from twin.retrieval.windows import WindowRecord, apply_holdout
from twin.services import Services
from twin.storage.retrieval_models import ExampleWindow

FOOD = [
    ("今天中午吃什么饭", "吃火锅吧"),
    ("晚饭想吃点什么", "随便吃面条也行"),
    ("你吃饭了吗", "刚吃完饭"),
    ("饿了想吃东西", "去吃饭吧"),
]
STUDY = [
    ("论文写完了没有", "还没呢在改稿"),
    ("今天上课累不累", "好累作业好多"),
    ("考试复习得怎么样", "在复习书太多"),
    ("实验报告交了没", "刚交上去了"),
]
FOOD_REPLIES = {answer for _, answer in FOOD}
WORKDAY = "workday"


def configure(services: Services, backend: HashingBackend) -> None:
    services.settings.time.source_timezone = "UTC"
    services.settings.retrieval.model = backend.info.model


def retriever(services: Services, backend: HashingBackend) -> ExampleRetriever:
    return ExampleRetriever(services, EmbeddingService(backend))


def ask(
    text: str,
    *,
    minute: float = 12 * 60,
    day_type: str = WORKDAY,
    k: int | None = 4,
    before: datetime | None = None,
) -> ExampleQuery:
    return ExampleQuery([QueryTurn(False, text)], minute, day_type, before, k)


def reply_texts(examples: list[Example]) -> list[str]:
    return ["".join(line.text for line in e.reply) for e in examples]


@pytest.fixture
def lib(services: Services, embedder: HashingBackend) -> Services:
    """Thirty conversations on two topics at lunchtime; the last three are held out."""
    configure(services, embedder)
    episodes = []
    for n in range(30):
        question, answer = (FOOD if n % 2 == 0 else STUDY)[(n // 2) % 4]
        episodes.append((day(n), [U(question), H(answer)]))
    write_dialogue(services, episodes)
    run_index(services)
    return services


# ------------------------------------------------------------------ relevance


async def test_a_question_about_eating_finds_her_answers_about_eating(
    lib: Services, embedder: HashingBackend
) -> None:
    got = retriever(lib, embedder)
    best = await got.query(ask("你吃饭了吗", k=1))
    assert reply_texts(best) == ["刚吃完饭"]  # the very same question
    assert best[0].similarity == pytest.approx(1.0, abs=0.05)
    lib.settings.retrieval.mmr_lambda = 1.0  # relevance alone: the first picks are all about food
    examples = await got.query(ask("你吃饭了吗", k=4))  # four different answers about food exist
    assert set(reply_texts(examples)) == FOOD_REPLIES
    context = "".join(line.text for turn in examples[0].context for line in turn.lines)
    assert "吃" in context
    assert all(e.reply and e.reply[0].reproducible for e in examples)
    assert examples[0].score >= examples[-1].score


async def test_a_question_about_studying_finds_study_windows(
    lib: Services, embedder: HashingBackend
) -> None:
    lib.settings.retrieval.mmr_lambda = 1.0
    examples = await retriever(lib, embedder).query(ask("论文 上课累 考试复习 实验报告", k=4))
    assert set(reply_texts(examples)) == {a for _, a in STUDY}


async def test_the_bot_conversation_is_context_but_the_results_are_only_hers(
    lib: Services, embedder: HashingBackend
) -> None:
    turns = [
        QueryTurn(False, "今天吃什么"),
        QueryTurn(True, "机器人专属暗号火锅"),  # the bot's own earlier reply
        QueryTurn(False, "你吃饭了吗"),
    ]
    examples = await retriever(lib, embedder).query(ExampleQuery(turns, 720, WORKDAY, None, 8))
    assert "机器人专属暗号火锅" in embedder.seen[-1]  # it shapes the query ...
    rendered = "\n".join(render_example(e) for e in examples)
    assert "机器人专属暗号" not in rendered  # ... but is never an example
    assert reply_texts(examples) and set(reply_texts(examples)) <= {a for _, a in (*FOOD, *STUDY)}


async def test_only_the_last_context_turns_make_the_query(
    lib: Services, embedder: HashingBackend
) -> None:
    turns = [QueryTurn(i % 2 == 0, f"第{i}轮话") for i in range(10)]
    await retriever(lib, embedder).query(ExampleQuery(turns, 720, WORKDAY, None, 2))
    seen = embedder.seen[-1]
    assert "第9轮话" in seen and "第4轮话" in seen and "第3轮话" not in seen


async def test_the_query_text_is_redacted_before_encoding(
    lib: Services, embedder: HashingBackend
) -> None:
    phone = "1" + "3800138000"
    await retriever(lib, embedder).query(ask(f"我的电话是{phone}吃饭吗"))
    assert phone not in embedder.seen[-1] and "[手机号]" in embedder.seen[-1]


async def test_nothing_comes_back_from_an_empty_library_or_an_empty_question(
    services: Services, embedder: HashingBackend, lib: Services
) -> None:
    got = retriever(lib, embedder)
    assert await got.query(ask("   ")) == []
    assert await got.query(ExampleQuery([], 720, WORKDAY)) == []
    assert await got.query(ask("你吃饭了吗", k=0)) == []


async def test_a_library_that_was_never_built_answers_nothing(
    services: Services, embedder: HashingBackend
) -> None:
    assert await retriever(services, embedder).query(ask("你吃饭了吗")) == []


async def test_the_default_number_of_examples_is_the_configured_one(
    lib: Services, embedder: HashingBackend
) -> None:
    lib.settings.engine.examples_k = 5
    got = retriever(lib, embedder)
    assert len(await got.query(ask("你吃饭了吗", k=None))) == 5
    assert len(await got.query(ask("你吃饭了吗", k=3))) == 3  # the budget level asks for fewer


async def test_a_library_made_with_another_model_is_refused(
    lib: Services, embedder: HashingBackend
) -> None:
    other = HashingBackend(weights="other-weights")
    with pytest.raises(IndexMismatchError, match="twin retrieval rebuild"):
        await ExampleRetriever(lib, EmbeddingService(other)).query(ask("你吃饭了吗"))


# ------------------------------------------------------------ time and recency


def test_the_time_of_day_bonus_is_a_bell_on_the_circular_distance() -> None:
    kw = {"weight": 0.06, "sigma_slots": 8.0}
    here = time_bonus(23 * 60 + 50, "workday", 0, "workday", **kw)  # slot 0 = 00:00-00:15
    far = time_bonus(23 * 60 + 50, "workday", 48, "workday", **kw)  # noon
    near_before_midnight = time_bonus(23 * 60 + 50, "workday", 95, "workday", **kw)
    assert here > far and near_before_midnight > here > 0.9 * 0.06 * 0.85
    assert far < 0.001
    exact = time_bonus(12 * 60 + 7.5, "workday", 48, "workday", **kw)
    assert exact == pytest.approx(0.06)
    assert time_bonus(12 * 60 + 7.5, "weekend", 48, "workday", **kw) == pytest.approx(0.03)


def test_recent_windows_score_higher_with_a_half_life() -> None:
    kw = {"weight": 0.04, "half_life_days": 180.0}
    assert recency_bonus(0, **kw) == pytest.approx(0.04)
    assert recency_bonus(180, **kw) == pytest.approx(0.02)
    assert recency_bonus(360, **kw) == pytest.approx(0.01)
    assert recency_bonus(-5, **kw) == pytest.approx(0.04)  # never negative age


async def test_a_window_from_the_same_hour_wins_when_the_wording_is_equal(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    services.settings.retrieval.recency_weight = 0.0
    episodes = []
    for n, (n_day, hour, answer) in enumerate(
        [(1, 8, "早饭吃面包"), (2, 8, "早饭吃面包"), (3, 8, "早饭吃面包")]
        + [(8, 23, "夜宵吃泡面"), (9, 23, "夜宵吃泡面"), (10, 23, "夜宵吃泡面")]
        + [(11, 15, "下午吃点心")] * 5
    ):
        episodes.append((day(n_day + (n if n > 5 else 0) * 0, hour), [U("今天吃什么"), H(answer)]))
    # distinct days for the filler windows (all on workdays)
    episodes = [
        (day([1, 2, 3, 8, 9, 10, 4, 5, 15, 16, 17, 18][i], episode[0].hour), episode[1])
        for i, episode in enumerate(episodes)
    ]
    write_dialogue(services, episodes)
    run_index(services)
    got = retriever(services, embedder)
    night = await got.query(ask("今天吃什么", minute=23 * 60 + 10, k=1))
    morning = await got.query(ask("今天吃什么", minute=8 * 60 + 5, k=1))
    assert reply_texts(night) == ["夜宵吃泡面"] and reply_texts(morning) == ["早饭吃面包"]


async def test_a_newer_window_wins_when_everything_else_is_equal(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    services.settings.retrieval.slot_weight = 0.0
    days = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16]
    episodes = [
        (day(d, 12), [U("今天吃什么"), H("很早以前的回答" if d < 6 else "最近的回答")])
        for d in days
    ]
    write_dialogue(services, episodes)
    run_index(services)
    got = retriever(services, embedder)
    best = await got.query(ask("今天吃什么", k=1))
    assert reply_texts(best) == ["最近的回答"]


# ------------------------------------------------------------------ MMR / dedup


def candidate(
    ident: str, score: float, vector: list[float], reply: str, *, slot: int = 0
) -> Ranked:
    window = WindowRecord(
        id=ident,
        conversation_id="c",
        reply_block_ids=(ident,),
        context_block_ids=(),
        reply_at_utc=datetime(2026, 3, 1, tzinfo=UTC),
        local_slot=slot,
        day_type=WORKDAY,
        reply_reproducible=1,
        holdout=False,
    )
    return Ranked(window, score, score, np.array(vector, dtype=np.float32), reply)


def test_mmr_trades_relevance_against_redundancy() -> None:
    pool = [
        candidate("copy-1", 0.90, [1, 0], "甲"),
        candidate("copy-2", 0.89, [1, 0], "乙乙"),
        candidate("copy-3", 0.88, [1, 0], "丙丙丙"),
        candidate("other", 0.60, [0, 1], "丁丁丁丁"),
    ]
    relevance_only = mmr_select(pool, 3, mmr_lambda=1.0, dedup_similarity=0.9)
    assert [r.window.id for r in relevance_only] == ["copy-1", "copy-2", "copy-3"]
    diverse = mmr_select(pool, 3, mmr_lambda=0.7, dedup_similarity=0.9)
    assert [r.window.id for r in diverse][:2] == ["copy-1", "other"]  # a different one comes second
    assert len({tuple(r.vector) for r in diverse}) == 2


def test_replies_that_are_nearly_the_same_text_are_not_picked_twice() -> None:
    pool = [
        candidate("a", 0.95, [1, 0, 0], "好的好的好的好的好的好的好的好的好的好的"),
        candidate("b", 0.94, [0, 1, 0], "好的好的好的好的好的好的好的好的好的好的呀"),
        candidate("c", 0.50, [0, 0, 1], "明天再说吧"),
    ]
    chosen = mmr_select(pool, 3, mmr_lambda=0.7, dedup_similarity=0.9)
    assert [r.window.id for r in chosen] == ["a", "c"]  # b is 95 % the same text as a
    loose = mmr_select(pool, 3, mmr_lambda=0.7, dedup_similarity=0.99)
    assert {r.window.id for r in loose} == {"a", "b", "c"}


def test_empty_replies_are_never_taken_for_duplicates() -> None:
    pool = [candidate("a", 0.9, [1, 0], ""), candidate("b", 0.8, [0, 1], "")]
    assert len(mmr_select(pool, 2, mmr_lambda=0.7, dedup_similarity=0.9)) == 2
    assert mmr_select([], 3, mmr_lambda=0.7, dedup_similarity=0.9) == []
    assert mmr_select(pool, 0, mmr_lambda=0.7, dedup_similarity=0.9) == []


async def test_many_copies_of_one_situation_do_not_fill_the_examples(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    episodes = []
    days = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]
    for i, d in enumerate(days):
        if i < 8:
            episodes.append((day(d), [U("你吃饭了吗"), H(f"吃过了第{i}号回答，真的吃过了")]))
        else:
            episodes.append(
                (day(d), [U("今天吃饭了没"), H(f"别的回答{'一二三四五六七八'[i - 8]}说法不一样")])
            )
    write_dialogue(services, episodes)
    run_index(services)
    got = retriever(services, embedder)
    services.settings.retrieval.dedup_similarity = 1.0  # isolate MMR from the text dedup
    services.settings.retrieval.mmr_lambda = 1.0
    plain = await got.query(ask("你吃饭了吗", k=4))
    services.settings.retrieval.mmr_lambda = 0.5
    varied = await got.query(ask("你吃饭了吗", k=4))

    def contexts(examples: list[Example]) -> set[str]:
        return {line.text for e in examples for turn in e.context for line in turn.lines}

    assert len(contexts(plain)) == 1  # four copies of the same question
    assert len(contexts(varied)) == 2  # variety: the other wording comes in


def test_a_reply_signature_ignores_events() -> None:
    def msg(kind: str, text: str | None, md5: str | None = None) -> MessageData:
        return MessageData("m", kind, False, text, None, md5, None, None, None, False)

    assert reply_signature(
        [msg("text", "好 的"), msg("image", None), msg("sticker", None, "ab" * 16)]
    ) == ("好 的\n[表情包:" + "ab" * 16 + "]")


# ----------------------------------------------------------------- as-of / hold-out


def stored(services: Services) -> list[WindowRecord]:
    with services.db.session() as session:
        rows = session.scalars(select(ExampleWindow).order_by(ExampleWindow.reply_at_utc))
        return [WindowRecord.from_row(row) for row in rows]


async def test_before_only_returns_windows_that_happened_earlier(
    lib: Services, embedder: HashingBackend
) -> None:
    lib.settings.retrieval.dedup_similarity = 1.0  # the library repeats replies on purpose
    found = stored(lib)
    boundary = found[12].reply_at_utc
    got = retriever(lib, embedder)
    everything = await got.query(ask("你吃饭了吗", k=30))
    earlier = await got.query(ask("你吃饭了吗", k=30, before=boundary))
    assert earlier and all(e.reply_at < boundary for e in earlier)
    assert any(e.reply_at >= boundary for e in everything)  # the filter did something
    assert {e.window_id for e in earlier} <= {w.id for w in found[:12]}
    # a window exactly at ``before`` is not before it
    assert found[12].id not in {e.window_id for e in earlier}
    just_after = await got.query(ask("你吃饭了吗", k=30, before=boundary + timedelta(seconds=1)))
    assert found[12].id in {e.window_id for e in just_after}


async def test_a_before_without_a_time_zone_is_refused(
    lib: Services, embedder: HashingBackend
) -> None:
    naive = datetime(2026, 3, 10, 12, 0)  # noqa: DTZ001 - the point of the test
    with pytest.raises(ValueError, match="naive datetime"):
        await retriever(lib, embedder).query(ask("你吃饭了吗", before=naive))


async def test_the_held_out_period_is_returned_only_for_moments_after_it(
    lib: Services, embedder: HashingBackend
) -> None:
    """Rewritten in round 18 (user decision, D-880): the held-out windows are in the library; a
    moment after them sees them, a moment at the cut-off sees none of them."""
    lib.settings.retrieval.dedup_similarity = 1.0
    holdout = get_holdout(lib)
    assert holdout is not None
    got = retriever(lib, embedder)
    future = holdout.cutoff + timedelta(days=100)
    examples = await got.query(ask("你吃饭了吗", k=30, before=future))
    assert len(examples) == 30 and sum(e.reply_at >= holdout.cutoff for e in examples) == 3
    at_the_cutoff = await got.query(ask("你吃饭了吗", k=30, before=holdout.cutoff))
    assert len(at_the_cutoff) == 27 and all(e.reply_at < holdout.cutoff for e in at_the_cutoff)


async def test_moving_the_holdout_flags_moves_no_vector(
    lib: Services, embedder: HashingBackend
) -> None:
    """Rewritten in round 18 (D-880): the flags are information; moving them keeps every vector,
    the running bot still finds every window and ``before`` alone decides for a past moment."""
    lib.settings.retrieval.dedup_similarity = 1.0
    found = stored(lib)
    versions = {w.id: w.embed_version for w in found}
    new_cutoff = found[20].reply_at_utc
    apply_holdout(lib, new_cutoff)
    assert window_table(lib).count() == 30
    after = stored(lib)
    assert {w.id: w.embed_version for w in after} == versions
    assert sum(w.holdout for w in after) == 10
    live = await retriever(lib, embedder).query(ask("你吃饭了吗", k=30))
    assert len(live) == 30
    past = await retriever(lib, embedder).query(ask("你吃饭了吗", k=30, before=new_cutoff))
    assert past and all(e.reply_at < new_cutoff for e in past)


# ------------------------------------- the held-out period in the library (R18-C3, D-880)


def question_of(episode: int) -> str:
    """The user's question of episode ``episode`` of the ``lib`` fixture."""
    return (FOOD if episode % 2 == 0 else STUDY)[(episode // 2) % 4][0]


def held_out_windows(services: Services) -> list[tuple[int, WindowRecord]]:
    """``(episode, window)`` of the windows at or after the stored cut-off (the last three)."""
    return [(n, w) for n, w in enumerate(stored(services)) if w.holdout]


async def test_r18_c3_the_held_out_windows_are_in_the_library_for_the_running_bot(
    lib: Services, embedder: HashingBackend
) -> None:
    """R18-C3 ① (user decision, D-880): the library holds every window, the held-out period
    included; the running bot (no ``before``) finds them like any other."""
    lib.settings.retrieval.dedup_similarity = 1.0
    held = held_out_windows(lib)
    assert len(held) == 3
    indexed = {row["window_id"] for row in window_table(lib).rows()}
    assert {w.id for _, w in held} <= indexed
    assert window_table(lib).count() == 30
    for episode, window in held:
        found = await retriever(lib, embedder).query(ask(question_of(episode), k=30))
        assert window.id in {e.window_id for e in found}


async def test_r18_c3_a_held_out_window_is_never_found_at_or_before_its_own_moment(
    lib: Services, embedder: HashingBackend
) -> None:
    """R18-C3 ① injection test (D-880): a held-out window is in the library, but a view of its own
    sample moment - the evaluation sandbox and the training export read ``AsOfView(t)``, whose
    examples carry ``before=t`` - never returns it, nor a view of any earlier moment."""
    from twin.memory.asof import AsOfSource

    lib.settings.retrieval.dedup_similarity = 1.0
    got = retriever(lib, embedder)
    source = AsOfSource(lib, retriever=got)
    for episode, window in held_out_windows(lib):
        question = question_of(episode)  # the very context of the window: it would rank first
        moment = window.reply_at_utc
        for before in (
            moment,
            moment - timedelta(microseconds=1),
            moment - timedelta(seconds=1),
            moment - timedelta(hours=1),
            moment - timedelta(days=3),
        ):
            found = await got.query(ask(question, k=30, before=before))
            assert found and window.id not in {e.window_id for e in found}
            assert all(e.reply_at < before for e in found)
        examples = await source.at(moment).examples([QueryTurn(False, question)], k=30)
        assert examples and window.id not in {e.window_id for e in examples}
        assert all(e.reply_at < moment for e in examples)
        later = await got.query(ask(question, k=30, before=moment + timedelta(seconds=1)))
        assert window.id in {e.window_id for e in later}  # the window is there to be found
    await got.aclose()


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(data=st.data())
def test_r18_c3_no_moment_up_to_a_window_reaches_it(
    lib: Services, embedder: HashingBackend, data: st.DataObject
) -> None:
    """R18-C3 ① hypothesis variant (D-880): for any window of the library - held out or not - and
    any moment at or before its reply, a query with that ``before`` never returns it."""
    lib.settings.retrieval.dedup_similarity = 1.0
    windows = stored(lib)
    episode = data.draw(st.integers(min_value=0, max_value=len(windows) - 1), label="episode")
    window = windows[episode]
    back = data.draw(
        st.floats(min_value=0.0, max_value=40 * 86_400.0, allow_nan=False), label="seconds back"
    )
    before = window.reply_at_utc - timedelta(seconds=back)
    ranked = retriever(lib, embedder).rank(ask(question_of(episode), k=30, before=before))
    assert window.id not in {r.window.id for r in ranked}
    assert all(r.window.reply_at_utc < before for r in ranked)


async def test_windows_without_a_row_are_skipped(lib: Services, embedder: HashingBackend) -> None:
    from sqlalchemy import delete

    victim = stored(lib)[0]
    with lib.db.transaction(bump_state=False) as session:
        session.execute(delete(ExampleWindow).where(ExampleWindow.id == victim.id))
    examples = await retriever(lib, embedder).query(ask("你吃饭了吗", k=30))
    assert victim.id not in {e.window_id for e in examples}


async def test_replies_of_only_events_rank_below_replies_with_words(
    services: Services, embedder: HashingBackend
) -> None:
    configure(services, embedder)
    episodes = []
    days = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16]
    for i, d in enumerate(days):
        reply = [H(None, kind="image")] if i == 0 else [H("吃过了")]
        episodes.append((day(d), [U("你吃饭了吗"), *reply]))
    write_dialogue(services, episodes)
    run_index(services)
    got = retriever(services, embedder)
    services.settings.retrieval.mmr_lambda = 1.0
    services.settings.retrieval.dedup_similarity = 1.0
    ranked = got.rank(ask("你吃饭了吗", k=12))
    event_only = [r for r in ranked if r.window.reply_reproducible == 0]
    assert len(event_only) == 1
    words = [r for r in ranked if r.window.reply_reproducible > 0]
    assert event_only[0].score < min(r.score for r in words) - 0.2  # multiplied by 0.6
    assert ranked[-1] is event_only[0]
