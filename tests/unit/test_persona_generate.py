"""Generating the automatic description: Map-Reduce with checked evidence
(R-PERS-001, R-PERS-005, R-LLM-003, R-LLM-009, R-TRN-013)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY, ok, request_json
from tests.support.persona import message_ids_after, sticker_scenario
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.llm.types import LedgerTag
from twin.profile.builder import rebuild
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona.generate import (
    ITEM_CHARS,
    MAX_ITEMS,
    Digest,
    EmotionItem,
    Item,
    PersonaGenerationError,
    batches_of,
    canonical_label,
    clean_digest,
    description_block,
    generate_digest,
    merge_digests,
)
from twin.profile.persona.jobs import generate_scope
from twin.profile.persona.refresh import count_her_messages
from twin.profile.persona.sampling import SampleRequest, build_sample
from twin.profile.persona.sections import AUTO, split_card
from twin.profile.persona.store import PersonaStore
from twin.profile.prompt_templates import PERSONA_MAP, PERSONA_REDUCE, TemplateStore
from twin.services import Services
from twin.storage.chat_models import Message

BOGUS = "她在国外读过博士学位"


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def labels_in(request: httpx.Request) -> list[str]:
    content = request_json(request)["messages"][1]["content"]
    return re.findall(r"【片段 (S\d+)】", content)


def digest_json(**lists: Any) -> str:
    return json.dumps(lists, ensure_ascii=False)


def mapper(request: httpx.Request) -> httpx.Response:
    """A model that answers a map call truthfully, plus the mistakes the program must catch."""
    labels = labels_in(request)
    first, second = labels[0], labels[-1]
    return ok(
        content=digest_json(
            tone=[
                {"text": "说话轻快", "evidence": [first, second]},
                {"text": BOGUS, "evidence": ["S99"]},  # no such segment
                {"text": "没有任何证据的一句", "evidence": []},
            ],
            catchphrases=[{"text": f"口头禅来自{first}", "evidence": [first.lower()]}],
            emotions=[
                {"emotion": "开心", "text": "哇塞好开心", "evidence": [first]},
                {"emotion": "无聊", "text": "不在六种之内", "evidence": [first]},
            ],
            facts=[{"text": f"在{first}里提到过一只猫", "evidence": [first]}],
        ),
        prompt=400,
        completion_tokens=80,
    )


def reducer(request: httpx.Request) -> httpx.Response:
    return ok(
        content=digest_json(
            tone=[
                {"text": "说话轻快", "evidence": ["S01", "S2"]},
                {"text": "合并时编出来的", "evidence": ["S77"]},
            ],
            facts=[{"text": "养了一只猫", "evidence": ["S01"]}],
            emotions=[{"emotion": "撒娇", "text": "嘛嘛", "evidence": ["S02"]}],
        ),
        prompt=900,
        completion_tokens=60,
    )


def router_answers(api: respx.MockRouter) -> respx.Route:
    def answer(request: httpx.Request) -> httpx.Response:
        prompt = request_json(request)["messages"][1]["content"]
        return reducer(request) if "份归纳结果" in prompt else mapper(request)

    return api.post(API).mock(side_effect=answer)


@pytest.fixture
def library(services: Services) -> Services:
    sticker_scenario(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return services


# --------------------------------------------------------------- evidence


def test_segment_labels_are_compared_by_their_number() -> None:
    assert canonical_label("S03") == canonical_label("s3") == canonical_label(" S003 ") == "3"
    assert canonical_label("X3") is None and canonical_label("S") is None


def test_statements_without_valid_evidence_are_dropped() -> None:
    digest = Digest(
        tone=[
            Item(text="  好  ", evidence=["s1", "S01", "S7"]),
            Item(text="无证据", evidence=[]),
            Item(text="假证据", evidence=["S9"]),
            Item(text="", evidence=["S1"]),
        ],
        emotions=[
            EmotionItem(emotion="撒娇时", text="嘛", evidence=["S2"]),
            EmotionItem(emotion="无聊", text="x", evidence=["S2"]),
        ],
    )
    result = clean_digest(digest, ["S01", "S02", "S03"])
    assert [(i.text, i.evidence) for i in result.digest.tone] == [("好", ["S01"])]
    assert [(e.emotion, e.evidence) for e in result.digest.emotions] == [("撒娇", ["S02"])]
    assert result.dropped == 4 and result.digest.count == 2


def test_digests_are_merged_and_batched() -> None:
    one = Digest(tone=[Item(text="a", evidence=["S1"])])
    two = Digest(tone=[Item(text="b", evidence=["S2"])], facts=[Item(text="c", evidence=["S2"])])
    merged = merge_digests([one, two])
    assert [i.text for i in merged.tone] == ["a", "b"] and merged.count == 3
    assert batches_of(list(range(25)), 10) == [
        list(range(10)),
        list(range(10, 20)),
        list(range(20, 25)),
    ]  # type: ignore[arg-type]


def test_the_description_text_has_labels_limits_and_no_duplicates() -> None:
    long_text = "很长" * 200
    digest = Digest(
        tone=[Item(text="轻快", evidence=["S1"]), Item(text="轻快", evidence=["S2"])],
        catchphrases=[Item(text=f"口头禅{i}", evidence=["S1"]) for i in range(20)],
        address_terms=[Item(text="宝宝", evidence=["S1"])],
        emotions=[EmotionItem(emotion="开心", text="哇", evidence=["S1"])],
        taboos=[Item(text="不说教", evidence=["S1"])],
        facts=[Item(text=long_text, evidence=["S3"])],
        topics=[Item(text="猫", evidence=["S3"])],
        attitude=[Item(text="亲近", evidence=["S3"])],
    )
    body, statements = description_block(digest)
    lines = body.splitlines()
    assert lines[0] == "### 风格" and "### 基本情况" in lines
    assert lines.count("- 语气：轻快") == 1
    assert sum(1 for line in lines if line.startswith("- 口头禅：")) == MAX_ITEMS
    assert "- 开心时：哇" in lines and "- 称呼：宝宝" in lines and "- 说话禁忌：不说教" in lines
    fact = next(line for line in lines if line.startswith("- 事实："))
    assert len(fact) <= len("- 事实：") + ITEM_CHARS and fact.endswith("…")
    assert lines.index("- 事实：" + long_text[: ITEM_CHARS - 1] + "…") < lines.index("- 话题：猫")
    assert statements[0] == {"part": "风格", "label": "语气", "text": "轻快", "evidence": ["S1"]}
    assert len(statements) == len([line for line in lines if line.startswith("- ")])


# ------------------------------------------------------------- the model calls


async def test_map_calls_per_batch_then_one_reduce_with_every_statement_checked(
    library: Services, api: respx.MockRouter
) -> None:
    route = router_answers(api)
    runtime = build_llm_runtime(library)
    sample = build_sample(library, SampleRequest("live", 7, 25, 40))
    templates = TemplateStore(library.db, library.clock)
    result = await generate_digest(
        runtime.client,
        sample,
        map_template=templates.active(PERSONA_MAP),
        reduce_template=templates.active(PERSONA_REDUCE),
        batch_size=10,
    )
    assert route.call_count == 4 == result.map_calls + 1
    assert [len(batch) for batch in result.batches] == [10, 10, 5]
    bodies = [request_json(call.request) for call in route.calls]
    # persona purpose: the offline model, no thinking, one system and one user message
    assert {b["model"] for b in bodies} == {"deepseek-flash"}
    assert all([m["role"] for m in b["messages"]] == ["system", "user"] for b in bodies)
    # the map calls show the segments ten at a time; the reduce call the checked map output
    assert [len(labels_in(call.request)) for call in route.calls[:3]] == [10, 10, 5]
    reduce_prompt = bodies[3]["messages"][1]["content"]
    assert "说话轻快" in reduce_prompt and BOGUS not in reduce_prompt
    assert "没有任何证据的一句" not in reduce_prompt and "不在六种之内" not in reduce_prompt
    # the answer of the reduce call is checked again against all segment labels
    texts = [i.text for i in result.digest.tone]
    assert texts == ["说话轻快"] and "合并时编出来的" not in texts
    assert [e.emotion for e in result.digest.emotions] == ["撒娇"]
    assert result.dropped == 3 * 3 + 1 and result.cost_usd > 0
    for call in route.calls:  # the redaction layer runs on every outgoing call
        assert request_json(call.request)["messages"][0]["role"] == "system"


async def test_one_batch_needs_no_reduce(library: Services, api: respx.MockRouter) -> None:
    route = router_answers(api)
    runtime = build_llm_runtime(library)
    sample = build_sample(library, SampleRequest("live", 7, 8, 40))
    templates = TemplateStore(library.db, library.clock)
    result = await generate_digest(
        runtime.client,
        sample,
        map_template=templates.active(PERSONA_MAP),
        reduce_template=templates.active(PERSONA_REDUCE),
        batch_size=10,
    )
    assert route.call_count == 1 and result.map_calls == 1
    assert result.digest.count == 4


async def test_a_model_that_proves_nothing_fails_instead_of_writing_a_card(
    library: Services, api: respx.MockRouter
) -> None:
    api.post(API).mock(
        return_value=ok(content=digest_json(tone=[{"text": "无证据", "evidence": ["S50"]}]))
    )
    runtime = build_llm_runtime(library)
    sample = build_sample(library, SampleRequest("live", 7, 8, 40))
    templates = TemplateStore(library.db, library.clock)
    with pytest.raises(PersonaGenerationError, match="valid evidence"):
        await generate_digest(
            runtime.client,
            sample,
            map_template=templates.active(PERSONA_MAP),
            reduce_template=templates.active(PERSONA_REDUCE),
            batch_size=10,
        )


async def test_nothing_to_describe_from_is_an_error(services: Services) -> None:
    from twin.profile.persona.sampling import Sample

    runtime = build_llm_runtime(services)
    empty = Sample("live", None, (), 0, 1)
    templates = TemplateStore(services.db, services.clock)
    with pytest.raises(PersonaGenerationError, match="no conversation segments"):
        await generate_digest(
            runtime.client,
            empty,
            map_template=templates.active(PERSONA_MAP),
            reduce_template=templates.active(PERSONA_REDUCE),
            batch_size=10,
        )


async def test_an_invalid_reply_is_sent_back_once_and_the_second_reply_is_used(
    library: Services, api: respx.MockRouter
) -> None:
    replies = iter([ok(content="这不是 JSON"), mapper])

    def answer(request: httpx.Request) -> httpx.Response:
        nxt = next(replies)
        return nxt if isinstance(nxt, httpx.Response) else nxt(request)

    route = api.post(API).mock(side_effect=answer)
    runtime = build_llm_runtime(library)
    sample = build_sample(library, SampleRequest("live", 7, 8, 40))
    templates = TemplateStore(library.db, library.clock)
    result = await generate_digest(
        runtime.client,
        sample,
        map_template=templates.active(PERSONA_MAP),
        reduce_template=templates.active(PERSONA_REDUCE),
        batch_size=10,
    )
    assert route.call_count == 2 and result.digest.count == 4
    retry = request_json(route.calls[1].request)["messages"]
    assert retry[-1]["role"] == "user" and "could not be used" in retry[-1]["content"]


# --------------------------------------------------------------- the card


async def test_a_live_run_writes_a_card_version_with_its_evidence(
    library: Services, api: respx.MockRouter
) -> None:
    rebuild(library, "all")
    router_answers(api)
    runtime = build_llm_runtime(library)
    card_id = await generate_scope(
        library, runtime, "live", seed=11, tag=LedgerTag("one_time", "persona-test")
    )
    store = PersonaStore(library.db, library.clock)
    card = store.get(card_id)
    assert card is not None and card.reason == "generate" and card.number == 1
    assert card.template_version == "persona_map@1,persona_reduce@1"
    assert card.described_her_messages == count_her_messages(library, "live") == 91
    body = split_card(card.text).body(AUTO)
    assert "- 语气：说话轻快" in body and "- 撒娇时：嘛嘛" in body and "养了一只猫" in body
    assert BOGUS not in card.text
    record = store.provenance(card_id)
    assert record is not None and record["scope"] == "live" and record["seed"] == 11
    assert record["cutoff"] is None and record["dropped_without_evidence"] > 0
    assert all(s["evidence"] for s in record["statements"])
    sampled_ids = {i for seg in record["segments"].values() for i in seg["messages"]}
    with library.db.session() as session:
        stored = set(session.scalars(select(Message.id)))
    assert sampled_ids and sampled_ids <= stored
    # the cost went to the one-time account of the batch, not to the daily budget
    assert runtime.ledger.batch_spent_usd("persona-test") > 0


async def test_the_past_card_is_made_without_a_single_message_from_the_held_out_period(
    library: Services, api: respx.MockRouter
) -> None:
    rebuild(library, "all")
    route = router_answers(api)
    cutoff = holdout_cutoff(library)
    runtime = build_llm_runtime(library)
    card_id = await generate_scope(library, runtime, "pre_holdout", seed=2, tag=LedgerTag())
    store = PersonaStore(library.db, library.clock)
    record = store.provenance(card_id)
    assert record is not None and record["cutoff"] == cutoff.isoformat()
    sampled_ids = {i for seg in record["segments"].values() for i in seg["messages"]}
    late = message_ids_after(library, cutoff)
    assert sampled_ids and late and sampled_ids.isdisjoint(late)
    with library.db.session() as session:
        late_texts = [m.text for m in session.scalars(select(Message)) if m.id in late and m.text]
    sent = "\n".join(
        json.dumps(request_json(call.request), ensure_ascii=False) for call in route.calls
    )
    assert late_texts and not any(text in sent for text in late_texts)
    card = store.get(card_id)
    assert card is not None and card.scope == "pre_holdout"
    assert split_card(card.text).names() == ("[自动-统计规则]", "[自动-描述]", "[手动]")
