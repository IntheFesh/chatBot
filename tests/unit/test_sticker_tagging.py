"""Tagging stickers by their picture and by her use (R-STK-003, R-LLM-003, R-LLM-004, R-TRN-013)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY, ok, request_json
from tests.support.persona import Scenario, message_ids_after, sticker_scenario
from twin.llm.capabilities import LlmCapabilities, save_capabilities
from twin.llm.errors import ImageError, StructuredOutputError
from twin.llm.images import ImageInput
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.profile.holdout import holdout_cutoff
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.tagging import (
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    MARKER,
    ContextReply,
    StickerTagger,
    VisionReply,
    context_due,
    context_of_use,
    her_use_counts_before,
    her_uses_before,
    one_line,
    reply_model,
    spread,
    uses_text,
)
from twin.stickers.tags import load_vocabulary
from twin.storage.chat_models import Message, Sticker


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def scenario(services: Services) -> Scenario:
    found = sticker_scenario(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return found


def tagger_of(services: Services) -> StickerTagger:
    return StickerTagger(services, build_llm_runtime(services).client)


def vision_reply(**fields: Any) -> httpx.Response:
    body = {"tags": ["开心"], "description": "一只笑着的猫", "use_cases": "回应好消息"} | fields
    return ok(content=json.dumps(body, ensure_ascii=False), prompt=300, completion_tokens=40)


def context_reply(**fields: Any) -> httpx.Response:
    body = {"tags": ["委屈"], "meaning": "她用它表示有点委屈"} | fields
    return ok(content=json.dumps(body, ensure_ascii=False), prompt=600, completion_tokens=30)


# --------------------------------------------------------------- the picture


async def test_the_picture_goes_to_the_vision_model_with_detail_low(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision_reply())
    tagger = tagger_of(services)
    record = StickerCatalog(services).require(scenario.md5s[0])
    seen = await tagger.see(tagger.image_of(record))
    assert (seen.tags, seen.description, seen.use_cases) == (["开心"], "一只笑着的猫", "回应好消息")
    body = request_json(route.calls[0].request)
    assert body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    parts = body["messages"][1]["content"]
    image = next(p for p in parts if p["type"] == "image_url")["image_url"]
    assert image["url"].startswith("data:image/png;base64,") and image["detail"] == "low"
    text = next(p for p in parts if p["type"] == "text")["text"]
    assert "开心、大笑、撒娇" in text and "JSON" in text  # the closed vocabulary is in the prompt


async def test_detail_is_left_out_when_the_probe_found_it_unsupported(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    with services.db.transaction() as session:
        save_capabilities(
            session,
            LlmCapabilities(detail_supported=False, measured_at="2026-10-09T00:00:00"),
            services.clock,
        )
    route = api.post(API).mock(return_value=vision_reply())
    tagger = tagger_of(services)
    await tagger.see(tagger.image_of(StickerCatalog(services).require(scenario.md5s[0])))
    parts = request_json(route.calls[0].request)["messages"][1]["content"]
    assert "detail" not in next(p for p in parts if p["type"] == "image_url")["image_url"]


async def test_a_tag_outside_the_vocabulary_is_sent_back_and_asked_for_again(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    replies = iter([vision_reply(tags=["超级开心"]), vision_reply(tags=["大笑", "开心"])])
    route = api.post(API).mock(side_effect=lambda request: next(replies))
    tagger = tagger_of(services)
    seen = await tagger.see(tagger.image_of(StickerCatalog(services).require(scenario.md5s[0])))
    assert seen.tags == ["大笑", "开心"] and route.call_count == 2
    retry = request_json(route.calls[1].request)["messages"]
    assert "超级开心" in retry[-2]["content"]  # the reply that was wrong
    asked_again = "".join(part["text"] for part in retry[-1]["content"] if part["type"] == "text")
    assert "not allowed" in asked_again and "开心、大笑" in asked_again


@pytest.mark.parametrize(
    "bad",
    [
        {"tags": []},
        {"tags": ["开心"], "description": "  "},
        {"tags": "开心", "description": "x"},
    ],
)
async def test_two_unusable_replies_fail_the_call(
    scenario: Scenario, services: Services, api: respx.MockRouter, bad: dict[str, Any]
) -> None:
    api.post(API).mock(return_value=vision_reply(**bad))
    tagger = tagger_of(services)
    with pytest.raises(StructuredOutputError):
        await tagger.see(tagger.image_of(StickerCatalog(services).require(scenario.md5s[0])))


async def test_more_than_three_tags_are_cut_to_three(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=vision_reply(tags=["开心", "大笑", "调皮", "害羞"]))
    tagger = tagger_of(services)
    seen = await tagger.see(tagger.image_of(StickerCatalog(services).require(scenario.md5s[0])))
    assert seen.tags == ["开心", "大笑", "调皮"]


async def test_the_description_is_redacted_and_shortened_before_it_is_kept(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    phone = "1" + "38" + "12345678"
    api.post(API).mock(
        return_value=vision_reply(
            description=f"招牌上写着电话 {phone}\n" + "很长" * 200, use_cases=f"打给{phone}"
        )
    )
    assert await tagger_of(services).tag_sticker(scenario.md5s[0]) is True
    record = StickerCatalog(services).require(scenario.md5s[0])
    assert record.description is not None and phone not in record.description
    assert "[手机号]" in record.description and "\n" not in record.description
    assert len(record.description) <= 200 and record.description.endswith("…")
    assert record.use_cases == "打给[手机号]"
    assert record.tags == ("开心",) and record.tag_source == "vision"


async def test_a_sticker_is_tagged_once_and_a_missing_file_is_an_image_error(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision_reply())
    tagger = tagger_of(services)
    md5 = scenario.md5s[0]
    assert await tagger.tag_sticker(md5) is True
    assert await tagger.tag_sticker(md5) is False and route.call_count == 1
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Sticker, scenario.md5s[1])
        assert row is not None
        row.sha256 = None
    with pytest.raises(ImageError, match="no stored file"):
        await tagger.tag_sticker(scenario.md5s[1])
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Sticker, scenario.md5s[2])
        assert row is not None
        row.status = "pending"
    assert await tagger.tag_sticker(scenario.md5s[2]) is False


# ------------------------------------------------------------------ her use


def test_her_uses_before_the_cutoff_are_counted_separately_from_later_ones(
    scenario: Scenario, services: Services
) -> None:
    cutoff = holdout_cutoff(services)
    a, b, c, d = scenario.md5s
    assert len(her_uses_before(services, a, cutoff)) == 4  # episodes 2, 5, 9, 14; 38 is later
    assert len(her_uses_before(services, b, cutoff)) == 3
    assert len(her_uses_before(services, d, cutoff)) == 0
    assert her_use_counts_before(services, cutoff, 3) == {a: 4, b: 3}
    assert her_use_counts_before(services, cutoff, 1) == {a: 4, b: 3, c: 1}
    times = [moment for _, moment in her_uses_before(services, a, cutoff)]
    assert times == sorted(times) and all(t < cutoff for t in times)


def test_the_uses_shown_are_spread_over_the_period() -> None:
    assert spread(4, 5) == [0, 1, 2, 3] and spread(0, 5) == []
    assert spread(10, 5) == [0, 2, 4, 7, 9]
    assert spread(10, 1) == [5] and spread(10, 2) == [0, 9]
    picked = spread(100, 5)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5


def test_a_use_is_shown_with_a_few_messages_before_and_after_and_marked(
    scenario: Scenario, services: Services
) -> None:
    cutoff = holdout_cutoff(services)
    message_id, moment = her_uses_before(services, scenario.md5s[0], cutoff)[0]  # episode 2
    lines = context_of_use(services, message_id, moment, cutoff, tag_of=None)
    texts = [line.text for line in lines]
    assert texts == ["今天的第2件事你听说了吗", "听说了呀2", "那你开心吗2", MARKER, "好啦晚点聊2"]
    assert [line.her for line in lines] == [False, True, False, True, True]
    assert CONTEXT_BEFORE == 4 and CONTEXT_AFTER == 2


def test_nothing_at_or_after_the_cutoff_is_read_around_a_use(
    scenario: Scenario, services: Services
) -> None:
    cutoff = holdout_cutoff(services)
    message_id, moment = her_uses_before(services, scenario.md5s[0], cutoff)[0]
    just_after = moment + timedelta(seconds=1)
    lines = context_of_use(services, message_id, moment, just_after, tag_of=None)
    assert [line.text for line in lines] == [
        "今天的第2件事你听说了吗",
        "听说了呀2",
        "那你开心吗2",
        MARKER,
    ]
    assert context_of_use(services, "no-such-message", moment, cutoff, tag_of=None) == []
    assert context_of_use(services, message_id, moment, moment, tag_of=None) == []


def test_the_text_of_the_uses_is_numbered_and_redacted() -> None:
    from twin.ingest.transcript import TranscriptLine
    from twin.llm.redaction import ConsistentRedactor

    phone = "1" + "38" + "12345678"
    blocks = [
        [TranscriptLine("a", False, f"打{phone}"), TranscriptLine("b", True, MARKER)],
        [TranscriptLine("c", True, f"{phone}对吗")],
    ]
    text = uses_text(blocks, ConsistentRedactor())
    assert text.startswith("场合 1：\n对方：打[手机号#1]\n她：" + MARKER)
    assert "场合 2：\n她：[手机号#1]对吗" in text and phone not in text


async def test_her_use_is_judged_from_the_surroundings_before_the_cutoff_only(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    cutoff = holdout_cutoff(services)
    route = api.post(API).mock(return_value=context_reply())
    tagger = tagger_of(services)
    md5 = scenario.md5s[0]
    assert await tagger.correct_with_context(md5) is True
    body = request_json(route.calls[-1].request)
    content = body["messages"][1]["content"]
    text = next(p for p in content if p["type"] == "text")["text"]
    for occasion in range(1, 5):
        assert f"场合 {occasion}：" in text
    assert "场合 5：" not in text and text.count(MARKER) == 4
    assert "那你开心吗2" in text and "那你开心吗14" in text
    # the use after the cutoff (episode 38) and everything said after the cutoff stay out
    with services.db.session() as session:
        late = [
            m.text
            for m in session.scalars(select(Message))
            if m.id in message_ids_after(services, cutoff) and m.text
        ]
    assert late and not any(t in json.dumps(body, ensure_ascii=False) for t in late)
    # the picture is shown too, with detail low, and the vocabulary is given
    image = next(p for p in content if p["type"] == "image_url")["image_url"]
    assert image["detail"] == "low" and "开心、大笑" in text
    record = StickerCatalog(services).require(md5)
    assert record.context_tags == ("委屈",) and record.context_note == "她用它表示有点委屈"
    assert record.context_uses == 4 and record.context_cutoff_at == cutoff
    assert record.tags == ("委屈",) and record.tag_source == "context"


async def test_at_most_the_configured_number_of_uses_is_shown_spread_over_the_period(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    services.settings.stickers.context_max_samples = 2
    route = api.post(API).mock(return_value=context_reply())
    assert await tagger_of(services).correct_with_context(scenario.md5s[0]) is True
    content = request_json(route.calls[0].request)["messages"][1]["content"]
    text = next(p for p in content if p["type"] == "text")["text"]
    assert "场合 2：" in text and "场合 3：" not in text
    assert (
        "那你开心吗2" in text and "那你开心吗14" in text
    )  # the first and the last, not the middle
    assert "那你开心吗5" not in text and "那你开心吗9" not in text


async def test_three_uses_are_enough_and_fewer_are_not(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=context_reply(tags=["晚安"]))
    tagger = tagger_of(services)
    assert await tagger.correct_with_context(scenario.md5s[1]) is True  # exactly three before
    assert await tagger.correct_with_context(scenario.md5s[2]) is False  # one
    assert await tagger.correct_with_context(scenario.md5s[3]) is False  # none before the cutoff
    assert route.call_count == 1
    services.settings.stickers.context_min_uses = 4
    assert await tagger.correct_with_context(scenario.md5s[0]) is True  # four
    services.settings.stickers.context_min_uses = 5
    assert await tagger_of(services).correct_with_context(scenario.md5s[2]) is False


async def test_the_correction_merges_with_the_picture_and_the_hand_wins(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    api.post(API).mock(
        side_effect=[vision_reply(tags=["开心", "晚安"]), context_reply(tags=["委屈"])]
    )
    tagger = tagger_of(services)
    md5 = scenario.md5s[0]
    await tagger.tag_sticker(md5)
    assert StickerCatalog(services).require(md5).tags == ("开心", "晚安")
    await tagger.correct_with_context(md5)
    corrected = StickerCatalog(services).require(md5)
    assert (
        corrected.tags == ("委屈",) and corrected.tag_source == "context"
    )  # use outweighs picture
    assert corrected.vision_tags == ("开心", "晚安")
    assert StickerCatalog(services).set_manual(md5, ["困"]).tag_source == "manual"


async def test_a_correction_is_not_made_twice_for_the_same_cutoff(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=context_reply())
    tagger = tagger_of(services)
    assert await tagger.correct_with_context(scenario.md5s[0]) is True
    assert await tagger.correct_with_context(scenario.md5s[0]) is False
    assert route.call_count == 1


async def test_the_surroundings_are_redacted_before_they_are_sent(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    phone = "1" + "38" + "12345678"
    with services.db.transaction(bump_state=False) as session:
        for message in session.scalars(select(Message)):
            if message.text == "那你开心吗2":
                message.text = f"那你开心吗 打我电话 {phone}"
    route = api.post(API).mock(return_value=context_reply())
    await tagger_of(services).correct_with_context(scenario.md5s[0])
    sent = json.dumps(request_json(route.calls[0].request), ensure_ascii=False)
    assert phone not in sent and "[手机号#1]" in sent


async def test_without_a_split_there_is_no_cutoff_and_so_no_correction(
    services: Services, api: respx.MockRouter
) -> None:
    from tests.support.embedding import H, U, day, write_dialogue
    from tests.support.persona import attach_files, sticker_files, sync_counters

    files = sticker_files(1)
    md5 = next(iter(files))
    write_dialogue(services, [(day(0), [U("你好"), H(kind="sticker", md5=md5)])])
    attach_files(services, files)
    sync_counters(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    route = api.post(API).mock(return_value=context_reply())
    assert await tagger_of(services).correct_with_context(md5) is False
    assert route.call_count == 0


# ---------------------------------------------------------------- when it is due


def test_a_correction_is_due_with_enough_uses_and_not_again_until_something_changes(
    scenario: Scenario, services: Services
) -> None:
    cutoff = holdout_cutoff(services)
    catalog = StickerCatalog(services)
    record = catalog.require(scenario.md5s[0])
    assert context_due(3, record, cutoff, 4) and not context_due(3, record, cutoff, 2)
    assert not context_due(5, record, cutoff, 4)
    done = catalog.save_context(record.md5, ["委屈"], "m", uses=4, cutoff=cutoff, at=cutoff)
    assert not context_due(3, done, cutoff, 4) and not context_due(3, done, cutoff, 7)
    assert context_due(3, done, cutoff, 8)  # her uses doubled
    assert context_due(3, done, cutoff + timedelta(days=1), 4)  # the cutoff moved
    unavailable = catalog.require(scenario.md5s[1])
    assert not context_due(3, replace(unavailable, status="pending"), cutoff, 9)


# ------------------------------------------------------------------ the schemas


def test_the_reply_models_know_only_their_vocabulary(services: Services) -> None:
    vocabulary = load_vocabulary(services.settings, services.paths.root)
    vision = reply_model(VisionReply, vocabulary)
    assert vision.model_validate({"tags": ["开心", "开心"], "description": "x"}).tags == ["开心"]
    with pytest.raises(ValueError, match="not allowed"):
        vision.model_validate({"tags": ["未知"], "description": "x"})
    context = reply_model(ContextReply, vocabulary)
    assert context.model_validate({"tags": ["晚安"]}).meaning == ""
    assert vision.__name__ == "VisionReply" and VisionReply.vocabulary == ()
    assert "tags" in vision.model_json_schema()["properties"]


def test_the_text_helper_makes_one_redacted_line() -> None:
    assert one_line("  a \n b  ") == "a b"
    assert one_line("x" * 300).endswith("…") and len(one_line("x" * 300)) == 200
    assert ImageInput.from_bytes(b"").detail is None
