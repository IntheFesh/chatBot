"""Recognising a sticker the user sent (R-STK-006)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.persona import Scenario, sticker_png, sticker_scenario
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.incoming import (
    FALLBACK_TEXT,
    NOTHING,
    StickerDescription,
    describe_incoming_sticker,
)
from twin.stickers.library import sticker_file_allowed

STRANGER = sticker_png(40)
STRANGER_MD5 = hashlib.md5(STRANGER, usedforsecurity=False).hexdigest()


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def scenario(services: Services) -> Scenario:
    found = sticker_scenario(services)
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    return found


def vision(**fields: Any) -> httpx.Response:
    body = {"tags": ["撒娇"], "description": "一只歪着头的猫", "use_cases": "撒娇时"} | fields
    return ok(content=json.dumps(body, ensure_ascii=False), prompt=300, completion_tokens=40)


async def test_a_sticker_of_the_library_is_answered_without_a_call(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    md5 = scenario.md5s[0]
    StickerCatalog(services).save_vision(
        md5, ["开心", "大笑"], "一只笑着的猫", "好消息", at=services.clock.now_utc()
    )
    found = await describe_incoming_sticker(services, md5=md5.upper())
    assert found == StickerDescription(md5, ("开心", "大笑"), "一只笑着的猫", "library")
    assert found.known and found.text == "[表情包：一只笑着的猫（情绪：开心、大笑）]"
    assert route.call_count == 0


async def test_a_sticker_of_the_library_without_a_description_is_looked_at_and_remembered(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    md5 = scenario.md5s[1]
    first = await describe_incoming_sticker(services, md5=md5)
    assert first.source == "vision" and first.tags == ("撒娇",) and first.md5 == md5
    assert route.call_count == 1
    body = request_json(route.calls[0].request)
    image = next(p for p in body["messages"][1]["content"] if p["type"] == "image_url")["image_url"]
    assert image["detail"] == "low" and image["url"].startswith("data:image/png;base64,")
    record = StickerCatalog(services).require(md5)
    assert (
        record.description == "一只歪着头的猫"
        and record.origin == "import"
        and record.her_uses == 4
    )
    second = await describe_incoming_sticker(services, md5=md5)
    assert second.source == "library" and route.call_count == 1  # no second call


async def test_an_unknown_picture_is_described_and_added_to_the_library_as_the_users(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    found = await describe_incoming_sticker(services, image=STRANGER)  # the md5 is worked out
    assert found.md5 == STRANGER_MD5 and found.source == "vision" and found.known
    record = StickerCatalog(services).require(STRANGER_MD5)
    assert record.origin == "incoming" and record.status == "available" and record.user_uses == 0
    assert record.tags == ("撒娇",) and record.description == "一只歪着头的猫"
    assert record.tag_source == "vision" and record.sha256 and record.mime == "image/png"
    assert services.media.read_bytes(record.sha256) == STRANGER
    with services.db.session() as session:
        assert sticker_file_allowed(session, record.sha256)  # a library sticker like the others
    again = await describe_incoming_sticker(services, md5=STRANGER_MD5)
    assert again.source == "library" and route.call_count == 1


async def test_a_picture_whose_md5_differs_from_the_message_is_kept_but_never_sendable(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=vision())
    announced = "0123456789abcdef0123456789abcdef"
    found = await describe_incoming_sticker(services, md5=announced, image=STRANGER)
    assert found.known and found.md5 == announced
    record = StickerCatalog(services).require(announced)
    assert record.status == "md5_mismatch" and not record.available and not record.usable
    assert record.sha256 is not None
    with services.db.session() as session:
        assert not sticker_file_allowed(session, record.sha256)


async def test_without_the_picture_or_an_md5_the_answer_is_the_plain_marker(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    assert await describe_incoming_sticker(services) == NOTHING
    unknown = await describe_incoming_sticker(services, md5="f" * 32)
    assert unknown.text == FALLBACK_TEXT == "[表情包]" and not unknown.known and unknown.tags == ()
    assert route.call_count == 0 and StickerCatalog(services).get("f" * 32) is None


async def test_a_slow_answer_gives_the_plain_marker_and_caches_nothing(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    services.settings.stickers.describe_timeout_s = 0.05

    async def hang(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    api.post(API).mock(side_effect=hang)
    found = await describe_incoming_sticker(services, image=STRANGER)
    assert found.text == FALLBACK_TEXT and found.source == "none" and found.md5 == STRANGER_MD5
    assert StickerCatalog(services).get(STRANGER_MD5) is None


@pytest.mark.parametrize(
    "response",
    [
        error(400),
        vision(tags=["不在词表里"]),
        ok(content="这不是 JSON"),
        vision(description=" "),
    ],
)
async def test_a_failed_description_gives_the_plain_marker_and_caches_nothing(
    scenario: Scenario, services: Services, api: respx.MockRouter, response: httpx.Response
) -> None:
    api.post(API).mock(return_value=response)
    found = await describe_incoming_sticker(services, image=STRANGER)
    assert found.text == FALLBACK_TEXT and not found.known
    assert StickerCatalog(services).get(STRANGER_MD5) is None
    known = await describe_incoming_sticker(services, md5=scenario.md5s[2])
    assert (
        known.text == FALLBACK_TEXT
        and StickerCatalog(services).require(scenario.md5s[2]).description is None
    )


async def test_bytes_that_are_no_picture_are_not_described(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    found = await describe_incoming_sticker(services, image=b"this is not a picture")
    assert found.text == FALLBACK_TEXT and route.call_count == 0


async def test_a_client_that_is_passed_in_is_used_and_left_open(
    scenario: Scenario, services: Services, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=vision())
    client = build_llm_runtime(services).client
    try:
        first = await describe_incoming_sticker(services, md5=scenario.md5s[0], client=client)
        second = await describe_incoming_sticker(services, md5=scenario.md5s[1], client=client)
    finally:
        await client.aclose()
    assert first.known and second.known and route.call_count == 2


def test_the_description_line_has_the_forms_the_reply_path_needs() -> None:
    assert StickerDescription("a", (), "猫", "library").text == "[表情包：猫]"
    assert StickerDescription("a", ("开心",), None, "library").text == FALLBACK_TEXT
    assert StickerDescription("a", ("开心",), "猫", "none").text == FALLBACK_TEXT
    assert not NOTHING.known and NOTHING.md5 is None
