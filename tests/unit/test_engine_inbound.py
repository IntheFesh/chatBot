"""How each kind of message of the user reads in the prompt (R-ENG-013, R-STK-006)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
import respx

from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.persona import sticker_png
from twin.channel.base import InboundMessage, MediaRef, MessageKind, QuoteInfo
from twin.engine.inbound import InboundRenderer, media_record, quote_line
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.services import Services
from twin.storage.media import MediaKind

AT = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)
PICTURE = sticker_png(3)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def renderer(services: Services) -> AsyncIterator[InboundRenderer]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    runtime: LlmRuntime = build_llm_runtime(services)
    made = InboundRenderer(services, client=runtime.client)
    yield made
    await runtime.client.aclose()


def stored(services: Services, kind: MediaKind, data: bytes = PICTURE) -> MediaRef:
    found = services.media.put(data, kind)
    return MediaRef(found.sha256, kind, len(data), "image/png", "报告.pdf")


def message(kind: MessageKind, **fields: object) -> InboundMessage:
    return InboundMessage("m1", AT, kind, **fields)  # type: ignore[arg-type]


async def test_text_is_the_words(renderer: InboundRenderer) -> None:
    item = await renderer.render(message(MessageKind.TEXT, text="  在吗  "))
    assert (item.id, item.at, item.kind, item.text, item.media) == ("m1", AT, "text", "在吗", None)


async def test_a_quote_reply_is_preceded_by_what_it_quotes(renderer: InboundRenderer) -> None:
    quote = QuoteInfo(text="今天\n吃什么", resolved=True)
    item = await renderer.render(message(MessageKind.TEXT, text="吃面", quote=quote))
    assert item.text == "[引用:今天 吃什么]\n吃面"
    assert item.media == {"quote": quote.to_dict()}
    unresolved = await renderer.render(
        message(MessageKind.TEXT, text="吃面", quote=QuoteInfo(svr_id="9"))
    )
    assert unresolved.text == "吃面"  # nothing to show of the quoted message
    titled = await renderer.render(
        message(MessageKind.TEXT, text="好", quote=QuoteInfo(title="一个链接"))
    )
    assert titled.text == "[引用:一个链接]\n好"
    assert quote_line(None) is None


async def test_a_picture_is_described_by_the_vision_model(
    services: Services, renderer: InboundRenderer, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content="一只趴在窗台上的橘猫"))
    ref = stored(services, MediaKind.IMAGE)
    item = await renderer.render(message(MessageKind.IMAGE, media_ref=ref))
    assert item.kind == "image" and item.text == "[图片：一只趴在窗台上的橘猫]"
    assert item.media == {"media_ref": ref.to_dict()}
    body = request_json(route.calls[0].request)
    assert any(p["type"] == "image_url" for p in body["messages"][-1]["content"])


async def test_a_picture_that_cannot_be_described_is_just_a_picture(
    services: Services, renderer: InboundRenderer, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=error(401, "bad key"))
    ref = stored(services, MediaKind.IMAGE)
    assert (await renderer.render(message(MessageKind.IMAGE, media_ref=ref))).text == "[图片]"
    assert (await renderer.render(message(MessageKind.IMAGE))).text == "[图片]"  # nothing stored


async def test_a_video_is_described_by_its_cover(
    services: Services, renderer: InboundRenderer, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content="海边的日落"))
    cover = stored(services, MediaKind.IMAGE)
    item = await renderer.render(message(MessageKind.VIDEO, media_ref=cover))
    assert item.kind == "video" and item.text == "[视频：海边的日落]"
    flagged = await renderer.render(message(MessageKind.VIDEO, flags=frozenset({"video_no_cover"})))
    assert flagged.text == "[视频]" and flagged.media == {"flags": ["video_no_cover"]}


async def test_a_voice_message_is_its_transcript(renderer: InboundRenderer) -> None:
    said = await renderer.render(message(MessageKind.VOICE, text="我马上到"))
    assert said.kind == "voice" and said.text == "[语音：我马上到]"
    silent = await renderer.render(
        message(MessageKind.VOICE, flags=frozenset({"voice_untranscribed"}))
    )
    assert silent.text == "[语音，未转写]"


async def test_a_file_is_its_name(services: Services, renderer: InboundRenderer) -> None:
    named = await renderer.render(message(MessageKind.FILE, text="周报.xlsx"))
    assert named.kind == "file" and named.text == "[文件：周报.xlsx]"
    ref = stored(services, MediaKind.FILE, b"data")
    from_ref = await renderer.render(message(MessageKind.FILE, media_ref=ref))
    assert from_ref.text == "[文件：报告.pdf]"
    assert (await renderer.render(message(MessageKind.FILE))).text == "[文件]"


async def test_anything_else_is_marked_as_such(renderer: InboundRenderer) -> None:
    item = await renderer.render(message(MessageKind.UNKNOWN, item_type=77))
    assert item.kind == "unknown" and item.text == "[其他消息]"


async def test_a_sticker_is_described_from_the_library_or_by_looking(
    services: Services, renderer: InboundRenderer, api: respx.MockRouter
) -> None:
    reply = {"tags": ["撒娇"], "description": "一只歪着头的猫", "use_cases": "撒娇时"}
    route = api.post(API).mock(return_value=ok(content=json.dumps(reply, ensure_ascii=False)))
    ref = stored(services, MediaKind.STICKER)
    item = await renderer.render(message(MessageKind.IMAGE, media_ref=ref))
    assert item.kind == "sticker"
    assert item.text == "[表情包：一只歪着头的猫（情绪：撒娇）]"
    again = await renderer.render(message(MessageKind.IMAGE, media_ref=ref))
    assert again.text == item.text and route.call_count == 1  # the library remembers it


async def test_several_messages_are_rendered_in_order(
    services: Services, renderer: InboundRenderer, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content="一杯咖啡"))
    ref = stored(services, MediaKind.IMAGE)
    items = await renderer.render_all(
        [
            InboundMessage("a", AT, MessageKind.TEXT, text="看"),
            InboundMessage("b", AT, MessageKind.IMAGE, media_ref=ref),
            InboundMessage("c", AT, MessageKind.VOICE, text="好看吗"),
        ]
    )
    assert [i.text for i in items] == ["看", "[图片：一杯咖啡]", "[语音：好看吗]"]
    assert media_record(message(MessageKind.TEXT, text="x")) is None
    await renderer.aclose()
