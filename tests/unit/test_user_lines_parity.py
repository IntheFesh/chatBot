"""R18-F2 (D-840): the user's sticker and quote read the same everywhere the style model reads them.

The running bot writes each message of the user as one line when it arrives
(:class:`~twin.engine.inbound.InboundRenderer`); the training export writes the user's turns of
her real history (:func:`~twin.training.export_text.context_text`); the evaluation sandbox hands
the drawn context to the same pipeline (:meth:`~twin.eval.samples.SampleDrawer._inbound`).  A
model trained on one notation and served another is measured on something it never sees, so the
three must agree character for character - for a sticker of the library and for a quote reply
whose quoted words are long and hold a closing bracket.  Synthetic data only.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import respx
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from tests.support.deepseek import TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.export_plans import run_with_plans
from tests.support.export_world import World, build_world
from tests.support.memory import mark_days_replayed
from tests.support.persona import sticker_scenario
from tests.support.synth_chat import MessageWriter
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.channel.base import InboundMessage, MediaRef, MessageKind, QuoteInfo
from twin.engine.inbound import InboundRenderer
from twin.engine.inbound import quote_line as online_quote_line
from twin.eval.render import message_lines, quote_fragment
from twin.eval.samples import SampleDrawer
from twin.llm.runtime import DEEPSEEK_SECRET, build_llm_runtime
from twin.retrieval.records import MessageData
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.stickers.incoming import describe_incoming_sticker
from twin.storage.chat_models import Message
from twin.storage.media import MediaKind
from twin.training.export import ExportOptions, TrainingSetExporter
from twin.training.export_blocks import LoadedMessage
from twin.training.export_text import quote_line as training_quote_line

AT = datetime(2026, 10, 9, 18, 30, tzinfo=UTC)
QUOTED = "我们周六下午三点在[学校]南门见面然后一起去吃那家新开的火锅店好不好"
DESCRIPTION = "一只笑着的猫"
TAGS = ["开心", "大笑"]


@pytest.fixture
async def renderer(services: Services) -> AsyncIterator[InboundRenderer]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    runtime = build_llm_runtime(services)
    with respx.mock(assert_all_called=False):
        made = InboundRenderer(services, client=runtime.client)
        yield made
        await made.aclose()
    await runtime.client.aclose()


def users(
    kind: str,
    *,
    text: str | None = None,
    md5: str | None = None,
    quote: dict[str, Any] | None = None,
) -> MessageData:
    """A message the user sent (``is_sent``) as the export and the sandbox read it."""
    return MessageData(
        id=f"u-{kind}",
        kind=kind,
        is_sent=True,
        text=text,
        quote=quote,
        sticker_md5=md5,
        call_status=None,
        call_duration_s=None,
        voice_seconds=None,
        has_transcript=False,
    )


def as_photo(services: Services, data: bytes) -> MediaRef:
    """What a channel delivers for a sticker the user sends: a plain image of those bytes."""
    found = services.media.put(data, MediaKind.IMAGE)
    return MediaRef(found.sha256, MediaKind.IMAGE, found.size, "image/png", "emoji.png")


# ------------------------------------------------------------------- online and the sandbox


async def test_r18_f2_a_user_sticker_reads_the_same_online_and_in_the_sandbox(
    services: Services, renderer: InboundRenderer
) -> None:
    """R18-F2 (D-840): the sandbox showed ``[表情包:开心]`` where the bot reads the description."""
    scenario = sticker_scenario(services)
    md5 = scenario.md5s[0]
    StickerCatalog(services).save_vision(
        md5, TAGS, DESCRIPTION, "好消息", at=services.clock.now_utc()
    )
    online = await renderer.render(
        InboundMessage(
            "m1", AT, MessageKind.IMAGE, media_ref=as_photo(services, scenario.files[md5])
        )
    )
    assert online.kind == "sticker"
    assert online.text == f"[表情包：{DESCRIPTION}（情绪：开心、大笑）]"
    drawn = SampleDrawer(services)._inbound(LoadedMessage(users("sticker", md5=md5), AT))
    assert (drawn["kind"], drawn["text"]) == ("sticker", online.text)


async def test_r18_f2_a_user_quote_reads_the_same_online_and_in_the_sandbox(
    services: Services, renderer: InboundRenderer
) -> None:
    """R18-F2 (D-840): one cut (30 characters) and one escape (``]`` as ``)``) on every path."""
    quote = QuoteInfo(text=QUOTED, resolved=True)
    online = await renderer.render(
        InboundMessage("m2", AT, MessageKind.TEXT, text="好呀", quote=quote)
    )
    header, words = online.text.split("\n")
    assert words == "好呀"
    fragment = QUOTED[:30].replace("]", ")")
    assert header == f"[引用:{fragment}]"  # cut, and the bracket cannot close the line early
    message = users("quote", text="好呀", quote={"quoteContent": QUOTED})
    drawn = SampleDrawer(services)._inbound(LoadedMessage(message, AT))
    assert (drawn["kind"], drawn["text"]) == ("text", online.text)
    # the blind-test screen shows the same words of the quote (its own brackets, R-EVAL-001)
    assert message_lines(message, lambda md5: None)[0] == f"[引用：{fragment}]"


QUOTE_TEXT = st.text(
    alphabet=st.sampled_from([*"好的猫火锅学校", "]", "[", " ", "\n", "\t", "a", "　"]),
    max_size=80,
)


@settings(max_examples=200, deadline=None)
@given(source=QUOTE_TEXT)
def test_r18_f2_the_quote_line_is_one_function_on_every_path(source: str) -> None:
    """R18-F2 (D-840): the reply path, the training export and the screen cut the same words."""
    online = online_quote_line(QuoteInfo(text=source, resolved=True))
    training = training_quote_line(users("quote", text="好", quote={"quoteContent": source}))
    assert online == training
    if online is None:
        assert quote_fragment(source) is None
        return
    shown = quote_fragment(source)
    assert shown is not None and online == f"[引用:{shown}]"
    assert "\n" not in online and "]" not in shown and len(shown) <= 30
    assert shown == shown.strip()


# ---------------------------------------------------------------------- training export

TOKENIZER = tiny_qwen_tokenizer()
STICKER = "f" * 32  # a sticker only the user sends in this history
MARKER = "收到你的表情包和引用了"


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=12)


def add_user_sticker_and_quote(world: World) -> datetime:
    """At night (she is asleep in the synthetic routine): a sticker and a quote, then her reply."""
    services = world.services
    moment = datetime(2026, 8, 6, 7, 0, tzinfo=UTC)  # 02:00 in Chicago, a quiet hour
    writer = MessageWriter(services)
    writer.add(moment, False, "sticker", None, STICKER)
    writer.add(moment + timedelta(seconds=20), False, "quote", "好呀")
    writer.add(moment + timedelta(seconds=90), True, "text", MARKER)
    writer.store(append=True)
    with services.db.transaction(bump_state=False) as session:
        row = session.scalars(
            select(Message).where(Message.kind == "quote", Message.is_sent.is_(True))
        ).one()
        row.quote = {"quoteContent": QUOTED, "quoteTitle": "她"}
    StickerCatalog(services).save_vision(
        STICKER, TAGS, DESCRIPTION, "好消息", at=services.clock.now_utc()
    )
    mark_days_replayed(services)
    return moment


async def test_r18_f2_a_training_sample_shows_the_users_sticker_and_quote_as_the_bot_reads_them(
    world: World, tmp_path: Path
) -> None:
    """R18-F2 (D-840): the export wrote ``[表情包:开心]`` and another quote than the bot reads."""
    services = world.services
    add_user_sticker_and_quote(world)
    exporter = TrainingSetExporter(
        services, TOKENIZER, ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    )
    result = run_with_plans(services, exporter)
    assert result.dataset is not None
    rows: list[dict[str, Any]] = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        path = result.dataset.file_path(name)
        rows.extend(json.loads(line) for line in path.read_text("utf-8").splitlines())
    (row,) = [r for r in rows if r["conversations"][-1]["value"] == MARKER]
    asked = row["conversations"][-2]
    assert asked["from"] == "human"
    # what the reply path writes for the same two messages (the library knows the sticker)
    sticker = (await describe_incoming_sticker(services, md5=STICKER)).text
    quote = online_quote_line(QuoteInfo(text=QUOTED, resolved=True))
    assert sticker == f"[表情包：{DESCRIPTION}（情绪：开心、大笑）]" and quote is not None
    # her reply answers the user's turn; its last lines are the two messages, line for line
    assert asked["value"].split("\n")[-3:] == [sticker, quote, "好呀"]
