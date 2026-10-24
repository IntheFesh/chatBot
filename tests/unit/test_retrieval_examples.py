"""Examples and their rendering: event lines as notes, pictures, quotes, stickers
(R-RET-005, R-SAFE-006, R-IMP-012)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from tests.support.embedding import H, HashingBackend, U, day, write_dialogue
from twin.ingest.captions import CAPTION_JOB, CaptionService
from twin.ops.jobs import JobQueue
from twin.retrieval.embedder import EmbeddingService
from twin.retrieval.examples import (
    DEFAULT_LABELS,
    Example,
    ExampleBuilder,
    ExampleLabels,
    ExampleLine,
    ExampleTurn,
    render_example,
    render_examples,
)
from twin.retrieval.indexer import run_index
from twin.retrieval.query import ExampleQuery, ExampleRetriever, QueryTurn
from twin.retrieval.windows import WindowRecord
from twin.services import Services
from twin.storage.chat_models import MediaAsset, Message
from twin.storage.retrieval_models import ExampleWindow

ASSET = "a" * 32


def configure(services: Services, backend: HashingBackend) -> None:
    services.settings.time.source_timezone = "UTC"
    services.settings.retrieval.model = backend.info.model
    services.settings.retrieval.dedup_similarity = 1.0


def line(
    text: str, *, kind: str = "text", ok: bool = True, quoted: str | None = None
) -> ExampleLine:
    return ExampleLine(text, kind, ok, quoted)


def example(**kwargs: object) -> Example:
    base: dict[str, object] = {
        "window_id": "w1",
        "reply_at": datetime(2026, 3, 4, 21, 35, tzinfo=UTC),
        "local_slot": 86,
        "day_type": "workday",
        "context": (),
        "reply": (),
    }
    base.update(kwargs)
    return Example(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ rendering


def test_an_example_is_rendered_as_context_then_her_reply_lines() -> None:
    text = render_example(
        example(
            context=(
                ExampleTurn(False, (line("今天吃什么"), line("我饿了"))),
                ExampleTurn(True, (line("还没想好"),)),
                ExampleTurn(False, (line("那吃火锅？"),)),
            ),
            reply=(line("好呀"), line("[表情包:开心]", kind="sticker")),
        ),
        number=2,
    )
    assert text.split("\n") == [
        "例子 2（21:30 左右，工作日）",
        "对方：今天吃什么",
        "对方：我饿了",
        "她：还没想好",
        "对方：那吃火锅？",
        "她的回复：",
        "好呀",
        "[表情包:开心]",
    ]


def test_lines_she_sent_that_the_bot_cannot_are_notes_not_lines_to_imitate() -> None:
    text = render_example(
        example(
            reply=(
                line("给你看个东西"),
                line("[图片：一碗面]", kind="image", ok=False),
                line("[通话 12 分钟]", kind="call", ok=False),
                line("好吃吧"),
            )
        )
    )
    lines = text.split("\n")
    assert lines[-4:] == [
        "给你看个东西",
        "（此处她发了：[图片：一碗面]）",
        "（此处她发了：[通话 12 分钟]）",
        "好吃吧",
    ]
    assert "她的回复：" in lines


def test_events_in_the_context_stay_event_text() -> None:
    text = render_example(
        example(
            context=(ExampleTurn(False, (line("[图片]", kind="image", ok=False), line("好看吗"))),),
            reply=(line("好看"),),
        )
    )
    assert "对方：[图片]" in text and "此处她发了" not in text


def test_quotes_show_what_they_quote_on_a_line_of_their_own() -> None:
    text = render_example(
        example(
            context=(ExampleTurn(False, (line("明天考试", kind="quote", quoted="加油"),)),),
            reply=(line("谢谢", kind="quote", quoted="明天考试"),),
        )
    )
    assert text.split("\n")[1:] == [
        "对方：[引用:加油]",
        "对方：明天考试",
        "她的回复：",
        "[引用:明天考试]",
        "谢谢",
    ]


def test_the_words_around_an_example_can_be_changed_by_the_prompt_builder() -> None:
    labels = ExampleLabels(
        user="你",
        her="我",
        reply="我当时回复",
        event_note="（我发了{event}）",
        heading="例{number}",
    )
    text = render_example(
        example(
            context=(ExampleTurn(False, (line("嗨"),)),),
            reply=(line("[语音 3 秒]", kind="voice", ok=False),),
        ),
        number=1,
        labels=labels,
    )
    assert text.split("\n") == ["例 1", "你：嗨", "我当时回复：", "（我发了[语音 3 秒]）"]
    assert DEFAULT_LABELS.reply == "她的回复"


def test_several_examples_are_numbered_from_one() -> None:
    two = [
        example(reply=(line("甲"),)),
        example(reply=(line("乙"),), local_slot=0, day_type="holiday"),
    ]
    text = render_examples(two)
    assert text.startswith("例子 1（21:30 左右，工作日）")
    assert "\n\n例子 2（00:00 左右，节假日）" in text
    assert render_examples([]) == ""
    assert example(reply=(line("好"), line("[图片]", ok=False))).reproducible_lines == (line("好"),)


# ----------------------------------------------------------- built from windows


def add_asset(
    services: Services, kind_message_text: str | None = None, caption: str | None = None
) -> str:
    """An available picture for the (first) image message; returns the asset id."""
    now = services.clock.now_utc()
    with services.db.transaction(bump_state=False) as session:
        message = next(m for m in session.scalars(select(Message)) if m.kind == "image")
        asset = MediaAsset(
            id=ASSET,
            conversation_id=message.conversation_id,
            message_id=message.id,
            kind="image",
            status="available",
            sha256="b" * 64,
            created_at=now,
            updated_at=now,
        )
        if caption:
            asset.caption = caption
        session.add(asset)
    return ASSET


def build_library(services: Services, backend: HashingBackend) -> None:
    configure(services, backend)
    episodes = [
        (
            day(1),
            [
                U("今天吃什么好呢"),
                H("你看这个"),
                H(None, kind="image"),
                H("好吃吗", kind="text"),
            ],
        ),
        (day(2), [U("晚饭吃什么好呢"), H(None, kind="voice")]),
        (day(3), [U("夜宵吃什么好呢"), H("我给你发个表情"), H(None, kind="sticker", md5="c" * 32)]),
        (day(4), [U("早餐吃什么好呢"), H("吃过了", kind="quote")]),
    ]
    episodes += [(day(8 + n), [U(f"问题{n}"), H(f"回答{n}")]) for n in range(8)]
    write_dialogue(services, episodes)


def retriever(services: Services, backend: HashingBackend, **kwargs: object) -> ExampleRetriever:
    return ExampleRetriever(services, EmbeddingService(backend), **kwargs)  # type: ignore[arg-type]


QUERY = ExampleQuery([QueryTurn(False, "吃什么好呢")], 12 * 60, "workday", None, 8)


async def test_pictures_without_a_description_read_picture_and_a_job_is_queued(
    services: Services, embedder: HashingBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_library(services, embedder)
    asset = add_asset(services)
    run_index(services)

    async def never(*args: object, **kwargs: object) -> str:
        raise AssertionError("a historic picture must never be described while a reply is built")

    monkeypatch.setattr(CaptionService, "generate", never)
    got = retriever(services, embedder)
    examples = await got.query(QUERY)
    shown = {tuple(line.text for line in e.reply): e for e in examples}
    with_picture = shown[("你看这个", "[图片]", "好吃吗")]
    assert [x.reproducible for x in with_picture.reply] == [True, False, True]
    assert "（此处她发了：[图片]）" in render_example(with_picture)

    queue = JobQueue(services.db, services.clock)
    jobs = queue.list_jobs(status="pending", job_type=CAPTION_JOB)
    assert len(jobs) == 1 and jobs[0].payload["asset_ids"] == [asset]
    await got.query(QUERY)  # asking again does not queue a second job
    assert len(queue.list_jobs(status="pending", job_type=CAPTION_JOB)) == 1


async def test_pictures_with_a_description_carry_it(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    add_asset(services, caption="一碗热腾腾的面")
    run_index(services)
    examples = await retriever(services, embedder).query(QUERY)
    picture = next(item for e in examples for item in e.reply if item.kind == "image")
    assert picture.text == "[图片：一碗热腾腾的面]" and not picture.reproducible
    assert (
        JobQueue(services.db, services.clock).list_jobs(status="pending", job_type=CAPTION_JOB)
        == []
    )


async def test_a_voice_message_is_a_note_with_its_event_text(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    run_index(services)
    examples = await retriever(services, embedder).query(QUERY)
    voice = next(e for e in examples if any(item.kind == "voice" for item in e.reply))
    assert voice.reproducible_lines == ()
    assert render_example(voice).split("\n")[-1].startswith("（此处她发了：[语音")


async def test_stickers_carry_their_label_when_one_is_known(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    run_index(services)
    plain = await retriever(services, embedder).query(QUERY)
    sticker = next(item for e in plain for item in e.reply if item.kind == "sticker")
    assert sticker.text == "[表情包]" and sticker.reproducible

    labelled = await retriever(
        services, embedder, sticker_label=lambda md5: "开心" if md5 == "c" * 32 else None
    ).query(QUERY)
    sticker = next(item for e in labelled for item in e.reply if item.kind == "sticker")
    assert sticker.text == "[表情包:开心]"


async def test_a_quote_carries_what_it_quotes(services: Services, embedder: HashingBackend) -> None:
    build_library(services, embedder)
    with services.db.transaction(bump_state=False) as session:
        quote = next(m for m in session.scalars(select(Message)) if m.kind == "quote")
        quote.quote = {"quoteContent": "早餐吃了吗   在不在", "quoteTitle": "忽略"}
    run_index(services)
    examples = await retriever(services, embedder).query(QUERY)
    line_ = next(item for e in examples for item in e.reply if item.kind == "quote")
    assert (line_.text, line_.quoted, line_.reproducible) == ("吃过了", "早餐吃了吗 在不在", True)
    assert "[引用:早餐吃了吗 在不在]\n吃过了" in render_example(
        next(e for e in examples if e.reply[0].kind == "quote")
    )


def records_of(services: Services) -> list[WindowRecord]:
    with services.db.session() as session:
        rows = session.scalars(select(ExampleWindow).order_by(ExampleWindow.reply_at_utc))
        return [WindowRecord.from_row(row) for row in rows]


async def test_a_damaged_window_never_puts_the_user_into_her_reply(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    run_index(services)
    window = records_of(services)[0]
    with services.db.session() as session:
        user_id = next(m.id for m in session.scalars(select(Message)) if m.is_sent)
    tampered = WindowRecord(
        **{**window.__dict__, "reply_block_ids": (user_id,)}  # type: ignore[arg-type]
    )
    builder = ExampleBuilder(services)
    try:
        assert await builder.build([tampered], {}) == []
        gone = WindowRecord(**{**window.__dict__, "reply_block_ids": ("no-such-message",)})  # type: ignore[arg-type]
        assert await builder.build([gone], {}) == []
        built = await builder.build([window], {window.id: (0.5, 0.6)})
    finally:
        await builder.aclose()
    assert len(built) == 1 and (built[0].similarity, built[0].score) == (0.5, 0.6)
    assert built[0].window_id == window.id and built[0].clock == "12:00"


async def test_messages_missing_from_a_context_are_left_out(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    run_index(services)
    window = records_of(services)[0]
    ghost = WindowRecord(
        **{
            **window.__dict__,
            "context_block_ids": (("no-such-message",), window.context_block_ids[0]),
        }  # type: ignore[arg-type]
    )
    builder = ExampleBuilder(services)
    try:
        (built,) = await builder.build([ghost], {})
    finally:
        await builder.aclose()
    assert len(built.context) == 1  # the turn made only of a missing message is dropped


async def test_the_user_may_appear_in_the_context_with_his_own_label(
    services: Services, embedder: HashingBackend
) -> None:
    build_library(services, embedder)
    run_index(services)
    examples = await retriever(services, embedder).query(QUERY)
    assert examples and all(e.context and not e.context[-1].her for e in examples)
