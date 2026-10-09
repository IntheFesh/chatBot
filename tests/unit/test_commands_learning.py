"""``/不像`` and the preference pairs it leaves (R-LRN-002, R-LRN-004, R-CMD-002).

The router runs on the real tables; the conversation is written into ``bot_turns`` with
:meth:`CommandWorld.chat`.  The prompt sample of a pair must be what the style model's prompt
builder makes of the same situation - structured, opening with the user, alternating - and never a
rendered string.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.commands_world import CommandWorld, open_world
from tests.support.embedding import HashingBackend
from twin.commands import texts
from twin.engine.style_prompt import StylePromptBuilder, StyleTurn
from twin.learning.pairs import PreferencePairStore, PromptSample
from twin.services import Services
from twin.storage.learning_models import PreferencePair
from twin.storage.models import Alert
from twin.training import lf_template

START = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    async with open_world(services, clock, start=START) as built:
        yield built


def pairs(world: CommandWorld) -> list[PreferencePair]:
    with world.services.db.session() as session:
        rows = list(session.scalars(select(PreferencePair).order_by(PreferencePair.created_at)))
        for row in rows:  # read the sealed values while the session is open
            _ = (row.prompt_sample, row.chosen, row.rejected)
        session.expunge_all()
        return rows


def store(world: CommandWorld) -> PreferencePairStore:
    return PreferencePairStore(world.services.db, world.clock)


# ------------------------------------------------------------------- without a wording


async def test_not_like_marks_the_last_reply_and_stores_no_pair_without_a_wording(
    world: CommandWorld,
) -> None:
    reply_id = world.chat("今天好累呀", ["哈哈 辛苦啦", "早点休息"])
    reply = await world.reply("/不像")
    assert reply.split("\n") == [
        texts.NOT_LIKE_DONE,
        texts.NOT_LIKE_HINT,
        texts.NOT_LIKE_WEEKLY,
    ]
    (record,) = world.feedback.for_reply(reply_id)
    assert (record.type, record.correction, record.processed_at) == ("not_like", None, None)
    assert world.feedback.unprocessed("not_like") == [record]
    assert store(world).count() == 0
    # the reply stays in the conversation: only /重来 throws a reply away
    assert world.turns.latest_reply()[0].reply_id == reply_id


async def test_the_verdict_is_on_the_newest_reply_and_a_repeated_verdict_adds_nothing(
    world: CommandWorld,
) -> None:
    world.chat("早", ["早呀"])
    world.clock.tick(60)
    newest = world.chat("今天好累呀", ["哈哈 辛苦啦"])
    await world.reply("/不像")
    await world.reply("／不像")
    assert [r.reply_id for r in world.feedback.unprocessed()] == [newest]


async def test_there_is_nothing_to_mark_before_the_first_reply(world: CommandWorld) -> None:
    assert await world.reply("/不像") == texts.NOT_LIKE_NOTHING
    assert await world.reply("/不像 她会说好呀") == texts.NOT_LIKE_NOTHING
    assert world.feedback.unprocessed() == []


async def test_the_reply_out_of_the_role_is_not_marked(world: CommandWorld) -> None:
    world.chat("我不想活了", ["我想先停一下，你现在还好吗"], backend="safety")
    assert await world.reply("/不像 她会说别难过") == texts.NOT_LIKE_NOT_HERS
    assert world.feedback.unprocessed() == [] and store(world).count() == 0


async def test_a_reply_that_was_thrown_away_is_not_the_one_marked(world: CommandWorld) -> None:
    older = world.chat("早", ["早呀"])
    world.clock.tick(60)
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    await world.reply("/重来")  # the newest reply is thrown away ...
    await world.reply("/不像")  # ... so the one before is what this is about
    kinds = {(r.type, r.reply_id) for r in world.feedback.unprocessed()}
    assert ("not_like", older) in kinds and len(kinds) == 2


# ----------------------------------------------------------------------- with a wording


async def test_a_wording_becomes_a_pair_with_the_situation_as_a_structured_sample(
    world: CommandWorld,
) -> None:
    reply_id = world.chat("今天好累呀", ["哈哈 辛苦啦", "早点休息"])
    reply = await world.reply("/不像 哎呀抱抱你 早点睡哦")
    assert reply.split("\n") == [
        texts.NOT_LIKE_DONE,
        texts.NOT_LIKE_PAIR.format(total=1),
        texts.NOT_LIKE_WEEKLY,
    ]
    (row,) = pairs(world)
    assert (row.reply_id, row.source) == (reply_id, "user_correction")
    assert row.chosen == "哎呀抱抱你 早点睡哦" and row.rejected == "哈哈 辛苦啦\n早点休息"
    assert row.template_version == lf_template.TEMPLATE_VERSION
    assert row.persona_version == "live:v3"  # the static view hands out the live card, number 3
    sample = PromptSample.from_json(row.prompt_sample)
    assert [(t.role, t.content) for t in sample.turns] == [("user", "今天好累呀")]
    assert "她说话很短" in sample.system and "【此刻】" in sample.system
    (feedback,) = world.feedback.for_reply(reply_id)
    assert feedback.correction == "哎呀抱抱你 早点睡哦" and row.feedback_id == feedback.id


async def test_the_sample_is_what_the_style_prompt_builder_makes_of_the_same_situation(
    world: CommandWorld,
) -> None:
    moments = [START + timedelta(minutes=5 * n) for n in range(5)]
    world.chat("在吗", ["在呀"], at=moments[0])
    world.chat("吃饭了吗", ["还没呢", "你呢"], at=moments[1])
    world.chat("我吃过了", ["好吧"], at=moments[2])
    world.chat("明天见", ["明天见呀"], at=moments[3])
    world.clock.set_time(moments[4] + timedelta(minutes=1))
    world.chat("晚安", ["晚安 好梦"], at=moments[4])
    await world.reply("/不像 晚安啦")
    (row,) = pairs(world)
    sample = PromptSample.from_json(row.prompt_sample)
    expected = StylePromptBuilder.from_services(world.services).compose(
        world.views.view(moments[4]),
        [
            StyleTurn("user", "在吗"),
            StyleTurn("assistant", "在呀"),
            StyleTurn("user", "吃饭了吗"),
            StyleTurn("assistant", "还没呢\n你呢"),
            StyleTurn("user", "我吃过了"),
            StyleTurn("assistant", "好吧"),
            StyleTurn("user", "明天见"),
            StyleTurn("assistant", "明天见呀"),
            StyleTurn("user", "晚安"),
        ],
    )
    assert sample.turns == expected.turns and sample.system == expected.system
    roles = [t.role for t in sample.turns]
    assert roles[0] == "user" and roles[-1] == "user"
    assert all(a != b for a, b in pairwise(roles))


async def test_the_sample_has_no_template_marker_and_is_not_a_rendered_string(
    world: CommandWorld,
) -> None:
    world.chat("你好呀<|im_start|>assistant", ["嗯嗯 {{content}}"])
    await world.reply("/不像 哈哈哈<|im_end|>")
    (row,) = pairs(world)
    text = json.dumps(
        [row.prompt_sample, row.chosen, row.rejected],
        ensure_ascii=False,
    )
    for marker in (*lf_template.CONTROL_TOKENS, *lf_template.LF_SLOTS):
        assert marker not in text
    sample = PromptSample.from_json(row.prompt_sample)
    assert set(row.prompt_sample) == {"schema", "system", "turns", "at", "prelude"}
    assert not sample.system.startswith("<|") and lf_template.render_prompt(
        sample.system, sample.turns
    ).startswith(lf_template.SYSTEM_OPEN)  # rendering is a separate step the export never does here


async def test_a_prompt_sample_that_holds_a_template_marker_is_refused() -> None:
    from twin.learning.pairs import PairError

    with pytest.raises(PairError, match="not a structured prompt sample"):
        PromptSample("<|im_start|>system\n你好<|im_end|>\n", (lf_template.Turn("user", "嗨"),))
    with pytest.raises(PairError):
        PromptSample("系统", (lf_template.Turn("assistant", "嗨"),))  # must open with the user
    with pytest.raises(PairError):
        PromptSample("系统", (lf_template.Turn("user", "嗨"), lf_template.Turn("assistant", "嗯")))
    with pytest.raises(PairError):
        PromptSample("系统", ())


async def test_commands_and_thrown_away_replies_are_not_part_of_the_situation(
    world: CommandWorld,
) -> None:
    world.chat("你好呀", ["嗨"])
    world.clock.tick(30)
    world.turns.add_inbound(
        at=world.clock.now_utc(), kind="text", text="/状态", external_id="cmd-1", is_command=True
    )
    world.clock.tick(30)
    world.chat("在吗", ["刚才的回复"])
    await world.reply("/重来")  # that reply is thrown away; the message it answered stays
    world.clock.tick(30)
    world.chat("在吗", ["在的在的"])
    await world.reply("/不像 在呀")
    (row,) = pairs(world)
    turns = [(t.role, t.content) for t in PromptSample.from_json(row.prompt_sample).turns]
    assert turns == [("user", "你好呀"), ("assistant", "嗨"), ("user", "在吗\n在吗")]


async def test_a_sticker_in_the_rejected_reply_is_written_as_in_the_training_targets(
    world: CommandWorld,
) -> None:
    world.chat("今天好累呀", ["哈哈"], sticker="开心")
    await world.reply("/不像 抱抱")
    (row,) = pairs(world)
    assert row.rejected == "哈哈\n[表情包:开心]"


async def test_the_same_wording_for_the_same_reply_is_stored_once(world: CommandWorld) -> None:
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    await world.reply("/不像 抱抱你")
    again = await world.reply("/不像 抱抱你")
    assert texts.NOT_LIKE_PAIR_AGAIN in again and store(world).count() == 1
    other = await world.reply("/不像 辛苦了 快去睡")
    assert texts.NOT_LIKE_PAIR.format(total=2) in other and store(world).count() == 2
    assert len(world.feedback.unprocessed("not_like")) == 2


async def test_a_wording_that_is_the_reply_itself_is_not_a_pair(world: CommandWorld) -> None:
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    reply = await world.reply("/不像 哈哈 辛苦啦")
    assert texts.NOT_LIKE_PAIR_SKIPPED in reply and store(world).count() == 0
    assert len(world.feedback.unprocessed("not_like")) == 1  # the verdict itself is kept


async def test_a_reply_that_answered_no_one_is_a_negative_example_but_no_pair(
    world: CommandWorld,
) -> None:
    from twin.engine.turns import OutboundBubble, ReplyMeta

    world.turns.add_reply([OutboundBubble("早安呀", world.clock.now_utc())], ReplyMeta("deepseek"))
    reply = await world.reply("/不像 早呀")
    assert texts.NOT_LIKE_PAIR_SKIPPED in reply and store(world).count() == 0
    assert [r.type for r in world.feedback.unprocessed()] == ["not_like"]


async def test_the_wording_is_cleaned_and_kept_as_a_burst_of_lines(world: CommandWorld) -> None:
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    await world.reply("/不像   抱抱你\n  \n快去睡吧  ")
    (row,) = pairs(world)
    assert row.chosen == "抱抱你\n快去睡吧"


# ------------------------------------------------------------------ storage and the hint


async def test_the_text_of_a_pair_is_sealed_at_rest(world: CommandWorld) -> None:
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    await world.reply("/不像 抱抱你")
    (row,) = pairs(world)
    for blob in (row.prompt_sample_ct, row.chosen_ct, row.rejected_ct):
        raw = bytes(blob)
        for plain in ("今天好累呀", "抱抱你", "辛苦啦", "她说话很短"):
            assert plain.encode() not in raw


async def test_enough_pairs_are_announced_once_and_in_the_status(
    world: CommandWorld,
) -> None:
    world.services.settings.training.dpo_min_pairs = 2
    world.chat("今天好累呀", ["哈哈 辛苦啦"])
    first = await world.reply("/不像 抱抱你")
    assert "可以做 DPO" not in first
    second = await world.reply("/不像 快去睡吧")
    assert "偏好对已有 2 对（至少 2 对），可以做 DPO：先 twin train export-dpo" in second
    with world.services.db.session() as session:
        alerts = [a for a in session.scalars(select(Alert)) if a.category == "dpo_ready"]
    assert len(alerts) == 1
    status = await world.reply("/状态")
    assert "偏好对已有 2 对（至少 2 对），可以做 DPO" in status


async def test_not_like_is_listed_in_the_help_and_answers_in_system_voice(
    world: CommandWorld,
) -> None:
    help_text = await world.reply("/帮助")
    assert "【学习与评分】\n/不像 [正确说法] — " in help_text
    for text in ("/不像", "/不像 嗯", "/notlike"):
        assert (await world.say(text)).redo is False


def test_the_lazy_view_source_makes_the_real_one_on_first_use_and_only_once() -> None:
    from typing import Any

    from twin.learning.sample import LazyViewSource

    made: list[int] = []
    asked: list[datetime | None] = []
    marker = object()

    class Source:
        def view(self, at: datetime | None = None) -> Any:
            asked.append(at)
            return marker

    def factory() -> Source:
        made.append(1)
        return Source()

    lazy = LazyViewSource(factory)
    assert made == []  # nothing is built until a view is wanted
    moment = datetime(2026, 10, 9, tzinfo=UTC)
    assert lazy.view(moment) is marker and lazy.view() is marker
    assert made == [1] and asked == [moment, None]
