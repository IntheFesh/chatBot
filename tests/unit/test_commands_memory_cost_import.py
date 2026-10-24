"""``/记住``, ``/忘掉``, ``/记忆``, ``/费用`` and ``/导入`` (R-MEM-009, R-LLM-006, R-ARCH-006).

The memory commands run on the real memory over the hashing embedder; the model that understands
a ``/记住`` is the scripted memory model behind ``respx``.  ``/导入`` runs on a synthetic export and
the real job queue; the report of its end is made by the same code the application runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import respx

from tests.fixtures.synth_export import SynthExport
from tests.support.clock import ManualClock
from tests.support.commands_world import CommandWorld, open_world
from tests.support.deepseek import API, TEST_KEY
from tests.support.embedding import HashingBackend
from tests.support.ingest import make_export, run_import
from tests.support.memory import FactRule, ScriptedMemoryModel, add_fact, add_followup, utc
from twin.commands import texts
from twin.commands.import_report import ImportNotifyStore, ImportReporter, render_end
from twin.config.runtime import TARGET_USERNAME
from twin.ingest.runs import get_run, latest_run
from twin.llm.ledger import LedgerRecord
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.llm.types import DAILY, CostBreakdown, LedgerTag, Usage
from twin.memory.replay import REPLAY_JOB
from twin.memory.store import NewEvent, NewFollowup
from twin.ops.jobs import JobQueue
from twin.services import Services

START = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)  # 07:00 in Chicago


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend
) -> AsyncIterator[CommandWorld]:
    services.settings.memory.conflict_min_similarity = 0.1
    async with open_world(services, clock, start=START) as built:
        yield built


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


# ------------------------------------------------------------------------------ /记住


async def test_remember_stores_the_words_as_a_fact_of_the_users_own_order(
    world: CommandWorld,
) -> None:
    reply = await world.reply("/记住 她喜欢蓝色")
    assert reply == "记住了（第 1 条）：她喜欢蓝色（没能分析这句话，按原话存下）"
    (fact,) = world.memory.corpus.facts.values()
    assert (fact.source, fact.text, fact.confidence) == ("user_command", "她喜欢蓝色", 1.0)


async def test_remember_keeps_the_text_as_it_was_written_colons_and_all(
    world: CommandWorld,
) -> None:
    reply = await world.reply("／记住：她的生日是 3月3日：别忘了")  # a full-width slash and colon
    assert "她的生日是 3月3日：别忘了" in reply
    assert [f.text for f in world.memory.corpus.facts.values()] == ["她的生日是 3月3日：别忘了"]


async def test_remember_says_what_the_model_understood_and_what_followed_from_it(
    world: CommandWorld, api: respx.MockRouter
) -> None:
    from tests.support.memory import FollowupRule

    world.services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    model = ScriptedMemoryModel(
        facts=[
            FactRule(
                "生日",
                {
                    "subject": "her",
                    "category": "anniversary",
                    "text": "她的生日是三月五日",
                    "importance": 5,
                    "event_date": "2000-03-05",
                    "recurrence": "yearly",
                },
            )
        ],
        followups=[
            FollowupRule(
                "生日",
                {"text": "她三月五日过生日，记得祝福", "due": "3月5日", "window_minutes": 600},
            )
        ],
    )
    api.post(API).mock(side_effect=model)
    reply = await world.reply("/记住 她的生日是三月五日")
    assert reply == "记住了（第 1 条）：她的生日是三月五日，另外记下了 1 条待跟进"
    assert world.memory.store.followups(only_open=True)


@pytest.mark.parametrize("bad", ["", "   "])
async def test_remember_without_words_gets_the_usage(world: CommandWorld, bad: str) -> None:
    reply = await world.reply(f"/记住{bad}")
    assert "用法：/记住 <内容>" in reply and "例如：/记住 她下周三有考试" in reply
    assert world.memory.corpus.facts == {}


async def test_what_was_remembered_can_be_listed_and_forgotten(world: CommandWorld) -> None:
    await world.reply("/记住 她养了一只猫")
    assert "1. 她养了一只猫" in await world.reply("/记忆")
    forgotten = await world.reply("/忘掉 1")
    assert forgotten == "已删除：\n第 1 条：她养了一只猫"
    assert await world.reply("/记忆") == texts.MEMORY_EMPTY


# ------------------------------------------------------------------------------ /忘掉


async def test_forget_by_number_lists_what_went_with_the_fact(world: CommandWorld) -> None:
    memory = world.memory
    keep = add_fact(memory, "她喜欢吃火锅", utc(2026, 3, 1))
    gone = add_fact(memory, "她周五要面试", utc(2026, 3, 2), source="user_said")
    memory.store.add_followup(
        NewFollowup(
            text="问问面试怎么样",
            due_at=utc(2026, 3, 6, 21),
            window_minutes=240,
            created_at=utc(2026, 3, 2),
            fact_id=gone.id,
        )
    )
    memory.store.add_event(
        NewEvent(
            local_date=date(2026, 3, 9),
            timezone="America/Chicago",
            activity="准备面试",
            source="improvised",
            fact_id=gone.id,
        )
    )
    memory.refresh()
    reply = await world.reply(f"/忘掉 {gone.number}")
    assert reply.split("\n") == [
        "已删除：",
        f"第 {gone.number} 条：她周五要面试",
        "待跟进：问问面试怎么样",
        "生活安排：准备面试",
    ]
    assert keep.id in memory.corpus.facts and gone.id not in memory.corpus.facts


async def test_forget_by_words_deletes_the_one_fact_they_name(world: CommandWorld) -> None:
    add_fact(world.memory, "她喜欢吃火锅", utc(2026, 3, 1))
    add_fact(world.memory, "她不吃香菜", utc(2026, 3, 2))
    reply = await world.reply("/忘掉 香菜")
    assert reply.startswith("已删除：\n第 2 条：她不吃香菜")
    assert [f.text for f in world.memory.corpus.facts.values()] == ["她喜欢吃火锅"]


async def test_words_that_match_several_facts_delete_nothing_and_list_the_numbers(
    world: CommandWorld,
) -> None:
    add_fact(world.memory, "她喜欢吃火锅", utc(2026, 3, 1))
    add_fact(world.memory, "她喜欢吃烧烤", utc(2026, 3, 2))
    reply = await world.reply("/忘掉 喜欢吃")
    lines = reply.split("\n")
    assert lines[0] == "有 2 条都符合，没有删除。请用编号再发一次，例如 /忘掉 1："
    assert lines[1:] == ["第 1 条：她喜欢吃火锅", "第 2 条：她喜欢吃烧烤"]
    assert len(world.memory.corpus.facts) == 2


async def test_a_follow_up_can_be_forgotten_by_its_words(world: CommandWorld) -> None:
    add_followup(world.memory, "问问她的面试", utc(2026, 3, 6, 21), utc(2026, 3, 2), status="open")
    reply = await world.reply("/忘掉 面试")
    assert reply == "已删除：\n待跟进：问问她的面试"


async def test_forgetting_what_is_not_there_says_so(world: CommandWorld) -> None:
    assert await world.reply("/忘掉 不存在的事") == texts.FORGET_NONE
    assert await world.reply("/忘掉 99") == texts.FORGET_NONE


async def test_forgetting_a_fact_that_replaced_another_names_the_one_that_is_valid_again(
    world: CommandWorld,
) -> None:
    old = add_fact(world.memory, "她住在北京", utc(2026, 3, 1))
    new = add_fact(world.memory, "她搬到了上海", utc(2026, 3, 10))
    world.memory.store.supersede_fact(old.id, new.id, at=new.known_at)
    world.memory.refresh()
    reply = await world.reply(f"/忘掉 {new.number}")
    assert reply.split("\n")[-1] == f"原来被它取代的第 {old.number} 条重新生效。"


async def test_forget_without_words_gets_the_usage(world: CommandWorld) -> None:
    assert "用法：/忘掉 <内容或编号>" in await world.reply("/忘掉")


# ------------------------------------------------------------------------------ /记忆


async def test_the_memory_is_listed_newest_first_ten_to_a_page(world: CommandWorld) -> None:
    assert await world.reply("/记忆") == texts.MEMORY_EMPTY
    for number in range(1, 13):
        add_fact(world.memory, f"第{number}件事", utc(2026, 3, number), embed=False)
    first = (await world.reply("/记忆")).split("\n")
    assert first[0] == "12. 第12件事" and first[9] == "3. 第3件事"
    assert first[-1] == "第 1/2 页，共 12 条；发 /记忆 2 看下一页"
    second = (await world.reply("/记忆 2")).split("\n")
    assert second == ["2. 第2件事", "1. 第1件事", "第 2/2 页，共 12 条"]
    assert (await world.reply("/记忆 9")).split("\n")[-1] == "第 2/2 页，共 12 条"  # clamped


async def test_the_memory_can_be_searched_by_a_word_and_shows_dates_and_follow_ups(
    world: CommandWorld,
) -> None:
    add_fact(world.memory, "她三月五日过生日", utc(2026, 3, 1), event_date=date(2026, 3, 5))
    add_fact(world.memory, "她喜欢吃火锅", utc(2026, 3, 2))
    add_followup(world.memory, "祝她生日快乐", utc(2026, 3, 5, 15), utc(2026, 3, 1), status="open")
    both = await world.reply("/记忆")
    assert "她三月五日过生日（2026-03-05）" in both and "待跟进：祝她生日快乐" in both
    found = await world.reply("/记忆 火锅")
    assert found.split("\n") == ["2. 她喜欢吃火锅", "第 1/1 页，共 1 条"]
    assert await world.reply("/记忆 不存在") == "没有包含「不存在」的记忆。"


# ------------------------------------------------------------------------------ /费用


def spend(
    world: CommandWorld,
    purpose: str,
    cost: float,
    *,
    hit: int = 0,
    miss: int = 100,
    at: datetime | None = None,
    tag: LedgerTag = DAILY,
) -> None:
    world.llm.ledger.record(
        LedgerRecord(
            "deepseek",
            "deepseek-flash",
            purpose,
            Usage(
                prompt_tokens=hit + miss,
                completion_tokens=10,
                cache_hit_tokens=hit,
                cache_miss_tokens=miss,
            ),
            CostBreakdown(cost / 2, cost / 2, 0.0, True, 1.0),
            False,
            1,
            at or world.clock.now_utc(),
            tag=tag,
        )
    )


async def test_the_cost_of_today_is_split_by_purpose_with_the_cache_and_the_budget_left(
    world: CommandWorld,
) -> None:
    spend(world, "reply", 0.30, hit=300, miss=100)
    spend(world, "extract", 0.10, hit=0, miss=100)
    lines = (await world.reply("/费用")).split("\n")
    assert lines[0] == "今天（2026-10-09）的费用"
    assert lines[1] == "合计 $0.4000，预算 $1.00，还剩 $0.6000（已用 40%）"
    assert lines[2] == "按用途：回复 $0.3000（75%）、抽取 $0.1000（25%）"
    assert lines[3] == "缓存命中率：60%（命中 300 / 共 500 个输入 token）"
    assert lines[4] == "预算级别：0（正常）"
    assert await world.reply("/费用 今天") == "\n".join(lines)
    assert await world.reply("／费用：today") == "\n".join(lines)


async def test_the_cost_of_the_month_adds_the_days_of_the_month_and_not_the_others(
    world: CommandWorld,
) -> None:
    spend(world, "reply", 0.50)
    spend(world, "reply", 0.25, at=START - timedelta(days=3))  # earlier this month
    spend(world, "reply", 9.00, at=START - timedelta(days=40))  # another month
    month = (await world.reply("/费用 本月")).split("\n")
    assert month[0] == "2026-10 的费用"
    assert month[1] == "合计 $0.7500，预算 $15.00，还剩 $14.2500（已用 5%）"
    day = (await world.reply("/费用")).split("\n")
    assert day[1].startswith("合计 $0.5000，")


async def test_one_time_batches_are_shown_apart_and_do_not_count_against_the_budget(
    world: CommandWorld,
) -> None:
    spend(world, "reply", 0.20)
    spend(world, "train_plan", 3.0, tag=LedgerTag("one_time", "batch-1"))
    lines = (await world.reply("/费用")).split("\n")
    assert lines[1].startswith("合计 $0.2000，预算 $1.00")
    assert "一次性批任务另计：$3.0000（不占预算）" in lines


async def test_a_day_without_calls_says_so(world: CommandWorld) -> None:
    lines = (await world.reply("/费用")).split("\n")
    assert lines[1] == "合计 $0.0000，预算 $1.00，还剩 $1.0000（已用 0%）"
    assert lines[2] == texts.COST_NO_CALLS


@pytest.mark.parametrize("bad", ["昨天", "all", "本年"])
async def test_the_cost_command_takes_today_or_this_month_only(
    world: CommandWorld, bad: str
) -> None:
    reply = await world.reply(f"/费用 {bad}")
    assert "用法：/费用 [今天|本月]" in reply and "例如：/费用 本月" in reply


# ------------------------------------------------------------------------------ /导入


@pytest.fixture
def export(tmp_path: Path) -> SynthExport:
    return make_export(tmp_path, target_messages=40, other_conversations=0)


async def test_an_import_from_the_chat_is_queued_like_the_command_line_does_it(
    world: CommandWorld, export: SynthExport
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    reply = await world.reply(f"/导入 {export.root}")
    run = latest_run(world.services.db)
    assert run is not None and run.status == "queued" and run.source_dir == str(export.root)
    assert reply == texts.IMPORT_STARTED.format(
        name=export.root.name, run=run.id, total=f"{run.total:,}"
    )
    queue = JobQueue(world.services.db, world.clock)
    [job] = queue.list_jobs(status="pending", job_type="import")
    assert job.payload == {"run_id": run.id}
    assert ImportNotifyStore(world.services).pending() == [run.id]  # its end will be reported


async def test_the_same_export_again_continues_the_unfinished_run(
    world: CommandWorld, export: SynthExport
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    first = latest_run(world.services.db)
    again = await world.reply(f'/导入 "{export.root}"')  # quotes around a path are taken off
    assert first is not None and again == texts.IMPORT_RESUMED.format(run=first.id)
    latest = latest_run(world.services.db)
    assert latest is not None and latest.id == first.id
    assert len(JobQueue(world.services.db, world.clock).list_jobs(status="pending")) == 1


async def test_a_first_import_cannot_choose_the_conversation_from_the_chat(
    world: CommandWorld, export: SynthExport
) -> None:
    reply = await world.reply(f"/导入 {export.root}")
    assert reply == texts.IMPORT_NEEDS_TARGET.format(path=str(export.root))
    assert latest_run(world.services.db) is None


async def test_a_folder_that_does_not_exist_or_is_no_export_is_said_so(
    world: CommandWorld, export: SynthExport, tmp_path: Path
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    missing = tmp_path / "nowhere"
    assert await world.reply(f"/导入 {missing}") == texts.IMPORT_NO_FOLDER.format(path=str(missing))
    (tmp_path / "empty").mkdir()
    reply = await world.reply(f"/导入 {tmp_path / 'empty'}")
    assert reply.startswith("这个文件夹不是可以导入的导出：")
    world.services.runtime.set(TARGET_USERNAME, "wxid" + "_somebody_else", by="test")
    assert (await world.reply(f"/导入 {export.root}")).startswith("已选定的会话不在这个导出里：")
    assert latest_run(world.services.db) is None


async def test_the_import_without_a_path_gets_the_usage(world: CommandWorld) -> None:
    assert "用法：/导入 <路径>" in await world.reply("/导入")


async def test_the_end_of_the_import_is_reported_with_counts_and_no_text(
    world: CommandWorld, export: SynthExport
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    run = latest_run(world.services.db)
    assert run is not None
    reporter = ImportReporter(world.services)
    assert reporter.collect() == []  # still queued: nothing to report
    assert "导入：任务" in await world.reply("/状态") and run.id in await world.reply("/状态")
    outcome = run_import(world.services, export, run_id=run.id)
    assert outcome.status == "done"
    [(reported, message)] = reporter.collect()
    done = get_run(world.services.db, run.id)
    assert reported == run.id and done is not None
    assert message == render_end(done) and message.startswith("导入完成：新增 40 条，重复 0 条")
    assert "无效 0 条" in message and "用时" in message
    assert export.texts
    for sentence in export.texts:  # no word of the messages is in it
        assert sentence not in message
    reporter.forget(run.id)
    assert reporter.collect() == [] and ImportNotifyStore(world.services).pending() == []


async def test_an_import_that_failed_is_reported_with_what_to_do(
    world: CommandWorld, export: SynthExport
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    run = latest_run(world.services.db)
    assert run is not None
    from twin.storage.chat_models import ImportRun

    with world.services.db.transaction() as session:
        row = session.get(ImportRun, run.id)
        assert row is not None
        row.status = "failed"
        row.error = "messages.json is cut off"
    [(_, message)] = ImportReporter(world.services).collect()
    assert (
        message.startswith("导入没有完成（failed）：已处理 0/")
        and "twin import --resume" in message
    )


async def test_a_run_that_vanished_is_forgotten_without_a_word(world: CommandWorld) -> None:
    ImportNotifyStore(world.services).add("no-such-run")
    reporter = ImportReporter(world.services)
    assert reporter.collect() == [] and reporter.notes.pending() == []


async def test_status_says_when_nothing_is_being_imported_or_replayed(world: CommandWorld) -> None:
    assert "导入与记忆回放：没有进行中的任务" in (await world.reply("/状态")).split("\n")


async def test_the_component_says_how_an_import_ended_once_it_has(
    world: CommandWorld, export: SynthExport
) -> None:
    from tests.support.waiting import wait_until
    from twin.commands.import_report import ImportReportComponent

    said: list[str] = []

    async def say(text: str) -> None:
        said.append(text)

    component = ImportReportComponent(world.services, say)
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    run = latest_run(world.services.db)
    assert run is not None
    await component.start()
    try:
        run_import(world.services, export, run_id=run.id)
        await world.clock.advance(world.services.settings.commands.import_poll_s)
        await wait_until(lambda: len(said) == 1)
        assert (
            said[0].startswith("导入完成：新增 40 条") and component.health().status.value == "ok"
        )
        # it is forgotten after it was said (a message lost is worse than one said twice)
        await wait_until(lambda: ImportNotifyStore(world.services).pending() == [])
    finally:
        await component.stop()


async def test_a_second_export_waits_until_the_first_import_is_over(
    world: CommandWorld, export: SynthExport, tmp_path: Path
) -> None:
    other = make_export(tmp_path / "second", target_messages=10, other_conversations=0)
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    first = latest_run(world.services.db)
    assert first is not None
    assert await world.reply(f"/导入 {other.root}") == texts.IMPORT_BUSY.format(run=first.id)
    latest = latest_run(world.services.db)
    assert latest is not None and latest.id == first.id
    assert len(JobQueue(world.services.db, world.clock).list_jobs(status="pending")) == 1


async def test_status_shows_the_import_under_way_and_the_replay_jobs_waiting(
    world: CommandWorld, export: SynthExport
) -> None:
    world.services.runtime.set(TARGET_USERNAME, export.target_username, by="test")
    await world.reply(f"/导入 {export.root}")
    run = latest_run(world.services.db)
    assert run is not None
    queue = JobQueue(world.services.db, world.clock)
    queue.enqueue(REPLAY_JOB, {})
    queue.enqueue(REPLAY_JOB, {})
    lines = (await world.reply("/状态")).split("\n")
    assert any(line.startswith("导入：任务 " + run.id) for line in lines)
    assert texts.STATUS_REPLAY.format(text="2 个回放任务在排队或进行") in lines
    assert "导入与记忆回放：没有进行中的任务" not in lines
