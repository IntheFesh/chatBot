"""The rules of ``[不要这样]``: what may be a rule, and the weekly consolidation (R-LRN-003).

The consolidator runs on the real tables with the real client behind ``respx``; the scripted model
answers the two prompts it asks (the merged list, and the second judgement of the new rules).  A
rule that is about a date, an event, an amount, a name or a place never reaches the persona card.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from tests.support.clock import ManualClock
from tests.support.commands_world import CommandWorld, open_world
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.policies import AlwaysOffPeak
from twin.learning.component import LearningComponent
from twin.learning.jobs import (
    LAST_QUEUED_KEY,
    RULES_JOB,
    handle_rules,
    last_queued,
    queue_if_due,
    queue_rules_job,
)
from twin.learning.rulecheck import check_rule, clean_rule
from twin.learning.rules import (
    ConsolidationReport,
    RuleConsolidator,
    merge_rules,
    same_rule,
)
from twin.llm.errors import LlmError
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.ops.jobs import HandlerRegistry, JobQueue, Worker
from twin.profile.persona import compose
from twin.profile.persona.api import read_corrections
from twin.profile.persona.refresh import persona_store
from twin.profile.persona.sections import AUTO, DONT, MANUAL, STATS
from twin.services import Services
from twin.storage.models import Alert
from twin.storage.settings_store import get_setting, put_setting

START = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)
LIMIT = 40


# --------------------------------------------------------------------------- the check


@pytest.mark.parametrize(
    "rule",
    [
        "不要用客服腔，少说“请问有什么可以帮您”",
        "回复要简短一点",
        "不要用句号结尾",
        "不要每句话都加表情",
        "少用‘亲爱的’称呼",
        "语气不要太客气，像对同事一样",
        "多用口语，少用书面语",
        "- 不要一次说太多",  # a bullet is taken off first
        "1. 少用感叹号",
    ],
)
def test_a_rule_about_the_manner_of_speaking_passes_the_local_check(rule: str) -> None:
    assert check_rule(rule, max_chars=LIMIT).ok


@pytest.mark.parametrize(
    ("rule", "kind"),
    [
        ("别在回复里写3月5日", "fact"),
        ("不要提周三的考试", "fact"),
        ("别说明天去逛街", "fact"),
        ("不要说两点半见面", "fact"),
        ("不要提十块钱的事", "fact"),
        ("别提工资", "fact"),
        ("不要提她的生日", "fact"),
        ("不要在晚上说早安", "fact"),
        ("别提上周的旅行", "fact"),
        ("别说她住在上海路12号", "fact"),
        ("不要发电话 " + "138" + "12345678", "fact"),  # built from parts: the privacy scan
        ("别告诉他邮箱 a@b.com", "name"),
        ("她住在北京大学", "name"),
        ("她在朝阳区工作", "name"),
        ("她名叫小红", "name"),
        ("这个人姓王", "name"),
    ],
)
def test_a_rule_with_a_date_an_event_an_amount_or_a_name_is_refused(rule: str, kind: str) -> None:
    verdict = check_rule(rule, max_chars=LIMIT)
    assert not verdict.ok and verdict.kind == kind, verdict


@pytest.mark.parametrize(
    "rule",
    ["", "短", "不要" * 30, "不要太客气\n少用句号", "为什么这么说话？", "你怎么这么客气呢"],
)
def test_a_rule_that_is_empty_too_short_too_long_in_lines_or_a_question_is_refused(
    rule: str,
) -> None:
    assert not check_rule(rule, max_chars=LIMIT).ok


def test_the_length_limit_is_the_setting() -> None:
    assert check_rule("不要太客气也不要太啰嗦", max_chars=20).ok
    assert not check_rule("不要太客气也不要太啰嗦", max_chars=8).ok


def test_a_rule_is_cleaned_to_one_line() -> None:
    assert clean_rule("  - 不要   太客气 ") == "不要 太客气"
    assert clean_rule("2) 少用句号") == "少用句号"


def test_rules_that_say_the_same_are_one_rule() -> None:
    assert same_rule("不要太客气", "不要太客气")
    assert same_rule("不要太客气", "不要太客气啦")
    assert same_rule("不要用客服腔说话", "不要用客服腔说话，要像朋友")
    assert not same_rule("不要太客气", "少用句号结尾")
    merged, repeats = merge_rules(
        ["不要太客气"], ["不要太客气啦", "少用句号结尾", "少用句号结尾。", "多用口语"]
    )
    assert merged == ["不要太客气", "少用句号结尾", "多用口语"] and repeats == 1


# --------------------------------------------------------------------- the consolidation


@dataclass
class RulesModel:
    """The model of the weekly job: the merged list it writes and its judgement of new rules."""

    rules: list[str] = field(default_factory=list)
    refuse: dict[str, str] = field(default_factory=dict)  # rule -> kind: judged not ok
    silent: set[str] = field(default_factory=set)  # rules it gives no verdict for
    fail: bool = False
    merges: list[str] = field(default_factory=list)  # the user messages of the merge calls
    checks: list[list[str]] = field(default_factory=list)  # the rules shown to the second look

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            return error(400, "scripted failure")
        messages = request_json(request)["messages"]
        system, user = messages[0]["content"], messages[1]["content"]
        if "整理一个聊天机器人" in system:
            self.merges.append(user)
            return ok(content=json.dumps({"rules": self.rules}, ensure_ascii=False))
        assert "规则审核员" in system, "the job asked the model something it never asks"
        listed = user.strip().split("\n\n", 1)[0]  # the client adds the schema after a blank line
        shown = [line.split(". ", 1)[1] for line in listed.split("\n")]
        self.checks.append(shown)
        verdicts: list[dict[str, Any]] = []
        for number, rule in enumerate(shown, start=1):
            if rule in self.silent:
                continue
            bad = rule in self.refuse
            verdicts.append(
                {"index": number, "ok": not bad, "kind": self.refuse.get(rule, "style")}
            )
        return ok(content=json.dumps({"verdicts": verdicts}))


@pytest.fixture
def model() -> RulesModel:
    return RulesModel()


@pytest.fixture
def api(model: RulesModel) -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        router.post(API).mock(side_effect=model)
        yield router


@pytest.fixture
async def world(
    services: Services, clock: ManualClock, embedder: HashingBackend, api: respx.MockRouter
) -> AsyncIterator[CommandWorld]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    async with open_world(services, clock, start=START) as built:
        yield built


def put_card(world: CommandWorld, rules: list[str]) -> None:
    """A live card with every section filled, so that a change of one can be told from the rest."""
    text = (
        compose.stats_block(["几乎不用句号", "单条三到八个字"])
        + compose.auto_block("### 风格\n- 爱说“哈哈”\n### 基本情况\n- 在读研究生")
        + compose.manual_block(["我手写的风格行"], ["我手写的事实行"])
        + compose.dont_block(rules)
    )
    persona_store(world.services).add_version("live", text, reason="test")


def consolidator(world: CommandWorld) -> RuleConsolidator:
    return RuleConsolidator(world.services, world.llm.client, world.feedback, world.turns)


def mark_replies(world: CommandWorld, wordings: dict[str, str | None]) -> list[str]:
    """One exchange per ``reply -> wording``; the reply is marked ``not_like`` with the wording."""
    ids = []
    for reply, wording in wordings.items():
        reply_id = world.chat(f"在吗{len(ids)}", [reply])
        world.clock.tick(60)
        world.feedback.add("not_like", reply_id, correction=wording)
        ids.append(reply_id)
    return ids


async def test_new_feedback_becomes_rules_in_a_new_version_that_keeps_every_other_section(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    before = persona_store(world.services).active("live")
    assert before is not None
    redone = world.chat("今天好累呀", ["亲爱的 您辛苦了"])
    world.clock.tick(60)
    world.feedback.add("redo", redone)
    mark_replies(world, {"好的呢 收到": "好呀好呀"})
    model.rules = ["不要太客气", "少用“您”这样的称呼", "不要用“收到”这类回复"]
    report = await consolidator(world).run()
    after = persona_store(world.services).active("live")
    assert after is not None and after.number == before.number + 1
    assert after.reason == "corrections"
    assert read_corrections(world.services) == model.rules
    for section in (STATS, AUTO, MANUAL):  # nothing but [不要这样] changed
        assert after.sections.block(section) == before.sections.block(section)
    assert after.sections.block(DONT) != before.sections.block(DONT)
    assert "我手写的风格行" in after.text and "我手写的事实行" in after.text
    assert (report.status, report.feedback, report.rules, report.added, report.kept) == (
        "written",
        2,
        3,
        2,
        1,
    )
    assert world.feedback.unprocessed() == []  # the feedback is used up
    assert report.version == after.id


async def test_the_model_sees_the_replies_the_wordings_and_the_rules_it_has(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气", "少用句号结尾"])
    redone = world.chat("今天好累呀", ["亲爱的 您辛苦了", "早点休息"])
    world.feedback.add("redo", redone)
    mark_replies(world, {"好的呢 收到": "好呀好呀"})
    model.rules = ["不要太客气"]
    await consolidator(world).run()
    (prompt,) = model.merges
    assert "1. 不要太客气\n2. 少用句号结尾" in prompt and "（共 2 条）" in prompt
    assert "[撤销重来] 机器人说：亲爱的 您辛苦了 / 早点休息" in prompt
    assert "[不像她] 机器人说：好的呢 收到；用户认为她会说：好呀好呀" in prompt
    assert "（共 2 条）" in prompt.split("这一周的新反馈")[1]


async def test_similar_rules_are_merged_and_the_older_wording_stays(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气啦", "少用句号结尾", "少用句号结尾。", "多用口语"]
    report = await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气", "少用句号结尾", "多用口语"]
    assert report.duplicates == 1 and report.kept == 1 and report.added == 2


async def test_at_most_thirty_rules_are_kept_and_the_rest_is_cut(
    world: CommandWorld, model: RulesModel
) -> None:
    letters = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒盈昃宿列张寒暑"
    many = [f"少用“{letter}”字开头的口头禅" for letter in letters[:35]]
    put_card(world, many[:5])
    mark_replies(world, {"您好": None})
    model.rules = many
    report = await consolidator(world).run()
    final = read_corrections(world.services)
    assert len(final) == 30 == report.rules and final == many[:30] and report.cut == 5


async def test_the_limit_is_a_setting(world: CommandWorld, model: RulesModel) -> None:
    world.services.settings.learning.rules_max = 2
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气", "少用句号结尾", "多用口语"]
    await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气", "少用句号结尾"]


async def test_a_rule_about_a_date_an_event_or_a_name_never_reaches_the_card(
    world: CommandWorld, model: RulesModel
) -> None:
    mark_replies(world, {"您好": None})
    model.rules = [
        "不要太客气",
        "别在回复里写3月5日",
        "不要提周三的考试",
        "别说明天去逛街",
        "少用句号结尾",
    ]
    report = await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气", "少用句号结尾"]
    assert (report.dropped_local, report.dropped_facts) == (3, 3)
    with world.services.db.session() as session:
        raised = [a.category for a in session.scalars(select(Alert))]
    assert [c for c in raised if c.startswith("learning")] == ["learning_fact_corrections"]
    # nothing that was refused was even shown to the second look
    assert all("3月5日" not in rule for shown in model.checks for rule in shown)


async def test_the_second_look_of_the_model_drops_what_the_local_check_cannot_see(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气", "别叫他张伟", "少用句号结尾", "要记得她爱吃火锅"]
    model.refuse = {"别叫他张伟": "name", "要记得她爱吃火锅": "fact"}
    report = await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气", "少用句号结尾"]
    assert model.checks == [["别叫他张伟", "少用句号结尾", "要记得她爱吃火锅"]]  # not the old rule
    assert (report.dropped_judged, report.dropped_facts) == (2, 1)


async def test_a_rule_the_model_says_nothing_about_is_not_approved(
    world: CommandWorld, model: RulesModel
) -> None:
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气", "少用句号结尾"]
    model.silent = {"少用句号结尾"}
    await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气"]


async def test_an_answer_without_rules_never_wipes_the_card(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气", "少用句号结尾"])
    mark_replies(world, {"您好": None})
    model.rules = []
    report = await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气", "少用句号结尾"]
    assert report.status == "unchanged" and world.feedback.unprocessed() == []
    top = persona_store(world.services).active("live")
    assert top is not None and top.reason == "test"  # no new version


async def test_an_answer_with_nothing_but_refused_rules_keeps_the_old_ones(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    mark_replies(world, {"您好": None})
    model.rules = ["不要提周三的考试"]
    report = await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气"] and report.status == "unchanged"


async def test_a_card_that_does_not_exist_yet_is_made_with_the_first_rules(
    world: CommandWorld, model: RulesModel
) -> None:
    assert persona_store(world.services).active("live") is None
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气"]
    await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气"]


async def test_the_same_result_writes_no_new_version(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气"]
    report = await consolidator(world).run()
    top = persona_store(world.services).active("live")
    assert report.status == "unchanged" and top is not None and top.reason == "test"


async def test_without_new_feedback_nothing_is_asked(
    world: CommandWorld, model: RulesModel
) -> None:
    report = await consolidator(world).run()
    assert report == ConsolidationReport("nothing_new") and model.merges == []


async def test_feedback_whose_replies_are_gone_is_used_up_without_a_call(
    world: CommandWorld, model: RulesModel
) -> None:
    world.feedback.add("redo", "01HZZZZZZZZZZZZZZZZZZZZZZZ")
    report = await consolidator(world).run()
    assert report.status == "unchanged" and model.merges == []
    assert world.feedback.unprocessed() == []


async def test_a_failing_model_changes_nothing_and_leaves_the_feedback_for_the_next_try(
    world: CommandWorld, model: RulesModel
) -> None:
    put_card(world, ["不要太客气"])
    mark_replies(world, {"您好": None})
    model.fail = True
    with pytest.raises(LlmError):
        await consolidator(world).run()
    assert read_corrections(world.services) == ["不要太客气"]
    assert len(world.feedback.unprocessed()) == 1


async def test_the_newest_feedback_is_what_one_run_looks_at(
    world: CommandWorld, model: RulesModel
) -> None:
    mark_replies(world, {f"回复{n}": None for n in range(70)})
    model.rules = ["不要太客气"]
    report = await consolidator(world).run()
    assert report.feedback == 60 and len(world.feedback.unprocessed()) == 10


# ----------------------------------------------------------------------------- the job


def worker_for(world: CommandWorld) -> Worker:
    registry = HandlerRegistry()
    registry.register(RULES_JOB, handle_rules)
    return Worker(
        JobQueue(world.services.db, world.clock),
        registry,
        world.clock,
        services=world.services,
        offpeak=AlwaysOffPeak(),
        alerts=world.services.alerts,
    )


async def test_the_job_runs_the_consolidation_off_peak_and_is_not_queued_twice(
    world: CommandWorld, model: RulesModel
) -> None:
    mark_replies(world, {"您好": None})
    model.rules = ["不要太客气"]
    job_id = queue_rules_job(world.services)
    assert job_id is not None and queue_rules_job(world.services) is None
    [job] = JobQueue(world.services.db, world.clock).list_jobs(job_type=RULES_JOB)
    assert job.offpeak_only and job.deadline is not None
    summary = await worker_for(world).run_until_idle()
    assert summary.done == 1
    assert read_corrections(world.services) == ["不要太客气"] and world.feedback.unprocessed() == []


async def test_a_job_that_cannot_ask_the_model_is_retried_and_changes_nothing(
    world: CommandWorld, model: RulesModel
) -> None:
    mark_replies(world, {"您好": None})
    model.fail = True
    queue_rules_job(world.services)
    summary = await worker_for(world).run_until_idle()
    assert summary.done == 0 and (summary.retried or summary.failed)
    assert len(world.feedback.unprocessed()) == 1


async def test_a_week_is_waited_between_two_runs_and_only_when_there_is_feedback(
    world: CommandWorld, model: RulesModel
) -> None:
    assert queue_if_due(world.services) is None  # nothing to learn from
    mark_replies(world, {"您好": None})
    first = queue_if_due(world.services)
    assert first is not None
    assert get_setting_value(world, LAST_QUEUED_KEY) == world.clock.now_utc().isoformat()
    model.rules = ["不要太客气"]
    await worker_for(world).run_until_idle()
    mark_replies(world, {"您好呀": None})
    world.clock.tick(6 * 86400)
    assert queue_if_due(world.services) is None  # not yet a week
    world.clock.tick(2 * 86400)
    assert queue_if_due(world.services) is not None


def get_setting_value(world: CommandWorld, key: str) -> Any:
    with world.services.db.session() as session:
        return get_setting(session, key)


async def test_the_interval_is_a_setting(world: CommandWorld) -> None:
    world.services.settings.learning.rules_interval_days = 1
    mark_replies(world, {"您好": None})
    assert queue_if_due(world.services) is not None
    JobQueue(world.services.db, world.clock).cancel(
        JobQueue(world.services.db, world.clock).list_jobs(job_type=RULES_JOB)[0].id
    )
    world.clock.tick(86400 + 60)
    assert queue_if_due(world.services) is not None


async def test_the_component_looks_when_it_starts_and_then_every_interval(
    world: CommandWorld, model: RulesModel
) -> None:
    from tests.support.waiting import wait_until

    world.services.settings.learning.check_interval_s = 86400.0  # a look a day
    component = LearningComponent(world.services)
    queue = JobQueue(world.services.db, world.clock)
    mark_replies(world, {"您好": None})
    await component.start()
    try:
        await wait_until(lambda: len(queue.list_jobs(job_type=RULES_JOB)) == 1)  # at the start
        assert component.health().status.value == "ok"
        model.rules = ["不要太客气"]
        await worker_for(world).run_until_idle()
        mark_replies(world, {"您好呀": None})
        await world.clock.advance(6 * 86400)
        assert len(queue.list_jobs(job_type=RULES_JOB)) == 1  # a week has not passed yet
        await world.clock.advance(86400)
        await wait_until(lambda: len(queue.list_jobs(job_type=RULES_JOB)) == 2)
    finally:
        await component.stop()
    assert await component.look() is None  # a job is waiting already


@pytest.mark.parametrize("raw", ["not-a-time", 42, ""])
async def test_a_last_queued_time_that_cannot_be_read_counts_as_never(
    world: CommandWorld, raw: object
) -> None:
    with world.services.db.transaction(bump_state=False) as session:
        put_setting(
            session, LAST_QUEUED_KEY, raw, clock=world.clock, by="test", record_history=False
        )
    assert last_queued(world.services) is None
    mark_replies(world, {"您好": None})
    assert queue_if_due(world.services) is not None  # never queued: it is due at once
