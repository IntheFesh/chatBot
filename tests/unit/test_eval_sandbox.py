"""The evaluation sandbox on a synthetic world (R-EVAL-009, R-TRN-013, R-LLM-014)."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import respx
from sqlalchemy import select, text

from tests.support.eval_world import (
    DeepSeekScript,
    make_stickers_available,
    open_kit,
    request_for,
    sample_of,
)
from tests.support.export_world import LATE_FACT, LIVE_STYLE, PAST_MARKER, World
from tests.support.memory import add_fact
from twin.channel.ilink.channel import IlinkChannel
from twin.engine.pipeline import ReplyPipeline
from twin.engine.turns import BotTurnStore
from twin.engine.types import Bubble, InboundItem
from twin.eval.blind import EvalError, check_backends, expand_backends
from twin.eval.channel import InMemoryChannel
from twin.eval.isolation import IsolationViolation, changes, isolated_writes, snapshot
from twin.eval.render import render_candidate
from twin.eval.samples import StickerDescriber
from twin.eval.sandbox import (
    CONTEXT_TURNS,
    EvalSandbox,
    SampleClock,
    SandboxError,
    SandboxKit,
    SandboxMode,
    SandboxRequest,
    backend_status,
    build_sandbox,
    merged_turns,
    stable_seed,
    with_backend,
)
from twin.eval.store import EvalStore
from twin.llm.tokens import CALIBRATION_KEY
from twin.memory.asof import AsOfView
from twin.memory.recent import Turn
from twin.retrieval.query import QueryTurn
from twin.stickers.catalog import StickerCatalog
from twin.storage.models import CostLedger
from twin.storage.settings_store import put_setting
from twin.storage.state import STATE_VERSION_KEY

SRC = Path(__file__).resolve().parents[2] / "src" / "twin"


async def test_a_reply_is_made_for_the_moment_of_the_sample(
    world: World, kit: SandboxKit, script: DeepSeekScript
) -> None:
    sample = sample_of(world)[0]
    reply = await kit.sandbox.reply(request_for(sample))
    assert reply.usable and not reply.fell_back and reply.backend == "deepseek"
    assert [line.text for line in reply.candidate.lines] == ["好呀", "哈哈[拥抱]"]
    assert len(reply.delivered) == 2 and all(m.at == sample.at for m in reply.delivered)
    assert len(script.requests) == 1
    assert world.services.settings.deepseek.chat_model == script.requests[0]["model"]
    with world.services.db.session() as session:
        rows = [
            (r.purpose, r.account, r.batch_id) for r in session.scalars(select(CostLedger)).all()
        ]
    # booked as evaluation on the one-time batch, not as a reply of the bot
    assert rows == [("eval", "one_time", "eval-test-1")]


async def test_the_prompt_of_a_past_moment_holds_only_what_was_known_then(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """R-TRN-013: a fact known at ``t`` or later is not in the prompt; the card is the old one."""
    sample = sample_of(world)[0]
    before = "一秒之前才知道的事蓝色自行车"
    after = "一秒之后才知道的事红色风筝"
    add_fact(world.memory, before, sample.at - timedelta(seconds=1), importance=5)
    add_fact(world.memory, after, sample.at + timedelta(seconds=1), importance=5)
    async with open_kit(world) as kit:
        await kit.sandbox.reply(request_for(sample))
    system, last = script.system_texts[0], script.last_texts[0]
    assert before in last and after not in last and LATE_FACT not in last
    assert PAST_MARKER in system and LIVE_STYLE not in system  # the pre-holdout card
    assert sample.real.lines[0].text not in last  # nor does the reply it is tested against leak
    local = sample.at.astimezone(ZoneInfo("America/Chicago"))
    # no strftime("%-m"): the flag that drops the leading zero is a glibc extension
    assert f"{local.year}年{local.month}月{local.day}日" in last


async def test_every_backend_is_given_the_same_eight_merged_turns(
    world: World, kit: SandboxKit, script: DeepSeekScript
) -> None:
    sample = sample_of(world)[0]
    request = request_for(sample)
    turns = tuple(
        Turn("bot" if i % 2 else "user", f"第{i}轮", sample.at, sample.at, (f"t{i}",))
        for i in range(12)
    )
    long = replace(request, history=turns)
    context = kit.sandbox.context(long)
    assert [t.text for t in context.history] == [f"第{i}轮" for i in range(5, 12)]
    assert len(context.history) == CONTEXT_TURNS - 1  # the round being answered is the eighth
    await kit.sandbox.reply(long)
    sent = " ".join(str(m["content"]) for m in script.requests[0]["messages"])
    assert all(f"第{i}轮" in sent for i in range(5, 12))  # the newest seven turns are there
    assert not any(f"第{i}轮" in sent for i in range(5))  # the older ones are not
    assert merged_turns(turns, 0) == ()
    # the channel decides whether the pipeline may quote: the in-memory channel can
    assert context.limits.supports_quote


async def test_a_run_changes_only_the_tables_the_sandbox_may_write(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """R-EVAL-009: rows and checksums of every table before and after; only the allowed differ."""
    samples = sample_of(world, 3)
    async with open_kit(world) as kit:
        before = snapshot(world.services.db)
        for sample in samples:
            assert (await kit.sandbox.reply(request_for(sample))).usable
        after = snapshot(world.services.db)
    found = changes(before, after)
    assert found.outside() == []
    assert found.tables == {"cost_ledger"}  # the calls were booked, nothing else moved
    for table in (
        "bot_turns",
        "conversation_state",
        "feedback",
        "facts",
        "lifeline_events",
        "daily_plans",
        "followups",
        "daily_summaries",
        "messages",
        "stickers",
        "sticker_uses",
        "profile_versions",
        "persona_cards",
        "example_windows",
    ):
        assert before.tables[table] == after.tables[table], table
    assert len(script.requests) == 3


async def test_building_the_sandbox_only_installs_the_default_prompt_texts(
    world: World, api: respx.MockRouter
) -> None:
    """The one thing construction writes is what the running engine's construction writes."""
    before = snapshot(world.services.db)
    async with open_kit(world):
        after = snapshot(world.services.db)
    found = changes(before, after)
    assert found.tables <= {"prompt_templates", "settings"}
    assert all(key.startswith("template.active.") for key in found.settings)


async def test_the_live_mode_is_isolated_too_and_reads_the_present(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """The memory test asks in ``live`` mode: the live card and all the memory, no other write."""
    now = world.services.clock.now_utc()
    fact = "刚刚才知道的事黄色雨伞"
    add_fact(world.memory, fact, now - timedelta(minutes=5), importance=5)
    async with open_kit(world, SandboxMode.LIVE) as kit:
        before = snapshot(world.services.db)
        question = InboundItem("q-1", now, "text", "你还记得吗")
        reply = await kit.sandbox.reply(SandboxRequest((question,), (), now))
        after = snapshot(world.services.db)
    assert reply.usable
    system, last = script.system_texts[0], script.last_texts[0]
    assert LIVE_STYLE in system and PAST_MARKER not in system
    assert fact in last  # the fact of five minutes ago: the live memory has all of it
    # today's day plan is not made by the sandbox: her state is simply not in the prompt
    assert changes(before, after).tables == {"cost_ledger"}
    assert before.tables["daily_plans"] == after.tables["daily_plans"]
    assert "她现在的状态" not in last


async def test_a_write_outside_the_allowed_tables_is_refused_inside_the_sandbox(
    world: World,
) -> None:
    services = world.services
    store = BotTurnStore(services.db, services.clock)
    with isolated_writes():
        with pytest.raises(IsolationViolation, match="bot_turns"):
            store.add_inbound(at=services.clock.now_utc(), kind="text", text="不该写进去")
        with pytest.raises(IsolationViolation, match="facts"):
            add_fact(world.memory, "不该写进去的事实", services.clock.now_utc())
        with (
            pytest.raises(IsolationViolation, match="schema"),
            services.db.transaction(bump_state=False) as session,
        ):
            session.execute(text("CREATE TABLE sneaky (x INTEGER)"))
        with (
            pytest.raises(IsolationViolation, match="conversation_state"),
            services.db.transaction(bump_state=False) as session,
        ):
            session.execute(text("DELETE FROM conversation_state"))
        with (
            pytest.raises(IsolationViolation, match="setting"),
            services.db.transaction(bump_state=False) as session,
        ):
            put_setting(session, "engine.paused_until", "2030-01-01", clock=services.clock)
    # what it may write passes: its own tables, the state version, the token calibration
    runs = EvalStore(services.db, services.clock)
    with isolated_writes():
        run = runs.create_run("blind", mode="holdout", backends=["deepseek"])
        with services.db.transaction(bump_state=True):
            pass
        with services.db.transaction(bump_state=False) as session:
            put_setting(session, CALIBRATION_KEY, {"x": 1}, clock=services.clock)
            put_setting(session, "onetime.batch.eval-x", {"a": 1}, clock=services.clock)
    assert runs.get_run(run.id).kind == "blind"
    # and outside the block nothing is held back
    store.add_inbound(at=services.clock.now_utc(), kind="text", text="在外面可以写")


async def test_the_snapshot_notices_a_change_and_names_what_is_not_allowed(world: World) -> None:
    services = world.services
    before = snapshot(services.db)
    BotTurnStore(services.db, services.clock).add_inbound(
        at=services.clock.now_utc(), kind="text", text="一条"
    )
    with services.db.transaction(bump_state=False) as session:
        put_setting(session, "engine.paused_until", "2030-01-01", clock=services.clock)
        put_setting(session, STATE_VERSION_KEY, 99, clock=services.clock)
    found = changes(before, snapshot(services.db))
    assert found.tables == {"bot_turns", "settings"}
    assert found.settings == {"engine.paused_until", STATE_VERSION_KEY}
    assert found.outside() == ["bot_turns", "settings:engine.paused_until"]


def test_the_style_and_hybrid_backends_are_refused_while_no_model_is_deployed(
    world: World,
) -> None:
    """Before round 14 nothing is registered: the sandbox says so instead of falling back."""
    services = world.services
    assert backend_status(services, "deepseek").available
    for name in ("style", "hybrid"):
        status = backend_status(services, name)
        assert not status.available and "未部署" in status.message and name in status.message
    unknown = backend_status(services, "gpt")
    assert not unknown.available and "unknown backend" in unknown.message
    with pytest.raises(EvalError, match="未部署"):
        check_backends(services, ["deepseek", "style"])
    check_backends(services, ["deepseek"])
    assert expand_backends("all") == ("deepseek", "style", "hybrid")
    with pytest.raises(EvalError, match="unknown backend"):
        expand_backends("gpt")


async def test_a_reply_by_another_backend_than_the_asked_one_is_marked_as_such(
    world: World, kit: SandboxKit, script: DeepSeekScript
) -> None:
    """The style model is not registered, so the pipeline falls back: the test must not count it."""
    sample = sample_of(world)[0]
    reply = await kit.sandbox.reply(request_for(sample, "style"))
    assert reply.fell_back and reply.backend == "deepseek" and reply.requested_backend == "style"
    assert "backend_error" in {a.step for a in reply.draft.actions}


async def test_the_sandbox_is_the_one_reply_pipeline_with_an_in_memory_channel(
    world: World, kit: SandboxKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-EVAL-009: the live engine and the sandbox run the same ``ReplyPipeline``."""
    assert type(kit.sandbox.pipeline) is ReplyPipeline
    assert type(kit.sandbox.channel) is InMemoryChannel
    # the running application builds its pipeline with the very constructor the sandbox uses
    component = (SRC / "engine" / "component.py").read_text(encoding="utf-8")
    assert "ReplyPipeline.from_services" in component
    seen: list[object] = []
    original = ReplyPipeline.run

    async def spy(self: ReplyPipeline, context: object, data: object) -> object:
        seen.append(data)
        return await original(self, context, data)  # type: ignore[arg-type]

    monkeypatch.setattr(ReplyPipeline, "run", spy)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the sandbox must never build the WeChat channel")

    monkeypatch.setattr(IlinkChannel, "__init__", refuse)
    sample = sample_of(world)[0]
    await kit.sandbox.reply(request_for(sample))
    assert len(seen) == 1 and isinstance(seen[0], AsOfView) and seen[0].at == sample.at
    # nothing in the evaluation produces a reply except through the pipeline
    for path in sorted((SRC / "eval").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "generate" not in called, path.name  # no backend is called directly


async def test_a_sticker_is_sent_through_the_library_and_shown_by_its_description(
    world: World, api: respx.MockRouter, script: DeepSeekScript
) -> None:
    """The sticker the bot picks goes out like a real one and is shown like hers (R-SAFE-006)."""
    md5s = make_stickers_available(world.services)
    catalog = StickerCatalog(world.services)
    for md5 in md5s:
        catalog.save_vision(
            md5, ["开心"], "笑得很开心的猫", "好消息", at=world.services.clock.now_utc()
        )
    script.reply = lambda body: "好呀\n[表情包:开心]"
    sample = sample_of(world, 6)[-1]  # a late one: her stickers were used before it
    async with open_kit(world) as kit:
        reply = await kit.sandbox.reply(request_for(sample))
        describe = StickerDescriber(StickerCatalog(world.services))
    assert reply.usable and reply.skipped_stickers == 0
    assert [line.kind for line in reply.candidate.lines] == ["text", "sticker"]
    assert reply.candidate.lines[1].sticker_md5 in md5s
    image = next(m for m in reply.delivered if m.kind == "image")
    record = catalog.require(reply.candidate.lines[1].sticker_md5 or "")
    assert image.sha256 == record.sha256 and image.mime == record.mime
    assert render_candidate(reply.candidate, describe) == "好呀\n[表情包：笑得很开心的猫]"


async def test_a_sticker_that_cannot_be_sent_is_left_out_like_the_bubble_sender_does(
    world: World, kit: SandboxKit
) -> None:
    sample = sample_of(world)[0]
    reply = await kit.sandbox.reply(request_for(sample))
    draft = replace(
        reply.draft,
        bubbles=(
            Bubble("text", "好呀"),
            Bubble("sticker", "[表情包:开心]", "f" * 32, "开心"),  # a sticker the library lacks
        ),
        quote=None,
    )
    delivered, candidate, skipped = await kit.sandbox._deliver(draft, request_for(sample).inbound)
    assert skipped == 1 and [line.text for line in candidate.lines] == ["好呀"]
    assert len(delivered) == 1


async def test_a_quote_goes_with_the_first_text_bubble_to_the_message_it_comes_from(
    world: World, kit: SandboxKit
) -> None:
    sample = sample_of(world)[0]
    request = request_for(sample)
    fragment = request.inbound[0].text[:6]
    reply = await kit.sandbox.reply(request)
    draft = replace(
        reply.draft, bubbles=(Bubble("text", "好呀"), Bubble("text", "再说一句")), quote=fragment
    )
    delivered, candidate, _ = await kit.sandbox._deliver(draft, request.inbound)
    assert candidate.quote == fragment
    assert delivered[0].quote is not None and delivered[0].quote.message_id == request.inbound[0].id
    assert delivered[1].quote is None  # only the first bubble carries it
    # a fragment that is in none of the messages quotes the last one
    odd = replace(draft, quote="完全没有出现过的话")
    again, shown, _ = await kit.sandbox._deliver(odd, request.inbound)
    assert again[0].quote is not None and again[0].quote.message_id == request.inbound[-1].id
    assert shown.quote == "完全没有出现过的话"


async def test_the_preview_is_the_prompt_the_generation_then_sends(
    world: World, kit: SandboxKit, script: DeepSeekScript
) -> None:
    """The estimate of a batch is priced from the very prompt the run uses (R-LLM-014)."""
    request = request_for(sample_of(world)[0])
    previewed = await kit.sandbox.preview(request)
    assert script.requests == []  # a preview sends nothing
    await kit.sandbox.reply(request)
    sent = script.requests[0]["messages"]
    assert [(m["role"], m["content"]) for m in previewed] == [
        (m["role"], m["content"]) for m in sent
    ]


async def test_a_sandbox_cannot_be_built_without_the_data_of_its_mode(world: World) -> None:
    kit_ = build_sandbox(world.services, mode=SandboxMode.HOLDOUT)
    try:
        sandbox = kit_.sandbox
        with pytest.raises(SandboxError, match="as-of source"):
            EvalSandbox(
                world.services,
                sandbox.pipeline,
                mode=SandboxMode.HOLDOUT,
                channel=sandbox.channel,
                clock=SampleClock(world.services.clock),
            )
        with pytest.raises(SandboxError, match="live data source"):
            EvalSandbox(
                world.services,
                sandbox.pipeline,
                mode=SandboxMode.LIVE,
                channel=sandbox.channel,
                clock=SampleClock(world.services.clock),
            )
    finally:
        await kit_.aclose()


def test_the_sample_clock_reads_the_moment_of_the_sample(world: World) -> None:
    clock = SampleClock(world.services.clock)
    assert clock.now_utc() == world.services.clock.now_utc()  # before any sample: the real time
    moment = datetime(2026, 8, 25, 21, 30, tzinfo=UTC)
    clock.set(moment)
    assert clock.now_utc() == moment
    assert clock.monotonic() == world.services.clock.monotonic()  # durations stay real
    assert stable_seed("a", "b") == stable_seed("a", "b") != stable_seed("b", "a")
    with pytest.raises(ValueError, match="naive"):
        clock.set(datetime(2026, 8, 25, 21, 30))  # noqa: DTZ001 - the refusal is the point
    assert with_backend(request_for(sample_of(world)[0]), "hybrid").backend == "hybrid"


async def test_the_data_view_of_the_sandbox_is_the_pre_holdout_one_with_earlier_examples_only(
    world: World, kit: SandboxKit
) -> None:
    """R-TRN-013: persona and profile are pre-holdout; every example is from before ``t``."""
    samples = sample_of(world, 6)
    for sample in samples:
        view = kit.sandbox.view(sample.at)
        assert isinstance(view, AsOfView) and view.at == sample.at
        full, compact = view.persona_full(), view.persona_compact()
        assert full is not None and full.scope == "pre_holdout"
        assert compact is not None and compact.scope == "pre_holdout"
        assert view.profile is not None and view.profile.metrics.scope == "pre_holdout"
        assert view.lifeline == () and view.bot_turns == ()  # no life line, no conversation then
        turns = [QueryTurn(her=False, text=line) for line in sample.shown[-1]["lines"]]
        found = await view.examples(turns, k=8)
        assert found and all(example.reply_at < sample.at for example in found)
        assert all(example.reply_at < world.cutoff for example in found)  # the hold-out never
        assert sample.real.lines[0].text not in {
            line.text for example in found for line in example.reply
        }
