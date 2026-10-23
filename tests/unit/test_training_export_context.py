"""The context of a training sample and the proactive form (R-TRN-002, R-TRN-005, R-TRN-013).

Round 17, item 6 (D-642, D-643) and item 49e (D-644): the context is what the running bot would
read - the last eight *merged* turns whatever the pauses between them were - and a reply that no
user turn precedes is a proactive sample that ends with the fixed cue turn the proactive path uses.
The scenarios are small hand-made conversations of July 2026 (before the synthetic chat starts) and
of the end of August (after its hold-out cutoff); nothing here is a real conversation.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.support.embedding import HashingBackend
from tests.support.export_plans import run_with_plans
from tests.support.export_world import World, build_world
from tests.support.memory import add_fact, mark_days_replayed
from tests.support.style_models import ScriptedStyleClient, register_model
from tests.support.synth_chat import ChatSpec, MessageWriter
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.engine.backend import BackendRequest
from twin.engine.style_backend import StyleWriter
from twin.engine.style_models import StyleModels
from twin.engine.style_prompt import (
    PROACTIVE_CUE_TEXT,
    WOKE_UP_LINE,
    StylePromptBuilder,
    StyleTurn,
    with_proactive_cue,
)
from twin.engine.types import InboundItem, ReplyContext, ReplyMaterial
from twin.memory.asof import AsOfSource
from twin.memory.recent import Turn
from twin.profile.holdout import holdout_cutoff
from twin.schedule.proactive import decide
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.training import lf_template
from twin.training.dataset_dir import DatasetDir
from twin.training.export import ExportOptions, ExportResult, TrainingSetExporter
from twin.training.export_blocks import MessageLoader, iter_block_refs
from twin.training.export_text import context_text
from twin.training.parity_cases import TAG_PROACTIVE
from twin.training.plans import plan_selected

TOKENIZER = tiny_qwen_tokenizer()
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "twin"


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=12)


def moment(day: int, hour: int, minute: int = 0, *, month: int = 7) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def put(world: World, lines: list[tuple[datetime, bool, str]]) -> None:
    """Add messages to the conversation: ``(when, hers, text)``."""
    writer = MessageWriter(world.services)
    for when, hers, text in lines:
        writer.add(when, hers, "text", text)
    writer.store(append=True)
    mark_days_replayed(world.services)  # the replay covers the new days as well (R18-F7)


def export(world: World, directory: Path, *, ratio: float = 0.0) -> tuple[ExportResult, DatasetDir]:
    exporter = TrainingSetExporter(
        world.services, TOKENIZER, ExportOptions(out_dir=directory, plan_ratio=ratio)
    )
    result = run_with_plans(world.services, exporter)
    assert result.dataset is not None, result.state
    return result, result.dataset


def rows_of(dataset: DatasetDir) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        path = dataset.file_path(name)
        found.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return found


def sample_at(world: World, dataset: DatasetDir, when: datetime) -> dict[str, Any]:
    """The sample whose reply block starts at ``when``."""
    ids = {r.sample_id for r in iter_block_refs(world.services) if r.reply_at == when}
    assert len(ids) == 1, f"no single reply block starts at {when}"
    (wanted,) = ids
    found = [row for row in rows_of(dataset) if row["id"] == wanted]
    assert found, f"the block at {when} is not a sample"
    return found[0]


def talk(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m["from"], m["value"]) for m in row["conversations"]]


# ------------------------------------------------------- a reply after a long pause (D-642)


def test_r17_6_a_reply_after_a_long_gap_is_a_sample_with_the_old_context(
    world: World, tmp_path: Path
) -> None:
    """R17-6: a reply 200 minutes after the user's message used to have no context and was lost."""
    reply_at = moment(1, 20, 25)
    put(
        world,
        [
            (moment(1, 17, 0), False, "今天去哪儿了"),
            (moment(1, 17, 1), True, "去公园了"),
            (moment(1, 17, 5), False, "好玩吗"),
            (reply_at, True, "好玩呀下次一起去"),
        ],
    )
    # what the evaluation draws and what the export reads are different on purpose
    default = next(r for r in iter_block_refs(world.services) if r.reply_at == reply_at)
    assert default.context_ids == ()
    across = next(
        r for r in iter_block_refs(world.services, across_gaps=True) if r.reply_at == reply_at
    )
    assert len(across.context_ids) == 3

    result, dataset = export(world, tmp_path / "ds")
    row = sample_at(world, dataset, reply_at)
    assert talk(row) == [
        ("human", "今天去哪儿了"),
        ("gpt", "去公园了"),
        ("human", "好玩吗"),
        ("gpt", "好玩呀下次一起去"),
    ]
    assert result.stats["samples"]["after_gap"] >= 1
    assert "no_user_turn_to_answer" not in result.stats["dropped"]


def test_r17_6_a_follow_up_burst_ends_with_the_cue_turn(world: World, tmp_path: Path) -> None:
    """R17-6: her second burst in a row answers nobody; the cue stands in for the user's turn."""
    follow_up = moment(2, 17, 10)
    put(
        world,
        [
            (moment(2, 17, 0), False, "在干嘛"),
            (moment(2, 17, 1), True, "刚下班"),
            (follow_up, True, "你呢"),
        ],
    )
    result, dataset = export(world, tmp_path / "ds")
    row = sample_at(world, dataset, follow_up)
    assert talk(row) == [
        ("human", "在干嘛"),
        ("gpt", "刚下班"),
        ("human", PROACTIVE_CUE_TEXT),
        ("gpt", "你呢"),
    ]
    assert result.stats["samples"]["proactive"] >= 1


def test_r17_6_her_first_message_is_a_cue_sample(world: World, tmp_path: Path) -> None:
    opening = moment(1, 12, 0, month=6)
    put(world, [(opening, True, "早呀")])
    _, dataset = export(world, tmp_path / "ds")
    row = sample_at(world, dataset, opening)
    assert talk(row) == [("human", PROACTIVE_CUE_TEXT), ("gpt", "早呀")]


def test_r17_6_the_context_is_eight_merged_turns_across_a_gap(world: World, tmp_path: Path) -> None:
    """R17-6: bursts of one side are one turn, as the live prompt merges them; pauses do not cut."""
    reply_at = moment(5, 17, 30)
    put(
        world,
        [
            (moment(5, 11, 30), False, "u0"),
            (moment(5, 11, 33), True, "h0"),
            (moment(5, 12, 0), False, "u1"),
            (moment(5, 12, 3), True, "h1"),
            (moment(5, 12, 6), True, "h1b"),  # a second burst of hers: the same turn as h1
            (moment(5, 12, 9), False, "u2"),
            (moment(5, 12, 12), True, "h2"),
            (moment(5, 12, 15), False, "u3"),
            (moment(5, 12, 18), False, "u3b"),
            (moment(5, 12, 21), True, "h3"),
            (moment(5, 12, 24), False, "u4"),
            (reply_at, True, "h4"),  # five hours later
        ],
    )
    ref = next(
        r for r in iter_block_refs(world.services, across_gaps=True) if r.reply_at == reply_at
    )
    assert [len(turn) for turn in ref.context_ids] == [1, 1, 2, 1, 1, 2, 1, 1]
    assert (
        next(r for r in iter_block_refs(world.services) if r.reply_at == reply_at).context_ids == ()
    )

    result, dataset = export(world, tmp_path / "ds")
    row = sample_at(world, dataset, reply_at)
    # eight merged turns: the oldest of them is hers and opens the context, so it is the prelude
    assert row["system"].endswith("【前文】\n她：h0")
    assert talk(row) == [
        ("human", "u1"),
        ("gpt", "h1\nh1b"),
        ("human", "u2"),
        ("gpt", "h2"),
        ("human", "u3\nu3b"),
        ("gpt", "h3"),
        ("human", "u4"),
        ("gpt", "h4"),
    ]
    assert result.stats["prelude_samples"] >= 1


def test_r17_6_the_default_context_of_the_evaluation_is_not_changed(
    world: World, tmp_path: Path
) -> None:
    """The draw of the evaluation (``across_gaps`` unset) still cuts at a pause and keeps bursts."""
    put(
        world,
        [
            (moment(6, 12, 0), False, "a"),
            (moment(6, 12, 3), True, "b"),
            (moment(6, 12, 6), True, "c"),
            (moment(6, 12, 9), False, "d"),
            (moment(6, 12, 12), True, "e"),
        ],
    )
    ref = next(r for r in iter_block_refs(world.services) if r.reply_at == moment(6, 12, 12))
    assert [len(turn) for turn in ref.context_ids] == [1, 1, 1, 1]  # bursts, not merged turns
    merged = next(
        r
        for r in iter_block_refs(world.services, across_gaps=True)
        if r.reply_at == moment(6, 12, 12)
    )
    assert [len(turn) for turn in merged.context_ids] == [1, 2, 1]


# ----------------------------------------------------------------- the cue is one constant


def test_r17_6_the_cue_text_is_one_constant() -> None:
    assert decide.CUE_TEXT is PROACTIVE_CUE_TEXT
    cued = with_proactive_cue([StyleTurn("assistant", "刚下班")])
    assert cued == [StyleTurn("assistant", "刚下班"), StyleTurn("user", PROACTIVE_CUE_TEXT)]
    assert with_proactive_cue([]) == [StyleTurn("user", PROACTIVE_CUE_TEXT)]
    literal = PROACTIVE_CUE_TEXT.strip("（）")
    holders = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in SOURCE_ROOT.rglob("*.py")
        if literal in path.read_text(encoding="utf-8")
    ]
    assert holders == ["engine/style_prompt.py"]


async def test_r17_6_the_cue_prompt_equals_what_the_proactive_path_renders(
    world: World, tmp_path: Path
) -> None:
    """The prompt of a cue sample and the one ``StyleWriter`` renders for the proactive path."""
    services = world.services
    opening, follow_up = moment(25, 14, 0, month=8), moment(25, 14, 10, month=8)
    put(world, [(opening, True, "起床啦"), (follow_up, True, "今天天气好好")])
    assert follow_up > holdout_cutoff(services)  # a test sample: it carries no plan
    _, dataset = export(world, tmp_path / "ds")
    row = sample_at(world, dataset, follow_up)
    assert row["conversations"][-2] == {"from": "human", "value": PROACTIVE_CUE_TEXT}
    assert "【规划】" not in row["system"]
    exported_prompt = lf_template.render_prompt(
        row["system"],
        [
            lf_template.Turn("user" if i % 2 == 0 else "assistant", m["value"])
            for i, m in enumerate(row["conversations"][:-1])
        ],
    )

    # the proactive path: the history up to now, the cue as the message that is "answered"
    ref = next(r for r in iter_block_refs(services, across_gaps=True) if r.reply_at == follow_up)
    (block,) = MessageLoader(services).load([ref])
    label = StickerCatalog(services).tag_lookup()
    history = tuple(
        Turn(
            "user" if turn[0].data.is_sent else "bot",
            context_text([m.data for m in turn], label),
            turn[0].at,
            turn[-1].at,
            tuple(m.data.id for m in turn),
        )
        for turn in block.context
    )
    register_model(services, persona_version="v1")
    client = ScriptedStyleClient()
    writer = StyleWriter(
        client=client,
        models=StyleModels(services.db, mode=services.settings.style_model.mode),
        builders=lambda locked: StylePromptBuilder.from_services(services, locked=locked),
    )
    view = AsOfSource(services).at(follow_up)
    context = ReplyContext(
        inbound=(InboundItem("proactive", follow_up, "text", decide.CUE_TEXT),),
        history=history,
        backend="style",
        woke_up=False,
    )
    material = ReplyMaterial(
        local=view.local,
        state=None,
        lifeline=None,
        persona="",
        persona_ref=None,
        memory_text="",
        memory_item_ids=(),
        examples=(),
        examples_text="",
        closing=None,
        asks_if_ai=False,
    )
    await writer.write(BackendRequest(context, view, material, (), 1))
    assert client.prompts[0].text == exported_prompt


# ----------------------------------------------- a plan for every proactive sample (D-643)


def test_r17_6_every_proactive_sample_before_the_holdout_carries_a_plan_whatever_the_ratio(
    world: World, tmp_path: Path
) -> None:
    """D6: online the proactive path always hands the style model a plan, so the samples do."""
    put(
        world,
        [
            (moment(2, 17, 0), False, "在干嘛"),
            (moment(2, 17, 1), True, "刚下班"),
            (moment(2, 17, 10), True, "你呢"),
        ],
    )
    exporter = TrainingSetExporter(
        world.services, TOKENIZER, ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    )
    first = exporter.run()
    assert first.state == "waiting_for_plans" and first.dataset is None
    assert not (tmp_path / "ds").exists()
    assert 0 < len(first.missing_plans) == first.plan_selected
    refs = {r.sample_id: r for r in iter_block_refs(world.services, across_gaps=True)}
    assert all(not plan_selected(i, 0.0) for i in first.missing_plans)  # not the ratio's choice
    assert all(refs[i].reply_at < world.cutoff for i in first.missing_plans)

    result = run_with_plans(world.services, exporter)
    assert result.dataset is not None
    planned: list[bool] = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        for line in result.dataset.file_path(name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cue = row["conversations"][-2]["value"] == PROACTIVE_CUE_TEXT
            has_plan = "【规划】" in row["system"]
            if name != "sft_test.jsonl":
                assert has_plan == cue, "a plan exactly where the context ends with the cue"
            else:
                assert not has_plan, "a test sample never carries a plan"
            planned.append(has_plan)
    assert any(planned)
    plans = result.stats["plans"]
    assert plans["selected"] == plans["planned"] == len(first.missing_plans)


def test_r17_6_the_ratio_adds_ordinary_samples_to_the_proactive_ones(
    world: World, tmp_path: Path
) -> None:
    exporter = TrainingSetExporter(
        world.services, TOKENIZER, ExportOptions(out_dir=tmp_path / "ds", plan_ratio=0.0)
    )
    proactive_only = exporter.run().plan_selected
    with_ratio = TrainingSetExporter(
        world.services, TOKENIZER, ExportOptions(out_dir=tmp_path / "ds2", plan_ratio=0.5)
    ).run()
    assert with_ratio.plan_selected > proactive_only > 0


# ----------------------------------------------------------- statistics and parity cases


def test_r17_6_the_statistics_count_the_proactive_and_the_after_gap_samples(
    world: World, tmp_path: Path
) -> None:
    result, dataset = export(world, tmp_path / "ds")
    samples = result.stats["samples"]
    cue_rows = [
        r for r in rows_of(dataset) if r["conversations"][-2]["value"] == PROACTIVE_CUE_TEXT
    ]
    assert samples["proactive"] == len(cue_rows) > 0
    assert 0 < samples["after_gap"] < samples["total"]
    assert result.stats["dropped"].get("no_user_turn_to_answer", 0) == 0


def test_r17_6_the_template_check_has_proactive_cases(world: World, tmp_path: Path) -> None:
    _, dataset = export(world, tmp_path / "ds")
    cases = [
        json.loads(line)
        for line in dataset.file_path("parity_cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    proactive = [case for case in cases if TAG_PROACTIVE in case["tags"]]
    assert proactive
    assert all(PROACTIVE_CUE_TEXT in case["prompt"] for case in proactive)


# --------------------------------------------------- nothing from the future (R-TRN-013)


def test_r17_6_a_fact_learned_at_or_after_a_cue_sample_is_not_in_its_prompt(
    world: World, tmp_path: Path
) -> None:
    """R-TRN-013 for the new samples: the cue turn is part of the memory query, not a leak."""
    follow_up = moment(2, 17, 10)
    put(
        world,
        [
            (moment(2, 17, 0), False, "在干嘛"),
            (moment(2, 17, 1), True, "刚下班买了点水果"),
            (follow_up, True, "你呢"),
        ],
    )
    secret = f"{PROACTIVE_CUE_TEXT} 刚下班买了点水果 只有之后才知道的秘密"
    add_fact(world.memory, secret, follow_up)  # known at the very moment: not before it
    _, dataset = export(world, tmp_path / "late")
    row = sample_at(world, dataset, follow_up)
    assert "只有之后才知道的秘密" not in json.dumps(row, ensure_ascii=False)
    times = {r.sample_id: r.reply_at for r in iter_block_refs(world.services, across_gaps=True)}
    earlier = [other for other in rows_of(dataset) if times[other["id"]] <= follow_up]
    assert earlier  # the samples of the same day and before, none of which may know it
    for other in earlier:
        assert "只有之后才知道的秘密" not in json.dumps(other, ensure_ascii=False)

    # the control: the same fact, known a second earlier, does reach the prompt of that sample
    add_fact(
        world.memory,
        f"{PROACTIVE_CUE_TEXT} 刚下班买了点水果 只有更早才知道的秘密",
        follow_up - timedelta(seconds=1),
    )
    _, again = export(world, tmp_path / "early")
    assert "只有更早才知道的秘密" in sample_at(world, again, follow_up)["system"]


# ----------------------------------------------------- she has just woken up (D-644, 49e)


@pytest.fixture
def night_world(services: Services, embedder: HashingBackend) -> World:
    """The user writes at 03:10 and she answers when she wakes (``ChatSpec.night_message``)."""
    return build_world(services, embedder, spec=ChatSpec(days=12, night_message=True))


def test_r17_49e_an_answer_to_a_night_message_says_she_has_just_woken_up(
    night_world: World, tmp_path: Path
) -> None:
    _, dataset = export(night_world, tmp_path / "ds")
    rows = rows_of(dataset)
    woke = [row for row in rows if WOKE_UP_LINE in row["system"]]
    assert woke, "the morning answers to the messages of the night"
    assert len(woke) < len(rows)
    for row in woke:
        assert row["system"].index(WOKE_UP_LINE) > row["system"].index("【此刻】")
    # the control: the daytime exchanges of the same data do not say it
    assert any(WOKE_UP_LINE not in row["system"] for row in rows)


def test_r17_49e_a_greeting_after_her_waking_says_she_has_just_woken_up(
    world: World, tmp_path: Path
) -> None:
    """A proactive sample 30 minutes after the routine's sleep ends is the greeting's situation."""
    # local time is UTC-5 in July: the routine has her deep asleep until about 07:00
    greeting = moment(10, 12, 30)  # 07:30 local
    afternoon = moment(10, 19, 0)  # 14:00 local: awake for hours
    put(
        world,
        [
            (moment(9, 3, 0), False, "晚安"),  # 22:00 local, the evening before
            (moment(9, 3, 5), True, "晚安晚安"),
            (greeting, True, "早呀"),
            (afternoon, True, "在忙吗"),
        ],
    )
    _, dataset = export(world, tmp_path / "ds")
    assert WOKE_UP_LINE in sample_at(world, dataset, greeting)["system"]
    assert WOKE_UP_LINE not in sample_at(world, dataset, afternoon)["system"]


def plan_section(row: dict[str, Any]) -> str:
    return str(row["system"]).split("【规划】\n", 1)[1].split("\n\n【", 1)[0]


def test_r17_49e_the_plan_carries_the_sticker_tag_of_the_target(
    world: World, tmp_path: Path
) -> None:
    """The online planner may name a sticker tag; the export derives it from the real target."""
    _, dataset = export(world, tmp_path / "ds", ratio=1.0)
    planned = [row for row in rows_of(dataset) if "【规划】" in row["system"]]
    with_sticker = [r for r in planned if "[表情包:" in r["conversations"][-1]["value"]]
    without = [r for r in planned if "[表情包" not in r["conversations"][-1]["value"]]
    assert with_sticker and without
    for row in with_sticker:
        target = row["conversations"][-1]["value"]
        first = re.search(r"\[表情包:([^\]]+)\]", target)
        assert first is not None
        assert plan_section(row).split("\n")[-1] == f"表情包：{first.group(1)}"
    for row in without:
        assert "表情包：" not in plan_section(row)


# ------------------------------------------------- the same arguments as the writer (49e)


async def test_r17_49e_the_training_prompt_uses_the_same_compose_arguments_as_the_writer(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever ``StyleWriter`` hands ``compose``, the export hands it too (D-644)."""
    seen: dict[str, set[str]] = {"export": set(), "writer": set()}
    original = StylePromptBuilder.compose

    def spy(self: StylePromptBuilder, *args: Any, **kwargs: Any) -> Any:
        seen[seen_for[0]].update(kwargs)
        return original(self, *args, **kwargs)

    seen_for = ["export"]
    monkeypatch.setattr(StylePromptBuilder, "compose", spy)
    export(world, tmp_path / "ds")

    seen_for[0] = "writer"
    services = world.services
    register_model(services, persona_version="v1")
    client = ScriptedStyleClient()
    writer = StyleWriter(
        client=client,
        models=StyleModels(services.db, mode=services.settings.style_model.mode),
        builders=lambda locked: StylePromptBuilder.from_services(services, locked=locked),
    )
    when = moment(25, 14, 0, month=8)
    view = AsOfSource(services).at(when)
    context = ReplyContext(inbound=(InboundItem("x", when, "text", "在吗"),), backend="style")
    material = ReplyMaterial(
        local=view.local,
        state=None,
        lifeline=None,
        persona="",
        persona_ref=None,
        memory_text="",
        memory_item_ids=(),
        examples=(),
        examples_text="",
        closing=None,
        asks_if_ai=False,
    )
    await writer.write(BackendRequest(context, view, material, (), 1))
    assert seen["writer"] and seen["export"] == seen["writer"]
    assert {"plan", "woke_up", "memory", "budget"} <= seen["export"]
