"""The training-set export on a synthetic conversation (R-TRN-002 to R-TRN-007, R-TRN-013).

The conversation, its profile, routine, persona cards, stickers and memory come from
``tests.support.export_world``; the tokenizer is a small byte-level BPE that splits text like
Qwen's.  Nothing here is a real conversation.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from tests.support.embedding import HashingBackend
from tests.support.export_plans import run_with_plans
from tests.support.export_world import (
    CORRECTION,
    FACT_MANUAL,
    LATE_FACT,
    LIVE_STYLE,
    PAST_MARKER,
    World,
    build_world,
    past_card,
)
from tests.support.memory import add_fact, mark_days_replayed
from tests.support.synth_chat import MessageWriter
from tests.support.tiny_tokenizer import tiny_qwen_tokenizer
from twin.engine.style_prompt import (
    PROACTIVE_CUE_TEXT,
    WOKE_UP_LINE,
    StylePromptBuilder,
    StyleTurn,
    normalize_context,
    with_proactive_cue,
)
from twin.ingest.events import default_detector
from twin.llm.redaction import ConsistentRedactor
from twin.memory.asof import AsOfSource
from twin.profile.api import load_profile
from twin.profile.holdout import holdout_cutoff
from twin.profile.persona import compose
from twin.retrieval.windows import window_id_of
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.storage.chat_models import Message
from twin.storage.engine_models import BotTurn
from twin.storage.ids import new_id
from twin.training import export as export_module
from twin.training import lf_template
from twin.training.dataset_dir import DatasetDir, load_dataset_dir
from twin.training.export import (
    ExportError,
    ExportOptions,
    ExportResult,
    TrainingSetExporter,
    scan_sample,
    val_count,
)
from twin.training.export_blocks import MessageLoader, iter_block_refs
from twin.training.export_text import collapse, context_text
from twin.training.profiles import CUTOFF_LEN

TOKENIZER = tiny_qwen_tokenizer()
PHONE_A = "138" + "12345678"  # built from parts: the privacy scan sees no number
PHONE_B = "139" + "87654321"


@pytest.fixture
def world(services: Services, embedder: HashingBackend) -> World:
    return build_world(services, embedder, days=12)


def run_export(
    services: Services, directory: Path, *, ratio: float = 0.0, **options: Any
) -> ExportResult:
    exporter = TrainingSetExporter(
        services, TOKENIZER, ExportOptions(out_dir=directory, plan_ratio=ratio, **options)
    )
    return run_with_plans(services, exporter)


def rows(dataset: DatasetDir, name: str) -> list[dict[str, Any]]:
    path = dataset.file_path(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def all_rows(dataset: DatasetDir) -> list[dict[str, Any]]:
    return [
        row
        for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl")
        for row in rows(dataset, name)
    ]


def sample_times(services: Services) -> dict[str, datetime]:
    return {ref.sample_id: ref.reply_at for ref in iter_block_refs(services)}


def exported(services: Services, directory: Path, **options: Any) -> DatasetDir:
    result = run_export(services, directory, **options)
    assert result.dataset is not None
    return result.dataset


# ------------------------------------------------------------------- only her real words


def test_every_target_line_is_a_message_she_wrote(world: World, tmp_path: Path) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    with world.services.db.session() as session:
        messages = [
            (m.is_sent, collapse(m.text)) for m in session.scalars(select(Message)) if m.text
        ]
    hers = {text for sent, text in messages if not sent}
    users = {text for sent, text in messages if sent}
    checked = 0
    for row in all_rows(dataset):
        for line in row["conversations"][-1]["value"].split("\n"):
            if line.startswith(("[表情包", "[引用:")):
                continue
            assert line in hers, "a target line is not a message of hers"
            assert line not in users - hers
            checked += 1
    assert checked > 100


def test_the_bots_own_words_never_reach_the_set(world: World, tmp_path: Path) -> None:
    marker = "机器人说过的独一无二的句子"
    with world.services.db.transaction(bump_state=False) as session:
        now = world.services.clock.now_utc()
        for number, direction in enumerate(("in", "out")):
            session.add(
                BotTurn(
                    id=new_id(),
                    at=datetime(2026, 8, 6, 15, number, tzinfo=UTC),
                    direction=direction,
                    kind="text",
                    text=marker,
                    is_command=False,
                    created_at=now,
                    updated_at=now,
                )
            )
    dataset = exported(world.services, tmp_path / "ds")
    for name in dataset.data_files():
        assert marker not in dataset.file_path(name).read_text(encoding="utf-8")


def test_samples_are_her_reply_blocks_in_time_order(world: World, tmp_path: Path) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    times = sample_times(world.services)
    for name in ("sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"):
        stamps = [times[row["id"]] for row in rows(dataset, name)]
        assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)
    ids = [row["id"] for row in all_rows(dataset)]
    assert len(ids) == len(set(ids)) and set(ids) <= set(times)
    first = next(iter_block_refs(world.services))
    assert first.sample_id == window_id_of(first.reply_ids[0])


# --------------------------------------------------------------------------------- splits


def test_the_split_is_by_time_and_the_test_set_starts_at_the_holdout_cutoff(
    world: World, tmp_path: Path
) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    times = sample_times(world.services)
    cutoff = holdout_cutoff(world.services)
    train, val, test = (
        [times[r["id"]] for r in rows(dataset, f"sft_{name}.jsonl")]
        for name in ("train", "val", "test")
    )
    assert cutoff == world.cutoff and dataset.meta.holdout_cutoff == cutoff.isoformat()
    assert test and all(t >= cutoff for t in test)
    assert train and val and all(t < cutoff for t in [*train, *val])
    assert max(train) < min(val), "the validation set is the latest part before the cut-off"
    before = len(train) + len(val)
    assert len(val) == val_count(before) and val_count(before) >= round(before * 0.05)
    counts = dataset.meta.counts
    assert (counts.train, counts.val, counts.test) == (len(train), len(val), len(test))


def test_every_reply_block_is_a_sample_or_a_counted_drop(world: World, tmp_path: Path) -> None:
    result = run_export(world.services, tmp_path / "ds")
    assert result.dataset is not None
    stats = result.stats
    blocks = sum(1 for ref in iter_block_refs(world.services) if ref.reproducible)
    dropped = sum(stats["dropped"].values())
    assert stats["samples"]["total"] + dropped == blocks
    # R17-6 (D-642): her opening of a conversation, a second block in a row and a reply after a
    # long pause are samples too (the first two end with the proactive cue), none is dropped for
    # lack of a user turn any more
    assert stats["samples"]["proactive"] > 0 and stats["samples"]["after_gap"] > 0
    assert "no_user_turn_to_answer" not in stats["dropped"]
    assert set(stats["dropped"]) <= {"empty_target", "over_budget", "target_too_long"}


def test_a_range_limits_the_samples_but_keeps_the_context_of_its_first_block(
    world: World, tmp_path: Path
) -> None:
    times = sample_times(world.services)
    ordered = sorted(times.values())
    since, until = ordered[len(ordered) // 3], ordered[-1] + timedelta(seconds=1)
    result = run_export(world.services, tmp_path / "ds", since=since, until=until)
    assert result.dataset is not None
    kept = {row["id"] for row in all_rows(result.dataset)}
    assert kept and all(since <= times[i] < until for i in kept)
    assert len(kept) < len(times)
    assert result.dataset.meta.range_from == since.isoformat()
    assert result.dataset.meta.range_to == until.isoformat()


# ------------------------------------------------------------------------- the structure


def test_the_context_starts_with_the_user_and_her_opening_turns_go_to_the_prelude(
    world: World, tmp_path: Path
) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    prelude = 0
    for row in all_rows(dataset):
        roles = [m["from"] for m in row["conversations"]]
        assert roles[0] == "human" and roles[-1] == "gpt" and len(roles) % 2 == 0
        assert roles == ["human", "gpt"] * (len(roles) // 2)
        if "【前文】" in row["system"]:
            prelude += 1
            section = row["system"].split("【前文】\n", 1)[1]
            assert section.startswith("她：")
    assert prelude > 0
    # a block that no user turn precedes still has a context that opens with the user: the cue
    cue = [r for r in all_rows(dataset) if r["conversations"][-2]["value"] == PROACTIVE_CUE_TEXT]
    assert cue and all(r["conversations"][0]["from"] == "human" for r in cue)


_PLAN_SECTION = re.compile(r"\n\n【规划】\n.*?(?=\n\n【|\Z)", re.DOTALL)


def test_the_prompt_is_the_one_the_running_bot_builds(world: World, tmp_path: Path) -> None:
    services = world.services
    dataset = exported(services, tmp_path / "ds")
    source = AsOfSource(services)
    builder = StylePromptBuilder.from_services(services)
    loader = MessageLoader(services)
    label = StickerCatalog(services).tag_lookup()
    by_id = {row["id"]: row for row in all_rows(dataset)}
    checked = proactive = 0
    for ref in iter_block_refs(services, across_gaps=True):
        row = by_id.get(ref.sample_id)
        if row is None:
            continue
        (block,) = loader.load([ref])
        turns = [
            StyleTurn(
                "user" if turn[0].data.is_sent else "assistant",
                context_text([m.data for m in turn], label),
            )
            for turn in block.context
        ]
        turns = [t for t in turns if t.text]
        if not normalize_context(turns).ends_with_user:  # nobody waits: she writes first
            turns = with_proactive_cue(turns)
            proactive += 1
        view = source.at(block.at)
        parts = builder.compose(view, turns, budget=None)
        # the plan and the waking line have tests of their own (test_training_export_context.py)
        written = _PLAN_SECTION.sub("", row["system"]).replace("\n" + WOKE_UP_LINE, "")
        assert written.split("【此刻】")[1] == parts.system.split("【此刻】")[1]
        assert written == parts.system
        rendered = lf_template.render_prompt(
            row["system"],
            [
                lf_template.Turn("user" if i % 2 == 0 else "assistant", m["value"])
                for i, m in enumerate(row["conversations"][:-1])
            ],
        )
        assert rendered.endswith("<|im_start|>assistant\n")
        assert [t.content for t in parts.turns] == [m["value"] for m in row["conversations"][:-1]]
        checked += 1
        if checked >= 40:
            break
    assert checked == 40 and proactive > 0


def test_the_system_message_has_the_moment_the_state_and_the_memory_of_that_time(
    world: World, tmp_path: Path
) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    first = rows(dataset, "sft_train.jsonl")[0]
    assert "【此刻】\n当地时间：2026年8月" in first["system"]
    assert "她现在的状态：" in first["system"]
    assert all(LATE_FACT not in r["system"] for r in all_rows(dataset))


# ------------------------------------------------------------- the persona, profile, routine


def test_the_card_the_profile_and_the_routine_are_the_pre_holdout_ones(
    world: World, tmp_path: Path
) -> None:
    services = world.services
    result = run_export(services, tmp_path / "ds")
    assert result.dataset is not None
    past = load_profile(services, "pre_holdout")
    live = load_profile(services, "live")
    assert past is not None and live is not None and past.version.id != live.version.id
    meta = result.dataset.meta
    assert meta.profile_version == past.version.id and meta.scope == "pre_holdout"
    source = AsOfSource(services)  # what the exporter reads: the pre-holdout routine and profile
    assert source.activity is not None and source.activity.scope == "pre_holdout"
    assert source.profile is not None and source.profile.metrics.scope == "pre_holdout"
    assert past.metrics.scope == "pre_holdout" and live.metrics.scope == "live"
    assert meta.persona_version == "v1"
    assert meta.template_version == lf_template.TEMPLATE_VERSION
    systems = [row["system"] for row in all_rows(result.dataset)]
    assert all(PAST_MARKER in system for system in systems)
    for forbidden in (LIVE_STYLE, FACT_MANUAL, CORRECTION):
        assert not any(forbidden in system for system in systems)
    assert result.stats["persona_scope"] == "pre_holdout"


def test_the_export_needs_the_pre_holdout_derivatives(
    services: Services, embedder: HashingBackend, tmp_path: Path
) -> None:
    from tests.support.synth_chat import ChatSpec, build_chat
    from twin.profile.builder import rebuild
    from twin.profile.persona.store import PersonaStore

    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=6))
    with pytest.raises(ExportError, match="persona card"):
        TrainingSetExporter(services, TOKENIZER)
    PersonaStore(services.db, services.clock).add_version("pre_holdout", past_card(), reason="x")
    with pytest.raises(ExportError, match="profile or routine"):
        TrainingSetExporter(services, TOKENIZER)
    rebuild(services, "all")
    mark_days_replayed(services)
    TrainingSetExporter(services, TOKENIZER)  # now it can start


def test_r18_f7_the_export_needs_the_memory_replay_and_the_persona_description(
    world: World, tmp_path: Path
) -> None:
    """R18-F7 (D-817): RUNBOOK §6.1 promises that the export names what is missing - the replay
    of the memory over the export range and the description part of the pre-holdout card were not
    checked at all (a card with statistics rules only counted as a card).  Both now stop the export
    with the command to run; ``allow_incomplete`` (``--allow-incomplete``) exports anyway and the
    statistics of the dataset record what was missing."""
    from twin.memory.localdate import MemoryClock
    from twin.memory.replay import count_message_days
    from twin.storage.memory_models import MemoryReplayDay

    services = world.services
    days = sorted(count_message_days(services, MemoryClock.from_services(services)))
    with services.db.transaction() as session:
        for row in session.scalars(select(MemoryReplayDay)).all():
            if row.local_date in (days[0], days[-1]):
                session.delete(row)
    with pytest.raises(ExportError, match="twin memory replay start") as caught:
        TrainingSetExporter(services, TOKENIZER)
    message = str(caught.value)
    assert "2 days" in message and str(days[0]) in message and str(days[-1]) in message
    assert "--allow-incomplete" in message
    # a range that leaves out the two days needs no more than its own days
    clock = MemoryClock.from_services(services)
    since = clock.real_bounds(days[1])[0]
    until = clock.real_bounds(days[-2])[1]
    TrainingSetExporter(services, TOKENIZER, ExportOptions(since=since, until=until))
    mark_days_replayed(services, {days[0], days[-1]})
    TrainingSetExporter(services, TOKENIZER)  # everything replayed: it starts
    stats_only = compose.stats_block(["几乎不用逗号"]) + compose.empty_auto_block()
    world.store.add_version("pre_holdout", stats_only + compose.manual_block([], None), reason="x")
    with pytest.raises(ExportError, match="twin persona regenerate --scope pre_holdout"):
        TrainingSetExporter(services, TOKENIZER)
    result = run_export(services, tmp_path / "ds", allow_incomplete=True)
    assert result.dataset is not None
    assert result.dataset.meta.stats["derived_data"] == {
        "complete": False,
        "memory_days_not_replayed": 0,
        "persona_description": False,
    }
    world.store.add_version("pre_holdout", past_card(), reason="generate")
    complete = run_export(services, tmp_path / "ds2")
    assert complete.dataset is not None
    assert complete.dataset.meta.stats["derived_data"]["complete"] is True


# --------------------------------------------------------------------- no future leakage


def a_train_sample(dataset: DatasetDir) -> dict[str, Any]:
    candidates = rows(dataset, "sft_train.jsonl")
    return candidates[len(candidates) // 2]


def memory_of(row: dict[str, Any]) -> str:
    return str(row["system"])


def inject(world: World, text: str, known_at: datetime) -> None:
    add_fact(world.memory, text, known_at)


def test_a_fact_learned_at_or_after_the_block_is_not_in_its_prompt(
    world: World, tmp_path: Path
) -> None:
    services = world.services
    baseline = exported(services, tmp_path / "base")
    row = a_train_sample(baseline)
    moment = sample_times(services)[row["id"]]
    asked = row["conversations"][-2]["value"].replace("\n", " ")
    secret = f"{asked} 只有目标块之后才知道的秘密"
    inject(world, secret, moment)  # known at the very moment of the block: not before it
    dataset = exported(services, tmp_path / "after")
    again = next(r for r in all_rows(dataset) if r["id"] == row["id"])
    assert "只有目标块之后才知道的秘密" not in memory_of(again)
    assert "只有目标块之后才知道的秘密" not in json.dumps(
        again["conversations"], ensure_ascii=False
    )
    later = [
        r
        for r in all_rows(dataset)
        if sample_times(services)[r["id"]] <= moment and "只有目标块之后才知道的秘密" in r["system"]
    ]
    assert later == []


def test_the_same_fact_known_a_moment_earlier_is_in_the_prompt(
    world: World, tmp_path: Path
) -> None:
    """The control of the injection test: the memory does reach the prompt when it may."""
    services = world.services
    baseline = exported(services, tmp_path / "base")
    row = a_train_sample(baseline)
    moment = sample_times(services)[row["id"]]
    asked = row["conversations"][-2]["value"].replace("\n", " ")
    inject(world, f"{asked} 只有目标块之后才知道的秘密", moment - timedelta(seconds=1))
    dataset = exported(services, tmp_path / "after")
    again = next(r for r in all_rows(dataset) if r["id"] == row["id"])
    assert "只有目标块之后才知道的秘密" in memory_of(again)


def test_a_word_of_the_target_block_is_in_no_earlier_prompt(world: World, tmp_path: Path) -> None:
    services = world.services
    marker = "独一无二的暗号"
    times = sample_times(services)
    ordered = sorted(times, key=times.__getitem__)
    target_id = ordered[len(ordered) // 2]
    ref = next(r for r in iter_block_refs(services) if r.sample_id == target_id)
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Message, ref.reply_ids[0])
        assert row is not None
        row.kind, row.text = "text", f"今天说{marker}"
    dataset = exported(services, tmp_path / "ds")
    before = [r for r in all_rows(dataset) if times[r["id"]] < times[target_id]]
    assert before
    for r in before:
        assert marker not in json.dumps(r, ensure_ascii=False)
    itself = next((r for r in all_rows(dataset) if r["id"] == target_id), None)
    if itself is not None:
        assert marker in itself["conversations"][-1]["value"]
        assert marker not in itself["system"]
        assert all(marker not in m["value"] for m in itself["conversations"][:-1])


# ---------------------------------------------------------------------- desensitising


def add_pii(world: World) -> None:
    """Phone numbers and an e-mail address in messages of both sides, early in the data."""
    writer = MessageWriter(world.services)
    day = datetime(2026, 8, 5, 20, 0, tzinfo=UTC)
    lines = [
        (True, f"我的电话是{PHONE_A}，别告诉别人"),
        (False, f"好的，{PHONE_A}我存了，邮箱是 a.b@example.com"),
        (True, f"那{PHONE_B}是谁的"),
        (False, f"{PHONE_B}是我同事的，别打错了"),
        (True, f"再说一遍你的号码{PHONE_A}"),
        (False, "对就是这个"),
    ]
    for number, (sent, text) in enumerate(lines):
        writer.add(day + timedelta(seconds=40 * number), not sent, "text", text)
    writer.store(append=True)
    mark_days_replayed(world.services)  # the replay covers the new day as well (R18-F7)


def test_the_same_person_is_the_same_placeholder_everywhere_and_nothing_is_left(
    world: World, tmp_path: Path
) -> None:
    add_pii(world)
    result = run_export(world.services, tmp_path / "ds")
    assert result.dataset is not None
    texts = [
        result.dataset.file_path(name).read_text(encoding="utf-8")
        for name in result.dataset.data_files()
    ]
    joined = "\n".join(texts)
    for secret in (PHONE_A, PHONE_B, "a.b@example.com"):
        assert secret not in joined
    assert joined.count("[手机号#1]") >= 2 and "[手机号#2]" in joined
    assert "[手机号#3]" not in joined
    assert result.stats["redacted_entities"] == {"phone": 2, "email": 1}
    for row in all_rows(result.dataset):
        scan_sample(export_module.sample_of(row))  # a second scan finds nothing


def test_a_redactor_that_misses_something_makes_the_export_fail_and_leave_nothing(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_pii(world)
    monkeypatch.setattr(ConsistentRedactor, "redact", lambda self, text: _unchanged(text))
    target = tmp_path / "ds"
    with pytest.raises(ExportError, match="personal data"):
        run_export(world.services, target)
    assert not target.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".export-")]


def _unchanged(text: str) -> Any:
    from twin.llm.redaction import RedactionResult

    return RedactionResult(text)


def test_event_text_in_a_target_fails_the_export(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = export_module.render_target

    def with_event(messages: Any, labeler: Any = None) -> Any:
        target = real(messages, labeler)
        target.lines.append("[图片]")
        return target

    monkeypatch.setattr(export_module, "render_target", with_event)
    with pytest.raises(ExportError, match="event text"):
        run_export(world.services, tmp_path / "ds")
    assert not (tmp_path / "ds").exists()


def test_the_scan_names_the_kind_of_personal_data_and_the_sample_never_its_text() -> None:
    sample = export_module.SftSample(
        "w1", "系统", (lf_template.Turn("user", f"打给{PHONE_A}"),), "好的"
    )
    with pytest.raises(ExportError) as raised:
        scan_sample(sample)
    assert "phone" in str(raised.value) and PHONE_A not in str(raised.value)
    clean = export_module.SftSample("w2", "系统", (lf_template.Turn("user", "你好"),), "在呢")
    scan_sample(clean)
    with pytest.raises(ExportError, match="event text"):
        scan_sample(export_module.SftSample("w3", "系统", clean.turns, "好\n[语音 5 秒，未转写]"))


@pytest.mark.parametrize(
    "where",
    ["system", "turn", "reply"],
)
@pytest.mark.parametrize(
    "text",
    ["call me at 312-555-0123.", "+1 312.555.0123.", "(312) 555-0123.", "Tel. 312-555-0123。"],
)
def test_r17_9_the_residual_scan_sees_a_number_at_the_end_of_a_sentence(
    where: str, text: str
) -> None:
    """R17-9: the scan uses the detector, so the dot guard had hidden these from both."""
    system = text if where == "system" else "系统"
    turn = text if where == "turn" else "你好"
    reply = text if where == "reply" else "在呢"
    sample = export_module.SftSample("w9", system, (lf_template.Turn("user", turn),), reply)
    with pytest.raises(ExportError, match=r"personal data \(phone\)"):
        scan_sample(sample)
    clean = export_module.SftSample(
        "w10",
        "系统",
        (lf_template.Turn("user", "版本 3.12.13，pi=3.14159，10.234.567.8901"),),
        "好",
    )
    scan_sample(clean)


# ----------------------------------------------------------------------- long samples


def long_segment(world: World, day: datetime, *, turn_chars: int, turns: int, reply: str) -> None:
    """A conversation of ``turns`` long turns of the user and short ones of hers, then ``reply``."""
    writer = MessageWriter(world.services)
    moment = day
    for number in range(turns):
        her = number % 2 == 1
        text = f"第{number}句" + ("长话" * (turn_chars // 2)) if not her else f"嗯{number}"
        writer.add(moment, her, "text", text)
        moment += timedelta(minutes=2)
    writer.add(moment, True, "text", reply)
    writer.store(append=True)
    mark_days_replayed(world.services)  # the replay covers the new day as well (R18-F7)


def sample_with(dataset: DatasetDir, marker: str) -> dict[str, Any] | None:
    for row in all_rows(dataset):
        if marker in row["conversations"][-1]["value"]:
            return row
    return None


def test_a_long_conversation_loses_its_oldest_turns_and_never_its_target(
    world: World, tmp_path: Path
) -> None:
    reply = "收到了长长的结尾标记"
    long_segment(
        world, datetime(2026, 7, 20, 15, 0, tzinfo=UTC), turn_chars=420, turns=9, reply=reply
    )
    result = run_export(world.services, tmp_path / "ds")
    assert result.dataset is not None
    row = sample_with(result.dataset, reply)
    assert row is not None
    kept = len(row["conversations"]) - 1
    assert kept < 9, "the oldest turns were dropped"
    assert row["conversations"][0]["from"] == "human"
    assert row["conversations"][-2]["value"].startswith("第8句")  # the newest user turn survived
    prompt = lf_template.render_prompt(
        row["system"],
        [
            lf_template.Turn("user" if i % 2 == 0 else "assistant", m["value"])
            for i, m in enumerate(row["conversations"][:-1])
        ],
    )
    total = TOKENIZER.count(prompt) + TOKENIZER.count(lf_template.render_response(reply))
    assert total <= CUTOFF_LEN
    assert result.stats["trimmed_samples"] >= 1
    # without the trimming the whole conversation would not have fitted
    full = "".join(f"第{n}句" + "长话" * 210 for n in range(0, 9, 2))
    assert TOKENIZER.count(full) > CUTOFF_LEN


def test_a_sample_that_cannot_fit_even_alone_is_dropped_and_counted(
    world: World, tmp_path: Path
) -> None:
    long_segment(
        world,
        datetime(2026, 7, 20, 15, 0, tzinfo=UTC),
        turn_chars=4000,
        turns=1,
        reply="短答复标记",
    )
    result = run_export(world.services, tmp_path / "ds")
    assert result.dataset is not None
    assert sample_with(result.dataset, "短答复标记") is None
    assert result.stats["dropped"]["over_budget"] == 1


def test_a_target_that_leaves_no_room_for_a_prompt_is_dropped_and_counted(
    world: World, tmp_path: Path
) -> None:
    long_segment(
        world, datetime(2026, 7, 20, 15, 0, tzinfo=UTC), turn_chars=20, turns=1, reply="嗯" * 3000
    )
    result = run_export(world.services, tmp_path / "ds")
    assert result.dataset is not None
    assert result.stats["dropped"]["target_too_long"] == 1


# -------------------------------------------------------------------- stats and files


def test_the_report_has_the_numbers_of_the_spec(world: World, tmp_path: Path) -> None:
    result = run_export(world.services, tmp_path / "ds")
    stats = result.stats
    assert stats["samples"]["total"] == sum(stats["samples"][k] for k in ("train", "val", "test"))
    assert stats["turns_per_sample"]["mean"] >= 1
    assert stats["target_chars"]["p90"] >= stats["target_chars"]["p50"] > 0
    sticker = stats["sticker_share"]
    assert abs(sticker["export"] - sticker["profile"]) < 0.06
    assert stats["emoji_code_rate"]["profile"] is not None
    # ratio 0: the only samples with a plan are the proactive ones before the cut-off (D-643)
    rows_before = [
        r for name in ("train", "val") for r in rows(result.dataset, f"sft_{name}.jsonl")
    ]
    with_plan = [r for r in rows_before if "【规划】" in r["system"]]
    assert with_plan and all(
        r["conversations"][-2]["value"] == PROACTIVE_CUE_TEXT for r in with_plan
    )
    assert stats["plans"] == {
        "selected": len(with_plan),
        "planned": len(with_plan),
        "share_of_train_and_val": round(len(with_plan) / len(rows_before), 4),
        "share_of_all": round(len(with_plan) / stats["samples"]["total"], 4),
    }
    assert stats["tokens"]["total"] > stats["tokens"]["train"] > 0
    assert set(stats["training_hours"]) == {"5090-8b", "5090-14b", "pro6000-14b", "pro6000-32b"}
    assert stats["epochs"] == 3
    assert stats["her_messages_covered"] > 0 and stats["cutoff_len"] == CUTOFF_LEN
    assert result.dataset is not None and result.dataset.meta.stats == stats


def test_the_directory_is_a_valid_dataset_with_template_check_cases(
    world: World, tmp_path: Path
) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    again = load_dataset_dir(dataset.path)
    assert again.meta.redacted and again.has_parity and again.meta.counts.parity <= 60
    cases = rows(dataset, "parity_cases.jsonl")
    assert cases and all(case["prompt"].endswith("<|im_start|>assistant\n") for case in cases)
    tags = {tag for case in cases for tag in case["tags"]}
    assert {"prelude", "sticker", "multiline", "longest", "proactive"} <= tags


def test_an_export_into_a_directory_that_is_in_use_is_refused(world: World, tmp_path: Path) -> None:
    target = tmp_path / "ds"
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ExportError, match="not empty"):
        run_export(world.services, target)
    assert (target / "keep.txt").exists()


def test_too_little_data_is_reported_in_counts(world: World, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="too few samples"):
        run_export(world.services, tmp_path / "ds", since=world.cutoff)  # nothing to train on
    assert not (tmp_path / "ds").exists()


def test_no_target_holds_event_text(world: World, tmp_path: Path) -> None:
    dataset = exported(world.services, tmp_path / "ds")
    for row in all_rows(dataset):
        assert not [
            line
            for line in row["conversations"][-1]["value"].split("\n")
            if default_detector.is_event_text(line)
        ]
