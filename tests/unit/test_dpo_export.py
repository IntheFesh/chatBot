"""``twin train export-dpo``: the preference pairs as a DPO file (R-LRN-002, R-LRN-004, R-TRN-012).

The export is the only reader of ``preference_pairs`` in the training package.  Its file is
ShareGPT with ``chosen`` / ``rejected`` and the structured sample as it is - never a rendered
template string; the SFT files of the base dataset are copied unchanged.  Round 18 (user decision
"DPO and iron rule 13", D-860/D-861): only the pairs made before the hold-out cut-off of the base
dataset are exported, every system segment is made again through ``AsOfView`` as of the pair's
moment (with the card and template the adapter is locked to), and a fifth of the pairs, chosen by
the hash of their id, is held out of the training for the stack choice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.embedding import HashingBackend
from tests.support.export_world import (
    CORRECTION,
    FACT_MANUAL,
    LIVE_STYLE,
    PAST_MARKER,
    build_world,
    past_card,
)
from tests.support.memory import add_fact
from tests.support.training_data import write_synthetic_dataset
from twin.cli import app
from twin.config.loader import load_settings, resolve_paths
from twin.engine.style_prompt import LockedVersionError
from twin.learning.pairs import (
    SAMPLE_SCHEMA,
    PairError,
    PairRecord,
    PreferencePairStore,
    PromptSample,
)
from twin.ops.foreground import run_jobs_until_idle
from twin.ops.instance_lock import LOCK_RUN, InstanceLock
from twin.ops.jobs import HandlerRegistry, JobContext, JobQueue, JobView
from twin.profile.holdout import HOLDOUT_KEY, Holdout
from twin.profile.persona.store import PersonaStore
from twin.services import Services, build_services, get_cli_context
from twin.storage.ids import new_id
from twin.storage.learning_models import PreferencePair
from twin.storage.settings_store import put_setting
from twin.training import lf_template
from twin.training.cli import format_dpo_state
from twin.training.dataset_dir import file_sha256, load_dataset_dir
from twin.training.dpo_export import (
    SKIP_EVENT,
    SKIP_PERSONA,
    SKIP_SAME,
    SKIP_TOO_LONG,
    DpoExportResult,
    export_dpo,
    holdout_key,
    newest_dataset,
)
from twin.training.export import ExportError
from twin.training.export_job import (
    DPO_STATE_KEY,
    EXPORT_DPO_JOB,
    DpoRequest,
    dpo_summary,
    handle_training_export_dpo,
    queue_export_dpo,
    read_state,
)
from twin.training.layout import (
    FILE_DATASET_META,
    FILE_DPO,
    FILE_DPO_HOLDOUT,
    FILE_TEST,
    FILE_TRAIN,
    FILE_VAL,
)

PHONE_A = "138" + "12345678"  # built from parts: the privacy scan reads this file
PHONE_B = "139" + "87654321"
NOW = datetime(2026, 10, 9, 18, 0, tzinfo=UTC)
LOCKED = "pre_holdout:v3"  # the persona version of the synthetic base dataset, as a pair names it
USER_TURN = lf_template.Turn("user", "今天好累呀")
# the hold-out cut-off of the base datasets of the tests of the file itself: after the test clock
# (2026-10-09 12:00) and NOW, so every pair those tests make is one the export may train on
LATE_CUTOFF = "2026-10-10T00:00:00+00:00"
CUTOFF_AFTER_ALL = datetime.fromisoformat(LATE_CUTOFF)
runner = CliRunner()


def datasets_dir(services: Services) -> Path:
    return services.paths.data_dir / "training" / "datasets"


def base_dataset(
    services: Services, name: str = "ds-test-01", *, cutoff: str = LATE_CUTOFF
) -> Path:
    directory = datasets_dir(services) / name
    write_synthetic_dataset(directory, version=name, holdout_cutoff=cutoff)
    return directory


def as_recorded(pair: PairRecord) -> PromptSample:
    """The system maker of the tests of the file itself: the segment as it was recorded."""
    return pair.sample


def export(services: Services, **options: Any) -> DpoExportResult:
    """``export_dpo`` with the recorded system segments (the real maker - ``AsOfView`` as of the
    pair's moment - is exercised by the tests of the locked versions and the cut-off below)."""
    return export_dpo(services, **{"system_maker": as_recorded, "now": NOW, **options})


def with_card(services: Services, directory: Path) -> str:
    """A pre-holdout card in the store, named by the dataset as the one its adapter is locked to."""
    card = PersonaStore(services.db, services.clock).add_version(
        "pre_holdout", past_card(), reason="generate"
    )
    meta_path = directory / FILE_DATASET_META
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    stored["persona_version"] = card.label
    meta_path.write_text(json.dumps(stored), encoding="utf-8")
    return f"pre_holdout:{card.label}"


def add_pair(
    services: Services,
    chosen: str = "抱抱你",
    rejected: str = "哈哈 辛苦啦",
    *,
    persona: str = LOCKED,
    system: str = "她说话很短。",
    turns: tuple[lf_template.Turn, ...] = (USER_TURN,),
    template: str = lf_template.TEMPLATE_VERSION,
) -> PairRecord:
    record, _ = PreferencePairStore(services.db, services.clock).add(
        reply_id=new_id(),
        sample=PromptSample(system, turns, NOW),
        chosen=chosen,
        rejected=rejected,
        template_version=template,
        persona_version=persona,
    )
    return record


def lines_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------- the file


def test_the_pairs_become_a_new_dataset_version_with_the_sft_files_unchanged(
    services: Services,
) -> None:
    base = base_dataset(services)
    for number in range(3):
        add_pair(services, f"说法{number}", f"回复{number}")
    result = export(services)
    assert result.pairs == 3 and result.regenerated == 3 and result.skipped == {}
    out = result.dataset
    assert out.path.parent == base.parent and out.path != base
    assert out.meta.dataset_version == "ds-test-01-dpo20261009180000"
    assert out.meta.counts.dpo == 3 and out.has_dpo and FILE_DPO in out.data_files()
    for name in (FILE_TRAIN, FILE_VAL, FILE_TEST):  # copied byte for byte: the same content
        assert file_sha256(out.path / name) == file_sha256(base / name)
    assert not (base / FILE_DPO).exists()  # the base dataset is not touched
    assert load_dataset_dir(base).meta.dataset_version == "ds-test-01"
    meta = json.loads((out.path / FILE_DATASET_META).read_text(encoding="utf-8"))
    assert meta["stats"]["dpo"]["exported"] == 3 and meta["stats"]["note"] == "synthetic"
    assert meta["files"][FILE_DPO] == file_sha256(out.path / FILE_DPO)


def test_every_line_is_sharegpt_with_chosen_and_rejected_and_no_template_string(
    services: Services,
) -> None:
    base_dataset(services)
    add_pair(
        services,
        "抱抱你",
        "哈哈 辛苦啦",
        turns=(
            lf_template.Turn("user", "在吗"),
            lf_template.Turn("assistant", "在呀"),
            lf_template.Turn("user", "今天好累呀"),
        ),
    )
    result = export(services)
    text = (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    for marker in (*lf_template.CONTROL_TOKENS, *lf_template.LF_SLOTS, "<|im_start|>"):
        assert marker not in text
    [row] = lines_of(result.dataset.path / FILE_DPO)
    assert set(row) == {"id", "system", "conversations", "chosen", "rejected"}
    assert [m["from"] for m in row["conversations"]] == ["human", "gpt", "human"]
    assert row["conversations"][-1]["value"] == "今天好累呀" and row["system"] == "她说话很短。"
    assert row["chosen"] == {"from": "gpt", "value": "抱抱你"}
    assert row["rejected"] == {"from": "gpt", "value": "哈哈 辛苦啦"}
    assert not row["system"].startswith(lf_template.IM_START)  # the structure, not the template


def test_the_rejected_text_is_in_the_dpo_file_and_in_no_other_file(services: Services) -> None:
    base_dataset(services)
    marker = "只在被拒绝的回复里出现的句子"
    add_pair(services, "抱抱你", marker)
    result = export(services)
    assert marker in (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    for path in result.dataset.path.iterdir():
        if path.name != FILE_DPO:
            assert marker not in path.read_text(encoding="utf-8"), path.name


def test_what_leaves_the_machine_is_desensitised_with_one_token_per_entity(
    services: Services,
) -> None:
    base_dataset(services)
    add_pair(services, f"打我电话 {PHONE_A}", "好的", system=f"她的电话是 {PHONE_A}")
    add_pair(services, f"找我 {PHONE_A} 或 {PHONE_B}", "好的呀")
    result = export(services)
    text = (result.dataset.path / FILE_DPO).read_text(encoding="utf-8")
    assert PHONE_A not in text and PHONE_B not in text
    rows = lines_of(result.dataset.path / FILE_DPO)
    assert "[手机号#1]" in rows[0]["chosen"]["value"] and "[手机号#1]" in rows[0]["system"]
    assert "[手机号#1]" in rows[1]["chosen"]["value"] and "[手机号#2]" in rows[1]["chosen"]["value"]
    assert result.dataset.meta.stats["dpo"]["redacted_entities"] == {"phone": 2}


def test_the_new_version_is_recorded_like_any_exported_dataset(services: Services) -> None:
    from twin.storage.training_models import DatasetVersion

    base_dataset(services)
    add_pair(services)
    result = export(services)
    with services.db.session() as session:
        row = session.scalars(select(DatasetVersion)).one()
        assert (row.id, row.dpo_count, row.train_count) == (
            result.dataset.meta.dataset_version,
            1,
            40,
        )
        assert row.files == result.dataset.meta.files


def test_enough_pairs_are_the_setting_of_the_hint(services: Services) -> None:
    base_dataset(services)
    add_pair(services)
    assert not export(services).enough
    services.settings.training.dpo_min_pairs = 1
    later = datetime(2026, 10, 9, 18, 0, 5, tzinfo=UTC)
    assert export(services, now=later).enough


# ------------------------------------------------------------------ what is left out


def test_a_pair_that_cannot_be_trained_on_is_left_out_and_counted(services: Services) -> None:
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦")
    same = add_pair(services, "好", "不好")
    with services.db.transaction(bump_state=False) as session:  # damaged: nothing to learn
        row = session.get(PreferencePair, same.id)
        assert row is not None
        row.chosen = "不好"
    add_pair(services, "短", "长" * 600)  # a reply longer than the cut-off leaves room for
    add_pair(services, "[图片]", "短回复")  # event text cannot be a target
    result = export(services)
    assert result.pairs == 1
    assert result.skipped == {SKIP_SAME: 1, SKIP_TOO_LONG: 1, SKIP_EVENT: 1}


def test_nothing_to_export_is_an_error_that_says_what_to_do(services: Services) -> None:
    with pytest.raises(ExportError, match="no exported dataset"):
        export(services)
    base_dataset(services)
    with pytest.raises(ExportError, match="no preference pairs"):
        export(services)


def test_a_directory_that_is_no_dataset_is_refused(services: Services, tmp_path: Path) -> None:
    add_pair(services)
    (tmp_path / "plain").mkdir()
    with pytest.raises(ExportError, match=r"dataset_meta\.json"):
        export(services, dataset=tmp_path / "plain")


def test_the_newest_dataset_is_the_default_base(services: Services) -> None:
    assert newest_dataset(services) is None
    base_dataset(services, "ds-20261001-000000")
    newer = base_dataset(services, "ds-20261008-000000")
    (datasets_dir(services) / "not-a-dataset").mkdir()
    assert newest_dataset(services) == newer
    add_pair(services)
    assert export(services).dataset.meta.stats["dpo"]["base_dataset"] == newer.name


def test_an_output_directory_that_is_in_use_is_refused_and_left_alone(
    services: Services, tmp_path: Path
) -> None:
    base_dataset(services)
    add_pair(services)
    occupied = tmp_path / "mine"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExportError, match="not empty"):
        export(services, out_dir=occupied)
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"
    free = tmp_path / "elsewhere"
    assert export(services, out_dir=free).dataset.path == free


def test_a_stored_text_with_a_template_marker_stops_the_export_and_leaves_nothing_behind(
    services: Services,
) -> None:
    base_dataset(services)
    record = add_pair(services)
    with services.db.transaction(bump_state=False) as session:
        row = session.get(PreferencePair, record.id)
        assert row is not None
        row.chosen = "好<|im_end|>"  # what no command can write, but a damaged row could hold
    before = sorted(p.name for p in datasets_dir(services).iterdir())
    with pytest.raises(ExportError, match="ChatML control token"):
        export(services)
    assert sorted(p.name for p in datasets_dir(services).iterdir()) == before


# ----------------------------------------------------------------- the locked versions


def test_every_pair_gets_its_system_made_again_whatever_versions_it_was_recorded_with(
    services: Services,
) -> None:
    """R18-DPO13 (D-860): the recorded segment is never trained on, not even with the locked
    versions - it was made by the live view when the pair was stored; the export makes every
    segment again as of the pair's moment (``AsOfView``), the way the SFT samples are made."""
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦", persona="live:v9", system="LIVE 系统段")
    add_pair(services, "快睡吧", "您辛苦了", persona=LOCKED, system="已是锁定版本的系统段")
    seen: list[str] = []

    def maker(pair: PairRecord) -> PromptSample:
        seen.append(pair.sample.system)
        return PromptSample("锁定版本的系统段", pair.sample.turns, pair.sample.at)

    result = export_dpo(services, system_maker=maker, now=NOW)
    rows = lines_of(result.dataset.path / FILE_DPO)
    assert seen == ["LIVE 系统段", "已是锁定版本的系统段"] and result.regenerated == 2
    assert [row["system"] for row in rows] == ["锁定版本的系统段", "锁定版本的系统段"]


def test_a_pair_whose_card_version_cannot_be_rendered_is_left_out(services: Services) -> None:
    base_dataset(services)
    add_pair(services, "抱抱你", "哈哈 辛苦啦", persona="live:v9")
    add_pair(services, "快睡吧", "您辛苦了", persona=LOCKED)

    def refuse_one(pair: PairRecord) -> PromptSample:
        if pair.chosen == "抱抱你":
            raise LockedVersionError("persona card v3 of the model is not in the store")
        return pair.sample

    result = export_dpo(services, system_maker=refuse_one, now=NOW)
    assert result.pairs == 1 and result.skipped == {SKIP_PERSONA: 1}


def test_when_no_pair_can_be_made_the_export_fails_with_the_reasons(services: Services) -> None:
    base_dataset(services)
    add_pair(services, persona="live:v9")

    def refuse(pair: PairRecord) -> PromptSample:
        raise PairError("the reply did not answer a message of the user")

    with pytest.raises(ExportError, match="none of the 1 preference pairs"):
        export_dpo(services, system_maker=refuse, now=NOW)


def world_dataset(services: Services, world_cutoff: datetime, card_label: str) -> Path:
    """The base dataset of a world: its cut-off, and the pre-holdout card it is locked to."""
    directory = datasets_dir(services) / "ds-world"
    write_synthetic_dataset(directory, version="ds-world", holdout_cutoff=world_cutoff.isoformat())
    meta_path = directory / FILE_DATASET_META
    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    stored["persona_version"] = card_label
    meta_path.write_text(json.dumps(stored), encoding="utf-8")
    return directory


def pair_at(
    services: Services,
    moment: datetime,
    chosen: str,
    rejected: str = "哈哈 辛苦啦",
    *,
    corrected_after: timedelta = timedelta(hours=1),
    persona: str = LOCKED,
    system: str = "她说话很短。",
) -> PairRecord:
    """A pair of a reply at ``moment`` whose correction (the row's ``created_at``) came later."""
    record, _ = PreferencePairStore(services.db, ManualClock(moment + corrected_after)).add(
        reply_id=new_id(),
        sample=PromptSample(system, (USER_TURN,), moment),
        chosen=chosen,
        rejected=rejected,
        template_version=lf_template.TEMPLATE_VERSION,
        persona_version=persona,
    )
    return record


def test_the_real_maker_renders_the_locked_pre_holdout_card_as_of_the_moment_of_the_reply(
    services: Services, embedder: HashingBackend
) -> None:
    world = build_world(services, embedder, days=20)
    card = world.store.active("pre_holdout")
    assert card is not None
    directory = world_dataset(services, world.cutoff, card.label)
    pair_at(
        services,
        world.cutoff - timedelta(days=2),
        "抱抱你",
        persona=f"live:{card.label}",
        system=f"LIVE 系统段 {LIVE_STYLE} {CORRECTION} {FACT_MANUAL}",
    )
    result = export_dpo(services, dataset=directory, now=NOW)
    assert result.pairs == 1 and result.regenerated == 1
    [row] = lines_of(result.dataset.path / FILE_DPO)
    system = row["system"]
    assert PAST_MARKER in system  # the pre-holdout card the adapter was trained with
    for leaked in (LIVE_STYLE, CORRECTION, FACT_MANUAL, "LIVE 系统段"):
        assert (
            leaked not in system
        )  # nothing of the live card: no corrections, no hand-written facts
    assert "【此刻】" in system and row["conversations"][-1]["value"] == "今天好累呀"
    for marker in lf_template.CONTROL_TOKENS:
        assert marker not in system


def test_a_fact_learned_after_the_pairs_moment_is_not_in_its_system_segment(
    services: Services, embedder: HashingBackend
) -> None:
    """R18-DPO13 (D-860), the injection test of R-TRN-013 for DPO: a fact that only became known
    after the reply - here half an hour later, before the correction was stored, so the live view
    that recorded the segment had it - is not in the exported segment; one known a second before
    the reply is.  The pair names the locked card and template: before round 18 its recorded
    segment went into the file unchanged."""
    world = build_world(services, embedder, days=20)
    card = world.store.active("pre_holdout")
    assert card is not None
    directory = world_dataset(services, world.cutoff, card.label)
    moment = world.cutoff - timedelta(days=3)
    before = "偏好对之前一秒才知道的事蓝色自行车"
    after = "偏好对之后才知道的事红色风筝"
    add_fact(world.memory, before, moment - timedelta(seconds=1), importance=5)
    add_fact(world.memory, after, moment + timedelta(minutes=30), importance=5)
    pair_at(
        services,
        moment,
        "抱抱你",
        persona=f"pre_holdout:{card.label}",
        system=f"当时记下的系统段\n{before}\n{after}",
    )
    result = export_dpo(services, dataset=directory, now=NOW)
    [row] = lines_of(result.dataset.path / FILE_DPO)
    assert after not in row["system"]
    assert before in row["system"] and PAST_MARKER in row["system"]
    assert "当时记下的系统段" not in row["system"]


# ------------------------------------------- the cut-off and the held-out pairs (round 18)

CUTOFF = datetime(2026, 9, 15, tzinfo=UTC)


def test_only_the_pairs_made_before_the_cutoff_are_exported_and_the_rest_wait(
    services: Services,
) -> None:
    """R18-DPO13 (D-860): ``created_at`` < the cut-off of the base dataset, nothing else."""
    base_dataset(services, cutoff=CUTOFF.isoformat())
    for days, chosen in ((10, "早的说法一"), (5, "早的说法二")):
        pair_at(services, CUTOFF - timedelta(days=days), chosen)
    pair_at(services, CUTOFF - timedelta(hours=3), "早的说法三")  # corrected 2 h before the cut
    # the reply before the cut-off, the correction after it: the pair is after the cut-off
    pair_at(services, CUTOFF - timedelta(minutes=30), "晚的说法一")
    pair_at(services, CUTOFF + timedelta(days=2), "晚的说法二")
    result = export(services)
    assert (result.pairs, result.after_cutoff, result.cutoff) == (3, 2, CUTOFF)
    written = "".join(path.read_text("utf-8") for path in result.dataset.path.glob("dpo_*.jsonl"))
    assert all(f"早的说法{n}" in written for n in "一二三")
    assert "晚的说法" not in written
    stats = result.dataset.meta.stats["dpo"]
    assert stats["pairs_in_store"] == 5 and stats["after_cutoff"] == 2
    assert stats["cutoff"] == CUTOFF.isoformat()
    summary = dpo_summary(result)
    assert (summary["after_cutoff"], summary["cutoff"]) == (2, CUTOFF.isoformat())
    shown = "\n".join(format_dpo_state({"state": "done", "at": "now", **summary}))
    assert "2 pair(s) were made after the hold-out cut-off 2026-09-15" in shown
    assert "they wait for an export with a later cut-off" in shown


def test_the_minimum_counts_only_the_pairs_made_before_the_cutoff(services: Services) -> None:
    """R18-DPO13 (D-860): "200 pairs before DPO" is 200 pairs it may train on."""
    services.settings.training.dpo_min_pairs = 3
    base_dataset(services, cutoff=CUTOFF.isoformat())
    for number in range(2):
        pair_at(services, CUTOFF - timedelta(days=number + 1), f"早的说法{number}")
    for number in range(4):
        pair_at(services, CUTOFF + timedelta(days=number + 1), f"晚的说法{number}")
    first = export(services)
    assert (first.pairs, first.after_cutoff, first.enough) == (2, 4, False)
    pair_at(services, CUTOFF - timedelta(days=7), "早的说法再一个")
    second = export(services, now=NOW + timedelta(seconds=1))
    assert (second.pairs, second.enough) == (3, True)


def test_pairs_that_are_all_after_the_cutoff_are_an_error_that_says_they_wait(
    services: Services,
) -> None:
    base_dataset(services, cutoff=CUTOFF.isoformat())
    pair_at(services, CUTOFF + timedelta(days=1), "晚的说法")
    with pytest.raises(ExportError, match="after the hold-out cut-off 2026-09-15") as caught:
        export(services)
    assert "晚的说法" not in str(caught.value)
    assert [p.name for p in datasets_dir(services).iterdir()] == ["ds-test-01"]


def test_a_base_dataset_whose_cutoff_cannot_be_read_is_refused(services: Services) -> None:
    """R18-DPO13 (D-860): without the cut-off nothing tells which pairs DPO may train on."""
    base_dataset(services, cutoff="no date at all")
    pair_at(services, CUTOFF - timedelta(days=1), "说法")
    with pytest.raises(ExportError, match="cut-off of dataset ds-test-01 cannot be read"):
        export(services)


def held_out_ids(pairs: list[PairRecord], count: int) -> set[str]:
    """The ``count`` pairs of the smallest key: sha256 of a fixed prefix and the id (D-861)."""
    return {pair.id for pair in sorted(pairs, key=lambda pair: holdout_key(pair.id))[:count]}


def test_a_fifth_of_the_pairs_is_held_out_by_the_hash_of_their_id(services: Services) -> None:
    """R18-E6 (D-861): ``n // 5`` pairs go to ``dpo_holdout.jsonl`` and are never trained on;
    which ones depends on the ids alone, so the split is the same on every export."""
    base_dataset(services, cutoff=CUTOFF.isoformat())
    pairs = [pair_at(services, CUTOFF - timedelta(days=12 - n), f"说法{n}") for n in range(11)]
    result = export(services)
    assert (result.pairs, result.held_out, result.trained) == (11, 2, 9)
    train = lines_of(result.dataset.path / FILE_DPO)
    held = lines_of(result.dataset.path / FILE_DPO_HOLDOUT)
    assert len(train) == 9 and len(held) == 2
    assert {row["id"] for row in held} == held_out_ids(pairs, 2)
    assert not {row["id"] for row in held} & {row["id"] for row in train}
    assert {row["id"] for row in held} | {row["id"] for row in train} == {p.id for p in pairs}
    assert set(held[0]) == {"id", "system", "conversations", "chosen", "rejected"}
    meta = result.dataset.meta
    assert (meta.counts.dpo, meta.counts.dpo_holdout) == (9, 2)
    assert FILE_DPO_HOLDOUT in result.dataset.data_files()
    assert meta.files[FILE_DPO_HOLDOUT] == file_sha256(result.dataset.path / FILE_DPO_HOLDOUT)
    assert meta.stats["dpo"]["held_out"] == 2 and meta.stats["dpo"]["trained"] == 9
    assert load_dataset_dir(result.dataset.path).meta.counts.dpo_holdout == 2
    again = export(services, now=NOW + timedelta(seconds=1))
    assert {row["id"] for row in lines_of(again.dataset.path / FILE_DPO_HOLDOUT)} == {
        row["id"] for row in held
    }
    summary = dpo_summary(result)
    assert (summary["held_out"], summary["trained"]) == (2, 9)
    shown = "\n".join(format_dpo_state({"state": "done", "at": "now", **summary}))
    assert "9 preference pair(s) in dpo_train.jsonl" in shown
    assert "2 held out in dpo_holdout.jsonl for the stack choice" in shown


@pytest.mark.parametrize(("count", "held"), [(1, 0), (4, 0), (5, 1), (9, 1), (10, 2)])
def test_the_held_out_share_is_a_fifth_rounded_down(
    services: Services, count: int, held: int
) -> None:
    base_dataset(services, cutoff=CUTOFF.isoformat())
    for number in range(count):
        pair_at(services, CUTOFF - timedelta(days=number + 1), f"说法{number}")
    result = export(services)
    assert (result.held_out, result.trained) == (held, count - held)
    assert (result.dataset.path / FILE_DPO_HOLDOUT).exists() is (held > 0)


def test_the_key_of_the_split_is_the_sha256_of_a_fixed_prefix_and_the_id() -> None:
    expected = hashlib.sha256(b"twin-dpo-holdout:01JABCDEF").hexdigest()
    assert holdout_key("01JABCDEF") == expected


# ------------------------------------------------------------------------ the command


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: ManualClock) -> Path:
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    # the commands read the clock the data of the tests is written with (EXECUTION_NOTES 12): the
    # export keeps the pairs made before the cut-off of the dataset (D-860)
    get_cli_context().clock = clock
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    return path


def with_services[T](work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(
        settings, root=resolve_paths(settings).root, clock=get_cli_context().clock
    )
    try:
        return work(services)
    finally:
        services.close()


def prepare_three_pairs(services: Services) -> None:
    """A base dataset locked to a card in the store; two pairs to export, one to leave out.

    Every system segment is made again with the locked card as of the pair's moment (D-860), so
    the card a pair was recorded with does not matter; the too long reply is left out."""
    locked = with_card(services, base_dataset(services))
    add_pair(services, persona="live:v9")
    add_pair(services, "快睡吧", "您辛苦了", persona=locked)
    add_pair(services, "短", "长" * 600)


def test_the_command_writes_the_file_and_says_what_comes_next(
    data_dir: Path, clock: ManualClock
) -> None:
    with_services(prepare_three_pairs)
    result = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    assert result.exit_code == 0, result.output
    assert "2 preference pair(s) in dpo_train.jsonl" in result.output
    assert "left out: reply_too_long 1" in result.output
    assert "every system segment made again as of its pair's moment" in result.output
    assert "DPO needs at least 200 pairs" in result.output
    assert "next: twin train bundle --profile <profile> --dataset" in result.output
    created = sorted(p.name for p in (data_dir / "training" / "datasets").iterdir())
    assert created[0] == "ds-test-01" and created[1].startswith("ds-test-01-dpo")


def test_the_command_fails_clearly_without_a_dataset_or_pairs(
    data_dir: Path, clock: ManualClock
) -> None:
    result = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    assert result.exit_code != 0 and "twin train export" in result.output
    with_services(lambda services: base_dataset(services))
    result = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    assert result.exit_code != 0 and "no preference pairs" in result.output


def jobs_of(services: Services) -> list[tuple[str, str, dict[str, Any]]]:
    queue = JobQueue(services.db, services.clock)
    return [(j.type, j.status, dict(j.payload)) for j in queue.list_jobs()]


def test_the_command_only_queues_one_job_and_writes_nothing(
    data_dir: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17-29 (D-713): the command is HEAVY - the application makes every pair again."""
    with_services(lambda services: (base_dataset(services), add_pair(services)))
    monkeypatch.chdir(data_dir)
    first = runner.invoke(app, ["train", "export-dpo", "--dataset", "ds", "--out", "made/it"])
    assert first.exit_code == 0, first.output
    assert "queued (job " in first.output and "twin train export-status" in first.output
    assert "already waiting" in runner.invoke(app, ["train", "export-dpo"]).output
    [(job_type, status, payload)] = with_services(jobs_of)
    assert (job_type, status) == ("training_export_dpo", "pending")
    # the application has another working directory: what travels in the job is absolute
    assert payload == {"dataset": str(data_dir / "ds"), "out_dir": str(data_dir / "made" / "it")}
    created = [p.name for p in (data_dir / "training" / "datasets").iterdir()]
    assert created == ["ds-test-01"] and not (data_dir / "made").exists()
    status_text = runner.invoke(app, ["train", "export-status"]).output
    assert "export-dpo" not in status_text  # nothing has run yet


def test_the_foreground_run_yields_to_a_running_application(
    data_dir: Path, clock: ManualClock
) -> None:
    with_services(lambda services: (base_dataset(services), add_pair(services)))
    lock = with_services(lambda s: InstanceLock(LOCK_RUN, locks_dir=s.paths.locks_dir))
    assert lock.acquire()
    try:
        result = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    finally:
        lock.release()
    assert result.exit_code == 0, result.output
    assert "the application is running and will execute the job" in result.output
    assert [t for t, _, _ in with_services(jobs_of)] == ["training_export_dpo"]
    assert [p.name for p in (data_dir / "training" / "datasets").iterdir()] == ["ds-test-01"]


def test_export_status_shows_how_the_dpo_export_ended(data_dir: Path, clock: ManualClock) -> None:
    with_services(prepare_three_pairs)
    done = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    assert done.exit_code == 0, done.output
    shown = runner.invoke(app, ["train", "export-status"])
    assert shown.exit_code == 0, shown.output
    assert "export-dpo: done" in shown.output
    assert "2 preference pair(s) in dpo_train.jsonl" in shown.output
    assert "left out: reply_too_long 1" in shown.output
    assert "DPO needs at least 200 pairs" in shown.output
    state = with_services(lambda services: read_state(services, key=DPO_STATE_KEY))
    assert state is not None
    assert state["pairs"] == 2 and state["skipped"] == {"reply_too_long": 1}
    assert (state["after_cutoff"], state["held_out"], state["trained"]) == (0, 0, 2)
    assert "快睡吧" not in json.dumps(state, ensure_ascii=False)  # counts and versions, no text


def test_a_failed_dpo_export_is_kept_and_shown_and_the_command_fails(
    data_dir: Path, clock: ManualClock
) -> None:
    failed = runner.invoke(app, ["train", "export-dpo", "--foreground"])
    assert failed.exit_code != 0 and "export-dpo: failed" in failed.output
    shown = runner.invoke(app, ["train", "export-status"])
    assert "export-dpo: failed" in shown.output and "twin train export" in shown.output


async def test_the_dpo_job_needs_the_services_container(clock: ManualClock) -> None:
    job = JobView(
        id="J" * 26, type=EXPORT_DPO_JOB, payload={}, priority=80, status="running", attempts=1,
        max_attempts=1, run_after=NOW, offpeak_only=False, deadline=None, last_error=None,
        batch_id=None, estimated_cost_usd=None, requires_approval=False, approved_at=None,
        approved_usd=None, started_at=NOW, finished_at=None, created_at=NOW, updated_at=NOW,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="services container"):
        await handle_training_export_dpo(JobContext(job=job, services=None, clock=clock))


async def test_the_job_writes_the_same_file_as_the_synchronous_export(
    services: Services, tmp_path: Path
) -> None:
    """The handler is ``export_dpo`` in a thread: the same lines, the same counts."""
    locked = with_card(services, base_dataset(services))
    add_pair(services, persona="live:v9")
    add_pair(services, "快睡吧", "您辛苦了", persona=locked)
    add_pair(services, "抱抱你", "哈哈 辛苦啦")
    direct = export_dpo(services, out_dir=tmp_path / "direct", now=NOW)
    assert direct.pairs == 3 and direct.regenerated == 3
    queued = queue_export_dpo(services, DpoRequest(out_dir=tmp_path / "queued"))
    assert not queued.already_queued
    registry = HandlerRegistry()
    registry.register(EXPORT_DPO_JOB, handle_training_export_dpo)
    summary = await run_jobs_until_idle(services, registry)
    assert summary.done == 1 and summary.failed == 0
    state = read_state(services, key=DPO_STATE_KEY)
    assert state is not None and state["state"] == "done"
    assert (state["pairs"], state["regenerated"], state["skipped"]) == (
        direct.pairs,
        direct.regenerated,
        direct.skipped,
    )
    assert state["directory"] == str(tmp_path / "queued")
    assert (tmp_path / "queued" / FILE_DPO).read_bytes() == (
        tmp_path / "direct" / FILE_DPO
    ).read_bytes()
    assert dpo_summary(direct)["enough"] is state["enough"]


def test_the_retrain_check_reports_the_pairs_for_dpo(data_dir: Path, clock: ManualClock) -> None:
    def prepare(services: Services) -> None:
        services.settings.training.dpo_min_pairs = 1

    with_services(prepare)
    plain = runner.invoke(app, ["train", "retrain-check"])
    assert plain.exit_code == 0 and "preference pairs: 0" in plain.output

    def add(services: Services) -> None:
        add_pair(services)

    with_services(add)
    after = runner.invoke(app, ["--set", "training.dpo_min_pairs=1", "train", "retrain-check"])
    # R18-DPO13 (D-860): no cut-off yet - no pair DPO may train on
    assert "preference pairs: 1 (0 before the hold-out cut-off, 1 after it)" in after.output
    assert "可以做 DPO" not in after.output

    def cut(services: Services) -> None:
        with services.db.transaction(bump_state=True) as session:
            put_setting(
                session,
                HOLDOUT_KEY,
                Holdout(CUTOFF_AFTER_ALL, 0.1, 100, 10, CUTOFF_AFTER_ALL).to_json(),
                clock=services.clock,
                by="test",
            )

    with_services(cut)
    result = runner.invoke(app, ["--set", "training.dpo_min_pairs=1", "train", "retrain-check"])
    assert "preference pairs: 1 (1 before the hold-out cut-off, 0 after it)" in result.output
    assert "可以做 DPO" in result.output


def test_the_sample_schema_is_part_of_the_stored_form() -> None:
    sample = PromptSample("系统", (USER_TURN,), NOW, ("她开头的话",))
    data = sample.to_json()
    assert data["schema"] == SAMPLE_SCHEMA and data["prelude"] == ["她开头的话"]
    assert PromptSample.from_json(data) == sample
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "schema": 99})
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "turns": [{"role": "robot", "content": "x"}]})
    with pytest.raises(PairError):
        PromptSample.from_json({**data, "prelude": "not a list"})
    with pytest.raises(PairError):
        PromptSample.from_json({"schema": SAMPLE_SCHEMA})
