"""The probe's plan format, its pictures and its verdict (R-CH-009, R-CH-010)."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import pytest
from PIL import Image

from tests.support.clock import ManualClock
from twin.channel.base import MediaNotAllowed
from twin.channel.policy import ProbeImageManifest, ensure_media_allowed, sha256_hex
from twin.channel.probe.images import (
    GIF_FRAMES,
    NAMES,
    make_gif,
    make_jpeg,
    make_png,
    make_probe_images,
)
from twin.channel.probe.model import (
    Action,
    ActionKind,
    Attempt,
    AttemptPhase,
    ProbeOptions,
    StepId,
    StepStatus,
    new_plan,
    plan_from_json,
    plan_to_json,
)
from twin.channel.probe.summary import (
    MARGIN,
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_UNDETERMINED,
    ChannelProbeSummary,
    judge,
    load_channel_probe_summary,
    measured_capabilities,
    save_summary,
    suggest_quota,
    suggest_window_h,
    summarize,
    window_verdict,
)
from twin.channel.state import ChannelStateStore
from twin.storage.db import Database

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------- the plan


def test_a_plan_round_trips_through_json_with_enums_and_times() -> None:
    plan = new_plan("run-1", NOW, ProbeOptions(empty_token_experiment=True))
    step = plan.step(StepId.COUNT)
    step.status = StepStatus.ACTIVE
    step.data = {"n": 8, "capped": False}
    step.attempts.append(
        Attempt(
            n=1,
            phase=AttemptPhase.RUNNING,
            armed_at=NOW,
            baseline_inbound_at=None,
            inbound_at=NOW,
            actions=[Action(id="send:1", kind=ActionKind.SEND_TEXT, label="x", due_at=NOW)],
        )
    )
    plan.add_event(NOW, "hello")
    encoded = plan_to_json(plan)
    assert encoded["status"] == "running" and encoded["created_at"].startswith("2026-10-09T12:00")
    restored = plan_from_json(encoded)
    assert restored == plan
    assert [s.id for s in restored.steps] == [
        StepId.COUNT,
        StepId.MEDIA,
        StepId.WINDOW,
        StepId.EMPTY_TOKEN,
    ]


def test_a_stored_plan_without_newer_fields_still_loads_and_unknown_fields_are_ignored() -> None:
    encoded: dict[str, Any] = plan_to_json(new_plan("run-2", NOW, ProbeOptions()))
    del encoded["options"]["watch_s"]
    encoded["something_from_the_future"] = 1
    restored = plan_from_json(encoded)
    assert restored.options.watch_s == ProbeOptions().watch_s


def test_the_default_options_are_the_ones_the_spec_names() -> None:
    options = ProbeOptions()
    assert options.interval_s == 120.0 and options.max_messages == 15
    assert options.window_hours == [1.0, 6.0, 12.0, 20.0, 23.0, 25.0]
    assert options.empty_token_experiment is False


def test_the_message_budget_is_n_minus_one_once_step_one_is_measured() -> None:
    plan = new_plan("run-3", NOW, ProbeOptions())
    assert plan.message_budget() is None
    plan.step(StepId.COUNT).data = {"n": 8}
    assert plan.message_budget() == 7
    plan.step(StepId.COUNT).data = {"n": 0}
    assert plan.message_budget() == 0


def test_a_step_that_is_not_in_the_plan_is_an_error() -> None:
    plan = new_plan("run-7", NOW, ProbeOptions())
    with pytest.raises(KeyError):
        plan.step(StepId.EMPTY_TOKEN)


def test_the_events_are_capped() -> None:
    plan = new_plan("run-4", NOW, ProbeOptions())
    for index in range(500):
        plan.add_event(NOW, f"event {index}")
    assert len(plan.events) == 400 and plan.events[-1].text == "event 499"


# --------------------------------------------------------------------- pictures


def test_the_probe_pictures_are_real_images_of_the_three_kinds() -> None:
    images = make_probe_images()
    assert set(images) == set(NAMES)
    assert images["jpg"].data.startswith(b"\xff\xd8\xff") and images["jpg"].mime == "image/jpeg"
    assert images["png"].data.startswith(b"\x89PNG") and images["png"].mime == "image/png"
    assert images["gif"].data.startswith(b"GIF89a") and images["gif"].mime == "image/gif"
    for image in images.values():
        with Image.open(io.BytesIO(image.data)) as opened:
            assert opened.size == (320, 240)
        assert len(image.data) < 120_000


def test_the_gif_has_several_different_frames_so_that_motion_can_be_seen() -> None:
    gif = make_gif()
    assert gif.frames == GIF_FRAMES > 1
    with Image.open(io.BytesIO(gif.data)) as opened:
        assert getattr(opened, "n_frames", 1) == GIF_FRAMES
        assert opened.info.get("loop") == 0  # it repeats
        frames = []
        for index in range(opened.n_frames):
            opened.seek(index)
            frames.append(opened.convert("RGB").tobytes())
    assert len(set(frames)) == GIF_FRAMES  # every frame differs from the others


def test_the_pictures_are_distinct_and_carry_a_visible_test_label() -> None:
    hashes = {sha256_hex(image.data) for image in make_probe_images().values()}
    assert len(hashes) == 3
    plain = Image.new("RGB", (320, 240), (30, 90, 200))
    with Image.open(io.BytesIO(make_jpeg().data)) as labelled:
        assert labelled.convert("RGB").tobytes() != plain.tobytes()
    with Image.open(io.BytesIO(make_png().data)) as labelled_png:
        top_left = labelled_png.convert("RGB").crop((16, 16, 160, 56))
        colours = top_left.getcolors(maxcolors=100_000) or []
        assert len(colours) > 20  # lettering with an outline, not a flat area


def test_only_registered_probe_pictures_pass_the_outbound_allow_list(
    db: Database, clock: ManualClock
) -> None:
    manifest = ProbeImageManifest(ChannelStateStore(db))
    images = make_probe_images()
    for image in images.values():
        with pytest.raises(MediaNotAllowed):
            ensure_media_allowed(manifest, image.data)  # not registered yet
    for image in images.values():
        manifest.register_bytes(image.data)
        ensure_media_allowed(manifest, image.data)
    other = Image.new("RGB", (4, 4), (1, 2, 3))
    buffer = io.BytesIO()
    other.save(buffer, "PNG")
    with pytest.raises(MediaNotAllowed):
        ensure_media_allowed(manifest, buffer.getvalue())  # a different picture stays refused


# ---------------------------------------------------------------------- verdict


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (12.0, None, True),
        (25.0, None, True),
        (12.0, 20.0, True),
        (11.99, 12.0, False),  # failed at 12 h: shorter than 12 h
        (6.0, 12.0, False),
        (None, 1.0, False),
        (None, 12.0, False),
        (6.0, 20.0, None),  # 12 h was not tried: cannot tell
        (None, 20.0, None),
        (None, None, None),
        (6.0, None, None),  # delivered at 6 h, nothing more was tried
    ],
)
def test_the_window_verdict_needs_a_point_at_or_beyond_twelve_hours(
    lower: float | None, upper: float | None, expected: bool | None
) -> None:
    assert window_verdict(lower, upper) is expected


@pytest.mark.parametrize(
    ("n", "lower", "upper", "verdict"),
    [
        (3, 12.0, None, VERDICT_MET),  # exactly the minimum on both counts
        (15, 25.0, None, VERDICT_MET),
        (2, 25.0, None, VERDICT_NOT_MET),  # one message short
        (0, None, None, VERDICT_NOT_MET),
        (8, 6.0, 12.0, VERDICT_NOT_MET),
        (2, 6.0, 12.0, VERDICT_NOT_MET),
        (None, 25.0, None, VERDICT_UNDETERMINED),  # step 1 missing
        (8, None, None, VERDICT_UNDETERMINED),  # step 3 missing
        (None, None, None, VERDICT_UNDETERMINED),
        (2, None, None, VERDICT_NOT_MET),  # one definite failure is enough
    ],
)
def test_the_verdict_is_not_met_on_any_proven_shortfall(
    n: int | None, lower: float | None, upper: float | None, verdict: str
) -> None:
    result, reasons = judge(n, lower, upper)
    assert result == verdict
    assert bool(reasons) == (verdict != VERDICT_MET)


def test_the_reasons_name_the_shortfall_in_numbers() -> None:
    _verdict, reasons = judge(2, 6.0, 12.0)
    assert any("only 2 message" in reason and "3" in reason for reason in reasons)
    assert any("12" in reason and "window" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("lower", "expected"),
    [(25.0, 22.5), (12.0, 10.8), (23.0, 20.7), (6.0, 5.4), (1.0, 0.9), (0.0, None), (None, None)],
)
def test_the_suggested_window_keeps_ten_percent_in_reserve(
    lower: float | None, expected: float | None
) -> None:
    assert MARGIN == 0.9
    assert suggest_window_h(lower) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [(15, 13), (8, 7), (10, 9), (3, 2), (2, 1), (1, 1), (0, None), (None, None)],
)
def test_the_suggested_count_keeps_ten_percent_in_reserve(
    n: int | None, expected: int | None
) -> None:
    assert suggest_quota(n) == expected


# --------------------------------------------------------------------- summary


def measured_plan() -> Any:
    plan = new_plan("run-5", NOW, ProbeOptions())
    count = plan.step(StepId.COUNT)
    count.status = StepStatus.DONE
    count.data = {"n": 8, "capped": False}
    media = plan.step(StepId.MEDIA)
    media.status = StepStatus.DONE
    media.data = {"gif_animated": True, "typing": {"visible": "yes"}}
    window = plan.step(StepId.WINDOW)
    window.status = StepStatus.DONE
    window.data = {"lower_bound_h": 23.0, "upper_bound_h": 25.0, "dropped_hours": []}
    return plan


def test_the_summary_collects_the_numbers_the_milestone_check_needs() -> None:
    summary = summarize(measured_plan())
    assert summary.verdict == VERDICT_MET and summary.meets_requirement is True
    assert (summary.n_messages, summary.window_lower_bound_h) == (8, 23.0)
    assert summary.suggestions == {
        "channel.proactive_window_safe_h": 20.7,
        "channel.outbound_quota_safe": 7,
    }
    assert summary.gif_animated is True and summary.typing_visible is True
    assert summary.quote_supported is False and summary.quota_shared is True
    assert not summary.complete  # the plan itself has not been completed


def test_the_requirement_is_true_false_or_unknown_by_the_verdict() -> None:
    assert summarize(measured_plan()).meets_requirement is True
    assert summarize(new_plan("r", NOW, ProbeOptions())).meets_requirement is None
    plan = measured_plan()
    plan.step(StepId.COUNT).data = {"n": 2}
    assert summarize(plan).meets_requirement is False


def test_unfinished_steps_stay_unknown_in_the_summary() -> None:
    plan = new_plan("run-6", NOW, ProbeOptions())
    summary = summarize(plan)
    assert summary.n_messages is None and summary.window_lower_bound_h is None
    assert summary.gif_animated is None and summary.typing_visible is None
    assert summary.quota_shared is None and summary.verdict == VERDICT_UNDETERMINED
    assert summary.suggestions == {
        "channel.proactive_window_safe_h": None,
        "channel.outbound_quota_safe": None,
    }


def test_the_summary_survives_storage_and_is_read_back_for_the_milestone_check(
    db: Database, clock: ManualClock
) -> None:
    summary = summarize(measured_plan())
    with db.session() as session:
        assert load_channel_probe_summary(session) is None
    with db.transaction() as session:
        save_summary(session, summary, clock)
    with db.session() as session:
        stored = load_channel_probe_summary(session)
    assert stored == summary
    assert isinstance(stored, ChannelProbeSummary)


def test_a_summary_of_another_schema_version_is_not_trusted(
    db: Database, clock: ManualClock
) -> None:
    from twin.storage.settings_store import put_setting

    payload = summarize(measured_plan()).to_json()
    payload["schema_version"] = 99
    with db.transaction() as session:
        put_setting(session, "m0.channel_probe", payload, clock=clock)
    with db.session() as session:
        assert load_channel_probe_summary(session) is None


def test_the_history_keeps_the_last_ten_runs(db: Database, clock: ManualClock) -> None:
    from twin.channel.probe.summary import HISTORY_KEY
    from twin.storage.settings_store import get_setting

    for index in range(12):
        plan = measured_plan()
        plan.run_id = f"run-{index}"
        with db.transaction() as session:
            save_summary(session, summarize(plan), clock)
    with db.session() as session:
        history = get_setting(session, HISTORY_KEY)
    assert [entry["run_id"] for entry in history] == [f"run-{i}" for i in range(2, 12)]


def test_measured_capabilities_feed_the_channel_without_inventing_values() -> None:
    assert measured_capabilities(None) == (None, None, None)
    assert measured_capabilities(summarize(measured_plan())) == (23.0, 8, True)
    assert measured_capabilities(summarize(new_plan("r", NOW, ProbeOptions()))) == (
        None,
        None,
        None,
    )
