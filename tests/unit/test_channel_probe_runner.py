"""The probe state machine on a simulated platform (R-CH-009, R-CH-010).

A virtual clock lets a 26-hour probe run in a moment; the simulated platform enforces a message
count and a window and records what reached the phone, which is what the simulated user reports
when the probe asks.  Every send goes through the real ``ProbeSendPolicy``.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import pairwise

import pytest

from tests.support.probe import Rig, finished_sends, make_rig, sent_texts
from tests.support.synthetic import mobile
from twin.channel.base import TEST_PREFIX, AuthState, OutboundKind, OutboundResult
from twin.channel.probe.model import (
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    StepId,
    StepStatus,
)
from twin.channel.probe.runner import ProbeRunner
from twin.channel.probe.summary import (
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_UNDETERMINED,
    load_channel_probe_summary,
    summarize,
)
from twin.storage.db import Database


def attempt_of(plan: ProbePlan, step: StepId) -> Attempt | None:
    return plan.step(step).current_attempt()


def waiting(plan: ProbePlan, step: StepId = StepId.COUNT) -> bool:
    attempt = attempt_of(plan, step)
    return attempt is not None and attempt.phase is AttemptPhase.WAITING


def running(plan: ProbePlan, step: StepId) -> bool:
    attempt = attempt_of(plan, step)
    return attempt is not None and attempt.phase is AttemptPhase.RUNNING


def done(step: StepId) -> object:
    return lambda plan: plan.step(step).status is StepStatus.DONE


@pytest.fixture
def rig(db: Database) -> Rig:
    return make_rig(db, quota=8, window_h=24.0)


# ----------------------------------------------------------------- the whole run


async def test_a_full_probe_measures_count_media_and_window(rig: Rig, db: Database) -> None:
    plan = await rig.run()
    assert plan.status is PlanStatus.COMPLETED
    count = plan.step(StepId.COUNT).data
    assert count["n"] == 8 and count["api_ok"] == 8 and not count["mismatch"]
    assert plan.step(StepId.MEDIA).data["gif_animated"] is True
    window = plan.step(StepId.WINDOW).data
    assert window["lower_bound_h"] is not None and 23 <= window["lower_bound_h"] < 24
    assert window["upper_bound_h"] is not None and window["upper_bound_h"] >= 25
    with db.session() as session:
        stored = load_channel_probe_summary(session)
    assert stored is not None and stored.complete and stored.meets_requirement is True
    assert stored.verdict == VERDICT_MET and stored.n_messages == 8
    assert stored.gif_animated is True and stored.typing_visible is True
    assert stored.quote_supported is False and stored.quota_shared is True


async def test_every_probe_text_starts_with_the_test_prefix(rig: Rig) -> None:
    await rig.run()
    texts = sent_texts(rig.sim)
    assert texts and all(text.startswith(TEST_PREFIX) for text in texts)


async def test_every_send_is_audited_and_none_was_refused(rig: Rig) -> None:
    await rig.run()
    audit = rig.store.audit()
    assert len(audit) == len(rig.sim.log)
    assert {entry["decision"] for entry in audit} == {"allowed"}
    plan = rig.store.load()
    assert plan is not None
    assert {entry["run_id"] for entry in audit} == {plan.run_id}
    assert {entry["step"] for entry in audit} == {"count", "media", "window"}


# --------------------------------------------- a fresh inbound message before each step


async def test_each_step_waits_for_a_fresh_message_announced_in_terminal_and_wechat(
    rig: Rig,
) -> None:
    rig.user.writes_when_asked = False
    plan = await rig.tick_until(waiting)
    for _ in range(5):  # nothing happens without the message
        await rig.runner.tick()
        await rig.clock.sleep(600)
    assert [entry.kind for entry in rig.sim.log] == ["text"]  # only the announcement
    announcement = rig.sim.log[0].text or ""
    assert announcement.startswith(TEST_PREFIX) and "第1步" in announcement
    assert rig.banner.shown and "send the bot a message" in rig.banner.shown[0][0]
    assert any("Step 1/3" in line for _title, lines in rig.banner.shown for line in lines)
    assert "Send the bot ONE message" in (plan.notice or "")
    assert "channel.probe" in rig.alerts.categories()


async def test_a_login_problem_is_shown_while_waiting_and_cleared_when_it_is_fixed(
    rig: Rig,
) -> None:
    rig.user.writes_when_asked = False
    rig.sim.auth = AuthState.NEEDS_RELOGIN
    await rig.tick_until(waiting)
    await rig.runner.tick()
    stored = rig.store.load()
    assert stored is not None and "twin channel login --force" in (stored.notice or "")
    assert "Send the bot ONE message" in (stored.notice or "")  # the original request stays
    rig.sim.auth = AuthState.OK
    await rig.runner.tick()
    fixed = rig.store.load()
    assert fixed is not None and "twin channel login" not in (fixed.notice or "")


async def test_a_message_from_before_the_announcement_does_not_start_a_step(rig: Rig) -> None:
    rig.user.writes_when_asked = False
    await rig.tick_until(waiting)
    for _ in range(3):
        await rig.runner.tick()
    plan = rig.store.load()
    assert plan is not None
    attempt = plan.step(StepId.COUNT).attempts[0]
    assert attempt.baseline_inbound_at == rig.sim.last_inbound  # the message sent at login
    assert attempt.phase is AttemptPhase.WAITING


async def test_the_measurement_begins_only_after_the_message_settled(rig: Rig) -> None:
    plan = await rig.tick_until(lambda p: running(p, StepId.COUNT))
    attempt = plan.step(StepId.COUNT).attempts[0]
    assert attempt.inbound_at == rig.sim.last_inbound
    assert attempt.started_at is not None and attempt.inbound_at is not None
    assert attempt.started_at - attempt.inbound_at >= timedelta(seconds=plan.options.settle_s)
    assert attempt.context_fp is not None


async def test_a_second_message_right_behind_the_first_moves_the_start(rig: Rig) -> None:
    await rig.tick_until(waiting)
    rig.sim.user_writes()
    await rig.runner.tick()  # sees the first message, lets it settle
    first = rig.sim.last_inbound
    await rig.clock.sleep(1)
    rig.sim.user_writes()  # a second message before the settling time is over
    plan = await rig.tick_until(lambda p: running(p, StepId.COUNT))
    assert plan.step(StepId.COUNT).attempts[0].inbound_at == rig.sim.last_inbound
    assert plan.step(StepId.COUNT).attempts[0].inbound_at != first
    assert plan.step(StepId.COUNT).voided_attempts == 0


# ------------------------------------------------------------------ step 1: count


async def test_the_count_stops_at_the_first_failure_and_is_never_retried(rig: Rig) -> None:
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT).data
    assert count["n"] == 8
    failure = count["first_failure"]
    assert failure["outcome"] == "window_rejected" and failure["ret"] == -2
    assert failure["code"] == -2 and failure["errmsg"] == "prepare failed"
    assert count["first_problem_index"] == 9
    results = [e.result for e in rig.sim.log if e.text and "条数测试" in e.text]
    assert results == ["ok"] * 8 + ["platform_rejected"]  # one failure, no second attempt


async def test_the_count_spaces_its_messages_two_minutes_apart(rig: Rig) -> None:
    await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    times = [e.at for e in rig.sim.log if e.text and "条数测试" in e.text]
    gaps = {(b - a).total_seconds() for a, b in pairwise(times)}
    assert gaps == {120.0}


async def test_the_count_stops_at_fifteen_without_a_failure(db: Database) -> None:
    rig = make_rig(db, quota=None, window_h=None)
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT).data
    assert count["n"] == 15 and count["capped"] is True and count["first_failure"] is None
    assert sum(1 for e in rig.sim.log if e.text and "条数测试" in e.text) == 15


async def test_the_phone_count_not_the_servers_ok_decides_n(db: Database) -> None:
    rig = make_rig(db, quota=8, window_h=24.0)
    rig.sim.swallow_after = 5  # the server accepts 8, the phone shows 5
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT).data
    assert (count["api_ok"], count["phone_received"], count["n"]) == (8, 5, 5)
    assert count["mismatch"] is True and count["first_problem_index"] == 6


async def test_when_nothing_is_accepted_n_is_zero_and_the_later_steps_are_skipped(
    db: Database,
) -> None:
    rig = make_rig(db, quota=0, window_h=24.0)
    plan = await rig.run()
    count = plan.step(StepId.COUNT).data
    assert count["n"] == 0 and count["api_ok"] == 0
    assert plan.step(StepId.MEDIA).status is StepStatus.SKIPPED
    assert plan.step(StepId.WINDOW).status is StepStatus.SKIPPED
    assert "0 message(s)" in (plan.step(StepId.MEDIA).skip_reason or "")
    with db.session() as session:
        stored = load_channel_probe_summary(session)
    assert stored is not None and stored.verdict == VERDICT_NOT_MET
    assert any("skipped" in note for note in stored.notes)


async def test_one_message_after_an_inbound_leaves_no_room_for_media_or_window(
    db: Database,
) -> None:
    rig = make_rig(db, quota=1, window_h=24.0)
    plan = await rig.run()
    assert plan.step(StepId.COUNT).data["n"] == 1
    assert plan.step(StepId.MEDIA).status is StepStatus.SKIPPED
    assert plan.step(StepId.WINDOW).status is StepStatus.SKIPPED
    assert summarize(plan).verdict == VERDICT_NOT_MET


# ----------------------------------------------------- step 2: pictures and typing


async def test_the_pictures_are_registered_then_sent_and_the_gif_is_judged_by_the_user(
    rig: Rig,
) -> None:
    plan = await rig.tick_until(done(StepId.MEDIA))  # type: ignore[arg-type]
    media = plan.step(StepId.MEDIA).data
    assert [e.text for e in rig.sim.log if e.kind == "image"] == [None, None, None]
    assert len(rig.sim.registered) == 3  # registered before they were sent
    assert {name: row["phone"] for name, row in media["images"].items()} == {
        "jpg": "arrived",
        "png": "arrived",
        "gif": "moving",
    }
    assert media["gif_animated"] is True
    assert media["typing"]["visible"] == "yes" and media["typing"]["hold_s"] == 30.0
    assert media["quote"] == "not_supported"
    assert rig.sim.typing == [True, False]


async def test_a_still_gif_and_invisible_typing_are_recorded(db: Database) -> None:
    rig = make_rig(db)
    rig.user.gif_moves = False
    rig.user.typing_visible = "no"
    plan = await rig.tick_until(done(StepId.MEDIA))  # type: ignore[arg-type]
    media = plan.step(StepId.MEDIA).data
    assert media["gif_animated"] is False and media["typing"]["visible"] == "no"
    stored = summarize(plan)
    assert stored.gif_animated is False and stored.typing_visible is False


async def test_a_picture_that_did_not_arrive_is_recorded_as_missing(db: Database) -> None:
    rig = make_rig(db)
    await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    rig.sim.swallow_after = 0  # from now on: accepted by the server, never on the phone
    plan = await rig.tick_until(done(StepId.MEDIA))  # type: ignore[arg-type]
    media = plan.step(StepId.MEDIA).data
    assert {row["phone"] for row in media["images"].values()} == {"missing"}
    assert all(row["api_ok"] for row in media["images"].values())
    assert media["gif_animated"] is None


async def test_the_typing_is_shown_only_after_the_user_says_they_are_looking(rig: Rig) -> None:
    plan = await rig.tick_until(
        lambda p: any(q.id.endswith("typing-ready") for q in rig.store.pending_questions(p))
    )
    assert rig.sim.typing == []  # not yet: it waits for the answer
    for question in rig.store.pending_questions(plan):
        rig.store.answer(question.id, "")
    await rig.tick_until(lambda _p: rig.sim.typing == [True, False])


async def test_a_small_n_splits_the_pictures_over_several_fresh_messages(db: Database) -> None:
    rig = make_rig(db, quota=2, window_h=24.0)  # N = 2, so one message per inbound
    plan = await rig.tick_until(done(StepId.MEDIA))  # type: ignore[arg-type]
    media_step = plan.step(StepId.MEDIA)
    assert [a.items for a in media_step.attempts] == [["jpg"], ["png"], ["gif"]]
    announcements = [t for t in sent_texts(rig.sim) if "第2步" in t]
    assert len(announcements) == 3  # a fresh message was asked for before every attempt
    assert len({key for key in rig.user.wrote_for if key[0] == "media"}) == 3
    # the typing check happens once, in the last attempt
    typing_attempts = [a for a in media_step.attempts if any(x.id == "typing" for x in a.actions)]
    assert typing_attempts == [media_step.attempts[-1]]


async def test_media_and_window_never_send_more_than_n_minus_one_per_inbound(db: Database) -> None:
    rig = make_rig(db, quota=4, window_h=24.0)  # N = 4: at most 3 per inbound afterwards
    plan = await rig.run()
    for step_id in (StepId.MEDIA, StepId.WINDOW):
        for attempt in plan.step(step_id).attempts:
            sends = [
                a
                for a in attempt.actions
                if a.kind in (ActionKind.SEND_TEXT, ActionKind.SEND_IMAGE) and a.send
            ]
            assert attempt.budget == 3 and len(sends) <= 3, (step_id, attempt.n)
    # and the platform confirms it: no inbound message was followed by more than N-1 probe sends
    assert all(
        entry.result in ("ok", "platform_rejected", "session_expired") for entry in rig.sim.log
    )


async def test_a_rejected_picture_stops_the_step_and_leaves_the_rest_untested(db: Database) -> None:
    rig = make_rig(db)
    rig.sim.reject_images_with = OutboundResult.failure(
        OutboundKind.REJECTED, "ret_-4", code=-4, ret=-4, errmsg="bad image"
    )
    plan = await rig.tick_until(done(StepId.MEDIA))  # type: ignore[arg-type]
    media = plan.step(StepId.MEDIA).data
    images = media["images"]
    assert images["jpg"]["phone"] == "not_delivered"
    assert images["jpg"]["failure"]["code"] == -4 and images["jpg"]["failure"]["errmsg"]
    assert images["png"]["phone"] == "not_tested" and images["gif"]["phone"] == "not_tested"
    assert media["gif_animated"] is None
    assert media["typing"]["sent"] is False  # nothing else is sent after a failure
    assert rig.sim.typing == []


# ----------------------------------------------------------------- step 3: window


@pytest.mark.parametrize(
    ("window_h", "lower", "upper", "verdict"),
    [
        (30.0, 25.0, None, VERDICT_MET),  # every point gets through
        (24.0, 23.0, 25.0, VERDICT_MET),
        (13.0, 12.0, 20.0, VERDICT_MET),  # the 12-hour point delivered is enough
        (11.9, 6.0, 12.0, VERDICT_NOT_MET),  # just short of 12 hours
        (5.0, 1.0, 6.0, VERDICT_NOT_MET),
        (0.5, None, 1.0, VERDICT_NOT_MET),  # not even one hour
    ],
)
async def test_the_window_is_found_between_the_measuring_points(
    db: Database, window_h: float, lower: float | None, upper: float | None, verdict: str
) -> None:
    rig = make_rig(db, quota=8, window_h=window_h)
    plan = await rig.run()
    data = plan.step(StepId.WINDOW).data
    if lower is None:
        assert data["lower_bound_h"] is None
    else:
        assert data["lower_bound_h"] == pytest.approx(lower, abs=0.01)
    if upper is None:
        assert data["upper_bound_h"] is None
    else:
        assert data["upper_bound_h"] == pytest.approx(upper, abs=0.01)
    assert summarize(plan).verdict == verdict


@pytest.mark.parametrize(
    ("quota", "verdict"), [(2, VERDICT_NOT_MET), (3, VERDICT_MET), (4, VERDICT_MET)]
)
async def test_three_messages_after_one_inbound_are_enough_and_two_are_not(
    db: Database, quota: int, verdict: str
) -> None:
    rig = make_rig(db, quota=quota, window_h=48.0)
    plan = await rig.run()
    stored = summarize(plan)
    assert stored.n_messages == quota and stored.verdict == verdict
    if verdict == VERDICT_MET:
        assert stored.window_lower_bound_h is not None and stored.window_lower_bound_h >= 25
    else:
        assert any("only 2 message" in reason for reason in stored.reasons)


async def test_the_window_sends_at_one_six_twelve_twenty_twenty_three_and_twenty_five_hours(
    db: Database,
) -> None:
    rig = make_rig(db, quota=8, window_h=48.0)
    plan = await rig.run()
    attempt = plan.step(StepId.WINDOW).attempts[0]
    assert attempt.inbound_at is not None
    hours = [
        (a.send.at - attempt.inbound_at).total_seconds() / 3600
        for a in attempt.actions
        if a.send is not None and a.send.at is not None
    ]
    assert hours == pytest.approx([1.0, 6.0, 12.0, 20.0, 23.0, 25.0], abs=0.001)
    assert plan.step(StepId.WINDOW).data["dropped_hours"] == []


async def test_a_small_n_keeps_the_latest_window_points_and_says_which_were_dropped(
    db: Database,
) -> None:
    rig = make_rig(db, quota=4, window_h=48.0)  # budget 3: only 20, 23 and 25 hours
    plan = await rig.run()
    data = plan.step(StepId.WINDOW).data
    assert [p["hours_planned"] for p in data["points"]] == [20.0, 23.0, 25.0]
    assert data["dropped_hours"] == [1.0, 6.0, 12.0]
    assert any("left out" in note for note in summarize(plan).notes)


async def test_a_failure_at_a_late_point_without_the_early_ones_cannot_decide(
    db: Database,
) -> None:
    rig = make_rig(db, quota=4, window_h=19.0)  # 20 h fails; 1, 6, 12 were never tried
    plan = await rig.run()
    data = plan.step(StepId.WINDOW).data
    assert data["lower_bound_h"] is None
    assert data["upper_bound_h"] == pytest.approx(20.0, abs=0.01)
    stored = summarize(plan)
    assert stored.verdict == VERDICT_UNDETERMINED and stored.window_lower_bound_h is None


async def test_the_window_stops_at_the_first_failure_and_never_retries(db: Database) -> None:
    rig = make_rig(db, quota=8, window_h=10.0)
    plan = await rig.run()
    results = [e.result for e in rig.sim.log if e.text and "窗口测试" in e.text]
    assert results == ["ok", "ok", "platform_rejected"]  # 1 h, 6 h, 12 h: then nothing more
    actions = plan.step(StepId.WINDOW).attempts[0].actions
    sends = [a for a in actions if a.kind is ActionKind.SEND_TEXT]
    assert [a.status for a in sends[3:]] == [ActionStatus.SKIPPED] * 3


async def test_the_phone_decides_which_window_points_were_delivered(db: Database) -> None:
    rig = make_rig(db, quota=8, window_h=48.0)
    await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    rig.sim.swallow_after = 3  # from now on the phone shows only three messages per inbound
    plan = await rig.run()
    data = plan.step(StepId.WINDOW).data
    assert (data["api_ok"], data["phone_received"], data["mismatch"]) == (6, 3, True)
    assert data["lower_bound_h"] == pytest.approx(12.0, abs=0.01)  # 1, 6, 12 delivered
    assert data["upper_bound_h"] == pytest.approx(20.0, abs=0.01)  # 20 accepted, not delivered


async def test_a_message_from_the_user_during_the_window_voids_the_attempt(db: Database) -> None:
    rig = make_rig(db, quota=8, window_h=48.0)
    await rig.tick_until(
        lambda p: running(p, StepId.WINDOW) and len(finished_sends(p, "window")) >= 2
    )
    rig.sim.user_writes()  # the user breaks the silence after the 6-hour message
    plan = await rig.run()
    window = plan.step(StepId.WINDOW)
    assert [a.outcome for a in window.attempts] == ["void", "complete"]
    assert "wrote to the bot" in (window.attempts[0].void_reason or "")
    assert window.voided_attempts == 1
    assert any("void" in title for title, _lines in rig.banner.shown)
    # the redo asked for a fresh message again and measured from it
    first, second = window.attempts[0].inbound_at, window.attempts[1].inbound_at
    assert first is not None and second is not None and second > first
    assert window.data["lower_bound_h"] >= 25


async def test_a_message_after_the_last_send_does_not_void_the_questions(db: Database) -> None:
    rig = make_rig(db, quota=8, window_h=48.0)
    plan = await rig.tick_until(
        lambda p: (
            bool(rig.store.pending_questions(p))
            and p.step(StepId.WINDOW).status is StepStatus.ACTIVE
        )
    )
    rig.sim.user_writes()  # the 25 hours are over: the user may write now
    for question in rig.store.pending_questions(plan):
        rig.store.answer(question.id, "6")
    plan = await rig.run()
    assert plan.step(StepId.WINDOW).voided_attempts == 0
    assert plan.step(StepId.WINDOW).data["phone_received"] == 6


# --------------------------------------------------------------- failures


async def test_a_platform_error_is_a_measurement_not_a_reason_to_redo(db: Database) -> None:
    rig = make_rig(db)
    rig.sim.script(
        "条数测试 1/",
        OutboundResult.failure(
            OutboundKind.REJECTED, "ret_-7", code=-7, ret=-7, errcode=None, errmsg="busy"
        ),
    )
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT)
    assert count.voided_attempts == 0 and len(count.attempts) == 1
    assert count.data["n"] == 0 and count.data["first_failure"]["code"] == -7
    assert count.data["first_failure"]["errmsg"] == "busy"


async def test_every_number_of_a_failure_is_recorded_and_the_message_is_redacted(
    db: Database,
) -> None:
    rig = make_rig(db)
    rig.sim.script(
        "条数测试 3/",
        OutboundResult.failure(
            OutboundKind.REJECTED,
            "ret_-3",
            code=-3,
            ret=-3,
            errcode=-5,
            errmsg=f"call {mobile()} failed",
            http_status=None,
        ),
    )
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    failure = plan.step(StepId.COUNT).data["first_failure"]
    assert (failure["code"], failure["ret"], failure["errcode"]) == (-3, -3, -5)
    assert mobile() not in failure["errmsg"]
    stored = summarize(plan)
    assert stored.failures[0]["errcode"] == -5 and stored.failures[0]["step"] == "count"


async def test_a_send_that_never_left_the_machine_is_repeated_a_minute_later(db: Database) -> None:
    rig = make_rig(db)
    rig.sim.script("条数测试 1/", OutboundResult.failure(OutboundKind.NETWORK, "connect_error"))
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    assert plan.step(StepId.COUNT).voided_attempts == 0
    assert sum(e.result.startswith("scripted") for e in rig.sim.log) == 1
    first_two = [e for e in rig.sim.log if e.text and "条数测试" in e.text][:2]
    assert (first_two[1].at - first_two[0].at).total_seconds() == 60.0
    assert plan.step(StepId.COUNT).data["n"] == 8


async def test_a_lasting_network_failure_voids_the_attempt_and_it_is_redone(db: Database) -> None:
    rig = make_rig(db)
    failure = OutboundResult.failure(OutboundKind.NETWORK, "connect_error")
    rig.sim.script("条数测试 1/", *[failure] * 6)  # the first try and five repeats
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT)
    assert [a.outcome for a in count.attempts] == ["void", "complete"]
    assert "interrupted" in (count.attempts[0].void_reason or "")
    assert count.data["n"] == 8  # measured on the second attempt


@pytest.mark.parametrize(
    "result",
    [
        OutboundResult.failure(OutboundKind.AUTH_EXPIRED, "needs_relogin", code=-14),
        OutboundResult.failure(OutboundKind.AMBIGUOUS, "timeout"),
        OutboundResult.failure(OutboundKind.UPLOAD_FAILED, "http_500"),
        OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "no_context_token"),
    ],
    ids=["login", "unknown-outcome", "upload", "local-refusal"],
)
async def test_failures_that_are_not_the_platforms_answer_void_the_attempt(
    db: Database, result: OutboundResult
) -> None:
    rig = make_rig(db)
    rig.sim.script("条数测试 1/", result)
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    count = plan.step(StepId.COUNT)
    assert count.attempts[0].outcome == "void" and count.voided_attempts == 1
    assert count.attempts[1].outcome == "complete"
    # the failed send was not retried within its attempt: exactly one scripted failure happened
    assert sum(e.result.startswith("scripted") for e in rig.sim.log) == 1


async def test_too_many_void_attempts_stop_the_plan(db: Database) -> None:
    rig = make_rig(db, options=ProbeOptions(watch_s=3600.0, max_attempts_per_step=2))
    ambiguous = OutboundResult.failure(OutboundKind.AMBIGUOUS, "timeout")
    rig.sim.script("条数测试 1/", *[ambiguous] * 5)
    plan = await rig.run()
    assert plan.status is PlanStatus.STOPPED
    assert "void" in (plan.stop_reason or "")


# ------------------------------------------------- restarts, stop, questions


async def test_a_restart_in_the_middle_continues_without_resending(db: Database) -> None:
    rig = make_rig(db)
    await rig.tick_until(lambda p: len(finished_sends(p, "count")) >= 3)
    rig.runner = ProbeRunner(  # the application restarts: a new runner on the same stored plan
        store=rig.store,
        channel=rig.sim,
        policy=rig.policy,
        clock=rig.clock,
        alerts=rig.alerts,
        banner=rig.banner,
    )
    plan = await rig.run()
    assert plan.status is PlanStatus.COMPLETED
    count_texts = [t for t in sent_texts(rig.sim) if "条数测试" in t]
    numbers = [int(t.split("条数测试 ")[1].split("/")[0]) for t in count_texts]
    assert numbers == list(range(1, len(numbers) + 1))  # 1, 2, 3, ... each exactly once


async def test_a_send_cut_off_by_a_restart_is_not_repeated_the_attempt_is_redone(
    db: Database,
) -> None:
    rig = make_rig(db)
    await rig.tick_until(lambda p: len(finished_sends(p, "count")) >= 2)

    def lose_the_result(plan: ProbePlan) -> None:
        action = next(
            a
            for a in plan.step(StepId.COUNT).attempts[0].actions
            if a.status is ActionStatus.PENDING
        )
        action.status = ActionStatus.ACTIVE  # the process died after marking it, before the result

    rig.store.update(lose_the_result)
    plan = await rig.tick_until(lambda p: p.step(StepId.COUNT).voided_attempts == 1)
    resent = [t for t in sent_texts(rig.sim) if "条数测试 3/" in t]
    assert resent == []  # the cut-off send was neither repeated nor counted as sent
    first = plan.step(StepId.COUNT).attempts[0]
    lost = next(a for a in first.actions if a.send and a.send.outcome == "unknown")
    assert lost.send is not None and lost.send.reason == "interrupted_by_restart"
    assert "restart" in (first.void_reason or "")
    plan = await rig.run()
    assert plan.step(StepId.COUNT).data["n"] == 8


async def test_stopping_the_plan_ends_the_sending_and_keeps_what_was_measured(rig: Rig) -> None:
    await rig.tick_until(lambda p: len(finished_sends(p, "count")) >= 2)
    rig.store.finish(PlanStatus.STOPPED, "stopped by the user")
    logged = len(rig.sim.log)
    for _ in range(5):
        await rig.runner.tick()
    assert len(rig.sim.log) == logged
    with rig.db.session() as session:
        stored = load_channel_probe_summary(session)
    assert stored is not None and stored.status == "stopped" and not stored.complete
    assert stored.n_messages is None  # step 1 was never finished: nothing is invented


async def test_the_probe_waits_for_an_answer_and_sends_nothing_meanwhile(db: Database) -> None:
    rig = make_rig(db)
    plan = await rig.tick_until(lambda p: bool(rig.store.pending_questions(p)))
    assert [q.id for q in rig.store.pending_questions(plan)] == ["count-1-count"]
    logged = len(rig.sim.log)
    for _ in range(10):
        await rig.runner.tick()
        await rig.clock.sleep(30)
    assert len(rig.sim.log) == logged  # unanswered: no new sends, and the step is not decided
    stored = rig.store.load()
    assert stored is not None and stored.step(StepId.COUNT).status is StepStatus.ACTIVE
    rig.store.answer("count-1-count", "7")
    plan = await rig.tick_until(done(StepId.COUNT))  # type: ignore[arg-type]
    assert plan.step(StepId.COUNT).data["n"] == 7 and plan.step(StepId.COUNT).data["mismatch"]


# ---------------------------------------------------- the optional experiment


async def test_the_empty_token_experiment_sends_one_text_without_a_context_token(
    db: Database,
) -> None:
    rig = make_rig(db, options=ProbeOptions(watch_s=3600.0, empty_token_experiment=True))
    plan = await rig.run()
    assert [s.id for s in plan.steps] == [
        StepId.COUNT,
        StepId.MEDIA,
        StepId.WINDOW,
        StepId.EMPTY_TOKEN,
    ]
    data = plan.step(StepId.EMPTY_TOKEN).data
    assert data["api_ok"] is True and data["delivered"] is True
    flagged = [e for e in rig.sim.log if e.empty_token]
    assert len(flagged) == 1 and "没有带 context_token" in (flagged[0].text or "")
    assert any(entry["empty_token"] for entry in rig.store.audit())


async def test_the_experiment_is_off_by_default(rig: Rig) -> None:
    plan = await rig.run()
    assert [s.id for s in plan.steps] == [StepId.COUNT, StepId.MEDIA, StepId.WINDOW]
    assert not any(e.empty_token for e in rig.sim.log)
