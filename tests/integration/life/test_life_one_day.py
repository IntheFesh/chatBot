"""A whole day of her life, from the night before to the morning after (R-ARCH-004 end to end).

The application of ``twin run`` runs on the terminal channel, a manual clock and a DeepSeek that is
made up (``tests/support/life_world.py``).  The user lives one Friday in Chicago:

* the evening before he says good night; she says good night to him, and sleeps;
* she wakes and greets him by herself, in the window the plan allows;
* at half past nine he writes three messages in a row: she answers once, after he is done;
* he sends a photo, and a few minutes later a sticker: she sees the photo through its description,
  answers the sticker with one of her own, and no photo of anyone ever goes out;
* he is silent through the noon: she writes first (a meal, a piece of her day) but never in her
  sleep and never closer than an hour apart;
* in the afternoon she is busy: his question waits for the busy window's latency;
* in the evening he says he has an exam tomorrow: half an hour later the memory holds a follow-up;
* she says good night before she goes to sleep;
* at ten past three, with her in deep sleep, he writes again: she answers when she has woken,
  as if just woken, and she brings up the exam, which the memory (not the script) put in front of
  the model;
* after the exam she asks how it went.

Each step asserts what the rules allow (the time of every message, the number of bubbles, the
backend), what was stored (``bot_turns``, ``proactive_log``, the memory) and, at the end, what
holds for any story (``tests/support/life_checks.py``) - among it that nothing of the conversation
reached the real messages, her profile or the library of examples (CLAUDE.md rule 7).
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pytest

from tests.fixtures.synth_export import make_image_bytes
from tests.integration.life.conftest import WorldFactory
from tests.support.embedding import HashingBackend
from tests.support.life_checks import (
    assert_bot_text_stays_out,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_proactive_rules,
    assert_screen_matches_records,
    assert_within_quota,
    snapshot_isolation,
)
from tests.support.memory import FollowupRule
from tests.support.proactive_world import opening_curve, proactive_model

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
SATURDAY = date(2026, 10, 10)
# her opening curve: the wake-up, the meals, and the half hour before she goes to bed
PEAKS = {31: 0.5, 32: 0.5, 48: 0.6, 49: 0.6, 73: 0.6, 74: 0.6, 90: 0.5, 91: 0.5, 92: 0.5}
PEAKS |= {93: 0.5, 94: 0.5, 95: 0.5}
QUIET_WINDOW_S = 15.0


def sent_kinds(world, day: date) -> list[str]:
    return [row.kind for row in world.proactive_rows(outcomes=["sent"]) if row.local_date == day]


async def test_one_day_of_her_life(make_world: WorldFactory, embedder: HashingBackend) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 3, 0, tzinfo=UTC),  # Thursday 22:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks=PEAKS)),
    )
    sticker = world.add_her_sticker("开心")  # part of her past: a sticker she used before
    before = snapshot_isolation(world)  # her real messages, as the story begins
    book = world.deepseek.book
    book.when("早饭", "吃啦", "你呢")
    book.when("[图片", "哇 看着好香")
    book.when("[表情包", "笑死我了", "[表情包:开心]")
    book.when("下午有课", "有 一点点")
    book.when("明天考试", "加油呀", "考完告诉我")
    book.when(
        "对方考试（", "对了 你考试怎么样了", in_context=True
    )  # the follow-up is in the memory
    world.deepseek.memory.followups.append(
        FollowupRule("明天考试", {"text": "对方考试", "due": "明天上午九点", "window_minutes": 180})
    )

    # ---- the evening before: good night, and she sleeps --------------------------------------
    await world.say("我先睡啦，晚安")
    await world.run_until_idle()
    first = world.persona_said[0]
    assert first.at - world.local(22, 0, on=date(2026, 10, 8)) < timedelta(minutes=5)

    # ---- the night, and her waking: a greeting of her own in the plan's window ----------------
    await world.run_until(world.local(9, 20, on=FRIDAY))
    plan = world.plan(FRIDAY)
    assert plan.morning is not None and plan.night is not None
    wake, onset = plan.morning.wake, plan.night.onset
    assert_never_in_deep_sleep(world)
    greeting = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "greeting"]
    assert len(greeting) == 1
    assert wake + timedelta(minutes=5) <= greeting[0].at <= wake + timedelta(minutes=40)
    assert greeting[0].her_state in ("free", "sleep_edge")
    assert [s.text for s in world.persona_said if s.at >= wake] == ["早啊", "刚醒"]

    # ---- three messages in a row: one answer, after he is done -------------------------------
    replies = world.deepseek.calls["reply"]
    await world.run_until(world.local(9, 30, on=FRIDAY))
    await world.say("在吗")
    await world.run_for(seconds=9)
    await world.say("今天好冷")
    await world.run_for(seconds=12)
    await world.say("你吃早饭了没")
    last_word = world.now
    await world.run_until_idle()
    assert world.deepseek.calls["reply"] == replies + 1  # one round, one request
    asked = world.deepseek.of_kind("reply")[-1].last_user
    assert all(text in asked for text in ("在吗", "今天好冷", "你吃早饭了没"))
    answer = [s for s in world.persona_said if s.at > last_word]
    assert [s.text for s in answer] == ["吃啦", "你呢"]
    assert answer[0].at - last_word >= timedelta(seconds=QUIET_WINDOW_S)  # after he is quiet
    assert answer[1].at - answer[0].at < timedelta(minutes=2)  # one burst
    reply_row = world.out_rows()[-1]
    assert reply_row.backend is None  # the second bubble of a reply; the first carries the meta
    assert any(a["step"] == "decided_free" for a in world.out_rows()[-2].actions)

    # ---- a photo, then a sticker --------------------------------------------------------------
    await world.run_until(world.local(10, 40, on=FRIDAY))
    await world.say_picture(make_image_bytes(random.Random(3), "PNG"))
    await world.run_until_idle()
    assert world.deepseek.calls["caption"] == 1
    photo = [r for r in world.in_rows() if r.kind == "image"]
    assert len(photo) == 1 and "咖啡" in photo[0].text  # she saw it through its description
    assert world.persona_said[-1].text == "哇 看着好香"
    await world.run_for(minutes=3)
    await world.say_sticker(make_image_bytes(random.Random(4), "PNG"))
    await world.run_until_idle()
    sent = world.persona_said[-2:]
    assert [s.kind for s in sent] == ["text", "sticker"] and sent[0].text == "笑死我了"
    sticker_rows = [r for r in world.out_rows() if r.kind == "sticker"]
    assert [r.sticker_md5 for r in sticker_rows] == [sticker]
    assert not [s for s in world.said if s.kind == "image"]  # no photo ever leaves

    # ---- noon: he is silent, she writes first, for her own reasons ----------------------------
    await world.run_until(world.local(13, 20, on=FRIDAY))
    noon = [
        r
        for r in world.proactive_rows(outcomes=["sent"])
        if r.local_date == FRIDAY and r.at > last_word
    ]
    assert noon and {r.kind for r in noon} <= {"meal", "share", "silence", "edge"}
    assert all(r.at - world.local(10, 43, on=FRIDAY) > timedelta(minutes=30) for r in noon)

    # ---- afternoon: she is busy, the answer takes as long as the busy window says -------------
    await world.run_until(world.local(13, 30, on=FRIDAY))
    asked_at = world.now
    await world.say("下午有课吗")
    await world.run_until_idle()
    late = world.persona_said[-1]
    assert late.text == "有 一点点"
    assert late.at - asked_at >= timedelta(minutes=20)  # the latency of her busy hours: 1500 s
    assert any(a["step"] == "decided_busy" for a in world.out_rows()[-1].actions)

    # ---- the evening: tomorrow's exam, and the memory ----------------------------------------
    await world.run_until(world.local(19, 0, on=FRIDAY))
    await world.say("我明天考试 好紧张")
    await world.run_until_idle()
    assert world.persona_said[-2:][0].text == "加油呀"
    await world.run_until(world.local(19, 45, on=FRIDAY))  # thirty quiet minutes later ...
    await world.drain_jobs()
    held = [f for f in world.followups() if "考试" in f.text]
    assert len(held) == 1 and held[0].status == "open"
    assert held[0].due_at.astimezone(world.zone).date() == SATURDAY

    # ---- good night, before she sleeps -------------------------------------------------------
    await world.run_until(onset)
    bedtime = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "bedtime"]
    friday_bedtime = [r for r in bedtime if r.local_date == FRIDAY]
    assert len(friday_bedtime) == 1
    assert onset - timedelta(minutes=60) <= friday_bedtime[0].at <= onset - timedelta(minutes=15)

    # ---- deep in the night he writes: she answers when she has woken -------------------------
    await world.run_until(world.local(3, 10, on=SATURDAY))
    spoken = len(world.persona_said)
    await world.say("还没睡 睡不着")
    wake_up = world.plan(SATURDAY).morning.wake
    await world.run_for(seconds=60)  # his quiet window is over: she has decided
    assert world.decision() is not None and world.decision().mode == "asleep"
    assert world.decision().send_at >= wake_up
    await world.run_until_idle()
    assert len(world.persona_said) == spoken + 1
    answer = world.persona_said[-1]
    assert answer.text == "对了 你考试怎么样了"  # the follow-up came from the memory block
    assert wake_up + timedelta(minutes=5) <= answer.at <= wake_up + timedelta(minutes=40)
    prompt = world.deepseek.of_kind("reply")[-1].last_user
    assert "你刚醒" in prompt  # the model is told she has just woken up

    # ---- after the exam she asks how it went -------------------------------------------------
    await world.run_until(world.local(12, 0, on=SATURDAY))
    followup = [r for r in world.proactive_rows(outcomes=["sent"]) if r.kind == "followup"]
    assert len(followup) == 1
    due = held[0].due_at
    window = timedelta(minutes=held[0].window_minutes)
    assert due + timedelta(minutes=10) <= followup[0].at <= due + window
    earlier = [r for r in world.proactive_rows(outcomes=["sent"]) if r.at < followup[0].at]
    assert followup[0].at - earlier[-1].at >= timedelta(minutes=60)  # never closer than an hour

    # ---- what holds for any day ---------------------------------------------------------------
    assert "greeting" in sent_kinds(world, FRIDAY) and "bedtime" in sent_kinds(world, FRIDAY)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_within_quota(world)
    assert_proactive_rules(world, FRIDAY, FRIDAY)
    assert world.deepseek.unexpected == []
    assert {c for c, _ in world.alerts()} <= {"lifeline_corrected"}
    assert not world.jobs().get("failed")
    await assert_bot_text_stays_out(world, before, embedder)
