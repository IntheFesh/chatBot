"""What the model writes that she must not say: event text and promises (R-SAFE-006, R-SAFE-002).

The reply model sometimes writes what only a person can do - ``[图片]`` lines copied from the
examples, "我拍给你看", "我给你打电话" - and the application of ``twin run`` must not let it through
(R-ENG-008):

* **event text** (``[图片…]``, ``[语音…]``, ``[通话…]``, ``[转账]``: every line of the template
  table of the import, R-IMP-007) is cut out of the reply; when nothing else is left the model is
  asked again, told why;
* **a promise** of something she cannot do (a photo, a call, a meeting) is a violation: the reply is
  asked for again with a note; on the last attempt the bubble is cut out when others remain; when
  the promise was all there was, the round fails like any other (late retries, then one short and
  natural answer and an alert, R-ENG-010) - and in no case does the user get the promise;
* **no photo ever goes out** because of it: the only picture she can send is a sticker of her
  library (R-SAFE-006); the channel is not even asked to send an image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.life.conftest import WorldFactory
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model
from twin.engine.prompt import VIOLATION_NOTES
from twin.engine.safety.commitments import CommitmentDetector
from twin.ingest.events import EVENT_TEMPLATES, default_detector, render_event_text

pytestmark = pytest.mark.integration

PROMISES = (
    "我拍给你看",
    "我给你打电话",
    "我发个语音给你",
    "我明天去找你",
    "我给你转点钱",
)


@dataclass(frozen=True)
class Event:
    """A message of the kinds she cannot reproduce, for ``render_event_text``."""

    kind: str
    is_sent: bool = True
    text: str | None = None
    call_status: str | None = None
    call_duration_s: int | None = None
    voice_seconds: int | None = None
    has_transcript: bool = False


def every_event_line() -> list[str]:
    """One rendering of every template of the table: what the examples of the import look like."""
    return [re.sub(r"\{\w+\}", "3", template) for template in EVENT_TEMPLATES.values()]


HER_ANSWERS = ("好的呀", "嗯嗯嗯", "知道啦")  # what she has said most often, in the past


async def quiet_world(make_world: WorldFactory) -> LifeWorld:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago, she is up
        model=proactive_model(opening_curve(base=0.02)),
        her_answers=HER_ANSWERS,
    )
    # the channel is asked for nothing but text: a picture would be a call of this method
    world.sent_images = []  # type: ignore[attr-defined]
    original = world.channel.send_image

    async def spy(*args: object, **kwargs: object) -> object:
        world.sent_images.append((args, kwargs))  # type: ignore[attr-defined]
        return await original(*args, **kwargs)  # type: ignore[arg-type]

    world.channel.send_image = spy  # type: ignore[method-assign]
    return world


def nothing_forbidden_went_out(world: LifeWorld) -> None:
    """No promise and no event line reached the user, the records, or the channel's image call."""
    assert world.sent_images == []  # type: ignore[attr-defined]
    detector = CommitmentDetector.from_services(world.services)
    for item in world.said:
        assert item.kind in ("text", "system"), f"{item.kind}: {item.text!r}"
        if item.persona:
            assert not default_detector.is_event_text(item.text), item.text
            assert not detector.promises(item.text), item.text
    for row in world.out_rows():
        assert row.kind in ("text", "sticker"), row.kind
        assert not default_detector.is_event_text(row.text) and not detector.promises(row.text)


def steps(row: object) -> dict[str, int]:
    return {a["step"]: a.get("count", 0) for a in getattr(row, "actions", None) or []}


async def test_event_text_is_cut_out_of_a_reply_and_the_rest_is_said(
    make_world: WorldFactory,
) -> None:
    world = await quiet_world(make_world)
    events = every_event_line()
    assert len(events) >= 15 and all(default_detector.is_event_text(line) for line in events)
    voice = render_event_text(Event("voice", voice_seconds=5))
    image = render_event_text(Event("image"), caption="一张桌子上放着一杯咖啡的照片")
    assert voice and image and image.startswith("[图片") and "咖啡" in image  # as in the examples
    world.deepseek.book.queue.append("\n".join([image, voice, "那个我也看到了", *events]))

    await world.say("你看到我发的了吗")
    await world.run_until_idle()
    assert [s.text for s in world.persona_said] == ["那个我也看到了"]  # the one real line
    assert world.deepseek.calls["reply"] == 1  # nothing was left to ask again about
    row = world.out_rows()[0]
    assert steps(row)["event_text_removed"] == len(events) + 2
    nothing_forbidden_went_out(world)
    assert_screen_matches_records(world)
    assert_clean_screen(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_reply_of_event_text_alone_is_asked_for_again_with_a_note(
    make_world: WorldFactory,
) -> None:
    world = await quiet_world(make_world)
    image = render_event_text(Event("image"), caption="一杯咖啡")
    assert image
    world.deepseek.book.queue.extend([image, "我看看"])

    await world.say("你看这个")
    await world.run_until_idle()
    first, second = world.deepseek.of_kind("reply")
    note = VIOLATION_NOTES["event_text_only"]
    assert note not in first.text and note in second.text  # she was told what was wrong
    assert [s.text for s in world.persona_said] == ["我看看"]
    row = world.out_rows()[0]
    assert steps(row)["regenerated"] == 1
    assert row.backend == "deepseek"
    nothing_forbidden_went_out(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


@pytest.mark.parametrize("promise", PROMISES)
async def test_a_promise_she_cannot_keep_is_rewritten_and_no_photo_is_sent(
    make_world: WorldFactory, promise: str
) -> None:
    world = await quiet_world(make_world)
    world.deepseek.book.queue.extend([f"{promise}\n等我一下", "现在不方便啦 下次吧"])

    await world.say("想看看你")
    await world.run_until_idle()
    first, second = world.deepseek.of_kind("reply")
    note = VIOLATION_NOTES["commitment"]
    assert note not in first.text and note in second.text
    assert [s.text for s in world.persona_said] == ["现在不方便啦 下次吧"]
    assert steps(world.out_rows()[0])["regenerated"] == 1
    nothing_forbidden_went_out(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_promise_that_stays_on_the_last_attempt_is_cut_out_and_the_rest_is_said(
    make_world: WorldFactory,
) -> None:
    world = await quiet_world(make_world)
    world.deepseek.book.when("想看看你", "我拍给你看", "我在吃饭呢")  # whatever she is told

    await world.say("想看看你")
    await world.run_until_idle()
    assert world.deepseek.calls["reply"] == 2  # once more, with the note; then it is cut out
    assert [s.text for s in world.persona_said] == ["我在吃饭呢"]
    performed = steps(world.out_rows()[0])
    assert performed["regenerated"] == 1 and performed["commitment_removed"] == 1
    nothing_forbidden_went_out(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


async def test_a_reply_that_is_only_a_promise_ends_in_one_short_natural_answer(
    make_world: WorldFactory,
) -> None:
    world = await quiet_world(make_world)
    world.deepseek.book.when("想看看你", "我拍给你看")  # on every attempt, in every round

    await world.say("想看看你")
    asked = world.now
    await world.run_until_idle()
    assert world.deepseek.calls["reply"] == 8  # two attempts, then three more rounds of them
    said = world.persona_said
    assert len(said) == 1 and said[0].text in HER_ANSWERS  # one of her own, not the promise
    assert said[0].at - asked >= timedelta(minutes=3 * 2)  # three late tries of 2 to 10 minutes
    row = world.out_rows()[0]
    assert row.backend == "fallback" and "fallback_answer" in steps(row)
    assert ("reply_failed", "warning") in world.alerts()  # and the user's computer says so
    nothing_forbidden_went_out(world)
    assert_screen_matches_records(world)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
