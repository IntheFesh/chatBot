"""``/评分 <1-5> [备注]``: the user's mark for the proactive messages (R-CMD-002, R-EVAL-005).

The answer is immediate and in the system's voice.  A rating is a command like any other:
its message and its answer are kept as audit rows flagged ``is_command`` and never reach the
memory, the learning, the retrieval or the training set (R-CMD-001); the score and the note are
written to ``ratings`` (the note sealed), which the weekly audit and the M3 gate read
(:mod:`twin.eval.proactive_audit`).

The argument is read forgivingly (R-CMD-003): ``4``, ``４``, ``四``, ``4分``, ``4/5``, ``4星``,
``五星``, with a colon or comma before the note, with or without the note.  Anything outside 1-5
(``0``, ``6``, ``10``, ``4.5``) or without a number is answered with the usage, and nothing is
written.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from datetime import timedelta

from twin.clock import Clock
from twin.commands import texts
from twin.commands.registry import CommandCall, CommandSpec, UsageError
from twin.schedule.proactive.store import RatingStore, average
from twin.schedule.time_service import TimeService

NAME = "评分"
WEEK = timedelta(days=7)
CHINESE_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
_SCORE = re.compile(
    r"^[\s:=]*(?P<num>[0-9]+(?:\.[0-9]+)?|[一二三四五])\s*"
    r"(?:/\s*5(?![0-9])|个?星|分)?[\s,，.。;；:：、-]*(?P<note>.*)$",
    re.DOTALL,
)


def parse_rating(args: str) -> tuple[int, str]:
    """The score and the note in the argument of ``/评分``; :class:`UsageError` if there is none."""
    text = unicodedata.normalize("NFKC", args).strip()
    if not text:
        raise UsageError("")
    found = _SCORE.match(text)
    if found is None:
        raise UsageError(texts.RATING_NO_NUMBER)
    token = found.group("num")
    if token in CHINESE_DIGITS:
        score = CHINESE_DIGITS[token]
    elif token.isdigit():
        score = int(token)
    else:
        raise UsageError(texts.RATING_NOT_WHOLE)
    if not 1 <= score <= 5:
        raise UsageError(texts.RATING_RANGE)
    note = found.group("note").strip()
    return score, note


class RatingCommand:
    """The handler of ``/评分`` (it writes to ``ratings`` and says what the week looks like)."""

    def __init__(self, store: RatingStore, clock: Clock, time: TimeService) -> None:
        self._store = store
        self._clock = clock
        self._time = time

    async def handle(self, call: CommandCall) -> str:
        score, note = parse_rating(call.args)
        now = call.context.at if call.context.at is not None else self._clock.now_utc()
        today = self._time.local_date(now)

        def write() -> tuple[int, float | None]:
            self._store.add(score, note or None, at=now, local_date=today)
            week = self._store.between(now - WEEK, now + timedelta(seconds=1))
            return len(week), average(week)

        count, mean = await asyncio.to_thread(write)
        reply = texts.RATING_DONE.format(score=score)
        if note:
            reply += texts.RATING_NOTE_KEPT
        if mean is not None:
            reply += texts.RATING_WEEK.format(count=count, mean=f"{mean:.1f}")
        return reply


def rating_command(store: RatingStore, clock: Clock, time: TimeService) -> CommandSpec:
    """The ``/评分`` entry of the command table (``router.register(rating_command(...))``)."""
    command = RatingCommand(store, clock, time)
    return CommandSpec(
        name=NAME,
        group="学习与评分",
        summary=texts.RATING_SUMMARY,
        syntax=texts.RATING_SYNTAX,
        example=texts.RATING_EXAMPLE,
        handler=command.handle,
        aliases=("rate", "rating"),
    )
