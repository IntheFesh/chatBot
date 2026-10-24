"""A synthetic week for the consistency-audit tests (round 15, R-EVAL-004).

``build_week`` fills a memory and the conversation with the bot with a week of invented content:
a life line with plan entries and one the bot improvised, replies in which "she" talks about
herself, and facts of every source.  Two contradictions are planted - the sentences say so on
purpose, they are not hidden - and ``AuditModel`` is the DeepSeek that finds them: a ``respx`` side
effect that reads the numbered records of the prompt, finds the lines that contain a marker and
answers in the real JSON shape.  Nothing here is a real conversation.

The clock of the tests (``ManualClock``) is Friday 9 October 2026, 12:00 UTC (07:00 in Chicago).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from tests.support.deepseek import completion, error, request_json
from tests.support.memory import add_event, add_fact
from twin.engine.turns import BotTurnStore, OutboundBubble, ReplyMeta
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.memory.memory import Memory
from twin.memory.records import FactRecord, LifelineRecord
from twin.services import Services

CHICAGO = ZoneInfo("America/Chicago")
KEY = "synthetic-test-key-0001"
LIBRARY = "在图书馆看书"
MEETING = "在公司开会"
HOME_ALL_DAY = "整天都在家躺着"
STAYED_HOME_OLD = "上个月一直在家躺着养病"
BEIJING = "她在北京读研"
SHANGHAI = "她刚搬到上海工作"
OLD_INVENTION = "她养过一只橘猫"
USER_JOB = "他在芝加哥一家银行上班"
SECRET_MARK = "只在评估里出现的暗号"


@dataclass
class Week:
    """Ids of what ``build_week`` stored."""

    memory: Memory
    library: LifelineRecord
    meeting: LifelineRecord
    home_reply_id: str
    real_fact: FactRecord
    invented_fact: FactRecord
    old_invention: FactRecord
    user_fact: FactRecord
    events: dict[str, LifelineRecord] = field(default_factory=dict)


def local(day: date, hour: int, minute: int = 0) -> datetime:
    """A moment on Chicago's wall clock, as UTC."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CHICAGO).astimezone(UTC)


def say(store: BotTurnStore, at: datetime, user: str, bot: str) -> str:
    """One exchange: what the user wrote and what the bot answered (one bubble per line)."""
    store.add_inbound(at=at, kind="text", text=user)
    bubbles = [
        OutboundBubble(line, at + timedelta(seconds=30 + 5 * number))
        for number, line in enumerate(bot.split("\n"))
    ]
    return store.add_reply(bubbles, ReplyMeta("deepseek"))[0].id


def today_of(services: Services) -> date:
    """The date on Chicago's wall clock now (the week is built relative to it)."""
    return services.clock.now_utc().astimezone(CHICAGO).date()


def build_week(services: Services) -> Week:
    """Store the week (the DeepSeek key as well); everything is dated relative to today.

    With the test clock (Friday 9 October 2026) the days are: the life line of Tuesday the 6th
    (library, meeting), of the 8th (a visit to the hospital, improvised), an entry of the 29th of
    September (before the window) and one of the 7th that is invalidated; replies on the 3rd, the
    6th and the 8th, and one on 25 September (before the window); facts known on 1 September (a
    real one, importance 5, and an old invention, and one about him) and a fresh invention of the
    7th.
    """
    services.secrets.set(DEEPSEEK_SECRET, KEY)
    today = today_of(services)

    def ago(days: int) -> date:
        return today - timedelta(days=days)

    online = local(ago(11), 7)
    known = local(ago(38), 7)
    tuesday = ago(3)
    memory = Memory(services)
    memory.store.mark_bot_online(online)
    library = add_event(memory, tuesday, LIBRARY, start="09:00", end="11:00", created_at=online)
    meeting = add_event(memory, tuesday, MEETING, start="13:00", end="15:00", created_at=online)
    older = add_event(
        memory, ago(10), "在咖啡店看书", start="10:00", end="11:00", created_at=online
    )  # before the window
    gone = add_event(
        memory, ago(2), "在跑步机上跑步", start="08:00", end="09:00", created_at=online
    )
    memory.store.invalidate_event(gone.id, by=None, at=online + timedelta(days=1))
    improvised = add_event(
        memory,
        ago(1),
        "去医院复查",
        start="14:00",
        end="16:00",
        source="improvised",
        created_at=online,
    )
    store = BotTurnStore(services.db, services.clock)
    say(store, local(ago(6), 20), "今天干嘛了", "刚下班，好累\n晚上想早点睡")
    home_reply = say(store, local(tuesday, 21, 30), "你今天忙吗", f"今天{HOME_ALL_DAY}，哪儿也没去")
    say(store, local(ago(1), 22), "周末有安排吗", f"我周末想去爬山 {SECRET_MARK}")
    say(store, local(ago(14), 20), "那天呢", STAYED_HOME_OLD)  # before the window
    real = add_fact(memory, BEIJING, known, source="real_record", importance=5, embed=False)
    invented = add_fact(
        memory,
        SHANGHAI,
        local(ago(2), 21),
        source="bot_invented",
        evidence={"kind": "bot_turns", "ids": [home_reply]},
        embed=False,
    )
    old_invention = add_fact(
        memory, OLD_INVENTION, known, source="bot_invented", embed=False
    )  # a fact of no relevance, from before the window
    user_fact = add_fact(
        memory, USER_JOB, known, source="user_said", subject="user", embed=False
    )  # about him, not her
    return Week(
        memory,
        library,
        meeting,
        home_reply,
        real,
        invented,
        old_invention,
        user_fact,
        {"older": older, "gone": gone, "improvised": improvised},
    )


# ---------------------------------------------------------------------- the model


@dataclass
class Rule:
    """When the prompt holds both markers, the model reports a contradiction between them."""

    first: str
    second: str
    fields: dict[str, Any] = field(default_factory=dict)


REF_LINE = re.compile(r"^([LRF]\d+)(?:[ （]).*$", re.MULTILINE)


def ref_of(prompt: str, marker: str) -> str | None:
    """The number of the first record line that contains ``marker``."""
    for line in prompt.splitlines():
        found = REF_LINE.match(line)
        if found and marker in line:
            return found.group(1)
    return None


@dataclass
class AuditModel:
    """A respx side effect that answers the audit prompt from rules the test sets."""

    rules: list[Rule] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)  # raw replies, used in turn, instead of rules
    prompt_tokens: int = 3000
    completion_tokens: int = 400
    failure: int | None = None  # answer every call with this HTTP status
    requests: list[dict[str, Any]] = field(default_factory=list)
    on_request: Callable[[dict[str, Any]], None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        self.requests.append(body)
        if self.on_request is not None:
            self.on_request(body)
        if self.failure is not None:
            return error(self.failure, "scripted failure")
        text = self.replies.pop(0) if self.replies else self._answer(body)
        return httpx.Response(
            200,
            json=completion(
                text,
                prompt=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                request_id=f"req-{len(self.requests)}",
            ),
        )

    @property
    def prompts(self) -> list[str]:
        """The user message of every audit request (the first one of each conversation)."""
        return [str(r["messages"][1]["content"]) for r in self.requests]

    def _answer(self, body: dict[str, Any]) -> str:
        prompt = str(body["messages"][1]["content"])
        found: list[dict[str, Any]] = []
        for rule in self.rules:
            first, second = ref_of(prompt, rule.first), ref_of(prompt, rule.second)
            if first is None or second is None:
                continue
            item: dict[str, Any] = {
                "time": "10-06 下午",
                "first": {"ref": first, "quote": None},
                "second": {"ref": second, "quote": None},
                "related": [],
                "severity": "obvious",
                "reason": "两处说法不可能同时为真",
                "keep": None,
                "rewrite": None,
            }
            item.update(rule.fields)
            found.append(item)
        return json.dumps({"contradictions": found}, ensure_ascii=False)


def the_two_contradictions() -> list[Rule]:
    """The library against "at home all day", and the real fact against the invented one."""
    return [
        Rule(LIBRARY, HOME_ALL_DAY, {"severity": "obvious"}),
        Rule(
            BEIJING,
            SHANGHAI,
            {
                "severity": "minor",
                "reason": "读研的城市和工作的城市对不上",
                "rewrite": "她在北京工作",
            },
        ),
    ]
