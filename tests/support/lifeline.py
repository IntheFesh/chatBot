"""A scripted DeepSeek for the life line: it draws the days and checks them as the test says.

``ScriptedLifelineModel`` is a ``respx`` side effect.  It reads which prompt it was given - the
generator's or the checker's - and answers with the next entry of the matching script (the last
entry repeats), in the real JSON shape.  Everything it was asked is kept in ``requests``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from tests.support.deepseek import ok, request_json

Event = dict[str, Any]


def event(
    start: str,
    end: str,
    activity: str,
    *,
    place: str = "家",
    mood: str = "平静",
    busy: bool = False,
    detail: str | None = None,
) -> Event:
    found: Event = {
        "start": start,
        "end": end,
        "activity": activity,
        "place": place,
        "mood": mood,
        "busy": busy,
    }
    if detail:
        found["detail"] = detail
    return found


def good_workday() -> list[Event]:
    """A consistent Friday for a routine that sleeps 23:30-07:30 and is busy 13:00-17:00."""
    return [
        event("07:45", "08:15", "吃早饭", detail="煎蛋配牛奶"),
        event("08:30", "12:00", "在图书馆看文献", place="图书馆"),
        event("12:10", "12:50", "吃午饭", place="食堂"),
        event("13:00", "17:00", "上课和做实验", place="实验室", mood="有点累", busy=True),
        event("17:30", "18:30", "吃晚饭", place="食堂"),
        event("19:00", "22:30", "写作业，顺便追剧", mood="放松"),
        event("22:45", "23:20", "洗漱，准备睡觉"),
    ]


@dataclass
class ScriptedLifelineModel:
    days: list[list[Event]] = field(default_factory=lambda: [good_workday()])
    checks: list[list[dict[str, Any]]] = field(default_factory=lambda: [[]])
    requests: list[dict[str, Any]] = field(default_factory=list)
    draws: int = 0
    reviews: int = 0
    fail_from_call: int | None = None
    prompt_tokens: int = 300
    completion_tokens: int = 120

    def prompts(self, kind: str) -> list[str]:
        """The user messages of the draws (``draw``) or of the reviews (``check``)."""
        marker = "编写“她”这一天的生活安排" if kind == "draw" else "日程审查员"
        return [
            body["messages"][1]["content"]
            for body in self.requests
            if marker in body["messages"][0]["content"]
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        self.requests.append(body)
        if self.fail_from_call is not None and len(self.requests) >= self.fail_from_call:
            return httpx.Response(400, json={"error": {"message": "scripted failure"}})
        system = body["messages"][0]["content"]
        if "编写“她”这一天的生活安排" in system:
            reply: dict[str, Any] = {"events": self.days[min(self.draws, len(self.days) - 1)]}
            self.draws += 1
        elif "日程审查员" in system:
            reply = {"contradictions": self.checks[min(self.reviews, len(self.checks) - 1)]}
            self.reviews += 1
        else:
            raise AssertionError("the model was asked something the life line never asks")
        return ok(
            content=json.dumps(reply, ensure_ascii=False),
            prompt=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            hit=0,
        )
