"""What the user sees: the terminal screen of the console channel, as messages with their time.

Shared by :mod:`tests.support.life_world` (the world of the scenarios) and
:mod:`tests.support.wechat_double` (the phone of the iLink scenarios), which both end in a list of
:class:`Said`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from tests.support.life_clock import LifeClock
from twin.channel.local import TYPING_TEXT
from twin.commands import texts

PREFIX = texts.PREFIX
CONTINUATION = " " * 5  # how the terminal indents the further lines of a message


@dataclass(frozen=True)
class Said:
    """One message she sent, as the terminal showed it, and when."""

    at: datetime
    kind: Literal["text", "sticker", "image", "system"]
    text: str

    @property
    def persona(self) -> bool:
        return self.kind != "system"


class TimedOutput:
    """The terminal screen: every line with the moment the clock showed when it was written."""

    def __init__(self, clock: LifeClock) -> None:
        self._clock = clock
        self.lines: list[tuple[datetime, str]] = []

    def write_line(self, text: str) -> None:
        self.lines.append((self._clock.now_utc(), text))

    def messages(self) -> list[Said]:
        """The messages of the bot (the further lines of a message put back together)."""
        found: list[Said] = []
        for at, line in self.lines:
            if line.startswith("bot: "):
                body = line.removeprefix("bot: ")
                kind: Literal["text", "sticker", "image", "system"] = "text"
                if body.startswith(PREFIX):
                    kind = "system"
                elif body.startswith("[表情包："):
                    kind = "sticker"
                elif body.startswith("[图片]"):
                    kind = "image"
                found.append(Said(at, kind, body))
            elif line.startswith(CONTINUATION) and found:
                found[-1] = replace(
                    found[-1], text=found[-1].text + "\n" + line[len(CONTINUATION) :]
                )
        return found

    def typing_times(self) -> list[datetime]:
        return [at for at, line in self.lines if line == TYPING_TEXT]
