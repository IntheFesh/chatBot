"""Scripted input and recorded output for the terminal channel tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


class ScriptedInput:
    """A ``TextInput`` fed by the test: ``feed`` a line, ``close`` for the end of the input."""

    def __init__(self, *lines: str, close: bool = False) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for line in lines:
            self.feed(line)
        if close:
            self.close()

    def feed(self, line: str) -> None:
        self._queue.put_nowait(line if line.endswith("\n") else line + "\n")

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def readline(self) -> str | None:
        return await self._queue.get()


@dataclass
class RecordingOutput:
    """A ``TextOutput`` that keeps every line."""

    lines: list[str] = field(default_factory=list)

    def write_line(self, text: str) -> None:
        self.lines.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class FixedLabels:
    """A ``StickerLabels`` with a fixed table."""

    def __init__(self, **labels: str) -> None:
        self._labels = labels

    def label(self, sha256: str) -> str | None:
        return self._labels.get(sha256)
