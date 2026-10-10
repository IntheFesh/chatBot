"""The terminal screens of the evaluation (R-EVAL-001, R-EVAL-003).

Two screens share one way of reading keys and one way of showing text:

* :class:`BlindSession` shows the conversation before a reply and two candidates, side by side in a
  random order, and waits for ``1`` (the left one is hers), ``2`` (the right one), ``s`` (skip - it
  is not counted) or ``q`` (save and leave).  Every decision is written the moment it is made, so
  a session that stops anywhere is continued with ``--resume``.
* :class:`MemorySession` (in :mod:`twin.eval.memory_test`) uses the same keys to review the
  automatic verdicts.

Input and output are injected: :class:`KeySource` hands out one key at a time (a line of text in
a test, a real key press in a terminal - :class:`TerminalKeys`), and the output is a rich
:class:`~rich.console.Console` on any stream.  Everything shown is a :class:`~rich.text.Text`, so
a message with brackets in it (``[拥抱]``) is never taken for markup.

The backend that wrote a candidate is never shown while judging.  When several backends are tested
on the same contexts, the pairs of one context are put far apart in the order the user meets them
(:func:`presentation_order`), so that a reply recognised from the earlier pair cannot give her
away.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TextIO

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from twin.eval.render import (
    HER_LABEL,
    USER_LABEL,
    Candidate,
    StickerLabel,
    render_candidate,
    render_turn,
)
from twin.eval.store import EvalStore, ItemView, RunView

KEY_LEFT = "1"
KEY_RIGHT = "2"
KEY_SKIP = "s"
KEY_QUIT = "q"
KEYS_BLIND = (KEY_LEFT, KEY_RIGHT, KEY_SKIP, KEY_QUIT)
CTRL_C = "\x03"
CTRL_D = "\x04"


class KeySource(Protocol):
    """Where the user's keys come from; ``None`` means the input ended."""

    def read_key(self) -> str | None: ...


class LineKeys:
    """One key per line of a text stream (the first character, lower case; Enter is ``""``)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def read_key(self) -> str | None:
        line = self._stream.readline()
        if line == "":
            return None
        text = line.strip()
        return text[0].lower() if text else ""


class TerminalKeys:
    """Real key presses from a terminal (no Enter needed); lines when the input is not a tty."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdin
        self._lines = LineKeys(self._stream)

    def read_key(self) -> str | None:
        if not self._stream.isatty():
            return self._lines.read_key()
        key = _press(self._stream)
        if key in (CTRL_C, CTRL_D):
            return KEY_QUIT
        if key in ("\r", "\n"):
            return ""
        return key.lower()


def _press(stream: TextIO) -> str:
    if sys.platform == "win32":  # pragma: win32-only
        import msvcrt

        return msvcrt.getwch()
    import termios
    import tty

    descriptor = stream.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor, termios.TCSADRAIN)  # keep what is typed ahead
        return stream.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)


def ask(keys: KeySource, console: Console, prompt: str, allowed: Sequence[str]) -> str | None:
    """Ask until one of ``allowed`` is pressed; ``None`` when the input ends."""
    while True:
        console.print(Text(prompt, style="bold"))
        key = keys.read_key()
        if key is None:
            return None
        if key in allowed:
            return key
        console.print(Text(f"（没有这个选项：{key!r}）", style="dim"))


# ----------------------------------------------------------------- blind screen


def presentation_order(items: Sequence[ItemView]) -> list[ItemView]:
    """The order the pairs are shown in: the pairs of one context are a whole pass apart.

    With one backend this is the order of the run.  With several, pass ``p`` shows every context
    once (with the backend that comes ``p`` places after a per-context starting point), so a
    context returns only after all the others were shown.
    """
    contexts: dict[str, list[ItemView]] = defaultdict(list)
    for item in items:
        contexts[item.sample_key].append(item)
    rank: dict[str, tuple[int, int]] = {}
    for number, group in enumerate(contexts.values()):
        group.sort(key=lambda i: i.seq)
        start = number % len(group)  # each backend comes first for a share of the contexts
        for position, item in enumerate(group):
            rank[item.id] = ((position - start) % len(group), number)
    return sorted(items, key=lambda i: (*rank[i.id], i.seq))


def sides_of(item: ItemView) -> tuple[Candidate, Candidate]:
    """``(left, right)``: the bot's reply is on the side the coin of the item says."""
    real = Candidate.from_json(item.payload["real"])
    bot = Candidate.from_json(item.payload["bot"])
    return (bot, real) if item.left_is_bot else (real, bot)


def correct_choice(item: ItemView, key: str) -> bool:
    """True when the key points at her real reply."""
    chose_left = key == KEY_LEFT
    return chose_left != bool(item.left_is_bot)


@dataclass
class SessionOutcome:
    judged: int = 0
    skipped: int = 0
    quit: bool = False
    remaining: int = 0


class BlindSession:
    """Shows the pairs of a run one after the other and records the decisions."""

    def __init__(
        self,
        store: EvalStore,
        run: RunView,
        label: StickerLabel,
        console: Console,
        keys: KeySource,
    ) -> None:
        self._store = store
        self._run = run
        self._label = label
        self._console = console
        self._keys = keys

    def pending(self) -> list[ItemView]:
        """The generated pairs the user has not decided yet, in the order they are shown."""
        every = self._store.items(self._run.id, status=["generated", "judged", "skipped"])
        return [i for i in presentation_order(every) if i.status == "generated"]

    def run(self) -> SessionOutcome:
        pending = self.pending()
        outcome = SessionOutcome(remaining=len(pending))
        total = len(pending)
        for position, item in enumerate(pending, start=1):
            self._show(item, position, total, outcome)
            key = ask(
                self._keys,
                self._console,
                "哪条是她？  [1] 左边  [2] 右边  [s] 跳过  [q] 保存退出",
                KEYS_BLIND,
            )
            if key is None or key == KEY_QUIT:
                outcome.quit = True
                return outcome
            outcome.remaining -= 1
            if key == KEY_SKIP:
                self._store.judge(item.id, None, payload={"chosen": "skip"})
                outcome.skipped += 1
                continue
            right = correct_choice(item, key)
            self._store.judge(
                item.id,
                "correct" if right else "wrong",
                score=1.0 if right else 0.0,
                payload={"chosen": "left" if key == KEY_LEFT else "right"},
            )
            outcome.judged += 1
        return outcome

    def _show(self, item: ItemView, position: int, total: int, outcome: SessionOutcome) -> None:
        left, right = sides_of(item)
        turns: list[Text] = []
        for turn in item.payload["shown"]:
            who = USER_LABEL if turn["who"] == "me" else HER_LABEL
            text = render_turn(who, turn["lines"])
            if text:
                turns.append(Text(text))
        table = Table(show_header=True, header_style="bold", expand=True, show_lines=True)
        table.add_column("1（左）", ratio=1)
        table.add_column("2（右）", ratio=1)
        table.add_row(
            Text(render_candidate(left, self._label)), Text(render_candidate(right, self._label))
        )
        title = f"第 {position}/{total} 对  已判 {outcome.judged}  跳过 {outcome.skipped}"
        self._console.print(Panel(Group(*turns), title="最近的对话"))
        self._console.print(Text(title, style="cyan"))
        self._console.print(table)
