"""The terminal screens: the blind-test judging and the keys (R-EVAL-001)."""

from __future__ import annotations

import io
import os
import sys
import threading
import types
from datetime import UTC, datetime

import pytest
from rich.console import Console

from twin.eval.render import Candidate, CandidateLine
from twin.eval.store import EvalStore, ItemView, NewItem
from twin.eval.ui import (
    BlindSession,
    LineKeys,
    TerminalKeys,
    ask,
    correct_choice,
    sides_of,
)
from twin.services import Services

AT = datetime(2026, 9, 1, 12, tzinfo=UTC)
LABELS = {"a" * 32: "笑得很开心的猫"}


def label(md5: str) -> str | None:
    return LABELS.get(md5)


def console_on(buffer: io.StringIO) -> Console:
    return Console(file=buffer, width=100, color_system=None, force_terminal=False, highlight=False)


def text_candidate(*lines: str) -> Candidate:
    return Candidate(tuple(CandidateLine("text", text=line) for line in lines))


def make_run(
    services: Services, sides: list[bool], *, backend: str = "deepseek"
) -> tuple[EvalStore, str, list[str]]:
    """A run of generated pairs: pair ``n`` has her reply ``真话n`` and the bot's ``假话n``."""
    store = EvalStore(services.db, services.clock)
    run = store.create_run("blind", mode="holdout", backends=[backend], status="running")
    store.add_items(
        run.id,
        [
            NewItem(
                f"k{n}",
                backend,
                AT,
                {
                    "shown": [
                        {"who": "me", "lines": [f"在吗{n}", "[拥抱] [bold]在干嘛"]},
                        {"who": "her", "lines": ["在呀"]},
                        {"who": "me", "lines": ["出来玩"]},
                    ],
                    "real": text_candidate(f"真话{n}").to_json(),
                },
                None,
                "night",
                "short",
                left,
            )
            for n, left in enumerate(sides)
        ],
    )
    ids = [i.id for i in store.items(run.id)]
    for n, item_id in enumerate(ids):
        store.save_generated(item_id, {"bot": text_candidate(f"假话{n}").to_json()}, cost_usd=0.001)
    return store, run.id, ids


def session_for(store: EvalStore, run_id: str, keys: str) -> tuple[BlindSession, io.StringIO]:
    buffer = io.StringIO()
    session = BlindSession(
        store, store.get_run(run_id), label, console_on(buffer), LineKeys(io.StringIO(keys))
    )
    return session, buffer


def test_a_choice_is_right_when_it_points_at_her_real_reply() -> None:
    def item(left_is_bot: bool) -> ItemView:
        return ItemView(
            "i", "r", 0, "k", "deepseek", None, None, None, AT, "generated", left_is_bot,
            None, None, None, 0.0, None, {},
        )  # fmt: skip

    assert correct_choice(item(True), "2") and not correct_choice(item(True), "1")
    assert correct_choice(item(False), "1") and not correct_choice(item(False), "2")


def test_the_bot_goes_on_the_side_its_coin_says(services: Services) -> None:
    store, run_id, _ = make_run(services, [True, False])
    on_left, on_right = store.items(run_id)
    assert sides_of(on_left)[0] == text_candidate("假话0") and sides_of(on_left)[
        1
    ] == text_candidate("真话0")
    assert sides_of(on_right)[0] == text_candidate("真话1") and sides_of(on_right)[
        1
    ] == text_candidate("假话1")


def test_the_screen_shows_the_conversation_and_two_candidates_and_not_the_backend(
    services: Services,
) -> None:
    store, run_id, _ = make_run(services, [True], backend="deepseek")
    session, out = session_for(store, run_id, "q\n")
    outcome = session.run()
    shown = out.getvalue()
    assert outcome.quit and outcome.judged == 0 and outcome.remaining == 1
    assert "我：在吗0" in shown and "她：在呀" in shown and "出来玩" in shown
    assert "[拥抱] [bold]在干嘛" in shown  # brackets are text here, never markup
    assert "真话0" in shown and "假话0" in shown and "1（左）" in shown and "2（右）" in shown
    assert shown.index("假话0") < shown.index("真话0")  # the coin said: the bot is on the left
    assert "deepseek" not in shown and "backend" not in shown  # nothing says who wrote what
    assert "第 1/1 对" in shown


def test_every_decision_is_stored_at_once_and_a_quit_keeps_the_rest(services: Services) -> None:
    store, run_id, ids = make_run(services, [True, False, True, False])
    # pair 0: bot left -> "2" is right; pair 1: bot right -> "2" is wrong; pair 2: skipped
    session, _ = session_for(store, run_id, "2\n2\ns\nq\n")
    outcome = session.run()
    assert (outcome.judged, outcome.skipped, outcome.quit) == (2, 1, True)
    first, second, third, fourth = (store.item(i) for i in ids)
    assert (first.status, first.outcome, first.score) == ("judged", "correct", 1.0)
    assert (second.status, second.outcome, second.score) == ("judged", "wrong", 0.0)
    assert (third.status, third.outcome) == ("skipped", None) and not third.valid
    assert fourth.status == "generated" and fourth.outcome is None  # not decided: left for later
    assert first.payload["chosen"] == "right" and third.payload["chosen"] == "skip"
    assert first.judged_at is not None
    # continuing asks only for what is left
    again, out = session_for(store, run_id, "1\n")
    assert [i.id for i in again.pending()] == [ids[3]]
    resumed = again.run()
    assert (resumed.judged, resumed.quit) == (1, False)
    assert store.item(ids[3]).outcome == "correct"  # bot right, "1" is her
    assert "第 1/1 对" in out.getvalue()
    assert again.pending() == []


def test_an_unknown_key_is_asked_again_and_the_end_of_the_input_leaves_the_rest(
    services: Services,
) -> None:
    store, run_id, ids = make_run(services, [True, True])
    session, out = session_for(store, run_id, "x\n\n3\n1\n")  # three wrong keys, then an answer
    outcome = session.run()
    assert outcome.judged == 1 and outcome.quit  # the input ended before the second pair
    assert out.getvalue().count("没有这个选项") == 3
    assert store.item(ids[0]).outcome == "wrong"  # bot is left: "1" picks the bot
    assert store.item(ids[1]).status == "generated"


def test_failed_and_ungenerated_pairs_are_not_shown(services: Services) -> None:
    store, run_id, ids = make_run(services, [True, True, True])
    store.mark_failed(ids[1], "no_reply")
    session, _ = session_for(store, run_id, "")
    assert [i.id for i in session.pending()] == [ids[0], ids[2]]


def test_ask_names_the_choices_and_returns_none_at_the_end_of_the_input() -> None:
    out = io.StringIO()
    assert ask(LineKeys(io.StringIO("")), console_on(out), "问？", ("a",)) is None
    assert ask(LineKeys(io.StringIO("B\na\n")), console_on(out), "问？", ("a", "")) == "a"
    assert ask(LineKeys(io.StringIO("\n")), console_on(out), "问？", ("a", "")) == ""


def test_line_keys_take_the_first_letter_in_lower_case() -> None:
    keys = LineKeys(io.StringIO("Quit\n  2  \n\n"))
    assert [keys.read_key(), keys.read_key(), keys.read_key(), keys.read_key()] == [
        "q",
        "2",
        "",
        None,
    ]


def test_without_a_terminal_the_keys_are_lines() -> None:
    keys = TerminalKeys(io.StringIO("s\n1\n"))
    assert [keys.read_key(), keys.read_key(), keys.read_key()] == ["s", "1", None]


class FakeTerminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_on_windows_a_key_press_is_read_with_msvcrt(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed = iter(["2", "\r", "S", "\x03"])
    module = types.ModuleType("msvcrt")
    module.getwch = lambda: next(pressed)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "msvcrt", module)
    monkeypatch.setattr(sys, "platform", "win32")
    keys = TerminalKeys(FakeTerminal())
    assert [keys.read_key() for _ in range(4)] == ["2", "", "s", "q"]  # Ctrl+C leaves


def read_key_within(stream: io.TextIOBase, seconds: float = 10.0) -> str | None:
    """``TerminalKeys.read_key`` on a worker thread, so that a test never hangs on a key."""
    found: list[str | None] = []
    worker = threading.Thread(
        target=lambda: found.append(TerminalKeys(stream).read_key()), daemon=True
    )
    worker.start()
    worker.join(seconds)
    assert found, "no key was read"
    return found[0]


@pytest.mark.skipif(sys.platform == "win32", reason="a pseudo-terminal exists on POSIX only")
def test_on_posix_a_key_press_needs_no_enter() -> None:
    import pty

    master, slave = pty.openpty()
    stream = os.fdopen(slave, "r", closefd=False)
    try:
        for typed, expected in ((b"2", "2"), (b"S", "s"), (b"\r", "")):  # bare keys, no newline
            os.write(master, typed)
            assert read_key_within(stream) == expected
    finally:
        os.close(slave)
        os.close(master)
