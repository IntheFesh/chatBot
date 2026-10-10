"""Example windows: her reply blocks with the conversation before them (R-RET-001, R-RET-003,
R-RET-004, R-STO-007)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from tests.support.embedding import H, Msg, U, day, write_dialogue
from tests.support.synth_chat import ChatSpec, build_chat
from twin.profile.holdout import get_holdout, holdout_cutoff
from twin.retrieval.records import (
    MessageData,
    NotHerMessageError,
    load_messages,
    require_her_message,
    require_message,
)
from twin.retrieval.windows import (
    WindowAssembler,
    WindowRecord,
    apply_holdout,
    assemble,
    build_window,
    sync_windows,
    window_from_ids,
    window_id_of,
)
from twin.services import Services
from twin.storage.chat_models import Message
from twin.storage.retrieval_models import ExampleWindow


@pytest.fixture
def utc(services: Services) -> Services:
    """Services whose source zone is UTC, so local clock time equals the stored UTC time."""
    services.settings.time.source_timezone = "UTC"
    return services


def windows(services: Services) -> list[WindowRecord]:
    with services.db.session() as session:
        rows = session.scalars(select(ExampleWindow).order_by(ExampleWindow.reply_at_utc))
        return [WindowRecord.from_row(row) for row in rows]


def texts_by_id(services: Services) -> dict[str, str | None]:
    with services.db.session() as session:
        return {m.id: m.text for m in session.scalars(select(Message))}


def readable(services: Services, window: WindowRecord) -> tuple[list[str], list[list[str]]]:
    names = texts_by_id(services)
    reply = [str(names[i]) for i in window.reply_block_ids]
    context = [[str(names[i]) for i in turn] for turn in window.context_block_ids]
    return reply, context


def alternating(count: int, first: str = "u") -> list[Msg]:
    """``count`` single-message turns, alternating speakers, texts ``t0``, ``t1``, ..."""
    out: list[Msg] = []
    who = first
    for index in range(count):
        out.append(Msg(who, f"t{index}"))
        who = "h" if who == "u" else "u"
    return out


# ----------------------------------------------------------------- R-RET-001


def test_the_context_is_the_turns_before_the_reply_in_the_same_segment(utc: Services) -> None:
    write_dialogue(
        utc,
        [
            (day(0), [U("早"), U("在吗"), H("在"), H("怎么啦"), U("没事")]),
            (day(1), alternating(10)),  # u,h,u,h,...: her last turn is the 10th message
            (day(2), [H("我先开口")]),
            (day(2, 15), [U("晚上好"), H("晚上好呀")]),
        ],
    )
    sync_windows(utc)
    found = windows(utc)
    by_reply = {tuple(readable(utc, w)[0]): w for w in found}

    first = by_reply[("在", "怎么啦")]  # a burst of two messages is one reply block
    assert readable(utc, first)[1] == [["早", "在吗"]]
    assert first.context_turns == 1 and first.reply_reproducible == 2

    capped = by_reply[("t9",)]  # t0..t8 precede it; only the last six turns are context
    assert readable(utc, capped)[1] == [["t3"], ["t4"], ["t5"], ["t6"], ["t7"], ["t8"]]
    assert capped.context_turns == 6

    opens_segment = by_reply[("我先开口",)]  # silence before it: no context from the day before
    assert opens_segment.context_block_ids == () and opens_segment.context_turns == 0

    after_gap = by_reply[("晚上好呀",)]
    assert readable(utc, after_gap)[1] == [["晚上好"]]


def test_a_new_segment_cuts_the_context_even_inside_one_day(utc: Services) -> None:
    write_dialogue(
        utc,
        [
            (day(0, 9), [U("上午的事"), H("嗯")]),
            (day(0, 11, 30), [U("下午的事"), H("好")]),  # 2.5 h later: another segment
        ],
    )
    sync_windows(utc)
    found = {tuple(readable(utc, w)[0]): readable(utc, w)[1] for w in windows(utc)}
    assert found[("好",)] == [["下午的事"]]
    assert found[("嗯",)] == [["上午的事"]]


def test_context_turns_follow_the_configuration(utc: Services) -> None:
    utc.settings.retrieval.context_turns = 2
    write_dialogue(utc, [(day(0), alternating(8))])
    sync_windows(utc)
    last = windows(utc)[-1]
    assert readable(utc, last)[1] == [["t5"], ["t6"]]


def test_every_burst_of_hers_is_a_window_and_event_only_ones_are_marked(utc: Services) -> None:
    write_dialogue(
        utc,
        [
            (day(0), [U("看这个"), H(None, kind="image"), H("好看吧")]),
            (day(1), [U("打电话"), H(None, kind="call")]),
            (day(2), [U("睡了吗"), H("睡了", kind="quote")]),
        ],
    )
    sync_windows(utc)
    found = windows(utc)
    assert [w.reply_reproducible for w in found] == [1, 0, 1]
    assert [len(w.reply_block_ids) for w in found] == [2, 1, 1]  # the image joins its burst


def test_system_notices_are_not_messages_of_either_side(utc: Services) -> None:
    write_dialogue(
        utc,
        [
            (day(0), [U("你好"), H("嗯"), Msg("u", "你撤回了一条消息", "system"), H("再说一句")]),
            (day(1), [U("明天见"), H("明天见")]),
        ],
    )
    sync_windows(utc)
    only = windows(utc)
    # the notice (80 s between her two messages, within the burst gap) does not split the burst
    assert [readable(utc, w)[0] for w in only] == [["嗯", "再说一句"], ["明天见"]]


def test_window_ids_are_stable_and_derived_from_the_first_reply_message(utc: Services) -> None:
    write_dialogue(utc, [(day(0), [U("a"), H("b"), H("c")]), (day(1), [U("d"), H("e")])])
    sync_windows(utc)
    with utc.db.session() as session:
        first_id = next(m.id for m in session.scalars(select(Message)) if m.text == "b")
    window = windows(utc)[0]
    assert window.id == window_id_of(first_id)
    assert len(window.id) <= 24
    ids = [w.id for w in windows(utc)]
    sync_windows(utc)
    assert [w.id for w in windows(utc)] == ids


def test_local_slot_and_day_type_follow_the_source_clock(services: Services) -> None:
    # 2026-03-01 is a Sunday; 22:30 UTC is 16:30 in Chicago (CST) and 06:30 the next day in Shanghai
    services.settings.time.source_timezone = "America/Chicago"
    write_dialogue(
        services,
        [(day(0, 22, 30), [U("在吗"), H("在")]), (day(1, 18, 0), [U("上班了吗"), H("上了")])],
    )
    sync_windows(services)
    sunday, monday = windows(services)
    assert (sunday.local_slot, sunday.day_type) == (66, "weekend")  # 16:30 -> slot 66
    assert (monday.local_slot, monday.day_type) == (48, "workday")  # 12:00 Monday


# ----------------------------------------------------------------- R-RET-003


def test_windows_and_the_holdout_agree_on_what_a_reply_block_is(services: Services) -> None:
    build_chat(services, ChatSpec(days=21))
    report = sync_windows(services)
    holdout = get_holdout(services)
    assert holdout is not None
    found = windows(services)
    reproducible = [w for w in found if w.reply_reproducible > 0]
    assert len(reproducible) == holdout.blocks  # the same bursts the cutoff was counted on
    held = [w for w in reproducible if w.holdout]
    assert len(held) == holdout.held_out_blocks
    assert min(w.reply_at_utc for w in held) == holdout.cutoff
    assert all(not w.holdout for w in found if w.reply_at_utc < holdout.cutoff)
    assert all(w.holdout for w in found if w.reply_at_utc >= holdout.cutoff)
    assert report.holdout == sum(1 for w in found if w.holdout)
    assert report.cutoff == holdout_cutoff(services)


def test_the_cutoff_does_not_move_when_windows_are_synced_again(services: Services) -> None:
    build_chat(services, ChatSpec(days=14))
    sync_windows(services)
    stored = holdout_cutoff(services)
    second = sync_windows(services)
    assert second.cutoff == stored and second.added == 0 and second.changed == 0


# ----------------------------------------------------------------- R-RET-006


def test_syncing_again_changes_nothing_and_new_messages_add_only_new_windows(
    utc: Services,
) -> None:
    write_dialogue(utc, [(day(n), [U(f"问{n}"), H(f"答{n}")]) for n in range(12)])
    first = sync_windows(utc)
    assert (first.total, first.added, first.removed) == (12, 12, 0)
    before = {w.id: w.signature for w in windows(utc)}

    again = sync_windows(utc)
    assert (again.added, again.changed, again.removed, again.unchanged) == (0, 0, 0, 12)

    write_dialogue(utc, [(day(20), [U("新问"), H("新答")])], append=True)
    third = sync_windows(utc)
    assert (third.added, third.changed, third.total) == (1, 0, 13)
    after = {w.id: w.signature for w in windows(utc)}
    assert {k: v for k, v in after.items() if k in before} == before


def test_a_message_imported_later_changes_only_the_windows_it_belongs_to(utc: Services) -> None:
    write_dialogue(
        utc,
        [(day(0), [U("一"), H("二"), U("三"), H("四")]), (day(3), [U("远"), H("方")])],
    )
    sync_windows(utc)
    before = {tuple(readable(utc, w)[0]): w.signature for w in windows(utc)}
    # an older export brings a message of the first conversation that was missing
    write_dialogue(utc, [(day(0, 12, 0), [Msg("u", "补", gap=None)])], append=True)
    report = sync_windows(utc)
    after = {tuple(readable(utc, w)[0]): w.signature for w in windows(utc)}
    assert report.changed >= 1
    assert after[("方",)] == before[("方",)]  # the distant window is untouched


def test_windows_of_vanished_messages_are_removed(utc: Services) -> None:
    write_dialogue(utc, [(day(0), [U("a"), H("b")]), (day(1), [U("c"), H("d")])])
    sync_windows(utc)
    with utc.db.transaction(bump_state=False) as session:
        gone = [m.id for m in session.scalars(select(Message)) if m.text in ("c", "d")]
        session.execute(delete(Message).where(Message.id.in_(gone)))
    report = sync_windows(utc)
    assert report.removed == 1 and len(report.removed_ids) == 1
    assert [readable(utc, w)[0] for w in windows(utc)] == [["b"]]


def test_apply_holdout_moves_flags_without_reading_messages(utc: Services) -> None:
    write_dialogue(utc, [(day(n), [U(f"问{n}"), H(f"答{n}")]) for n in range(10)])
    sync_windows(utc)
    found = windows(utc)
    new_cutoff = found[6].reply_at_utc
    entering, released = apply_holdout(utc, new_cutoff)
    assert released == 0 and len(entering) == len(
        [w for w in found if not w.holdout and w.reply_at_utc >= new_cutoff]
    )
    flags = {w.id: w.holdout for w in windows(utc)}
    assert all(flags[w.id] == (w.reply_at_utc >= new_cutoff) for w in found)
    later = found[9].reply_at_utc
    entering, released = apply_holdout(utc, later)
    assert entering == () and released == 3  # windows 6..8 are released again


# ----------------------------------------------------------------- R-RET-004


class BotTurnLike:
    """What a row of the bot's own turns (round 09) would look like: not a ``messages`` row."""

    def __init__(self) -> None:
        self.id = "bot-1"
        self.is_sent = False
        self.kind = "text"
        self.text = "我是机器人说的话"
        self.conversation_id = "c"
        self.create_time_utc = datetime(2026, 3, 1, tzinfo=UTC)


def message(services: Services, ident: str, *, sent: bool, text: str = "x") -> Message:
    return Message(
        id=ident,
        conversation_id="c",
        create_time_utc=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        is_sent=sent,
        kind="text",
        text=text,
        raw={},
        source_export_id="e",
    )


def test_a_bot_reply_cannot_enter_the_library(services: Services) -> None:
    real = message(services, "m1", sent=False)
    with pytest.raises(TypeError, match="messages table"):
        build_window([BotTurnLike()], [])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="messages table"):
        build_window([real], [[BotTurnLike()]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="messages table"):
        WindowAssembler(120, 3600, 6).add(BotTurnLike())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        require_message(BotTurnLike())
    with pytest.raises(TypeError):
        MessageData.from_row(BotTurnLike())


def test_the_user_cannot_be_the_reply_but_may_be_the_context(services: Services) -> None:
    hers = message(services, "m1", sent=False, text="她说")
    his = message(services, "m2", sent=True, text="他说")
    with pytest.raises(NotHerMessageError):
        build_window([his], [])
    with pytest.raises(NotHerMessageError):
        build_window([hers, his], [])
    with pytest.raises(NotHerMessageError):
        require_her_message(his)
    window = build_window([hers], [[his]])
    assert window.reply_ids == ("m1",) and window.context_ids == (("m2",),)
    assert window.reproducible == 1
    with pytest.raises(ValueError, match="at least one reply"):
        build_window([], [[his]])


def test_windows_by_id_look_the_ids_up_in_the_messages_table(utc: Services) -> None:
    write_dialogue(utc, [(day(0), [U("问"), H("答")])])
    with utc.db.session() as session:
        rows = {m.text: m.id for m in session.scalars(select(Message))}
        window = window_from_ids(session, [rows["答"]], [[rows["问"]]])
        assert window.id == window_id_of(rows["答"])
        with pytest.raises(NotHerMessageError):
            window_from_ids(session, [rows["问"]], [])
        with pytest.raises(LookupError, match="not rows of the messages table"):
            window_from_ids(session, ["bot-turn-9"], [[rows["问"]]])
        assert set(load_messages(session, [rows["问"], "bot-turn-9"])) == {rows["问"]}


def test_nothing_the_bot_says_can_come_out_of_the_library(utc: Services) -> None:
    """The library reads ``messages`` only; whatever is stored in it is a real export row."""
    write_dialogue(utc, [(day(0), [U("你好"), H("你好呀")]), (day(1), [U("早"), H("早呀")])])
    sync_windows(utc)
    columns = set(ExampleWindow.__table__.columns.keys())  # type: ignore[attr-defined]
    assert not {"text", "content", "body", "bot_turn_id"} & columns
    record = WindowRecord.from_row(windows(utc)[0])
    assert record.reply_block_ids and all(isinstance(i, str) for i in record.reply_block_ids)


def test_assemble_streams_windows_in_time_order(utc: Services) -> None:
    write_dialogue(utc, [(day(n), [U(f"q{n}"), H(f"a{n}")]) for n in range(4)])
    from twin.ingest.corpus import conversation_skeleton

    with utc.db.session() as session:
        drafts = list(assemble(session.scalars(conversation_skeleton()), 120, 3600, 6))
    assert [d.reply_at for d in drafts] == sorted(d.reply_at for d in drafts)
    assert len(drafts) == 4 and all(len(d.context_ids) == 1 for d in drafts)


def test_the_library_reads_messages_only_through_the_corpus_statements() -> None:
    """No ``select(Message...)`` in the retrieval package, and nothing of the bot's turns."""
    root = Path(__file__).resolve().parents[2] / "src" / "twin" / "retrieval"
    offenders = []
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "select":
                names = {
                    n.id for arg in node.args for n in ast.walk(arg) if isinstance(n, ast.Name)
                }
                if "Message" in names:
                    offenders.append(f"{path.name}:{node.lineno}")
        assert "bot_turns" not in source and "BotTurn" not in source, path.name
    assert offenders == []
