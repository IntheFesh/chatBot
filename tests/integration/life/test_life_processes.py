"""Other processes of the machine work on the same files while the application runs (R-ARCH-006).

``twin run`` is one process; ``twin persona edit``, ``twin timezone set``, ``twin import`` and the
rest of the command line are others, on the same database.  The cooperation is the one of
R-ARCH-006: a READ command only looks, a LIGHT command writes and counts up ``state_version``, a
HEAVY command only queues the work, and an EXCLUSIVE command is refused while the application or
its supervisor holds its lock.  Here the command line really runs (``LifeWorld.cli``: its own
container and database connection, the machine's wall clock), and the application of the world
reacts the way the real one does:

* **the persona card** edited in an editor is in the very next reply, and the state watcher has
  seen the change within its two seconds;
* **the time zone** switched on the command line is noticed within two seconds: the plan is the
  one of the new zone, what was planned is void, ``/状态`` says the new zone; and the switch the
  user makes in the chat is what the command line shows;
* **an import** queued by the command line (and one started by ``/导入``) is executed by the
  application's job worker, in batches, with its progress in ``twin import status`` and in
  ``/状态``, and her real messages grow while nothing of the conversation with the bot gets in;
* **an exclusive command** (``twin purge --all``) is refused while either lock is held, before
  anything is asked or deleted, and the conversation goes on.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import func, select

from tests.fixtures.synth_export import SynthExport
from tests.integration.life.conftest import WorldFactory
from tests.support.embedding import HashingBackend
from tests.support.ingest import make_export
from tests.support.life_checks import (
    assert_bot_text_not_in_her_data,
    assert_bot_text_stays_out,
    assert_clean_screen,
    assert_never_in_deep_sleep,
    assert_screen_matches_records,
    snapshot_isolation,
)
from tests.support.life_world import LifeWorld
from tests.support.proactive_world import opening_curve, proactive_model
from twin.config.runtime import BOT_TIMEZONE
from twin.ingest.runs import get_run
from twin.ops.instance_lock import LOCK_RUN, LOCK_SUPERVISOR, InstanceLock
from twin.ops.process_model import ExitCode
from twin.profile.persona import compose
from twin.profile.persona import edit as persona_edit
from twin.profile.persona.store import PersonaStore
from twin.services import Services
from twin.storage.chat_models import Message
from twin.storage.state import read_state_version

pytestmark = pytest.mark.integration

FRIDAY = date(2026, 10, 9)
CHICAGO, SHANGHAI = "America/Chicago", "Asia/Shanghai"
SOON = 2.0  # the state watcher looks every two seconds (R-ARCH-006)
WIFE = "synthetic-her"


def state_version(services: Services) -> int:
    with services.db.session() as session:
        return read_state_version(session)


async def noticed_within(world: LifeWorld, version: int, limit_s: float = SOON) -> float:
    """Seconds until the application's state watcher has seen ``version`` (at most ``limit_s``)."""
    started = world.now
    while world.assembly.watcher.version != version:
        waited = (world.now - started).total_seconds()
        assert waited < limit_s, f"after {waited:.2f} s the application still has not noticed"
        await world.clock.step(0.25)
    return (world.now - started).total_seconds()


def replacing(old: str, new: str) -> Callable[[Path], None]:
    """An editor that changes ``old`` into ``new`` in the file it is given, and closes."""

    def editor(path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        assert old in text, "the card in the editor is not the card of the story"
        path.write_text(text.replace(old, new), encoding="utf-8")

    return editor


def write_card(world: LifeWorld, address: str) -> None:
    """Her card in force, the way ``twin persona regenerate`` leaves it (no model involved)."""
    text = (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block("### 风格\n- 口头禅：笑死我了\n\n### 基本情况\n- 事实：养猫\n\n")
        + compose.manual_block([f"- 称呼：{address}"], ["- 她养了一只猫"])
        + compose.dont_block(["不要用句号"])
    )
    PersonaStore(world.services.db, world.services.clock).add_version(
        "live", text, reason="generate", described_her_messages=100
    )


# ------------------------------------------------------------------------- the persona card


async def test_a_persona_edit_in_another_process_is_in_the_next_reply_within_two_seconds(
    make_world: WorldFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    world.deepseek.book.when("在吗", "在呀")
    world.deepseek.book.when("早点睡", "知道啦")
    write_card(world, "宝宝")
    await world.clock.step(SOON)  # the application has seen the card
    await world.say("在吗")
    await world.run_until_idle()
    first = world.deepseek.of_kind("reply")[-1].text
    assert "称呼：宝宝" in first and "称呼：小笨蛋" not in first

    # ---- `twin persona edit` in a terminal of its own -----------------------------------------
    monkeypatch.setattr(
        persona_edit, "make_editor", lambda: replacing("称呼：宝宝", "称呼：小笨蛋")
    )
    before = state_version(world.services)
    code, out = world.cli("persona", "edit")
    assert code == 0 and "saved as version v2" in out, out
    after = state_version(world.services)
    assert after == before + 1  # a LIGHT command counts the change up, once, in its transaction
    assert world.assembly.watcher.version == before  # the application has not looked yet
    waited = await noticed_within(world, after)
    assert 0 < waited <= SOON  # and now it has: within the two seconds of the specification

    # ---- the next reply is written with the card as it is now ---------------------------------
    await world.say("早点睡")
    await world.run_until_idle()
    second = world.deepseek.of_kind("reply")[-1].text
    assert "称呼：小笨蛋" in second and "称呼：宝宝" not in second
    assert world.persona_said[-1].text == "知道啦"
    card = PersonaStore(world.services.db, world.services.clock).active("live")
    assert card is not None and card.reason == "edit" and card.number == 2
    # the edit is her card's, not a message of anybody: the file of the editor is gone
    assert list(world.services.paths.tmp_dir.glob("persona-*")) == []
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


# --------------------------------------------------------------------------- the time zone


async def test_a_zone_switched_in_another_process_is_noticed_and_the_chat_switch_is_seen_there(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago
        model=proactive_model(opening_curve(base=0.02, peaks={48: 0.6, 49: 0.6, 73: 0.6})),
    )
    await world.say("我下午有课")
    await world.run_until_idle()
    chicago_plan = world.plan()
    assert chicago_plan.timezone == CHICAGO
    planned = list(world.assembly.proactive.scheduler.candidates.pending())
    assert planned, "nothing was planned: the switch has nothing to void"

    # ---- `twin timezone set` in a terminal of its own ------------------------------------------
    version = state_version(world.services)
    switch_at = world.now
    code, out = world.cli("timezone", "set", SHANGHAI)
    assert code == 0 and f"switched {CHICAGO} -> {SHANGHAI}" in out, out
    assert world.services.runtime.get(BOT_TIMEZONE) == SHANGHAI
    assert state_version(world.services) > version
    await noticed_within(world, state_version(world.services))
    await world.settle()
    plan = world.plan(date(2026, 10, 10))  # one o'clock in the night in Shanghai: the next day
    assert plan.timezone == SHANGHAI and plan.reason == "timezone_switch"
    assert plan.effective_from == switch_at  # made by the other process, from that moment
    voided = [
        r for r in world.proactive_rows(outcomes=["expired"]) if r.reason == "timezone_switch"
    ]
    assert voided and all(r.at >= switch_at for r in voided)  # the application voided the old ones
    assert not list(world.assembly.proactive.scheduler.candidates.pending())  # she is asleep now
    await world.say("/状态")
    assert f"时区：{SHANGHAI}" in world.system_said[-1].text

    # ---- `/时区` in the chat: what the command line shows is the same ---------------------------
    await world.say(f"/时区 {CHICAGO}")
    assert f"{SHANGHAI} 切到 {CHICAGO}" in world.system_said[-1].text
    code, shown = world.cli("timezone", "show")
    assert code == 0 and f"bot time zone: {CHICAGO}" in shown, shown
    code, history = world.cli("timezone", "history")
    assert code == 0 and "cli" in history and "command" in history, history
    code, listed = world.cli("settings", "list")
    assert code == 0 and CHICAGO in listed and SHANGHAI not in listed
    assert world.plan().timezone == CHICAGO and world.plan().reason == "timezone_switch"
    await world.run_for(minutes=30)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []


# ----------------------------------------------------------------------------- the import


def an_export(root: Path, *, seed: int, start: datetime, name: str) -> SynthExport:
    return make_export(
        root / name,
        target_messages=300,
        other_conversations=1,
        other_messages=10,
        include_group=False,
        media=False,
        seed=seed,
        start=start,
        export_id=f"life-export-{name}",
        target_username=WIFE,
    )


def serve_stickers(api: respx.MockRouter, *exports: SynthExport) -> None:
    """The sticker server of the export's links, which the post-import hook downloads from."""
    served = {info.url: info for export in exports for info in export.stickers.values()}

    def answer(request: httpx.Request) -> httpx.Response:
        found = served.get(str(request.url))
        if found is None:
            return httpx.Response(404)
        return httpx.Response(200, content=found.data, headers={"content-type": found.mime})

    api.get(host="stickers.example.test").mock(side_effect=answer)


def count_messages(services: Services) -> int:
    with services.db.session() as session:
        return int(session.scalar(select(func.count()).select_from(Message)) or 0)


def small_batches(services: Services) -> None:
    services.settings.ingest.batch_size = 50  # 300 messages: six commits, six moments to look


async def test_an_import_from_another_process_runs_in_the_application_with_visible_progress(
    make_world: WorldFactory,
    api: respx.MockRouter,
    tmp_path: Path,
    embedder: HashingBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twin.ingest import jobs as import_jobs
    from twin.ingest.cli import format_status
    from twin.ingest.importer import BatchEvent, ImportRunner

    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
        configure=small_batches,
    )
    world.deepseek.book.default = ("收到收到",)
    world.watch.add("commands/import_report.py")  # looks every 5 s whether an import has ended
    shots: list[tuple[int, int, str]] = []  # (batch, processed per the event, status as shown)

    class Watched(ImportRunner):
        """The production runner, which also writes down what `twin import status` would show."""

        def __init__(self, services: Services, **options: object) -> None:
            super().__init__(services, on_batch=self.look, **options)  # type: ignore[arg-type]

        def look(self, event: BatchEvent) -> None:
            view = get_run(self._services.db, event.run_id)
            assert view is not None
            shots.append((event.batch_number, view.processed, format_status(view)[0]))

    monkeypatch.setattr(import_jobs, "ImportRunner", Watched)
    first = an_export(tmp_path, seed=31, start=datetime(2026, 8, 1, tzinfo=UTC), name="one")
    serve_stickers(api, first)
    messages = count_messages(world.services)

    # ---- `twin import <dir>`: queued, and nothing happens in this process ------------------------
    code, out = world.cli("settings", "set", "target.username", WIFE)  # she is the one to imitate
    assert code == 0, out
    code, out = world.cli("import", str(first.root))
    assert code == 0 and "the running application executes it" in out, out
    assert count_messages(world.services) == messages and shots == []
    code, status = world.cli("import", "status")
    assert code == 0 and "queued" in status and "processed 0 / 300" in status, status
    await world.say("/状态")
    assert (
        "导入：任务" in world.system_said[-1].text and "已处理 0/300" in world.system_said[-1].text
    )

    # ---- the application's worker takes the job: six batches, each visible -----------------------
    await world.drain_jobs(approval_ok=True)
    assert [batch for batch, _, _ in shots] == [1, 2, 3, 4, 5, 6]
    assert [processed for _, processed, _ in shots] == [50, 100, 150, 200, 250, 300]
    assert all("running" in line for _, _, line in shots[:-1])
    assert count_messages(world.services) == messages + 300
    code, status = world.cli("import", "status")
    assert code == 0 and "done" in status and "processed 300 / 300" in status, status
    assert "post-import hooks" in status
    assert not [row for row in world.job_rows() if row[1] in ("failed", "dead")]
    await world.say("/状态")
    ended_status = world.system_said[-1].text
    assert "导入：任务" not in ended_status  # the import is over ...
    assert "个回放任务在排队或进行" in ended_status  # ... and the memory replay waits for him
    waiting = world.awaiting_approval()
    assert waiting and {kind for kind, *_ in waiting} >= {"memory_replay"}  # one-time costs
    assert all(attempts == 0 for _, _, attempts, _ in waiting)  # nothing is spent unapproved

    # ---- `/导入 <dir>` from the chat, the same files again: the same worker, and he is told ------
    await world.say(f"/导入 {first.root}")
    assert "已开始导入" in world.system_said[-1].text and "约 300 条" in world.system_said[-1].text
    await world.say("/状态")
    assert "导入：任务" in world.system_said[-1].text
    await world.drain_jobs(approval_ok=True)
    await world.run_for(seconds=30)  # the reporter looks every five seconds
    ended = [s.text for s in world.system_said if s.text.startswith("⚙️ 导入完成")]
    assert len(ended) == 1 and "新增 0 条，重复 300 条" in ended[0], ended  # counts, never a word
    assert count_messages(world.services) == messages + 300  # the same messages are not twice
    assert len(shots) == 12  # (six batches again)

    # ---- the new real messages are hers; nothing of the conversation with the bot got in ---------
    before = snapshot_isolation(world)
    await world.say("在吗")
    await world.run_until_idle()
    await world.drain_jobs(approval_ok=True)
    assert world.persona_said[-1].text == "收到收到"
    await assert_bot_text_stays_out(world, before, embedder)
    assert_clean_screen(world)
    assert_never_in_deep_sleep(world)
    assert_screen_matches_records(world)
    assert world.deepseek.unexpected == []


# ------------------------------------------------------------------------ exclusive commands


async def test_an_exclusive_command_is_refused_while_a_lock_is_held_and_nothing_is_deleted(
    make_world: WorldFactory,
) -> None:
    world = await make_world(
        datetime(2026, 10, 9, 16, 0, tzinfo=UTC),  # 11:00 in Chicago
        model=proactive_model(opening_curve(base=0.02)),
    )
    world.deepseek.book.when("在吗", "在呀")
    locks_dir = world.services.paths.locks_dir
    messages = count_messages(world.services)
    database = world.services.paths.db_path

    def refused_with(*args: str, named: str) -> None:
        code, out = world.cli(*args, answer="删除她的全部数据\n")
        assert code == ExitCode.BUSY, out
        assert "exclusive access" in out and f"'{named}'" in out, out
        assert "twin service stop" in out  # it says what to do
        assert database.is_file() and count_messages(world.services) == messages

    # `twin run` holds the "run" lock for as long as the application lives
    running = InstanceLock(LOCK_RUN, locks_dir=locks_dir)
    assert running.acquire()
    try:
        refused_with("purge", "--all", named=LOCK_RUN)
        refused_with("purge", "--training-only", named=LOCK_RUN)
        refused_with("db", "upgrade", named=LOCK_RUN)
        code, out = world.cli("run")  # and a second application is refused as well
        assert code == ExitCode.BUSY and "already running" in out, out
        code, out = world.cli("settings", "list")  # while the commands that cooperate are not
        assert code == 0 and "time.bot_timezone" in out
        await world.say("在吗")
        await world.run_until_idle()
        assert world.persona_said[-1].text == "在呀"  # the application did not notice a thing
    finally:
        running.release()

    # the supervisor (`twin supervise`) holds the other lock, and the same command is refused
    supervising = InstanceLock(LOCK_SUPERVISOR, locks_dir=locks_dir)
    assert supervising.acquire()
    try:
        refused_with("purge", "--all", named=LOCK_SUPERVISOR)
    finally:
        supervising.release()

    # nobody holds a lock: an exclusive command goes through
    code, out = world.cli("db", "upgrade")
    assert code == 0, out
    assert database.is_file() and count_messages(world.services) == messages
    assert_clean_screen(world)
    assert_screen_matches_records(world)
    assert_bot_text_not_in_her_data(world)
    assert world.deepseek.unexpected == []
