"""The world of the end-to-end scenarios: the whole application, a user, a made-up DeepSeek.

``tests/integration/life`` runs the application of ``twin run`` - **assembled by the same function**
(:func:`twin.assembly.assemble`) - on the terminal channel, a manual clock and a database that is
the migrated test database.  Everything the production code reaches out to is a test double:

* **the user** types into a :class:`~tests.support.console.ScriptedInput` and reads a
  :class:`TimedOutput` that stamps every line with the clock;
* **DeepSeek** is :class:`LifeDeepSeek`, a ``respx`` side effect that answers by the *content* of
  the request (which template the system message is made from, what the user said) with fixed,
  deterministic results; it composes the scripted models of the memory, the life line and the
  proactive planner that the unit tests use, and adds the ones the engine needs (the reply, the
  crisis judgement, a picture's description);
* **time** is a :class:`~tests.support.life_clock.LifeClock`;
* **her routine** is a synthetic student (sleeps 23:30 to 07:30, busy 13:00 to 17:00 on workdays),
  stored as the active activity model, so the plan, the greeting and the proactive curve are the
  production ones, computed from it.

Nothing here is imported by ``src``.  The helpers read like the story of a scenario: ``await
world.say(...)``, ``await world.run_until(...)``, ``world.said``, ``world.proactive_rows()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx
import respx
from sqlalchemy import select

from tests.fixtures.synth_export import make_image_bytes
from tests.support.console import ScriptedInput
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.life_clock import HOUSEKEEPING, LifeClock
from tests.support.lifeline import ScriptedLifelineModel
from tests.support.memory import ScriptedMemoryModel
from tests.support.persona import attach_files, sync_counters
from tests.support.proactive_world import ProactiveScript, proactive_model
from tests.support.synth_chat import ChatSpec, MessageWriter, build_chat
from tests.support.waiting import wait_until
from twin.assembly import Assembly, assemble
from twin.channel.base import MediaRef, MessageKind
from twin.channel.local import TYPING_TEXT, LocalConsoleChannel
from twin.commands import texts
from twin.config.runtime import BOT_TIMEZONE
from twin.engine.roundstate import RoundData
from twin.engine.turns import BotTurnStore
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.profile.activity_model import ActivityModel
from twin.profile.builder import rebuild
from twin.profile.store import ACTIVITY_ACTIVE, VersionStore
from twin.retrieval.indexer import run_index
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.store import LogEntry, ProactiveLogStore
from twin.schedule.service import schedule_kit
from twin.schedule.store import SALT_KEY
from twin.services import Services
from twin.stickers.catalog import StickerCatalog
from twin.storage.engine_models import BotTurn
from twin.storage.media import MediaKind
from twin.storage.models import Alert, Job
from twin.storage.profile_models import ActivityModelVersion
from twin.storage.settings_store import put_setting

PREFIX = texts.PREFIX
CONTINUATION = " " * 5  # how the terminal indents the further lines of a message
USER_WORDS = re.compile(r"对方这一轮说的话：\s*(.*)\Z", re.DOTALL)
CHICAGO = "America/Chicago"
PLAN_SALT = "life-world"


# ------------------------------------------------------------------------------ the screen


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


# ------------------------------------------------------------------------------ DeepSeek


@dataclass
class Call:
    """One request DeepSeek was asked, as the world saw it."""

    kind: str
    at: datetime
    body: dict[str, Any]

    @property
    def last_user(self) -> str:
        return str(self.body["messages"][-1]["content"])

    @property
    def text(self) -> str:
        """Everything that was sent, as one string (for 'did X reach the model')."""
        return json.dumps(self.body["messages"], ensure_ascii=False)


@dataclass
class ReplyRule:
    needle: str
    lines: tuple[str, ...]
    in_context: bool = False  # look in the whole last message (memory block ...), not the words


@dataclass
class ReplyBook:
    """What the reply model says: the next queued answer, else the first rule that fits."""

    rules: list[ReplyRule] = field(default_factory=list)
    queue: deque[str] = field(default_factory=deque)
    default: tuple[str, ...] = ("嗯嗯",)

    def when(self, needle: str, *lines: str, in_context: bool = False) -> None:
        self.rules.append(ReplyRule(needle, lines, in_context))

    def answer(self, words: str, whole: str) -> str:
        if self.queue:
            return self.queue.popleft()
        for rule in self.rules:
            if rule.needle in (whole if rule.in_context else words):
                return "\n".join(rule.lines)
        return "\n".join(self.default)


def _markers() -> dict[str, str]:
    """The first words of the system message of every template, to tell the requests apart."""
    from twin.profile.prompt_templates import newest_file_template, template_files

    return {
        name: newest_file_template(name).system.strip()[:24]
        for name in template_files()
        if name != "reply_rules"
    } | {"reply_rules": newest_file_template("reply_rules").system.strip()[:24]}


KIND_OF_TEMPLATE = {
    "reply_rules": "reply",
    "reply_plan": "reply_plan",
    "proactive_plan": "proactive",
    "crisis_check": "crisis",
    "memory_extract": "memory",
    "memory_extract_bot": "memory",
    "memory_conflict": "memory",
    "memory_summary": "memory",
    "memory_summary_merge": "memory",
    "lifeline_generate": "lifeline",
    "lifeline_check": "lifeline",
    "correction_check": "correction",
    "sticker_tag": "sticker_tag",
    "sticker_context": "sticker_tag",
}


class LifeDeepSeek:
    """A DeepSeek that is made up: deterministic answers chosen by what the request is about."""

    def __init__(self, clock: LifeClock) -> None:
        self.clock = clock
        self.memory = ScriptedMemoryModel()
        self.lifeline = ScriptedLifelineModel()
        self.proactive = ProactiveScript()
        self.book = ReplyBook()
        self.caption = "一张桌子上放着一杯咖啡的照片"
        self.sticker_tags = ["开心"]
        self.sticker_description = "一只笑着的小猫"
        self.calls: Counter[str] = Counter()
        self.log: list[Call] = []
        self.unexpected: list[str] = []
        # failures to hand out before the model works again: kind -> HTTP statuses, in order
        self.failures: dict[str, deque[int]] = {}
        self.prompt_tokens = 400
        self.cache_hit_tokens = 300
        self.completion_tokens = 12
        self._markers = _markers()

    # ---- the knobs of a scenario -------------------------------------------------

    def fail(self, kind: str, status: int = 500, times: int = 1) -> None:
        self.failures.setdefault(kind, deque()).extend([status] * times)

    def bill(self, prompt_tokens: int, hit: int = 0, completion: int = 12) -> None:
        """What the following answers say they cost (a way to spend the budget in a scenario)."""
        self.prompt_tokens, self.cache_hit_tokens, self.completion_tokens = (
            prompt_tokens,
            hit,
            completion,
        )

    def of_kind(self, kind: str) -> list[Call]:
        return [call for call in self.log if call.kind == kind]

    # ---- the transport -------------------------------------------------------------

    def classify(self, body: dict[str, Any]) -> str:
        messages = body["messages"]
        for message in messages:
            content = message["content"]
            if isinstance(content, list) and any(
                part.get("type") == "image_url" for part in content
            ):
                system = str(messages[0]["content"]) if messages[0]["role"] == "system" else ""
                return (
                    "sticker_tag"
                    if self._markers["sticker_tag"] in system or "表情包" in system[:200]
                    else "caption"
                )
        system = str(messages[0]["content"])
        for name, marker in self._markers.items():
            if system.strip().startswith(marker):
                return KIND_OF_TEMPLATE.get(name, name)
        return "unknown"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        kind = self.classify(body)
        self.calls[kind] += 1
        self.log.append(Call(kind, self.clock.now_utc(), body))
        pending = self.failures.get(kind)
        if pending:
            return error(pending.popleft(), "scripted failure of the model")
        if kind == "reply":
            return self._reply(body)
        if kind == "reply_plan":
            return self._plain(
                json.dumps(
                    {
                        "reply": True,
                        "intent": "接住对方的话",
                        "facts_to_use": [],
                        "tone": "随意",
                        "bubble_hint": "一条短句",
                        "sticker_hint": "",
                    },
                    ensure_ascii=False,
                )
            )
        if kind == "proactive":
            return self.proactive(request)
        if kind == "memory":
            return self.memory(request)
        if kind == "lifeline":
            return self.lifeline(request)
        if kind == "crisis":
            return self._crisis(body)
        if kind == "correction":
            return self._plain('{"is_correction": false}')
        if kind == "caption":
            return self._plain(self.caption)
        if kind == "sticker_tag":
            seen = {
                "tags": self.sticker_tags,
                "description": self.sticker_description,
                "use_cases": "回应开心的事",
            }
            return self._plain(json.dumps(seen, ensure_ascii=False))
        self.unexpected.append(str(body["messages"][0]["content"])[:80])
        return error(400, "the made-up DeepSeek was asked something it does not know")

    def _plain(self, content: str) -> httpx.Response:
        return ok(
            content=content,
            prompt=self.prompt_tokens,
            hit=self.cache_hit_tokens,
            completion_tokens=self.completion_tokens,
        )

    def _reply(self, body: dict[str, Any]) -> httpx.Response:
        whole = str(body["messages"][-1]["content"])
        found = USER_WORDS.search(whole)
        words = found.group(1) if found else whole
        return self._plain(self.book.answer(words, whole))

    @staticmethod
    def _crisis_lines(body: dict[str, Any]) -> str:
        return str(body["messages"][-1]["content"])

    def _crisis(self, body: dict[str, Any]) -> httpx.Response:
        lines = self._crisis_lines(body)
        sure = any(word in lines for word in ("不想活", "结束生命", "自杀"))
        verdict = {"is_crisis": sure, "severity": "high" if sure else "none", "reason": "scripted"}
        return self._plain(json.dumps(verdict, ensure_ascii=False))


# --------------------------------------------------------------------------- the routine


def install_routine(services: Services, model: ActivityModel) -> str:
    """Store ``model`` as the active activity model of the live scope (what a rebuild would do)."""
    store = VersionStore(services.db, services.clock)
    parent = store.active_activity("live")
    profile = store.active_profile("live")
    now = services.clock.now_utc()
    digest = hashlib.sha256(json.dumps(model.to_json(), sort_keys=True).encode()).hexdigest()
    with services.db.transaction(bump_state=True) as session:
        row = ActivityModelVersion(
            scope="live",
            parent_id=parent.id if parent else None,
            profile_version_id=profile.id if profile else None,
            reason="life-world",
            input_hash=digest,
            her_messages=1000,
            data_range={},
            model=model.to_json(),
            diff=[],
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.flush()
        put_setting(
            session, ACTIVITY_ACTIVE + "live", row.id, clock=services.clock, by="life-world"
        )
        return row.id


# ------------------------------------------------------------------------------ the world


@dataclass
class LifeWorld:
    """A running application, a user at the terminal, and the doubles around them."""

    services: Services
    clock: LifeClock
    deepseek: LifeDeepSeek
    assembly: Assembly
    keyboard: ScriptedInput
    screen: TimedOutput
    channel: LocalConsoleChannel
    api: respx.MockRouter
    workdir: Path
    watch: set[str] = field(default_factory=set)
    started: bool = False
    pictures: int = 0

    # ---- building it ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        services: Services,
        clock: LifeClock,
        embedder: HashingBackend,
        api: respx.MockRouter,
        *,
        workdir: Path,
        start: datetime | None = None,
        model: ActivityModel | None = None,
        history_days: int = 21,
        seed: int = 7,
        window_h: float | None = None,
        quota: int | None = None,
        start_application: bool = True,
        configure: Callable[[Services], None] | None = None,
    ) -> LifeWorld:
        """Build the world and, unless told not to, start the application (see the module text)."""
        settings = services.settings
        settings.retrieval.model = embedder.info.model
        settings.channel.kind = "console"
        settings.pricing.offpeak_multiplier = 1.0  # a job may run at any hour
        if window_h is not None:
            settings.channel.proactive_window_safe_h = window_h
        if quota is not None:
            settings.channel.outbound_quota_safe = quota
        if configure is not None:
            configure(services)
        moment = start or clock.now_utc()
        first = (moment.astimezone(ZoneInfo(CHICAGO)).date()) - timedelta(days=history_days)
        build_chat(services, ChatSpec(start=first, days=history_days))
        rebuild(services, "all")  # her profile, the examples and the like, from the past
        run_index(services)
        install_routine(services, model or proactive_model())
        services.runtime.initialize()
        with services.db.transaction(bump_state=False) as session:
            put_setting(
                session, SALT_KEY, PLAN_SALT, clock=clock, by="life-world", record_history=False
            )
        services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
        clock.set_time(moment)
        clock.track_threads()
        deepseek = LifeDeepSeek(clock)
        api.post(API).mock(side_effect=deepseek)
        api.route(host="127.0.0.1").pass_through()  # a made-up llama-server is a real local one
        keyboard, screen = ScriptedInput(), TimedOutput(clock)
        assembly = assemble(
            services, console_input=keyboard, console_output=screen, rng=random.Random(seed)
        )
        channel = assembly.channel
        assert isinstance(channel, LocalConsoleChannel)
        world = cls(services, clock, deepseek, assembly, keyboard, screen, channel, api, workdir)
        if start_application:
            await world.start()
        return world

    async def start(self) -> None:
        await self.assembly.application.start()
        self.started = True
        await self.clock.settle()

    async def close(self) -> None:
        """Stop the application the way ``twin run`` does at the end, and close the model client."""
        if self.started:
            self.started = False
            await self.assembly.application.stop()
        await self.assembly.llm.client.aclose()

    # ---- time -------------------------------------------------------------------------

    @property
    def now(self) -> datetime:
        return self.clock.now_utc()

    @property
    def kit(self) -> Any:
        return schedule_kit(self.services)

    @property
    def zone(self) -> ZoneInfo:
        return self.kit.time.bot_timezone()

    def local(self, hour: int, minute: int = 0, *, on: date | None = None) -> datetime:
        """The instant at which the bot's wall clock shows ``hour:minute`` (today, or ``on``)."""
        day = on or self.kit.time.local_date(self.now)
        return self.kit.time.local_to_utc(day, hour * 60 + minute)

    def today(self) -> date:
        return self.kit.time.local_date(self.now)

    def local_text(self, moment: datetime) -> str:
        return f"{moment.astimezone(self.zone):%Y-%m-%d %H:%M:%S}"

    async def settle(self) -> None:
        await self.clock.settle()

    def meaningful_wake_in(self) -> float | None:
        return self.clock.next_wake_in(HOUSEKEEPING - self.watch)

    async def run_until(
        self,
        moment: datetime,
        *,
        until: Callable[[], bool] | None = None,
        max_step_s: float = 300.0,
        real_limit_s: float = 300.0,
    ) -> None:
        """Let time pass up to ``moment``: from one meaningful wake-up to the next (see LifeClock).

        A step is never longer than ``max_step_s``.  When nothing is waiting for a time but the
        application is still busy (a request is on its way), real time passes, not the clock's.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + real_limit_s
        while self.now < moment:
            await self.clock.settle()
            if until is not None and until():
                return
            wake = self.meaningful_wake_in()
            remaining = (moment - self.now).total_seconds()
            step = min(remaining, max_step_s, wake if wake is not None else max_step_s)
            if step <= 0:
                step = 0.0
            await self.clock.step(step)
            if loop.time() > deadline:
                raise AssertionError(
                    f"the world did not reach {moment} within {real_limit_s}s of real time"
                )
        await self.clock.settle()

    async def run_for(self, **delta: float) -> None:
        await self.run_until(self.now + timedelta(**delta))

    async def run_until_idle(self, *, limit_s: float = 36 * 3600.0) -> None:
        """Let time pass until she has answered everything (the engine is idle, nothing waits)."""
        end = self.now + timedelta(seconds=limit_s)
        await self.run_until(end, until=self.idle)
        assert self.idle(), "the engine never became idle"

    def idle(self) -> bool:
        snap = self.assembly.engine.engine.snapshot()
        return snap.state == "IDLE" and not snap.pending

    # ---- the user ---------------------------------------------------------------------------

    @property
    def handled(self) -> int:
        return self.assembly.engine.handled

    async def say(self, text: str) -> None:
        """The user types ``text``; returns when the engine took it (stored, queued, answered)."""
        before = self.handled
        self.keyboard.feed(text)
        await wait_until(lambda: self.handled > before, limit_s=15.0, interval=0.002)
        await self.clock.settle()

    async def say_picture(self, data: bytes, suffix: str = ".png") -> None:
        """The user sends a photo (``/img <path>`` of the terminal)."""
        self.pictures += 1
        path = self.workdir / f"photo-{self.pictures}{suffix}"
        path.write_bytes(data)
        await self.say(f"/img {path}")

    async def say_sticker(self, data: bytes) -> None:
        """The user sends a sticker: the terminal has no key for it, the channel's inbox does."""
        before = self.handled
        stored = await asyncio.to_thread(self.services.media.put, data, MediaKind.STICKER)
        ref = MediaRef(stored.sha256, MediaKind.STICKER, stored.size, "image/png", "sticker.png")
        self.channel._deliver(MessageKind.IMAGE, media=ref)  # the terminal has no sticker key
        await wait_until(lambda: self.handled > before, limit_s=15.0, interval=0.002)
        await self.clock.settle()

    # ---- what happened ------------------------------------------------------------------------

    @property
    def said(self) -> list[Said]:
        """Everything the bot said on the terminal, in order."""
        return self.screen.messages()

    @property
    def persona_said(self) -> list[Said]:
        return [item for item in self.said if item.persona]

    @property
    def system_said(self) -> list[Said]:
        return [item for item in self.said if item.kind == "system"]

    def said_between(self, first: datetime, last: datetime) -> list[Said]:
        return [item for item in self.persona_said if first <= item.at < last]

    def rows(self) -> list[BotTurn]:
        with self.services.db.session() as session:
            found = list(session.scalars(select(BotTurn).order_by(BotTurn.at, BotTurn.id)))
            session.expunge_all()
        return found

    def out_rows(self) -> list[BotTurn]:
        """The messages of the bot that are conversation (no command answers)."""
        return [row for row in self.rows() if row.direction == "out" and not row.is_command]

    def in_rows(self) -> list[BotTurn]:
        return [row for row in self.rows() if row.direction == "in"]

    def proactive_log(self) -> ProactiveLogStore:
        return ProactiveLogStore(self.services.db, self.services.clock)

    def proactive_rows(self, *, outcomes: Iterable[str] | None = None) -> list[LogEntry]:
        return self.proactive_log().entries(
            outcomes=list(outcomes) if outcomes else None, with_text=True
        )

    def alerts(self) -> list[tuple[str, str]]:
        with self.services.db.session() as session:
            rows = session.scalars(select(Alert).order_by(Alert.created_at, Alert.id))
            return [(row.category, row.severity) for row in rows]

    def jobs(self) -> Counter[str]:
        """The jobs by status (``queued``, ``running``, ``done`` ...)."""
        with self.services.db.session() as session:
            return Counter(row.status for row in session.scalars(select(Job)))

    def plan(self, day: date | None = None) -> DailyPlan:
        zone = self.zone
        found = self.kit.planner.store.current_for(day or self.today(), zone.key)
        assert found is not None, "there is no plan for that day"
        return found

    def decision(self) -> Any:
        return RoundData.of(self.assembly.engine.engine.snapshot()).decision

    @property
    def memory_store(self) -> Any:
        """The store of the memory the engine reads (facts, summaries, follow-ups, life line)."""
        return self.assembly.engine.engine.kit.data.memory.store  # type: ignore[union-attr]

    def followups(self) -> list[Any]:
        return list(self.memory_store.followups())

    def facts(self) -> list[Any]:
        return list(self.memory_store.all_facts())

    def turn_store(self) -> BotTurnStore:
        return BotTurnStore(self.services.db, self.services.clock)

    async def drain_jobs(self, *, limit_s: float = 600.0) -> None:
        """Let the job worker finish what is queued (the memory, the life line, the summaries)."""
        end = self.now + timedelta(seconds=limit_s)
        step = 5.0
        while self.now < end:
            counts = self.jobs()
            if not counts.get("queued") and not counts.get("running"):
                return
            await self.clock.step(step)
        raise AssertionError(f"jobs still waiting: {dict(self.jobs())}")

    def add_her_sticker(self, tag: str = "开心", seed: int = 1, uses: int = 3) -> str:
        """A sticker she has used in the past and carries ``tag``; returns its MD5.

        The library is what the import made: a sticker she may send is one she used before.
        """
        data = make_image_bytes(random.Random(seed), "PNG")
        md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
        writer = MessageWriter(self.services)
        long_ago = self.now - timedelta(days=30)
        for number in range(uses):
            writer.add(long_ago + timedelta(hours=number), True, "sticker", None, md5)
        writer.store(append=True)
        attach_files(self.services, {md5: data})
        sync_counters(self.services)
        run_index(self.services)  # her library of examples includes the new messages
        StickerCatalog(self.services).save_vision(
            md5, [tag], "一只笑着的小猫", "开心的时候", at=self.now
        )
        return md5

    def set_timezone(self, name: str) -> None:
        self.services.runtime.set(BOT_TIMEZONE, name, by="life-world")
