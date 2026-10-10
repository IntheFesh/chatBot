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
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from unittest import mock
from zoneinfo import ZoneInfo

import httpx
import respx
from sqlalchemy import select

from tests.fixtures.synth_export import make_image_bytes
from tests.support.console import ScriptedInput
from tests.support.deepseek import API, TEST_KEY, error, ok, request_json
from tests.support.embedding import HashingBackend
from tests.support.ilink import API as ILINK_API
from tests.support.ilink import BOT, CTX, TOKEN, USER
from tests.support.life_clock import HOUSEKEEPING, LifeClock
from tests.support.life_screen import Said, TimedOutput
from tests.support.lifeline import ScriptedLifelineModel
from tests.support.memory import ScriptedMemoryModel
from tests.support.network import OfflineTransport
from tests.support.persona import attach_files, sync_counters
from tests.support.proactive_world import ProactiveScript, proactive_model
from tests.support.style_models import register_model
from tests.support.synth_chat import ChatSpec, MessageWriter, build_chat
from tests.support.waiting import wait_until
from tests.support.wechat_double import Phone, WeChatDouble
from twin.assembly import Assembly, assemble
from twin.channel.base import Channel, MediaRef, MessageKind
from twin.channel.ilink.store import Credentials, IlinkStore
from twin.channel.state import ChannelStateStore
from twin.clock import set_active_clock
from twin.config.runtime import BOT_TIMEZONE
from twin.engine.roundstate import RoundData
from twin.engine.turns import BotTurnStore
from twin.llm.runtime import DEEPSEEK_SECRET
from twin.profile.activity_model import ActivityModel
from twin.profile.builder import rebuild
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.profile.store import ACTIVITY_ACTIVE, VersionStore
from twin.retrieval.indexer import run_index
from twin.schedule.plan_model import DailyPlan
from twin.schedule.proactive.store import LogEntry, ProactiveLogStore
from twin.schedule.service import schedule_kit
from twin.schedule.store import SALT_KEY
from twin.services import CliContext, Services, build_services, get_cli_context, set_cli_context
from twin.stickers.catalog import StickerCatalog
from twin.storage.crypto import set_active_keyring
from twin.storage.engine_models import BotTurn
from twin.storage.media import MediaKind
from twin.storage.models import Alert, Job
from twin.storage.profile_models import ActivityModelVersion
from twin.storage.settings_store import put_setting

USER_WORDS = re.compile(r"对方这一轮说的话：\s*(.*)\Z", re.DOTALL)
CHICAGO = "America/Chicago"
PLAN_SALT = "life-world"
START_GRACE_S = 0.3  # real seconds the start of the application has to end by itself


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
    reasoning: str | None = None  # what the model "thought", sent when thinking is on

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


# what she says on her own, kind by kind: every time a different one, so that the post-processing
# (which drops a bubble that repeats the last one) never empties a proactive message
VARIANTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "greeting": (("早啊", "刚醒"), ("早呀",), ("醒啦",), ("早安",)),
    "meal": (("吃饭了吗",), ("你吃了没",), ("我饿了",), ("该吃饭啦",)),
    "bedtime": (("我先睡啦", "晚安"), ("困了 晚安",), ("要睡了",), ("睡觉咯 晚安",)),
    "followup": (("考试怎么样了",), ("考完了吗",), ("结果出来了吗",)),
    "silence": (("在干嘛呀",), ("人呢",), ("忙完了吗",), ("在不在",)),
    "share": (
        ("刚刚在图书馆看文献", "有点困"),
        ("今天的课好长",),
        ("路上看到一只好胖的猫",),
        ("晚饭有点咸",),
    ),
    "edge": (("睡不着",), ("刚醒",), ("还没睡",)),
}


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
    "correction_rules": "rules",
    "correction_rule_check": "rules_check",
    "sticker_tag": "sticker_tag",
    "sticker_context": "sticker_tag",
}


class LifeDeepSeek:
    """A DeepSeek that is made up: deterministic answers chosen by what the request is about."""

    def __init__(self, clock: LifeClock) -> None:
        self.clock = clock
        self.memory = ScriptedMemoryModel()
        self.lifeline = ScriptedLifelineModel()
        self.proactive = ProactiveScript(by_hand=self._plan_by_hand)
        self._variant: Counter[str] = Counter()
        self.on_her_own: dict[str, tuple[tuple[str, ...], ...]] = {}  # replaces VARIANTS by kind
        self.book = ReplyBook()
        self.caption = "一张桌子上放着一杯咖啡的照片"
        self.rules = ["不要说得太客气", "少用句号"]  # what the weekly consolidation finds
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

    def _plan_by_hand(self, kind: str, _body: dict[str, Any]) -> dict[str, Any]:
        """The planner's answer: send, with the next variant of what she says for this kind."""
        if kind in self.proactive.decline:
            return {"send": False, "kind": kind, "messages": [], "reason": "现在发不合适"}
        options = self.on_her_own.get(kind) or VARIANTS.get(kind, (("嗯",),))
        lines = options[self._variant[kind] % len(options)]
        self._variant[kind] += 1
        return {
            "send": True,
            "kind": kind,
            "messages": list(lines),
            "sticker_hint": "",
            "reason": f"{kind} 的理由",
            "intent": "找他聊聊",
            "tone": "随意",
        }

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
        if kind == "rules":
            return self._plain(json.dumps({"rules": self.rules}, ensure_ascii=False))
        if kind == "rules_check":
            asked = str(body["messages"][-1]["content"])
            numbered = re.findall(r"^(\d+)\. ", asked, re.MULTILINE)
            verdicts = [{"index": int(n), "ok": True, "kind": "style"} for n in numbered]
            return self._plain(json.dumps({"verdicts": verdicts}))
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
        thinking = body.get("thinking", {}).get("type") == "enabled"
        return ok(
            content=self.book.answer(words, whole),
            reasoning=self.book.reasoning if thinking else None,
            prompt=self.prompt_tokens,
            hit=self.cache_hit_tokens,
            completion_tokens=self.completion_tokens,
        )

    def _crisis(self, body: dict[str, Any]) -> httpx.Response:
        """The second judgement: it reads the newest line (the last of the list)."""
        prompt = str(body["messages"][-1]["content"])
        lines = [x for x in prompt.splitlines() if x.startswith("- ")]
        newest = lines[-1] if lines else ""
        sure = any(word in newest for word in ("不想活", "结束生命", "自杀"))
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


def add_her_answers(
    services: Services, day: date, answers: tuple[str, ...], times: int = 6
) -> None:
    """Short answers that she gave ``times`` times each in the past: the words of her profile.

    The generated chat has no repeated sentence; what she says most often (and what the
    last-resort answer of R-ENG-010 is drawn from) comes from here.
    """
    writer = MessageWriter(services)
    opening = datetime.combine(day, time(18, 0), tzinfo=ZoneInfo("UTC")) + timedelta(days=1)
    for number in range(times):
        for index, text in enumerate(answers):
            at = opening + timedelta(minutes=10 * (number * len(answers) + index))
            writer.add(at, True, "text", text)
    writer.store(append=True)
    sync_counters(services)


def log_in_and_bind(services: Services, clock: LifeClock) -> None:
    """The user has scanned the code and been bound (what ``twin channel login`` and bind leave)."""
    store = IlinkStore(ChannelStateStore(services.db), clock)
    store.save_credentials(
        Credentials(
            bot_token=TOKEN,
            ilink_bot_id=BOT,
            ilink_user_id=USER,
            api_base_url=ILINK_API,
            saved_at=clock.now_utc().isoformat(),
        )
    )
    store.bind(USER, context_token=CTX)


def register_style_model(services: Services) -> str:
    """A registered, active model that passed the gate, locked to a stored persona card v1."""
    text = (
        compose.stats_block(["几乎不用逗号"])
        + compose.auto_block("### 风格\n- 训练时的口头禅是嘿嘿\n\n### 基本情况\n- 一件事实\n\n")
        + compose.manual_block([], None)
    )
    PersonaStore(services.db, services.clock).add_version("pre_holdout", text, reason="generate")
    return register_model(services, persona_version="v1")


# ------------------------------------------------------------------------------ the world


@dataclass
class LifeWorld:
    """A running application, a user at the terminal (or the phone), and the doubles around them."""

    services: Services
    clock: LifeClock
    deepseek: LifeDeepSeek
    assembly: Assembly
    keyboard: ScriptedInput | None
    screen: TimedOutput | Phone
    channel: Channel
    api: respx.MockRouter
    workdir: Path
    seed: int = 7
    double: WeChatDouble | None = None
    watch: set[str] = field(default_factory=set)
    started: bool = False
    pictures: int = 0
    before_start: set[asyncio.Task[Any]] = field(default_factory=set)
    retired: list[Services] = field(default_factory=list)

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
        platform: Literal["console", "ilink"] = "console",
        platform_window_h: float = 24.0,
        platform_quota: int = 10,
        her_answers: tuple[str, ...] = (),
    ) -> LifeWorld:
        """Build the world and, unless told not to, start the application (see the module text)."""
        settings = services.settings
        settings.retrieval.model = embedder.info.model
        settings.channel.kind = "console" if platform == "console" else "ilink"
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
        if her_answers:
            add_her_answers(services, first, her_answers)
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
        double: WeChatDouble | None = None
        keyboard: ScriptedInput | None = None
        if platform == "ilink":  # the WeChat platform and the user's phone, made up
            double = WeChatDouble(clock, window_h=platform_window_h, quota=platform_quota)
            double.mount(api)
            log_in_and_bind(services, clock)
            double.last_user_at = clock.now_utc()  # the message that bound him was just now
            screen: TimedOutput | Phone = double.phone
            assembly = assemble(services, rng=random.Random(seed))
        else:
            keyboard, screen = ScriptedInput(), TimedOutput(clock)
            assembly = assemble(
                services, console_input=keyboard, console_output=screen, rng=random.Random(seed)
            )
        world = cls(
            services,
            clock,
            deepseek,
            assembly,
            keyboard,
            screen,
            assembly.channel,
            api,
            workdir,
            seed,
            double,
        )
        clock.fast = world.watch  # what a scenario watches is woken at every step, too
        if start_application:
            await world.start()
        return world

    async def start(self) -> None:
        """Start the application; virtual time moves only if it cannot finish without.

        The start of some components waits for a time (the first tick of the schedule may send, and
        pause, on a platform that has a window): then the clock is stepped.  Otherwise it ends in
        real milliseconds, and not one second of the world's time goes by.
        """
        self.before_start = asyncio.all_tasks()
        starting = asyncio.ensure_future(self.assembly.application.start())
        loop = asyncio.get_running_loop()
        waiting_since = loop.time()
        while not starting.done():
            await self.clock.settle()
            if starting.done():
                break
            if loop.time() - waiting_since < START_GRACE_S:
                await asyncio.sleep(0.002)  # a hop between a thread and the loop, not a sleeper
                continue
            wake = self.meaningful_wake_in()
            if wake is None:
                await asyncio.sleep(0.001)
            else:
                await self.clock.step(min(wake, 60.0))
                waiting_since = loop.time()
        await starting
        self.started = True
        await self.clock.settle()

    async def close(self) -> None:
        """Stop the application the way ``twin run`` does at the end, and close the model client."""
        if self.started:
            self.started = False
            await self.assembly.application.stop()
        await self.assembly.llm.client.aclose()
        for old in self.retired:
            old.close()

    async def kill(self) -> None:
        """The process dies: every task of the application ends where it is, nothing is cleaned up.

        No component's ``stop`` runs (the channel does not say goodbye, the engine does not
        finish its round); what is on disk is all that is left.
        """
        self.started = False
        victims = asyncio.all_tasks() - self.before_start - {asyncio.current_task()}
        for task in victims:
            task.cancel()
        await asyncio.gather(*victims, return_exceptions=True)
        await self.assembly.llm.client.aclose()
        await self.clock.settle()

    async def restart(self) -> None:
        """A new process on the same files: new services, a new application, the same world around.

        The user's side - the terminal keyboard or the platform with its phone - and the made-up
        DeepSeek are the same objects: they are not part of the process that died.
        """
        old = self.services
        services = build_services(
            old.settings, root=old.paths.root, secrets=old.secrets, clock=self.clock
        )
        services.runtime.initialize()
        self.retired.append(old)
        self.services = services
        if self.double is not None:
            self.assembly = assemble(services, rng=random.Random(self.seed + 1))
        else:
            assert isinstance(self.screen, TimedOutput) and self.keyboard is not None
            self.assembly = assemble(
                services,
                console_input=self.keyboard,
                console_output=self.screen,
                rng=random.Random(self.seed + 1),
            )
        self.channel = self.assembly.channel
        await self.start()

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
        crawled = 0  # steps in a row that moved the clock by less than a second
        while self.now < moment:
            await self.clock.settle()
            if until is not None and until():
                return
            wake = self.meaningful_wake_in()
            remaining = (moment - self.now).total_seconds()
            step = min(remaining, max_step_s, wake if wake is not None else max_step_s)
            crawled = crawled + 1 if step < 1.0 else 0
            if crawled > 200:
                raise AssertionError(
                    f"time crawls at {self.local_text(self.now)}: {self.clock.sleepers()[:8]}"
                )
            await self.clock.step(max(0.0, step))
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
        if self.double is not None:
            self.double.user_types(text)
            for _ in range(30):  # the channel's poll runs once a second
                await self.clock.step(1.0)
                if self.handled > before:
                    return
            raise AssertionError("the message never reached the engine")
        assert self.keyboard is not None
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
        """The jobs by status (``pending``, ``running``, ``done``, ``failed`` ...)."""
        with self.services.db.session() as session:
            return Counter(row.status for row in session.scalars(select(Job)))

    def job_rows(self) -> list[tuple[str, str, int, str | None]]:
        """``(type, status, attempts, last error)`` of every job, oldest first."""
        with self.services.db.session() as session:
            rows = session.scalars(select(Job).order_by(Job.created_at, Job.id))
            return [(row.type, row.status, row.attempts, row.last_error) for row in rows]

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

    def awaiting_approval(self) -> list[tuple[str, str, int, str | None]]:
        """The jobs of a one-time batch that wait for ``twin jobs approve`` (R-LLM-014)."""
        with self.services.db.session() as session:
            rows = session.scalars(
                select(Job)
                .where(Job.requires_approval.is_(True), Job.approved_at.is_(None))
                .where(Job.status == "pending")
                .order_by(Job.created_at, Job.id)
            )
            return [(row.type, row.status, row.attempts, row.last_error) for row in rows]

    async def drain_jobs(self, *, limit_s: float = 600.0, approval_ok: bool = False) -> None:
        """Let the job worker finish what is queued (the memory, the life line, the summaries).

        ``approval_ok``: jobs that wait for the user's approval of their cost are not waited for.
        """
        end = self.now + timedelta(seconds=limit_s)
        step = 5.0
        while self.now < end:
            counts = self.jobs()
            waiting = counts.get("pending", 0) - (
                len(self.awaiting_approval()) if approval_ok else 0
            )
            if not waiting and not counts.get("running"):
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

    def cli(self, *args: str, answer: str | None = None) -> tuple[int, str]:
        """Another process of the machine runs ``twin <args>``: ``(exit code, output)``.

        The command line builds its own container on the same files - its own database
        connection, the same key - as a second process does.  The wall clock is the machine's, so
        it is the clock of the world (the real command line has the system clock, which every
        process of the machine shares).  The application of this process goes on as it was.
        """
        from typer.testing import CliRunner

        from tests.support.cli_runner import invoke  # (imports the whole command line)

        before = get_cli_context()
        set_cli_context(
            CliContext(secrets=self.services.secrets, http_transport=OfflineTransport())
        )
        options = ["--set", f"paths.data_dir={self.services.paths.data_dir}"]
        try:
            with mock.patch("twin.services.SystemClock", return_value=self.clock):
                return invoke(CliRunner(), [*options, *args], answer=answer)
        finally:
            set_cli_context(before)
            set_active_clock(self.clock)
            set_active_keyring(self.services.keyring)
