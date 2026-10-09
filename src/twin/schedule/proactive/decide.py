"""The two-step decision: should she write, and what (R-PRO-006).

A candidate that passed the hard constraints goes to DeepSeek (:class:`ProactiveDecider`):

1. **the plan** - the template ``proactive_plan`` (versioned, ``twin/profile/templates``) with the
   full persona card, the recent conversation, and one last message that carries what varies: why
   a message is being considered, the local time and her state, her day so far (the life line, with
   what was told already), the memory block with the open follow-ups, how she really opened
   conversations around this hour (:mod:`twin.retrieval.openers`) and, if the user left her last
   message unanswered, what she said.  The reply is a JSON object (``thinking.proactive_planner``,
   on by default; the budget may switch it off)::

       {"send": true, "kind": "...", "messages": ["..."], "sticker_hint": "", "reason": "...", ...}

   ``send: false`` ends it: the reason is kept, the scheduler decides what happens to the slot.
2. **the words** - with the ``deepseek`` backend the planner's own messages are the text; with
   ``style`` or ``hybrid`` the plan goes into the ``【规划】`` field of the style model's prompt
   (the same field the hybrid backend fills for a reply) and the style model writes the bubbles.
   The style model's prompt must end with a turn of the user's, and nobody has spoken, so the
   conversation ends with a fixed note that she is writing first (:data:`CUE_TEXT`).  A style
   model that fails costs nothing: the planner's messages are used.

The text then goes through the **same post-processing** as a reply
(:meth:`~twin.engine.pipeline.ReplyPipeline.post_process`): AI phrases, event text, punctuation,
length, emoji codes, stickers, promises and the platform quota.  A text that breaks a hard rule - or
that says again what her unanswered last message said (a chase has to be different, R-PRO-007) - is
asked for once more with what was wrong; after that the slot is given up (``generation_failed``) -
a proactive message is never worth a bad message, and there is no fallback line.

Everything the plan and the words need is gathered by a :class:`MaterialSource`
(:class:`LiveMaterial` reads the database); the decider itself reads nothing.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from twin.config.runtime import THINKING_PROACTIVE, RuntimeSettings, ThinkingMode
from twin.engine.backend import BackendRequest
from twin.engine.backend_select import BackendChoice
from twin.engine.dataview import ReplyDataView
from twin.engine.deepseek_backend import usage_of
from twin.engine.postprocess import PostResult
from twin.engine.prompt import NO_PERSONA, STATE_LINES, VIOLATION_NOTES, lay_out_turns, moment_text
from twin.engine.refusal import FILTER_FINISHES
from twin.engine.style_backend import StyleWriter
from twin.engine.style_prompt import PlanFields
from twin.engine.types import (
    Bubble,
    InboundItem,
    PostAction,
    ReplyContext,
    ReplyMaterial,
    SendLimits,
    UsageSummary,
)
from twin.llm.deepseek import DeepSeekClient
from twin.llm.errors import InvalidRequestError, StructuredOutputError
from twin.llm.types import Purpose
from twin.memory.asof import LocalMoment
from twin.memory.blocks import MemoryQuery
from twin.memory.lifeline import LifelineStore, PlannedEvent, minutes_of
from twin.memory.recent import Turn
from twin.memory.records import LifelineRecord
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import PromptText
from twin.retrieval.examples import render_examples
from twin.schedule.proactive.types import KIND_LABELS, Candidate, Reason, TriggerKind

if TYPE_CHECKING:
    from twin.memory.followups import FollowupStore
    from twin.retrieval.openers import OpenerExamples

log = get_logger("twin.schedule.proactive.decide")

TEMPLATE = "proactive_plan"
CUE_TEXT = "（对方这会儿没有说话，是她自己想主动找他聊几句。）"
HISTORY_TURNS = 12  # merged turns of the recent conversation the planner sees
HISTORY_MESSAGES = 60  # messages read to find them
PLAN_TEMPERATURE = 0.8
PLAN_MAX_TOKENS = 700
PLAN_MAX_TOKENS_THINKING = 4000
MAX_FACTS = 4
LIFELINE_SHOWN = 12
SHARE_PREFIX = "L"
MEAL_NAMES = {"breakfast": "早饭", "lunch": "午饭", "dinner": "晚饭"}
REPEAT_KIND = "repeat_previous"
REPEAT_RATIO = 0.85  # a chase this much like the unanswered message is the same message again
REPEAT_NOTE = "这条和上一条主动消息几乎一样，换一种说法，别重复上一条的内容"
NOTES = {**VIOLATION_NOTES, REPEAT_KIND: REPEAT_NOTE}
_NOT_WORDS = re.compile(r"[\s\W_]+")


# --------------------------------------------------------------------------- the plan


class NewDetail(BaseModel):
    """A small detail of her day that the plan makes up (written to the life line)."""

    model_config = ConfigDict(extra="ignore")

    activity: str
    start: str | None = None
    end: str | None = None
    place: str | None = None
    mood: str | None = None

    @field_validator("start", "end", "place", "mood", mode="before")
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


class ProactivePlan(BaseModel):
    """The planner's JSON (R-PRO-006); only ``send`` is required, a loose field is repaired."""

    model_config = ConfigDict(extra="ignore")

    send: bool
    kind: str = ""
    messages: list[str] = Field(default_factory=list)
    sticker_hint: str = ""
    reason: str = ""
    intent: str = ""
    facts_to_use: list[str] = Field(default_factory=list)
    tone: str = ""
    bubble_hint: str = ""
    shares: list[str] = Field(default_factory=list)
    new_detail: NewDetail | None = None
    followup_done: bool = False

    @field_validator(
        "kind", "sticker_hint", "reason", "intent", "tone", "bubble_hint", mode="before"
    )
    @classmethod
    def _as_text(cls, value: object) -> str:
        return _text(value)

    @field_validator("messages", "facts_to_use", "shares", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list | tuple) else [value]
        return [text for text in (_text(item) for item in items) if text]

    @field_validator("new_detail", mode="before")
    @classmethod
    def _detail(cls, value: object) -> object:
        if isinstance(value, dict) and not _text(value.get("activity")):
            return None
        return value if isinstance(value, dict) else None

    @model_validator(mode="after")
    def _a_message_to_send(self) -> ProactivePlan:
        if self.send and not self.messages and not self.sticker_hint:
            raise ValueError("send is true but there is no message and no sticker")
        return self

    def fields(self) -> PlanFields:
        """The fields the style model's prompt carries."""
        return PlanFields(
            intent=self.intent or self.reason,
            facts_to_use=tuple(self.facts_to_use[:MAX_FACTS]),
            tone=self.tone,
            bubble_hint=self.bubble_hint or f"{max(1, len(self.messages))}条",
            sticker_hint=self.sticker_hint,
        )

    def raw_text(self) -> str:
        """The planner's own messages as the one-bubble-per-line text the pipeline parses."""
        lines = list(self.messages)
        if self.sticker_hint:
            lines.append(f"[表情包:{self.sticker_hint}]")
        return "\n".join(lines)


# ------------------------------------------------------------------------- the material


@dataclass(frozen=True)
class Brief:
    """What the scheduler knows about this send and tells the decider."""

    now: datetime
    unanswered: int = 0  # proactive messages the user has not answered (the chase number)
    max_bubbles: int = 8  # the most bubbles the platform leaves for it
    previous: tuple[str, ...] = ()  # the bubbles of those unanswered messages, oldest first


@dataclass(frozen=True)
class Material:
    """Everything the plan reads (see :class:`LiveMaterial`)."""

    candidate: Candidate
    local: LocalMoment
    state: str | None
    persona: str
    memory_text: str
    day_entries: tuple[LifelineRecord, ...]
    current_entry: LifelineRecord | None
    examples_text: str
    history: tuple[Turn, ...]
    unanswered_texts: tuple[str, ...]
    followup_text: str | None
    followup_due: datetime | None
    since_last_min: int | None
    edge_side: str | None
    data: ReplyDataView
    extras: dict[str, Any] = field(default_factory=dict)


class MaterialSource(Protocol):
    """Where the decider gets what the plan reads."""

    async def gather(self, candidate: Candidate, brief: Brief) -> Material: ...


class PostProcessing(Protocol):
    """The part of the reply pipeline the decider needs (``ReplyPipeline.post_process``)."""

    def post_process(
        self,
        raw: str,
        *,
        data: ReplyDataView,
        history: Sequence[Turn],
        limits: SendLimits,
        recent_stickers: Sequence[str | None] = (),
        rng: random.Random | None = None,
        final: bool = False,
    ) -> PostResult: ...


class BackendPick(Protocol):
    async def choose(self) -> BackendChoice: ...


# ----------------------------------------------------------------------------- the draft


@dataclass(frozen=True)
class Draft:
    """The decider's answer for one candidate."""

    send: bool
    reason: str
    bubbles: tuple[Bubble, ...] = ()
    plan: dict[str, Any] | None = None
    backend: str = "deepseek"
    thinking: bool = False
    cost_usd: float = 0.0
    usage: UsageSummary = field(default_factory=UsageSummary)
    actions: tuple[PostAction, ...] = ()
    shared_ids: tuple[str, ...] = ()
    new_detail: PlannedEvent | None = None
    followup_done: bool = False
    failure: Reason | None = None
    failure_detail: str | None = None
    attempts: int = 1
    latency_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.send and bool(self.bubbles) and self.failure is None


def _plain(text: str) -> str:
    return _NOT_WORDS.sub("", text)


def repeats_previous(bubbles: Sequence[Bubble], previous: Sequence[str]) -> bool:
    """Is this message the one she sent before and nobody answered, said again (R-PRO-007)?

    A chase has to say something else.  The texts are compared without spaces and punctuation:
    the same words, or nearly the same (``REPEAT_RATIO``), or every bubble one she already sent.
    """
    old = [_plain(text) for text in previous if _plain(text)]
    new = [_plain(b.text) for b in bubbles if not b.is_sticker and _plain(b.text)]
    if not old or not new:
        return False
    if all(text in old for text in new):
        return True
    return SequenceMatcher(None, "".join(new), "".join(old)).ratio() >= REPEAT_RATIO


def failed(reason: Reason, detail: str | None = None, **fields: Any) -> Draft:
    return Draft(False, detail or reason.value, failure=reason, failure_detail=detail, **fields)


# --------------------------------------------------------------------------- the decider


def kind_line(candidate: Candidate, material: Material) -> str:
    """The "reason to write" paragraph of the plan's context."""
    label = KIND_LABELS[candidate.kind]
    kind = candidate.kind
    if kind is TriggerKind.MEAL:
        meal = MEAL_NAMES.get(str(candidate.detail.get("meal")), "吃饭")
        return f"{kind.value}（{label}）：快到她{meal}的时候了。"
    if kind is TriggerKind.GREETING:
        return f"{kind.value}（{label}）：她刚起床不久。"
    if kind is TriggerKind.BEDTIME:
        return f"{kind.value}（{label}）：她快要睡觉了。"
    if kind is TriggerKind.FOLLOWUP:
        return f"{kind.value}（{label}）：对方之前说过的一件事到时间了，问问结果。"
    if kind is TriggerKind.SILENCE:
        since = material.since_last_min
        gap = f"，已经有 {_span(since)} 没说话了" if since else ""
        return f"{kind.value}（{label}）：你们有一阵没聊了{gap}。"
    if kind is TriggerKind.SHARE:
        return f"{kind.value}（{label}）：想把今天的一件小事讲给对方听。"
    side = "快要睡着，睡不着" if material.edge_side == "falling" else "刚醒"
    return f"{kind.value}（{label}）：她{side}。"


def _span(minutes: int) -> str:
    hours, rest = divmod(max(0, minutes), 60)
    if hours >= 24:
        return f"{hours // 24} 天多"
    if hours:
        return f"{hours} 小时" + (f" {rest} 分钟" if rest else "")
    return f"{rest} 分钟"


def render_context(material: Material, notes: Sequence[str]) -> str:
    """The context block of the plan's last message (see the module description)."""
    candidate = material.candidate
    sections = ["【这次的由头】\n类型：" + kind_line(candidate, material)]
    now = [f"当地时间：{moment_text(material.local)}"]
    if material.state in STATE_LINES:
        now.append(f"她现在的状态：{STATE_LINES[material.state]}")
    if material.current_entry is not None:
        now.append(f"她这会儿大概在：{material.current_entry.line()}")
    sections.append("【此刻】\n" + "\n".join(now))
    if material.day_entries:
        lines = []
        for number, entry in enumerate(material.day_entries[:LIFELINE_SHOWN], start=1):
            told = "（已经对他说过）" if entry.shared else ""
            lines.append(f"{SHARE_PREFIX}{number} {entry.line()}{told}")
        sections.append("【她今天的生活（编号 L1、L2……）】\n" + "\n".join(lines))
    else:
        sections.append("【她今天的生活】\n（今天还没有记下什么）")
    if material.followup_text:
        due = (
            f"（{material.followup_due.astimezone(material.local.local.tzinfo):%m-%d %H:%M} 前后）"
            if material.followup_due is not None and material.local.local.tzinfo is not None
            else ""
        )
        sections.append(f"【要跟进的事】\n{material.followup_text}{due}")
    if material.memory_text.strip():
        sections.append(material.memory_text.strip())
    if material.examples_text.strip():
        sections.append(
            "【她过去自己先开口时是怎么说的（仅供模仿语气，不要照抄内容）】\n"
            + material.examples_text.strip()
        )
    if material.unanswered_texts:
        said = "\n".join(f"- {text}" for text in material.unanswered_texts)
        sections.append(f"【上一条主动消息没有回（对方一直没说话）】\n她之后说过：\n{said}")
    if notes:
        sections.append(
            "【上一次的内容有这些问题，这次要避免】\n" + "\n".join(f"- {n}" for n in notes)
        )
    return (
        "（以下是这次的情况，只用来决定怎么发，不要复述给对方）\n"
        + "\n\n".join(sections)
        + "\n（以上是这次的情况）"
    )


class ProactiveDecider:
    """Plans and writes one proactive message (see the module description)."""

    def __init__(
        self,
        *,
        client: DeepSeekClient,
        template: PromptText,
        sticker_tags: Sequence[str],
        material: MaterialSource,
        pipeline: PostProcessing,
        runtime: RuntimeSettings,
        planner_thinking_allowed: Callable[[], bool] = lambda: True,
        backends: BackendPick | None = None,
        style_writer: StyleWriter | None = None,
        auto_rules: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        self._client = client
        self._template = template
        self._tags = tuple(sticker_tags)
        self._material = material
        self._pipeline = pipeline
        self._runtime = runtime
        self._thinking_allowed = planner_thinking_allowed
        self._backends = backends
        self._style = style_writer
        self._auto_rules = auto_rules
        self._rng = rng or random.Random()  # noqa: S311 - a draw, not security

    @property
    def template_ref(self) -> str:
        return self._template.ref

    async def decide(self, candidate: Candidate, brief: Brief) -> Draft:
        """The plan for ``candidate`` and, if it says send, the checked bubbles."""
        material = await self._material.gather(candidate, brief)
        total = UsageSummary()
        cost = 0.0
        notes: tuple[str, ...] = ()
        last: Draft | None = None
        for attempt in (1, 2):
            planned = await self._plan(
                material, notes, thinking_wanted=self._wants_thinking(candidate)
            )
            if isinstance(planned, Draft):  # the planner failed or refused
                return planned
            plan, plan_cost, plan_usage, thinking, latency = planned
            cost += plan_cost
            total = total.plus(plan_usage)
            if not plan.send:
                return Draft(
                    False,
                    plan.reason or Reason.PLANNER_DECLINED.value,
                    plan=plan.model_dump(),
                    thinking=thinking,
                    cost_usd=cost,
                    usage=total,
                    followup_done=plan.followup_done,
                    attempts=attempt,
                    latency_ms=latency,
                )
            written = await self._write(material, plan, brief, attempt)
            cost += written.cost
            total = total.plus(written.usage)
            post = await asyncio.to_thread(
                self._pipeline.post_process,
                written.text,
                data=material.data,
                history=material.history,
                limits=SendLimits(supports_quote=False, max_bubbles=brief.max_bubbles),
                rng=self._rng,
                final=attempt == 2,
            )
            actions = tuple(post.actions)
            kinds = [v.kind for v in post.violations]
            if (
                post.ok
                and post.bubbles
                and repeats_previous(post.bubbles, material.unanswered_texts)
            ):
                kinds = [REPEAT_KIND]
            elif post.ok and post.bubbles:
                shared, detail = _shares_of(plan, material)
                return Draft(
                    True,
                    plan.reason,
                    bubbles=tuple(post.bubbles),
                    plan=plan.model_dump(),
                    backend=written.backend,
                    thinking=thinking,
                    cost_usd=cost,
                    usage=total,
                    actions=actions,
                    shared_ids=shared,
                    new_detail=detail,
                    followup_done=plan.followup_done,
                    attempts=attempt,
                    latency_ms=latency + written.latency_ms,
                    meta=written.meta,
                )
            kinds = kinds or ["empty"]
            notes = tuple(NOTES[k] for k in kinds if k in NOTES)
            last = failed(
                Reason.GENERATION_FAILED,
                ",".join(kinds),
                plan=plan.model_dump(),
                backend=written.backend,
                thinking=thinking,
                cost_usd=cost,
                usage=total,
                actions=actions,
                attempts=attempt,
            )
            log.info("proactive_violations", attempt=attempt, kinds=kinds)
        return last if last is not None else failed(Reason.GENERATION_FAILED, "no_attempt")

    # ------------------------------------------------------------------ the plan call

    def _wants_thinking(self, candidate: Candidate) -> bool:
        mode: ThinkingMode = self._runtime.get(THINKING_PROACTIVE)
        if not self._thinking_allowed():
            return False
        if mode == "on":
            return True
        if mode == "auto" and self._auto_rules:
            return candidate.kind in (TriggerKind.FOLLOWUP, TriggerKind.SILENCE)
        return False

    async def _plan(
        self, material: Material, notes: Sequence[str], *, thinking_wanted: bool
    ) -> Draft | tuple[ProactivePlan, float, UsageSummary, bool, int]:
        rendered = self._template.render(
            persona=material.persona.strip() or NO_PERSONA,
            sticker_tags="、".join(self._tags) or "（没有可用的表情包标签，留空）",
            emoji_codes=_emoji_text(material.data),
            context=render_context(material, notes),
        )
        layout = lay_out_turns(rendered, material.history)
        try:
            result = await self._client.chat_json(
                layout,
                ProactivePlan,
                purpose=Purpose.PROACTIVE,
                thinking=thinking_wanted,
                temperature=None if thinking_wanted else PLAN_TEMPERATURE,
                max_tokens=PLAN_MAX_TOKENS_THINKING if thinking_wanted else PLAN_MAX_TOKENS,
            )
        except StructuredOutputError as exc:
            log.warning("proactive_plan_invalid")
            return failed(Reason.PLANNER_FAILED, "invalid_json", meta={"error": type(exc).__name__})
        except InvalidRequestError as exc:
            kind = "content_risk" if "risk" in str(exc).lower() else "invalid_request"
            log.warning("proactive_plan_refused", kind=kind)
            return failed(Reason.PLANNER_FAILED, kind)
        except Exception as exc:  # DeepSeek is down, out of money, or slow: nothing is sent
            log.warning("proactive_plan_failed", error=type(exc).__name__)
            return failed(Reason.PLANNER_FAILED, type(exc).__name__)
        chat = result.chat
        if chat.finish_reason in FILTER_FINISHES:
            log.warning("proactive_plan_refused", kind=chat.finish_reason)
            return failed(Reason.PLANNER_FAILED, "content_filter", cost_usd=result.total_cost_usd)
        return result.value, result.total_cost_usd, usage_of(chat), chat.thinking, chat.latency_ms

    # -------------------------------------------------------------------- the words

    async def _write(
        self, material: Material, plan: ProactivePlan, brief: Brief, attempt: int
    ) -> _Written:
        """The raw text: the planner's own messages, or the style model's version of the plan."""
        chosen = "deepseek"
        if self._backends is not None and self._style is not None:
            try:
                chosen = (await self._backends.choose()).name
            except Exception as exc:  # a broken selector costs the style model, not the message
                log.warning("proactive_backend_choice_failed", error=type(exc).__name__)
        if chosen not in ("style", "hybrid") or self._style is None:
            return _Written(plan.raw_text(), "deepseek", UsageSummary(), 0, 0.0, {})
        cue = InboundItem(id="proactive", at=brief.now, kind="text", text=CUE_TEXT)
        context = ReplyContext(
            inbound=(cue,),
            history=material.history,
            backend=chosen,
            limits=SendLimits(supports_quote=False, max_bubbles=brief.max_bubbles),
            woke_up=material.candidate.kind is TriggerKind.GREETING,
            seed=self._rng.randrange(2**31),
        )
        request = BackendRequest(context, material.data, _blank_material(material), (), attempt)
        try:
            written = await self._style.write(request, plan.fields())
        except Exception as exc:  # the style model is down or cannot render: use the planner's text
            log.warning("proactive_style_failed", error=type(exc).__name__)
            return _Written(
                plan.raw_text(), "deepseek", UsageSummary(), 0, 0.0, {"style_failed": True}
            )
        return _Written(
            written.text, chosen, written.usage, written.latency_ms, 0.0, dict(written.meta)
        )


@dataclass(frozen=True)
class _Written:
    text: str
    backend: str
    usage: UsageSummary
    latency_ms: int
    cost: float
    meta: dict[str, Any]


def _emoji_text(data: ReplyDataView) -> str:
    emoji = data.emoji_codes
    codes = emoji.codes if emoji is not None else ()
    return " ".join(codes[:30]) or "（她几乎不用微信表情代码，不要用）"


def _blank_material(material: Material) -> ReplyMaterial:
    """The material of a reply as far as the style writer reads it: nothing but the moment."""
    return ReplyMaterial(
        local=material.local,
        state=material.state,
        lifeline=None,
        persona="",
        persona_ref=None,
        memory_text="",
        memory_item_ids=(),
        examples=(),
        examples_text="",
        closing=None,
        asks_if_ai=False,
    )


def _shares_of(
    plan: ProactivePlan, material: Material
) -> tuple[tuple[str, ...], PlannedEvent | None]:
    """The life line entries the plan told (their ids) and the new detail it made up."""
    entries = material.day_entries[:LIFELINE_SHOWN]
    ids: list[str] = []
    for token in plan.shares:
        cleaned = token.strip().upper().removeprefix(SHARE_PREFIX)
        if cleaned.isdigit() and 1 <= int(cleaned) <= len(entries):
            ids.append(entries[int(cleaned) - 1].id)
    found = plan.new_detail
    detail: PlannedEvent | None = None
    if found is not None:
        start = found.start if minutes_of(found.start) is not None else None
        end = found.end if minutes_of(found.end) is not None else None
        detail = PlannedEvent(found.activity, start, end, found.place, found.mood)
    return tuple(dict.fromkeys(ids)), detail


# ------------------------------------------------------------------------ live material


class LiveMaterial:
    """:class:`MaterialSource` over the running application's data (one per process)."""

    def __init__(
        self,
        *,
        data_view: Callable[[datetime], ReplyDataView],
        lifeline: LifelineStore,
        followups: FollowupStore,
        openers: OpenerExamples | None,
        read_turns: Callable[[], Sequence[Turn]],
        local_date_of: Callable[[datetime], date],
        examples_k: Callable[[], int],
        memory_tokens: int,
        last_interaction: Callable[[], datetime | None],
    ) -> None:
        self._data_view = data_view
        self._lifeline = lifeline
        self._followups = followups
        self._openers = openers
        self._read_turns = read_turns
        self._local_date_of = local_date_of
        self._examples_k = examples_k
        self._memory_tokens = memory_tokens
        self._last_interaction = last_interaction

    async def gather(self, candidate: Candidate, brief: Brief) -> Material:
        now = brief.now
        view = self._data_view(now)

        def read() -> tuple[
            Any, Any, Any, Any, tuple[LifelineRecord, ...], LifelineRecord | None, tuple[Turn, ...]
        ]:
            turns = tuple(self._read_turns())[-HISTORY_TURNS:]
            followup = self._followups.get(candidate.followup_id) if candidate.followup_id else None
            topic_parts = [turn.text for turn in turns[-2:]]
            if followup is not None:
                topic_parts.append(followup.text)
            topic = "\n".join(topic_parts)
            memory = view.memory_block(MemoryQuery(text=topic), self._memory_tokens)
            entries = tuple(self._lifeline.day(self._local_date_of(now)))
            current = self._lifeline.at(now)
            return view.persona_full(), memory, view.her_state(), followup, entries, current, turns

        persona, memory, state, followup, entries, current, turns = await asyncio.to_thread(read)
        local = view.local
        examples_text = ""
        wanted = self._examples_k()
        if self._openers is not None and wanted > 0:
            try:
                found = await self._openers.query(
                    local_minute=local.minute, day_type=local.day_type, now=now, k=wanted
                )
                examples_text = render_examples(found) if found else ""
            except Exception as exc:  # the examples are optional: the message goes on without
                log.warning("proactive_examples_unavailable", error=type(exc).__name__)
        last = self._last_interaction()
        since = int((now - last).total_seconds() // 60) if last is not None else None
        edge_side: str | None = None
        if candidate.kind is TriggerKind.EDGE:
            edge_side = str(candidate.detail.get("side") or "falling")
        return Material(
            candidate=candidate,
            local=local,
            state=state,
            persona=persona.text if persona is not None else "",
            memory_text=memory.text,
            day_entries=entries,
            current_entry=current,
            examples_text=examples_text,
            history=turns,
            unanswered_texts=brief.previous,
            followup_text=followup.text if followup is not None else None,
            followup_due=followup.due_at if followup is not None else None,
            since_last_min=since,
            edge_side=edge_side,
            data=view,
        )


__all__ = [
    "CUE_TEXT",
    "Brief",
    "Draft",
    "LiveMaterial",
    "Material",
    "ProactiveDecider",
    "ProactivePlan",
    "failed",
    "render_context",
    "repeats_previous",
]
