"""``StylePromptBuilder``: the style model's prompt, for inference and training (R-TRN-011).

One class renders the prompt of the fine-tuned style model **everywhere**: the running bot (the
``style`` and ``hybrid`` backends), the training-set export (round 13, ``twin train export``) and
the evaluation sandbox (round 09b).  They differ only in the data they hand in: the bot passes
the live data view, the export and the sandbox pass ``AsOfView(t)`` of the moment of the sample
(R-TRN-013), so nothing from the future can reach a prompt.

The format is the chat template the model is trained with, LLaMA-Factory's ``qwen3_nothink``
(plain ChatML, no think marker).  Nothing in this module spells a piece of that format: every
delimiter comes from :mod:`twin.training.lf_template`, so the text built here and the text the
training side encodes are the same by construction (character-level tests compare them).

What goes into the prompt
=========================

The **system segment** (``system`` below) is, in this order and each part only when it has
content:

1. the compact persona card - style only, no facts, no corrections (R-PERS-004);
2. ``【此刻】`` - the local time with the weekday, and what she is doing (her state);
3. the memory block, within ``memory_tokens`` (R-MEM-008);
4. ``【规划】`` - the plan of the hybrid backend (intent, facts to use, tone, ...), if any;
5. ``【前文】`` - her turns that open the conversation part (see below), if any.

The **conversation segment** is the last ``max_turns`` (8, as in the training set) *merged*
turns - consecutive messages of one side are one turn, joined with a newline - and ends with the
user's turn that is being answered; the prompt ends with ``<|im_start|>assistant\\n`` and the model
continues with her reply and ``<|im_end|>``.

ShareGPT (and therefore LLaMA-Factory) needs the conversation to start with the user and to
alternate.  :func:`normalize_context` makes any run of turns so, and it is the function the export
and the live backends share: adjacent turns of one side are merged, and turns of hers that open
the context cannot be a "gpt" turn without a "human" turn before them, so they move to the
``前文`` section of the system segment.

Fitting a token budget
======================

LLaMA-Factory cuts a sample at ``cutoff_len`` (2048); with ``mask_history`` the part that is cut
is the **end** of the prompt side of the last turn - the ``<|im_start|>assistant\\n`` opener - which
silently trains the model on a prompt it never sees at inference.  ``build(..., budget=...)``
therefore drops what is oldest until the prompt fits: first the ``前文``, then the oldest turns
(keeping the context a valid one, so a her-turn that would open it goes too).  The budget and the
tokenizer are the caller's: ``TokenBudget(max_tokens, count)``.  What does not fit even after
everything droppable is gone is reported (``meta.over_budget``) and left to the caller.

Locked versions
===============

A registered model is bound to the persona card version and the template version it was trained
with (``model_registry``, R-SRV-001).  A builder made with ``locked=LockedVersions(...)`` renders
the pre-holdout card of exactly that version and refuses a template version it cannot render
(:class:`LockedVersionError`), so a model is never served with a prompt it was not trained on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from twin.engine.dataview import ReplyDataView
from twin.engine.prompt import STATE_LINES, moment_text
from twin.llm.style_client import IM_END, PromptMeta, RenderedPrompt
from twin.memory.blocks import MemoryBlock, MemoryQuery
from twin.profile.persona.render import RenderedPersona, count_tokens
from twin.training import lf_template
from twin.training.lf_template import Turn
from twin.training.registry import LockedVersions

if TYPE_CHECKING:
    from twin.services import Services

STYLE_CONTEXT_TURNS = 8
"""Merged turns of the conversation part: the same number the training set uses (R-TRN-002)."""
TOPIC_TURNS = 2  # turns before the last one that the memory search also reads (as the pipeline)
DEFAULT_MEMORY_TOKENS = 300
PRE_HOLDOUT = "pre_holdout"

NOW_HEADING = "【此刻】"
PLAN_HEADING = "【规划】"
PRELUDE_HEADING = "【前文】"
WOKE_UP_LINE = "你刚醒，刚看到对方发来的这些消息。"
PLAN_FIELD_CHARS = 80
PLAN_FACTS_SHOWN = 6
_BLANK_LINES = re.compile(r"\n{3,}")

Side = Literal["user", "assistant"]
RenderPersona = Callable[[str], RenderedPersona | None]


class StylePromptError(ValueError):
    """A prompt cannot be built from what was handed in."""


class LockedVersionError(StylePromptError):
    """The versions a model is locked to cannot be rendered by this code."""


class TurnLike(Protocol):
    """A turn of the conversation as the memory (``memory.recent.Turn``) hands it out."""

    @property
    def role(self) -> str: ...

    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class StyleTurn:
    """One turn of the context: who said it and what (``assistant`` is her)."""

    role: Side
    text: str


def style_turns(turns: Iterable[TurnLike]) -> list[StyleTurn]:
    """Turns as the memory hands them out (``user`` / ``bot``) as :class:`StyleTurn` objects."""
    converted: list[StyleTurn] = []
    for turn in turns:
        if turn.role == "user":
            converted.append(StyleTurn("user", turn.text))
        elif turn.role in {"bot", "assistant", "her"}:
            converted.append(StyleTurn("assistant", turn.text))
        else:
            raise StylePromptError(f"unknown side {turn.role!r}; a turn is the user's or hers")
    return converted


# ----------------------------------------------------------------------------- text hygiene


def scrub(text: str) -> str:
    """``text`` without what the template would misread as format.

    ChatML control tokens (``<|im_start|>``, ``<|im_end|>``, ``<|endoftext|>``) are tokenised as
    one special token and LLaMA-Factory's slot names (``{{content}}``, ``{{idx}}``) are
    substituted, so a message that holds one would not mean to the model what it says.  Training
    and inference both pass every piece of text through here, so both see the same string.
    """
    for token in (*lf_template.CONTROL_TOKENS, *lf_template.LF_SLOTS):
        text = text.replace(token, "")
    return text


def _clean(text: str) -> str:
    return scrub(text).strip()


# -------------------------------------------------------------------- the conversation part


@dataclass(frozen=True)
class NormalizedContext:
    """The conversation part, in the shape ShareGPT needs.

    ``turns`` starts with the user and alternates; ``prelude`` holds the texts of her turns that
    opened the context (they go to the system segment).  ``merged`` is how many merged turns
    the window held before the prelude was split off.
    """

    turns: tuple[Turn, ...]
    prelude: tuple[str, ...]
    merged: int

    @property
    def ends_with_user(self) -> bool:
        return bool(self.turns) and self.turns[-1].role == "user"


def normalize_context(
    turns: Sequence[StyleTurn], *, max_turns: int = STYLE_CONTEXT_TURNS
) -> NormalizedContext:
    """Make ``turns`` a valid ShareGPT context (training and inference share this function).

    1. blank turns are dropped, text is scrubbed and stripped;
    2. adjacent turns of one side are merged into one (texts joined with a newline - the merged
       block rule of the conversation window);
    3. only the newest ``max_turns`` merged turns are kept;
    4. turns of hers that open what is left are taken out as the ``prelude`` - the conversation
       part must start with the user.

    The result may still end with a turn of hers (or be empty); the builder refuses that, because
    a prompt must end with the user's turn that is answered.
    """
    merged: list[tuple[Side, list[str]]] = []
    for turn in turns:
        text = _clean(turn.text)
        if not text:
            continue
        if merged and merged[-1][0] == turn.role:
            merged[-1][1].append(text)
        else:
            merged.append((turn.role, [text]))
    window = merged[-max_turns:] if max_turns > 0 else []
    start = 0
    while start < len(window) and window[start][0] == "assistant":
        start += 1
    prelude = tuple("\n".join(parts) for _, parts in window[:start])
    kept = tuple(Turn(side, "\n".join(parts)) for side, parts in window[start:])
    return NormalizedContext(kept, prelude, len(window))


# ---------------------------------------------------------------------------- the plan field


@dataclass(frozen=True)
class PlanFields:
    """The plan the hybrid backend hands to the style model (R-ENG-006, R-TRN-005).

    Every field is optional; only those with content are written.  The training set uses the
    same fields for the share of samples that carry a plan.
    """

    intent: str = ""
    facts_to_use: tuple[str, ...] = ()
    tone: str = ""
    bubble_hint: str = ""
    sticker_hint: str = ""

    @property
    def empty(self) -> bool:
        return not render_plan(self)


def _one_line(text: str) -> str:
    return " ".join(_clean(text).split())[:PLAN_FIELD_CHARS]


def render_plan(plan: PlanFields) -> str:
    """The lines of the ``【规划】`` section (without its heading); empty when nothing is set."""
    lines: list[str] = []
    if intent := _one_line(plan.intent):
        lines.append(f"想表达：{intent}")
    facts = [fact for fact in (_one_line(f) for f in plan.facts_to_use) if fact]
    if facts:
        lines.append("会用到：" + "；".join(facts[:PLAN_FACTS_SHOWN]))
    if tone := _one_line(plan.tone):
        lines.append(f"语气：{tone}")
    if bubbles := _one_line(plan.bubble_hint):
        lines.append(f"气泡：{bubbles}")
    if sticker := _one_line(plan.sticker_hint):
        lines.append(f"表情包：{sticker}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- the budget


@dataclass(frozen=True)
class TokenBudget:
    """How long the prompt may be and how to count it (the caller injects its tokenizer)."""

    max_tokens: int
    count: Callable[[str], int] = count_tokens

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("a token budget needs at least one token")

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.max_tokens


# --------------------------------------------------------------------------- the prompt parts


@dataclass(frozen=True)
class PromptParts:
    """A prompt before it is rendered: what ShareGPT calls ``system`` and ``conversations``.

    The training export writes ``system`` and ``lf_template.sharegpt_conversations(turns,
    reply)``; the backends call :meth:`render`.  Both come from the same object, so what the model
    is trained on is what it is shown.
    """

    system: str
    turns: tuple[Turn, ...]
    meta: PromptMeta
    memory: MemoryBlock = field(default_factory=lambda: MemoryBlock(""))

    def render(self) -> RenderedPrompt:
        text = lf_template.render_prompt(self.system, self.turns)
        return RenderedPrompt(
            text, template=lf_template.TEMPLATE_VERSION, stop=(IM_END,), meta=self.meta
        )


class StylePromptBuilder:
    """Builds the style model's prompt (see the module description).

    ``locked`` binds the builder to the versions of a registered model, ``render_persona`` turns
    a locked card version into the compact card (``from_services`` wires the persona store).
    Without ``locked`` the card is the one the data view hands out.
    """

    def __init__(
        self,
        *,
        locked: LockedVersions | None = None,
        render_persona: RenderPersona | None = None,
        memory_tokens: int = DEFAULT_MEMORY_TOKENS,
        max_turns: int = STYLE_CONTEXT_TURNS,
    ) -> None:
        if locked is not None:
            if locked.template_version != lf_template.TEMPLATE_VERSION:
                raise LockedVersionError(
                    f"the model is bound to the template {locked.template_version!r}, "
                    f"this program renders {lf_template.TEMPLATE_VERSION!r}"
                )
            if render_persona is None:
                raise LockedVersionError("a locked builder needs a way to load the locked card")
        if memory_tokens < 0 or max_turns < 1:
            raise ValueError("memory_tokens must be >= 0 and max_turns >= 1")
        self._locked = locked
        self._render_persona = render_persona
        self._memory_tokens = memory_tokens
        self._max_turns = max_turns

    @classmethod
    def from_services(
        cls,
        services: Services,
        *,
        locked: LockedVersions | None = None,
        memory_tokens: int | None = None,
    ) -> StylePromptBuilder:
        """The builder of the running bot (``locked``: the versions of the active model)."""
        from twin.profile.persona.api import render_compact
        from twin.profile.persona.store import PersonaVersionError

        def render_locked(version: str) -> RenderedPersona | None:
            try:
                return render_compact(services, PRE_HOLDOUT, version=version)
            except PersonaVersionError as exc:
                raise LockedVersionError(
                    f"persona card {version} of the model is not in the store"
                ) from exc

        return cls(
            locked=locked,
            render_persona=render_locked,
            memory_tokens=(
                memory_tokens
                if memory_tokens is not None
                else services.settings.style_model.memory_tokens
            ),
        )

    @property
    def locked(self) -> LockedVersions | None:
        return self._locked

    # ------------------------------------------------------------------------- pieces

    def persona(self, data: ReplyDataView) -> RenderedPersona | None:
        """The compact card to render: the locked version, else the data view's."""
        if self._locked is not None and self._render_persona is not None:
            return self._render_persona(self._locked.persona_version)
        return data.persona_compact()

    def memory_for(self, data: ReplyDataView, turns: Sequence[StyleTurn]) -> MemoryBlock:
        """The memory block for the conversation ``turns`` (the topic is the last three turns)."""
        if self._memory_tokens <= 0:
            return MemoryBlock("")
        normalized = normalize_context(turns, max_turns=max(self._max_turns, TOPIC_TURNS + 1))
        recent = [*normalized.prelude, *(turn.content for turn in normalized.turns)]
        topic = "\n".join(recent[-(TOPIC_TURNS + 1) :])
        return data.memory_block(MemoryQuery(text=topic), self._memory_tokens)

    @staticmethod
    def _now_section(data: ReplyDataView, *, woke_up: bool) -> str:
        lines = [NOW_HEADING, f"当地时间：{moment_text(data.local)}"]
        state = data.her_state()
        if state in STATE_LINES:
            lines.append(f"她现在的状态：{STATE_LINES[state]}")
        if woke_up:
            lines.append(WOKE_UP_LINE)
        return "\n".join(lines)

    @staticmethod
    def _system(
        persona: str,
        now: str,
        memory: str,
        plan: str,
        prelude: Sequence[str],
    ) -> str:
        parts = [persona, now, memory]
        if plan:
            parts.append(f"{PLAN_HEADING}\n{plan}")
        if prelude:
            parts.append(PRELUDE_HEADING + "\n" + "\n".join(f"她：{text}" for text in prelude))
        joined = "\n\n".join(part.strip() for part in (scrub(p) for p in parts) if part.strip())
        return _BLANK_LINES.sub("\n\n", joined)

    # --------------------------------------------------------------------------- build

    def compose(
        self,
        data: ReplyDataView,
        turns: Sequence[StyleTurn],
        *,
        plan: PlanFields | None = None,
        woke_up: bool = False,
        budget: TokenBudget | None = None,
        memory: MemoryBlock | None = None,
    ) -> PromptParts:
        """The system segment and the conversation segment of one prompt.

        ``turns`` is the conversation up to and including the user's turn that is answered (the
        newest last); ``memory`` is the memory block when the caller already fetched it
        (:meth:`memory_for`), else it is fetched here.  ``budget`` makes the prompt fit by
        dropping the oldest context (see the module description).
        """
        context = normalize_context(turns, max_turns=self._max_turns)
        if not context.turns:
            raise StylePromptError("a prompt needs the user's turn that is answered")
        if not context.ends_with_user:
            raise StylePromptError("the context must end with the user's turn that is answered")
        card = self.persona(data)
        block = memory if memory is not None else self.memory_for(data, turns)
        plan_text = render_plan(plan) if plan is not None else ""
        now = self._now_section(data, woke_up=woke_up)
        persona_text = card.text if card is not None else ""

        kept, prelude, trimmed = list(context.turns), list(context.prelude), 0
        while True:
            system = self._system(persona_text, now, block.text, plan_text, prelude)
            text = lf_template.render_prompt(system, kept)
            if budget is None or budget.fits(text):
                over = False
                break
            if prelude:  # the oldest content of all goes first
                trimmed += len(prelude)
                prelude = []
            elif len(kept) > 1:
                del kept[0]
                trimmed += 1
                if kept[0].role == "assistant":  # the context must still open with the user
                    del kept[0]
                    trimmed += 1
            else:
                over = True
                break
        meta = PromptMeta(
            template_version=lf_template.TEMPLATE_VERSION,
            persona_scope=card.scope if card is not None else None,
            persona_version=f"v{card.number}" if card is not None else None,
            persona_id=card.version_id if card is not None else None,
            locked=self._locked is not None,
            context_turns=len(kept),
            prelude_turns=len(prelude),
            trimmed_turns=trimmed,
            over_budget=over,
            has_plan=bool(plan_text),
            memory_item_ids=tuple(item.item_id for item in block.items),
        )
        return PromptParts(system, tuple(kept), meta, block)

    def build(
        self,
        data: ReplyDataView,
        turns: Sequence[StyleTurn],
        *,
        plan: PlanFields | None = None,
        woke_up: bool = False,
        budget: TokenBudget | None = None,
        memory: MemoryBlock | None = None,
    ) -> RenderedPrompt:
        """The prompt string of the style model: ``compose(...).render()``."""
        return self.compose(
            data, turns, plan=plan, woke_up=woke_up, budget=budget, memory=memory
        ).render()
