"""The post-processing pipeline: raw model output in, bubbles and an audit trail out (R-ENG-008).

:class:`PostProcessor` runs the steps of :mod:`twin.engine.postprocess.steps` in the order of the
requirement on the parsed output and returns a :class:`PostResult`.  It is pure with respect to the
conversation: everything it needs is in the :class:`~twin.engine.postprocess.model.PostContext`
(her style numbers, the vocabularies, the sticker selector and share controller, the channel's
abilities), so the live engine and the evaluation sandbox use the same code.  A result with
violations is not to be sent - the pipeline asks the model again (``twin.engine.pipeline``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from twin.engine.parsing import parse_reply
from twin.engine.postprocess.model import Line, PostContext, Working
from twin.engine.postprocess.quota import fit_bubbles_to_quota
from twin.engine.postprocess.steps import STEPS, Step, remove_think
from twin.engine.types import Bubble, PostAction, Violation


@dataclass(frozen=True)
class PostResult:
    """What the post-processing made of one output."""

    bubbles: tuple[Bubble, ...]
    quote: str | None
    no_reply: bool
    actions: tuple[PostAction, ...]
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def to_bubbles(work: Working) -> list[Bubble]:
    """The lines as bubbles; the quote goes with the first text bubble."""
    bubbles: list[Bubble] = []
    quote = work.quote
    for line in work.lines:
        if line.kind == "sticker":
            bubbles.append(
                Bubble(
                    "sticker", f"[表情包:{line.text}]", sticker_md5=line.md5, sticker_tag=line.text
                )
            )
        elif line.kind == "text":
            bubbles.append(Bubble("text", line.text, quote=quote))
            quote = None
    return bubbles


class PostProcessor:
    """Runs the steps in order (see :mod:`twin.engine.postprocess.steps`)."""

    def __init__(self, steps: Sequence[tuple[str, Step]] = STEPS) -> None:
        self._steps = tuple(steps)

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._steps)

    def process(self, raw: str, ctx: PostContext) -> PostResult:
        text, thoughts = remove_think(raw)
        parsed = parse_reply(text)
        work = Working(parsed.quote, [Line(line.kind, line.text) for line in parsed.lines])
        if thoughts:
            work.act("think_removed", thoughts)
            work.violate("think_tag")
        work.act("stray_quote_removed", parsed.stray_quotes)
        work.act("speaker_label_removed", parsed.labels_removed)
        for _, step in self._steps:
            step(work, ctx)
        bubbles = to_bubbles(work)
        actions = list(work.actions)
        if ctx.quota is not None and len(bubbles) > max(1, ctx.quota):
            bubbles, quota_actions = fit_bubbles_to_quota(bubbles, ctx.quota)
            actions.extend(quota_actions)
        return PostResult(
            tuple(bubbles), work.quote, work.no_reply, tuple(actions), tuple(work.violations)
        )

    def with_quota(self, result: PostResult, available: int) -> PostResult:
        """``result`` fitted into ``available`` bubbles (the engine calls this when it sends)."""
        bubbles, actions = fit_bubbles_to_quota(result.bubbles, available)
        return replace(result, bubbles=tuple(bubbles), actions=(*result.actions, *actions))
