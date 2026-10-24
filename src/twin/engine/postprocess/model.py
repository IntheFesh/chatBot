"""The working data of the post-processing: lines, what was done to them, what they violated."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from twin.engine.parsing import LineKind
from twin.engine.postprocess.phrases import AiPhrases
from twin.engine.postprocess.style import StyleLimits
from twin.engine.safety.commitments import CommitmentDetector
from twin.engine.types import PostAction, Violation
from twin.ingest.events import EventTextDetector, default_detector
from twin.stickers.catalog import StickerRecord
from twin.stickers.emoji_codes import EmojiCodePolicy
from twin.stickers.rate import StickerRateController

StickerChooser = Callable[[str, str, Sequence[str | None]], StickerRecord | None]


@dataclass
class Line:
    """One line of the reply while it is being worked on."""

    kind: LineKind
    text: str  # text: the line; sticker: the tag that was asked for
    md5: str | None = None  # a sticker line: the sticker it was resolved to
    promises: bool = False  # a text line that promises something the bot cannot do


@dataclass
class Working:
    """The reply as the steps see it, with the audit trail they leave."""

    quote: str | None
    lines: list[Line]
    actions: list[PostAction] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    no_reply: bool = False

    def act(self, step: str, count: int = 1, detail: str | None = None) -> None:
        if count > 0:
            self.actions.append(PostAction(step, count, detail))

    def violate(self, kind: str, detail: str | None = None) -> None:
        if all(found.kind != kind for found in self.violations):
            self.violations.append(Violation(kind, detail))

    def text_lines(self) -> list[Line]:
        return [line for line in self.lines if line.kind == "text"]


@dataclass
class PostContext:
    """Everything the steps need besides the lines (built once per reply)."""

    style: StyleLimits
    emoji: EmojiCodePolicy | None
    ai_phrases: AiPhrases
    commitments: CommitmentDetector | None
    rng: random.Random
    detector: EventTextDetector = default_detector
    allow_ai_admission: bool = False  # the user sincerely asked whether she is an AI (R-SAFE-003)
    user_text: str = ""  # what the user said this round: words it already uses are not a slip
    supports_quote: bool = True
    quota: int | None = None  # the bubbles the platform quota leaves for this reply
    no_reply_allowed: bool = False  # the user's last message is a closing one and she does this
    known_tags: Sequence[str] = ()  # the sticker tag vocabulary; empty: not checked
    chooser: StickerChooser | None = None
    rate: StickerRateController | None = None
    sticker_context: str = ""  # the text a sticker is matched against (the conversation's end)
    recent_stickers: Sequence[str | None] = ()
    last_attempt: bool = False  # no regeneration is left: a promise is cut out instead
