"""Local token estimation with self-calibration (R-LLM-012).

The estimator counts characters by class - CJK characters, Latin words, digit runs,
punctuation, other symbols - and images, then multiplies by a calibration factor learned from
the ``usage`` the API reports (an exponential moving average of ``actual / estimated``).  The
ratios per character class come from the token page of the DeepSeek documentation (checked
2026-10-09): about 0.6 tokens per Chinese character and 0.3 per English character.  The factor
is persisted in the ``settings`` table so a restart does not lose what was learned.

Images are estimated separately: the documented cap of 1024 tokens per image, or the measured
values of the M0 probe (:class:`ImageTokenTable`) once they exist.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.llm.official import MAX_IMAGE_TOKENS, TOKENS_PER_CJK_CHAR, TOKENS_PER_LATIN_CHAR
from twin.llm.types import ChatMessage, ContentPart
from twin.storage.settings_store import get_setting, put_setting

CALIBRATION_KEY = "llm.token_calibration"
MESSAGE_OVERHEAD_TOKENS = 3.0
TOKENS_PER_DIGIT = 0.5
TOKENS_PER_PUNCTUATION = 1.0
TOKENS_PER_OTHER_CHAR = 1.5
MIN_FACTOR, MAX_FACTOR = 0.3, 3.0
MIN_RATIO, MAX_RATIO = 0.2, 5.0

_CJK = r"㐀-䶿一-鿿豈-﫿぀-ヿ가-힯\U00020000-\U0002fa1f"
_TOKEN_RE = re.compile(
    rf"(?P<cjk>[{_CJK}])"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<digits>[0-9]+)"
    r"|(?P<space>\s+)"
    r"|(?P<punct>[!-/:-@[-`{-~"
    r"　-〿＀-￯ -⁯])"
    r"|(?P<other>.)",
    re.DOTALL,
)


@dataclass(frozen=True)
class Calibration:
    factor: float = 1.0
    samples: int = 0


class ImageTokenTable:
    """Measured tokens per image for a few sizes; nearest size (by pixel count) wins."""

    def __init__(self, entries: Sequence[tuple[int, int]] = ()) -> None:
        self._entries = sorted(
            (int(pixels), int(tokens)) for pixels, tokens in entries if pixels > 0
        )

    def __bool__(self) -> bool:
        return bool(self._entries)

    def tokens_for(self, width: int | None, height: int | None) -> int:
        """Tokens for an image of the given size; the documented cap when nothing is known."""
        if not self._entries or not width or not height:
            return MAX_IMAGE_TOKENS
        pixels = width * height
        nearest = min(
            self._entries, key=lambda entry: abs(math.log(max(entry[0], 1) / max(pixels, 1)))
        )
        return min(MAX_IMAGE_TOKENS, nearest[1])


class TokenEstimator:
    """Estimates tokens of text and messages and learns a correction factor."""

    def __init__(
        self,
        *,
        calibration: Calibration | None = None,
        alpha: float = 0.1,
        image_tokens: ImageTokenTable | None = None,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._calibration = calibration or Calibration()
        self._alpha = alpha
        self._images = image_tokens or ImageTokenTable()

    # ---------------------------------------------------------------- estimation

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def factor(self) -> float:
        return self._calibration.factor

    def set_image_table(self, table: ImageTokenTable) -> None:
        self._images = table

    def estimate_image(self, width: int | None = None, height: int | None = None) -> int:
        return self._images.tokens_for(width, height)

    @staticmethod
    def raw_text(text: str) -> float:
        """Uncalibrated estimate of ``text``."""
        total = 0.0
        for match in _TOKEN_RE.finditer(text):
            kind = match.lastgroup
            piece = match.group(0)
            if kind == "cjk":
                total += TOKENS_PER_CJK_CHAR
            elif kind == "word":
                total += max(1.0, len(piece) * TOKENS_PER_LATIN_CHAR)
            elif kind == "digits":
                total += max(1.0, len(piece) * TOKENS_PER_DIGIT)
            elif kind == "punct":
                total += TOKENS_PER_PUNCTUATION
            elif kind == "other":
                total += TOKENS_PER_OTHER_CHAR
        return total

    def estimate_text(self, text: str) -> int:
        """Calibrated token estimate of ``text`` (at least 1 for non-empty text)."""
        if not text:
            return 0
        return max(1, math.ceil(self.raw_text(text) * self.factor))

    def _split(self, messages: Sequence[ChatMessage]) -> tuple[float, int]:
        """``(uncalibrated text tokens, image tokens)`` of a message list."""
        text = 0.0
        images = 0
        for message in messages:
            text += MESSAGE_OVERHEAD_TOKENS
            content = message["content"]
            if isinstance(content, str):
                text += self.raw_text(content)
                continue
            for part in content:
                text_part, image_part = self._part_tokens(part)
                text += text_part
                images += image_part
        return text, images

    def _part_tokens(self, part: ContentPart) -> tuple[float, int]:
        if part["type"] == "text":
            return self.raw_text(part["text"]), 0
        return 0.0, self.estimate_image()

    def estimate_messages(self, messages: Sequence[ChatMessage]) -> int:
        """Calibrated estimate of the prompt tokens of a message list, images included."""
        text, images = self._split(messages)
        return math.ceil(text * self.factor) + images

    # --------------------------------------------------------------- calibration

    def observe(self, messages: Sequence[ChatMessage], actual_prompt_tokens: int) -> Calibration:
        """Learn from a real call: ``actual_prompt_tokens`` is ``usage.prompt_tokens``."""
        text, images = self._split(messages)
        actual_text = actual_prompt_tokens - images
        if text <= 0 or actual_text <= 0:
            return self._calibration
        ratio = min(MAX_RATIO, max(MIN_RATIO, actual_text / text))
        if self._calibration.samples == 0:
            factor = ratio
        else:
            factor = (1 - self._alpha) * self._calibration.factor + self._alpha * ratio
        factor = min(MAX_FACTOR, max(MIN_FACTOR, factor))
        self._calibration = Calibration(factor, self._calibration.samples + 1)
        return self._calibration

    # --------------------------------------------------------------- persistence

    def save(self, session: Session, clock: Clock) -> bool:
        """Store the calibration in ``settings``; returns ``True`` if it changed."""
        value = {"factor": round(self._calibration.factor, 6), "samples": self._calibration.samples}
        return put_setting(
            session, CALIBRATION_KEY, value, clock=clock, by="llm", record_history=False
        )

    def load(self, session: Session) -> Calibration:
        """Restore the stored calibration (the default when none or invalid)."""
        raw = get_setting(session, CALIBRATION_KEY, None)
        if isinstance(raw, dict):
            try:
                factor = float(raw["factor"])
                samples = int(raw["samples"])
            except (KeyError, TypeError, ValueError):
                return self._calibration
            if MIN_FACTOR <= factor <= MAX_FACTOR and samples >= 0:
                self._calibration = Calibration(factor, samples)
        return self._calibration
