"""The WeChat emoji codes she uses (``[拥抱]``, ``[亲亲]``, ...) and how often (R-STK-001).

:class:`EmojiCodePolicy` is built from her profile (R-PROF-002): the codes she actually used,
their frequencies, the share of her text messages that carry a code, and how many codes she
puts in a row.  The reply generator (round 09) uses it to

* delete every code that is not in her vocabulary (:meth:`strip_disallowed`) - the model may
  not invent ``[旺柴]`` if she never wrote it,
* keep the frequency near hers (:meth:`expected_rate`),
* and draw how many codes go in a row (:meth:`sample_repeat`).

Two kinds of words are never in the vocabulary.  A bracket text that is not a WeChat emoji code
of the table ``profile.emoji_codes_file`` is not an emoji (it is just text she typed in
brackets).  And a code that is also the *event text* of a message type - ``[红包]`` is the event
text of a red packet - is excluded (D-138): a line like that is deleted as an event text
(R-SAFE-006) and must not come back as an "emoji".

The profile is read per scope: ``live`` for the running bot, ``pre_holdout`` for the training
export and the evaluation sandbox (R-TRN-013).
"""

from __future__ import annotations

import random
import re
from collections.abc import Collection, Mapping

from twin.config.lists import load_word_list, locate_list_file
from twin.ingest.events import EventTextDetector, default_detector
from twin.profile.api import load_profile
from twin.profile.distribution import EmpiricalDistribution
from twin.profile.textstats import EMOJI_CODE_RE
from twin.services import Services

_SPACES = re.compile(r"[ \t]{2,}")


def bare(code: str) -> str:
    """``拥抱`` for ``[拥抱]`` or ``拥抱``."""
    text = code.strip()
    return text[1:-1] if text.startswith("[") and text.endswith("]") else text


def bracketed(code: str) -> str:
    return f"[{bare(code)}]"


class EmojiCodePolicy:
    """Which emoji codes may appear, how often, and how many in a row."""

    def __init__(
        self,
        frequencies: Mapping[str, float],
        *,
        rate: float,
        run_lengths: EmpiricalDistribution | None = None,
        known: Collection[str] | None = None,
        detector: EventTextDetector = default_detector,
    ) -> None:
        allowed: dict[str, float] = {}
        for code, share in frequencies.items():
            name = bare(code)
            if share <= 0 or (known is not None and name not in known):
                continue
            if detector.is_event_text(bracketed(name)):
                continue
            allowed[name] = allowed.get(name, 0.0) + share
        self._frequencies = dict(sorted(allowed.items(), key=lambda item: (-item[1], item[0])))
        self._rate = max(0.0, min(1.0, rate)) if self._frequencies else 0.0
        self._runs = run_lengths

    @classmethod
    def from_profile(cls, services: Services, scope: str = "live") -> EmojiCodePolicy | None:
        """The policy of her profile in ``scope``; ``None`` if no profile has been computed."""
        profile = load_profile(services, scope)
        if profile is None:
            return None
        settings = services.settings.profile
        known = load_word_list(locate_list_file(services.paths.root, settings.emoji_codes_file))
        return cls(
            profile.metrics.rates("her", "emoji_code_freq"),
            rate=profile.metrics.scalar("her", "emoji_code_rate") or 0.0,
            run_lengths=profile.metrics.distribution("her", "emoji_code_run_length"),
            known=known,
        )

    # ------------------------------------------------------------------ queries

    @property
    def codes(self) -> tuple[str, ...]:
        """Her codes with brackets, most frequent first."""
        return tuple(bracketed(name) for name in self._frequencies)

    def is_allowed(self, code: str) -> bool:
        """Is this code (with or without brackets) in her vocabulary?"""
        return bare(code) in self._frequencies

    def frequency(self, code: str) -> float:
        """The share of her codes that this one makes (0 if she never used it)."""
        total = sum(self._frequencies.values())
        return self._frequencies.get(bare(code), 0.0) / total if total else 0.0

    def expected_rate(self) -> float:
        """The share of her text messages that carry at least one emoji code."""
        return self._rate

    def sample_repeat(self, rng: random.Random) -> int:
        """How many codes in a row (at least 1), drawn from her distribution."""
        if self._runs is None or self._runs.is_empty:
            return 1
        return max(1, round(self._runs.sample(rng)))

    def sample_code(self, rng: random.Random) -> str | None:
        """One of her codes, drawn by frequency (``None`` if she uses none)."""
        if not self._frequencies:
            return None
        names = list(self._frequencies)
        return bracketed(rng.choices(names, weights=list(self._frequencies.values()), k=1)[0])

    # ------------------------------------------------------------- post-processing

    def strip_disallowed(self, text: str) -> str:
        """``text`` without the bracket codes that are not in her vocabulary."""

        def keep(match: re.Match[str]) -> str:
            return match.group(0) if self.is_allowed(match.group(0)) else ""

        stripped = EMOJI_CODE_RE.sub(keep, text)
        return _SPACES.sub(" ", stripped).strip(" \t") if stripped != text else text
