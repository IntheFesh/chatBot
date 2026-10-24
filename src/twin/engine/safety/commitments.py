"""Promises of things the bot cannot do (R-SAFE-002).

The bot lives in a chat window.  It cannot phone, video-call, send a voice message or a photo,
meet anybody, transfer money, send a red packet or post a parcel, so it must never say it will.
``config/lists/commitment_patterns.txt`` (``engine.commitment_patterns_file``) holds regular
expressions for the typical wordings; a bubble that matches is a violation: the engine asks the
model for the reply again with a note (R-ENG-008), and as a last resort the bubble is removed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from twin.config.lists import load_regex_list, locate_list_file

if TYPE_CHECKING:
    from twin.services import Services


class CommitmentDetector:
    """Finds sentences that promise a real-world action."""

    def __init__(self, patterns: Sequence[re.Pattern[str]]) -> None:
        self._patterns = tuple(patterns)

    @classmethod
    def from_file(cls, path: Path) -> CommitmentDetector:
        return cls(load_regex_list(path))

    @classmethod
    def from_services(cls, services: Services) -> CommitmentDetector:
        return cls.from_file(
            locate_list_file(services.paths.root, services.settings.engine.commitment_patterns_file)
        )

    def __len__(self) -> int:
        return len(self._patterns)

    def find(self, text: str) -> int | None:
        """Index (line of the list) of the first pattern ``text`` matches, else ``None``."""
        for index, pattern in enumerate(self._patterns):
            if pattern.search(text):
                return index
        return None

    def promises(self, text: str) -> bool:
        return self.find(text) is not None
