"""What the M0 probe learned about the API (R-LLM-004, R-LLM-013).

The documentation says the API accepts ``detail`` and animated GIFs and that thinking mode
can return JSON; the probe checks it for real and stores the outcome here.  Until a probe has
run, the documented behaviour is assumed.  The values change what the client sends:

* ``detail_supported`` false - :class:`~twin.llm.images.ImageInput` stops sending ``detail``;
* ``gif_supported`` false - a GIF is reduced to its first frame (as PNG) before sending;
* ``json_in_thinking`` false - structured calls with thinking enabled are refused by
  :meth:`DeepSeekClient.chat_json` instead of failing at random (the proactive planner needs
  to know);
* ``image_tokens`` - measured tokens per image size, used by the estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from twin.clock import Clock
from twin.llm.tokens import ImageTokenTable
from twin.storage.db import Database
from twin.storage.settings_store import get_setting, put_setting

CAPABILITIES_KEY = "llm.capabilities"
CACHE_TTL_S = 30.0


@dataclass(frozen=True)
class LlmCapabilities:
    detail_supported: bool = True
    gif_supported: bool = True
    json_in_thinking: bool = True
    image_tokens: tuple[tuple[int, int], ...] = ()
    measured_at: str | None = None  # ISO time of the probe run, None while assumed

    @property
    def measured(self) -> bool:
        return self.measured_at is not None

    def image_table(self) -> ImageTokenTable:
        return ImageTokenTable(self.image_tokens)

    def to_json(self) -> dict[str, object]:
        return {
            "detail_supported": self.detail_supported,
            "gif_supported": self.gif_supported,
            "json_in_thinking": self.json_in_thinking,
            "image_tokens": [[pixels, tokens] for pixels, tokens in self.image_tokens],
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_json(cls, raw: object) -> LlmCapabilities:
        """Parse a stored value; anything malformed falls back to the documented defaults."""
        if not isinstance(raw, dict):
            return cls()
        try:
            tokens = tuple(
                (int(pixels), int(count)) for pixels, count in raw.get("image_tokens", [])
            )
            measured_at = raw.get("measured_at")
            return cls(
                detail_supported=bool(raw.get("detail_supported", True)),
                gif_supported=bool(raw.get("gif_supported", True)),
                json_in_thinking=bool(raw.get("json_in_thinking", True)),
                image_tokens=tokens,
                measured_at=str(measured_at) if measured_at else None,
            )
        except (TypeError, ValueError):
            return cls()


DOCUMENTED = LlmCapabilities()


def save_capabilities(session: Session, caps: LlmCapabilities, clock: Clock) -> bool:
    """Store ``caps``; returns ``True`` if the stored value changed."""
    return put_setting(
        session, CAPABILITIES_KEY, caps.to_json(), clock=clock, by="llm_probe", record_history=True
    )


def load_capabilities(session: Session) -> LlmCapabilities:
    return LlmCapabilities.from_json(get_setting(session, CAPABILITIES_KEY, None))


class CapabilityStore:
    """Cached access for a long-running process (refreshed every :data:`CACHE_TTL_S`)."""

    def __init__(self, db: Database, clock: Clock, *, ttl_s: float = CACHE_TTL_S) -> None:
        self._db = db
        self._clock = clock
        self._ttl_s = ttl_s
        self._value: LlmCapabilities | None = None
        self._loaded_at = 0.0

    def get(self) -> LlmCapabilities:
        now = self._clock.monotonic()
        if self._value is None or now - self._loaded_at >= self._ttl_s:
            with self._db.session() as session:
                self._value = load_capabilities(session)
            self._loaded_at = now
        return self._value

    def invalidate(self) -> None:
        self._value = None
