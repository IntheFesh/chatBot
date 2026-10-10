"""More doubles for the engine tests: routine model, command router, stickers, restart."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from tests.fixtures.synth_export import make_image_bytes
from tests.support.engine_harness import Harness, build_harness
from twin.profile.distribution import EmpiricalDistribution
from twin.services import Services
from twin.stickers.catalog import StickerCatalog, StickerRecord
from twin.stickers.library import store_sticker_file
from twin.storage.chat_models import Sticker


@dataclass
class Window:
    """A busy window of the routine model: only its latency distribution matters here."""

    latency: EmpiricalDistribution


class Routine:
    """What the pacing asks of the routine model: busy windows, slot latency, rate, state."""

    def __init__(
        self, busy: list[Window] | None = None, rates: dict[int, float] | None = None
    ) -> None:
        self.busy = busy or []
        self.rates = rates or {}
        self.calls: list[tuple[str, int | None]] = []

    def busy_windows(self, day_type: str, weekday: int | None = None) -> list[Window]:
        self.calls.append((day_type, weekday))
        return self.busy

    def latency_distribution(self, slot: int) -> EmpiricalDistribution:
        return EmpiricalDistribution.empty()

    def rate_at(self, slot: int, day_type: str) -> float:
        return self.rates.get(slot, 1.0)

    def typical_state(self, local_time: Any, day_type: str) -> str:
        return "deep_sleep" if local_time.hour < 7 else "free"


class ScriptedCommands:
    """A command router of a test: ``replies`` maps a command (no slash) to its outcome."""

    def __init__(self, **replies: Any) -> None:
        self.replies = {f"/{name}": outcome for name, outcome in replies.items()}
        self.seen: list[tuple[str, Any]] = []
        self.before: Callable[[str, Any], None] | None = None
        self.error: Exception | None = None

    async def handle(self, text: str, context: Any) -> Any:
        self.seen.append((text, context))
        if self.before is not None:
            self.before(text, context)
        if self.error is not None:
            raise self.error
        return self.replies.get(text.replace("／", "/").strip())


PICTURES: dict[str, bytes] = {}  # the bytes of the stickers added by ``add_sticker``, by MD5


def add_sticker(
    services: Services, seed: int = 1, *, tags: Sequence[str] = ("开心",)
) -> StickerRecord:
    """A sticker of the library with a real picture."""
    data = make_image_bytes(random.Random(seed), "PNG")
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    with services.db.transaction() as session:
        row = Sticker(md5=md5, status="pending", attempts=0, her_uses=3, user_uses=0)
        session.add(row)
        session.flush()
        store_sticker_file(row, data, services.media, services.clock.now_utc())
        row.tags = list(tags)
    record = StickerCatalog(services).get(md5)
    assert record is not None and record.available
    PICTURES[md5] = data
    return record


async def restart(harness: Harness, *, clock_jump_s: float = 0.0, **changes: Any) -> Harness:
    """Stop the engine as a kill would (the state stays on disk) and start a new one on it."""
    await harness.engine.stop()
    if clock_jump_s:
        harness.clock.tick(clock_jump_s)
    options: dict[str, Any] = {
        "day": harness.day,
        "crisis": harness.crisis,
        "commands": harness.commands,
        **changes,
    }
    fresh = build_harness(harness.services, harness.clock, **options)
    await fresh.engine.start()
    return fresh
