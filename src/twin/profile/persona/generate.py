"""Writing the automatic description of the persona card: Map-Reduce with checked evidence.

**Map.**  The sampled segments go to DeepSeek ten at a time (``persona.batch_segments``) with the
``persona_map`` template; each reply is a :class:`Digest` - tone, catchphrases, forms of
address, topics, what she says in six moods, her attitude, what she does not say, facts about
her - and every statement names the segments that show it.

**Reduce.**  The cleaned digests are merged by one call with the ``persona_reduce`` template,
which may only combine what the digests already say.

**Evidence.**  The model is asked for evidence, the program *checks* it (R-PERS-001): a segment
number is accepted only if it is one of the numbers shown in that call; a statement without a
single accepted number is dropped, and so is a mood outside the six.  What survives is written
into ``[自动-描述]`` and its evidence is kept in the version's provenance record.  So no fact
reaches the card without a segment of the real record behind it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import DAILY, LedgerTag, Purpose
from twin.profile.persona.sampling import Sample, SampledSegment
from twin.profile.persona.sections import (
    BASICS,
    EMOTIONS,
    LABEL_ADDRESS,
    LABEL_ATTITUDE,
    LABEL_CATCHPHRASE,
    LABEL_FACT,
    LABEL_TABOO,
    LABEL_TONE,
    LABEL_TOPIC,
    STYLE,
    subsection,
)
from twin.profile.prompt_templates import PromptText

MAX_ITEMS = 8
ITEM_CHARS = 160
_SEGMENT_NUMBER = re.compile(r"^S0*(\d+)$", re.IGNORECASE)


class Item(BaseModel):
    """One statement and the segments that show it."""

    model_config = ConfigDict(extra="ignore")

    text: str = ""
    evidence: list[str] = Field(default_factory=list)


class EmotionItem(Item):
    """What she says in one mood."""

    emotion: str = ""


class Digest(BaseModel):
    """What the model reports about her (the reply format of both calls)."""

    model_config = ConfigDict(extra="ignore")

    tone: list[Item] = Field(default_factory=list)
    catchphrases: list[Item] = Field(default_factory=list)
    address_terms: list[Item] = Field(default_factory=list)
    topics: list[Item] = Field(default_factory=list)
    emotions: list[EmotionItem] = Field(default_factory=list)
    attitude: list[Item] = Field(default_factory=list)
    taboos: list[Item] = Field(default_factory=list)
    facts: list[Item] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return sum(len(getattr(self, name)) for name in LIST_FIELDS)


LIST_FIELDS = (
    "tone",
    "catchphrases",
    "address_terms",
    "topics",
    "emotions",
    "attitude",
    "taboos",
    "facts",
)


class PersonaGenerationError(RuntimeError):
    """The description could not be produced (nothing usable came back)."""


# ----------------------------------------------------------------------- evidence


def canonical_label(value: str) -> str | None:
    """``S3`` / ``s03`` / ``S003`` as the number it names; ``None`` if it is not a segment label."""
    match = _SEGMENT_NUMBER.match(value.strip())
    return str(int(match.group(1))) if match else None


@dataclass(frozen=True)
class CleanResult:
    digest: Digest
    dropped: int  # statements removed because no valid evidence was left


def clean_digest(digest: Digest, valid_labels: Iterable[str]) -> CleanResult:
    """Keep only the statements with evidence among ``valid_labels`` (and valid moods)."""
    valid = {}
    for label in valid_labels:
        number = canonical_label(label)
        if number is not None:
            valid[number] = label
    dropped = 0

    def keep(item: Item) -> Item | None:
        nonlocal dropped
        text = " ".join(item.text.split())
        accepted: list[str] = []
        for raw in item.evidence:
            number = canonical_label(raw)
            if number is not None and number in valid and valid[number] not in accepted:
                accepted.append(valid[number])
        if not text or not accepted:
            dropped += 1
            return None
        return item.model_copy(update={"text": text, "evidence": accepted})

    cleaned: dict[str, list[Any]] = {}
    for name in LIST_FIELDS:
        kept: list[Item] = []
        for item in getattr(digest, name):
            if isinstance(item, EmotionItem) and item.emotion.strip().rstrip("时") not in EMOTIONS:
                dropped += 1
                continue
            result = keep(item)
            if result is not None:
                if isinstance(result, EmotionItem):
                    result = result.model_copy(
                        update={"emotion": result.emotion.strip().rstrip("时")}
                    )
                kept.append(result)
        cleaned[name] = kept
    return CleanResult(Digest(**cleaned), dropped)


def merge_digests(digests: Sequence[Digest]) -> Digest:
    """All statements of several digests in one (used when there is nothing to reduce)."""
    combined: dict[str, list[Any]] = {name: [] for name in LIST_FIELDS}
    for digest in digests:
        for name in LIST_FIELDS:
            combined[name].extend(getattr(digest, name))
    return Digest(**combined)


# ----------------------------------------------------------------------- prompts


def segments_text(segments: Sequence[SampledSegment]) -> str:
    return "\n\n".join(segment.render() for segment in segments)


def digest_json(digest: Digest) -> str:
    return json.dumps(digest.model_dump(), ensure_ascii=False, separators=(",", ":"))


def map_messages(template: PromptText, segments: Sequence[SampledSegment]) -> Any:
    return template.render(count=len(segments), segments=segments_text(segments))


def reduce_messages(template: PromptText, digests: Sequence[Digest]) -> Any:
    items = "\n".join(f"第 {i} 份：{digest_json(d)}" for i, d in enumerate(digests, start=1))
    return template.render(count=len(digests), items=items, max_items=MAX_ITEMS)


def batches_of(segments: Sequence[SampledSegment], size: int) -> list[list[SampledSegment]]:
    return [list(segments[start : start + size]) for start in range(0, len(segments), size)]


# ----------------------------------------------------------------------- the run


@dataclass
class GenerationResult:
    """What the Map-Reduce produced."""

    digest: Digest
    dropped: int
    cost_usd: float
    map_calls: int
    batches: list[list[str]] = field(default_factory=list)  # the labels of each map call


async def generate_digest(
    client: DeepSeekClient,
    sample: Sample,
    *,
    map_template: PromptText,
    reduce_template: PromptText,
    batch_size: int,
    tag: LedgerTag = DAILY,
    on_progress: Callable[[str], None] | None = None,
) -> GenerationResult:
    """Map every batch of segments, check the evidence, then reduce (R-PERS-001)."""
    if not sample.segments:
        raise PersonaGenerationError("there are no conversation segments to describe her from")
    batches = batches_of(sample.segments, batch_size)
    cleaned: list[Digest] = []
    cost = 0.0
    dropped = 0
    for number, batch in enumerate(batches, start=1):
        reply = await client.chat_json(
            map_messages(map_template, batch),
            Digest,
            purpose=Purpose.PERSONA,
            tag=tag,
            temperature=0.2,
        )
        cost += reply.total_cost_usd
        result = clean_digest(reply.value, [segment.label for segment in batch])
        dropped += result.dropped
        cleaned.append(result.digest)
        if on_progress is not None:
            on_progress(f"map {number}/{len(batches)}")
    usable = [digest for digest in cleaned if digest.count]
    if not usable:
        raise PersonaGenerationError("no statement of the model came with valid evidence")
    labels = [segment.label for segment in sample.segments]
    if len(usable) == 1:
        final = usable[0]
    else:
        merged = await client.chat_json(
            reduce_messages(reduce_template, usable),
            Digest,
            purpose=Purpose.PERSONA,
            tag=tag,
            temperature=0.2,
        )
        cost += merged.total_cost_usd
        reduced = clean_digest(merged.value, labels)
        dropped += reduced.dropped
        final = reduced.digest
        if not final.count:
            raise PersonaGenerationError("the merged description kept no statement with evidence")
        if on_progress is not None:
            on_progress("reduce")
    return GenerationResult(
        final, dropped, cost, len(batches), [[s.label for s in b] for b in batches]
    )


# ----------------------------------------------------------------------- the text


def _line(label: str, text: str) -> str:
    shortened = text if len(text) <= ITEM_CHARS else text[: ITEM_CHARS - 1] + "…"
    return f"- {label}：{shortened}"


def _unique(items: Sequence[Item]) -> list[Item]:
    seen: set[str] = set()
    out: list[Item] = []
    for item in items:
        if item.text not in seen:
            seen.add(item.text)
            out.append(item)
    return out[:MAX_ITEMS]


def description_block(digest: Digest) -> tuple[str, list[dict[str, Any]]]:
    """The body of ``[自动-描述]`` and the statements in it with their evidence."""
    style: list[str] = []
    basics: list[str] = []
    statements: list[dict[str, Any]] = []

    def add(target: list[str], part: str, label: str, items: Sequence[Item]) -> None:
        for item in _unique(items):
            target.append(_line(label, item.text))
            statements.append(
                {"part": part, "label": label, "text": item.text, "evidence": item.evidence}
            )

    add(style, STYLE, LABEL_TONE, digest.tone)
    add(style, STYLE, LABEL_CATCHPHRASE, digest.catchphrases)
    add(style, STYLE, LABEL_ADDRESS, digest.address_terms)
    for emotion in EMOTIONS:
        add(style, STYLE, f"{emotion}时", [e for e in digest.emotions if e.emotion == emotion])
    add(style, STYLE, LABEL_TABOO, digest.taboos)
    add(basics, BASICS, LABEL_FACT, digest.facts)
    add(basics, BASICS, LABEL_TOPIC, digest.topics)
    add(basics, BASICS, LABEL_ATTITUDE, digest.attitude)
    body = subsection(STYLE, style) + subsection(BASICS, basics)
    return body, statements
