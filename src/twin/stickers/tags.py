"""The closed tag vocabulary of stickers and how tags from different sources are combined.

The vocabulary (``stickers.tags_file``) and the table of nearby tags (``stickers.neighbors_file``)
are configuration files, not code (R-STK-003, R-STK-004).  A sticker carries up to
:data:`MAX_TAGS` tags from three sources:

``vision``
    what the picture itself shows (the vision model, R-STK-003);
``context``
    what she used it for in the conversations before the hold-out cutoff (R-STK-003);
``manual``
    what the user set with ``twin stickers tag`` - when present it replaces everything else.

:func:`effective_tags` combines them.  Evidence of use outweighs the picture: the context tags
come first, and a visual tag is kept only if it is the same tag or a nearby one (the neighbour
table); a visual tag that contradicts how she really used the sticker is dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from twin.config.lists import WordListError, load_word_list, locate_list_file
from twin.config.settings import Settings

MAX_TAGS = 3
SOURCE_VISION = "vision"
SOURCE_CONTEXT = "context"
SOURCE_MANUAL = "manual"
TAG_SOURCES = (SOURCE_VISION, SOURCE_CONTEXT, SOURCE_MANUAL)


class TagError(ValueError):
    """A tag is not in the vocabulary."""


@dataclass(frozen=True)
class TagVocabulary:
    """The tags a sticker may carry and the tags that stand in for each other."""

    tags: tuple[str, ...]
    neighbors: Mapping[str, tuple[str, ...]]

    def __contains__(self, tag: object) -> bool:
        return tag in self.tags

    def check(self, tags: Sequence[str]) -> list[str]:
        """``tags`` without duplicates, in order; any tag outside the vocabulary is an error."""
        unknown = [tag for tag in tags if tag not in self.tags]
        if unknown:
            raise TagError(
                f"not in the tag vocabulary: {', '.join(unknown)}; allowed: {'、'.join(self.tags)}"
            )
        return list(dict.fromkeys(tags))

    def near(self, tag: str) -> tuple[str, ...]:
        """Tags that may stand in for ``tag``, nearest first."""
        return self.neighbors.get(tag, ())

    def related(self, tag: str, other: str) -> bool:
        return tag == other or other in self.near(tag) or tag in self.near(other)


def parse_neighbors(text: str, tags: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """The neighbour table from YAML; every name in it must be a tag of the vocabulary."""
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise WordListError("the neighbour table must be a mapping from a tag to a list of tags")
    known = set(tags)
    table: dict[str, tuple[str, ...]] = {}
    for key, values in loaded.items():
        if not isinstance(values, list):
            raise WordListError(f"neighbours of {key!r} must be a list")
        names = [str(key), *map(str, values)]
        unknown = [name for name in names if name not in known]
        if unknown:
            raise WordListError(f"neighbour table: {unknown[0]!r} is not in the tag vocabulary")
        table[str(key)] = tuple(dict.fromkeys(str(v) for v in values if str(v) != str(key)))
    return table


def load_vocabulary_files(tags_file: Path, neighbors_file: Path) -> TagVocabulary:
    tags = load_word_list(tags_file)
    if not tags:
        raise WordListError(f"{tags_file} lists no tags")
    try:
        text = neighbors_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise WordListError(f"cannot read {neighbors_file}: {exc}") from exc
    return TagVocabulary(tuple(tags), parse_neighbors(text, tags))


def load_vocabulary(settings: Settings, root: Path) -> TagVocabulary:
    """The vocabulary named by the settings (looked up below ``root``, else in the checkout)."""
    return load_vocabulary_files(
        locate_list_file(root, settings.stickers.tags_file),
        locate_list_file(root, settings.stickers.neighbors_file),
    )


def merge_tags(
    vision: Sequence[str], context: Sequence[str], vocabulary: TagVocabulary
) -> list[str]:
    """Visual and context tags as one list; the evidence of use has the last word."""
    if not context:
        return list(vision)[:MAX_TAGS]
    merged = list(context)[:MAX_TAGS]
    for tag in vision:
        if len(merged) >= MAX_TAGS:
            break
        if tag not in merged and any(vocabulary.related(tag, used) for used in context):
            merged.append(tag)
    return merged


def effective_tags(
    vision: Sequence[str] | None,
    context: Sequence[str] | None,
    manual: Sequence[str] | None,
    vocabulary: TagVocabulary,
) -> tuple[list[str], str | None]:
    """``(tags, source)`` a sticker is selected by; the source says which evidence decided."""
    if manual:
        return list(manual)[:MAX_TAGS], SOURCE_MANUAL
    if context:
        return merge_tags(vision or (), context, vocabulary), SOURCE_CONTEXT
    if vision:
        return list(vision)[:MAX_TAGS], SOURCE_VISION
    return [], None
