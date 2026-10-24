"""Changing a value in ``config/config.yaml`` on the user's behalf, after they confirmed.

The file is the user's: it has their comments and their layout.  Only the lines of the one
top-level section being changed are replaced (with the section's merged values), everything
else is kept byte for byte.  The result is loaded through :class:`~twin.config.settings.Settings`
before it is kept; if it does not validate the original text is put back.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from twin.config.loader import ConfigError, load_settings


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    """``(first line, line after the last)`` of the top-level ``section`` entry, if present."""
    pattern = re.compile(rf"^{re.escape(section)}\s*:")
    for start, line in enumerate(lines):
        if not pattern.match(line):
            continue
        end = start + 1
        last_used = start
        while end < len(lines):
            current = lines[end]
            if current.strip() and not current[0].isspace():
                break  # the next top-level key (or a comment that belongs to it)
            if current.strip():
                last_used = end
            end += 1
        return start, last_used + 1
    return None


def set_section_values(path: Path, section: str, values: dict[str, Any]) -> str:
    """Set ``section.<key> = value`` for every item of ``values``; returns the new text.

    Creates the file when it does not exist.  Raises :class:`ConfigError` (and leaves the file
    as it was) if the result is not a valid configuration.
    """
    raw = path.read_bytes().decode("utf-8") if path.is_file() else ""
    eol = "\r\n" if "\r\n" in raw else "\n"  # a file edited on Windows keeps its line endings
    original = raw.replace("\r\n", "\n")
    data = yaml.safe_load(original) if original.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a mapping at the top level; edit it by hand")
    current = data.get(section)
    merged = {**(current if isinstance(current, dict) else {}), **values}
    block = yaml.safe_dump(
        {section: merged}, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).splitlines()
    lines = original.splitlines()
    bounds = _section_bounds(lines, section)
    if bounds is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    else:
        start, end = bounds
        lines[start:end] = block
    text = eol.join(lines) + eol
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    try:
        load_settings(temporary)
    except ConfigError:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, path)
    return text
