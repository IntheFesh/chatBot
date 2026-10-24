"""Masking of the effective configuration before it is printed (R-CFG-001).

Real secrets are never part of :class:`~twin.config.settings.Settings`, but a few
values are personal (a WeChat id, e-mail addresses, an SSH host).  Those are
partially masked, and any key whose name looks like a credential is replaced
entirely, so a future setting added without thinking about it is still safe.
"""

from __future__ import annotations

import re
from typing import Any

from twin.config.settings import Settings

MASK = "***"
_SECRET_KEY = re.compile(r"(password|passwd|secret|token|api_?key|credential)", re.IGNORECASE)
_PARTIAL_PATHS = frozenset(
    {
        "target.username",
        "ops.smtp.user",
        "ops.smtp.to",
        "safety.emergency_contact.email",
        "autodl.host",
    }
)


def mask_value(value: str) -> str:
    """Keep the first two and last two characters of a long-enough value."""
    if len(value) <= 6:
        return MASK
    return f"{value[:2]}{MASK}{value[-2:]}"


def _walk(node: Any, path: str) -> Any:
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _SECRET_KEY.search(str(key)) and value not in (None, "", False):
                result[key] = MASK
            else:
                result[key] = _walk(value, child_path)
        return result
    if isinstance(node, list):
        return [_walk(item, path) for item in node]
    if isinstance(node, str) and path in _PARTIAL_PATHS:
        return mask_value(node)
    return node


def masked_settings(settings: Settings) -> dict[str, Any]:
    """The effective configuration as plain data with sensitive values masked."""
    result: dict[str, Any] = _walk(settings.to_plain(), "")
    return result
