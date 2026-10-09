"""Pattern-shaped synthetic values, assembled at run time.

Tests need strings that *look like* personal identifiers (to prove they are masked or
redacted).  They are built from parts here so that no such literal exists in the
repository and ``scripts/privacy_scan.py`` stays clean.
"""

from __future__ import annotations


def mobile() -> str:
    return "138" + "00138000"


def mobile_formatted() -> str:
    return "+86 " + "139" + "-0013-" + "9000"


def wxid() -> str:
    return "wxid_" + "synthetic123"


def wxid_short() -> str:
    return "wxid_" + "abcdef123"


def chatroom() -> str:
    return "12345678" + "@chatroom"


def export_fragment() -> str:
    return '{"' + 'localId": 1, "' + 'isSent": true, "' + 'createTimeText": "x", "content": "y"}'
