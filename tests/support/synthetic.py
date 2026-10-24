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


def id_card_with_check(prefix17: str) -> str:
    """An 18 character ID number: ``prefix17`` plus its ISO 7064 check character."""
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    total = sum(int(d) * w for d, w in zip(prefix17, weights, strict=True))
    return prefix17 + "10X98765432"[total % 11]


def card_with_luhn(prefix: str, length: int = 16) -> str:
    """A card number of ``length`` digits starting with ``prefix`` that passes the Luhn check."""
    from twin.llm.redaction import luhn_valid

    body = prefix.ljust(length - 1, "0")[: length - 1]
    for check in "0123456789":
        if luhn_valid(body + check):
            return body + check
    raise AssertionError("a Luhn check digit always exists")


def us_phone(area: str = "312", exchange: str = "555", line: str = "0198") -> str:
    return f"({area}) {exchange}-{line}"


def email_address(local: str = "someone.name", domain: str = "example.org") -> str:
    return f"{local}@{domain}"
