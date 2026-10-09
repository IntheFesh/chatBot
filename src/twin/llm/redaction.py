"""Redaction of personal identifiers before text leaves the machine or hits a log.

Replaces phone numbers (mainland China and US formats), e-mail addresses,
mainland ID card numbers (18 digits with a valid check digit), bank card numbers
(16-19 digits passing the Luhn check) and WeChat ids with type tokens such
as ``[手机号]``.  The logging layer uses :func:`redact` for DEBUG output; the LLM
layer calls it on everything sent to DeepSeek or AutoDL (CLAUDE.md rule 6).
"""

from __future__ import annotations

import re

REPLACEMENTS = {
    "phone": "[手机号]",
    "email": "[邮箱]",
    "id_card": "[身份证号]",
    "bank_card": "[银行卡号]",
    "wxid": "[微信号]",
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_WXID = re.compile(r"\bwxid_[A-Za-z0-9_\-]{6,}\b|\b[A-Za-z0-9_\-]{6,}@chatroom\b")
_ID_CARD = re.compile(
    r"(?<![0-9A-Za-z])[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
    r"(?![0-9A-Za-z])"
)
_BANK_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
_CN_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d[- ]?\d{4}[- ]?\d{4}(?!\d)")
_US_PHONE = re.compile(
    r"(?<![\d.])(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{2}[-. ]?\d{4}(?![\d.])"
)

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


def is_valid_id_card(number: str) -> bool:
    """Check the ISO 7064 mod 11-2 check digit of an 18-character ID number."""
    if len(number) != 18 or not number[:17].isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(number[:17], _ID_WEIGHTS, strict=True))
    return _ID_CHECK[total % 11] == number[17].upper()


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a string of digits."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _sub_id(match: re.Match[str]) -> str:
    return REPLACEMENTS["id_card"] if is_valid_id_card(match.group(0)) else match.group(0)


def _sub_bank(match: re.Match[str]) -> str:
    digits = re.sub(r"[ -]", "", match.group(0))
    if 16 <= len(digits) <= 19 and luhn_valid(digits):
        return REPLACEMENTS["bank_card"]
    return match.group(0)


def redact(text: str) -> str:
    """Return ``text`` with personal identifiers replaced by type tokens."""
    text = _EMAIL.sub(REPLACEMENTS["email"], text)
    text = _WXID.sub(REPLACEMENTS["wxid"], text)
    text = _ID_CARD.sub(_sub_id, text)
    text = _BANK_CARD.sub(_sub_bank, text)
    text = _CN_PHONE.sub(REPLACEMENTS["phone"], text)
    return _US_PHONE.sub(REPLACEMENTS["phone"], text)
