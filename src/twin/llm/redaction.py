"""Redaction of personal identifiers before text leaves the machine or hits a log (R-LLM-009).

What is replaced, and how false positives are kept down:

* **phone numbers** - mainland China (``1[3-9]`` + 9 digits, optional ``+86``/``0086``, optional
  separators) and North American (NANP) numbers; the area code and the exchange must not start
  with 0 or 1, so ordinary 10 digit numbers such as timestamps are left alone;
* **e-mail addresses**;
* **mainland ID card numbers** - 18 characters, a real birth date and a correct ISO 7064
  check digit; a number with a wrong check digit is not an ID card and stays;
* **bank card numbers** - 16 to 19 digits (single spaces or hyphens allowed between digits)
  that pass the Luhn check;
* **detailed addresses** - Chinese street addresses (a street name with a house number and
  ``号``, optionally with a leading province/city/district and a trailing building/unit/room),
  stand-alone building addresses (at least two of ``栋/幢/号楼``, ``单元``, ``室``) and North
  American street addresses.  A bare ``路`` or ``号`` never counts;
* **WeChat ids** - ``wxid_...``, ``...@chatroom`` and an id that follows a label such as
  ``微信号：``.

Matching happens on a copy of the text in which full-width digits and a few full-width
symbols are mapped one-to-one to ASCII, so offsets stay valid and ``１３８...`` is caught.

:func:`redact` returns a :class:`RedactionResult`; :func:`redact_text` is the string-only
shortcut used by the logging layer.  :class:`ConsistentRedactor` keeps one numbered token per
distinct entity (``[手机号#1]``) for training exports (R-TRN-007).  :func:`find_tokens`
detects tokens in a model reply (R-ENG-012).

This module is a leaf: it imports nothing from the rest of the project, so the logging and job
layers can use it without cycles.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

REPLACEMENTS = {
    "phone": "[手机号]",
    "email": "[邮箱]",
    "id_card": "[身份证号]",
    "bank_card": "[银行卡号]",
    "address": "[地址]",
    "wxid": "[微信号]",
}
KINDS = tuple(REPLACEMENTS)
# lower value wins when two detectors claim overlapping text
_PRIORITY = {"email": 0, "wxid": 1, "id_card": 2, "bank_card": 3, "phone": 4, "address": 5}

_TOKEN_NAMES = "|".join(re.escape(token[1:-1]) for token in REPLACEMENTS.values())
TOKEN_PATTERN = re.compile(rf"\[(?:{_TOKEN_NAMES})(?:#\d+)?\]")

# ----------------------------------------------------------------- matching helpers

_FULLWIDTH = str.maketrans(
    {
        **{chr(0xFF10 + i): str(i) for i in range(10)},
        "＋": "+",
        "－": "-",
        "（": "(",
        "）": ")",
        "＠": "@",
        "．": ".",
        "　": " ",
    }
)

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


def is_valid_id_card(number: str) -> bool:
    """Check the birth date and the ISO 7064 mod 11-2 check digit of an 18-character ID number."""
    if len(number) != 18 or not number[:17].isdigit() or number[0] == "0":
        return False
    try:
        date(int(number[6:10]), int(number[10:12]), int(number[12:14]))
    except ValueError:
        return False
    total = sum(int(d) * w for d, w in zip(number[:17], _ID_WEIGHTS, strict=True))
    return _ID_CHECK[total % 11] == number[17].upper()


def luhn_valid(digits: str) -> bool:
    """Luhn checksum over a string of ASCII digits."""
    if not digits or not digits.isascii() or not digits.isdigit():
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


@dataclass(frozen=True)
class _Hit:
    kind: str
    start: int
    end: int
    key: str  # normalised entity value, used by ConsistentRedactor


_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"
)
_WXID = re.compile(r"(?<![A-Za-z0-9_\-])wxid_[A-Za-z0-9_\-]{6,}|[A-Za-z0-9_\-]{6,}@chatroom\b")
_WXID_LABELLED = re.compile(
    r"(?:微信号|微信ID|vx号|wx号)\s*[:：是为]?\s*([A-Za-z][A-Za-z0-9_\-]{5,19})(?![A-Za-z0-9_\-])"
    r"|(?:微信|vx|wx|wechat(?:\s*id)?)\s*[:：]\s*([A-Za-z][A-Za-z0-9_\-]{5,19})(?![A-Za-z0-9_\-])",
    re.IGNORECASE,
)
_ID_CARD = re.compile(r"(?<![0-9A-Za-z])[1-9]\d{16}[\dXx](?![0-9A-Za-z])")
_BANK_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){15,18}(?!\d)")
_CN_PHONE = re.compile(r"(?<![\d])(?:(?:\+|00)?86[-\s]?)?1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}(?![\d])")
_US_PHONE = re.compile(
    r"(?<![\d.])(?:\+?1[-.\s]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[-.\s]?[2-9]\d{2}[-.\s]?\d{4}(?![\d.])"
)

_PROVINCE = (
    r"(?:北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北"
    r"|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)"
    r"(?:省|市|自治区|特别行政区)?"
)
_CJK = r"[一-鿿]"
_STREET = rf"{_CJK}{{1,6}}(?:大街|大道|街道|路|街|巷|弄|胡同)"
_HOUSE_NUMBER = r"\d{1,5}(?:[-－]\d{1,4})?\s?号"
_CN_STREET_ADDRESS = re.compile(
    rf"(?:{_PROVINCE})?(?:{_CJK}{{2,3}}(?:市|州|盟))?(?:{_CJK}{{1,4}}(?:区|县|旗))?"
    rf"(?:{_CJK}{{1,4}}(?:镇|乡|街道))?{_STREET}\s?{_HOUSE_NUMBER}"
    rf"(?:\s?\d{{1,3}}\s?(?:栋|幢|座|号楼))?(?:\s?\d{{1,2}}\s?单元)?(?:\s?\d{{1,2}}\s?层)?"
    rf"(?:\s?\d{{1,4}}\s?(?:室|房))?"
)
_CN_BUILDING_ADDRESS = re.compile(
    r"(?:(?:\d{1,3}\s?(?:栋|幢|号楼)\s?\d{0,2}\s?单元\s?\d{1,4}\s?(?:室|房|号)?)"
    r"|(?:\d{1,3}\s?(?:栋|幢|号楼)\s?\d{1,4}\s?(?:室|房))"
    r"|(?:\d{1,2}\s?单元\s?\d{1,4}\s?(?:室|房)))"
)
_US_SUFFIX = (
    r"(?i:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|place|pl"
    r"|parkway|pkwy|terrace|highway|hwy)"
)
_US_ADDRESS = re.compile(
    rf"(?<![\w#])\d{{1,6}}\s+(?:(?:[NSEW]\.?|North|South|East|West)\s+)?"
    rf"(?:[A-Z][A-Za-z']*\.?\s+){{1,3}}?{_US_SUFFIX}\b\.?"
    r"(?:\s*,?\s*(?i:apt|unit|suite|ste)\.?\s*#?[\w-]+|\s*,?\s*#\s*[\w-]+)?"
    r"(?:,\s*[A-Z][A-Za-z.' ]{1,25},\s*[A-Z]{2}(?:\s+\d{5}(?:-\d{4})?)?)?"
)
_HAS_DIGIT_WORD = re.compile(r"\d\s+[A-Za-z]")
# words that tend to stand right before an address and must not be swallowed by the match
_LEADING_FUNCTION_CHARS = frozenset(
    "我你他她它咱们在到去于住来回搬是的了从往正过和跟与及或就都也还又才再把被让给对向由自按约离近靠有没这那哪"
)


def _trim_leading(text: str, start: int) -> int:
    """Move ``start`` right past leading function characters (``我住在`` before a street)."""
    while start < len(text) and text[start] in _LEADING_FUNCTION_CHARS:
        start += 1
    return start


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _phone_key(value: str, *, national: bool) -> str:
    """Digits that identify the number: 11 digits for mainland China, 10 for North America."""
    digits = _digits_only(value)
    if national:
        for prefix in ("0086", "86"):
            if digits.startswith(prefix) and len(digits) > 11:
                return digits[len(prefix) :]
        return digits
    return digits[-10:]


def _detect_email(text: str) -> Iterable[_Hit]:
    for match in _EMAIL.finditer(text):
        yield _Hit("email", match.start(), match.end(), match.group(0).lower())


def _detect_wxid(text: str) -> Iterable[_Hit]:
    for match in _WXID.finditer(text):
        yield _Hit("wxid", match.start(), match.end(), match.group(0))
    for match in _WXID_LABELLED.finditer(text):
        group = 1 if match.group(1) else 2
        yield _Hit("wxid", match.start(group), match.end(group), match.group(group))


def _detect_id_card(text: str) -> Iterable[_Hit]:
    for match in _ID_CARD.finditer(text):
        if is_valid_id_card(match.group(0)):
            yield _Hit("id_card", match.start(), match.end(), match.group(0).upper())


def _detect_bank_card(text: str) -> Iterable[_Hit]:
    for match in _BANK_CARD.finditer(text):
        digits = _digits_only(match.group(0))
        if 16 <= len(digits) <= 19 and luhn_valid(digits):
            yield _Hit("bank_card", match.start(), match.end(), digits)


def _detect_phone(text: str) -> Iterable[_Hit]:
    for match in _CN_PHONE.finditer(text):
        yield _Hit("phone", match.start(), match.end(), _phone_key(match.group(0), national=True))
    for match in _US_PHONE.finditer(text):
        yield _Hit("phone", match.start(), match.end(), _phone_key(match.group(0), national=False))


def _detect_address(text: str) -> Iterable[_Hit]:
    candidates: list[tuple[re.Pattern[str], bool]] = []
    if "号" in text:
        candidates.append((_CN_STREET_ADDRESS, True))
    if any(token in text for token in ("栋", "幢", "号楼", "单元")):
        candidates.append((_CN_BUILDING_ADDRESS, True))
    if _HAS_DIGIT_WORD.search(text):
        candidates.append((_US_ADDRESS, False))
    for pattern, trim in candidates:
        for match in pattern.finditer(text):
            start = _trim_leading(text, match.start()) if trim else match.start()
            if start >= match.end():
                continue
            value = re.sub(r"\s+", "", text[start : match.end()]).lower()
            yield _Hit("address", start, match.end(), value)


_DETECTORS: tuple[Callable[[str], Iterable[_Hit]], ...] = (
    _detect_email,
    _detect_wxid,
    _detect_id_card,
    _detect_bank_card,
    _detect_phone,
    _detect_address,
)


def _select(hits: list[_Hit]) -> list[_Hit]:
    """Pick non-overlapping hits: higher priority first, then longer, then earlier."""
    accepted: list[_Hit] = []
    for hit in sorted(hits, key=lambda h: (_PRIORITY[h.kind], -(h.end - h.start), h.start)):
        if all(hit.end <= other.start or hit.start >= other.end for other in accepted):
            accepted.append(hit)
    accepted.sort(key=lambda h: h.start)
    return accepted


# ----------------------------------------------------------------------- public API


@dataclass(frozen=True)
class RedactionSpan:
    """One replaced region: offsets refer to the *input* text; the original value is not kept."""

    kind: str
    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    spans: tuple[RedactionSpan, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return bool(self.spans)

    def counts(self) -> dict[str, int]:
        """Number of replacements per kind (safe to log: contains no content)."""
        counts: dict[str, int] = {}
        for span in self.spans:
            counts[span.kind] = counts.get(span.kind, 0) + 1
        return counts


def _find(text: str) -> list[_Hit]:
    probe = text.translate(_FULLWIDTH)
    hits: list[_Hit] = []
    for detector in _DETECTORS:
        hits.extend(detector(probe))
    return _select(hits)


def _apply(text: str, hits: list[_Hit], token_for: Callable[[_Hit], str]) -> RedactionResult:
    pieces: list[str] = []
    spans: list[RedactionSpan] = []
    cursor = 0
    for hit in hits:
        token = token_for(hit)
        pieces.append(text[cursor : hit.start])
        pieces.append(token)
        spans.append(RedactionSpan(hit.kind, hit.start, hit.end, token))
        cursor = hit.end
    pieces.append(text[cursor:])
    return RedactionResult("".join(pieces), tuple(spans))


def redact(text: str) -> RedactionResult:
    """Replace personal identifiers in ``text`` with type tokens such as ``[手机号]``."""
    if not text:
        return RedactionResult(text)
    hits = _find(text)
    if not hits:
        return RedactionResult(text)
    return _apply(text, hits, lambda hit: REPLACEMENTS[hit.kind])


def redact_text(text: str) -> str:
    """:func:`redact` without the span list (logging and error messages)."""
    return redact(text).text


def find_tokens(text: str) -> list[str]:
    """Redaction tokens present in ``text`` (``[手机号]``, ``[邮箱#2]``, ...).

    A model reply must never contain one (R-ENG-012): it would mean the reply echoes a
    token copied from the prompt instead of speaking naturally.
    """
    return TOKEN_PATTERN.findall(text)


def contains_token(text: str) -> bool:
    return TOKEN_PATTERN.search(text) is not None


class ConsistentRedactor:
    """Numbered tokens that stay the same for the same entity (R-LLM-009, R-TRN-007).

    ``[手机号#1]`` always stands for one phone number, ``[手机号#2]`` for the next distinct
    one, and so on, for every text redacted by the same instance.  Entities are remembered by
    a salted hash of their normalised value, so :meth:`state` can be stored with a dataset
    without keeping any identifier in clear text.
    """

    def __init__(self, *, salt: str = "twin-redaction-v1") -> None:
        self._salt = salt
        self._numbers: dict[str, dict[str, int]] = {kind: {} for kind in KINDS}

    def _fingerprint(self, hit: _Hit) -> str:
        digest = hashlib.sha256(f"{self._salt}\x1f{hit.kind}\x1f{hit.key}".encode()).hexdigest()
        return digest[:32]

    def _token(self, hit: _Hit) -> str:
        numbers = self._numbers[hit.kind]
        fingerprint = self._fingerprint(hit)
        number = numbers.get(fingerprint)
        if number is None:
            number = len(numbers) + 1
            numbers[fingerprint] = number
        return f"{REPLACEMENTS[hit.kind][:-1]}#{number}]"

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text)
        hits = _find(text)
        if not hits:
            return RedactionResult(text)
        return _apply(text, hits, self._token)

    def redact_text(self, text: str) -> str:
        return self.redact(text).text

    def counts(self) -> dict[str, int]:
        """Number of distinct entities seen per kind."""
        return {kind: len(numbers) for kind, numbers in self._numbers.items() if numbers}

    def state(self) -> dict[str, dict[str, int]]:
        """Hash -> number tables, for storing next to a dataset and resuming later."""
        return {kind: dict(numbers) for kind, numbers in self._numbers.items() if numbers}

    @classmethod
    def from_state(
        cls, state: dict[str, dict[str, int]], *, salt: str = "twin-redaction-v1"
    ) -> ConsistentRedactor:
        redactor = cls(salt=salt)
        for kind, numbers in state.items():
            if kind not in redactor._numbers:
                raise ValueError(f"unknown redaction kind {kind!r}")
            redactor._numbers[kind] = {str(k): int(v) for k, v in numbers.items()}
        return redactor
