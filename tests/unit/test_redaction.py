"""Basic identifier redaction used by logging and (from round 01) the LLM layer."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from twin.llm.redaction import REPLACEMENTS, is_valid_id_card, luhn_valid, redact


def make_id_card(prefix17: str) -> str:
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    total = sum(int(d) * w for d, w in zip(prefix17, weights, strict=True))
    return prefix17 + "10X98765432"[total % 11]


def make_card(prefix15: str) -> str:
    for check in "0123456789":
        if luhn_valid(prefix15 + check):
            return prefix15 + check
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("打我电话 13800138000 谢谢", "phone"),
        ("+86 139-0013-9000", None),
        ("call (312) 555-0198 now", "phone"),
        ("312-555-0198", "phone"),
        ("mail me at someone.name+tag@example.org please", "email"),
        ("加我 wxid_abcdef123 吧", "wxid"),
        ("群 12345678@chatroom", "wxid"),
    ],
)
def test_identifiers_are_replaced(text: str, token: str | None) -> None:
    result = redact(text)
    assert result != text
    if token:
        assert REPLACEMENTS[token] in result


def test_valid_id_cards_and_bank_cards_are_replaced_but_invalid_numbers_are_not() -> None:
    card = make_id_card("11010519491231002")
    assert is_valid_id_card(card)
    assert redact(f"身份证 {card}") == f"身份证 {REPLACEMENTS['id_card']}"
    broken = card[:-1] + ("0" if card[-1] != "0" else "1")
    assert not is_valid_id_card(broken)
    assert redact(f"编号 {broken}") == f"编号 {broken}"

    bank = make_card("411111111111111")
    assert luhn_valid(bank)
    assert redact(f"卡号 {bank}") == f"卡号 {REPLACEMENTS['bank_card']}"
    spaced = " ".join([bank[:4], bank[4:8], bank[8:12], bank[12:]])
    assert redact(spaced) == REPLACEMENTS["bank_card"]
    not_luhn = bank[:-1] + str((int(bank[-1]) + 1) % 10)
    assert redact(not_luhn) == not_luhn


@pytest.mark.parametrize(
    "text",
    ["今天 2026-10-09 见", "一共 1234 元", "版本 3.12.13", "12345678901234", "你好，世界", "", "pi=3.14159"],
)
def test_ordinary_text_is_untouched(text: str) -> None:
    assert redact(text) == text


def test_helpers_reject_malformed_input() -> None:
    assert not is_valid_id_card("123")
    assert not is_valid_id_card("11010519491231002X1")
    assert not luhn_valid("12a4")


@given(st.text(max_size=200))
def test_redaction_never_raises_and_is_idempotent(text: str) -> None:
    once = redact(text)
    assert redact(once) == once


@given(st.integers(min_value=3, max_value=9), st.integers(min_value=0, max_value=999_999_999))
def test_generated_mobile_numbers_never_survive(second: int, rest: int) -> None:
    number = f"1{second}{rest:09d}"
    assert number not in redact(f"我的号码是{number}，打给我")
