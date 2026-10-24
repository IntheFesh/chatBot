"""Redaction of personal identifiers (R-LLM-009, R-PRIV-002)."""

from __future__ import annotations

import re
import string
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.support.synthetic import (
    card_with_luhn,
    chatroom,
    email_address,
    id_card_with_check,
    mobile,
    mobile_formatted,
    us_phone,
    wxid_short,
)
from twin.llm.redaction import (
    REPLACEMENTS,
    ConsistentRedactor,
    contains_token,
    find_tokens,
    is_valid_id_card,
    luhn_valid,
    prompt_redactor,
    redact,
    redact_prompt,
    redact_text,
    use_prompt_redactor,
)

# ----------------------------------------------------------------- per kind


def dotted_mobile() -> str:
    number = mobile()
    return f"{number[:3]}.{number[3:7]}.{number[7:]}"


def grouped_id_card(separator: str) -> str:
    """A valid ID number written in groups of 6, 8 and 4 (R18-A4)."""
    card = id_card_with_check("11010519491231002")
    return separator.join([card[:6], card[6:14], card[14:]])


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (f"打我电话 {mobile()} 谢谢", "phone"),
        (f"打我电话{mobile_formatted()}谢谢", "phone"),
        (f"备用 +86-{mobile()[:3]}-{mobile()[3:7]}-{mobile()[7:]}", "phone"),
        (f"call {us_phone()} now", "phone"),
        ("312-555-0198", "phone"),
        ("+1 312.555.0198", "phone"),
        # R17-9: a number that ends a sentence is still a number
        ("call me at 312-555-0123.", "phone"),
        ("+1 312.555.0123.", "phone"),
        ("(312) 555-0123.", "phone"),
        ("312.555.0123.", "phone"),
        ("312-555-0123。", "phone"),
        ("Tel. 312-555-0123.", "phone"),
        (f"mail me at {email_address('a.b+tag')} please", "email"),
        (f"加我 {wxid_short()} 吧", "wxid"),
        (f"群 {chatroom()}", "wxid"),
        ("我的微信号：Abc_12345 欢迎", "wxid"),
        ("wechat: lovely_cat9 hi", "wxid"),
        # R18-A4 (D-766): the ways people actually write them used to pass untouched
        ("我的微信是abc_def12", "wxid"),
        ("我微信 abc_def12", "wxid"),
        ("加我vx abc_def12", "wxid"),
        ("加我微信吧 zhang_san88", "wxid"),
        ("我微信abc_def12，加一下", "wxid"),
        ("v信：Lucky-cat2024", "wxid"),
        ("上海市浦东新区张江路88弄3号501室", "address"),
        ("地址：长宁区愚园路1088弄5号601", "address"),
        ("成都市武侯区人民南路四段27号", "address"),
        ("成都市高新区天府大道北段1700号", "address"),
        ("住在建国路2段16号3楼", "address"),
        (f"手机：{dotted_mobile()}", "phone"),
        (f"手机 +86.{dotted_mobile()}。", "phone"),
        (f"身份证 {grouped_id_card(' ')}", "id_card"),
        (f"身份证：{grouped_id_card('-')}，帮我填一下", "id_card"),
    ],
)
def test_identifiers_are_replaced_by_type_tokens(text: str, kind: str) -> None:
    result = redact(text)
    assert result.changed
    assert REPLACEMENTS[kind] in result.text
    assert [span.kind for span in result.spans] == [kind]
    span = result.spans[0]
    assert text[span.start : span.end] not in result.text


def test_full_width_digits_are_caught_and_offsets_stay_valid() -> None:
    wide = "".join(chr(0xFF10 + int(digit)) for digit in mobile())
    text = f"电话：{wide}。"
    result = redact(text)
    assert result.text == f"电话：{REPLACEMENTS['phone']}。"
    assert text[result.spans[0].start : result.spans[0].end] == wide


def test_valid_id_cards_and_bank_cards_are_replaced_but_invalid_numbers_are_not() -> None:
    card = id_card_with_check("11010519491231002")
    assert is_valid_id_card(card)
    assert redact_text(f"身份证 {card}") == f"身份证 {REPLACEMENTS['id_card']}"
    broken = card[:-1] + ("0" if card[-1] != "0" else "1")
    assert not is_valid_id_card(broken)
    assert redact_text(f"编号 {broken}") == f"编号 {broken}"

    bank = card_with_luhn("411111111111111")
    assert luhn_valid(bank)
    assert redact_text(f"卡号 {bank}") == f"卡号 {REPLACEMENTS['bank_card']}"
    spaced = " ".join([bank[:4], bank[4:8], bank[8:12], bank[12:]])
    assert redact_text(spaced) == REPLACEMENTS["bank_card"]
    hyphenated = "-".join([bank[:4], bank[4:8], bank[8:12], bank[12:]])
    assert redact_text(hyphenated) == REPLACEMENTS["bank_card"]
    not_luhn = bank[:-1] + str((int(bank[-1]) + 1) % 10)
    assert redact_text(not_luhn) == not_luhn
    long_card = card_with_luhn("62220212345678", 19)
    assert redact_text(long_card) == REPLACEMENTS["bank_card"]


def test_id_card_with_an_impossible_birth_date_is_not_an_id_card() -> None:
    impossible = id_card_with_check("11010519490230000")  # 31 February
    assert not is_valid_id_card(impossible)
    assert redact_text(impossible) == impossible


def test_id_card_check_character_x_is_accepted_in_either_case() -> None:
    cards = [id_card_with_check(f"1101051949123100{digit}") for digit in "0123456789"]
    with_x = [card for card in cards if card.endswith("X")]
    assert with_x, "ten consecutive sequence numbers cover every check character"
    assert redact_text(with_x[0].lower()) == REPLACEMENTS["id_card"]
    assert redact_text(with_x[0]) == REPLACEMENTS["id_card"]


@pytest.mark.parametrize(
    "text",
    [
        "我住在北京市朝阳区建国路88号",
        "住址：上海市浦东新区张江路 12 号 3栋2单元501室",
        "广东省深圳市南山区科技路1号",
        "他家在3栋2单元501室，到了打电话",
        "5号楼301室",
        "1600 Pennsylvania Avenue",
        "我在 1234 N State St, Chicago, IL 60601 等你",
        "apartment at 221 Baker Street Apt 4B",
    ],
)
def test_detailed_addresses_are_replaced(text: str) -> None:
    result = redact(text)
    assert [span.kind for span in result.spans] == ["address"], result.text
    assert REPLACEMENTS["address"] in result.text


def test_r18_a4_lane_and_section_addresses_are_replaced_whole() -> None:
    """R18-A4 (D-766): ``N弄`` and ``N段`` between the street and the house number, and a room
    number written straight after ``号``, are part of the address; a year after it is not."""
    assert redact_text("上海市浦东新区张江路88弄3号501室") == REPLACEMENTS["address"]
    assert redact_text("地址：长宁区愚园路1088弄5号601") == f"地址：{REPLACEMENTS['address']}"
    assert redact_text("成都市武侯区人民南路四段27号") == REPLACEMENTS["address"]
    assert redact_text("成都市高新区天府大道北段1700号到了") == f"{REPLACEMENTS['address']}到了"
    assert redact_text("建国路88号2026年拆了") == f"{REPLACEMENTS['address']}2026年拆了"


def test_r18_a4_the_natural_wechat_id_forms_keep_the_words_around_the_id() -> None:
    """R18-A4 (D-766): only the id goes; the label and the words around it stay."""
    assert redact_text("我的微信是abc_def12") == f"我的微信是{REPLACEMENTS['wxid']}"
    assert redact_text("加我微信吧 zhang_san88 备注一下") == (
        f"加我微信吧 {REPLACEMENTS['wxid']} 备注一下"
    )
    assert redact_text("加我vx abc_def12") == f"加我vx {REPLACEMENTS['wxid']}"
    # an ASCII label needs a separator: "vxlan123" is a word, not "vx" + an id
    assert redact_text("开了vxlan12345") == "开了vxlan12345"


def test_leading_function_words_are_not_swallowed_by_an_address() -> None:
    assert redact_text("我住在建国路88号") == f"我住在{REPLACEMENTS['address']}"
    assert redact_text("我们明天约在朝阳区建国路88号门口见") == (
        f"我们明天约在{REPLACEMENTS['address']}门口见"
    )


@pytest.mark.parametrize(
    "text",
    [
        "路上堵车了",
        "今天是5月号外",
        "我是3号选手",
        "坐5号线到终点",
        "你的QQ号我记得是123456",
        "100路公交车",
        "我知道88号球员很强",
        "三号楼",
        "第二单元考试",
        "在路边等你",
        "wxid 是什么",
        "we met at 5 pm on Main Street",
        "I want 3 big dogs",
        "2026-10-09 见",
        "一共 1234 元",
        "版本 3.12.13",
        "12345678901234",
        "pi=3.14159",
        # R17-9: the sentence-final dot is allowed after a number, a decimal point is not
        "192.168.100.1234",
        "312-555-0123.5",
        "10.234.567.8901",
        "2.312-555-0123",
        "你好，世界",
        "",
        # R18-A4 (D-766): the wider address, WeChat id and phone forms must not catch these
        "第3段",
        "5弄堂",
        "这段路好难走",
        "微信上说",
        "我微信 好的",
        "我微信没电了",
        "加我微信吧",
        "微信 hello",
        "wx 号码给你了",
        "走到人民南路四段再说",
        "他住3弄",
        "第2段第5号",
        "版本 138.1234.5678.9",
        "138.12345678",
        "WeChat polling keeps failing; check the network connection",
        "the WeChat channel serves the one bound user",
    ],
)
def test_ordinary_text_is_untouched(text: str) -> None:
    result = redact(text)
    assert result.text == text and not result.changed


def test_overlapping_matches_prefer_the_stronger_kind() -> None:
    card = id_card_with_check("11010519491231002")
    # an ID number is not also reported as a bank card even though it has 18 digits
    result = redact(f"{card}")
    assert [s.kind for s in result.spans] == ["id_card"]
    mixed = f"{email_address()} {mobile()}"
    assert [s.kind for s in redact(mixed).spans] == ["email", "phone"]


def test_result_helpers_and_counts() -> None:
    text = f"{mobile()} 和 {email_address()} 以及 {email_address('b', 'x.org')}"
    result = redact(text)
    assert result.counts() == {"phone": 1, "email": 2}
    assert all(span.replacement in REPLACEMENTS.values() for span in result.spans)
    assert redact("没有敏感信息").counts() == {}
    assert redact_text(text) == result.text


def test_helpers_reject_malformed_input() -> None:
    assert not is_valid_id_card("123")
    assert not is_valid_id_card("11010519491231002X1")
    assert not is_valid_id_card("01010519491231002X")
    assert not luhn_valid("12a4")
    assert not luhn_valid("")
    assert not luhn_valid("１２３４")


# ------------------------------------------------------- consistent placeholders


def test_the_same_entity_gets_the_same_numbered_token() -> None:
    redactor = ConsistentRedactor()
    other = mobile()[:-1] + str((int(mobile()[-1]) + 1) % 10)
    first = redactor.redact_text(f"A: {mobile()} B: {other}")
    again = redactor.redact_text(f"再说一遍 {mobile()}，邮箱 {email_address()}")
    assert first == "A: [手机号#1] B: [手机号#2]"
    assert again == "再说一遍 [手机号#1]，邮箱 [邮箱#1]"
    assert redactor.counts() == {"phone": 2, "email": 1}


def test_equivalent_spellings_map_to_one_entity() -> None:
    redactor = ConsistentRedactor()
    plain = redactor.redact_text(mobile())
    formatted = redactor.redact_text(f"+86 {mobile()[:3]}-{mobile()[3:7]}-{mobile()[7:]}")
    shouted = redactor.redact_text(email_address().upper())
    lower = redactor.redact_text(email_address())
    assert plain == formatted == "[手机号#1]"
    assert shouted == lower == "[邮箱#1]"
    assert redactor.redact_text(us_phone()) == redactor.redact_text("312-555-0198") == "[手机号#2]"


def test_state_round_trip_resumes_numbering_without_keeping_identifiers() -> None:
    redactor = ConsistentRedactor()
    redactor.redact_text(f"{mobile()} {email_address()}")
    state = redactor.state()
    assert mobile() not in str(state) and email_address() not in str(state)
    resumed = ConsistentRedactor.from_state(state)
    assert resumed.redact_text(email_address()) == "[邮箱#1]"
    assert resumed.redact_text(email_address("new", "x.org")) == "[邮箱#2]"
    with pytest.raises(ValueError, match="unknown redaction kind"):
        ConsistentRedactor.from_state({"passport": {"x": 1}})


def test_different_salts_give_different_fingerprints() -> None:
    one = ConsistentRedactor(salt="a")
    two = ConsistentRedactor(salt="b")
    one.redact_text(mobile())
    two.redact_text(mobile())
    assert set(one.state()["phone"]) != set(two.state()["phone"])


def test_consistent_redaction_of_empty_and_clean_text() -> None:
    redactor = ConsistentRedactor()
    assert redactor.redact("").text == ""
    assert redactor.redact("你好").spans == ()


# ------------------------------------------------------------ token detection


def test_leaked_tokens_are_found_in_replies() -> None:
    assert find_tokens("好呀[手机号]，记得[邮箱#12]") == ["[手机号]", "[邮箱#12]"]
    assert contains_token("它在[地址]")
    assert not contains_token("没有 [括号里的话] 也没有 #1")
    assert find_tokens("") == []


# ------------------------------------------------------------ property tests

_CLEAN_CHARS = "你我他她的了是在有不这就都和也很好想去吃玩看说今天明晚上下午早安"
_CLEAN_ALPHABET = st.sampled_from(list(_CLEAN_CHARS + "abcdefghijklmnopqrstuvwxyz ，。！？,.!?~"))
# R18-A4 (D-766): ``vx abcdefg`` written with these letters is a WeChat id now, not clean text
_ASCII_WECHAT_LABEL = re.compile(r"(?i)(?<![a-z0-9])(?:vx|wx)[\s,，]")
_clean_text = st.text(alphabet=_CLEAN_ALPHABET, min_size=0, max_size=60).filter(
    lambda text: _ASCII_WECHAT_LABEL.search(text) is None
)
_separator = st.sampled_from(["，", "。", " ", "！", "\n", ", "])


@st.composite
def _phone_numbers(draw: st.DrawFn) -> str:
    second = draw(st.integers(min_value=3, max_value=9))
    rest = draw(st.integers(min_value=0, max_value=999_999_999))
    digits = f"1{second}{rest:09d}"
    style = draw(st.sampled_from(["plain", "dashed", "spaced", "plus86"]))
    if style == "dashed":
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    if style == "spaced":
        return f"{digits[:3]} {digits[3:7]} {digits[7:]}"
    if style == "plus86":
        return f"+86{digits}"
    return digits


@st.composite
def _emails(draw: st.DrawFn) -> str:
    local = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789._", min_size=1, max_size=10)
    )
    domain = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8))
    tld = draw(st.sampled_from(["com", "org", "cn", "edu"]))
    return f"{local}@{domain}.{tld}"


@st.composite
def _id_cards(draw: st.DrawFn) -> str:
    region = draw(st.integers(min_value=110000, max_value=659999))
    year = draw(st.integers(min_value=1950, max_value=2005))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    sequence = draw(st.integers(min_value=0, max_value=999))
    return id_card_with_check(f"{region}{year}{month:02d}{day:02d}{sequence:03d}")


@st.composite
def _bank_cards(draw: st.DrawFn) -> str:
    length = draw(st.integers(min_value=16, max_value=19))
    prefix = draw(st.text(alphabet="0123456789", min_size=6, max_size=6))
    body = card_with_luhn(prefix + "0" * 12, length)
    if draw(st.booleans()):
        return " ".join(body[i : i + 4] for i in range(0, len(body), 4))
    return body


@st.composite
def _wxids(draw: st.DrawFn) -> str:
    tail = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=6, max_size=14))
    return f"wxid_{tail}"


@st.composite
def _addresses(draw: st.DrawFn) -> str:
    province = draw(st.sampled_from(["北京市", "上海市", "广东省", "浙江省", "四川省"]))
    district = draw(st.sampled_from(["朝阳区", "浦东新区", "南山区", "西湖区", ""]))
    street = draw(st.sampled_from(["建国路", "中山大道", "解放街", "人民大街", "科技路"]))
    number = draw(st.integers(min_value=1, max_value=999))
    unit = draw(st.sampled_from(["", "3栋2单元501室", "12号楼301室"]))
    return f"{province}{district}{street}{number}号{unit}"


_PII = st.one_of(
    st.tuples(st.just("phone"), _phone_numbers()),
    st.tuples(st.just("email"), _emails()),
    st.tuples(st.just("id_card"), _id_cards()),
    st.tuples(st.just("bank_card"), _bank_cards()),
    st.tuples(st.just("wxid"), _wxids()),
    st.tuples(st.just("address"), _addresses()),
)


@settings(max_examples=250, deadline=None)
@given(
    pieces=st.lists(st.tuples(_clean_text, _separator, _PII), min_size=1, max_size=4),
    tail=_clean_text,
)
def test_every_inserted_identifier_is_replaced_and_everything_else_is_preserved(
    pieces: list[tuple[str, str, tuple[str, str]]], tail: str
) -> None:
    source: list[str] = []
    expected: list[str] = []
    for clean, separator, (kind, value) in pieces:
        source.extend([clean, separator, value, separator])
        expected.extend([clean, separator, REPLACEMENTS[kind], separator])
    source.append(tail)
    expected.append(tail)
    text = "".join(source)
    result = redact(text)
    assert result.text == "".join(expected)
    for _, _, (_, value) in pieces:
        assert value not in result.text


@settings(max_examples=250, deadline=None)
@given(_clean_text)
def test_text_without_identifiers_is_never_changed(text: str) -> None:
    assert redact_text(text) == text


# ------------------------------------------- R18-A4: everyday spellings that used to slip through


@st.composite
def _dotted_phones(draw: st.DrawFn) -> str:
    second = draw(st.integers(min_value=3, max_value=9))
    rest = draw(st.integers(min_value=0, max_value=999_999_999))
    digits = f"1{second}{rest:09d}"
    prefix = draw(st.sampled_from(["", "+86 ", "+86.", "0086 "]))
    return f"{prefix}{digits[:3]}.{digits[3:7]}.{digits[7:]}"


@st.composite
def _grouped_id_cards(draw: st.DrawFn) -> str:
    card = draw(_id_cards())
    if draw(st.booleans()):
        card = card.lower()
    separator = draw(st.sampled_from([" ", "-"]))
    return separator.join([card[:6], card[6:14], card[14:]])


@st.composite
def _lane_addresses(draw: st.DrawFn) -> str:
    city = draw(st.sampled_from(["上海市", "成都市", "杭州市", ""]))
    district = draw(st.sampled_from(["浦东新区", "长宁区", "武侯区", "高新区", ""]))
    street = draw(st.sampled_from(["张江路", "愚园路", "人民南路", "天府大道", "延安西路"]))
    if draw(st.booleans()):
        middle = f"{draw(st.integers(min_value=1, max_value=2000))}弄"
    else:
        middle = draw(st.sampled_from([*"一二三四五六东南西北中", "1", "2", "12"])) + "段"
    house = draw(st.integers(min_value=1, max_value=9999))
    room = draw(st.sampled_from(["", "501室", "3栋2单元501室", "601", " 1203室"]))
    return f"{city}{district}{street}{middle}{house}号{room}"


_R18_PII = st.one_of(
    st.tuples(st.just("phone"), _dotted_phones()),
    st.tuples(st.just("id_card"), _grouped_id_cards()),
    st.tuples(st.just("address"), _lane_addresses()),
)


@settings(max_examples=300, deadline=None)
@given(
    pieces=st.lists(st.tuples(_clean_text, _separator, _R18_PII), min_size=1, max_size=4),
    tail=_clean_text,
)
def test_r18_a4_dotted_phones_grouped_id_cards_and_lane_addresses_are_replaced(
    pieces: list[tuple[str, str, tuple[str, str]]], tail: str
) -> None:
    """R18-A4 (D-766): ``138.1234.5678``, ``110105 19491231 002X``, ``愚园路1088弄5号``."""
    source: list[str] = []
    expected: list[str] = []
    for clean, separator, (kind, value) in pieces:
        source.extend([clean, separator, value, separator])
        expected.extend([clean, separator, REPLACEMENTS[kind], separator])
    source.append(tail)
    expected.append(tail)
    assert redact_text("".join(source)) == "".join(expected)


_WECHAT_LABELS = [
    "我的微信是",
    "我微信 ",
    "我微信",
    "加我微信吧 ",
    "加我微信，",
    "微信号：",
    "微信号是 ",
    "微信: ",
    "v信 ",
    "威信号",
    "wx号",
    "vx号：",
    "加我vx ",
    "vx: ",
    "wx：",
    "wechat: ",
    "WeChat ID: ",
    "微信ID ",
]


@st.composite
def _wechat_ids(draw: st.DrawFn) -> str:
    """The shape WeChat allows: a letter, then 5 to 19 letters, digits, ``_`` or ``-``."""
    first = draw(st.sampled_from(string.ascii_letters))
    rest = draw(
        st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=5, max_size=19)
    )
    return first + rest


@settings(max_examples=300, deadline=None)
@given(
    _clean_text,
    _separator,
    st.sampled_from(_WECHAT_LABELS),
    _wechat_ids(),
    _separator,
    _clean_text,
)
def test_r18_a4_a_wechat_id_after_a_natural_label_is_replaced(
    before: str, lead: str, label: str, wechat_id: str, trail: str, after: str
) -> None:
    """R18-A4 (D-766): ``我微信 abc_def12``, ``加我vx abc_def12``: the label stays, the id goes."""
    text = f"{before}{lead}{label}{wechat_id}{trail}{after}"
    assert redact_text(text) == f"{before}{lead}{label}{REPLACEMENTS['wxid']}{trail}{after}"


_LOOKALIKES = st.sampled_from(
    [
        "第3段",
        "5弄堂",
        "这段路",
        "微信上说",
        "我微信 好的",
        "我微信没电了",
        "加我微信吧",
        "微信 hello",
        "人民南路四段",
        "他住3弄",
        "第2段第5号",
        "版本 138.1234.5678.9",
        "wx 号码",
        "vxlan12345",
        "3.14159",
        "192.168.100.1234",
    ]
)


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(_clean_text, _LOOKALIKES), max_size=5), _clean_text)
def test_r18_a4_lookalikes_of_the_wider_forms_are_never_changed(
    pieces: list[tuple[str, str]], tail: str
) -> None:
    """R18-A4 (D-766): the wider patterns leave ordinary words alone (a sentence ends between)."""
    text = "".join(f"{clean}。{lookalike}。" for clean, lookalike in pieces) + tail
    assert redact_text(text) == text


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=200))
def test_redaction_never_raises_and_is_idempotent(text: str) -> None:
    once = redact_text(text)
    assert redact_text(once) == once
    numbered = ConsistentRedactor().redact_text(text)
    assert redact_text(numbered) == numbered


@settings(max_examples=200, deadline=None)
@given(_phone_numbers(), _clean_text, _clean_text)
def test_consistent_redaction_maps_one_number_to_one_token(
    number: str, before: str, after: str
) -> None:
    redactor = ConsistentRedactor()
    first = redactor.redact_text(f"{before}，{number}，{after}")
    second = redactor.redact_text(f"{after}，{number}，{before}")
    assert "[手机号#1]" in first and "[手机号#1]" in second
    assert "[手机号#2]" not in first + second


# ------------------------------------------------- R17-9: North American numbers at a sentence end


@st.composite
def _us_phones(draw: st.DrawFn) -> str:
    """A NANP number in the spellings people type (hyphen, dot, parentheses, ``+1``, plain)."""
    area = draw(st.integers(min_value=200, max_value=999))
    exchange = draw(st.integers(min_value=200, max_value=999))
    line = draw(st.integers(min_value=0, max_value=9999))
    style = draw(st.sampled_from(["hyphen", "dot", "paren", "plus1", "plus1dot", "plain"]))
    if style == "hyphen":
        return f"{area}-{exchange}-{line:04d}"
    if style == "dot":
        return f"{area}.{exchange}.{line:04d}"
    if style == "paren":
        return f"({area}) {exchange}-{line:04d}"
    if style == "plus1":
        return f"+1 {area}-{exchange}-{line:04d}"
    if style == "plus1dot":
        return f"+1 {area}.{exchange}.{line:04d}"
    return f"{area}{exchange}{line:04d}"


_SENTENCE_END = st.sampled_from(["", ".", "。", "!", "?", ",", ". ", ".\n", "。\n", ")."])
_LEAD = st.sampled_from(["", "Tel. ", "Tel.", "call me at ", "电话：", "（"])


@settings(max_examples=300, deadline=None)
@given(_clean_text, _LEAD, _us_phones(), _SENTENCE_END, _clean_text)
def test_r17_9_a_north_american_number_is_replaced_at_the_end_of_a_sentence(
    before: str, lead: str, number: str, ending: str, after: str
) -> None:
    """R17-9: ``call me at 312-555-0123.`` used to pass untouched (the dot guard was too wide)."""
    separator = "，" if before else ""
    text = f"{before}{separator}{lead}{number}{ending}{after}"
    result = redact(text)
    assert result.text == f"{before}{separator}{lead}{REPLACEMENTS['phone']}{ending}{after}"
    assert [span.kind for span in result.spans] == ["phone"]


@settings(max_examples=300, deadline=None)
@given(
    st.integers(min_value=200, max_value=999),
    st.integers(min_value=200, max_value=999),
    st.integers(min_value=0, max_value=9999),
    st.sampled_from([0, 2, 3, 4, 5, 6, 7, 8, 9]),  # "1." in front is the country code
    st.sampled_from(["-", "."]),
)
def test_r17_9_a_digit_after_the_dot_means_it_was_a_decimal_point_not_a_number(
    area: int, exchange: int, line: int, digit: int, sep: str
) -> None:
    """R17-9: the dot may end a sentence; a dot followed by a digit continues a longer number."""
    text = f"x {area}{sep}{exchange}{sep}{line:04d}.{digit}"
    assert redact_text(text) == text
    dotted_prefix = f"{digit}.{area}{sep}{exchange}{sep}{line:04d}"
    assert redact_text(dotted_prefix) == dotted_prefix


# ----------------------------------------- R17-20: the redactor of the prompts of the style model


def test_the_prompt_redactor_is_one_numbered_redactor_for_the_whole_process() -> None:
    """R17-20: the same instance every time, with the numbered placeholders of the training set."""
    with use_prompt_redactor() as fresh:
        assert prompt_redactor() is fresh is prompt_redactor()
        assert isinstance(fresh, ConsistentRedactor)
        assert redact_prompt(f"打 {mobile()}") == "打 [手机号#1]"
        assert redact_prompt(f"{us_phone()} 和 {mobile()}") == "[手机号#2] 和 [手机号#1]"
        assert fresh.counts() == {"phone": 2}


def test_borrowing_the_prompt_redactor_puts_the_previous_one_back() -> None:
    before = prompt_redactor()
    with use_prompt_redactor(ConsistentRedactor()) as borrowed:
        assert borrowed is not before and prompt_redactor() is borrowed
        with pytest.raises(RuntimeError, match="boom"), use_prompt_redactor():
            raise RuntimeError("boom")
        assert prompt_redactor() is borrowed  # the inner borrow was undone, even by an error
    assert prompt_redactor() is before


def test_numbers_asked_from_many_threads_stay_one_per_entity() -> None:
    redactor = ConsistentRedactor()
    numbers = [f"1380013{index:04d}" for index in range(40)]
    results: list[list[str]] = [[] for _ in range(8)]

    def work(slot: int) -> None:
        for number in numbers:
            results[slot].append(redactor.redact_text(f"号码{number}。"))

    threads = [threading.Thread(target=work, args=(slot,)) for slot in range(len(results))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert redactor.counts() == {"phone": len(numbers)}
    for slot in results:  # every thread saw the same token for the same number
        assert slot == results[0]
    assert len(set(results[0])) == len(numbers)


@settings(max_examples=100, deadline=None)
@given(st.lists(_phone_numbers(), min_size=1, max_size=6), _clean_text)
def test_r17_20_numbered_prompt_placeholders_are_the_ones_the_training_redactor_writes(
    numbers: list[str], filler: str
) -> None:
    """Whatever is in the prompt, the process redactor and a dataset redactor agree on the text."""
    text = "，".join(f"{filler}{number}" for number in numbers)
    with use_prompt_redactor():
        assert redact_prompt(text) == ConsistentRedactor().redact_text(text)
