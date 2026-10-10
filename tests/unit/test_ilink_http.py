"""Headers, error mapping, wire models and log privacy of the iLink HTTP layer."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
import respx

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    CTX,
    TOKEN,
    USER,
    Harness,
    image_item,
    make_harness,
    message,
    now_ms,
    text_item,
    updates,
)
from twin.channel.base import OutboundResult
from twin.channel.ilink.http import (
    IlinkHttp,
    IlinkHttpError,
    IlinkProtocolError,
    IlinkResponse,
    IlinkTransportError,
    base_info,
    random_uin,
)
from twin.channel.ilink.poller import system_jitter
from twin.channel.ilink.wire import WireMessage, as_text
from twin.ops.logging import JsonFormatter, configure_logging, shutdown_logging
from twin.storage.db import Database

SECRET_WORDS = "绝密的聊天内容-QQ"
URL = f"{API}/ilink/bot/getupdates"


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def http() -> AsyncIterator[IlinkHttp]:
    client = IlinkHttp()
    yield client
    await client.aclose()


def parsed(data: dict[str, object]) -> IlinkResponse:
    return IlinkResponse(200, data, httpx.Headers())


# --------------------------------------------------------- response codes


@pytest.mark.parametrize(
    ("data", "code", "expired"),
    [
        ({}, None, False),
        ({"ret": 0}, None, False),
        ({"ret": 0, "errcode": 0}, None, False),
        ({"ret": -14}, -14, True),
        ({"errcode": -14}, -14, True),
        ({"ret": 0, "errcode": -14}, -14, True),
        ({"ret": -14, "errcode": 5}, -14, True),
        ({"ret": 5, "errcode": -14}, -14, True),
        ({"ret": -2}, -2, False),
        ({"errcode": 7, "ret": 1}, 7, False),  # errcode is reported before ret
        ({"ret": True}, None, False),  # booleans and strings are not error numbers
        ({"ret": "-14"}, None, False),
        ({"ret": 1.0}, 1, False),
    ],
)
def test_the_error_number_follows_the_documented_precedence(
    data: dict[str, object], code: int | None, expired: bool
) -> None:
    response = parsed(data)
    assert response.error_code == code and response.auth_expired is expired
    assert response.ok is (code is None)


def test_the_error_message_is_only_text() -> None:
    assert parsed({"errmsg": "ERRMSG"}).errmsg == "ERRMSG"
    assert parsed({"errmsg": ""}).errmsg is None and parsed({"errmsg": 5}).errmsg is None


# ---------------------------------------------------------------- requests


async def test_every_request_carries_the_documented_headers_and_a_fresh_uin(
    api: respx.MockRouter, http: IlinkHttp
) -> None:
    route = api.post(URL).respond(200, json={})
    for _ in range(3):
        await http.post("getupdates", {"x": 1}, base_url=API + "/", token=TOKEN, timeout_s=5)
    uins = {call.request.headers["x-wechat-uin"] for call in route.calls}
    assert len(uins) == 3  # a new number for every request
    for call in route.calls:
        headers = call.request.headers
        assert headers["authorization"] == f"Bearer {TOKEN}"
        assert headers["authorizationtype"] == "ilink_bot_token"
        assert headers["ilink-app-id"] == "bot" and headers["ilink-app-clientversion"] == "132105"
        assert headers["content-type"] == "application/json"
        assert call.request.url == URL  # the trailing slash of the base is handled


def test_the_uin_is_the_base64_of_a_32_bit_decimal_number() -> None:
    for _ in range(20):
        number = int(base64.b64decode(random_uin()).decode("ascii"))
        assert 0 <= number < 2**32


def test_base_info_names_this_program_honestly() -> None:
    info = base_info()
    assert info["bot_agent"].startswith("WechatTwin/") and "OpenClaw" not in info["bot_agent"]
    assert info["channel_version"] and info["channel_version"] in info["bot_agent"]


async def test_a_request_without_a_token_sends_no_bearer(
    api: respx.MockRouter, http: IlinkHttp
) -> None:
    route = api.post(URL).respond(200, json={})
    await http.post("getupdates", {}, base_url=API, token=None, timeout_s=5, with_base_info=False)
    assert "authorization" not in route.calls.last.request.headers
    assert b"base_info" not in route.calls.last.request.content


# ----------------------------------------------------------------- errors


async def test_transport_problems_say_whether_the_request_left_the_machine(
    api: respx.MockRouter, http: IlinkHttp
) -> None:
    cases = [
        (httpx.ConnectError("x"), False, False),
        (httpx.ConnectTimeout("x"), False, True),
        (httpx.ReadTimeout("x"), True, True),
        (httpx.WriteTimeout("x"), True, True),
        (httpx.ReadError("x"), True, False),
        (httpx.RemoteProtocolError("x"), True, False),
    ]
    for failure, sent, timeout in cases:
        api.post(URL).mock(side_effect=failure)
        with pytest.raises(IlinkTransportError) as caught:
            await http.post("getupdates", {}, base_url=API, token=TOKEN, timeout_s=5)
        assert (caught.value.request_sent, caught.value.timeout) == (sent, timeout), failure


async def test_http_errors_and_unreadable_answers_are_distinct_exceptions(
    api: respx.MockRouter, http: IlinkHttp
) -> None:
    api.post(URL).respond(502)
    with pytest.raises(IlinkHttpError) as status:
        await http.post("getupdates", {}, base_url=API, token=TOKEN, timeout_s=5)
    assert status.value.status == 502 and str(status.value) == "getupdates: HTTP 502"
    for body in ("<html>", "[1, 2]", "42"):
        api.post(URL).respond(200, text=body)
        with pytest.raises(IlinkProtocolError):
            await http.post("getupdates", {}, base_url=API, token=TOKEN, timeout_s=5)
    api.post(URL).respond(200, content=b"  \n")
    answer = await http.post("getupdates", {}, base_url=API, token=TOKEN, timeout_s=5)
    assert answer.data == {} and answer.ok


async def test_a_client_passed_in_is_not_closed_by_the_wrapper() -> None:
    client = httpx.AsyncClient()
    wrapper = IlinkHttp(client)
    await wrapper.aclose()
    assert not client.is_closed
    await client.aclose()


# ------------------------------------------------------------ wire models


def test_unknown_fields_are_ignored_and_ids_are_exact() -> None:
    raw = message(text_item("x", future_field=[1, 2]), mid=18_446_744_073_709_551_615)
    raw["surprise"] = {"nested": True}
    wire = WireMessage.model_validate(raw)
    assert wire.dedup_id() == "18446744073709551615"
    assert as_text(0) == "0" and as_text("") is None and as_text(None) is None


def test_the_fallback_ids_are_stable_and_ignore_the_context_token() -> None:
    first = WireMessage.model_validate(message(text_item("甲"), mid=None, context_token="A"))
    same = WireMessage.model_validate(message(text_item("甲"), mid=None, context_token="B"))
    other = WireMessage.model_validate(message(text_item("乙"), mid=None))
    assert first.dedup_id() == same.dedup_id() != other.dedup_id()
    with_item_id = WireMessage.model_validate(
        message(text_item("甲", msg_id=123), image_item("P"), mid=None)
    )
    assert with_item_id.dedup_id() == "123"


def test_the_jitter_factor_is_between_one_half_and_one_and_a_half() -> None:
    values = [system_jitter() for _ in range(200)]
    assert all(0.5 <= value < 1.5 for value in values) and len(set(values)) > 100


def test_a_failure_result_cannot_claim_to_be_ok() -> None:
    from twin.channel.base import OutboundKind

    with pytest.raises(ValueError, match="cannot have the kind OK"):
        OutboundResult.failure(OutboundKind.OK, "nonsense")


# --------------------------------------------------------------- privacy


class Collector(logging.Handler):
    """Keeps every record as the JSON line the production file handler would write."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.formatter_ = JsonFormatter()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.formatter_.format(record))


@pytest.fixture
def collected() -> Iterator[Collector]:
    """Production logging setup (DEBUG, third-party loggers quieted) plus a collector."""
    configure_logging(None, level="DEBUG", console=False)
    collector = Collector()
    twin_logger, root = logging.getLogger("twin"), logging.getLogger()
    twin_logger.addHandler(collector)
    root.addHandler(collector)
    yield collector
    twin_logger.removeHandler(collector)
    root.removeHandler(collector)
    shutdown_logging()


def test_the_production_logging_setup_silences_the_http_client_url_log() -> None:
    configure_logging(None, console=False)
    try:
        for name in ("httpx", "httpcore"):
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        shutdown_logging()


async def test_logs_never_contain_message_text_tokens_or_ids(
    api: respx.MockRouter,
    db: Database,
    clock: ManualClock,
    tmp_path: Path,
    collected: Collector,
) -> None:
    harness: Harness = make_harness(db, clock, tmp_path, RecordingAlerts())
    harness.login()
    harness.bind()
    try:
        batch = [
            message(text_item(SECRET_WORDS), mid=61, created_ms=now_ms(clock)),
            message({"type": 99, "x": SECRET_WORDS}, mid=62, created_ms=now_ms(clock)),
            message(image_item("PARAM-SECRET", hex_key="00" * 16), mid=63),
        ]
        api.get(url__startswith="https://novac2c.cdn.weixin.qq.com").respond(404)
        api.post(URL).mock(
            side_effect=[
                httpx.Response(200, json=updates(batch, cursor="CURSOR-SECRET")),
                httpx.Response(500),
                httpx.Response(200, json={"ret": -14, "errmsg": f"token {TOKEN} bad"}),
            ]
        )
        api.post(f"{API}/ilink/bot/sendmessage").mock(
            side_effect=[
                httpx.Response(200, json={"ret": 3, "errmsg": SECRET_WORDS}),
                httpx.Response(500),
            ]
        )
        for _ in range(3):
            await harness.channel.poll_once()
        harness.store.mark_auth_ok()
        await harness.channel.send_text(SECRET_WORDS)
        await harness.channel.send_text(SECRET_WORDS)
    finally:
        await harness.channel.stop()
    logged = "\n".join(collected.lines)
    assert logged  # something was logged, so the absence below means something
    for secret in (SECRET_WORDS, TOKEN, CTX, USER, "PARAM-SECRET", "CURSOR-SECRET"):
        assert secret not in logged, secret
