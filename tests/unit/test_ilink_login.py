"""QR-code login against the synthetic responses S-01 to S-08 (R-CH-003)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from PIL import Image

from tests.support.clock import ManualClock
from tests.support.ilink import API, BOT, TOKEN, USER, drive, request_json
from twin.channel.ilink import qr
from twin.channel.ilink.http import IlinkHttp
from twin.channel.ilink.login import (
    IlinkLogin,
    LoginError,
    LoginResult,
    normalise_base_url,
    redirect_base,
)

NEW_API = "https://other-host.example.com"


class ScriptedUI:
    """Records what the login shows and answers verification-number questions."""

    def __init__(self, *codes: str) -> None:
        self.codes = list(codes)
        self.shown: list[tuple[str, int, int]] = []
        self.infos: list[str] = []
        self.asked: list[bool] = []

    def show_qr(self, content: str, *, number: int, total: int) -> None:
        self.shown.append((content, number, total))

    def ask_verify_code(self, *, previous_was_wrong: bool) -> str:
        self.asked.append(previous_was_wrong)
        return self.codes.pop(0)

    def info(self, message: str) -> None:
        self.infos.append(message)


def confirmed(**overrides: Any) -> dict[str, Any]:
    body = {
        "status": "confirmed",
        "bot_token": TOKEN,
        "ilink_bot_id": BOT,
        "baseurl": "https://api-host.example.com",
        "ilink_user_id": USER,
    }
    body.update(overrides)
    return body


def code_response(number: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={"qrcode": f"QRCODE-{number}", "qrcode_img_content": f"https://qr.example/{number}"},
    )


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@pytest.fixture
async def http() -> Iterator[IlinkHttp]:  # type: ignore[misc]
    client = IlinkHttp()
    yield client
    await client.aclose()


def statuses(
    router: respx.MockRouter, *answers: dict[str, Any] | httpx.Response | Exception
) -> Any:
    side: list[Any] = []
    for answer in answers:
        if isinstance(answer, dict):
            side.append(httpx.Response(200, json=answer))
        else:
            side.append(answer)
    return router.get(f"{API}/ilink/bot/get_qrcode_status").mock(side_effect=side)


async def login(http: IlinkHttp, clock: ManualClock, ui: ScriptedUI, **kwargs: Any) -> LoginResult:
    return await drive(IlinkLogin(http, clock, ui, **kwargs).run(), clock, step_s=1.0)  # type: ignore[arg-type]


async def test_a_scan_and_confirmation_produce_the_credentials(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    qr_route = router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    status_route = statuses(router, {"status": "wait"}, {"status": "scaned"}, confirmed())
    ui = ScriptedUI()
    result = await login(http, clock, ui)
    assert result == LoginResult(TOKEN, BOT, USER, "https://api-host.example.com")
    assert ui.shown == [("https://qr.example/1", 1, 3)]
    assert any("confirm the login on your phone" in text for text in ui.infos)

    request = qr_route.calls.last.request
    assert request.method == "POST"
    assert parse_qs(urlsplit(str(request.url)).query) == {"bot_type": ["3"]}
    assert request_json(request) == {"local_token_list": []}  # no base_info, no token list
    assert "authorization" not in request.headers
    assert request.headers["authorizationtype"] == "ilink_bot_token"
    assert request.headers["x-wechat-uin"]
    assert request.headers["ilink-app-id"] == "bot"
    assert request.headers["ilink-app-clientversion"] == "132105"

    polled = status_route.calls[0].request
    assert polled.method == "GET"
    assert parse_qs(urlsplit(str(polled.url)).query) == {"qrcode": ["QRCODE-1"]}
    assert "authorization" not in polled.headers and "x-wechat-uin" not in polled.headers
    assert polled.headers["ilink-app-id"] == "bot"


async def test_an_expired_code_is_replaced_up_to_three_codes(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(
        side_effect=[code_response(1), code_response(2), code_response(3)]
    )
    status_route = statuses(router, {"status": "expired"}, {"status": "expired"}, confirmed())
    ui = ScriptedUI()
    result = await login(http, clock, ui)
    assert result.bot_token == TOKEN
    assert [number for _, number, _ in ui.shown] == [1, 2, 3]
    queried = [
        parse_qs(urlsplit(str(call.request.url)).query)["qrcode"][0] for call in status_route.calls
    ]
    assert queried == ["QRCODE-1", "QRCODE-2", "QRCODE-3"]


async def test_the_login_fails_after_the_third_code_expires(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    code_route = router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(
        side_effect=[code_response(1), code_response(2), code_response(3), code_response(4)]
    )
    statuses(router, *[{"status": "expired"}] * 4)
    with pytest.raises(LoginError, match="expired 3 times"):
        await login(http, clock, ScriptedUI())
    assert code_route.call_count == 3


async def test_a_verification_number_is_asked_for_and_sent_with_the_next_poll(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    status_route = statuses(
        router,
        {"status": "need_verifycode"},
        {"status": "need_verifycode"},  # the first number was wrong
        {"status": "scaned"},
        confirmed(),
    )
    ui = ScriptedUI("1111", "2222")
    await login(http, clock, ui)
    assert ui.asked == [False, True]
    sent = [
        parse_qs(urlsplit(str(call.request.url)).query).get("verify_code")
        for call in status_route.calls
    ]
    assert sent == [None, ["1111"], ["2222"], None]  # cleared once the phone accepted it


async def test_too_many_wrong_numbers_fetch_a_fresh_code(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    code_route = router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(
        side_effect=[code_response(1), code_response(2)]
    )
    statuses(router, {"status": "need_verifycode"}, {"status": "verify_code_blocked"}, confirmed())
    ui = ScriptedUI("0000")
    await login(http, clock, ui)
    assert code_route.call_count == 2
    assert any("Too many wrong" in text for text in ui.infos)


async def test_a_redirect_moves_the_status_polling_to_the_new_host(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    first = statuses(
        router, {"status": "scaned_but_redirect", "redirect_host": "other-host.example.com"}
    )
    second = router.get(f"{NEW_API}/ilink/bot/get_qrcode_status").mock(
        return_value=httpx.Response(200, json=confirmed())
    )
    result = await login(http, clock, ScriptedUI())
    assert first.call_count == 1 and second.call_count == 1
    assert result.api_base_url == "https://api-host.example.com"


async def test_a_redirect_without_a_host_keeps_the_current_one(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    route = statuses(router, {"status": "scaned_but_redirect"}, confirmed())
    await login(http, clock, ScriptedUI())
    assert route.call_count == 2


async def test_the_server_saying_the_bot_is_already_bound_is_an_error(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    statuses(router, {"status": "binded_redirect"})
    with pytest.raises(LoginError, match="already bound"):
        await login(http, clock, ScriptedUI())


@pytest.mark.parametrize("missing", ["bot_token", "ilink_bot_id"])
async def test_a_confirmation_without_credentials_is_a_failure(
    missing: str, router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    body = confirmed()
    del body[missing]
    statuses(router, body)
    with pytest.raises(LoginError, match="no credentials"):
        await login(http, clock, ScriptedUI())


async def test_network_trouble_while_waiting_counts_as_still_waiting(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    route = statuses(
        router,
        httpx.Response(524),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("down"),
        httpx.Response(200, text="<html>gateway</html>"),
        confirmed(),
    )
    result = await login(http, clock, ScriptedUI())
    assert result.bot_token == TOKEN and route.call_count == 5


async def test_waiting_longer_than_eight_minutes_gives_up(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    router.get(f"{API}/ilink/bot/get_qrcode_status").respond(200, json={"status": "wait"})
    with pytest.raises(LoginError, match="timed out"):
        await drive(
            IlinkLogin(http, clock, ScriptedUI(), total_timeout_s=30).run(), clock, step_s=1.0
        )


async def test_an_unreachable_login_service_is_reported(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(LoginError, match="cannot reach"):
        await login(http, clock, ScriptedUI())
    router.post(f"{API}/ilink/bot/get_bot_qrcode").respond(200, json={"qrcode": "Q"})
    with pytest.raises(LoginError, match="did not return a code"):
        await login(http, clock, ScriptedUI())


async def test_the_old_token_list_is_sent_only_when_given(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    route = router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    statuses(router, confirmed())
    tokens = tuple(f"T{i}" for i in range(12))
    await login(http, clock, ScriptedUI(), local_token_list=tokens)
    assert request_json(route.calls.last.request)["local_token_list"] == list(tokens[:10])


def test_the_api_address_from_the_server_must_be_a_plain_https_host() -> None:
    fallback = "https://fallback.example.com"
    assert normalise_base_url(None, fallback) == fallback
    assert normalise_base_url("https://api.example.com/", fallback) == "https://api.example.com"
    assert normalise_base_url("https://api.example.com:8443/x", fallback) == (
        "https://api.example.com:8443"
    )
    for bad in ("http://api.example.com", "ftp://x", "https://", "https://a b.com", "not a url"):
        with pytest.raises(LoginError, match="plain https host"):
            normalise_base_url(bad, fallback)


def test_redirect_hosts_are_validated() -> None:
    assert redirect_base("h.example.com", "https://old") == "https://h.example.com"
    assert redirect_base("https://h.example.com/", "https://old") == "https://h.example.com"
    assert redirect_base(None, "https://old") == "https://old"
    with pytest.raises(LoginError, match="invalid host"):
        redirect_base("bad host/../x", "https://old")


# -------------------------------------------------------------------- QR


def test_the_qr_matrix_is_square_has_a_quiet_zone_and_depends_on_the_content() -> None:
    matrix = qr.qr_matrix("https://qr.example/1", border=2)
    assert len(matrix) == len(matrix[0]) and len(matrix) > 21
    assert not any(matrix[0]) and not any(matrix[1])  # the border is light
    assert any(any(row) for row in matrix)
    assert qr.qr_matrix("https://qr.example/2", border=2) != matrix


def test_the_terminal_rendering_uses_two_module_rows_per_line() -> None:
    matrix = qr.qr_matrix("https://qr.example/1", border=2)
    text = qr.render_terminal("https://qr.example/1", border=2)
    lines = text.split("\n")
    assert len(lines) == (len(matrix) + 1) // 2
    assert all(len(line) == len(matrix[0]) for line in lines)
    assert set(text) <= {"█", "▀", "▄", " ", "\n"}
    first_dark = next(i for i, row in enumerate(matrix) if any(row))
    assert lines[first_dark // 2].strip()  # the first dark module shows up in its text line


def test_the_png_is_written_locally_and_is_a_valid_picture(tmp_path: Path) -> None:
    folder = tmp_path / "tmp"
    path = qr.save_png("https://qr.example/1", folder)
    assert path.parent == folder and path.name.startswith(qr.PNG_PREFIX)
    with Image.open(path) as image:
        assert image.format == "PNG" and image.width == image.height >= 100
    other = qr.save_png("https://qr.example/2", folder)
    assert other != path
    qr.remove_old_pngs(folder)
    assert not list(folder.glob("*.png"))
    qr.remove_old_pngs(tmp_path / "missing")  # nothing to do, nothing raised


def test_the_viewer_launcher_is_injectable_and_failures_are_quiet(tmp_path: Path) -> None:
    seen: list[Path] = []

    def opener(path: Path) -> bool:
        seen.append(path)
        return True

    target = tmp_path / "x.png"
    assert qr.open_in_viewer(target, opener=opener) is True and seen == [target]
    assert qr.open_in_viewer(target, opener=lambda _p: False) is False


async def test_an_unknown_status_is_ignored_and_the_wait_continues(
    router: respx.MockRouter, http: IlinkHttp, clock: ManualClock
) -> None:
    router.post(f"{API}/ilink/bot/get_bot_qrcode").mock(side_effect=[code_response(1)])
    route = statuses(router, {"status": "something_new"}, {"status": "wait"}, confirmed())
    result = await login(http, clock, ScriptedUI())
    assert result.bot_token == TOKEN and route.call_count == 3
