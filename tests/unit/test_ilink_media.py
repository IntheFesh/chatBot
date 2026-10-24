"""iLink media: AES-128-ECB, key forms, URLs, CDN upload and download (R-CH-005, R-CH-006)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator, Iterator
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    CDN,
    TOKEN,
    USER,
    b64_hex,
    b64_raw,
    drive,
    encrypt,
    gif_bytes,
    image_bytes,
    request_json,
)
from twin.channel.ilink.http import ApiAuth, IlinkHttp
from twin.channel.ilink.media import (
    CdnClient,
    MediaCryptoError,
    MediaTransferError,
    aes_ecb_decrypt,
    aes_ecb_encrypt,
    cipher_size,
    download_url,
    encode_media_key,
    parse_hex_key,
    parse_media_key,
    sniff_image_mime,
    upload_url,
)
from twin.channel.ilink.wire import CdnMedia

KEY = bytes(range(16))
AUTH = ApiAuth(API, TOKEN)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mock:
        yield mock


# ---------------------------------------------------------------- crypto


def test_the_cipher_matches_the_published_aes_128_test_vector() -> None:
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    plain = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    assert aes_ecb_encrypt(key, plain)[:16].hex() == "3ad77bb40d7a3660a89ecaf32466ef97"


@pytest.mark.parametrize("size", [0, 1, 15, 16, 17, 31, 32, 1000])
def test_encryption_round_trips_and_has_the_documented_size(size: int) -> None:
    data = bytes((i * 7) % 256 for i in range(size))
    cipher = aes_ecb_encrypt(KEY, data)
    assert len(cipher) == cipher_size(size) == math.ceil((size + 1) / 16) * 16
    assert cipher == encrypt(KEY, data)  # same bytes as an independent implementation
    assert aes_ecb_decrypt(KEY, cipher) == data


def test_a_whole_block_of_plaintext_gets_a_full_block_of_padding() -> None:
    assert cipher_size(16) == 32 and cipher_size(0) == 16 and cipher_size(15) == 16


def test_decryption_rejects_damaged_input() -> None:
    cipher = encrypt(KEY, b"synthetic")
    with pytest.raises(MediaCryptoError, match="whole number"):
        aes_ecb_decrypt(KEY, cipher[:-1])
    with pytest.raises(MediaCryptoError, match="whole number"):
        aes_ecb_decrypt(KEY, b"")
    with pytest.raises(MediaCryptoError, match="padding"):
        aes_ecb_decrypt(bytes(16), cipher)  # a wrong key almost always breaks the padding
    with pytest.raises(MediaCryptoError, match="16 bytes"):
        aes_ecb_decrypt(b"short", cipher)
    with pytest.raises(MediaCryptoError, match="16 bytes"):
        aes_ecb_encrypt(b"short", b"x")


def test_media_keys_are_accepted_in_both_documented_spellings() -> None:
    assert parse_media_key(b64_raw(KEY)) == KEY  # base64 of the 16 raw bytes (24 characters)
    assert parse_media_key(b64_hex(KEY)) == KEY  # base64 of the 32 hex digits (44 characters)
    assert len(b64_raw(KEY)) == 24 and len(b64_hex(KEY)) == 44
    assert parse_hex_key(KEY.hex()) == KEY
    assert parse_hex_key(KEY.hex().upper()) == KEY


@pytest.mark.parametrize("bad", ["", "!!!", "QUJD", b64_raw(bytes(20)), b64_hex(KEY)[:-4]])
def test_malformed_media_keys_are_refused(bad: str) -> None:
    with pytest.raises(MediaCryptoError):
        parse_media_key(bad)


@pytest.mark.parametrize("bad", ["", "zz" * 16, "00" * 15, "00" * 17])
def test_malformed_hex_keys_are_refused(bad: str) -> None:
    with pytest.raises(MediaCryptoError):
        parse_hex_key(bad)


def test_the_key_is_sent_as_base64_of_its_hex_text() -> None:
    assert encode_media_key(KEY) == b64_hex(KEY)


def test_image_types_are_recognised_by_their_first_bytes() -> None:
    assert sniff_image_mime(image_bytes("PNG")) == "image/png"
    assert sniff_image_mime(image_bytes("JPEG")) == "image/jpeg"
    assert sniff_image_mime(gif_bytes()) == "image/gif"
    assert sniff_image_mime(image_bytes("WEBP")) == "image/webp"
    assert sniff_image_mime(image_bytes("BMP")) == "image/bmp"
    assert sniff_image_mime(b"%PDF-1.7") is None
    assert sniff_image_mime(b"") is None


# ------------------------------------------------------------------- URLs


def test_download_urls_prefer_the_full_url_then_use_the_parameter() -> None:
    full = CdnMedia(full_url="https://cdn.example/x?y=1", encrypt_query_param="P")
    assert download_url(full) == "https://cdn.example/x?y=1"
    built = download_url(CdnMedia(encrypt_query_param="a b/c+d="))
    assert built == f"{CDN}/download?encrypted_query_param=a%20b%2Fc%2Bd%3D"
    assert download_url(CdnMedia()) is None


def test_upload_urls_prefer_the_full_url_then_use_parameter_and_filekey() -> None:
    assert upload_url(upload_full_url="https://cdn.example/u", upload_param="P", filekey="K") == (
        "https://cdn.example/u"
    )
    built = upload_url(upload_full_url=None, upload_param="a b/c", filekey="k/1")
    assert built == f"{CDN}/upload?encrypted_query_param=a%20b%2Fc&filekey=k%2F1"
    assert upload_url(upload_full_url=None, upload_param=None, filekey="K") is None


# ---------------------------------------------------------- CDN transfers


@pytest.fixture
async def cdn(clock: ManualClock) -> AsyncIterator[CdnClient]:
    http = IlinkHttp()
    yield CdnClient(http, clock)
    await http.aclose()


async def test_download_returns_the_ciphertext_without_custom_headers(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    route = router.get(f"{CDN}/download").respond(200, content=b"cipher-bytes")
    assert await cdn.download(f"{CDN}/download?encrypted_query_param=P") == b"cipher-bytes"
    sent = route.calls.last.request
    assert "authorization" not in sent.headers and "x-wechat-uin" not in sent.headers
    assert parse_qs(urlsplit(str(sent.url)).query) == {"encrypted_query_param": ["P"]}


async def test_download_retries_server_errors_but_not_client_errors(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    route = router.get(f"{CDN}/download").mock(
        side_effect=[
            httpx.Response(502),
            httpx.ConnectError("x"),
            httpx.Response(200, content=b"ok"),
        ]
    )
    assert await drive(cdn.download(f"{CDN}/download?p=1"), clock) == b"ok"
    assert route.call_count == 3

    router.get(f"{CDN}/gone").respond(404)
    with pytest.raises(MediaTransferError) as excinfo:
        await cdn.download(f"{CDN}/gone")
    assert excinfo.value.status == 404 and not excinfo.value.retryable


async def test_a_download_that_never_works_reports_the_last_failure(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    router.get(f"{CDN}/down").respond(503)
    with pytest.raises(MediaTransferError, match="HTTP 503"):
        await drive(cdn.download(f"{CDN}/down"), clock)


async def test_upload_returns_the_download_parameter_header(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    route = router.post(f"{CDN}/upload").respond(200, headers={"x-encrypted-param": "DL-PARAM"})
    assert await cdn.upload(f"{CDN}/upload?encrypted_query_param=P&filekey=K", b"cipher") == (
        "DL-PARAM"
    )
    request = route.calls.last.request
    assert request.headers["content-type"] == "application/octet-stream"
    assert request.content == b"cipher"


async def test_upload_without_the_header_is_retried_and_then_fails(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    route = router.post(f"{CDN}/upload").respond(200)
    with pytest.raises(MediaTransferError, match="x-encrypted-param"):
        await drive(cdn.upload(f"{CDN}/upload", b"x"), clock)
    assert route.call_count == 3


async def test_upload_gives_up_at_once_on_a_client_error(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    route = router.post(f"{CDN}/upload").respond(400, headers={"x-error-message": "bad request"})
    with pytest.raises(MediaTransferError) as excinfo:
        await cdn.upload(f"{CDN}/upload", b"x")
    assert route.call_count == 1 and not excinfo.value.retryable and excinfo.value.status == 400


async def test_upload_survives_two_server_errors(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    route = router.post(f"{CDN}/upload").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, headers={"x-encrypted-param": "OK"}),
        ]
    )
    assert await drive(cdn.upload(f"{CDN}/upload", b"x"), clock) == "OK"
    assert route.call_count == 3


async def test_upload_image_requests_an_address_and_sends_the_encrypted_picture(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    picture = image_bytes("PNG", size=12)
    url_route = router.post(f"{API}/ilink/bot/getuploadurl").respond(
        200, json={"upload_param": "UP PARAM"}
    )
    upload_route = router.post(url__startswith=f"{CDN}/upload").respond(
        200, headers={"x-encrypted-param": "DL-1"}
    )
    uploaded = await cdn.upload_image(
        picture, auth=AUTH, to_user_id=USER, key=KEY, filekey="FILEKEY1"
    )
    body = request_json(url_route.calls.last.request)
    assert body["media_type"] == 1 and body["no_need_thumb"] is True
    assert body["to_user_id"] == USER and body["filekey"] == "FILEKEY1"
    assert body["rawsize"] == len(picture)
    assert body["rawfilemd5"] == hashlib.md5(picture, usedforsecurity=False).hexdigest()
    assert body["filesize"] == math.ceil((len(picture) + 1) / 16) * 16
    assert body["aeskey"] == KEY.hex() and len(body["aeskey"]) == 32
    assert "base_info" in body
    sent = upload_route.calls.last.request
    assert sent.content == encrypt(KEY, picture)  # exactly the protocol's ciphertext
    query = parse_qs(urlsplit(str(sent.url)).query)
    assert query == {"encrypted_query_param": ["UP PARAM"], "filekey": ["FILEKEY1"]}
    assert uploaded.download_param == "DL-1"
    assert uploaded.aes_key_b64 == b64_hex(KEY)
    assert uploaded.cipher_size == len(encrypt(KEY, picture))


async def test_upload_image_prefers_the_full_upload_url(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    router.post(f"{API}/ilink/bot/getuploadurl").respond(
        200, json={"upload_full_url": f"{CDN}/upload?encrypted_query_param=FULL&filekey=K"}
    )
    route = router.post(url__startswith=f"{CDN}/upload").respond(
        200, headers={"x-encrypted-param": "DL-2"}
    )
    result = await cdn.upload_image(b"x" * 20, auth=AUTH, to_user_id=USER)
    assert result.download_param == "DL-2"
    assert "FULL" in str(route.calls.last.request.url)


async def test_upload_image_fails_without_an_upload_address_and_never_uploads(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    router.post(f"{API}/ilink/bot/getuploadurl").respond(200, json={"ret": 0})
    cdn_route = router.post(url__startswith=f"{CDN}/upload").respond(200)
    with pytest.raises(MediaTransferError, match="no upload address"):
        await cdn.upload_image(b"x", auth=AUTH, to_user_id=USER)
    assert cdn_route.call_count == 0


async def test_upload_image_reports_an_expired_login_and_other_failures(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    route = router.post(f"{API}/ilink/bot/getuploadurl")
    route.respond(200, json={"ret": -14})
    with pytest.raises(MediaTransferError) as expired:
        await cdn.upload_image(b"x", auth=AUTH, to_user_id=USER)
    assert expired.value.auth_expired
    route.respond(500)
    with pytest.raises(MediaTransferError) as broken:
        await cdn.upload_image(b"x", auth=AUTH, to_user_id=USER)
    assert not broken.value.auth_expired
    route.mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(MediaTransferError, match="getuploadurl"):
        await cdn.upload_image(b"x", auth=AUTH, to_user_id=USER)


async def test_a_download_larger_than_the_limit_is_refused(
    router: respx.MockRouter, cdn: CdnClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("twin.channel.ilink.media.MAX_DOWNLOAD_BYTES", 10)
    router.get(f"{CDN}/big").respond(200, content=b"x" * 11)
    with pytest.raises(MediaTransferError, match="too large"):
        await cdn.download(f"{CDN}/big")


class CountingBody:
    """An endless response body that counts the chunks the client actually pulled."""

    def __init__(self, chunk: int = 1024, chunks: int = 10_000) -> None:
        self.chunk = chunk
        self.chunks = chunks
        self.pulled = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            self.pulled += 1
            yield b"x" * self.chunk


async def test_r18_a6_a_download_stops_reading_as_soon_as_it_passes_the_limit(
    router: respx.MockRouter, cdn: CdnClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-A6 (D-768): the body used to be read whole into memory and measured afterwards; the
    server decides how big it is.  Now reading stops one chunk past the limit."""
    monkeypatch.setattr("twin.channel.ilink.media.MAX_DOWNLOAD_BYTES", 10 * 1024)
    body = CountingBody()
    router.get(f"{CDN}/endless").mock(return_value=httpx.Response(200, content=body))
    with pytest.raises(MediaTransferError, match="too large") as raised:
        await cdn.download(f"{CDN}/endless")
    assert not raised.value.retryable
    assert body.pulled <= 10 + 1


async def test_r18_a6_a_declared_size_over_the_limit_is_refused_without_reading(
    router: respx.MockRouter, cdn: CdnClient
) -> None:
    """R18-A6 (D-768): ``Content-Length`` above 100 MiB: refused before the first byte."""
    body = CountingBody(chunks=4)
    router.get(f"{CDN}/huge").mock(
        return_value=httpx.Response(
            200, headers={"content-length": str(200 * 1024 * 1024)}, content=body
        )
    )
    with pytest.raises(MediaTransferError, match="too large"):
        await cdn.download(f"{CDN}/huge")
    assert body.pulled == 0


async def test_r18_a6_a_download_within_the_limit_still_arrives_whole(
    router: respx.MockRouter, cdn: CdnClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R18-A6 (D-768): exactly at the limit is fine, in many chunks."""
    monkeypatch.setattr("twin.channel.ilink.media.MAX_DOWNLOAD_BYTES", 8 * 1024)
    body = CountingBody(chunk=1024, chunks=8)
    router.get(f"{CDN}/exact").mock(return_value=httpx.Response(200, content=body))
    assert await cdn.download(f"{CDN}/exact") == b"x" * 8 * 1024


async def test_an_upload_retries_after_a_dropped_connection(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    route = router.post(f"{CDN}/upload").mock(
        side_effect=[
            httpx.ConnectError("dropped"),
            httpx.Response(200, headers={"x-encrypted-param": "OK"}),
        ]
    )
    assert await drive(cdn.upload(f"{CDN}/upload", b"x"), clock) == "OK"
    assert route.call_count == 2


def test_cdn_media_addresses_may_be_plain_http_or_https_and_nothing_else() -> None:
    """R17-46 (D-681) replaces ``test_addresses_that_are_not_https_are_never_used``.

    The old rule ("only https") was too strict for the CDN: the server may hand out an ``http://``
    ``full_url`` (the official client uses whatever it is given).  What travels there is ciphertext
    and carries no token, so ``http`` is accepted for ``full_url`` / ``upload_full_url``; any other
    scheme (or no host) still falls back to the address built from the parameter.  The hosts that
    do receive the bearer token stay https-only (``test_ilink_login``: ``baseurl``,
    ``redirect_host``)."""
    plain = CdnMedia(full_url="http://cdn.example/x?y=1", encrypt_query_param="P")
    assert download_url(plain) == "http://cdn.example/x?y=1"
    assert download_url(CdnMedia(full_url="http://cdn.example/x")) == "http://cdn.example/x"
    assert download_url(CdnMedia(full_url="HTTPS://cdn.example/x")) == "HTTPS://cdn.example/x"
    assert upload_url(upload_full_url="http://cdn.example/u", upload_param="P", filekey="K") == (
        "http://cdn.example/u"
    )
    for bad in (
        "ftp://cdn.example/x",
        "file:///etc/passwd",
        "//cdn.example/x",
        "cdn.example/x",
        "https://",
        "http:///x",
        "javascript:alert(1)",
        " http://cdn.example/x",
        "http://[::1",
    ):
        fallback = f"{CDN}/download?encrypted_query_param=P"
        assert download_url(CdnMedia(full_url=bad, encrypt_query_param="P")) == fallback, bad
        assert download_url(CdnMedia(full_url=bad)) is None, bad
        assert upload_url(upload_full_url=bad, upload_param=None, filekey="K") is None, bad
        assert upload_url(upload_full_url=bad, upload_param="P", filekey="K") == (
            f"{CDN}/upload?encrypted_query_param=P&filekey=K"
        ), bad


async def test_a_download_follows_a_redirect_to_the_final_address(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    """R17-46 (D-681): the CDN answers some requests with a redirect (302 for a GET)."""
    router.get(f"{CDN}/download").respond(302, headers={"location": "http://edge.example/blob/1"})
    final = router.get("http://edge.example/blob/1").respond(200, content=b"cipher-bytes")
    got = await drive(cdn.download(f"{CDN}/download?encrypted_query_param=P"), clock)
    assert got == b"cipher-bytes"
    sent = final.calls.last.request
    assert "authorization" not in sent.headers and "x-wechat-uin" not in sent.headers


async def test_an_upload_follows_a_method_preserving_redirect_with_its_ciphertext(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    """R17-46 (D-681): a 307 keeps the POST and its body (like ``fetch`` in the official client)."""
    router.post(f"{CDN}/upload").respond(
        307, headers={"location": "https://edge.example/upload/2?k=1"}
    )
    final = router.post("https://edge.example/upload/2").respond(
        200, headers={"x-encrypted-param": "DL-REDIRECTED"}
    )
    uploaded = await drive(cdn.upload(f"{CDN}/upload?encrypted_query_param=P", b"cipher"), clock)
    assert uploaded == "DL-REDIRECTED"
    request = final.calls.last.request
    assert request.method == "POST" and request.content == b"cipher"
    assert "authorization" not in request.headers


async def test_a_redirect_that_never_ends_fails_instead_of_looping(
    router: respx.MockRouter, cdn: CdnClient, clock: ManualClock
) -> None:
    router.get(f"{CDN}/loop").respond(302, headers={"location": f"{CDN}/loop"})
    with pytest.raises(MediaTransferError):
        await drive(cdn.download(f"{CDN}/loop"), clock)
