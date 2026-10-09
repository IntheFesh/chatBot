"""iLink media: AES-128-ECB encryption, CDN download and upload (protocol document section 7).

The CDN holds files encrypted with AES-128-ECB (PKCS7 padding, one key per file, no
chunking).  Downloaded media carry their key in the message; uploads use a fresh random
key that is sent in two spellings: as a 32-digit hex string to ``getuploadurl`` and as
``base64(hex text)`` in the message item.  The pure functions here are tested by encrypting
synthetic pictures with the very same algorithm and decrypting them again.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from twin.channel.ilink.http import (
    TIMEOUT_SEND_S,
    ApiAuth,
    IlinkError,
    IlinkHttp,
    IlinkTransportError,
)
from twin.channel.ilink.wire import CDN_BASE, MEDIA_IMAGE, CdnMedia
from twin.clock import Clock

BLOCK = 16
UPLOAD_ATTEMPTS = 3
CDN_TIMEOUT_S = 60.0
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
_HEX = frozenset("0123456789abcdefABCDEF")


class MediaCryptoError(Exception):
    """A key has the wrong shape or the ciphertext does not decrypt."""


class MediaTransferError(Exception):
    """A CDN download or upload failed; ``retryable`` says whether trying again can help."""

    def __init__(
        self,
        reason: str,
        *,
        retryable: bool = True,
        status: int | None = None,
        auth_expired: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable
        self.status = status
        self.auth_expired = auth_expired


# ------------------------------------------------------------------ crypto


def cipher_size(plain_size: int) -> int:
    """Ciphertext length with PKCS7: ``ceil((n + 1) / 16) * 16``."""
    return (plain_size // BLOCK + 1) * BLOCK


def aes_ecb_encrypt(key: bytes, data: bytes) -> bytes:
    if len(key) != BLOCK:
        raise MediaCryptoError("an AES-128 key has 16 bytes")
    padder = padding.PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()  # noqa: S305
    return encryptor.update(padded) + encryptor.finalize()


def aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    if len(key) != BLOCK:
        raise MediaCryptoError("an AES-128 key has 16 bytes")
    if not data or len(data) % BLOCK:
        raise MediaCryptoError("the ciphertext is not a whole number of 16-byte blocks")
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()  # noqa: S305
    padded = decryptor.update(data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    try:
        return unpadder.update(padded) + unpadder.finalize()
    except ValueError:
        raise MediaCryptoError("the padding is invalid (wrong key or damaged file)") from None


def sniff_image_mime(data: bytes) -> str | None:
    """The image type by its first bytes (PNG, JPEG, GIF, WebP, BMP), or ``None``."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def parse_hex_key(value: str) -> bytes:
    """``image_item.aeskey``: 32 hex digits for the 16 raw key bytes."""
    text = value.strip()
    if len(text) != 32 or not set(text) <= _HEX:
        raise MediaCryptoError("an aeskey is 32 hexadecimal digits")
    return bytes.fromhex(text)


def parse_media_key(value: str) -> bytes:
    """``media.aes_key``: base64 of either the 16 raw bytes or of the 32-digit hex text."""
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise MediaCryptoError("the media key is not valid base64") from None
    if len(raw) == BLOCK:
        return raw
    if len(raw) == 32 and set(raw.decode("ascii", errors="replace")) <= _HEX:
        return bytes.fromhex(raw.decode("ascii"))
    raise MediaCryptoError("the media key must decode to 16 bytes or to 32 hex digits")


def encode_media_key(key: bytes) -> str:
    """The key as sent in an outgoing item: ``base64(hex(key) as ASCII text)``."""
    return base64.b64encode(key.hex().encode("ascii")).decode("ascii")


# -------------------------------------------------------------------- URLs


def download_url(media: CdnMedia, *, cdn_base: str = CDN_BASE) -> str | None:
    """The full https URL if the message gave one, else built from ``encrypt_query_param``."""
    if media.full_url and media.full_url.startswith("https://"):
        return media.full_url
    if media.encrypt_query_param:
        return (
            f"{cdn_base}/download?encrypted_query_param={quote(media.encrypt_query_param, safe='')}"
        )
    return None


def upload_url(
    *, upload_full_url: str | None, upload_param: str | None, filekey: str, cdn_base: str = CDN_BASE
) -> str | None:
    if upload_full_url and upload_full_url.startswith("https://"):
        return upload_full_url
    if upload_param:
        return (
            f"{cdn_base}/upload?encrypted_query_param={quote(upload_param, safe='')}"
            f"&filekey={quote(filekey, safe='')}"
        )
    return None


# ----------------------------------------------------------------- transfer


@dataclass(frozen=True)
class UploadedImage:
    """What a ``sendmessage`` image item needs after a successful upload."""

    download_param: str
    aes_key_b64: str
    cipher_size: int


class CdnClient:
    """Downloads and uploads encrypted media on the CDN."""

    def __init__(self, http: IlinkHttp, clock: Clock, *, cdn_base: str = CDN_BASE) -> None:
        self._http = http
        self._clock = clock
        self._cdn_base = cdn_base

    async def download(self, url: str, *, attempts: int = 3) -> bytes:
        """GET the ciphertext (no custom headers); 4xx is final, other failures retry."""
        last = MediaTransferError("no attempt was made")
        for attempt in range(attempts):
            try:
                response = await self._http.client.get(url, timeout=CDN_TIMEOUT_S)
            except httpx.HTTPError as exc:
                last = MediaTransferError(type(exc).__name__)
            else:
                if 200 <= response.status_code < 300:
                    if len(response.content) > MAX_DOWNLOAD_BYTES:
                        raise MediaTransferError("the file is too large", retryable=False)
                    return response.content
                retryable = response.status_code >= 500
                last = MediaTransferError(
                    f"HTTP {response.status_code}", retryable=retryable, status=response.status_code
                )
                if not retryable:
                    raise last
            if attempt + 1 < attempts:
                await self._clock.sleep(float(attempt + 1))
        raise last

    async def upload(self, url: str, ciphertext: bytes) -> str:
        """POST the ciphertext; returns the ``x-encrypted-param`` header (the download parameter).

        4xx answers are final; anything else (including a 200 without the header) is tried
        up to three times in total, like the official client.
        """
        last = MediaTransferError("no attempt was made")
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                response = await self._http.client.post(
                    url,
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=CDN_TIMEOUT_S,
                )
            except httpx.HTTPError as exc:
                last = MediaTransferError(type(exc).__name__)
            else:
                if 400 <= response.status_code < 500:
                    raise MediaTransferError(
                        f"HTTP {response.status_code}", retryable=False, status=response.status_code
                    )
                param = str(response.headers.get("x-encrypted-param", ""))
                if 200 <= response.status_code < 300 and param:
                    return param
                last = MediaTransferError(
                    f"HTTP {response.status_code}"
                    if not 200 <= response.status_code < 300
                    else "the answer carries no x-encrypted-param header",
                    status=response.status_code,
                )
            if attempt + 1 < UPLOAD_ATTEMPTS:
                await self._clock.sleep(float(attempt + 1))
        raise last

    async def upload_image(
        self,
        data: bytes,
        *,
        auth: ApiAuth,
        to_user_id: str,
        key: bytes | None = None,
        filekey: str | None = None,
    ) -> UploadedImage:
        """``getuploadurl`` then the CDN upload (section 7.3 steps 1-4)."""
        key = key or secrets.token_bytes(BLOCK)
        filekey = filekey or secrets.token_hex(BLOCK)
        ciphertext = aes_ecb_encrypt(key, data)
        body = {
            "filekey": filekey,
            "media_type": MEDIA_IMAGE,
            "to_user_id": to_user_id,
            "rawsize": len(data),
            "rawfilemd5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "filesize": len(ciphertext),
            "no_need_thumb": True,
            "aeskey": key.hex(),
        }
        try:
            answer = await self._http.post(
                "getuploadurl",
                body,
                base_url=auth.base_url,
                token=auth.token,
                timeout_s=TIMEOUT_SEND_S,
            )
        except IlinkTransportError as exc:
            raise MediaTransferError(f"getuploadurl: {exc.reason}") from None
        except IlinkError as exc:
            raise MediaTransferError(str(exc)) from None
        if answer.auth_expired:
            raise MediaTransferError(
                "getuploadurl: the login is no longer valid", retryable=False, auth_expired=True
            )
        url = upload_url(
            upload_full_url=_text(answer.data.get("upload_full_url")),
            upload_param=_text(answer.data.get("upload_param")),
            filekey=filekey,
            cdn_base=self._cdn_base,
        )
        if url is None:
            raise MediaTransferError("getuploadurl returned no upload address", retryable=False)
        param = await self.upload(url, ciphertext)
        return UploadedImage(param, encode_media_key(key), len(ciphertext))


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
