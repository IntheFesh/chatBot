"""Image input for vision requests (R-LLM-004).

* The format is read from the file header (JPEG, PNG, GIF, WebP) - never from a file name or
  a declared type - and becomes the MIME type of the ``data:`` URL.
* Images are sent unchanged unless they exceed an API limit (8192 px per side, 4096 px when a
  request has 15 or more images, 32 MiB per inline image); then they are scaled down with
  Pillow.  An animated GIF is kept as a GIF (all frames scaled) when it must be shrunk and
  reduced to its first frame, as PNG, only if that is not enough or if the probe found that
  the API rejects GIFs.
* ``detail`` is optional; it is dropped when the M0 probe found that the API rejects it.
* Images may only appear in ``user`` messages; :func:`validate_image_placement` raises for
  anything else (the API would answer 400).

Limits and behaviour were checked against the vision page of the DeepSeek documentation on
2026-10-09 (see :mod:`twin.llm.official`).
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from twin.llm.capabilities import DOCUMENTED, LlmCapabilities
from twin.llm.errors import ImagePlacementError, ImageTooLargeError, UnsupportedImageError
from twin.llm.official import (
    DETAIL_VALUES,
    MAX_IMAGE_SIDE_PX,
    MAX_IMAGE_SIDE_PX_MANY,
    MAX_INLINE_IMAGE_BYTES,
)
from twin.llm.types import ChatMessage, ContentPart, ImagePart, ImageUrl, TextPart

if TYPE_CHECKING:
    from twin.storage.media import MediaStore

DETAIL_STICKER = "low"
DETAIL_PHOTO = "auto"
# an inline image is sent as base64, which is 4/3 of the file size
MAX_RAW_IMAGE_BYTES = MAX_INLINE_IMAGE_BYTES * 3 // 4
MIN_SIDE_PX = 32
SHRINK_STEP = 0.75
MAX_SHRINK_ROUNDS = 12
JPEG_QUALITY = 90

_Frame = Image.Image


def sniff_mime(data: bytes) -> str:
    """MIME type from the file header; raises :class:`UnsupportedImageError` if unknown."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise UnsupportedImageError("not a JPEG, PNG, GIF or WebP image (checked the file header)")


@dataclass(frozen=True)
class MediaRef:
    """An image held in the encrypted media store."""

    store: MediaStore
    sha256: str


@dataclass(frozen=True)
class ImageInput:
    """An image to show the model: raw bytes, a file, or a media store entry."""

    source: bytes | Path | MediaRef
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.detail is not None and self.detail not in DETAIL_VALUES:
            raise ValueError(f"detail must be one of {', '.join(DETAIL_VALUES)}")

    @classmethod
    def from_bytes(cls, data: bytes, detail: str | None = None) -> ImageInput:
        return cls(data, detail)

    @classmethod
    def from_path(cls, path: Path | str, detail: str | None = None) -> ImageInput:
        return cls(Path(path), detail)

    @classmethod
    def from_media(cls, store: MediaStore, sha256: str, detail: str | None = None) -> ImageInput:
        return cls(MediaRef(store, sha256), detail)

    def read(self) -> bytes:
        """The raw bytes (blocking: file or encrypted media read)."""
        if isinstance(self.source, bytes):
            return self.source
        if isinstance(self.source, Path):
            return self.source.read_bytes()
        return self.source.store.read_bytes(self.source.sha256)


@dataclass(frozen=True)
class PreparedImage:
    """An image ready for the request body."""

    data: bytes
    mime: str
    width: int
    height: int
    frames: int
    detail: str | None
    note: str | None = None  # what was changed locally, if anything

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"

    def part(self) -> ImagePart:
        image_url: ImageUrl = {"url": self.data_url()}
        if self.detail is not None:
            image_url["detail"] = self.detail
        return {"type": "image_url", "image_url": image_url}


# ------------------------------------------------------------------ preparation


def _open(data: bytes) -> _Frame:
    try:
        image = Image.open(io.BytesIO(data))
        if getattr(image, "n_frames", 1) == 1:
            image.load()  # a still image must decode completely, a truncated file is an error
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnsupportedImageError(f"cannot decode the image: {type(exc).__name__}") from exc
    return image


def _png_bytes(image: _Frame) -> bytes:
    buffer = io.BytesIO()
    frame = image.convert("RGBA") if image.mode not in ("RGB", "RGBA", "L", "LA", "P") else image
    frame.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _first_frame_png(image: _Frame) -> bytes:
    image.seek(0)
    return _png_bytes(image.copy())


def _scaled(size: tuple[int, int], factor: float) -> tuple[int, int]:
    return max(1, int(size[0] * factor)), max(1, int(size[1] * factor))


def _encode_static(image: _Frame, fmt: str, size: tuple[int, int]) -> bytes:
    upright = ImageOps.exif_transpose(image) or image
    resized = upright.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if fmt == "JPEG":
        resized.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    elif fmt == "WEBP":
        resized.save(buffer, format="WEBP", quality=JPEG_QUALITY)
    else:
        resized.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _encode_animated(image: _Frame, fmt: str, size: tuple[int, int]) -> bytes:
    frames: list[_Frame] = []
    durations: list[int] = []
    for frame in ImageSequence.Iterator(image):
        durations.append(int(frame.info.get("duration", 100)))
        resized = frame.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        frames.append(
            resized.convert("P", palette=Image.Palette.ADAPTIVE) if fmt == "GIF" else resized
        )
    buffer = io.BytesIO()
    options: dict[str, object] = {"duration": durations, "loop": image.info.get("loop", 0)}
    if fmt == "GIF":
        options["optimize"] = True
    frames[0].save(buffer, format=fmt, save_all=True, append_images=frames[1:], **options)
    return buffer.getvalue()


def prepare_image(
    raw: bytes,
    *,
    detail: str | None = None,
    capabilities: LlmCapabilities = DOCUMENTED,
    many_images: bool = False,
) -> PreparedImage:
    """Check ``raw``, shrink it if an API limit requires, and describe the result.

    ``many_images`` is true when the request holds 15 or more images (the side limit drops).
    """
    mime = sniff_mime(raw)
    image = _open(raw)
    width, height = image.size
    frames = int(getattr(image, "n_frames", 1))
    chosen_detail = detail if capabilities.detail_supported else None
    note: str | None = None

    if mime == "image/gif" and frames > 1 and not capabilities.gif_supported:
        png = _first_frame_png(image)
        reopened = _open(png)
        return PreparedImage(
            png,
            "image/png",
            reopened.size[0],
            reopened.size[1],
            1,
            chosen_detail,
            "GIF reduced to its first frame (the API rejected GIFs in the probe)",
        )

    side_limit = MAX_IMAGE_SIDE_PX_MANY if many_images else MAX_IMAGE_SIDE_PX
    if max(width, height) <= side_limit and len(raw) <= MAX_RAW_IMAGE_BYTES:
        return PreparedImage(raw, mime, width, height, frames, chosen_detail)

    fmt = {"image/jpeg": "JPEG", "image/png": "PNG", "image/gif": "GIF", "image/webp": "WEBP"}[mime]
    animated = frames > 1
    factor = min(1.0, side_limit / max(width, height))
    size = _scaled((width, height), factor)
    data = raw
    for _ in range(MAX_SHRINK_ROUNDS):
        data = _encode_animated(image, fmt, size) if animated else _encode_static(image, fmt, size)
        if len(data) <= MAX_RAW_IMAGE_BYTES:
            note = f"scaled down from {width}x{height} to {size[0]}x{size[1]}"
            return PreparedImage(data, mime, size[0], size[1], frames, chosen_detail, note)
        if min(size) <= MIN_SIDE_PX:
            break
        size = _scaled(size, SHRINK_STEP)
    if animated:  # last resort for a huge animation: the first frame only
        png = _first_frame_png(image)
        if len(png) <= MAX_RAW_IMAGE_BYTES:
            side = min(side_limit, max(width, height))
            still = _open(png)
            if max(still.size) > side:
                png = _encode_static(still, "PNG", _scaled(still.size, side / max(still.size)))
            final = _open(png)
            return PreparedImage(
                png,
                "image/png",
                final.size[0],
                final.size[1],
                1,
                chosen_detail,
                "animation reduced to its first frame to fit the size limit",
            )
    raise ImageTooLargeError(
        f"image of {width}x{height} px and {len(raw)} bytes still exceeds the API limits"
    )


# ------------------------------------------------------------ message helpers


def user_message(text: str, images: Sequence[PreparedImage]) -> ChatMessage:
    """A user message with the text first and the images after it."""
    parts: list[ContentPart] = []
    if text:
        text_part: TextPart = {"type": "text", "text": text}
        parts.append(text_part)
    parts.extend(image.part() for image in images)
    return {"role": "user", "content": parts}


def attach_images(
    messages: Sequence[ChatMessage], images: Sequence[PreparedImage]
) -> list[ChatMessage]:
    """Copy of ``messages`` with ``images`` added to the last user message."""
    result = list(messages)
    if not images:
        return result
    for index in range(len(result) - 1, -1, -1):
        message = result[index]
        if message["role"] != "user":
            continue
        content = message["content"]
        parts: list[ContentPart] = (
            [{"type": "text", "text": content}] if isinstance(content, str) and content else []
        )
        if not isinstance(content, str):
            parts = list(content)
        parts.extend(image.part() for image in images)
        result[index] = {"role": "user", "content": parts}
        return result
    raise ImagePlacementError("images need a user message to go into; there is none")


def count_images(messages: Sequence[ChatMessage]) -> int:
    return sum(
        1
        for message in messages
        if not isinstance(message["content"], str)
        for part in message["content"]
        if part["type"] == "image_url"
    )


def validate_image_placement(messages: Sequence[ChatMessage]) -> None:
    """Raise :class:`ImagePlacementError` if an image sits in a system or assistant message."""
    for index, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "user" or isinstance(content, str):
            continue
        if any(part["type"] == "image_url" for part in content):
            raise ImagePlacementError(
                f"message {index} has role {message['role']!r}: "
                "images are only allowed in user messages"
            )


def estimate_encoded_bytes(images: Sequence[PreparedImage]) -> int:
    """Size of the images inside a request body (base64)."""
    return sum(4 * math.ceil(len(image.data) / 3) for image in images)
