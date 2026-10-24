"""The synthetic pictures the probe sends (R-CH-009 step 2, R-SAFE-006).

They are drawn with Pillow from nothing but colours and the word TEST: no photograph and no
file from the user's data is involved.  Each carries "[TEST]" in its pixels, so the person
looking at the phone can tell a probe picture from anything else.  The GIF has several frames
(a square moves across), so "is it animated?" has an answer the eye can check.

The SHA-256 of every picture is registered in the channel's probe manifest before it is sent
(:class:`twin.channel.policy.ProbeImageManifest`); that is the only way these bytes pass the
outbound image allow list.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

WIDTH = 320
HEIGHT = 240
GIF_FRAMES = 8
GIF_FRAME_MS = 160

NAMES = ("jpg", "png", "gif")


@dataclass(frozen=True)
class ProbeImage:
    name: str  # "jpg", "png" or "gif"
    mime: str
    data: bytes
    frames: int


def _font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=30)
    except (TypeError, OSError):  # a Pillow build without FreeType: the small bitmap font
        return ImageFont.load_default()


def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    for row in range(HEIGHT):
        mix = row / (HEIGHT - 1)
        colour = tuple(round(a + (b - a) * mix) for a, b in zip(top, bottom, strict=True))
        draw.line([(0, row), (WIDTH, row)], fill=(colour[0], colour[1], colour[2]))
    return image


def _label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    draw.text(
        (16, 16), text, fill=(255, 255, 255), font=_font(), stroke_width=2, stroke_fill=(0, 0, 0)
    )


def make_jpeg() -> ProbeImage:
    image = _gradient((30, 90, 200), (240, 200, 60))
    _label(image, "[TEST] JPG")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return ProbeImage("jpg", "image/jpeg", buffer.getvalue(), 1)


def make_png() -> ProbeImage:
    image = _gradient((200, 40, 90), (40, 180, 120))
    draw = ImageDraw.Draw(image)
    draw.ellipse((90, 90, 230, 210), outline=(255, 255, 255), width=6)
    _label(image, "[TEST] PNG")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return ProbeImage("png", "image/png", buffer.getvalue(), 1)


def make_gif() -> ProbeImage:
    frames: list[Image.Image] = []
    span = WIDTH - 80
    for index in range(GIF_FRAMES):
        frame = _gradient((20, 20, 60), (60, 60, 140))
        left = 20 + round(span * index / (GIF_FRAMES - 1))
        ImageDraw.Draw(frame).rectangle((left, 110, left + 40, 170), fill=(250, 220, 40))
        _label(frame, "[TEST] GIF")
        frames.append(frame.quantize(colors=64))
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
    )
    return ProbeImage("gif", "image/gif", buffer.getvalue(), GIF_FRAMES)


def make_probe_images() -> dict[str, ProbeImage]:
    """All three pictures by name."""
    return {image.name: image for image in (make_jpeg(), make_png(), make_gif())}
