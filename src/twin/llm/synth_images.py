"""Programmatically drawn images for the M0 probes (R-LLM-013, R-CH-009).

The probes must never use a real photo.  These functions draw simple scenes - a coloured shape
on a coloured background - with Pillow, so the content is known and the answer to "what shape
and colour do you see?" can be checked.  Output is deterministic for the same arguments.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

SCENE_SHAPE = "circle"
SCENE_COLOR = "red"


def _scene(width: int, height: int, *, shift: int = 0) -> Image.Image:
    image = Image.new("RGB", (width, height), (30, 60, 200))
    draw = ImageDraw.Draw(image)
    radius = max(2, min(width, height) // 4)
    centre_x = width // 2 + shift
    centre_y = height // 2
    draw.ellipse(
        (centre_x - radius, centre_y - radius, centre_x + radius, centre_y + radius),
        fill=(220, 30, 30),
    )
    return image


def draw_jpeg(width: int = 256, height: int = 256) -> bytes:
    """A red circle on a blue background as JPEG."""
    buffer = io.BytesIO()
    _scene(width, height).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def draw_png(width: int = 256, height: int = 256) -> bytes:
    """The same scene as PNG."""
    buffer = io.BytesIO()
    _scene(width, height).save(buffer, format="PNG")
    return buffer.getvalue()


def draw_gif(width: int = 128, height: int = 128, frames: int = 4) -> bytes:
    """A multi-frame GIF in which the circle moves from frame to frame."""
    if frames < 2:
        raise ValueError("an animated GIF needs at least two frames")
    step = max(1, width // (4 * frames))
    images = [
        _scene(width, height, shift=index * step - (frames * step) // 2).convert(
            "P", palette=Image.Palette.ADAPTIVE
        )
        for index in range(frames)
    ]
    buffer = io.BytesIO()
    images[0].save(
        buffer, format="GIF", save_all=True, append_images=images[1:], duration=120, loop=0
    )
    return buffer.getvalue()
