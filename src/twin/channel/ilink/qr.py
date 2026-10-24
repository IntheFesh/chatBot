"""Local QR code rendering for the login (R-CH-003, R-OPS-004).

The login content is turned into a picture on this machine only: a PNG in the controlled
temporary directory, opened with the default viewer, and a character version for the terminal.
No online QR service is ever used and the code is never sent anywhere.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_M

PNG_PREFIX = "ilink-login-"
_FULL, _UPPER, _LOWER, _EMPTY = "█", "▀", "▄", " "


def qr_matrix(content: str, *, border: int = 2) -> list[list[bool]]:
    """The QR modules (``True`` = dark) including a quiet zone of ``border`` modules."""
    code = qrcode.QRCode(error_correction=ERROR_CORRECT_M, border=border)
    code.add_data(content)
    code.make(fit=True)
    return [[bool(cell) for cell in row] for row in code.get_matrix()]


def render_terminal(content: str, *, border: int = 2) -> str:
    """Two module rows per text line using half-block characters (dark on a light terminal)."""
    matrix = qr_matrix(content, border=border)
    if len(matrix) % 2:
        matrix.append([False] * len(matrix[0]))
    lines: list[str] = []
    for top, bottom in zip(matrix[::2], matrix[1::2], strict=True):
        chars = []
        for upper, lower in zip(top, bottom, strict=True):
            if upper and lower:
                chars.append(_FULL)
            elif upper:
                chars.append(_UPPER)
            elif lower:
                chars.append(_LOWER)
            else:
                chars.append(_EMPTY)
        lines.append("".join(chars))
    return "\n".join(lines)


def save_png(content: str, directory: Path) -> Path:
    """Write the QR code as a PNG into ``directory`` (created, owner-only on POSIX)."""
    directory.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        directory.chmod(0o700)
    path = directory / f"{PNG_PREFIX}{secrets.token_hex(6)}.png"
    code = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=10, border=4)
    code.add_data(content)
    code.make(fit=True)
    image = code.make_image(fill_color="black", back_color="white")
    image.save(str(path))
    return path


def remove_old_pngs(directory: Path) -> None:
    """Delete leftover login pictures; a viewer that still holds one open is left alone."""
    if not directory.is_dir():
        return
    for old in directory.glob(f"{PNG_PREFIX}*.png"):
        with contextlib.suppress(OSError):
            old.unlink()


def open_in_viewer(path: Path, *, opener: Callable[[Path], bool] | None = None) -> bool:
    """Open ``path`` with the system's default application; ``False`` if that was not possible."""
    if opener is not None:
        return opener(path)
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606  # pragma: win32-only
            return True  # pragma: win32-only
        return webbrowser.open(path.as_uri())
    except OSError:
        return False
