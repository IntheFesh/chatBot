"""Terminal interaction for the channel commands, behind small interfaces.

Everything that asks the user a question or shouts at them goes through :class:`Prompter` or
:class:`AlertBanner`, so the login and binding flows can be driven by a scripted input stream
in tests and replaced by a graphical front end later.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import typer
from rich.console import Console
from rich.panel import Panel


class Prompter(Protocol):
    """Questions and messages for the person at the keyboard."""

    def confirm(self, message: str, *, default: bool = False) -> bool: ...

    def ask(self, message: str) -> str: ...

    def say(self, message: str) -> None: ...


class TyperPrompter:
    """Reads from and writes to the terminal (``typer``'s prompts, so tests can pipe input)."""

    def confirm(self, message: str, *, default: bool = False) -> bool:
        return typer.confirm(message, default=default)

    def ask(self, message: str) -> str:
        return str(typer.prompt(message, default="", show_default=False))

    def say(self, message: str) -> None:
        typer.echo(message)


class AlertBanner(Protocol):
    """A conspicuous message for the console of a running application (R-OPS-004)."""

    def show(self, title: str, lines: Sequence[str]) -> None: ...


class StderrBanner:
    """A red panel on standard error."""

    def show(self, title: str, lines: Sequence[str]) -> None:
        console = Console(stderr=True, highlight=False)
        console.print(Panel("\n".join(lines), title=title, border_style="bold red"))
