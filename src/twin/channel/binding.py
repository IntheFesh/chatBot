"""Recipient binding (R-CH-007, R-SAFE-004, R-PRIV-006).

The bot talks to exactly one person.  This module holds the pieces that make that
checkable:

* :func:`mask_user_id`: what the terminal shows instead of a full id;
* :class:`RecipientGuard`: every send resolves its recipient through it, so naming anyone
  other than the bound user raises :class:`~twin.channel.base.RecipientNotAllowed`;
* :func:`confirm_binding` and :func:`confirm_unbind`: the two questions put to the user.
"""

from __future__ import annotations

from collections.abc import Callable

from twin.channel.base import RecipientNotAllowed
from twin.channel.console import Prompter

UNBIND_WORD = "unbind"
BIND_ANYWAY_WORD = "bind"


def mask_user_id(user_id: str) -> str:
    """``o9cq1234abcd@im.wechat`` -> ``o9cq****@im.wechat`` (id kept to four characters)."""
    local, at, domain = user_id.partition("@")
    visible = local[:4] if len(local) > 8 else local[:2]
    return f"{visible}****{at}{domain}"


class RecipientGuard:
    """Resolves the recipient of an outgoing message to the bound user, or refuses."""

    def __init__(self, bound_user: Callable[[], str | None]) -> None:
        self._bound_user = bound_user

    def resolve(self, recipient: str | None = None) -> str:
        bound = self._bound_user()
        if bound is None:
            raise RecipientNotAllowed("no user is bound to this channel yet (R-CH-007)")
        if recipient is not None and recipient != bound:
            raise RecipientNotAllowed("the channel only sends to the bound user (R-CH-007)")
        return bound


def confirm_binding(
    prompter: Prompter, candidate_id: str, *, matches_expected: bool | None
) -> bool:
    """Ask whether ``candidate_id`` is the user.  Returns ``True`` only on explicit consent.

    ``matches_expected`` compares the sender with the id the login returned for the person
    who scanned the code: when they differ the question is asked again in stronger words and
    the user must type a word, not just press a key (decision D-012).
    """
    shown = mask_user_id(candidate_id)
    prompter.say(f"The first message came from the account {shown}.")
    if matches_expected is False:
        prompter.say(
            "WARNING: this is NOT the account that scanned the login code. Binding it means "
            "the bot will talk to someone else than the person who logged in."
        )
        answer = prompter.ask(
            f"Type '{BIND_ANYWAY_WORD}' to bind it anyway, anything else to refuse"
        )
        return answer.strip().lower() == BIND_ANYWAY_WORD
    if matches_expected is None:
        prompter.say("(the login did not report which account scanned the code; cannot compare)")
    return prompter.confirm(
        f"Is {shown} you? The bot will talk only to this account", default=False
    )


def confirm_unbind(prompter: Prompter, bound_id: str) -> bool:
    """Two steps: a yes/no question, then typing the word ``unbind``."""
    shown = mask_user_id(bound_id)
    if not prompter.confirm(
        f"Unbind {shown}? The bot stops talking to them until you bind someone again",
        default=False,
    ):
        return False
    answer = prompter.ask(f"Type '{UNBIND_WORD}' to confirm")
    return answer.strip().lower() == UNBIND_WORD
