"""What later rounds import from the persona card (R-PERS-004, R-TRN-013).

::

    from twin.profile.persona.api import load_persona, render_compact, render_full

    full = render_full(services)                      # live card, all sections, <= 1,500 tokens
    compact = render_compact(services, "pre_holdout") # style only, <= 400 tokens: training prompts
    eval_full = render_full(services, "pre_holdout", with_live_corrections=True)
    text, version, scope = compact.text, compact.number, compact.scope

``render_full`` is for the DeepSeek backend and the hybrid planner, ``render_compact`` for the
style model's backend and for the training set.  The compact rendering has no facts and no
corrections; the full pre-holdout rendering gets the live card's ``[不要这样]`` lines added when
``with_live_corrections`` is set - that is how the evaluation sandbox feeds the DeepSeek backend
(R-TRN-013: those lines only describe a manner of speaking).  Each function takes an explicit
card version (``version``) so that a model trained with version N can be served with version N.
"""

from __future__ import annotations

from twin.profile.persona.compose import correction_lines
from twin.profile.persona.refresh import persona_store, read_corrections, write_corrections
from twin.profile.persona.render import RenderedPersona, render_card
from twin.profile.persona.store import PersonaCardView
from twin.services import Services

__all__ = [
    "PersonaCardView",
    "RenderedPersona",
    "load_persona",
    "read_corrections",
    "render_compact",
    "render_full",
    "write_corrections",
]


def load_persona(
    services: Services, scope: str = "live", version: str | None = None
) -> PersonaCardView | None:
    """The card in force for ``scope``, or the version named by ``version`` (``vN``, ``~N``, id)."""
    store = persona_store(services)
    if version is None:
        return store.active(scope)
    return store.resolve(version, scope)


def render_full(
    services: Services,
    scope: str = "live",
    *,
    version: str | None = None,
    with_live_corrections: bool = False,
) -> RenderedPersona | None:
    """The full rendering (all sections); ``None`` when the scope has no card yet."""
    card = load_persona(services, scope, version)
    if card is None:
        return None
    extra: list[str] = []
    if with_live_corrections and scope != "live":
        live = persona_store(services).active("live")
        if live is not None:
            extra = correction_lines(live.sections)
    return render_card(
        card.text,
        kind="full",
        scope=card.scope,
        version_id=card.id,
        number=card.number,
        budget=services.settings.persona.full_max_tokens,
        extra_dont=extra,
    )


def render_compact(
    services: Services, scope: str = "live", *, version: str | None = None
) -> RenderedPersona | None:
    """The compact, style-only rendering; ``None`` when the scope has no card yet."""
    card = load_persona(services, scope, version)
    if card is None:
        return None
    return render_card(
        card.text,
        kind="compact",
        scope=card.scope,
        version_id=card.id,
        number=card.number,
        budget=services.settings.persona.compact_max_tokens,
    )
