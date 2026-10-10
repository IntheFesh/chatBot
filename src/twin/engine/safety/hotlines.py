"""Which help line to give, by the time zone the bot lives in (R-SAFE-001).

``safety.timezone_country`` maps the bot's IANA time zone to a country code, ``safety.hotlines``
maps a country code to the text of its help line (US: 988, CN: 12356).  A zone the table does not
know - or a country without a line - gets every line, so nobody is left without a number.
"""

from __future__ import annotations

from twin.config.settings import SafetyConfig


def country_of(safety: SafetyConfig, timezone_name: str) -> str | None:
    """The country code of a time zone, or ``None`` if the table does not know the zone."""
    return safety.timezone_country.get(timezone_name)


def hotlines_for(safety: SafetyConfig, timezone_name: str) -> tuple[str, ...]:
    """The help lines to give for ``timezone_name`` (all of them when it is not mapped)."""
    country = country_of(safety, timezone_name)
    if country is not None and country in safety.hotlines:
        return (safety.hotlines[country],)
    return tuple(safety.hotlines.values())
