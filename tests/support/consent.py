"""Her consent, written explicitly by the test fixtures (R-SCOPE-003, R18-F10, D-825).

``consent.confirmed_at`` has no default any more: without it every command except ``twin setup``
and ``twin doctor`` (and the few that stop or delete, D-825) refuses with exit code 3.  The suite
records the date the way a user's configuration would - through the settings sources - so the
autouse fixture (``tests/conftest.py``), the life environment and the development tools that
build their own environment set :data:`CONSENT_ENV`.  A test about the missing consent removes it.
"""

from __future__ import annotations

CONSENT_ENV = "TWIN_CONSENT__CONFIRMED_AT"
CONSENT_DATE = "2026-10-08"
