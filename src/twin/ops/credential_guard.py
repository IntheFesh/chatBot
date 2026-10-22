"""The commands that must not start on a weak credential store (D-744, R-CFG-002).

``twin run``, ``twin supervise`` and ``twin service install`` call :func:`require_credential_store`
first.  On Windows the encrypted-file store (``TWIN_KEYRING_BACKEND=file``) is a development and
test switch; without ``TWIN_ALLOW_FILE_KEYRING=1`` the command stops with a message instead of
starting an application that keeps its keys next to its secrets.
"""

from __future__ import annotations

import sys

from twin.config.secrets import SecretStore, SecretStoreError, file_keyring_refusal
from twin.ops.process_model import CliError, ExitCode


def current_platform() -> str:
    """``sys.platform``; a seam, so that the Windows rule can be tested on any machine."""
    return sys.platform


def require_credential_store(secrets: SecretStore | None) -> None:
    """Raise :class:`~twin.ops.process_model.CliError` if this credential store is not allowed."""
    try:
        store = secrets or SecretStore.default()
    except SecretStoreError as exc:
        raise CliError(str(exc), ExitCode.SECRETS) from exc
    refusal = file_keyring_refusal(store.info, platform=current_platform())
    if refusal is not None:
        raise CliError(refusal, ExitCode.SECRETS)
