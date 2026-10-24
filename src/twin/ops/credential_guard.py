"""The early check of the credential store in ``twin run``, ``supervise`` and ``service install``.

Every command refuses the encrypted-file store on Windows without ``TWIN_ALLOW_FILE_KEYRING=1``
where it opens the store (:meth:`twin.services.CliContext.secret_store`,
:func:`twin.config.secrets.refuse_weak_store`; D-744, D-827).  The three commands that start or
register the application ask before anything else happens - ``twin supervise`` and ``twin service
install`` do not open the store themselves - through :func:`require_credential_store`, the same
check.
"""

from __future__ import annotations

from twin.config.secrets import SecretStore, SecretStoreError, refuse_weak_store
from twin.ops.process_model import CliError, ExitCode


def require_credential_store(secrets: SecretStore | None) -> None:
    """Raise :class:`~twin.ops.process_model.CliError` if this credential store is not allowed."""
    try:
        refuse_weak_store(secrets or SecretStore.default(allow_weak_file_store=True))
    except SecretStoreError as exc:
        raise CliError(str(exc), ExitCode.SECRETS) from exc
