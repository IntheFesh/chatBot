"""In-memory credential backends for tests (never touch the real credential store)."""

from __future__ import annotations

from keyring.errors import PasswordDeleteError, PasswordSetError


class MemoryCredentials:
    """Dictionary-backed credential backend (structurally a ``CredentialBackend``)."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str, /) -> str | None:
        return self.items.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str, /) -> None:
        self.items[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str, /) -> None:
        if (service_name, username) not in self.items:
            raise PasswordDeleteError("not found")
        del self.items[(service_name, username)]


class BrokenCredentials:
    """A backend whose every call fails, like a locked or missing credential store."""

    def get_password(self, service_name: str, username: str, /) -> str | None:
        raise OSError("credential store unavailable")

    def set_password(self, service_name: str, username: str, password: str, /) -> None:
        raise OSError("credential store unavailable")

    def delete_password(self, service_name: str, username: str, /) -> None:
        raise OSError("credential store unavailable")


class CredentialManagerDouble(MemoryCredentials):
    """Stands for the Windows Credential Manager behind ``keyring.get_keyring()`` (R18-A2).

    ``twin secrets migrate`` moves the entries of the encrypted file store into the system
    credential store; the tests install this double in place of ``keyring.get_keyring`` so that
    the real credential manager of a Windows CI machine is never touched.  ``priority`` makes it a
    system candidate for :func:`twin.config.secrets.select_backend`; ``fail_on_write`` makes the
    n-th ``set_password`` of a test fail as a locked or full credential store would.
    """

    priority = 5.0

    def __init__(self, *, fail_on_write: int | None = None) -> None:
        super().__init__()
        self.fail_on_write = fail_on_write
        self.writes = 0

    def set_password(self, service_name: str, username: str, password: str, /) -> None:
        self.writes += 1
        if self.fail_on_write is not None and self.writes >= self.fail_on_write:
            raise PasswordSetError("the credential store refused the write")
        super().set_password(service_name, username, password)
