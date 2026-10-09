"""In-memory credential backends for tests (never touch the real credential store)."""

from __future__ import annotations

from keyring.errors import PasswordDeleteError


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
