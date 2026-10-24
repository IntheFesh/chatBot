"""Login to the AutoDL instance (R-TRN-010, CLAUDE.md section 7).

The connection is made with ``asyncssh`` (pure Python, so Windows needs no OpenSSH).  Login is by
password - read from the credential store entry ``autodl_password`` and never written anywhere -
or by a key file.  The host, port and user come from the ``autodl.*`` settings.

Host keys: AutoDL instances get a new host key every time one is created, so there is no key to
check against on the first connection.  The first connection to a ``host:port`` therefore shows
the fingerprint and asks whether to trust it (``twin train remote connect``); the key is then
saved in ``training/known_hosts`` below the data directory, and every later connection to that
address must present the same key.  A different key is refused with a message that says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import asyncssh

from twin.config.secrets import SecretStore
from twin.config.settings import AutoDlConfig

KEEPALIVE_SECONDS = 15
KEEPALIVE_MAX_MISSED = 4
CONNECT_TIMEOUT_SECONDS = 30.0


class RemoteError(RuntimeError):
    """The instance cannot be reached or a remote step failed."""


class HostKeyError(RemoteError):
    """The host key is unknown and was not trusted, or it changed."""


class HashMismatchError(RemoteError):
    """A transferred file does not have the expected sha256 (damaged in transit, or wrong file)."""


class LoginRefusedError(RemoteError):
    """The instance refused the password or the key; trying again does not help."""


@dataclass(frozen=True)
class RemoteTarget:
    """Where and how to log in; the password never shows up in ``repr``."""

    host: str
    port: int
    user: str
    auth: Literal["password", "key"]
    known_hosts: Path
    key_path: Path | None = None
    password: str | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


def target_from_settings(
    config: AutoDlConfig, secrets: SecretStore, known_hosts: Path
) -> RemoteTarget:
    """Build the login from ``autodl.*`` and the credential store."""
    if not config.host or not config.port:
        raise RemoteError(
            "autodl.host and autodl.port are not set; copy them from the SSH login command in "
            "the AutoDL console into config/config.yaml"
        )
    if config.auth == "key":
        if not config.key_path:
            raise RemoteError("autodl.auth is 'key' but autodl.key_path is not set")
        key = Path(config.key_path).expanduser()
        if not key.is_file():
            raise RemoteError(f"the key file {key} does not exist")
        return RemoteTarget(config.host, config.port, config.user, "key", known_hosts, key_path=key)
    password = secrets.get("autodl_password")
    if not password:
        raise RemoteError("the AutoDL password is not set; run: twin secrets set autodl_password")
    return RemoteTarget(
        config.host, config.port, config.user, "password", known_hosts, password=password
    )


def known_host_entry(host: str, port: int) -> str:
    return f"[{host}]:{port}"


def has_known_host(path: Path, host: str, port: int) -> bool:
    if not path.is_file():
        return False
    prefix = known_host_entry(host, port) + " "
    return any(line.startswith(prefix) for line in path.read_text(encoding="utf-8").splitlines())


def remember_host(path: Path, host: str, port: int, key: asyncssh.SSHKey) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = key.export_public_key("openssh").decode("ascii").strip().split(" ")[:2]
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{known_host_entry(host, port)} {line[0]} {line[1]}\n")


TrustPrompt = Callable[[str, str], bool]
"""``(label, fingerprint) -> trusted``: asked once for an address that has no saved host key."""


async def open_connection(
    target: RemoteTarget,
    *,
    trust: TrustPrompt | None = None,
    limit_s: float = CONNECT_TIMEOUT_SECONDS,
) -> asyncssh.SSHClientConnection:
    """Log in to ``target``; see the module docstring for the host key policy."""
    known = has_known_host(target.known_hosts, target.host, target.port)
    options: dict[str, object] = {
        "port": target.port,
        "username": target.user,
        "config": [],
        "agent_path": None,
        "keepalive_interval": KEEPALIVE_SECONDS,
        "keepalive_count_max": KEEPALIVE_MAX_MISSED,
        "connect_timeout": limit_s,
        "login_timeout": limit_s,
        "known_hosts": str(target.known_hosts) if known else None,
    }
    if target.auth == "key":
        options.update(client_keys=[str(target.key_path)], password=None)
    else:
        options.update(
            client_keys=None,
            password=target.password,
            preferred_auth=["password", "keyboard-interactive"],
        )
    try:
        connection = await asyncssh.connect(target.host, **options)
    except asyncssh.HostKeyNotVerifiable as exc:
        raise HostKeyError(
            f"the host key of {target.label} differs from the one saved in {target.known_hosts}; "
            "if the instance was recreated, delete that line and connect again"
        ) from exc
    except asyncssh.PermissionDenied as exc:
        raise LoginRefusedError(
            f"{target.label} refused the login ({exc.reason}); check the password or key"
        ) from exc
    except (OSError, asyncssh.Error, TimeoutError) as exc:
        raise RemoteError(f"cannot connect to {target.label}: {exc}") from exc
    if not known:
        key = connection.get_server_host_key()
        if key is None:
            connection.close()
            raise HostKeyError(f"{target.label} did not present a host key")
        fingerprint = key.get_fingerprint("sha256")
        if trust is None or not trust(target.label, fingerprint):
            connection.close()
            await connection.wait_closed()
            raise HostKeyError(
                f"the host key {fingerprint} of {target.label} is not trusted yet; "
                "run `twin train remote connect` and confirm it"
            )
        remember_host(target.known_hosts, target.host, target.port, key)
    return connection
