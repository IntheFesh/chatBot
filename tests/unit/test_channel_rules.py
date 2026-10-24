"""Rules about the channel code itself, enforced by scanning it (R-SAFE-004, R-PRIV-006)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from twin.channel.base import Channel
from twin.channel.ilink.channel import IlinkChannel

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
ILINK = SRC / "channel" / "ilink"

# every endpoint the program may call (docs/ILINK_PROTOCOL.md section 5); there is no endpoint
# for changing the bot's name or avatar, and the code must stay that way (R-PRIV-006)
ALLOWED_ENDPOINTS = {
    "get_bot_qrcode",
    "get_qrcode_status",
    "getupdates",
    "sendmessage",
    "getconfig",
    "sendtyping",
    "getuploadurl",
    "msg/notifystart",
    "msg/notifystop",
}


def endpoint_literals(path: Path) -> set[str]:
    """Endpoint names passed to ``IlinkHttp.post`` / ``IlinkHttp.get``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"post", "get"}
            and ast.unparse(node.func.value).endswith("http")  # the IlinkHttp wrapper
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and not node.args[0].value.startswith(("http", "/"))
        ):
            found.add(node.args[0].value)
    return found


def test_only_the_documented_endpoints_are_ever_called() -> None:
    used: set[str] = set()
    for path in sorted(ILINK.glob("*.py")):
        used |= endpoint_literals(path)
    assert used and used <= ALLOWED_ENDPOINTS, sorted(used - ALLOWED_ENDPOINTS)


def test_messages_are_sent_only_from_inside_the_channel_package() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(SRC.rglob("*.py"))
        if SRC / "channel" not in path.parents and "sendmessage" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_the_channel_hosts_are_the_only_places_urls_with_the_wechat_domains_appear() -> None:
    allowed = {ILINK / "wire.py"}
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(SRC.rglob("*.py"))
        if path not in allowed and "weixin.qq.com" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders


def test_every_send_method_takes_a_recipient_that_the_guard_checks() -> None:
    for owner in (Channel, IlinkChannel):
        for name in ("send_text", "send_image", "send_typing"):
            parameters = inspect.signature(getattr(owner, name)).parameters
            assert "recipient" in parameters, f"{owner.__name__}.{name}"
            assert parameters["recipient"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_normal_send_path_has_no_bypass_default() -> None:
    for name in ("send_text", "send_image"):
        parameter = inspect.signature(getattr(IlinkChannel, name)).parameters["bypass"]
        assert parameter.default is None  # only the probe passes one (R-CH-009)
