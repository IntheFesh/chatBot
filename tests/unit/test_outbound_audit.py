"""R-SAFE-004: the bot contacts nobody but the user - the audit of every way out (round 16).

What leaves the machine towards a person, and what limits who that can be:

==================  =========================================  =================================
way out             who can receive                            where it is held to that
==================  =========================================  =================================
the chat channels   the one bound user (WeChat, terminal,      ``RecipientGuard.resolve`` in
                    in-memory evaluation channel)              every ``send_*``
alert e-mail        ``ops.smtp.to``, the user's own address    ``AlertDelivery(recipient=...)``
monthly cost mail   ``ops.smtp.to``                            ``cost_report_task``
emergency e-mail    ``safety.emergency_contact.email``, only   ``SmtpEmergencyNotifier.notify``
                    when ``enabled``
setup test mail     the address the user just typed            ``setup_cli``
==================  =========================================  =================================

The tests below scan the code for the list being complete (a new way out fails here until it is
reviewed and added), run all of them at once on a real service container and look at every mail
that results, and put a sentence of chat into every free field of the three mail templates to see
that none of it comes out.  The guards themselves are tested where they live; the tests cited in
``GUARD_TESTS`` are looked up so that renaming one cannot silently drop the guard.
"""

from __future__ import annotations

import ast
import dataclasses
import typing
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from tests.support.clock import ManualClock
from tests.support.nodes import missing_nodes
from tests.support.ops import RecordingMailer, RecordingNotifier
from twin.config.settings import Settings
from twin.engine.safety.notifier import EmergencyEmail, EmergencyNotice, render_notice
from twin.llm.ledger import SpendSummary
from twin.ops.alert_delivery import AlertDelivery
from twin.ops.alert_text import render_alert
from twin.ops.alerts import ALIASES, RECOVERIES, SPECS, UNKNOWN_LABEL, AlertView
from twin.ops.cost import CostReport
from twin.ops.mail import OutgoingMail, build_message
from twin.ops.scheduler import cost_report_task
from twin.ops.wiring import build_emergency_notifier
from twin.services import Services

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
CHICAGO = ZoneInfo("America/Chicago")
ME = "me@example.org"
FRIEND = "friend@example.org"
CHAT = "今晚想吃火锅还是烧烤，晚点再说"
CHAT_ASCII = "see you at the hotpot place tonight"
CANARIES = (CHAT, CHAT_ASCII, "火锅")

# the guard tests of the channels and the mail, cited so that they cannot disappear unnoticed
GUARD_TESTS = [
    "tests/unit/test_ilink_outbound.py::test_only_the_bound_user_can_be_named_as_the_recipient",
    "tests/unit/test_ilink_outbound.py::test_nothing_can_be_sent_while_nobody_is_bound",
    "tests/unit/test_channel_local.py::test_only_the_local_user_can_be_named_as_recipient",
    "tests/unit/test_eval_channel.py::test_it_only_talks_to_the_one_bound_user",
    "tests/unit/test_channel_policy_binding.py::test_the_guard_resolves_only_the_bound_user",
    "tests/unit/test_channel_rules.py::test_messages_are_sent_only_from_inside_the_channel_package",
    "tests/unit/test_channel_rules.py::test_every_send_method_takes_a_recipient_that_the_guard_checks",
    "tests/unit/test_channel_rules.py::test_only_the_documented_endpoints_are_ever_called",
    "tests/unit/test_ops_emergency.py::test_only_the_configured_address_is_ever_written_to",
    "tests/unit/test_ops_emergency.py::test_nothing_is_sent_unless_switched_on_with_an_address",
    "tests/unit/test_ops_emergency.py::test_the_notice_has_no_field_that_could_carry_what_was_said",
    "tests/unit/test_engine_safety.py::test_the_notice_has_no_field_for_chat_content",
    "tests/unit/test_ops_alerts.py::test_the_notice_is_made_of_wording_and_numbers_only",
    "tests/unit/test_ops_mail.py::test_an_alert_arrives_by_mail_and_holds_no_chat_content",
    "tests/unit/test_ops_scheduler.py::test_the_cost_report_goes_out_on_the_first_at_nine",
]


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def sources() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(SRC.rglob("*.py")):
        yield path, parse(path)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_the_guard_tests_this_audit_relies_on_exist() -> None:
    assert missing_nodes(GUARD_TESTS) == []


# =========================================================================== the chat channels


def classes_deriving_from(base: str) -> dict[str, str]:
    """Class name -> file for the classes of ``src`` that list ``base`` among their bases."""
    found: dict[str, str] = {}
    for path, tree in sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and base in {ast.unparse(b) for b in node.bases}:
                found[node.name] = rel(path)
    return found


def test_there_are_exactly_three_channel_implementations_and_each_has_a_guard() -> None:
    """A fourth way to talk to somebody must be added here, with its guard, on purpose."""
    assert classes_deriving_from("Channel") == {
        "IlinkChannel": "src/twin/channel/ilink/channel.py",
        "LocalConsoleChannel": "src/twin/channel/local.py",
        "InMemoryChannel": "src/twin/eval/channel.py",
    }


def methods_of(tree: ast.Module, class_name: str) -> dict[str, ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {m.name: m for m in node.body if isinstance(m, ast.AsyncFunctionDef)}
    raise AssertionError(f"{class_name} not found")


def calls_in(node: ast.AST) -> list[str]:
    return [ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


@pytest.mark.parametrize(
    ("file", "class_name", "resolver"),
    [
        ("channel/ilink/outbound.py", "IlinkSender", "self._recipients.resolve"),
        ("channel/local.py", "LocalConsoleChannel", "self.guard.resolve"),
        ("eval/channel.py", "InMemoryChannel", "self.guard.resolve"),
    ],
)
def test_every_send_method_resolves_its_recipient_through_the_guard(
    file: str, class_name: str, resolver: str
) -> None:
    methods = methods_of(parse(SRC / file), class_name)
    for name in ("send_text", "send_image", "send_typing"):
        assert resolver in calls_in(methods[name]), f"{class_name}.{name} does not ask the guard"


def test_the_wechat_channel_hands_every_send_to_the_guarded_sender() -> None:
    methods = methods_of(parse(SRC / "channel" / "ilink" / "channel.py"), "IlinkChannel")
    for name in ("send_text", "send_image"):
        assert f"self._sender.{name}" in calls_in(methods[name]), name
    assert "self._sender.send_typing" in calls_in(methods["send_typing"])


def test_nobody_outside_the_channel_package_names_a_recipient() -> None:
    """The engine, the scheduler and the evaluation never pass ``recipient=``: the default is
    the bound user, and a send without one goes there (R-CH-007)."""
    offenders = []
    for path, tree in sources():
        if "channel" in path.relative_to(SRC).parts[:1] or path == SRC / "eval" / "channel.py":
            continue
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"send_text", "send_image", "send_typing"}
                and any(keyword.arg == "recipient" for keyword in call.keywords)
            ):
                offenders.append(f"{rel(path)}:{call.lineno}")
    assert offenders == []


# ================================================================================ e-mail


def outgoing_mail_sites() -> dict[str, list[str]]:
    """File -> the expressions given as the recipient of each ``OutgoingMail(...)`` there."""
    found: dict[str, list[str]] = {}
    for path, tree in sources():
        if path == SRC / "ops" / "mail.py":
            continue
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "OutgoingMail"
            ):
                found.setdefault(rel(path), []).append(ast.unparse(call.args[0]))
    return found


def assigned_from(path: str, name: str) -> list[str]:
    """What the variable ``name`` is set to in a file (the right-hand sides, as source)."""
    values = []
    for node in ast.walk(parse(ROOT / path)):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            values.append(ast.unparse(node.value) if node.value is not None else "")
    return values


def test_mail_is_written_in_four_places_each_with_a_recipient_from_the_settings() -> None:
    sites = outgoing_mail_sites()
    assert sites == {
        "src/twin/ops/alert_delivery.py": ["to"],
        "src/twin/ops/emergency.py": ["configured"],
        "src/twin/ops/scheduler.py": ["recipient"],
        "src/twin/ops/setup_cli.py": ["recipient"],
    }
    # alerts: the address comes from the callable the component was given ...
    delivery = ast.unparse(parse(SRC / "ops" / "alert_delivery.py"))
    assert "mailer, to = (self._mailer, self._recipient())" in delivery
    # ... which every construction of the component sets to ops.smtp.to
    for file in ("src/twin/ops/wiring.py", "src/twin/ops/service_cli.py"):
        constructions = [
            keyword
            for call in ast.walk(parse(ROOT / file))
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "AlertDelivery"
            for keyword in call.keywords
            if keyword.arg == "recipient"
        ]
        assert len(constructions) == 1, file
        assert ast.unparse(constructions[0].value).endswith("settings.ops.smtp.to"), file
    # the monthly report: ops.smtp.to
    assert assigned_from("src/twin/ops/scheduler.py", "recipient") == [
        "services.settings.ops.smtp.to"
    ]
    # the emergency notice: the configured contact, and only when it is switched on
    assert assigned_from("src/twin/ops/emergency.py", "configured") == [
        "(self._safety.emergency_contact.email or '').strip()"
    ]
    # the setup wizard: the address the user typed for his own alerts
    assert assigned_from("src/twin/ops/setup_cli.py", "recipient") == [
        "_ask('收件地址', current.to or user)"
    ]


def test_only_the_mail_module_speaks_smtp_and_no_other_messenger_is_used() -> None:
    smtp_users, banned = [], []
    messengers = {
        "aiosmtplib",
        "yagmail",
        "imaplib",
        "poplib",
        "ftplib",
        "telnetlib",
        "slack_sdk",
        "telegram",
        "twilio",
        "discord",
        "itchat",
        "wxpy",
        "pywechat",
    }
    for path, tree in sources():
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top == "smtplib":
                    smtp_users.append(rel(path))
                if top in messengers:
                    banned.append(f"{rel(path)}: {name}")
    assert sorted(set(smtp_users)) == ["src/twin/ops/mail.py"]
    assert banned == []


def test_the_settings_have_exactly_two_places_for_an_address_to_write_to() -> None:
    """If a setting for another recipient (a second user, a Cc, a webhook) is ever added, this
    fails and the new way out gets reviewed."""
    found: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, info in model.model_fields.items():
            annotation = info.annotation
            args = typing.get_args(annotation) or (annotation,)
            nested = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
            if nested:
                walk(nested[0], f"{prefix}{name}.")
            elif str in args and name in {
                "to",
                "cc",
                "bcc",
                "email",
                "recipient",
                "recipients",
                "webhook",
                "url",
            }:
                found.append(f"{prefix}{name}")

    walk(Settings, "")
    assert sorted(found) == ["ops.smtp.to", "safety.emergency_contact.email"]


def test_a_mail_has_one_recipient_and_no_cc_bcc_or_reply_to() -> None:
    message = build_message("bot@example.org", OutgoingMail(ME, "s", "t", "<p>t</p>"))
    assert message.get_all("To") == [ME]
    for header in ("Cc", "Bcc", "Reply-To", "Resent-To", "Resent-Cc", "Resent-Bcc"):
        assert message.get_all(header) is None, header


# ----------------------------------------------------------------- all of it at once


@pytest.fixture
def world(services: Services, clock: ManualClock) -> Services:
    clock.set_time(datetime(2026, 10, 1, 14, 5, tzinfo=UTC))  # 09:05 on the 1st, Chicago
    services.settings.ops.smtp.to = ME
    services.settings.safety.emergency_contact.enabled = True
    services.settings.safety.emergency_contact.email = FRIEND
    return services


def hostile_detail() -> dict[str, Any]:
    return {
        "text": CHAT,
        "reply": CHAT,
        "message": CHAT_ASCII,
        "reason": CHAT,  # a sentence in a field that normally holds a code
        "error": CHAT_ASCII,
        "note": CHAT,
        "lines": [CHAT, CHAT_ASCII],
        "nested": {"text": CHAT},
        "count": 3,
    }


def everything(mail: OutgoingMail) -> str:
    return "\n".join([mail.to, mail.subject, mail.text, mail.html or ""])


async def deliver_everything(world: Services, mailer: RecordingMailer) -> RecordingNotifier:
    """Every category (and every old name of one) with chat in all its fields, then delivered."""
    notifier = RecordingNotifier()
    delivery = AlertDelivery(
        world.alerts,
        world.clock,
        notifier=notifier,
        mailer=mailer,
        recipient=lambda: world.settings.ops.smtp.to,
        zone=lambda: CHICAGO,
    )
    names = [*SPECS, *ALIASES, *RECOVERIES, CHAT, CHAT_ASCII]
    for name in names:
        world.alerts.raise_alert(
            name, f"{CHAT} / {CHAT_ASCII}", severity="critical", detail=hostile_detail()
        )
    for name in {*SPECS, *ALIASES.values()}:
        world.alerts.recover(str(name), f"{CHAT} / {CHAT_ASCII}")
    while await delivery.deliver_once():
        pass
    return notifier


async def test_every_mail_the_program_can_send_goes_to_the_user_or_the_contact_and_holds_no_chat(
    world: Services, clock: ManualClock
) -> None:
    mailer = RecordingMailer()
    notifier = await deliver_everything(world, mailer)
    alert_mails = list(mailer.sent)
    assert len(alert_mails) > 25  # the announced categories, their recoveries, the critical ones

    emergency = build_emergency_notifier(world, mailer)
    assert await emergency.notify(EmergencyNotice(clock.now_utc(), FRIEND)) is True
    emergency_mail = mailer.sent[-1]

    clock.set_time(datetime(2026, 10, 1, 14, 5, tzinfo=UTC))
    report_task = cost_report_task(world, mailer)
    await report_task.run(datetime(2026, 10, 1, 14, 0, tzinfo=UTC))
    report_mail = mailer.sent[-1]
    assert report_mail is not emergency_mail and "费用报告" in report_mail.text

    assert {m.to for m in alert_mails} == {ME}
    assert emergency_mail.to == FRIEND and report_mail.to == ME
    assert {m.to for m in mailer.sent} == {ME, FRIEND}
    for mail in mailer.sent:
        text = everything(mail)
        for canary in CANARIES:
            assert canary not in text, (mail.subject, canary)
    for title, body in notifier.shown:
        for canary in CANARIES:
            assert canary not in title + body


async def test_without_the_switch_the_contact_is_never_written_to(
    world: Services, clock: ManualClock
) -> None:
    world.settings.safety.emergency_contact.enabled = False
    mailer = RecordingMailer()
    emergency = build_emergency_notifier(world, mailer)
    assert await emergency.notify(EmergencyNotice(clock.now_utc(), FRIEND)) is False
    world.settings.safety.emergency_contact.enabled = True
    world.settings.safety.emergency_contact.email = None
    assert await emergency.notify(EmergencyNotice(clock.now_utc(), FRIEND)) is False
    assert mailer.sent == [] and mailer.attempts == 0


async def test_nobody_else_can_be_named_as_the_contact(world: Services, clock: ManualClock) -> None:
    from twin.ops.mail import MailError

    mailer = RecordingMailer()
    emergency = build_emergency_notifier(world, mailer)
    with pytest.raises(MailError) as caught:
        await emergency.notify(EmergencyNotice(clock.now_utc(), "stranger@example.net"))
    assert caught.value.code == "recipient" and mailer.sent == []


# ============================================================ the fields of the three templates


def view_with(**changes: Any) -> AlertView:
    base = AlertView(
        id="a1",
        category="disk_low",
        kind="alert",
        severity="critical",
        title=f"{CHAT} {CHAT_ASCII}",
        detail=hostile_detail(),
        created_at=datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
        toast_state="pending",
        mail_state="pending",
        toast_at=None,
        mail_at=None,
        mail_attempts=0,
        mail_next_at=None,
        resolved_at=None,
        suppressed=False,
    )
    return dataclasses.replace(base, **changes)


def rendered_text(view: AlertView) -> str:
    rendered = render_alert(view, CHICAGO)
    return "\n".join(
        [rendered.subject, rendered.text, rendered.html, rendered.toast_title, rendered.toast_body]
    )


@pytest.mark.parametrize("kind", ["alert", "recovery"])
@pytest.mark.parametrize("category", [*SPECS, "something_new", CHAT, CHAT_ASCII])
def test_no_field_of_the_alert_template_can_carry_a_sentence(category: str, kind: str) -> None:
    """The category, the title and the details all hold chat; none of it is in the notice."""
    text = rendered_text(view_with(category=category, kind=kind))
    for canary in CANARIES:
        assert canary not in text, (category, canary)


def test_an_unknown_category_is_named_only_when_it_is_a_plain_identifier() -> None:
    assert "something_new" in rendered_text(view_with(category="something_new"))
    assert UNKNOWN_LABEL in rendered_text(view_with(category=CHAT))
    assert UNKNOWN_LABEL in rendered_text(view_with(category="Not A Name"))


def test_the_fields_of_the_alert_notice_are_wording_time_and_filtered_numbers() -> None:
    text = render_alert(view_with(), CHICAGO).text
    allowed_prefixes = (
        "严重：",
        "时间：",
        "怎么办：",
        "count = ",
        "（这封通知不包含任何聊天内容。）",
    )
    assert all(line.startswith(allowed_prefixes) for line in text.splitlines()), text


def test_the_emergency_notice_and_its_mail_have_no_field_for_text() -> None:
    assert [f.name for f in dataclasses.fields(EmergencyNotice)] == ["at", "recipient"]
    assert [f.name for f in dataclasses.fields(EmergencyEmail)] == ["to", "subject", "body"]
    notice = EmergencyNotice(datetime(2026, 10, 9, 12, 0, tzinfo=UTC), f"{CHAT_ASCII}@example.org")
    mail = render_notice(notice, CHICAGO)
    assert CHAT_ASCII not in mail.body and CHAT_ASCII not in mail.subject
    assert "2026-10-09 07:00（America/Chicago）" in mail.body


def test_the_cost_report_holds_only_numbers_dates_and_closed_names() -> None:
    hints = typing.get_type_hints(CostReport)
    allowed = {"month": str, "zone": str}
    for name, hint in hints.items():
        if name in allowed:
            assert hint is allowed[name]
        else:
            assert hint in {SpendSummary, float, int, tuple[SpendSummary, ...]}, name
    summary_hints = typing.get_type_hints(SpendSummary)
    assert {n for n, h in summary_hints.items() if h is str} == {"key"}  # a purpose or a model


def test_the_names_in_the_cost_report_come_from_the_closed_purpose_list() -> None:
    """The one writer of ``cost_ledger`` stores ``Purpose.value`` and the model's own name."""
    from twin.llm.types import Purpose

    writers = []
    for path, tree in sources():
        for call in ast.walk(tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "LedgerRecord"
            ):
                keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
                writers.append((rel(path), keywords["purpose"], keywords["model"]))
    assert writers == [("src/twin/llm/deepseek.py", "result.purpose.value", "result.model")]
    assert {p.value for p in Purpose} >= {"reply", "plan", "proactive", "summary", "persona"}
