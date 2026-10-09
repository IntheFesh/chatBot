"""The safety boundaries: crisis, promises, "are you an AI?", the emergency contact (R-SAFE)."""

from __future__ import annotations

import ast
import dataclasses
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import respx
from sqlalchemy import select

from tests.support.deepseek import API, TEST_KEY, error, ok
from twin.config.runtime import BOT_TIMEZONE
from twin.config.settings import SafetyConfig
from twin.engine.safety.commitments import CommitmentDetector
from twin.engine.safety.crisis import CrisisHandler, CrisisScreen, crisis_bubbles
from twin.engine.safety.hotlines import country_of, hotlines_for
from twin.engine.safety.identity import asks_if_ai
from twin.engine.safety.notifier import (
    BODY_TEMPLATE,
    SUBJECT,
    EmergencyEmail,
    EmergencyNotice,
    render_notice,
)
from twin.llm.runtime import DEEPSEEK_SECRET, LlmRuntime, build_llm_runtime
from twin.schedule.service import time_service_for
from twin.services import Services
from twin.storage.models import Alert

LISTS = Path(__file__).resolve().parents[2] / "config" / "lists"
SAID = "我真的不想活了，今晚就想结束这一切"
VERDICT = '{"is_crisis": true, "severity": "high", "reason": "有明确的轻生想法"}'


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def runtime(services: Services) -> AsyncIterator[LlmRuntime]:
    services.secrets.set(DEEPSEEK_SECRET, TEST_KEY)
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


class RecordingNotifier:
    def __init__(self, result: bool = True, fail: bool = False) -> None:
        self.notices: list[EmergencyNotice] = []
        self.result = result
        self.fail = fail

    async def notify(self, notice: EmergencyNotice) -> bool:
        self.notices.append(notice)
        if self.fail:
            raise OSError("the mail server is down")
        return self.result


def handler(
    services: Services, runtime: LlmRuntime, notifier: RecordingNotifier | None = None
) -> CrisisHandler:
    return CrisisHandler.from_services(
        services, runtime.client, time_service=time_service_for(services), notifier=notifier
    )


@dataclasses.dataclass(frozen=True)
class Raised:
    category: str
    severity: str
    title: str
    detail: dict[str, object] | None


def alerts(services: Services) -> list[Raised]:
    with services.db.session() as session:
        found = session.scalars(
            select(Alert).where(Alert.category.in_(("crisis", "emergency_contact")))
        )
        return [Raised(a.category, a.severity, a.title, a.detail) for a in found]


# ------------------------------------------------------------------- the keyword screen


def test_the_screen_finds_the_words_of_the_list_wherever_they_are() -> None:
    screen = CrisisScreen.from_file(LISTS / "crisis_keywords.txt")
    assert len(screen) > 150
    assert screen.hits(SAID) >= 2
    assert screen.hits("I WANT TO DIE tonight") >= 1  # case does not matter
    assert screen.hits("今天吃什么") == 0 and screen.hits("") == 0
    assert CrisisScreen([" ", "A", "a"]).hits("banana") == 1


async def test_a_message_without_a_keyword_costs_nothing(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content=VERDICT))
    assert await handler(services, runtime).assess(["今天好开心", "晚上吃面"]) is None
    assert await handler(services, runtime).handle(["今天好开心"]) is None
    assert route.call_count == 0 and alerts(services) == []


# --------------------------------------------------------- confirmed crisis and the reply


async def test_a_confirmed_crisis_steps_out_of_the_role_with_the_help_line(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    route = api.post(API).mock(return_value=ok(content=VERDICT))
    outcome = await handler(services, runtime).handle([SAID], ["今天过得怎么样", "不太好"])
    assert outcome is not None and outcome.assessment.severity == "high"
    assert outcome.assessment.judged and outcome.assessment.keyword_hits >= 2
    text = "\n".join(outcome.bubbles)
    assert "988" in text and "12356" not in text  # America/Chicago -> the US line
    assert (
        "身边" in text and "专业" in text and "担心你" in text
    )  # people nearby, professional help
    assert "马上联系身边的人" in text  # high severity: act now
    assert (
        outcome.hotlines == ("988（美国心理危机热线，电话或短信）",)
        and not outcome.contact_notified
    )
    asked = json.loads(route.calls[0].request.content)
    assert SAID in asked["messages"][-1]["content"] and "不太好" in asked["messages"][-1]["content"]
    assert asked["messages"][-1]["content"].index("今天过得怎么样") < asked["messages"][-1][
        "content"
    ].index("不太好")


async def test_the_help_line_follows_the_time_zone_of_the_bot(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    services.runtime.set(BOT_TIMEZONE, "Asia/Shanghai")
    china = await handler(services, runtime).handle([SAID])
    assert china is not None and "12356" in "\n".join(china.bubbles)
    assert "988" not in "\n".join(china.bubbles)
    services.runtime.set(BOT_TIMEZONE, "Europe/London")
    elsewhere = await handler(services, runtime).handle([SAID])
    assert elsewhere is not None
    both = "\n".join(elsewhere.bubbles)
    assert "988" in both and "12356" in both  # not mapped: every line is given


def test_the_mapping_picks_the_line_of_the_country() -> None:
    safety = SafetyConfig()
    assert (
        country_of(safety, "America/Chicago") == "US"
        and country_of(safety, "Asia/Shanghai") == "CN"
    )
    assert country_of(safety, "Europe/Paris") is None
    assert hotlines_for(safety, "America/Chicago") == (safety.hotlines["US"],)
    assert hotlines_for(safety, "Asia/Shanghai") == (safety.hotlines["CN"],)
    assert hotlines_for(safety, "Europe/Paris") == tuple(safety.hotlines.values())
    mapped_without_a_line = SafetyConfig(
        hotlines={"US": "988"}, timezone_country={"Asia/Tokyo": "JP"}
    )
    assert hotlines_for(mapped_without_a_line, "Asia/Tokyo") == ("988",)


def test_the_reply_is_fixed_text_and_the_urgent_line_only_for_high_severity() -> None:
    plain = crisis_bubbles(("988",), "medium")
    urgent = crisis_bubbles(("988",), "high")
    assert len(urgent) == len(plain) + 1 and "急救电话" in "".join(urgent)
    assert "急救电话" not in "".join(plain) and "988" in "".join(plain)
    assert "988" not in "".join(crisis_bubbles((), "high"))


async def test_the_alert_says_that_it_happened_and_never_what_was_said(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    await handler(services, runtime).handle([SAID])
    (alert,) = alerts(services)
    assert alert.category == "crisis" and alert.severity == "critical"
    assert alert.detail is not None and alert.detail["severity"] == "high"
    assert alert.detail["judged"] is True and alert.detail["contact_notified"] is False
    blob = f"{alert.title} {alert.detail}"
    assert "不想活" not in blob and SAID not in blob


async def test_a_keyword_the_model_dismisses_is_not_a_crisis(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(
        return_value=ok(content='{"is_crisis": false, "severity": "none", "reason": "开玩笑"}')
    )
    assert await handler(services, runtime).handle(["笑死我了，气得想杀了他"]) is None
    assessment = await handler(services, runtime).assess(["笑死我了，气得想杀了他"])
    assert assessment is not None and not assessment.is_crisis and assessment.judged
    assert alerts(services) == []


@pytest.mark.parametrize(
    "response",
    [
        error(401, "bad key"),
        ok(content="not json at all"),
        error(400, "invalid"),
    ],
)
async def test_when_the_model_cannot_be_asked_the_keyword_hit_stands(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter, response: object
) -> None:
    api.post(API).mock(side_effect=[response] * 4)
    outcome = await handler(services, runtime).handle([SAID])
    assert outcome is not None and outcome.assessment.severity == "unknown"
    assert not outcome.assessment.judged
    assert "988" in "\n".join(outcome.bubbles) and len(alerts(services)) == 1


async def test_with_no_key_at_all_a_hit_still_gets_the_caring_answer(services: Services) -> None:
    runtime = build_llm_runtime(services)  # no key stored
    try:
        outcome = await handler(services, runtime).handle([SAID])
    finally:
        await runtime.client.aclose()
    assert outcome is not None and not outcome.assessment.judged


# --------------------------------------------------------------------- the emergency contact


async def test_the_emergency_contact_is_off_by_default(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    notifier = RecordingNotifier()
    outcome = await handler(services, runtime, notifier).handle([SAID])
    assert outcome is not None and not outcome.contact_notified and notifier.notices == []
    assert services.settings.safety.emergency_contact.enabled is False


async def test_when_switched_on_with_an_address_one_notice_goes_out(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    services.settings.safety.emergency_contact.enabled = True
    services.settings.safety.emergency_contact.email = "friend@example.org"
    notifier = RecordingNotifier()
    outcome = await handler(services, runtime, notifier).handle([SAID])
    assert outcome is not None and outcome.contact_notified
    (notice,) = notifier.notices
    assert notice.recipient == "friend@example.org" and notice.at == services.clock.now_utc()
    (alert,) = alerts(services)
    assert alert.detail is not None and alert.detail["contact_notified"] is True


async def test_switched_on_without_an_address_nothing_is_sent(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    services.settings.safety.emergency_contact.enabled = True
    services.settings.safety.emergency_contact.email = "  "
    notifier = RecordingNotifier()
    outcome = await handler(services, runtime, notifier).handle([SAID])
    assert outcome is not None and notifier.notices == [] and not outcome.contact_notified


async def test_a_mail_problem_never_stops_the_answer_to_the_user(
    services: Services, runtime: LlmRuntime, api: respx.MockRouter
) -> None:
    api.post(API).mock(return_value=ok(content=VERDICT))
    services.settings.safety.emergency_contact.enabled = True
    services.settings.safety.emergency_contact.email = "friend@example.org"
    broken = await handler(services, runtime, RecordingNotifier(fail=True)).handle([SAID])
    assert broken is not None and broken.bubbles and not broken.contact_notified
    categories = sorted(a.category for a in alerts(services))
    assert categories == ["crisis", "emergency_contact"]
    nobody = await handler(services, runtime, None).handle([SAID])
    assert nobody is not None and not nobody.contact_notified
    assert sorted(a.category for a in alerts(services)).count("emergency_contact") == 2
    refused = await handler(services, runtime, RecordingNotifier(result=False)).handle([SAID])
    assert refused is not None and not refused.contact_notified


def test_the_notice_has_no_field_for_chat_content() -> None:
    assert {f.name for f in dataclasses.fields(EmergencyNotice)} == {"at", "recipient"}
    assert {f.name for f in dataclasses.fields(EmergencyEmail)} == {"to", "subject", "body"}
    assert "{time}" in BODY_TEMPLATE and BODY_TEMPLATE.count("{") == 1  # the time is all there is


def test_the_email_is_made_of_the_time_alone() -> None:
    at = datetime(2026, 10, 9, 3, 30, tzinfo=UTC)
    email = render_notice(EmergencyNotice(at, "friend@example.org"), ZoneInfo("America/Chicago"))
    assert email.to == "friend@example.org" and email.subject == SUBJECT
    assert "2026-10-08 22:30（America/Chicago）" in email.body
    assert "他可能需要关心" in email.body and "不包含任何聊天内容" in email.body
    assert SAID not in email.body
    with pytest.raises(ValueError, match="recipient"):
        EmergencyNotice(at, " ")
    with pytest.raises(ValueError, match="naive"):
        EmergencyNotice(datetime(2026, 1, 1), "a@b.c")  # noqa: DTZ001


# ----------------------------------------------------------------------------- promises


def test_the_promise_patterns_load_from_the_file_of_the_settings(services: Services) -> None:
    detector = CommitmentDetector.from_services(services)
    assert len(detector) > 30
    for promise in (
        "我给你发个语音",
        "等下给你拍张照片",
        "我晚上给你打电话",
        "我们视频吧",
        "周末见面吧",
        "我转你200",
        "红包发你了",
        "我去找你",
    ):
        assert detector.promises(promise), promise
    for talk in ("你吃饭了吗", "哈哈哈哈", "今天好累啊", "我刚到家", "想你了"):
        assert detector.find(talk) is None, talk


# ----------------------------------------------------------------------- "are you an AI?"


@pytest.mark.parametrize(
    "question",
    [
        "你是不是AI啊",
        "你是不是机器人",
        "你是真人吗",
        "你是机器人吗",
        "你是AI吗",
        "你到底是不是人工智能",
        "你是人还是机器",
        "你是不是在用ai回我",
        "说真的，你是不是真人",
    ],
)
def test_questions_about_being_an_ai_are_recognised(question: str) -> None:
    assert asks_if_ai(question), question


@pytest.mark.parametrize(
    "talk", ["我今天看了人工智能的新闻", "你是不是饿了", "你是我的宝贝", "机器坏了", "好累啊"]
)
def test_other_talk_is_not_taken_for_that_question(talk: str) -> None:
    assert not asks_if_ai(talk), talk


# ------------------------------------------------------------------------ no pictures out


def test_nothing_in_the_engine_package_sends_a_picture_by_itself() -> None:
    """R-SAFE-006: pictures leave only as library stickers, through the sticker sender alone."""
    root = Path(__file__).resolve().parents[2] / "src" / "twin" / "engine"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "sticker_sender.py":  # the one door, guarded by its own test (09-3)
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "send_image":
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
