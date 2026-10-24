"""The user may be in danger: step out of the role and give help (R-SAFE-001).

Three stages, cheapest first:

1. a **keyword screen** (``safety.crisis_keywords_file``, wide on purpose): most messages stop
   here and cost nothing;
2. the model's **second judgement** (``is_crisis``, ``severity``, ``reason``, JSON): a hit is
   confirmed or dismissed with the last lines of the conversation as context - "笑死我了" is not
   a crisis, "我真的不想活了" is.  The call is a ``reply`` purpose, so the budget never holds it
   back.  When the model cannot be asked (no key, network, circuit open) the keyword hit counts as
   confirmed: a missed crisis costs more than one out-of-character answer;
3. a **confirmed crisis**: the reply leaves the role - concern, a pointer to people nearby and to
   professional help, and the help line of the country the bot lives in - an alert is written
   (never with chat content), and, if the user enabled it and gave an address, the emergency
   contact is told that he may need care (:mod:`twin.engine.safety.notifier`).

The reply is fixed text, not generated: what is said in such a moment must not depend on a model's
mood.  After it, the engine does not play the role for that round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from twin.clock import Clock
from twin.config.lists import load_word_list, locate_list_file
from twin.config.settings import SafetyConfig
from twin.engine.safety.hotlines import hotlines_for
from twin.engine.safety.notifier import EmergencyNotice, EmergencyNotifier
from twin.llm.deepseek import DeepSeekClient
from twin.llm.types import Purpose
from twin.ops.alerts import AlertSink
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import PromptText, TemplateStore
from twin.schedule.time_service import TimeService

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.engine.safety.crisis")

TEMPLATE = "crisis_check"
MAX_JUDGE_TOKENS = 200
MAX_CONTEXT_LINES = 6
Severity = Literal["none", "low", "medium", "high", "unknown"]
CATEGORY = "crisis"


class CrisisJudgement(BaseModel):
    """The model's second judgement (R-SAFE-001)."""

    model_config = ConfigDict(extra="ignore")

    is_crisis: bool
    severity: Literal["none", "low", "medium", "high"] = "none"
    reason: str = ""


@dataclass(frozen=True)
class CrisisAssessment:
    """What the screen and the judgement found out."""

    is_crisis: bool
    severity: Severity
    keyword_hits: int
    judged: bool  # the model's judgement was obtained (else the keyword hit stood alone)


@dataclass(frozen=True)
class CrisisOutcome:
    """A confirmed crisis: what to say instead of the role, and what was done about it."""

    assessment: CrisisAssessment
    bubbles: tuple[str, ...]
    hotlines: tuple[str, ...]
    contact_notified: bool


class CrisisScreen:
    """The keyword screen: case-insensitive substrings from the list file."""

    def __init__(self, keywords: Sequence[str]) -> None:
        self._keywords = tuple(dict.fromkeys(k.lower() for k in keywords if k.strip()))

    @classmethod
    def from_file(cls, path: Path) -> CrisisScreen:
        return cls(load_word_list(path))

    @classmethod
    def from_services(cls, services: Services) -> CrisisScreen:
        return cls.from_file(
            locate_list_file(services.paths.root, services.settings.safety.crisis_keywords_file)
        )

    def __len__(self) -> int:
        return len(self._keywords)

    def hits(self, text: str) -> int:
        """How many different keywords occur in ``text``."""
        lowered = text.lower()
        return sum(1 for keyword in self._keywords if keyword in lowered)


def crisis_bubbles(hotlines: Sequence[str], severity: Severity) -> tuple[str, ...]:
    """The reply that steps out of the role: care, people nearby, professional help, help lines."""
    lines = [
        "我想先停一下，不聊刚才的话题了。你刚才说的话让我很担心你。",
        "你现在还好吗？如果你觉得撑不住了，先不要一个人扛着，找身边信得过的人陪陪你，"
        "家人或朋友都可以，哪怕只是待在一起。",
    ]
    if severity == "high":
        lines.append("如果你现在就有危险，请马上联系身边的人，或者拨打当地的急救电话。")
    if hotlines:
        lines.append("也可以马上联系专业的人：" + "；".join(hotlines) + "。")
    lines.append("我在这儿听你说。你愿意的话，慢慢告诉我发生了什么。")
    return tuple(lines)


class CrisisHandler:
    """Screens a user message, confirms a hit with the model and builds the answer."""

    def __init__(
        self,
        *,
        screen: CrisisScreen,
        client: DeepSeekClient,
        templates: TemplateStore,
        safety: SafetyConfig,
        time_service: TimeService,
        alerts: AlertSink,
        clock: Clock,
        notifier: EmergencyNotifier | None = None,
    ) -> None:
        self._screen = screen
        self._client = client
        self._templates = templates
        self._safety = safety
        self._time = time_service
        self._alerts = alerts
        self._clock = clock
        self._notifier = notifier

    @classmethod
    def from_services(
        cls,
        services: Services,
        client: DeepSeekClient,
        *,
        time_service: TimeService,
        notifier: EmergencyNotifier | None = None,
    ) -> CrisisHandler:
        return cls(
            screen=CrisisScreen.from_services(services),
            client=client,
            templates=TemplateStore(services.db, services.clock),
            safety=services.settings.safety,
            time_service=time_service,
            alerts=services.alerts,
            clock=services.clock,
            notifier=notifier,
        )

    # --------------------------------------------------------------------- assessing

    def screen(self, texts: Sequence[str]) -> int:
        """Keyword hits in the user's new messages (the first, free stage)."""
        return sum(self._screen.hits(text) for text in texts)

    async def assess(
        self, texts: Sequence[str], context: Sequence[str] = ()
    ) -> CrisisAssessment | None:
        """``None`` when the screen finds nothing; else the model's judgement of the hit.

        ``texts`` are the user's messages of this round, ``context`` the lines before them.
        """
        hits = self.screen(texts)
        if hits == 0:
            return None
        lines = [*context, *texts][-MAX_CONTEXT_LINES:]
        try:
            template: PromptText = self._templates.active(TEMPLATE)
            judged = await self._client.chat_json(
                template.render(lines="\n".join(f"- {line}" for line in lines)),
                CrisisJudgement,
                purpose=Purpose.REPLY,
                max_tokens=MAX_JUDGE_TOKENS,
            )
        except Exception as exc:  # the model could not be asked: the keyword hit stands
            log.warning("crisis_judgement_unavailable", reason=type(exc).__name__, hits=hits)
            return CrisisAssessment(True, "unknown", hits, False)
        verdict = judged.value
        log.info("crisis_judged", is_crisis=verdict.is_crisis, severity=verdict.severity, hits=hits)
        return CrisisAssessment(verdict.is_crisis, verdict.severity, hits, True)

    # ------------------------------------------------------------------- responding

    async def handle(
        self, texts: Sequence[str], context: Sequence[str] = ()
    ) -> CrisisOutcome | None:
        """The whole procedure: ``None`` for an ordinary message, else the out-of-role answer."""
        assessment = await self.assess(texts, context)
        if assessment is None or not assessment.is_crisis:
            return None
        hotlines = hotlines_for(self._safety, self._time.bot_timezone().key)
        notified = await self._tell_the_contact()
        self._alerts.raise_alert(
            CATEGORY,
            "The user may be in a crisis; the bot stepped out of the role",
            severity="critical",
            detail={
                "severity": assessment.severity,
                "keyword_hits": assessment.keyword_hits,
                "judged": assessment.judged,
                "contact_notified": notified,
            },
        )
        return CrisisOutcome(
            assessment,
            crisis_bubbles(hotlines, assessment.severity),
            hotlines,
            notified,
        )

    async def _tell_the_contact(self) -> bool:
        """Send the one fixed message to the emergency contact, if it is switched on."""
        contact = self._safety.emergency_contact
        email = (contact.email or "").strip()
        if not contact.enabled or not email:
            return False
        if self._notifier is None:
            self._alerts.raise_alert(
                "emergency_contact",
                "The emergency contact is switched on but no mail service is set up",
                severity="warning",
            )
            return False
        try:
            return await self._notifier.notify(EmergencyNotice(self._clock.now_utc(), email))
        except Exception as exc:  # a mail problem must not stop the answer to the user
            log.warning("emergency_notice_failed", reason=type(exc).__name__)
            self._alerts.raise_alert(
                "emergency_contact",
                "The message to the emergency contact could not be sent",
                severity="warning",
                detail={"reason": type(exc).__name__},
            )
            return False
