"""A simulated WeChat platform and phone for the channel probe tests.

:class:`PlatformSim` implements the runner's ``ProbeChannel`` interface with the rules a real
ClawBot might have (a number of messages after each inbound message, a window of hours) so a
whole 26-hour probe can run in a fraction of a second on a virtual clock.  It records what
"reached the phone", which is what :class:`AutoUser` reports when the probe asks.  Every send
goes through the real ``ProbeSendPolicy`` exactly as the real channel would call it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tests.support.alerts import RecordingAlerts
from tests.support.clock import InstantClock
from tests.support.ilink import RecordingBanner
from twin.channel.base import (
    AuthState,
    BypassRequest,
    OutboundKind,
    OutboundResult,
    SendBypass,
)
from twin.channel.probe.images import ProbeImage, make_probe_images
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    AttemptPhase,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    QuestionKind,
)
from twin.channel.probe.policy import ProbeSendPolicy
from twin.channel.probe.runner import InboundMarker, ProbeRunner
from twin.channel.probe.store import ProbeStore
from twin.storage.db import Database


@dataclass
class Delivered:
    at: datetime
    kind: str  # "text" or "image"
    text: str | None = None
    image: str | None = None  # "jpg", "png" or "gif"
    after_inbound_h: float = 0.0
    epoch: int = 0  # which inbound message it followed


@dataclass
class SentLog:
    at: datetime
    kind: str
    text: str | None
    result: str
    empty_token: bool = False


class PlatformSim:
    """The platform, the phone and the login, driven by a virtual clock."""

    def __init__(
        self,
        clock: InstantClock,
        images: dict[str, ProbeImage],
        *,
        quota: int | None = 8,
        window_h: float | None = 24.0,
    ) -> None:
        self.clock = clock
        self._names = {
            hashlib.sha256(image.data).hexdigest(): name for name, image in images.items()
        }
        self.quota = quota  # messages accepted after one inbound message (None = no limit)
        self.window_h = window_h  # hours after the inbound message (None = no limit)
        self.auth = AuthState.OK
        self.bound = True
        self.last_inbound: datetime | None = None
        self.sent_since_inbound = 0
        self.expired = False
        self.delivered: list[Delivered] = []
        self.log: list[SentLog] = []
        self.registered: list[str] = []
        self.typing: list[bool] = []
        self.token = 0
        self.scripted: dict[str, list[OutboundResult]] = {}  # text fragment -> results to return
        self.swallow_after: int | None = None  # the phone shows only this many per inbound
        self.empty_token_delivers = True
        self.reject_images_with: OutboundResult | None = None
        self.since_phone = 0

    def script(self, needle: str, *results: OutboundResult) -> None:
        """Answer the next sends whose text contains ``needle`` with ``results``, in order."""
        self.scripted.setdefault(needle, []).extend(results)

    # ------------------------------------------------------------ the user

    def user_writes(self) -> None:
        """The user sends the bot a message now (resets the count and the window)."""
        now = self.clock.now_utc()
        if self.last_inbound is not None and now <= self.last_inbound:
            now = self.last_inbound + timedelta(seconds=1)
        self.last_inbound = now
        self.sent_since_inbound = 0
        self.since_phone = 0
        self.expired = False
        self.token += 1

    # ------------------------------------------------- ProbeChannel interface

    def inbound_marker(self) -> InboundMarker:
        fingerprint = hashlib.sha256(f"ctx-{self.token}".encode()).hexdigest()[:8]
        return InboundMarker(
            last_inbound_at=self.last_inbound,
            context_fingerprint=fingerprint if self.token else None,
            auth=self.auth,
            bound=self.bound,
        )

    def register_image(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        self.registered.append(digest)
        return digest

    async def send_text(self, text: str, *, bypass: SendBypass) -> OutboundResult:
        return self._send("text", text, None, bypass)

    async def send_image(self, data: bytes, mime: str, *, bypass: SendBypass) -> OutboundResult:
        digest = hashlib.sha256(data).hexdigest()
        names = self._names
        if digest not in self.registered:
            raise AssertionError("a picture was sent without being registered first")
        return self._send("image", None, names.get(digest, "?"), bypass)

    async def send_typing(self, active: bool) -> None:
        self.typing.append(active)

    # ---------------------------------------------------------- the platform

    def _send(
        self, kind: str, text: str | None, image: str | None, bypass: SendBypass
    ) -> OutboundResult:
        now = self.clock.now_utc()
        grant = bypass.authorize(
            BypassRequest("text" if kind == "text" else "image", text, now, None)
        )  # type: ignore[arg-type]
        empty = bool(grant and grant.empty_context_token)
        label = text or f"picture {image}"
        for needle, queue in self.scripted.items():
            if queue and needle in label:
                result = queue.pop(0)
                self.log.append(SentLog(now, kind, text, f"scripted {result.kind.value}", empty))
                return result
        if self.last_inbound is None or self.token == 0:
            self.log.append(SentLog(now, kind, text, "no_context_token", empty))
            return OutboundResult.failure(OutboundKind.WINDOW_REJECTED, "no_context_token")
        if self.expired:
            self.log.append(SentLog(now, kind, text, "session_expired", empty))
            return OutboundResult.failure(
                OutboundKind.WINDOW_REJECTED, "session_expired", session_expired=True
            )
        elapsed = (now - self.last_inbound).total_seconds() / 3600.0
        too_late = self.window_h is not None and elapsed > self.window_h
        too_many = self.quota is not None and self.sent_since_inbound >= self.quota
        if too_late or too_many:
            self.expired = True
            self.log.append(SentLog(now, kind, text, "platform_rejected", empty))
            return OutboundResult.failure(
                OutboundKind.WINDOW_REJECTED,
                "platform_rejected",
                code=-2,
                ret=-2,
                errmsg="prepare failed",
                session_expired=True,
                client_id="c-" + label[:6],
            )
        if kind == "image" and self.reject_images_with is not None:
            self.log.append(SentLog(now, kind, text, "image_rejected", empty))
            return self.reject_images_with
        self.sent_since_inbound += 1
        reaches = not (empty and not self.empty_token_delivers) and (
            self.swallow_after is None or self.since_phone < self.swallow_after
        )
        if reaches:
            self.since_phone += 1
            self.delivered.append(Delivered(now, kind, text, image, elapsed, self.token))
        self.log.append(SentLog(now, kind, text, "ok", empty))
        return OutboundResult.success(message_id=f"m{len(self.log)}", client_id=f"c{len(self.log)}")

    # --------------------------------------------------------- the phone

    def phone_count(self, marker: str) -> int:
        """Messages containing ``marker`` on the phone since the latest inbound they followed."""
        matching = [d for d in self.delivered if d.text and marker in d.text]
        if not matching:
            return 0
        return sum(1 for d in matching if d.epoch == matching[-1].epoch)

    def phone_has(self, image: str) -> bool:
        return any(d.image == image for d in self.delivered)


@dataclass
class AutoUser:
    """Behaves like a cooperative person: writes when asked, tells the truth about the phone."""

    sim: PlatformSim
    store: ProbeStore
    gif_moves: bool = True
    typing_visible: str = "yes"
    writes_when_asked: bool = True
    answers: dict[str, str] = field(default_factory=dict)
    wrote_for: set[tuple[str, int]] = field(default_factory=set)
    interrupt_at: dict[tuple[str, int], timedelta] = field(default_factory=dict)

    def react(self, plan: ProbePlan) -> None:
        step = plan.active_step()
        attempt = step.current_attempt() if step else None
        if step is None or attempt is None:
            return
        key = (step.id.value, attempt.n)
        if (
            attempt.phase is AttemptPhase.WAITING
            and self.writes_when_asked
            and key not in self.wrote_for
        ):
            self.wrote_for.add(key)
            self.sim.user_writes()
        for question in ProbeStore.pending_questions(plan):
            self.store.answer(
                question.id, self._answer(question.id, question.kind, question.choices)
            )

    def _answer(self, question_id: str, kind: QuestionKind, choices: list[str]) -> str:
        if question_id in self.answers:
            return self.answers[question_id]
        sim = self.sim
        if kind is QuestionKind.READY:
            return "ready"
        if question_id.endswith("-count"):
            if question_id.startswith("count"):
                return str(sim.phone_count("条数测试"))
            if question_id.startswith("window"):
                return str(sim.phone_count("窗口测试"))
            return str(sim.phone_count("空令牌实验"))
        if question_id.endswith("-gif"):
            if not sim.phone_has("gif"):
                return "missing"
            return "moving" if self.gif_moves else "still"
        if question_id.endswith(("-jpg", "-png")):
            return "arrived" if sim.phone_has(question_id[-3:]) else "missing"
        if question_id.endswith("-typing-seen"):
            return self.typing_visible
        raise AssertionError(f"unexpected question {question_id}")


@dataclass
class Rig:
    """Everything a probe test needs, wired together."""

    clock: InstantClock
    db: Database
    store: ProbeStore
    sim: PlatformSim
    user: AutoUser
    runner: ProbeRunner
    policy: ProbeSendPolicy
    banner: RecordingBanner
    alerts: RecordingAlerts

    async def run(self, *, max_ticks: int = 20000) -> ProbePlan:
        """Run the probe to the end of the plan, like the application would."""
        for _ in range(max_ticks):
            delay = await self.runner.tick()
            plan = self.store.load()
            assert plan is not None
            if plan.status is not PlanStatus.RUNNING:
                return plan
            self.user.react(plan)
            await self.clock.sleep(delay)
        raise AssertionError("the probe did not finish")

    async def tick_until(
        self, check: Callable[[ProbePlan], bool], *, max_ticks: int = 5000
    ) -> ProbePlan:
        """Tick (and let the user react) until ``check(plan)`` holds."""
        for _ in range(max_ticks):
            delay = await self.runner.tick()
            plan = self.store.load()
            assert plan is not None
            if check(plan):
                return plan
            self.user.react(plan)
            await self.clock.sleep(delay)
        raise AssertionError("the condition was not reached")


def make_rig(
    db: Database,
    *,
    quota: int | None = 8,
    window_h: float | None = 24.0,
    options: ProbeOptions | None = None,
    start: bool = True,
) -> Rig:
    from tests.support.clock import DEFAULT_START

    clock = InstantClock(DEFAULT_START)
    store = ProbeStore(db, clock)
    images = make_probe_images()
    sim = PlatformSim(clock, images, quota=quota, window_h=window_h)
    sim.user_writes()  # the login just happened: the user has written once
    policy = ProbeSendPolicy(store)
    banner = RecordingBanner()
    alerts = RecordingAlerts()
    runner = ProbeRunner(
        store=store,
        channel=sim,
        policy=policy,
        clock=clock,
        alerts=alerts,
        banner=banner,
        images=images,
    )
    if start:
        store.create(options or fast_options())
    return Rig(clock, db, store, sim, AutoUser(sim, store), runner, policy, banner, alerts)


def fast_options(**changes: object) -> ProbeOptions:
    """The SPEC timings, with long polling waits shortened so tests run quickly."""
    options = ProbeOptions(watch_s=3600.0)
    for name, value in changes.items():
        setattr(options, name, value)
    return options


def sent_texts(sim: PlatformSim) -> list[str]:
    return [entry.text for entry in sim.log if entry.text is not None]


def action_by_id(plan: ProbePlan, step: str, action_id: str) -> list[Action]:
    return [
        a
        for s in plan.steps
        if s.id.value == step
        for attempt in s.attempts
        for a in attempt.actions
        if a.id == action_id
    ]


def finished_sends(plan: ProbePlan, step: str) -> list[Action]:
    return [
        a
        for s in plan.steps
        if s.id.value == step
        for attempt in s.attempts
        for a in attempt.actions
        if a.kind in (ActionKind.SEND_TEXT, ActionKind.SEND_IMAGE) and a.status is ActionStatus.DONE
    ]
