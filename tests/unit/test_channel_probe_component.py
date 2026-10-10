"""The probe as an application component on the real WeChat channel (R-CH-009).

A fake WeChat server (respx) enforces a message count and a window; the manual clock jumps over
the 26 hours; the real ``IlinkChannel`` sends, the real ``ProbeSendPolicy`` authorises, the real
state machine decides.  Only the platform and the person are simulated.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from tests.support.clock import ManualClock
from tests.support.ilink import (
    API,
    BOT,
    CDN,
    CTX,
    TOKEN,
    USER,
    RecordingBanner,
    message,
    now_ms,
    request_json,
    text_item,
    updates,
)
from tests.support.waiting import wait_until
from twin.app import Application, HealthStatus
from twin.channel.base import TEST_PREFIX
from twin.channel.component import ChannelComponent, register_channel
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.ilink.store import Credentials
from twin.channel.probe.component import ChannelProbe, build_runner, register_probe
from twin.channel.probe.model import (
    AttemptPhase,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    QuestionKind,
    StepId,
)
from twin.channel.probe.store import ProbeStore
from twin.channel.probe.summary import VERDICT_MET, load_channel_probe_summary
from twin.config.loader import load_settings
from twin.services import Services

SEND = f"{API}/ilink/bot/sendmessage"
GET_UPDATES = f"{API}/ilink/bot/getupdates"
UPLOAD_URL = f"{API}/ilink/bot/getuploadurl"
GETCONFIG = f"{API}/ilink/bot/getconfig"
SENDTYPING = f"{API}/ilink/bot/sendtyping"


class WeChatServer:
    """Just enough of the iLink server: counts and a window after each message of the user."""

    def __init__(
        self, api: respx.MockRouter, clock: ManualClock, *, quota: int, window_h: float
    ) -> None:
        self.clock = clock
        self.quota = quota
        self.window_h = window_h
        self.epoch = 0
        self.sent = 0
        self.expired = False
        self.last_inbound: datetime | None = clock.now_utc()  # the message sent at login
        self.delivered: list[tuple[int, str, str | None]] = []  # (epoch, kind, text)
        self.requests: list[dict[str, Any]] = []
        self._queue: list[dict[str, Any]] = []
        self._ids = itertools.count(1)
        api.post(SEND).mock(side_effect=self._send)
        api.post(GET_UPDATES).mock(side_effect=self._poll)
        api.post(UPLOAD_URL).respond(200, json={"upload_param": "UP"})
        api.post(url__startswith=f"{CDN}/upload").respond(200, headers={"x-encrypted-param": "D"})
        api.post(GETCONFIG).respond(200, json={"ret": 0, "typing_ticket": "TICKET"})
        self.typing_route = api.post(SENDTYPING).respond(200, json={})
        for notice in ("notifystart", "notifystop"):
            api.post(f"{API}/ilink/bot/msg/{notice}").respond(200, json={"ret": 0})

    def user_writes(self) -> None:
        self.epoch += 1
        self.sent = 0
        self.expired = False
        self.last_inbound = self.clock.now_utc()
        self._queue.append(
            message(text_item("hi"), mid=next(self._ids), created_ms=now_ms(self.clock))
        )

    def _poll(self, _request: httpx.Request) -> httpx.Response:
        batch, self._queue = self._queue, []
        return httpx.Response(200, json=updates(batch, cursor=f"C{next(self._ids)}"))

    def _send(self, request: httpx.Request) -> httpx.Response:
        body = request_json(request)["msg"]
        self.requests.append(body)
        assert self.last_inbound is not None
        elapsed_h = (self.clock.now_utc() - self.last_inbound).total_seconds() / 3600
        if self.expired or self.sent >= self.quota or elapsed_h > self.window_h:
            self.expired = True
            return httpx.Response(200, json={"ret": -2, "errmsg": "prepare failed"})
        self.sent += 1
        item = body["item_list"][0]
        text = item.get("text_item", {}).get("text") if item["type"] == 1 else None
        self.delivered.append((self.epoch, "text" if item["type"] == 1 else "image", text))
        return httpx.Response(200, json={})

    def count(self, marker: str) -> int:
        texts = [(e, t) for e, k, t in self.delivered if t and marker in t]
        if not texts:
            return 0
        return sum(1 for e, _t in texts if e == texts[-1][0])

    def images(self) -> int:
        epochs = [e for e, kind, _t in self.delivered if kind == "image"]
        return sum(1 for e in epochs if e == epochs[-1]) if epochs else 0


@pytest.fixture
def api() -> Any:
    with respx.mock(assert_all_called=False) as router:
        yield router


def answer_for(server: WeChatServer, question_id: str, kind: QuestionKind) -> str:
    if kind is QuestionKind.READY:
        return ""
    if question_id.endswith("-count"):
        marker = "条数测试" if question_id.startswith("count") else "窗口测试"
        return str(server.count(marker))
    if question_id.endswith("-typing-seen"):
        return "yes"
    position = {"jpg": 1, "png": 2, "gif": 3}[question_id[-3:]]
    if question_id.endswith("-gif"):
        return "moving" if server.images() >= position else "missing"
    return "arrived" if server.images() >= position else "missing"


# ----------------------------------------------------------- the component


async def test_the_probe_component_is_added_after_the_channel_and_only_with_it(
    services: Services, tmp_path: Path
) -> None:
    console_settings = load_settings(
        None, {"paths": {"data_dir": str(tmp_path / "x")}, "channel": {"kind": "console"}}
    )
    bare = Application()
    assert register_probe(bare, replace(services, settings=console_settings)) is None
    application = Application()
    assert register_probe(application, services) is None  # no channel component yet
    channel = register_channel(application, services)
    probe = register_probe(application, services)
    assert isinstance(channel, ChannelComponent) and isinstance(probe, ChannelProbe)
    assert [c.name for c in application.start_order()] == ["channel", "channel_probe"]
    assert probe.depends_on == ("channel",)


async def test_the_idle_component_does_nothing_and_stops_cleanly(
    services: Services, api: respx.MockRouter, clock: ManualClock
) -> None:
    server = WeChatServer(api, clock, quota=8, window_h=24)
    application = Application()
    register_channel(application, services)
    probe = register_probe(application, services)
    assert probe is not None
    await application.start()
    try:
        await wait_until(lambda: clock.pending_sleepers >= 1, interval=0.002)
        assert application.health()["channel_probe"].status is HealthStatus.OK
        assert server.requests == []  # no plan: nothing is sent
    finally:
        await application.stop()


# ------------------------------------------------- the whole probe, end to end


async def test_the_application_carries_the_probe_out_over_the_real_channel(
    services: Services, api: respx.MockRouter, clock: ManualClock
) -> None:
    server = WeChatServer(api, clock, quota=10, window_h=24)  # more than the safe 8 and 22 h
    application = Application()
    # no background polling: the test hands the user's messages over itself (poll_once)
    channel = IlinkChannel.from_services(services, poll=False)
    application.register(ChannelComponent(channel))
    channel.store.save_credentials(Credentials(TOKEN, BOT, USER, API, clock.now_utc().isoformat()))
    channel.store.bind(USER, context_token=CTX)
    banner = RecordingBanner()
    probe = ChannelProbe(build_runner(services, channel, banner=banner), clock, services.alerts)
    application.register(probe)
    store = ProbeStore(services.db, clock)
    plan = store.create(ProbeOptions(watch_s=3600.0, typing_hold_s=0.0, image_gap_s=0.0))
    injected: set[tuple[str, int]] = set()

    async def react(current: ProbePlan) -> None:
        step = current.active_step()
        attempt = step.current_attempt() if step else None
        if step is not None and attempt is not None:
            key = (step.id.value, attempt.n)
            if attempt.phase is AttemptPhase.WAITING and key not in injected:
                injected.add(key)
                server.user_writes()
                await channel.poll_once()
        for question in store.pending_questions(current):
            store.answer(question.id, answer_for(server, question.id, question.kind))

    await application.start()
    try:
        for _ in range(3000):
            current = store.load()
            assert current is not None
            if current.status is not PlanStatus.RUNNING:
                break
            await react(current)
            await wait_until(lambda: clock.pending_sleepers >= 1, interval=0.002)
            await clock.advance(clock.sleeps[-1])
        else:
            raise AssertionError("the probe did not finish")
    finally:
        await application.stop()

    final = store.load()
    assert final is not None and final.status is PlanStatus.COMPLETED
    assert final.run_id == plan.run_id
    # what the server did: more than the safe count and window were used, but only by the probe
    assert (
        max(len(list(g)) for _, g in itertools.groupby(e for e, _k, _t in server.delivered)) == 10
    )
    texts = [t for _e, kind, t in server.delivered if kind == "text"]
    assert texts and all(t and t.startswith(TEST_PREFIX) for t in texts)
    assert {body["to_user_id"] for body in server.requests} == {USER}
    assert {body["context_token"] for body in server.requests} == {CTX}
    assert server.typing_route.call_count == 2  # shown, then cleared
    assert [e["decision"] for e in store.audit()].count("refused") == 0
    assert len(store.audit()) == len(server.requests)
    # the measured result, as the milestone check reads it
    with services.db.session() as session:
        summary = load_channel_probe_summary(session)
    assert summary is not None and summary.complete and summary.verdict == VERDICT_MET
    assert summary.n_messages == 10 and summary.gif_animated is True
    assert summary.window_lower_bound_h is not None and 23 <= summary.window_lower_bound_h < 24
    assert summary.window_upper_bound_h is not None and summary.window_upper_bound_h >= 25
    failures = [f for f in summary.failures if f["step"] == "count"]
    assert failures and failures[0]["ret"] == -2 and failures[0]["errmsg"] == "prepare failed"
    # three fresh messages were asked for, in the terminal and in WeChat
    assert injected == {("count", 1), ("media", 1), ("window", 1)}
    assert sum("send the bot a message" in title for title, _lines in banner.shown) == 3
    sent_texts = [
        body["item_list"][0]["text_item"]["text"]
        for body in server.requests
        if body["item_list"][0]["type"] == 1
    ]
    announcements = [t for t in sent_texts if "即将开始" in t]
    # WeChat can only carry the reminder while the platform still takes messages: after step 1
    # ended with the platform's refusal the reminder for step 2 stays in the terminal
    assert [("第1步" in t, "第3步" in t) for t in announcements] == [(True, False), (False, True)]
    assert all("请现在给我发一条任意消息" in t for t in announcements)
    media_attempt = final.step(StepId.MEDIA).attempts[0]
    assert media_attempt.announce_note == "not delivered (window_rejected: session_expired)"
    assert final.step(StepId.COUNT).attempts[0].announce_note == "sent"
    # and what was measured now shows in the channel's capabilities
    measured = IlinkChannel.from_services(services, poll=False).capabilities()
    assert measured.outbound_quota == 10 and measured.gif_animated is True
    assert measured.proactive_window_h == summary.window_lower_bound_h
    assert StepId.COUNT.value in summary.steps


async def test_the_configured_thresholds_are_not_changed_by_what_was_measured(
    services: Services,
) -> None:
    channel = IlinkChannel.from_services(services, poll=False)
    assert channel.window().window_h == services.settings.channel.proactive_window_safe_h
    assert channel.window().quota == services.settings.channel.outbound_quota_safe
    assert channel.capabilities().proactive_window_h is None  # nothing measured yet


async def test_twin_run_adds_the_probe_next_to_the_channel(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twin.cli import _serve
    from twin.llm.runtime import DEEPSEEK_SECRET

    services.secrets.set(DEEPSEEK_SECRET, "synthetic-test-key-0001")
    seen: list[str] = []

    async def record(self: Application, stop_event: object, *, signals: object = None) -> None:
        seen.extend(sorted(self.components))

    monkeypatch.setattr(Application, "run", record)
    await _serve(services)
    assert seen == [
        "backend_monitor",  # the style model's health, looked at while nobody talks (round 09)
        "channel",
        "channel_probe",
        "engine",
        "heartbeat",
        "import_report",  # says how an import started from the chat ended (round 11)
        "job_worker",
        "learning",  # queues the weekly consolidation of the correction rules (round 11)
        "power_events",
        "proactive",  # she writes first, never while she sleeps (round 10)
        "schedule",
        "state_watcher",
    ]
