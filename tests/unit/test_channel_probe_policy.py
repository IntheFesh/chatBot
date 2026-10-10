"""``ProbeSendPolicy``: the probe, and only the probe, may skip the safe thresholds (R-CH-009)."""

from __future__ import annotations

import ast
import inspect
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx

from tests.support.alerts import RecordingAlerts
from tests.support.clock import ManualClock
from tests.support.ilink import API, CTX, Harness, make_harness, request_json
from twin.channel.base import (
    TEST_PREFIX,
    BypassGrant,
    BypassRefused,
    BypassRequest,
    OutboundKind,
)
from twin.channel.ilink.channel import IlinkChannel
from twin.channel.probe.model import (
    Action,
    ActionKind,
    ActionStatus,
    Attempt,
    AttemptPhase,
    PlanStatus,
    ProbeOptions,
    ProbePlan,
    StepId,
    StepStatus,
)
from twin.channel.probe.policy import ProbeSendPolicy
from twin.channel.probe.store import AUDIT_MAX, ProbeStore
from twin.storage.db import Database

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "twin"
SEND = f"{API}/ilink/bot/sendmessage"


@pytest.fixture
def store(db: Database, clock: ManualClock) -> ProbeStore:
    return ProbeStore(db, clock)


def request(
    text: str | None = f"{TEST_PREFIX} hello",
    *,
    kind: str = "text",
    clock: ManualClock | None = None,
) -> BypassRequest:
    now = (clock or ManualClock()).now_utc()
    return BypassRequest(kind, text, now, "quota_exhausted")  # type: ignore[arg-type]


def open_attempt(
    store: ProbeStore,
    phase: AttemptPhase = AttemptPhase.RUNNING,
    step: StepId = StepId.COUNT,
    *,
    options: ProbeOptions | None = None,
    actions: list[Action] | None = None,
) -> ProbePlan:
    store.create(options)

    def change(p: ProbePlan) -> None:
        state = p.step(step)
        for earlier in p.steps:
            if earlier.id is step:
                break
            earlier.status = StepStatus.DONE
        state.status = StepStatus.ACTIVE
        now = p.created_at
        state.attempts.append(
            Attempt(n=1, phase=phase, armed_at=now, baseline_inbound_at=None, actions=actions or [])
        )

    store.update(change)
    loaded = store.load()
    assert loaded is not None
    return loaded


# ------------------------------------------------------------- when it refuses


def test_nothing_is_authorised_while_no_plan_exists(store: ProbeStore) -> None:
    policy = ProbeSendPolicy(store)
    with pytest.raises(BypassRefused, match="probe_not_active"):
        policy.authorize(request())
    [entry] = store.audit()
    assert entry["decision"] == "refused" and entry["why"] == "probe_not_active"
    assert entry["run_id"] is None and entry["gate_reason"] == "quota_exhausted"


def test_nothing_is_authorised_between_attempts(store: ProbeStore) -> None:
    store.create()  # a plan, but no attempt has begun
    with pytest.raises(BypassRefused, match="no_attempt_in_progress"):
        ProbeSendPolicy(store).authorize(request())
    assert store.audit()[-1]["why"] == "no_attempt_in_progress"


def test_an_attempt_that_is_over_authorises_nothing(store: ProbeStore) -> None:
    open_attempt(store, AttemptPhase.FINISHED)
    with pytest.raises(BypassRefused, match="no_attempt_in_progress"):
        ProbeSendPolicy(store).authorize(request())


@pytest.mark.parametrize("text", ["hello", "你好", " [测试] x", "测试 [测试]", "", "[test] hi"])
def test_a_text_without_the_test_prefix_is_refused_and_not_quoted_in_the_audit(
    store: ProbeStore, text: str
) -> None:
    open_attempt(store)
    with pytest.raises(BypassRefused, match="missing_test_prefix"):
        ProbeSendPolicy(store).authorize(request(text))
    entry = store.audit()[-1]
    assert entry["decision"] == "refused" and entry["text_chars"] == len(text)
    assert entry["text_head"] is None  # could be chat content: never stored


def test_a_text_that_is_missing_refuses(store: ProbeStore) -> None:
    open_attempt(store)
    with pytest.raises(BypassRefused, match="missing_test_prefix"):
        ProbeSendPolicy(store).authorize(request(None))


def test_a_picture_that_comes_with_text_is_refused(store: ProbeStore) -> None:
    open_attempt(store)
    with pytest.raises(BypassRefused, match="text_with_picture"):
        ProbeSendPolicy(store).authorize(request(f"{TEST_PREFIX} x", kind="image"))


def test_an_unknown_kind_of_send_is_refused(store: ProbeStore) -> None:
    open_attempt(store)
    with pytest.raises(BypassRefused, match="unknown_kind"):
        ProbeSendPolicy(store).authorize(request(None, kind="voice"))


@pytest.mark.parametrize("status", [PlanStatus.STOPPED, PlanStatus.COMPLETED])
def test_ending_the_plan_withdraws_the_permission_at_once(
    store: ProbeStore, status: PlanStatus
) -> None:
    open_attempt(store)
    policy = ProbeSendPolicy(store)
    assert policy.authorize(request()) is None
    store.finish(status, "done")
    with pytest.raises(BypassRefused, match="probe_not_active"):
        policy.authorize(request())


def test_an_unreadable_plan_is_never_a_permission(store: ProbeStore) -> None:
    open_attempt(store)
    store.state.put("probe.plan", {"garbage": True})
    with pytest.raises(BypassRefused, match="plan_unreadable"):
        ProbeSendPolicy(store).authorize(request())
    assert store.audit()[-1]["why"] == "plan_unreadable"


# ------------------------------------------------------------- when it allows


@pytest.mark.parametrize("phase", [AttemptPhase.ANNOUNCE, AttemptPhase.RUNNING])
def test_a_test_text_is_allowed_while_an_attempt_announces_or_measures(
    store: ProbeStore, phase: AttemptPhase
) -> None:
    plan = open_attempt(store, phase)
    assert ProbeSendPolicy(store).authorize(request()) is None
    [entry] = store.audit()
    assert entry["decision"] == "allowed" and entry["why"] == "ok"
    assert entry["run_id"] == plan.run_id and entry["step"] == "count"
    assert entry["text_head"] == f"{TEST_PREFIX} hello" and entry["empty_token"] is False


def test_a_picture_without_text_is_allowed_and_audited(store: ProbeStore) -> None:
    open_attempt(store, step=StepId.MEDIA)
    assert ProbeSendPolicy(store).authorize(request(None, kind="image")) is None
    entry = store.audit()[-1]
    assert entry["kind"] == "image" and entry["decision"] == "allowed"
    assert entry["text_chars"] is None


def test_every_authorisation_writes_an_audit_record_with_the_reason_it_was_needed(
    store: ProbeStore,
) -> None:
    open_attempt(store)
    policy = ProbeSendPolicy(store)
    for _ in range(3):
        policy.authorize(request())
    assert [(e["decision"], e["gate_reason"]) for e in store.audit()] == [
        ("allowed", "quota_exhausted")
    ] * 3


def test_the_audit_trail_is_capped(store: ProbeStore) -> None:
    open_attempt(store)
    policy = ProbeSendPolicy(store)
    for _ in range(AUDIT_MAX + 5):
        policy.authorize(request())
    assert len(store.audit()) == AUDIT_MAX


# ------------------------------------------------------- the empty-token grant


def experiment_action(status: ActionStatus) -> Action:
    return Action(
        id="send:1",
        kind=ActionKind.SEND_TEXT,
        label="empty",
        text=f"{TEST_PREFIX} x",
        empty_token=True,
        status=status,
    )


def test_the_empty_token_grant_exists_only_for_the_experiments_own_send(store: ProbeStore) -> None:
    options = ProbeOptions(empty_token_experiment=True)
    open_attempt(
        store,
        step=StepId.EMPTY_TOKEN,
        options=options,
        actions=[experiment_action(ActionStatus.ACTIVE)],
    )
    grant = ProbeSendPolicy(store).authorize(request())
    assert grant == BypassGrant(empty_context_token=True)
    assert store.audit()[-1]["empty_token"] is True


def test_the_grant_is_not_given_before_the_send_starts_or_in_other_steps(store: ProbeStore) -> None:
    options = ProbeOptions(empty_token_experiment=True)
    open_attempt(
        store,
        step=StepId.EMPTY_TOKEN,
        options=options,
        actions=[experiment_action(ActionStatus.PENDING)],
    )
    assert ProbeSendPolicy(store).authorize(request()) is None  # the announcement is a normal send


def test_a_normal_step_never_gets_an_empty_token(db: Database, clock: ManualClock) -> None:
    store = ProbeStore(db, clock)
    open_attempt(store, actions=[experiment_action(ActionStatus.ACTIVE)])
    assert ProbeSendPolicy(store).authorize(request()) is None


# ---------------------------------------- the engine and proactive paths have none


def test_only_the_probe_package_ever_names_the_policy() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(SRC.rglob("*.py"))
        if "ProbeSendPolicy" in path.read_text(encoding="utf-8")
        and SRC / "channel" / "probe" not in path.parents
    ]
    assert offenders == []


def test_no_module_outside_the_channel_package_passes_a_bypass() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if SRC / "channel" in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "bypass":
                offenders.append(f"{path.relative_to(ROOT)}:{node.value.lineno}")
    assert offenders == []


def test_within_the_channel_package_only_the_probe_and_the_channels_use_bypass() -> None:
    allowed = {
        SRC / "channel" / "ilink" / "channel.py",
        SRC / "channel" / "ilink" / "outbound.py",
        SRC / "channel" / "local.py",
        SRC / "channel" / "base.py",
    }
    users = {
        path
        for path in sorted((SRC / "channel").rglob("*.py"))
        if "bypass" in path.read_text(encoding="utf-8")
        and SRC / "channel" / "probe" not in path.parents
    }
    assert users <= allowed, sorted(str(p) for p in users - allowed)


def test_the_normal_send_methods_default_to_no_bypass() -> None:
    from twin.channel.local import LocalConsoleChannel

    for owner in (IlinkChannel, LocalConsoleChannel):
        for name in ("send_text", "send_image"):
            parameter = inspect.signature(getattr(owner, name)).parameters["bypass"]
            assert parameter.default is None, f"{owner.__name__}.{name}"


# --------------------------------------------- with the real channel and sender


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def h(db: Database, clock: ManualClock, tmp_path: Path) -> AsyncIterator[Harness]:
    harness = make_harness(db, clock, tmp_path, RecordingAlerts(), window_h=22, quota=2)
    harness.login()
    harness.bind()
    yield harness
    await harness.channel.stop()


async def test_the_real_channel_sends_past_the_safe_thresholds_only_with_the_policy(
    api: respx.MockRouter, h: Harness, db: Database, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    store = ProbeStore(db, clock)
    open_attempt(store)
    policy = ProbeSendPolicy(store)
    assert (await h.channel.send_text(f"{TEST_PREFIX} 1")).ok
    assert (await h.channel.send_text(f"{TEST_PREFIX} 2")).ok
    refused = await h.channel.send_text(f"{TEST_PREFIX} 3")  # past the safe count: refused
    assert refused.kind is OutboundKind.WINDOW_REJECTED and refused.reason == "quota_exhausted"
    assert route.call_count == 2
    for number in range(3, 6):  # the probe goes past the count with its policy
        assert (await h.channel.send_text(f"{TEST_PREFIX} {number}", bypass=policy)).ok
    clock.tick(30 * 3600)  # and past the safe window
    assert (await h.channel.send_text(f"{TEST_PREFIX} late", bypass=policy)).ok
    assert route.call_count == 6
    assert request_json(route.calls.last.request)["msg"]["context_token"] == CTX
    assert [e["decision"] for e in store.audit()] == ["allowed"] * 4


async def test_a_text_without_the_prefix_never_leaves_the_machine_even_with_the_policy(
    api: respx.MockRouter, h: Harness, db: Database, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    store = ProbeStore(db, clock)
    open_attempt(store)
    with pytest.raises(BypassRefused):
        await h.channel.send_text("hello there", bypass=ProbeSendPolicy(store))
    assert route.call_count == 0
    assert h.channel.session_state().outbound_since_inbound == 0


async def test_the_policy_cannot_reopen_a_window_the_platform_closed(
    api: respx.MockRouter, h: Harness, db: Database, clock: ManualClock
) -> None:
    route = api.post(SEND).mock(
        side_effect=[
            httpx.Response(200, json={"ret": -2, "errmsg": "prepare failed"}),
            httpx.Response(200, json={}),
        ]
    )
    store = ProbeStore(db, clock)
    open_attempt(store)
    policy = ProbeSendPolicy(store)
    first = await h.channel.send_text(f"{TEST_PREFIX} a", bypass=policy)
    assert first.session_expired and first.code == -2 and first.ret == -2
    assert first.errmsg == "prepare failed"
    second = await h.channel.send_text(f"{TEST_PREFIX} b", bypass=policy)
    assert second.session_expired and second.reason == "session_expired"
    assert route.call_count == 1  # the second one was refused locally, not retried
    clock.tick(timedelta(seconds=1).total_seconds())


async def test_the_failure_numbers_are_all_kept_on_the_result(
    api: respx.MockRouter, h: Harness
) -> None:
    api.post(SEND).respond(200, json={"ret": -7, "errcode": -9, "errmsg": "busy"})
    result = await h.channel.send_text(f"{TEST_PREFIX} x")
    assert result.kind is OutboundKind.REJECTED
    assert (result.ret, result.errcode, result.code) == (-7, -9, -9)  # errcode wins over ret
    assert result.errmsg == "busy"


async def test_an_empty_context_token_is_sent_only_with_the_grant(
    api: respx.MockRouter, h: Harness, db: Database, clock: ManualClock
) -> None:
    route = api.post(SEND).respond(200, json={})
    store = ProbeStore(db, clock)
    open_attempt(
        store,
        step=StepId.EMPTY_TOKEN,
        options=ProbeOptions(empty_token_experiment=True),
        actions=[experiment_action(ActionStatus.ACTIVE)],
    )
    assert (await h.channel.send_text(f"{TEST_PREFIX} t", bypass=ProbeSendPolicy(store))).ok
    assert request_json(route.calls.last.request)["msg"]["context_token"] == ""
    assert (await h.channel.send_text(f"{TEST_PREFIX} u")).ok
    assert request_json(route.calls.last.request)["msg"]["context_token"] == CTX
