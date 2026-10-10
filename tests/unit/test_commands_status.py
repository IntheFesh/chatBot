"""``/状态``: every item, and what it says when its source does not exist yet (R-CMD-002).

The report never invents a number: an item whose source is not there says so (``暂无``, or that
the proactive messages are not enabled), and one that fails costs only its own line.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.style_models import ScriptedStyleClient, register_model
from twin.channel.base import AuthState, SessionState
from twin.commands import texts
from twin.commands.router import CommandRouter
from twin.commands.status import ProactiveStatus, StatusReport, StatusSources, format_span
from twin.config.runtime import (
    BACKEND_ACTIVE,
    BACKEND_FALLBACK,
    SHOW_THINKING,
    THINKING_CHAT,
)
from twin.engine.backend_select import BackendSelector
from twin.engine.command_port import CommandContext
from twin.engine.style_models import StyleModels
from twin.llm.ledger import LedgerRecord
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import CostBreakdown, Usage
from twin.schedule.time_service import BotTimeService
from twin.services import Services

PREFIX = "⚙️ "


@pytest.fixture
async def llm(services: Services) -> AsyncIterator[LlmRuntime]:
    built = build_llm_runtime(services)
    yield built
    await built.client.aclose()


class Rig:
    def __init__(self, services: Services, llm: LlmRuntime, clock: ManualClock) -> None:
        services.runtime.initialize()
        self.services, self.llm, self.clock = services, llm, clock
        self.client = ScriptedStyleClient()
        self.selector = BackendSelector(
            runtime=services.runtime,
            models=StyleModels(services.db, mode=services.settings.style_model.mode),
            client=self.client,
            config=services.settings.backend,
            clock=clock,
            alerts=services.alerts,
            limits=llm.budget.limits,
        )
        llm.budget.set_style_status(self.selector)  # what the application does at start-up

    def sources(self, **changes: object) -> StatusSources:
        values: dict[str, object] = {
            "runtime": self.services.runtime,
            "settings": self.services.settings,
            "time": self.llm.time_service,
            "selector": self.selector,
            "db": self.services.db,
            "clock": self.clock,
        }
        values.update(changes)
        return StatusSources(**values)  # type: ignore[arg-type]

    async def report(self, **changes: object) -> str:
        return await StatusReport(self.sources(**changes)).render()

    def spend(self, usd: float, *, hit: int = 0, miss: int = 1) -> None:
        self.llm.ledger.record(
            LedgerRecord(
                "deepseek",
                "deepseek-flash",
                "reply",
                Usage(
                    prompt_tokens=hit + miss,
                    completion_tokens=1,
                    cache_hit_tokens=hit,
                    cache_miss_tokens=miss,
                ),
                CostBreakdown(usd, 0.0, 0.0, True, 1.0),
                False,
                1,
                self.clock.now_utc(),
            )
        )
        self.llm.budget.status(force=True)


@pytest.fixture
def rig(services: Services, llm: LlmRuntime, clock: ManualClock) -> Rig:
    return Rig(services, llm, clock)


def lines_of(report: str) -> dict[str, str]:
    """The report by the label in front of the first colon (the alert list is under its label)."""
    found: dict[str, str] = {}
    for line in report.split("\n"):
        label, _, rest = line.partition("：")
        found.setdefault(label, rest)
    return found


async def test_a_fresh_installation_reports_what_it_has_and_says_so_for_the_rest(rig: Rig) -> None:
    report = await rig.report(ledger=rig.llm.ledger, budget=rig.llm.budget)
    found = lines_of(report)
    assert report.startswith(texts.STATUS_HEADER)
    assert found["时区"].startswith("America/Chicago，当地时间 2026-10-09 周五 07:00")
    assert found["她此刻"].count("，到 ") == 1 and "还有约" in found["她此刻"]
    assert found["后端"] == "deepseek"
    assert found["思考模式"] == "关；显示思考：关"
    # this process has no scheduler: only `twin run` writes first
    assert found["主动消息"] == "没有在运行（只有 twin run 会主动发消息）"
    assert found["平台窗口"] == "暂无"
    assert found["今日费用"] == "$0.0000 / $1.00，今天还没有调用"
    assert found["预算级别"] == "0（正常）"
    assert found["风格模型"] == "还没有登记"
    assert found["最近告警"] == "没有"
    assert found["重训提醒"] == "暂无"
    assert "提醒：有" not in report  # nothing is left on a rented machine


async def test_without_a_day_plan_her_state_is_not_invented(rig: Rig) -> None:
    unplanned = BotTimeService(rig.clock, lambda: "America/Chicago")  # no plans attached
    found = lines_of(await rig.report(time=unplanned))
    assert found["她此刻"] == "暂无（今天还没有日程）"
    assert found["时区"].startswith("America/Chicago，当地时间 ")


async def test_the_thinking_mode_and_the_shown_thinking_follow_the_settings(rig: Rig) -> None:
    rig.services.runtime.set(THINKING_CHAT, "auto")
    rig.services.runtime.set(SHOW_THINKING, True)
    assert lines_of(await rig.report())["思考模式"] == "自动；显示思考：开"


async def test_the_platform_window_says_how_long_and_how_many(rig: Rig) -> None:
    state = SessionState(
        auth=AuthState.OK,
        bound=True,
        last_inbound_at=rig.clock.now_utc(),
        outbound_since_inbound=2,
        expired=False,
        remaining_quota=6,
        window_remaining=timedelta(hours=21, minutes=12),
        has_context_token=True,
    )
    window = lines_of(await rig.report(session_state=lambda: state))["平台窗口"]
    assert window == "剩余 21 小时 12 分，还能发 6 条；可主动 6 条"
    expired = SessionState(AuthState.OK, True, None, 0, True, 0, None, False)
    assert lines_of(await rig.report(session_state=lambda: expired))["平台窗口"].startswith(
        "已过期"
    )
    waiting = SessionState(AuthState.OK, True, None, 0, False, 8, None, False)
    assert "暂无" in lines_of(await rig.report(session_state=lambda: waiting))["平台窗口"]
    assert lines_of(await rig.report(session_state=lambda: None))["平台窗口"] == "暂无"


async def test_the_proactive_range_and_the_count_of_today_come_from_the_scheduler(rig: Rig) -> None:
    status = ProactiveStatus(low=1, high=6, enabled=True, sent_today=2)
    found = lines_of(await rig.report(proactive=lambda: status))
    assert found["主动消息"] == "每天 1-6 条（开），今天已发 2 条"
    off = ProactiveStatus(0, 0, False, 0)
    assert "（关）" in lines_of(await rig.report(proactive=lambda: off))["主动消息"]
    assert "没有在运行" in lines_of(await rig.report(proactive=lambda: None))["主动消息"]


async def test_the_cost_of_today_comes_with_the_cache_hit_rate_and_the_budget_level(
    rig: Rig,
) -> None:
    rig.services.settings.budget.daily_usd = 1.0
    rig.spend(0.25, hit=300, miss=100)
    found = lines_of(await rig.report(ledger=rig.llm.ledger, budget=rig.llm.budget))
    assert found["今日费用"] == "$0.2500 / $1.00，缓存命中率 75%"
    assert found["预算级别"] == "0（正常）"
    rig.spend(1.0)  # 125% of the day: level 2
    found = lines_of(await rig.report(ledger=rig.llm.ledger, budget=rig.llm.budget))
    assert found["预算级别"] == "2（例子减到 3 条、记忆预算减半）"
    only_ledger = lines_of(await rig.report(ledger=rig.llm.ledger))
    assert only_ledger["今日费用"].startswith("$1.2500 / $1.00")
    only_budget = lines_of(await rig.report(budget=rig.llm.budget))
    assert only_budget["今日费用"] == "$1.2500 / $1.00" and "暂无" not in only_budget["预算级别"]


async def test_the_last_three_alerts_are_listed_newest_first_by_their_titles(rig: Rig) -> None:
    for number in range(1, 5):
        rig.services.alerts.raise_alert("test", f"告警标题{number}")
        rig.clock.tick(60)
    report = await rig.report()
    shown = report.split(texts.STATUS_ALERTS_HEADER + "\n", 1)[1].split("\n")[:3]
    assert [line.split(" ", 3)[3] for line in shown] == ["告警标题4", "告警标题3", "告警标题2"]
    assert "告警标题1" not in report
    assert shown[0].startswith("- 10-09 07:03 ")  # on the clock of the bot's time zone


async def test_the_style_model_is_described_by_its_state(rig: Rig) -> None:
    async def style_line() -> str:
        rig.clock.tick(31)  # the server is looked at every 30 seconds
        return lines_of(await rig.report())["风格模型"]

    register_model(rig.services, run_id="r1", active=False, gate_passed=None)
    assert await style_line() == "已登记 1 个文件，还没有启用"
    register_model(rig.services, run_id="r2", gate_passed=True)
    assert await style_line() == "r2 Q5_K_M，已通过上线门槛，运行正常"
    rig.client.healthy, rig.client.detail = False, "model is still loading"
    rig.clock.tick(31)
    assert await style_line() == "r2 Q5_K_M，已通过上线门槛，不可用（model is still loading）"
    register_model(rig.services, run_id="r3", gate_passed=False)
    rig.client.healthy = True
    rig.clock.tick(31)
    assert await style_line() == "r3 Q5_K_M，未通过门槛（强制启用），运行正常"


async def test_a_style_model_on_a_rented_machine_comes_with_the_reminder_about_the_bill(
    rig: Rig,
) -> None:
    rig.services.settings.style_model.mode = "vllm_completion"
    rig.selector = BackendSelector(
        runtime=rig.services.runtime,
        models=StyleModels(rig.services.db, mode="vllm_completion"),
        client=rig.client,
        config=rig.services.settings.backend,
        clock=rig.clock,
        alerts=rig.services.alerts,
    )
    assert texts.STATUS_REMOTE_HOURLY not in await rig.report()  # no model yet
    register_model(rig.services, quant="lora", kind="adapter")
    rig.clock.tick(31)
    assert texts.STATUS_REMOTE_HOURLY in await rig.report()


async def test_a_fallback_and_a_budget_takeover_are_visible_in_the_backend_line(rig: Rig) -> None:
    register_model(rig.services, gate_passed=True)
    rig.services.runtime.set(BACKEND_ACTIVE, "hybrid")
    rig.client.healthy = False
    await rig.selector.choose()
    found = lines_of(await rig.report())
    assert found["后端"] == "hybrid（风格模型不可用，眼下由 deepseek 回复：unhealthy）"
    rig.services.runtime.set(BACKEND_FALLBACK, None)
    rig.services.runtime.set(BACKEND_ACTIVE, "deepseek")
    rig.client.healthy = True
    rig.clock.tick(31)
    await rig.selector.probe()
    rig.services.settings.budget.daily_usd = 0.01
    rig.spend(0.05)  # five times the day: the last level
    budgeted = lines_of(await rig.report(budget=rig.llm.budget))
    assert budgeted["后端"] == "deepseek（预算已到最后一级，眼下由风格模型回复）"


async def test_training_data_left_on_a_rented_machine_and_the_retraining_reminder(rig: Rig) -> None:
    reminder = "真实消息比上次训练时增加了 12%"
    report = await rig.report(uncleaned=lambda: 2, retrain=lambda: reminder)
    found = lines_of(report)
    assert found["重训提醒"] == reminder
    assert texts.STATUS_UNCLEANED.format(count=2) in report
    assert "twin train remote cleanup" in report


async def test_the_retraining_reminder_is_read_from_the_training_records(
    rig: Rig, services: Services
) -> None:
    """R-TRN-012: without a source passed in, ``/状态`` compares her messages with the training."""
    from tests.support.synth_chat import MessageWriter
    from tests.support.training_history import record_training

    writer = MessageWriter(services)
    for number in range(100):
        writer.add(
            rig.clock.now_utc() - timedelta(minutes=200 - number), True, "text", f"她的第{number}句"
        )
    writer.store()
    assert lines_of(await rig.report())["重训提醒"] == texts.STATUS_NONE  # nothing trained yet
    record_training(services, covered=95)
    assert lines_of(await rig.report())["重训提醒"] == texts.STATUS_NONE  # 5 %: not yet
    record_training(services, "r2", "ds-2", covered=80)
    reminder = lines_of(await rig.report())["重训提醒"]
    assert "25%" in reminder and "建议重新训练风格模型" in reminder


async def test_what_later_rounds_add_appears_at_the_end_and_one_broken_source_costs_one_line(
    rig: Rig,
) -> None:
    def broken() -> SessionState | None:
        raise RuntimeError("a secret 暗号甲")

    def failing_extra() -> str | None:
        raise RuntimeError("也坏了")

    report = await rig.report(
        session_state=broken,
        extra=(lambda: "建议的静默窗口：18 秒", lambda: None, failing_extra),
    )
    found = lines_of(report)
    assert found["平台窗口"] == "暂无" and found["后端"] == "deepseek"  # the rest is still there
    assert report.split("\n")[-1] == "建议的静默窗口：18 秒"
    assert "暗号甲" not in report and "也坏了" not in report


async def test_an_extra_line_may_be_a_coroutine_and_a_broken_one_costs_only_itself(
    rig: Rig,
) -> None:
    async def asked() -> str | None:
        return "等你说完：15 秒；按你的连发习惯建议 33 秒（自适应：关）"

    async def broken() -> str | None:
        raise RuntimeError("协程里坏了")

    async def nothing() -> str | None:
        return None

    report = await rig.report(extra=(asked, broken, nothing))
    assert report.split("\n")[-1] == "等你说完：15 秒；按你的连发习惯建议 33 秒（自适应：关）"
    assert "协程里坏了" not in report and lines_of(report)["后端"] == "deepseek"


async def test_the_command_sends_the_same_report(
    rig: Rig, services: Services, llm: LlmRuntime
) -> None:
    router = CommandRouter.from_services(services, llm, selector=rig.selector)
    outcome = await router.handle("/状态", CommandContext(rig.clock.now_utc(), "in-1"))
    assert outcome is not None and outcome.reply.startswith(PREFIX + texts.STATUS_HEADER)
    found = lines_of(outcome.reply)
    assert found["后端"] == "deepseek" and found["风格模型"] == "还没有登记"
    assert "今日费用" in found and found["平台窗口"] == "暂无"


def test_a_span_is_told_in_hours_and_minutes() -> None:
    assert format_span(timedelta(hours=2, minutes=5)) == "2 小时 5 分"
    assert format_span(timedelta(hours=3)) == "3 小时"
    assert format_span(timedelta(minutes=42, seconds=20)) == "42 分钟"
    assert format_span(timedelta(seconds=-5)) == "0 分钟"
