"""The test of readiness and what it says is in the way (R18-B8, D-870, D-873).

Ready = the live profile, the live routine model and a live persona card **with its description**
exist.  The explanation names the import (and its progress), the recomputation and the
description batch (with the ``twin jobs approve`` command) - numbers and commands, never a word of
a message.  ``/状态`` and ``twin health`` show the same ``未就绪：缺 …``.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.support.clock import ManualClock
from tests.support.readiness import (
    install_card,
    install_profile,
    install_ready,
    install_routine,
    record_import,
)
from tests.support.synth_chat import MessageWriter
from twin.engine.readiness import (
    IMPORT_DONE,
    IMPORT_NONE,
    PERSONA_AFTER_PROFILE,
    PERSONA_APPROVED,
    PERSONA_FAILED,
    PERSONA_MISSING,
    PERSONA_RUNNING,
    PROFILE_AFTER_IMPORT,
    PROFILE_FAILED,
    PROFILE_MISSING,
    PROFILE_QUEUED,
    ReadinessGate,
    check_readiness,
    explain,
    status_lines,
)
from twin.ops.health import HealthCollector, HealthLevel, check_ready
from twin.ops.jobs import JobQueue
from twin.profile.persona.jobs import PERSONA_JOB
from twin.profile.persona.refresh import refresh_stats
from twin.profile.queue import PROFILE_JOB
from twin.services import Services
from twin.storage.models import Job


def her_messages(services: Services, count: int) -> None:
    """``count`` synthetic messages of hers (enough to describe her from at 30)."""
    writer = MessageWriter(services)
    start = services.clock.now_utc() - timedelta(days=3)
    for number in range(count):
        writer.add(start + timedelta(minutes=7 * number), True, "text", f"合成的第{number}句")
    writer.store()


def set_status(services: Services, job_id: str, status: str) -> None:
    with services.db.transaction(bump_state=False) as session:
        row = session.get(Job, job_id)
        assert row is not None
        row.status = status


def persona_batch(services: Services, batch: str, *, usd: float = 0.15) -> list[str]:
    """The description batch of both scopes, waiting for approval (what the import hook queues)."""
    queue = JobQueue(services.db, services.clock)
    return [
        queue.enqueue(
            PERSONA_JOB,
            {"scope": scope, "batch_id": batch},
            batch_id=batch,
            estimated_cost_usd=usd,
            requires_approval=True,
            offpeak_only=True,
        )
        for scope in ("live", "pre_holdout")
    ]


def test_r18_b8_ready_means_profile_routine_and_a_described_card(services: Services) -> None:
    """D-870: each of the three is needed; the first card (statistics rules only) is not enough."""
    assert check_readiness(services).missing == ("profile", "routine", "persona")
    install_profile(services)
    assert check_readiness(services).missing == ("routine", "persona")
    install_routine(services)
    assert check_readiness(services).missing == ("persona",)
    assert refresh_stats(services, "live").status == "created"  # the free first card
    readiness = check_readiness(services)
    assert readiness.missing == ("persona",) and readiness.label == "人设卡"
    assert readiness.status_line() == "未就绪：缺 人设卡"
    install_card(services)
    assert check_readiness(services).ready
    assert check_readiness(services).status_line() == "已就绪"


def test_r18_b8_a_routine_model_without_a_sleep_window_counts_as_a_routine(
    services: Services,
) -> None:
    """D-870 with D-231: the model exists, it inferred no sleep - ready, and the plan warns."""
    install_ready(services)  # the neutral model of the support has no sleep window at all
    assert check_readiness(services).ready


def test_r18_b8_before_anything_was_imported_the_import_comes_first(services: Services) -> None:
    lines = explain(services, check_readiness(services))
    assert lines == [
        IMPORT_NONE,
        PROFILE_AFTER_IMPORT,
        "人设卡：她的消息只有 0 条，不够生成描述（至少 30 条），先导入更多聊天记录",
    ]


@pytest.mark.parametrize(
    ("status", "processed", "total", "expected"),
    [
        (
            "running",
            500,
            2000,
            "导入：进行中，已处理 500 / 2,000（25%）；`twin import status` 看进度",
        ),
        ("queued", 0, None, "导入：进行中，已处理 0 / ?；`twin import status` 看进度"),
        (
            "interrupted",
            120,
            2000,
            "导入：中断了（已处理 120 / 2,000），`twin import --resume` 接着导",
        ),
        ("failed", 7, None, "导入：失败了（已处理 7 / ?），`twin import --resume` 接着导"),
        ("done", 2000, 2000, IMPORT_DONE),
    ],
)
def test_r18_b8_the_progress_of_the_import_is_told(
    services: Services, status: str, processed: int, total: int | None, expected: str
) -> None:
    record_import(services, status=status, processed=processed, total=total)
    assert explain(services, check_readiness(services))[0] == expected


def test_r18_b8_the_recomputation_is_told_where_it_stands(services: Services) -> None:
    record_import(services, processed=100, total=100)
    readiness = check_readiness(services)
    assert explain(services, readiness)[1] == PROFILE_MISSING
    queue = JobQueue(services.db, services.clock)
    job = queue.enqueue(PROFILE_JOB, {"scope": "all"})
    assert explain(services, readiness)[1] == PROFILE_QUEUED
    set_status(services, job, "failed")
    assert explain(services, readiness)[1] == PROFILE_FAILED
    install_profile(services)
    install_routine(services)
    lines = explain(services, check_readiness(services))
    assert len(lines) == 2 and not any(line.startswith("画像与作息") for line in lines)


def test_r18_b8_the_description_batch_is_told_with_its_command_and_price(
    services: Services, clock: ManualClock
) -> None:
    """D-871: the batch that waits for ``twin jobs approve`` - the whole batch's estimate."""
    her_messages(services, 40)
    record_import(services, processed=40, total=40)
    install_profile(services)
    install_routine(services)
    readiness = check_readiness(services)
    assert explain(services, readiness)[-1] == PERSONA_MISSING
    live, _ = persona_batch(services, "persona-20261009-abcd")
    assert explain(services, readiness)[-1] == (
        "人设卡：描述生成等你批准，`twin jobs approve persona-20261009-abcd`（估算 $0.30）"
    )
    JobQueue(services.db, clock).approve_batch("persona-20261009-abcd", max_usd=1.0)
    assert explain(services, readiness)[-1] == PERSONA_APPROVED
    set_status(services, live, "running")
    assert explain(services, readiness)[-1] == PERSONA_RUNNING
    set_status(services, live, "failed")
    assert explain(services, readiness)[-1] == PERSONA_FAILED


def test_r18_b8_with_enough_messages_but_no_profile_the_card_comes_after_it(
    services: Services,
) -> None:
    her_messages(services, 35)
    assert explain(services, check_readiness(services))[-1] == PERSONA_AFTER_PROFILE


def test_r18_b8_the_notice_and_the_status_never_carry_a_message(services: Services) -> None:
    her_messages(services, 35)
    record_import(services, processed=35, total=35)
    gate = ReadinessGate(services)
    readiness = gate.check()
    text = gate.notice(readiness, services.clock.now_utc(), None)
    assert text is not None and text.startswith("还没准备好，先不回复：缺 画像、作息模型、人设卡")
    assert "合成的第" not in text
    lines = status_lines(services)
    assert lines[0] == "未就绪：缺 画像、作息模型、人设卡（先不回复、不主动）"
    assert lines[1:] == [f"- {line}" for line in explain(services, readiness)]
    assert not any("合成的第" in line for line in lines)
    gate.noted(readiness, services.clock.now_utc())
    assert gate.notice(readiness, services.clock.now_utc(), None) is None  # told: not again now
    install_ready(services)
    assert status_lines(services) == ["已就绪"]
    assert explain(services, check_readiness(services)) == []  # nothing stands in the way
    assert gate.notice(gate.check(), services.clock.now_utc(), None) is None  # ready: nothing


def test_r18_b8_a_broken_explanation_still_says_what_is_missing(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object) -> list[str]:
        raise RuntimeError("the job table is locked")

    monkeypatch.setattr("twin.engine.readiness.explain", broken)
    gate = ReadinessGate(services)
    text = gate.notice(gate.check(), services.clock.now_utc(), None)
    assert text is not None and "缺 画像、作息模型、人设卡" in text and "\n" not in text


def test_r18_b8_twin_health_warns_until_the_bot_is_ready(services: Services) -> None:
    """D-873: a warning (no alert category) with the same ``未就绪：缺 …``; ok once ready."""
    assert check_ready(True, "已就绪").level is HealthLevel.OK
    found = check_ready(False, "未就绪：缺 人设卡")
    assert (found.level, found.category) == (HealthLevel.WARN, None)
    checks, _, _ = HealthCollector(services).stored_checks()
    (readiness,) = [check for check in checks if check.name == "readiness"]
    assert readiness.level is HealthLevel.WARN
    assert readiness.detail == "未就绪：缺 画像、作息模型、人设卡"
    install_ready(services)
    checks, _, _ = HealthCollector(services).stored_checks()
    (readiness,) = [check for check in checks if check.name == "readiness"]
    assert (readiness.level, readiness.detail) == (HealthLevel.OK, "已就绪")
