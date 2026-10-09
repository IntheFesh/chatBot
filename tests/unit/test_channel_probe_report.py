"""``docs/CHANNEL_REPORT.md``: a pending template until real data exists (R-CH-009, R-CH-010)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twin.channel.probe.report import (
    MEASURED_MARKER,
    PENDING_MARKER,
    render_pending_report,
    render_report,
    write_report,
)
from twin.channel.probe.summary import (
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_UNDETERMINED,
    ChannelProbeSummary,
)

ROOT = Path(__file__).resolve().parents[2]


def sample(**changes: Any) -> ChannelProbeSummary:
    steps: dict[str, dict[str, Any]] = {
        "count": {
            "status": "done",
            "attempts": 1,
            "voided_attempts": 0,
            "finished_at": "2026-10-09T12:40:00+00:00",
            "n": 8,
            "capped": False,
            "api_ok": 8,
            "phone_received": 8,
            "mismatch": False,
            "first_problem_index": 9,
            "first_failure": {
                "outcome": "window_rejected",
                "ret": -2,
                "errcode": None,
                "errmsg": "prepare failed",
            },
            "interval_s": 120.0,
        },
        "media": {
            "status": "done",
            "attempts": 1,
            "voided_attempts": 0,
            "finished_at": "2026-10-09T13:00:00+00:00",
            "images": {
                "jpg": {"api_ok": True, "phone": "arrived"},
                "png": {"api_ok": True, "phone": "arrived"},
                "gif": {"api_ok": True, "phone": "moving"},
            },
            "gif_animated": True,
            "typing": {"sent": True, "visible": "yes", "hold_s": 30.0},
            "quote": "not_supported",
        },
        "window": {
            "status": "done",
            "attempts": 1,
            "voided_attempts": 0,
            "finished_at": "2026-10-10T14:00:00+00:00",
            "points": [
                {
                    "hours_planned": 20.0,
                    "hours_actual": 20.0,
                    "api_ok": True,
                    "phone_delivered": True,
                },
                {
                    "hours_planned": 23.0,
                    "hours_actual": 23.0,
                    "api_ok": True,
                    "phone_delivered": True,
                },
                {
                    "hours_planned": 25.0,
                    "hours_actual": 25.0,
                    "api_ok": False,
                    "phone_delivered": False,
                },
            ],
            "dropped_hours": [1.0, 6.0, 12.0],
            "lower_bound_h": 23.0,
            "upper_bound_h": 25.0,
            "mismatch": False,
            "inbound_at": "2026-10-09T13:10:00+00:00",
        },
    }
    values: dict[str, Any] = {
        "run_id": "20261009T120000Z-ab12",
        "status": "completed",
        "started_at": "2026-10-09T12:00:00+00:00",
        "finished_at": "2026-10-10T14:00:00+00:00",
        "complete": True,
        "verdict": VERDICT_MET,
        "reasons": [],
        "n_messages": 8,
        "n_capped": False,
        "window_lower_bound_h": 23.0,
        "window_upper_bound_h": 25.0,
        "gif_animated": True,
        "typing_visible": True,
        "quote_supported": False,
        "quota_shared": True,
        "quota_basis": "one send call for replies and proactive messages",
        "suggestions": {"channel.proactive_window_safe_h": 20.7, "channel.outbound_quota_safe": 7},
        "steps": steps,
        "failures": [
            {
                "step": "count",
                "attempt": 1,
                "action": "count 9/15",
                "outcome": "window_rejected",
                "reason": "platform_rejected",
                "code": -2,
                "ret": -2,
                "errcode": None,
                "errmsg": "prepare failed",
                "http_status": None,
                "hours_after_inbound": 0.27,
                "at": "2026-10-09T12:31:00+00:00",
            }
        ],
        "notes": ["window points left out for lack of message budget: 1, 6, 12 h"],
    }
    values.update(changes)
    return ChannelProbeSummary(**values)


# ------------------------------------------------------------ the shipped file


def test_the_report_in_the_repository_is_the_pending_template() -> None:
    shipped = (ROOT / "docs" / "CHANNEL_REPORT.md").read_text(encoding="utf-8")
    assert shipped == render_pending_report()
    assert PENDING_MARKER in shipped and MEASURED_MARKER not in shipped
    assert "待实测" in shipped


def test_the_pending_template_states_no_measurement() -> None:
    text = render_pending_report()
    for claim in ("已实测", "实测结果如下", "测得", "手机上实际收到了", "达标：", "R-CH-010 判定"):
        assert claim not in text
    assert "没有任何" in text and "实测数字" in text
    assert "**未达标**" in text  # it explains the rule, it does not apply it
    assert "估算" not in text


def test_the_pending_template_explains_how_to_get_real_results() -> None:
    text = render_pending_report()
    for command in (
        "twin channel login",
        "twin channel probe start",
        "twin channel probe answer",
        "twin channel probe report",
    ):
        assert command in text
    assert "手机为准" in text and "合成图" in text


# ---------------------------------------------------------------- measured report


def test_a_measured_report_shows_every_result_with_its_time() -> None:
    text = render_report(sample())
    assert MEASURED_MARKER in text and PENDING_MARKER not in text
    assert "20261009T120000Z-ab12" in text and "2026-10-09 12:00:00 UTC" in text
    assert "R-CH-010 判定：**达标**" in text
    assert "| 手机上实际收到（N） | 8 |" in text and "| 与手机对账 | 一致 |" in text
    assert "GIF 在手机上会动 | 是" in text and "“对方正在输入”可见 | 看到" in text
    assert "引用 | 不支持" in text
    assert "窗口下限（最长一次送达） | 23.00 小时" in text
    assert "窗口上限（首次未送达） | 25.00 小时" in text
    assert "因条数不足而省略的测量点 | 1, 6, 12" in text
    assert "2026-10-09 13:10:00 UTC" in text  # the inbound time of the window step


def test_the_failure_table_has_every_number_the_server_returned() -> None:
    text = render_report(sample())
    row = next(line for line in text.splitlines() if "count 9/15" in line)
    for fragment in ("window_rejected:platform_rejected", "-2", "prepare failed", "0.27"):
        assert fragment in row
    assert "| 失败的 ret / errcode / errmsg | -2 / — / prepare failed |" in text


def test_a_report_without_failures_says_so() -> None:
    assert "没有失败的发送" in render_report(sample(failures=[]))


def test_suggestions_are_shown_as_suggestions_that_need_confirmation() -> None:
    text = render_report(sample())
    assert "| `channel.proactive_window_safe_h` | 20.7 |" in text
    assert "| `channel.outbound_quota_safe` | 7 |" in text
    assert "确认后才写入" in text


def test_not_met_says_so_and_recommends_stopping() -> None:
    text = render_report(
        sample(
            verdict=VERDICT_NOT_MET,
            n_messages=2,
            reasons=["only 2 message(s) reached the phone after one inbound message"],
        )
    )
    assert "R-CH-010 判定：**未达标**" in text
    assert "**建议停机**" in text and "企业微信通道不在本规格范围" in text
    assert "only 2 message(s)" in text


def test_undetermined_says_what_is_missing() -> None:
    text = render_report(
        sample(verdict=VERDICT_UNDETERMINED, reasons=["the window (step 3) was not measured"])
    )
    assert "无法判定" in text and "the window (step 3)" in text and "建议停机" not in text


def test_a_capped_count_is_labelled_as_a_lower_bound() -> None:
    summary = sample(n_messages=15, n_capped=True)
    summary.steps["count"].update(n=15, capped=True, api_ok=15, phone_received=15)
    text = render_report(summary)
    assert "达到上限，N 是下限" in text and "达到探针上限，实际可能更多" in text


def test_a_mismatch_between_server_and_phone_is_called_out() -> None:
    summary = sample()
    summary.steps["count"].update(api_ok=8, phone_received=5, n=5, mismatch=True)
    text = render_report(summary)
    assert "**不一致**" in text and "| 接口接受 | 8 |" in text


def test_a_skipped_step_is_reported_with_its_reason() -> None:
    summary = sample()
    summary.steps["media"] = {
        "status": "skipped",
        "skip_reason": "only 1 message(s) reached the phone",
        "attempts": 0,
        "voided_attempts": 0,
        "finished_at": None,
    }
    summary.steps["window"] = {
        "status": "skipped",
        "skip_reason": "only 1 message(s) reached the phone",
        "attempts": 0,
        "voided_attempts": 0,
    }
    text = render_report(summary)
    assert "已跳过：only 1 message(s) reached the phone" in text
    assert "状态：skipped：only 1 message(s)" in text


def test_an_unfinished_probe_is_labelled_as_such() -> None:
    text = render_report(sample(status="stopped", finished_at=None, complete=False))
    assert "已停止（未做完）" in text
    assert "结束于 —" in text


def test_the_optional_experiment_appears_only_when_it_ran() -> None:
    assert "空 context_token" not in render_report(sample())
    summary = sample()
    summary.steps["empty_token"] = {
        "status": "done",
        "attempts": 1,
        "voided_attempts": 0,
        "api_ok": True,
        "delivered": False,
        "failure": None,
    }
    text = render_report(summary)
    assert "可选实验：空 context_token" in text and "手机收到 | 否" in text


def test_an_experiment_that_did_not_finish_is_reported_as_such() -> None:
    summary = sample()
    summary.steps["empty_token"] = {"status": "pending", "attempts": 0, "voided_attempts": 0}
    assert "状态：pending" in render_report(summary)


def test_the_pipe_character_cannot_break_a_table_row() -> None:
    summary = sample()
    summary.failures[0]["errmsg"] = "a|b"
    assert "a/b" in render_report(summary)


def test_a_report_is_written_as_utf_8_with_unix_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "nested" / "CHANNEL_REPORT.md"
    write_report(target, render_report(sample()))
    raw = target.read_bytes()
    assert b"\r\n" not in raw and "达标".encode() in raw
