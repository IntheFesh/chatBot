"""``docs/CHANNEL_REPORT.md``: the written result of the M0 channel probe (R-CH-009, R-CH-010).

Two renderings exist.  :func:`render_pending_report` is what the repository ships until the
probe has run on a real phone: it explains the measurements and the verdict rules and states
**no measured number** - none can exist without a WeChat account.  :func:`render_report` turns
a real :class:`~twin.channel.probe.summary.ChannelProbeSummary` into the final report: only
measurements and times, the failures with their numbers, the verdict on R-CH-010, and the
suggested configuration values (which change nothing until the user confirms them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twin.channel.probe.model import ProbeOptions
from twin.channel.probe.summary import (
    MARGIN,
    MIN_MESSAGES,
    MIN_WINDOW_H,
    VERDICT_MET,
    VERDICT_NOT_MET,
    ChannelProbeSummary,
)

PENDING_MARKER = "<!-- channel-report: pending -->"
MEASURED_MARKER = "<!-- channel-report: measured -->"

_defaults = ProbeOptions()
_HOURS = " / ".join(f"{hour:g}" for hour in _defaults.window_hours)


def render_pending_report() -> str:
    """The report as shipped before any real probe: explains, but states no measurement."""
    return f"""# 微信通道实测报告（M0 · ClawBot / iLink）

{PENDING_MARKER}

> 状态：**待实测**。
> 编写本轮代码的环境没有微信账号，`twin channel probe` 还没有在真机上运行过，所以本文件里没有任何
> 实测数字。探针跑完后运行 `twin channel probe report`，本文件会被实测结果整体覆盖。

## 怎样得到实测结果

```
uv run twin channel login            # 扫码并绑定
uv run twin run                      # 探针由运行中的应用执行，保持它开着
uv run twin channel probe start      # 约 26 小时；按终端和微信里的提示操作
uv run twin channel probe answer     # 回答探针的问题（手机上实际收到几条等）
uv run twin channel probe report     # 生成本文件
```

探针在你自己的 ClawBot 会话里发消息，所有消息以“[测试]”开头，图片是程序画出来的合成图
（不含任何真实照片）。计划保存在数据库里，应用重启后继续。

## 四项测量与顺序

每一步开始前，终端与微信都会提示你先给机器人发一条新消息，收到后才开始，使窗口与计数复位。
每一步结束后探针会问你手机上实际收到几条，**以手机为准**，不以接口返回的成功为准。

| # | 测量 | 做法 |
| --- | --- | --- |
| 1 | 条数 | 收到你的新消息后，每 {_defaults.interval_s / 60:g} 分钟主动发一条，直到首次失败或 {_defaults.max_messages} 条；手机上实际收到的条数记为 N |
| 2 | 媒体与交互 | 合成的 JPG、PNG、多帧 GIF 各一张（询问 GIF 是否在动）；发送“正在输入”并询问是否可见；引用：协议源码不支持出站引用，跳过（D-009）。本步总条数不超过 N-1，不够时拆成几次，每次之前都要你先发一条新消息 |
| 3 | 窗口 | 收到你的新消息后约 {max(_defaults.window_hours):g} 小时内你不要给机器人发任何消息（发了则本步作废重做）；探针在 {_HOURS} 小时各发一条文字（总条数不超过 N-1，不够时从后往前保留测量点并在报告中写明） |
| 4 | 配额是否共用 | 回复与主动出站是否消耗同一计数：依据协议源码（两者是同一个发送接口、同一个 `context_token`）与第 1 项的结果在报告中写明 |

可选实验（默认关，`--empty-token-experiment`）：不带 `context_token` 的文字能否送达。

遇到协议失败（包括会话过期）立即停止当前一步，不重试；每次失败的 `ret`、`errcode`、`errmsg`
（脱敏）都会记录在本文件里，用来区分社区上互相矛盾的几种解释（窗口过期、速率限制、每次回复条数上限）。

## 判定口径（R-CH-010）

- 窗口短于 {MIN_WINDOW_H:g} 小时，或一次入站后手机上实际收到的连发条数少于 {MIN_MESSAGES} 条：报告写“**未达标**”，
  并建议停机：主动消息（第 10 轮）依赖这两个条件，请把报告交给维护者决定；企业微信通道不在本规格范围，需要你确认后另开一轮。
- 两项都测到且都满足：报告写“达标”。
- 其余情况写“无法判定”，并说明缺什么。

## 建议的配置值

报告会给出 `channel.proactive_window_safe_h` 与 `channel.outbound_quota_safe` 的建议值：
实测值留 {round((1 - MARGIN) * 100)}% 余量（`twin channel probe report` 会先问你，确认后才写入 `config/config.yaml`）。
在拿到实测数据之前，`config/config.example.yaml` 里的默认值只是保守猜测，不是协议事实。

## 已知线索（来自协议源码，非实测）

协议源码里没有任何发送计数、窗口计时或配额检查；社区对 `ret=-2` 的解释互相矛盾，见
`docs/ILINK_PROTOCOL.md` 第 9 节。窗口长度、可连发条数、回复与主动是否共用配额、GIF 是否会动、
“正在输入”是否可见，全部以本探针的实测为准。
"""


# ------------------------------------------------------------------ formatting


def _hours(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f} 小时"


def _yes_no(value: Any) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "未确定"


def _time(value: Any) -> str:
    return "—" if not value else str(value).replace("T", " ")[:19] + " UTC"


def _cell(value: Any) -> str:
    return "—" if value in (None, "") else str(value).replace("|", "/")


def _verdict_block(summary: ChannelProbeSummary) -> str:
    reasons = "\n".join(f"- {reason}" for reason in summary.reasons)
    if summary.verdict == VERDICT_MET:
        return (
            "R-CH-010 判定：**达标**\n\n"
            f"一次入站后手机上实际收到 {summary.n_messages} 条"
            f"{'（达到探针上限，实际可能更多）' if summary.n_capped else ''}，"
            f"窗口至少 {_hours(summary.window_lower_bound_h)}，"
            f"满足窗口 ≥ {MIN_WINDOW_H:g} 小时且连发 ≥ {MIN_MESSAGES} 条的要求。"
        )
    if summary.verdict == VERDICT_NOT_MET:
        return (
            "R-CH-010 判定：**未达标**\n\n"
            f"{reasons}\n\n"
            "**建议停机**：主动消息（第 10 轮）依赖“窗口 ≥ "
            f"{MIN_WINDOW_H:g} 小时”和“一次入站后连发 ≥ {MIN_MESSAGES} 条”。请把本报告交给维护者，"
            "由用户决定下一步；企业微信通道不在本规格范围，需要用户确认后另开一轮。"
        )
    return f"R-CH-010 判定：**无法判定**\n\n{reasons}"


def _count_section(step: dict[str, Any], summary: ChannelProbeSummary) -> str:
    if step.get("status") != "done":
        return f"状态：{step.get('status', '未运行')}\n"
    first = step.get("first_failure") or {}
    lines = [
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 手机上实际收到（N） | {step['n']}{'（达到上限，N 是下限）' if summary.n_capped else ''} |",
        f"| 接口接受 | {step['api_ok']} |",
        f"| 与手机对账 | {'**不一致**' if step['mismatch'] else '一致'} |",
        f"| 首个有问题的序号 | {_cell(step.get('first_problem_index'))} |",
        f"| 首次失败 | {_cell(first.get('outcome'))} |",
        f"| 失败的 ret / errcode / errmsg | {_cell(first.get('ret'))} / "
        f"{_cell(first.get('errcode'))} / {_cell(first.get('errmsg'))} |",
        f"| 间隔 | {step['interval_s']:g} 秒 |",
    ]
    return "\n".join(lines) + "\n"


def _media_section(step: dict[str, Any], summary: ChannelProbeSummary) -> str:
    if step.get("status") == "skipped":
        return f"已跳过：{_cell(step.get('skip_reason'))}\n"
    images = step.get("images") or {}
    names = {"jpg": "JPEG", "png": "PNG", "gif": "GIF"}
    lines = ["| 图片 | 接口接受 | 手机上 |", "| --- | --- | --- |"]
    shown = {
        "arrived": "收到",
        "moving": "收到，会动",
        "still": "收到，不会动",
        "missing": "没收到",
        "not_delivered": "接口拒绝",
        "not_tested": "未测",
        "unanswered": "未回答",
    }
    for key in ("jpg", "png", "gif"):
        row = images.get(key)
        if row is None:
            lines.append(f"| {names[key]} | 未测 | 未测 |")
        else:
            phone = shown.get(str(row.get("phone")), _cell(row.get("phone")))
            lines.append(f"| {names[key]} | {_yes_no(row.get('api_ok'))} | {phone} |")
    typing = step.get("typing") or {}
    visible = {"yes": "看到", "no": "没看到", "unsure": "不确定"}.get(
        str(typing.get("visible")), "未测"
    )
    lines += [
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
        f"| GIF 在手机上会动 | {_yes_no(summary.gif_animated)} |",
        f"| “对方正在输入”可见 | {visible} |",
        "| 引用 | 不支持：协议源码没有出站引用（D-009），未测试 |",
    ]
    return "\n".join(lines) + "\n"


def _window_section(step: dict[str, Any]) -> str:
    if step.get("status") != "done":
        reason = f"：{step['skip_reason']}" if step.get("skip_reason") else ""
        return f"状态：{step.get('status', '未运行')}{reason}\n"
    lines = [
        "| 计划（小时） | 实际（入站后小时） | 接口 | 手机 |",
        "| --- | --- | --- | --- |",
    ]
    for point in step.get("points", []):
        api = {True: "接受", False: "失败", None: "未发"}[point.get("api_ok")]
        phone = {True: "收到", False: "没收到", None: "—"}[point.get("phone_delivered")]
        actual = point.get("hours_actual")
        lines.append(
            f"| {float(point['hours_planned']):g} | {'—' if actual is None else f'{actual:.2f}'}"
            f" | {api} | {phone} |"
        )
    dropped = step.get("dropped_hours") or []
    lines += [
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 窗口下限（最长一次送达） | {_hours(step.get('lower_bound_h'))} |",
        f"| 窗口上限（首次未送达） | {_hours(step.get('upper_bound_h'))} |",
        f"| 与手机对账 | {'**不一致**' if step.get('mismatch') else '一致'} |",
        f"| 因条数不足而省略的测量点 | {', '.join(f'{float(h):g}' for h in dropped) or '无'} |",
        f"| 入站时间 | {_time(step.get('inbound_at'))} |",
    ]
    return "\n".join(lines) + "\n"


def _empty_token_section(step: dict[str, Any]) -> str:
    if step.get("status") != "done":
        return f"状态：{step.get('status', '未运行')}\n"
    failure = step.get("failure") or {}
    return (
        "| 项目 | 值 |\n| --- | --- |\n"
        f"| 接口接受 | {_yes_no(step.get('api_ok'))} |\n"
        f"| 手机收到 | {_yes_no(step.get('delivered'))} |\n"
        f"| 失败的 ret / errcode / errmsg | {_cell(failure.get('ret'))} / "
        f"{_cell(failure.get('errcode'))} / {_cell(failure.get('errmsg'))} |\n"
    )


def _failures_section(summary: ChannelProbeSummary) -> str:
    if not summary.failures:
        return "没有失败的发送。\n"
    lines = [
        "| 步骤 | 第几次 | 发送 | 结果 | code | ret | errcode | errmsg（脱敏） | HTTP | 入站后小时 | 时间 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.failures:
        hours = row.get("hours_after_inbound")
        lines.append(
            f"| {row['step']} | {row['attempt']} | {_cell(row['action'])} | "
            f"{_cell(row['outcome'])}:{_cell(row['reason'])} | {_cell(row['code'])} | "
            f"{_cell(row['ret'])} | {_cell(row['errcode'])} | {_cell(row['errmsg'])} | "
            f"{_cell(row['http_status'])} | {'—' if hours is None else f'{hours:.2f}'} | "
            f"{_time(row['at'])} |"
        )
    return "\n".join(lines) + "\n"


def _suggestions_section(summary: ChannelProbeSummary) -> str:
    window = summary.suggestions.get("channel.proactive_window_safe_h")
    quota = summary.suggestions.get("channel.outbound_quota_safe")
    lines = [
        "| 配置键 | 建议值 | 依据 |",
        "| --- | --- | --- |",
        f"| `channel.proactive_window_safe_h` | {_cell(window)} | 窗口下限 × {MARGIN:g} |",
        f"| `channel.outbound_quota_safe` | {_cell(quota)} | N × {MARGIN:g}（向下取整，至少 1） |",
        "",
        "这只是建议：`twin channel probe report` 会先问你，确认后才写入 `config/config.yaml`。",
    ]
    return "\n".join(lines) + "\n"


def _steps_section(summary: ChannelProbeSummary) -> str:
    lines = ["| 步骤 | 状态 | 尝试次数 | 作废次数 | 结束时间 |", "| --- | --- | --- | --- | --- |"]
    for name, row in summary.steps.items():
        lines.append(
            f"| {name} | {row['status']} | {row['attempts']} | {row['voided_attempts']} | "
            f"{_time(row.get('finished_at'))} |"
        )
    return "\n".join(lines) + "\n"


def render_report(summary: ChannelProbeSummary) -> str:
    """The final report for a probe that ran on a real phone."""
    state = {
        "completed": "已完成",
        "stopped": "已停止（未做完）",
        "running": "进行中（未做完）",
    }.get(summary.status, summary.status)
    notes = "\n".join(f"- {note}" for note in summary.notes) or "无。"
    empty = summary.steps.get("empty_token")
    empty_block = (
        f"\n### 可选实验：空 context_token\n\n{_empty_token_section(empty)}" if empty else ""
    )
    return f"""# 微信通道实测报告（M0 · ClawBot / iLink）

{MEASURED_MARKER}

> 状态：**已实测（{state}）**。运行编号 `{summary.run_id}`，开始于 {_time(summary.started_at)}，
> 结束于 {_time(summary.finished_at)}。本文件只含测量结果与时间，不含任何对话内容。

## 结论

{_verdict_block(summary)}

## 实测结果

### 1. 单次入站后可发条数

{_count_section(summary.steps.get("count", {}), summary)}
### 2. 媒体与交互

{_media_section(summary.steps.get("media", {}), summary)}
### 3. 主动发送窗口

{_window_section(summary.steps.get("window", {}))}
### 4. 回复与主动出站是否共用配额

结论：{_yes_no(summary.quota_shared)}（共用）。依据：{summary.quota_basis}。
{empty_block}
## 失败记录

{_failures_section(summary)}
## 建议的配置值

{_suggestions_section(summary)}
## 各步状态与备注

{_steps_section(summary)}
备注：

{notes}
"""


def write_report(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` (UTF-8, ``\\n`` line ends, directories created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
