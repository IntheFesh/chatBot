"""``docs/LLM_REPORT.md``: the written result of the M0 probe (R-LLM-013).

Two renderings exist.  :func:`render_pending_report` is what the repository ships before the
probe has run against the real API: it explains the checks and the verdict rules and lists what
the official documentation says, but contains **no measured number** - none can exist without
an API key.  :func:`render_report` turns a real :class:`~twin.llm.probe.ProbeReport` into the
final report.  The report holds technical results only (sizes, counts, times, cost); the probe
sends no conversation content, so there is none to leak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twin.llm import official
from twin.llm.probe import CHECKS, GATE_CHECKS, CheckResult, ProbeReport

PENDING_MARKER = "<!-- llm-report: pending -->"
MEASURED_MARKER = "<!-- llm-report: measured -->"

_TITLES = {
    1: "思考开关：开启与关闭都成功；开启时返回 reasoning_content，关闭时不返回",
    2: "上下文缓存：同一长前缀重复请求，prompt_cache_hit_tokens > 0",
    3: "看图：程序生成的 JPEG、PNG、多帧 GIF 各一张（GIF 只记录结果）",
    4: "JSON 输出：思考关闭与开启两种情况下都能被解析",
    5: "detail 参数：带与不带各请求一次，是否被接受",
    6: "每张图的实际计费 token 数：不同尺寸各一张，由 usage 差值推算",
    7: "各请求的耗时与费用",
}


def _documentation_section() -> str:
    flash = official.PEAK_PRICES_USD_PER_MTOK[official.FLASH_MODEL]
    pro = official.PEAK_PRICES_USD_PER_MTOK[official.PRO_MODEL]
    pages = "\n".join(f"- {name}: {url}" for name, url in official.DOC_PAGES.items())
    return f"""## 官方文档核对（{official.CHECKED_ON.isoformat()}，非实测）

核对的页面：

{pages}

| 项目 | 官方文档的说法 | 实现中的处理 |
| --- | --- | --- |
| 模型名 | `{official.FLASH_MODEL}`（支持看图）、`{official.PRO_MODEL}`（不支持看图）；旧名 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 仍可用并按 Flash 计价 | 与 SPEC 一致；旧名按 Flash 计价；`vision_model` 配置为不能看图的模型时拒绝启动 |
| 思考开关 | 默认开启；`extra_body={{"thinking": {{"type": "enabled"|"disabled"}}}}`；`reasoning_effort` 取 low/high/max；开启时 `temperature`、`presence_penalty`、`frequency_penalty` 不报错但无效；没有 `tools` 参数时历史回合的 `reasoning_content` 会被 API 忽略 | 两种状态都显式发送；思考时不发送上述三个参数并记录一次警告；任何历史都不带 `reasoning_content` |
| 价格（美元/百万 token，高峰） | Flash：缓存命中 {flash["cache_hit"]}、未命中 {flash["cache_miss"]}、输出 {flash["output"]}；Pro：{pro["cache_hit"]}、{pro["cache_miss"]}、{pro["output"]}；非高峰为一半 | 与 SPEC R-CFG-004 的价格表逐项一致（有测试） |
| 高峰时段 | 周一至周五（不含中国法定节假日）UTC 01:00–04:00、06:00–10:00（北京时间 09:00–12:00、14:00–18:00）；周末与法定节假日全天为空闲时段 | 用 `chinese_calendar.is_workday()` 判定（含调休上班的周末，见 DECISIONS.md D-004） |
| 缓存字段 | usage 中有 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`；命中要求前缀已被持久化，构建需要数秒，尽力而为 | 逐次记账；探针对同一请求等待后重试 |
| 看图限制 | 仅 user 消息；JPEG/PNG/GIF/WebP 按文件头识别；`detail` 取 low/high/original/auto；每张图最多 {official.MAX_IMAGE_TOKENS} token；单图 32 MiB，请求体 48 MiB，每边 {official.MAX_IMAGE_SIDE_PX} px（15 张以上 {official.MAX_IMAGE_SIDE_PX_MANY} px），每次最多 {official.MAX_IMAGES_PER_REQUEST} 张 | 超限时用 Pillow 本地缩放；system/assistant 消息里放图直接抛异常 |
| JSON 输出 | `response_format={{"type": "json_object"}}`；提示词里要有 "json"；要设置 `max_tokens`；偶尔返回空内容 | 自动补充指令；空内容当作无效并重试一次 |
| 错误码 | 400/422 请求有误，401 鉴权，402 余额不足，429 限速，500/503 服务端 | 429/5xx/超时/连接错误重试；其余不重试，401/402/403 立即告警 |
"""


def render_pending_report() -> str:
    """The report as shipped before any real probe: explains, but states no measurement."""
    rows = "\n".join(
        f"| {number} | `{CHECKS[number]}` | {_TITLES[number]} | "
        f"{'是' if number in GATE_CHECKS else '否（测量项）'} |"
        for number in range(1, 8)
    )
    return f"""# LLM 层实测报告（M0 · DeepSeek）

{PENDING_MARKER}

> 状态：**待实测**。
> 编写本轮代码的环境没有 DeepSeek API Key，`twin llm probe` 还没有对真实接口运行过，所以本文件
> 里没有任何实测数字。运行探针后，本文件会被实测结果整体覆盖。

## 怎样得到实测结果

```
uv run twin secrets set deepseek_api_key     # 粘贴 Key
uv run twin llm probe                        # 约几十个很小的请求，费用远低于 0.1 美元
```

探针把结果写入 `docs/LLM_REPORT.md`，同时以结构化形式存入数据库（`settings` 表，键 `m0.llm_probe`，
`schema_version` 1），第 09b 轮的 M0 判定读取它；探针学到的能力（`detail` 是否被接受、GIF 是否被
接受、思考开启时 JSON 是否可靠、各尺寸图片的计费 token）存入键 `llm.capabilities`，随后改变客户端
的行为。探针不发送任何对话内容，图片都是程序画出来的（红色圆形）。

## 七项检查与 M0 判定口径

| # | 标识 | 内容 | 计入 M0 |
| --- | --- | --- | --- |
{rows}

M0 通过 = 第 1、2、4 项成功，且第 3 项中 JPEG 与 PNG 成功（GIF 的结果只记录，不被接受时看图前取首帧
转 PNG）。第 5、6、7 项是测量项，只要求记录。

如果第 4 项显示思考开启时 JSON 不可靠，探针会在报告里写明并以非零退出码结束：主动消息规划依赖
这一点，需要先停下来商量。

{_documentation_section()}
"""


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return f"`{value}`"
    return str(value)


def _check_section(check: CheckResult) -> str:
    verdict = "通过" if check.passed else ("未运行" if not check.ran else "未通过")
    kind = "计入 M0" if check.gate else "测量项"
    lines = [
        f"### {check.number}. {_TITLES[check.number]}",
        "",
        f"结果：**{verdict}**（{kind}）",
        "",
    ]
    if check.metrics:
        lines += ["| 指标 | 值 |", "| --- | --- |"]
        for key, value in check.metrics.items():
            if key == "per_image" and isinstance(value, list):
                continue
            lines.append(f"| {key} | {_format_value(value)} |")
        lines.append("")
    per_image = check.metrics.get("per_image")
    if isinstance(per_image, list) and per_image:
        lines += ["| 尺寸 | 像素 | 计费 token |", "| --- | --- | --- |"]
        for row in per_image:
            lines.append(f"| {row['width']}×{row['height']} | {row['pixels']} | {row['tokens']} |")
        lines.append("")
    for note in check.notes:
        lines.append(f"- 注意：{note}")
    if check.notes:
        lines.append("")
    if check.requests:
        lines += [
            "| 请求 | 成功 | 耗时 ms | 提示 token | 输出 token | 缓存命中 | 费用 USD |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in check.requests:
            lines.append(
                f"| {r.label} | {'是' if r.ok else '否'} | {r.latency_ms} | {r.prompt_tokens} | "
                f"{r.completion_tokens} | {r.cache_hit_tokens} | {r.cost_usd:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_report(report: ProbeReport) -> str:
    """The final report for a probe that ran against the real API."""
    caps = report.capabilities
    verdict = "通过" if report.m0_passed else "未通过"
    gate_rows = "\n".join(
        f"| {c.number} | `{c.id}` | {'通过' if c.passed else ('未运行' if not c.ran else '未通过')} |"
        for c in report.checks
        if c.gate
    )
    json_on = report.check("json_output")
    warnings: list[str] = []
    if report.fatal:
        warnings.append(f"探针中途停止：{report.fatal}")
    if json_on.ran and not json_on.metrics.get("thinking_on_ok"):
        warnings.append(
            "思考开启时 JSON 输出不可靠：主动消息规划（第 11 轮）依赖它，需要先停下来商量，"
            "不要继续往下实现。"
        )
    adjustments = [
        f"- `detail` 参数：{'发送' if caps.detail_supported else '不再发送（API 拒绝了它）'}",
        f"- GIF：{'原样发送' if caps.gif_supported else '看图前取首帧转 PNG（API 拒绝了 GIF）'}",
        f"- 思考开启时的 JSON：{'可靠' if caps.json_in_thinking else '不可靠'}",
        f"- 图片 token 估算：{'按实测值（见第 6 项）' if caps.image_tokens else '沿用文档上限 1024'}",
    ]
    sections = "\n".join(_check_section(c) for c in report.checks)
    warning_block = "\n".join(f"> **{w}**" for w in warnings)
    return f"""# LLM 层实测报告（M0 · DeepSeek）

{MEASURED_MARKER}

> 状态：**已实测**。运行编号 `{report.run_id}`，开始于 {report.started_at}，结束于 {report.finished_at}（UTC）。
> 模型：文本 `{report.model}`，看图 `{report.vision_model}`。总费用 {report.total_cost_usd:.4f} 美元（`one_time` 账目，批次 `{report.run_id}`）。

## 结论

M0 判定：**{verdict}**

| # | 标识 | 结果 |
| --- | --- | --- |
{gate_rows}

{warning_block}

探针学到的、已写入 `llm.capabilities` 并改变客户端行为的内容：

{chr(10).join(adjustments)}

## 实测结果

{sections}
{_documentation_section()}
"""


def write_report(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` (UTF-8, ``\\n`` line ends, directories created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
