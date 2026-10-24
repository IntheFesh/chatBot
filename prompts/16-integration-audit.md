# 第 16 轮：端到端集成、长时间运行验证、文档与需求覆盖总审计

> 里程碑：全部收尾 · 前置：第 00–15 轮全绿；M0–M4 门槛已通过（M5 通过与否均可）

## 先读
`CLAUDE.md`；`docs/SPEC.md` 全文；`docs/TRACEABILITY.md`；全部 `docs/*_REPORT.md`。

## 本轮目标
证明整个系统作为一个整体可用、可长期运行、文档完整，并且 SPEC 中**每一个**需求编号都有实现与测试；找出并补齐任何遗漏，而不是把遗漏写进"已知问题"。

## 必须实现的需求
R-ARCH-004（端到端验证）、R-SCOPE-008（非目标未出现半成品的审计）、R-SCOPE-009（取舍原则的审计）、R-SAFE-004（对外联系审计）、R-NFR-001～006、R-PRIV-001（总审计）、`docs/TRACEABILITY.md` 中全部编号状态为"已实现"。

## 详细要求

### A. 端到端测试（integration）
1. 场景测试（`tests/integration/`；`LocalConsoleChannel` + 注入时钟 + respx 假 DeepSeek（按请求内容返回确定性结果）+ `tests/support/` 的假 llama-server）：
   - 一天完整生活：起床问候 → 用户上午连发三条 → 她忙碌时延迟回复 → 用户发图片与表情包 → 中午沉默 → 她分享生活线 → 用户说明天考试 → 晚上睡前道晚安 → 用户深夜发消息 → 她第二天醒来才回并跟进考试；
   - 时区切换：中午在芝加哥切到上海；
   - 夏令时结束日 2026-11-01 的一整天；
   - 预算逐级降级到 3 级再恢复；
   - 微信会话窗口过期与恢复；
   - 风格模型掉线回退与自动恢复；
   - 危机消息跳出角色；
   - 进程在 SENDING 中途被杀后重启恢复；
   - 全部指令走一遍；
   - 电脑睡眠 3 小时后唤醒（墙钟跳变）：过期主动不补发、计划重建；
   - 平台条数将尽：回复气泡被合并、主动消息被截短或抑制；
   - 模型输出含事件文字与"我拍给你看"之类承诺：被删除或重写，没有任何照片被发送；
   - 应用运行时在另一个进程执行 `twin persona edit`、`/时区`、`twin import`：缓存在 2 秒内失效、导入由应用执行且进度可查；运行时执行 `twin purge --all` 被拒绝。
   每个场景断言：发送时刻落在规则允许范围、无报错外泄、`bot_turns`/`proactive_log`/记忆写入正确、机器人回复未进入检索与训练入口。
2. 长时间运行：`scripts/soak.py --days 14 --accelerated`（开发工具，导入 `tests/support/` 的假模型与可快进时钟，驱动生产代码；不是 `twin` 子命令，`src/` 中不出现任何假实现）：在本地通道上运行 14 个模拟日，监控内存增长（不得超过 R-NFR-002）、任务队列积压、数据库大小、异常计数；输出报告。真实环境的 7 天无人值守验证已在第 12 轮 M4 门槛完成（R-EVAL-006）。

### B. 非功能核对（R-NFR-001～005）
- 性能：测量生成耗时 p95（假模型下引擎自身开销 + 真实 DeepSeek 抽样 20 次的 live 测试，标记 live）、常驻内存、空闲 CPU、启动耗时；结果写入 `docs/PERFORMANCE.md`。
- 代码质量：ruff 零告警、**全部** `src/twin` mypy strict 通过、全部 `src/twin` 覆盖率 ≥ 85% 且每个子包 ≥ 75%（列出各子包覆盖率）、单元测试零网络访问（用一个全局 fixture 禁止未拦截的网络连接）。
- 时间逻辑：确认所有时间相关模块都有注入时钟测试（用扫描脚本列出调用 `TimeService`/`Clock` 的模块与对应测试文件）。

### C. 文档（R-NFR-006）
1. `README.md`：这是什么、硬件与系统要求、安装（指向 `install.ps1`）、日常使用（指令速查）、隐私说明。
2. `docs/RUNBOOK.md`，至少包含以下章节且每章有可直接执行的命令：
   - 首次部署（install、setup、consent、DeepSeek Key、SMTP）与服务管理（`twin service start|stop|status`、哪些命令需要先停止应用）；
   - 微信扫码登录与绑定、掉线后重新登录；
   - 导入全部聊天记录与以后增量导入（`twin import inspect`、`twin import status`、一次性批任务的费用确认 `twin jobs approve`）；
   - 人设卡审阅与手改；
   - 训练风格模型：AutoDL 租卡（RTX 5090 或 RTX PRO 6000）、按档位扩容数据盘、填写 `autodl.*` 与密码、`twin train remote all`、清理与释放实例；
   - 评估与门槛：盲测、记忆测试、主动审计、稳定性、`twin eval gate`、`twin eval report`；
   - 部署风格模型：本地 llama.cpp 或远程隧道、评估与激活门槛；
   - 回国切换时区（`/时区 北京`）与回美国切回；
   - 备份、校验、恢复；
   - 一键删除（她要求时）；
   - 常见故障排查（DeepSeek 余额不足、微信掉线、主动消息被窗口抑制、风格模型掉线、磁盘满）。
3. `docs/ARCHITECTURE.md`：组件图（Mermaid）、数据流、状态机、表结构概览、关键设计决策（与 SPEC 一致）。

### D. 需求覆盖总审计
1. 运行 `uv run python scripts/trace_check.py`（不带 `--round`，检查全部编号）：每个 `R-xxx` 必须为"已实现"、有实现位置与真实存在的测试。任何缺口在本轮补齐实现与测试。
2. 另做一次"反向审计"：读产品文档对应的 `docs/TRACEABILITY.md` 第一张表（产品需求 → SPEC），逐条在系统中演示或用测试证明（列出命令或测试名），把结果填进该表的"验证方式"列。
3. 代码扫描 `scripts/stub_scan.py`（纳入 CI）：
   - 关键词：`src/` 下不得出现 `TODO`、`FIXME`、`XXX`、`NotImplementedError`、`stub`、`placeholder`、`mock`、`fake`、`dummy`、`简化`、`示例`、`暂不`、`以后再`（标识符、注释与字符串都查，误报需在白名单文件中逐条说明理由）；
   - AST：函数或方法体只有 `pass`、`...`、文档字符串、`return None` 或返回字面常量（空列表、空字典、`True`/`False`、固定数字或字符串）的，除 `Protocol` 成员、`@abstractmethod`、`@overload` 与白名单外一律报错；
   - `src/` 不导入 `tests`；没有 `twin` 子命令依赖假模型；
   - 隐私扫描通过。
4. 取舍审计（R-SCOPE-009）：`docs/DECISIONS.md` 至少覆盖 SPEC 列出的取舍点，每条写明顺序、优先级理由与测试节点 id；`scripts/decisions_check.py` 核对每个测试节点在 `pytest --collect-only` 中存在。
5. 对外联系审计（R-SAFE-004）：扫描测试确认全部出站途径（通道发送、SMTP）的收件人只来自绑定用户、用户本人邮箱或显式开启的紧急联系人，邮件模板中没有聊天正文字段。

### E. 发布
- 打 `v1.0.0` 标签前的检查清单（写在 `docs/RELEASE_CHECKLIST.md`）：全部测试绿、trace_check 全绿、隐私扫描绿、`twin doctor` 绿、备份恢复演练完成、M0–M4 门槛报告、（若已训练）M5 门槛报告。

## 验收
```
uv run pytest -q
uv run pytest -q -m integration
uv run python scripts/soak.py --days 14 --accelerated
uv run python scripts/stub_scan.py
uv run python scripts/decisions_check.py
uv run python scripts/trace_check.py
uv run python scripts/privacy_scan.py
uv run twin eval report
```

## 不要做
- 不要把任何未完成项留到"以后"；本轮的定义就是"全部需求都已实现并验证"。
- 不要修改 SPEC 来迁就实现；若确需修改，先停下来说明理由并征得我同意。

## 完成后汇报
按 CLAUDE.md 格式；附 trace_check 全量输出摘要、覆盖率表、soak 报告摘要、`eval` 门槛表、发布检查清单状态。
