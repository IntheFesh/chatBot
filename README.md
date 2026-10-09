# wechat-twin

只给一个人使用的微信拟人聊天机器人：在用户微信的 ClawBot 会话里，模仿**已知情并授权**的她的说话方式，记得真实聊天记录里的事，有作息、有回复延迟、会在合适的时间主动开口。全部聊天记录只在本机保存并加密。

- 需求与规格：[`docs/SPEC.md`](docs/SPEC.md)（唯一事实源）、追踪表 [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md)
- 项目规则：[`CLAUDE.md`](CLAUDE.md)；取舍与偏差：[`docs/DECISIONS.md`](docs/DECISIONS.md)；需要你手动做的事：[`docs/PENDING_USER_ACTIONS.md`](docs/PENDING_USER_ACTIONS.md)

> 当前进度：第 04 轮（风格统计画像与作息模型）。已有：配置与秘密、加密存储与媒体库、持久化任务队列、CLI 进程模型、应用生命周期、日志、Windows 单实例与防休眠、`twin doctor`；DeepSeek 客户端（思考开关、JSON、看图、重试与熔断）、费用记账与非高峰判定、预算降级、一次性批任务预算、脱敏、风格模型客户端、M0 探针；聊天记录导入（流式、可续传、增量、媒体与表情包、图片描述、导入报告与导入后钩子）；风格统计画像（可抽样的经验分布、双窗口、版本化与回滚）、作息活动模型（睡眠/忙碌/先开口/回复延迟，按当地钟点学习，手动修正）与唯一的留出切分点 `holdout_cutoff()`（live 与 pre_holdout 两个范围）。通道、聊天等逐轮加入。

## 安装（Windows 10/11 x64 或 Linux）

```powershell
# 1. 安装 uv（https://docs.astral.sh/uv/），然后：
uv python install 3.12
uv sync                      # 按 uv.lock 安装依赖

# 2. 检查环境
uv run twin doctor

# 3. 初始化数据库并启动
uv run twin db upgrade
uv run twin run              # Ctrl+C 优雅退出
```

可选：`copy config\config.example.yaml config\config.yaml` 后编辑（该文件不会被提交）。时区默认 `America/Chicago`，回国后用 `twin settings set time.bot_timezone Asia/Shanghai`（第 11 轮起也可在微信里用 `/时区`）。

## 日常命令

| 命令 | 作用 | 进程类别 |
| --- | --- | --- |
| `twin run` | 启动应用（任务队列、状态监视、心跳） | 独占（持有 `run` 锁） |
| `twin doctor` | 检查 Python、依赖、时区库、凭据存储、磁盘、数据库版本 | 只读 |
| `twin config show` | 打印生效配置（个人信息打码） | 只读 |
| `twin settings list\|set\|history` | 查看 / 修改运行时设置（存数据库，应用 2 秒内感知） | 只读 / 轻量修改 |
| `twin secrets set\|delete\|list\|check <名字>` | 凭据管理器里的秘密（从不显示值） | 轻量修改 / 只读 |
| `twin secrets rotate-db-key` | 轮换数据库主密钥（可中断续跑） | 独占 |
| `twin db upgrade\|status` | 数据库迁移 | 独占 / 只读 |
| `twin llm probe` | M0 探针：对真实 DeepSeek 接口做七项检查并写 `docs/LLM_REPORT.md`（需要 Key；先 `twin secrets set deepseek_api_key`） | 轻量修改 |
| `twin llm status` | 模型、探针学到的能力、预算级别与今日花费 | 只读 |
| `twin jobs list\|show\|retry\|cancel\|approve` | 任务队列；`jobs run --until-idle` 在应用未运行时前台执行 | 只读 / 轻量修改 / 重任务 |
| `twin import <目录> [--foreground] [--resume] [--target N]` | 导入聊天记录：首次让你选目标会话；默认入队后立即返回，`--foreground` 在应用未运行时前台执行（进度条），`--resume` 续跑 | 重任务 |
| `twin import status [--watch]` | 阶段、已处理条数 / 总数、速度、预计剩余时间与各导入后钩子 | 只读 |
| `twin import inspect <目录>` | 只输出导出目录的结构（键名、类型、枚举取值计数），不含任何值；写 `data/reports/inspect-*.md` | 只读 |
| `twin stickers download [--retry-failed]` | 下载导出里没有文件的表情包（并发 4、每秒 ≤ 4） | 重任务 |
| `twin images caption-backfill [--days N]` | 为最近 N 天的图片排队描述（一次性批任务，先给估算费用，`twin jobs approve` 后执行） | 重任务 |
| `twin profile rebuild [--scope live\|pre_holdout\|all] [--foreground] [--force]` | 重算风格画像与作息模型（导入后自动排队；结果与上一版相同时不写新版本） | 重任务 |
| `twin profile show [版本] [--scope live\|pre_holdout]` | 风格数字（与 SPEC §0 样本并列）、数字风格规则、作息概览（当地时间；睡眠落在白天时醒目警告）并请你确认睡眠时段 | 只读 |
| `twin profile history\|diff <a> <b>\|phrases` | 版本列表；两版相差超过 10% 的指标；她的高频整句/字组/称呼候选（仅本机屏幕） | 只读 |
| `twin profile rollback <版本>` | 切回旧版本（连同当时的作息模型） | 轻量修改 |
| `twin routine list` | 手动作息修正列表 | 只读 |
| `twin routine add sleep\|busy\|holiday …` / `remove\|enable\|disable <id>` | 手动修正睡眠区间、每周忙碌时段、节假日日期区间（优先于推断） | 轻量修改 |
| `twin retrieval rebuild [--full] [--foreground]` | 建 / 修复“她在类似情况下怎么回”的检索库（真实向量模型，首次使用下载到 `data/models/embeddings/`；可续跑；显示进度与预计剩余；导入后自动增量） | 重任务 |
| `twin retrieval stats` | 窗口数、留出数、已编码数、向量模型与最近一次编码速度（只有计数，不含聊天内容） | 只读 |
| `twin retrieval resplit [--yes] [--foreground]` | 把留出集切分点移到今天数据的最新 10%（评估结果前后不可比）；立刻把新留出的窗口移出索引 | 重任务 |

全局选项：`--config <文件>`、`--set 键.路径=值`（可重复）、`--log-level`。配置优先级：命令行 > 环境变量（`TWIN_` 前缀，嵌套用 `__`）> `config/config.yaml` > 默认值。

## 开发

```powershell
uv run ruff check . ; uv run ruff format --check .
uv run mypy src/twin
uv run pytest -q --cov=src/twin --cov-report=json
uv run python scripts/coverage_gate.py                 # 总体 >= 85%，每个子包 >= 75%
uv run python scripts/trace_check.py --round 01        # 需求追踪
uv run python scripts/privacy_scan.py                  # 提交前隐私扫描
powershell -File scripts/check.ps1 -Round 01           # 以上全部一键运行（Windows）
```

规则摘要（完整见 `CLAUDE.md`）：不留桩和玩具实现；时间一律带时区并通过 `twin.clock`；真实数据永远不进 git，测试只用合成数据；日志 INFO 及以上不含正文；秘密只存凭据管理器；机器人自己的回复永远不进风格样本、检索库和训练集。

## 目录

```
src/twin/   config storage channel llm ingest profile stickers retrieval memory engine
            schedule commands learning ops training eval  cli.py app.py clock.py services.py
config/     config.example.yaml、lists/（AI 腔短语、承诺句式、危机关键词）
docs/       SPEC / TRACEABILITY / DECISIONS / PENDING_USER_ACTIONS / EXECUTION_NOTES ...
prompts/    各轮提示词（只读）
scripts/    trace_check.py coverage_gate.py privacy_scan.py simulate_proactive.py check.ps1 ...
tests/      unit/ integration/ fixtures/（全部合成数据） support/（共享测试替身）
data/       运行时数据（gitignore）
```
