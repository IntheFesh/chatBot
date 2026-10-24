# CLAUDE.md — wechat-twin 项目记忆

> 本文件放在仓库根目录，Claude Code 每次会话都会读取。任何一轮开发都必须遵守这里的规则。
> 完整需求见 `docs/SPEC.md`（唯一事实源，需求编号 `R-xxx`）；需求追踪见 `docs/TRACEABILITY.md`。

## 1. 项目是什么

`wechat-twin` 是一个只给一个人（仓库主人，下称"用户"）使用的微信拟人聊天机器人：

- 模仿用户女朋友（下称"她"，已知情并授权）的聊天风格，在用户微信的 **ClawBot 会话**里和用户聊天；
- 记得真实聊天记录里的事和与机器人聊过的事；
- 有作息、有回复延迟、会在合理时间主动发消息，睡觉时间基本不打扰；
- 理解/记忆/规划/看图用 **DeepSeek API**（`deepseek-flash` 为主）；
- 另在 **AutoDL 单卡**（RTX 5090 32GB 或 RTX PRO 6000 96GB）上 LoRA 微调一个 Qwen3 开源模型作为"风格模型"，作为可切换的生成后端；
- 运行在用户的 **Windows 10/11 x64 常开电脑**上，全部聊天记录只在本地；
- 时区可切换：默认 `America/Chicago`，用户回国后切到 `Asia/Shanghai`。

## 2. 铁律（违反任何一条都视为该轮未完成）

1. **不做玩具实现。** 交付代码中不得出现 `TODO`、`FIXME`、`pass` 占位、`raise NotImplementedError`、"简化版/示例版/mock 版"逻辑、写死的样例数据、只打印不执行的分支。遇到外部行为不确定，写**验证脚本 + 报告**，再按验证结果实现，而不是留桩。
2. **生产代码路径里不得有假实现。** Mock/Fake 只允许出现在 `tests/` 下（共享的测试替身放 `tests/support/`）。`src/` 绝不导入 `tests`，也不提供跑在假模型上的命令；需要用假模型做模拟或长时间运行验证的开发工具放在 `scripts/`，可以导入 `tests/support/`。需要离线可运行的替代（如本地控制台通道 `LocalConsoleChannel`）必须是功能完整的真实实现，并在 SPEC 中有编号。
3. **每个需求都有测试。** 新增/修改的每个 `R-xxx` 至少有一个测试覆盖其关键行为；**全部** `src/twin` 行覆盖率 ≥ 85%（每个子包 ≥ 75%），全部 `src/twin` 通过 `mypy --strict`。测试要验证行为，不能只为让 `trace_check` 找到测试名而写空壳测试。
4. **不得为了让测试通过而削弱测试**（删断言、放宽阈值、跳过用例、`xfail`）。测试失败就修实现；确属测试本身错误，要在汇报里说明理由。
5. **时间一律带时区。** 禁止 naive `datetime`；存储一律 UTC（`datetime` with `tzinfo=UTC` 或 epoch 秒）；展示与作息计算用 `zoneinfo.ZoneInfo`。Windows 没有系统 IANA 时区库，必须依赖 `tzdata` 包。
6. **隐私：**
   - 真实聊天数据、媒体、数据库、向量库、训练集、模型文件**永远不进 git**（`.gitignore` 覆盖 `data/`、`*.db`、`*.enc`、`exports/`、`models/`、`backups/`、`.env`）。
   - 测试只用 `tests/fixtures/` 下**合成**数据，不得复制任何真实聊天内容。
   - 日志在 INFO 及以上级别不得出现消息正文；DEBUG 级别正文也必须经过脱敏。
   - 发往 DeepSeek 或 AutoDL 的内容必须先过 `twin.llm.redaction`。
   - 通道层只允许给已绑定的唯一用户发消息，代码层面拒绝任何其他收件人。
7. **机器人自己的回复永远不进入风格样本库、检索库和训练集。** 这是防止风格漂移的硬约束，必须有测试守护。
8. **一处实现，处处复用。** 提示词渲染、模板、时区换算、脱敏、费用计算、留出集切分（`holdout_cutoff()`）、事件文字（`render_event_text()`）各只有一份实现；训练集导出与线上风格模型推理用同一个 `StylePromptBuilder`（与 LLaMA-Factory 模板逐 token 一致，有专门测试）。
9. **不确定就停下来问。** 当 SPEC 与现实冲突（例如协议行为与预期不符），先写清楚证据，停止该分支并向用户报告，不要自行降级需求。
10. **秘密不落盘明文。** DeepSeek API Key、SMTP 密码、数据库密钥、AutoDL 密码存 Windows 凭据管理器（`keyring`），不写入配置文件或日志。
11. **配置三处一致。** 新增任何可调参数，必须在同一轮同时加入 `twin.config.Settings`、`config/config.example.yaml` 和 `docs/SPEC.md` 的 R-CFG-004，并在汇报中列出（R-CFG-005 有测试比对三处）。这是唯一不需要先征得同意的 SPEC 修改。
12. **里程碑门槛是硬门槛。** 提示词里有"门槛检查"的轮次，必须运行 `twin eval gate <里程碑>` 并通过才算完成；未通过就留在该轮改进（不得调低门槛或改统计口径）。每轮开始时先确认上一个门槛已通过（`twin eval gate <里程碑> --check`）。唯一例外是 M5：第 14 轮只要求运行并记录 `twin eval gate M5`，未通过时保留 DeepSeek 后端，照常进入第 15 轮。
13. **训练与评估不得看到未来。** 训练集导出与评估沙盒只能通过 `AsOfView(t)` 和 pre_holdout 范围的派生数据读取上下文（R-TRN-013），不得直接查询全量数据。

## 3. 技术栈（固定，改动需在 SPEC 中先改）

- Python 3.12（`uv` 管理依赖与虚拟环境，`pyproject.toml` 锁定版本，提交 `uv.lock`）
- 异步：`asyncio` + `httpx`；CLI：`typer`；配置：`pydantic` v2 + `pydantic-settings` + YAML
- 存储：SQLite（WAL）+ SQLAlchemy 2.x + Alembic；敏感字段 AES-256-GCM（`cryptography`），密钥在 `keyring`
- 向量：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`（可配置 `BAAI/bge-m3`），索引 LanceDB（只存向量与 id，不存明文）
- 流式 JSON：`ijson`；时区：`zoneinfo` + `tzdata`；中国节假日：`chinese-calendar`
- LLM：`openai` SDK 指向 `https://api.deepseek.com`；风格模型：llama.cpp `llama-server` 的 `/completion`（本地）或 vLLM 的 `/v1/completions`（AutoDL 远程，经 `asyncssh` 隧道）——两者都只接收我们渲染好的提示词字符串，不用服务端聊天模板
- 远程：`asyncssh`（AutoDL 用密码登录；不依赖 Windows 的 OpenSSH）
- 训练：LLaMA-Factory（在 AutoDL 上，固定版本），模板 `qwen3_nothink`（纯 ChatML），`mask_history: true`
- 测试：`pytest`、`pytest-asyncio`、`respx`、`time-machine`、`hypothesis`；质量：`ruff`、`mypy --strict`（全部 `src/twin`）

## 4. 目录结构（由第 00 轮建立，之后保持）

```
wechat-twin/
  CLAUDE.md  README.md  pyproject.toml  uv.lock  alembic.ini
  config/            config.example.yaml（无秘密）、lists/（AI 腔短语、承诺句式、危机关键词词表）
  docs/              SPEC.md TRACEABILITY.md CHANNEL_REPORT.md LLM_REPORT.md DECISIONS.md RUNBOOK.md 等
  prompts/           各轮提示词（只读参考；轮次编号 00–16，另有 09b）
  src/twin/
    config/ storage/ channel/ llm/ ingest/ profile/ stickers/ retrieval/
    memory/ engine/ schedule/ commands/ learning/ ops/ training/ eval/
    cli.py app.py
  training/          AutoDL 侧脚本与 LLaMA-Factory 配置模板
  scripts/           trace_check.py、privacy_scan.py、基准与模拟等开发工具（可导入 tests/support）
  scripts/windows/   PowerShell 安装、计划任务、llama.cpp 下载
  tests/             unit/ integration/ fixtures/（全部合成数据）support/（共享测试替身）
  data/              运行时数据（gitignore）
```

## 5. 常用命令

```
uv sync                       # 安装依赖
uv run pytest -q              # 全部测试
uv run python scripts/shard_tests.py --shard 1 --of 3   # 本地取 CI 的一片测试文件（见 docs/CI.md）
uv run ruff check . && uv run ruff format --check .
uv run mypy src/twin
uv run twin --help            # CLI
uv run alembic upgrade head   # 数据库迁移
```

## 6. 每轮的工作方式

1. 先读本文件、`docs/SPEC.md` 中本轮涉及的章节、`docs/TRACEABILITY.md`；若提示词的"前置"写了门槛，先运行 `twin eval gate <里程碑> --check` 确认已通过。
2. 先用计划模式列出：要改的文件、要实现的 `R-xxx`、要写的测试、风险点。计划获得用户确认后再动手。
3. 实现 → 写测试 → 跑 `ruff`、`mypy`、`pytest` 全部通过。
4. 更新 `docs/TRACEABILITY.md`：本轮实现的每个 `R-xxx` 填上"实现位置（文件:符号）"和"测试名"。多轮共同负责的编号，只有最后一个负责轮次完成时才把状态改为"已实现"，之前的轮次在实现位置后注明"（部分：第 NN 轮）"。
5. 每个新 CLI 命令声明进程模型类别（只读 / 轻量修改 / 重任务 / 独占，R-ARCH-006）。注册导入后钩子的轮次，同时提供对应的回填命令（R-IMP-011）。
6. 汇报格式（每轮结束必须输出）：
   - 已实现的 `R-xxx` 列表，每条附实现位置与测试名
   - 跑过的命令与结果（贴关键输出）
   - 需要用户手动做的事（扫码、填 Key、放数据路径等）
   - 已知问题与下一轮注意事项（不得把未完成的需求藏在这里——未完成就是本轮未完成）

## 7. Windows 注意事项

- 一律用 `pathlib`；不要拼接反斜杠；注意长路径与中文路径（导出目录名含中文和特殊符号）。
- 控制台编码：程序启动时强制 UTF-8（`PYTHONUTF8=1` 与 `sys.stdout.reconfigure`）。
- 子进程（`llama-server.exe`、`twin supervise` 托管的 `twin run`）用 `asyncio.create_subprocess_exec`，并在退出时可靠终止（Job Object 或进程树终止）。SSH 一律用 `asyncssh`，不调用 `ssh.exe`。
- 防休眠：运行期间调用 `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`；仍要处理手动睡眠/合盖后的唤醒（R-SCH-005）。
- 单实例：两把独立的锁（命名互斥体或文件锁）——`run` 锁由 `twin run` 持有，`supervisor` 锁由 `twin supervise` 持有，互不冲突；独占命令检查两把锁；其他 CLI 命令按 R-ARCH-006 与运行中的应用协作。
- 计划任务以 `InteractiveToken` 登录类型运行（keyring、Windows 通知、二维码窗口需要用户会话）；不要用 Windows 服务。
