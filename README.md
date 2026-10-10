# wechat-twin

只给一个人使用的微信拟人聊天机器人：在你微信的 ClawBot 会话里，模仿**已知情并授权**的她的说话方式，记得真实聊天记录里的事和你们跟机器人聊过的事，有作息、有回复延迟、会在合适的时间主动开口，睡觉时间基本不打扰。全部聊天记录只在你自己的电脑上保存，并且加密。

- 需求与规格：[`docs/SPEC.md`](docs/SPEC.md)（唯一事实源）；追踪表 [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md)；取舍与偏差 [`docs/DECISIONS.md`](docs/DECISIONS.md)
- 怎么部署和运维：[`docs/RUNBOOK.md`](docs/RUNBOOK.md)；系统是怎么搭的：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)；打 `v1.0.0` 之前的检查：[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md)
- 需要你亲手做的步骤（扫码、填 Key、真实数据、GPU、观察期）：[`docs/PENDING_USER_ACTIONS.md`](docs/PENDING_USER_ACTIONS.md)
- 项目规则：[`CLAUDE.md`](CLAUDE.md)

> **重要：** 机器人说的话不代表她本人。它是模仿，不是她；用户真诚地问“你是不是 AI”时，它不会否认自己是模拟出来的。她知情并授权的日期记录在配置 `consent.confirmed_at`，缺失时程序拒绝启动。

## 它做什么

| 能力 | 怎么做到 |
| --- | --- |
| 像她 | 从真实记录统计她的说话习惯（句长、标点、连发、表情），生成人设卡；回复前检索她在类似情况下的真实回复；可选再微调一个 Qwen3 风格模型 |
| 记得住 | 近期对话原文、每日摘要、带来源与时间的事实库、她的生活线、待跟进的事；`/记住`、`/忘掉`、`/记忆` 随时管理 |
| 像真人 | 等你说完再回、回复有延迟（睡觉时不回，醒来再回）、气泡逐条发出、偶尔发表情包、不追求秒回 |
| 会主动 | 起床问候、饭点、睡前晚安、跟进你说过的事、偶尔分享；每天条数在范围内，深睡时段 0 次 |
| 可控 | 微信里的指令（见下）；运行时设置存数据库，改了几秒内生效；预算超出会逐级降级但永远不停止回复 |
| 会成长 | `/不像`、`/重来` 变成纠正规则和偏好对；导入新记录后画像、检索库、记忆自动更新 |

## 硬件与系统要求

| 项目 | 要求 |
| --- | --- |
| 运行机器 | **Windows 10/11 x64**，常开、不休眠（程序运行时会请求系统不要睡眠）。全部聊天记录只在这台机器上导入和保存 |
| 软件 | [uv](https://docs.astral.sh/uv/)（它会安装并管理 Python 3.12）；`install.ps1` 会在缺少时安装 uv |
| 内存与磁盘 | 应用常驻内存目标 < 1.5 GB（不含本机风格模型的 `llama-server`）；数据库、媒体、向量库和备份都在 `data\`，磁盘剩余低于 5 GB 会告警 |
| 网络 | 能访问 DeepSeek（`api.deepseek.com`）和微信 ClawBot 服务器（`ilinkai.weixin.qq.com`）；`twin doctor` 会逐个检查 |
| 手机 | 微信版本支持 ClawBot（iOS ≥ 8.0.70 / Android ≥ 8.0.69），“我 → 设置 → 插件”里有 ClawBot |
| 账号 | DeepSeek API Key 与余额；可选：SMTP 邮箱（告警邮件，Gmail 要应用专用密码，QQ/163 要授权码） |
| 可选：风格模型 | 训练：AutoDL 上租一张 RTX 5090（32 GB）或 RTX PRO 6000（96 GB），数据盘按档位扩容；推理：本机 NVIDIA 显卡跑 llama.cpp，或留在 AutoDL 上经 SSH 隧道使用 |
| Linux / macOS | 只用于开发和试聊（`twin chat --local`、全部测试）；计划任务、防休眠、系统通知、二维码窗口是 Windows 的 |

## 安装（Windows）

```powershell
# 在仓库根目录，用 PowerShell：
powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1
```

`install.ps1` 依次：检查并安装 uv → `uv sync --frozen`（按 `uv.lock` 安装，从不重新解析版本）→ `twin setup`（同意日期、DeepSeek Key、告警邮件、目标会话、时区、紧急联系人）→ `twin db upgrade` → `twin service install`（登录 Windows 时自启的计划任务）。加 `-StartNow` 立即启动，加 `-SkipSetup` 跳过向导。反向操作是 `scripts\windows\uninstall.ps1`（数据不动）。详细步骤与自动登录的风险见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md) 第 1 章。

手动安装也可以：

```powershell
uv python install 3.12
uv sync --frozen
uv run twin doctor          # 体检：Python、依赖、时区库、凭据存储、磁盘、数据库、网络、计划任务
uv run twin db upgrade      # 建库 / 迁移
uv run twin setup           # 第一次配置向导
```

## 第一次使用

1. **登录并绑定微信**：`uv run twin channel login`，用手机扫码，再给 ClawBot 发一条消息确认“这就是我”（绑定后机器人只会对这个账号说话）。
2. **导入聊天记录**：先 `uv run twin import inspect "<导出目录>"` 只看结构，再 `uv run twin import "<导出目录>"`，用 `uv run twin import status` 看进度。需要花钱的一次性任务会先给估算，你确认后 `uv run twin jobs approve <批次号>` 才执行。
3. **看画像、核对作息**：`uv run twin profile show`，睡眠时段不对就用 `uv run twin routine add sleep 01:00 08:30` 修正；`uv run twin persona show` 看人设卡，`uv run twin persona edit` 手改。
4. **试聊**：`uv run twin chat --local`（终端里聊，她按真实节奏回复）。
5. **常驻运行**：`uv run twin service start`（或下次登录 Windows 时自动启动）。

每一步的细节、预期输出和出问题怎么办都在 [`docs/RUNBOOK.md`](docs/RUNBOOK.md)。

## 日常使用

### 微信里的指令

在 ClawBot 会话里发给机器人。回复以 `⚙️` 开头、是系统口吻，不进入记忆、学习和训练。全角斜杠、多余空格、中英文冒号都能识别；写错了会回帮助摘要；以 `/` 开头但不是指令的（例如微信旧表情 `/::)`）仍然是普通聊天。许多指令也有英文别名（例如 `/redo`、`/backend`），`/帮助 <指令>` 会列出。

| 指令 | 作用 | 例子 |
| --- | --- | --- |
| `/帮助 [指令名]` | 列出全部指令，或某个指令的用法 | `/帮助 思考` |
| `/状态` | 时区、她此刻的状态、后端、思考模式、主动范围、平台窗口、今日费用、最近告警、重训提醒 | `/状态` |
| `/思考 开\|关\|自动` | 聊天生成的思考模式；自动 = 提问、聊情绪或长消息时开 | `/思考 自动` |
| `/显示思考 开\|关` | 是否把思考内容以系统消息发给你（调试用） | `/显示思考 开` |
| `/后端 deepseek\|style\|hybrid` | 切换生成后端（风格模型未启用或不健康时不能切到 style / hybrid） | `/后端 deepseek` |
| `/重来` | 撤销她上一轮回复（记为负例）并重新生成 | `/重来` |
| `/时区 <IANA 名称>\|查看` | 查看或切换机器人所在时区（可以写北京、芝加哥） | `/时区 北京` |
| `/暂停 <时长>` | 暂停回复和主动消息一段时间 | `/暂停 2小时` |
| `/恢复` | 结束暂停 | `/恢复` |
| `/主动 <最少>-<最多>\|开\|关` | 主动消息每天的条数范围，或者开关 | `/主动 2-5` |
| `/作息 睡 <HH:MM>-<HH:MM> \| 忙 <星期> <HH:MM>-<HH:MM> \| 假期 <日期>[..<日期>] \| 查看 \| 删除 <编号>` | 手动修正她的睡眠、忙碌时段和假期 | `/作息 睡 01:00-08:30` |
| `/记住 <内容>` | 让她记住一件事（来源是你说的，优先级最高） | `/记住 她下周三有考试` |
| `/忘掉 <内容或编号>` | 删除一条记忆，和只由它产生的待跟进、生活安排 | `/忘掉 3` |
| `/记忆 [页码\|关键词]` | 分页看她现在记得的事，或者按关键词找 | `/记忆 2` |
| `/不像 [正确说法]` | 标记她上一轮不像她；附上她会怎么说就存成一对对照 | `/不像 那你早点睡吧` |
| `/评分 <1-5> [备注]` | 给她最近一周的主动消息和整体体验打分 | `/评分 4 晚安发得很自然` |
| `/费用 [今天\|本月]` | 费用、按用途拆分、缓存命中率、预算剩余 | `/费用 本月` |
| `/导入 <路径>` | 导入新的聊天记录导出（电脑上的文件夹路径） | `/导入 D:\聊天记录\导出` |

### 电脑上的命令

`twin` 的命令都属于四类之一（只读 / 轻量修改 / 重任务 / 独占），决定它能不能在应用运行时执行，见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md) 第 0 章和附录 A。常用的：

| 命令 | 作用 |
| --- | --- |
| `twin service start\|stop\|status` | 启动 / 优雅停止 / 查看常驻的计划任务 |
| `twin health` | 通道、DeepSeek、风格模型、磁盘、任务队列、备份、预算一览 |
| `twin doctor` | 体检整个安装 |
| `twin chat --local` | 在终端里和她聊（与 `twin run` 互斥） |
| `twin channel status` | 微信登录、绑定、窗口与条数 |
| `twin import status` | 导入进度、速度、预计剩余时间与各导入后钩子 |
| `twin jobs list` | 后台任务队列；一次性任务用 `twin jobs approve <批次号>` 确认费用 |
| `twin profile show` | 风格数字与作息概览，请你确认睡眠时段 |
| `twin persona show` | 看人设卡；`twin persona edit` 手改 `[手动]` 区块 |
| `twin memory list` | 她现在记得的事 |
| `twin proactive log` | 主动消息的每次决定（默认不显示正文） |
| `twin cost report` | 按日、用途、模型的费用，缓存命中率，与预算对比 |
| `twin backup now` | 立即做一份加密备份（每天 4 点自动做） |
| `twin eval gate M1` | 里程碑门槛判定，M0 到 M5 |
| `twin timezone set Asia/Shanghai` | 回国后切换时区（微信里发 `/时区 北京` 也行） |
| `twin purge --all` | 她要求时一键彻底删除全部数据 |

## 隐私说明

- **数据只在本机。** 聊天记录、媒体、数据库、向量库、训练集和模型文件都在 `data\` 下，永远不进 git（`.gitignore` 与提交前隐私扫描守着）；测试只用合成数据。
- **加密。** 数据库里的消息正文、记忆、摘要、人设卡、机器人对话等敏感字段用 AES-256-GCM 加密，媒体与备份同样加密；主密钥存在 Windows 凭据管理器，不在任何文件里。DeepSeek Key、SMTP 密码、AutoDL 密码也只存凭据管理器。
- **发给 DeepSeek 的只有当次需要的少量片段，并且先脱敏**（手机号、邮箱、身份证号、银行卡号、详细地址、wxid 换成类型占位符），绝不整库上传。
- **训练数据上云时**只上传脱敏、加密的训练包（口令不落盘）；训练完执行清理脚本并提醒你到 AutoDL 控制台释放实例。
- **日志里没有聊天正文**（INFO 及以上；DEBUG 也先脱敏）；告警邮件和 Windows 通知里没有聊天内容，二维码只在本机弹出。
- **她知情并授权**；记录在 `consent.confirmed_at`。她要求时可以一键删除全部与她相关的数据（含训练集、模型和备份，并销毁密钥使残留备份无法解密）：`twin purge --all`。
- **机器人只对你一个人说话**，代码层面拒绝任何其他收件人；它自己的回复永远不进入风格样本、检索库和训练集。
- **紧急联系人**默认关闭；显式开启并填写邮箱后，只有检测到危机时才发一封固定模板的邮件（只有时间，没有任何聊天内容）。

## 局限与非目标

v1 不做：语音或视频通话、发送语音、生成她的照片、给你以外的任何人发消息、群聊。机器人做不到现实里的事（打电话、见面、转账……），你提出时它会用她的语气自然带过。主动消息能不能在窗口外送达取决于 ClawBot 的平台限制，要等真实的 M0 通道探针才知道（[`docs/RUNBOOK.md`](docs/RUNBOOK.md) 第 13 章）。

## 开发

```powershell
uv run ruff check . ; uv run ruff format --check .
uv run mypy src/twin
uv run pytest -q --cov=src/twin --cov-report=json
uv run python scripts/coverage_gate.py                 # 总体 >= 85%，每个子包 >= 75%
uv run python scripts/trace_check.py                   # 需求追踪（每个 R-xxx 有实现位置与测试）
uv run python scripts/privacy_scan.py                  # 提交前隐私扫描
uv run python scripts/stub_scan.py                     # src/ 里没有桩、占位和玩具实现
uv run python scripts/decisions_check.py               # DECISIONS 的编号、取舍表与引用的测试
powershell -File scripts/check.ps1 -Round 16           # 以上全部一键运行（Windows）
```

GitHub Actions 把这些检查拆成矩阵分片并行运行（Windows 5 片、Linux 3 片，覆盖率合并后再过同一道门槛）；结构、分片器 `scripts/shard_tests.py` 和排查见 [`docs/CI.md`](docs/CI.md)。

规则摘要（完整见 `CLAUDE.md`）：不留桩和玩具实现；时间一律带时区并通过 `twin.clock`；真实数据永远不进 git；日志 INFO 及以上不含正文；秘密只存凭据管理器；机器人自己的回复永远不进风格样本、检索库和训练集。

## 目录

```
src/twin/   config storage channel llm ingest profile stickers retrieval memory engine
            schedule commands learning ops serving training eval  cli.py app.py clock.py services.py
config/     config.example.yaml、lists/（AI 腔短语、承诺句式、危机关键词等词表）
docs/       SPEC / TRACEABILITY / DECISIONS / RUNBOOK / ARCHITECTURE / RELEASE_CHECKLIST /
            PENDING_USER_ACTIONS / CHANNEL_REPORT / LLM_REPORT / ILINK_PROTOCOL / SERVING_NOTES ...
prompts/    各轮提示词（只读）
training/   AutoDL 侧脚本与 LLaMA-Factory 配置模板
scripts/    trace_check.py privacy_scan.py stub_scan.py decisions_check.py coverage_gate.py shard_tests.py ...
scripts/windows/   install.ps1 uninstall.ps1 get_llamacpp.ps1
tests/      unit/ integration/ fixtures/（全部合成数据）support/（共享测试替身）
data/       运行时数据（gitignore）
```
