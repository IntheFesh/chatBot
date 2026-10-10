# 运维手册（RUNBOOK）

> 给仓库主人（你）看的：从零部署、日常运维、训练和部署风格模型、换时区、备份恢复、一键删除、出问题怎么办。
> 命令都在 **PowerShell**、**仓库根目录**里执行，前面的 `uv run` 表示用项目的虚拟环境（`install.ps1` 装好后也可以把 `.venv\Scripts` 放进 PATH 直接写 `twin`）。
> 每条命令都是真实存在的：`tests/unit/test_docs_commands.py` 逐条核对本文件里写出的 `twin` 命令（含选项）和微信里的指令，附录 A 的命令表与代码里的声明逐条比对。
> 需要你的真实环境才能完成的步骤（扫码、真实数据、GPU、观察期）在 [`PENDING_USER_ACTIONS.md`](PENDING_USER_ACTIONS.md) 里按轮次列出，第 13 章把它们汇总成 M0–M5 清单。

## 目录

0. [先读：命令的进程类别与“先停应用”规则](#0-先读命令的进程类别与先停应用规则)
1. [首次部署与服务管理](#1-首次部署与服务管理)
2. [微信扫码登录与绑定、掉线后重新登录](#2-微信扫码登录与绑定掉线后重新登录)
3. [导入全部聊天记录与以后的增量导入](#3-导入全部聊天记录与以后的增量导入)
4. [人设卡审阅与手改](#4-人设卡审阅与手改)
5. [试聊、微信里的指令与日常观察](#5-试聊微信里的指令与日常观察)
6. [训练风格模型（AutoDL）](#6-训练风格模型autodl)
7. [评估与门槛](#7-评估与门槛)
8. [部署风格模型](#8-部署风格模型)
9. [回国切换时区（以及切回美国）](#9-回国切换时区以及切回美国)
10. [备份、校验、恢复](#10-备份校验恢复)
11. [一键删除（她要求时）](#11-一键删除她要求时)
12. [常见故障排查](#12-常见故障排查)
13. [M0–M5 里程碑清单](#13-m0m5-里程碑清单)
- [附录 A：全部 CLI 命令与进程类别](#附录-a全部-cli-命令与进程类别)

---

## 0. 先读：命令的进程类别与“先停应用”规则

常驻的应用（`twin run`，由计划任务里的 `twin supervise` 拉起）和你手敲的命令同时存在。每个命令都声明了一个类别（R-ARCH-006），类别决定它能不能在应用运行时执行：

| 类别 | 做什么 | 应用运行时能执行吗 | 例子 |
| --- | --- | --- | --- |
| **只读** | 直接读库；任何写库都会被拒绝 | 能 | `twin health`、`twin profile show`、`twin cost report` |
| **轻量修改** | 短事务写库，并让运行中的应用在 2 秒内感知（重读人设卡、重建今天的计划、启停 `llama-server` 等） | 能 | `twin settings set`、`twin timezone set`、`twin persona edit`、`twin backup now` |
| **重任务** | 只把任务**排进队列**，由运行中的应用执行；应用没在运行时加 `--foreground`，或用 `twin jobs run --until-idle` 在前台执行 | 能（入队） | `twin import <目录>`、`twin profile rebuild`、`twin train export` |
| **独占** | 数据库迁移、恢复备份、一键删除、密钥轮换等：`run` 锁或 `supervisor` 锁被任何一个进程持有就拒绝执行，退出码 4 | **不能**，先 `twin service stop` | `twin backup restore`、`twin purge`、`twin db upgrade` |

**必须先停止应用的命令**（`twin service stop` 之后再执行，做完 `twin service start`）：

- `twin backup restore`、`twin purge`、`twin db upgrade`、`twin secrets rotate-db-key`、`twin service install`、`twin service uninstall`。

**与应用互斥的命令**（它们自己持有 `run` 锁，同一时间只能有一个在轮询微信或回复）：

- `twin run`、`twin chat`、`twin channel listen`、`twin channel echo-test`；`twin supervise` 持有另一把 `supervisor` 锁，不能开第二个。

退出码：0 成功；1 失败；2 用法错误；3 缺少 `consent.confirmed_at`（她的同意）；4 被锁拒绝；5 数据库版本过旧（先 `twin db upgrade`）；6 秘密缺失或密钥错误；7 配置错误。附录 A 列出每个命令的类别。

---

## 1. 首次部署与服务管理

### 1.1 开始之前

- 一台 Windows 10/11 x64 电脑，常开、关掉自动睡眠（设置 → 系统 → 电源 → 睡眠：从不）。`twin doctor` 的 `power-plan` 一行会提醒。
- 手机微信已升级到支持 ClawBot 的版本，“我 → 设置 → 插件”里能看到 ClawBot。
- 一个 DeepSeek API Key，账户里有余额（日常月费用目标不超过 15 美元；一次性批任务单批上限 30 美元，每批都要你确认）。
- 可选：能发邮件的 SMTP 账号（Gmail 用应用专用密码，QQ/163 用授权码）。

### 1.2 安装

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1
```

它依次做：检查并安装 uv → `uv sync --frozen` → `twin setup` 向导 → `twin db upgrade` → `twin service install`。想马上启动加 `-StartNow`；已经配置过加 `-SkipSetup`。卸载用 `powershell -ExecutionPolicy Bypass -File scripts\windows\uninstall.ps1`（停止并移除计划任务，数据不动）。

### 1.3 配置向导 `twin setup`

向导一次问一件事，已经有的答案会显示出来、不改就保留，所以可以重复运行：

```powershell
uv run twin setup
```

| 问什么 | 写到哪里 | 说明 |
| --- | --- | --- |
| 她同意的日期（consent） | `config\config.yaml` 的 `consent.confirmed_at` | 缺失或不是日期时，除 `twin doctor` 等少数命令外一律拒绝运行（退出码 3，R-SCOPE-003） |
| DeepSeek API Key | Windows 凭据管理器 `deepseek_api_key` | 输入时不回显，从不写入文件或日志 |
| 告警邮件（SMTP） | `ops.smtp.*`；密码进凭据管理器 `smtp_password` | 端口 465 用隐式 TLS，其余端口 STARTTLS；可发一封测试邮件 |
| 她的 wxid（目标会话） | `target.username` | 留空就在第一次导入时从非群聊会话里选 |
| 机器人时区 | `time.bot_timezone` | 默认 `America/Chicago`，回国后切到 `Asia/Shanghai`（第 9 章） |
| 紧急联系人（默认关） | `safety.emergency_contact` | 开启前屏幕上会原样显示将要发出的固定邮件（只有时间，没有任何聊天内容） |

不想用向导的话，等价的手动命令：

```powershell
uv run twin secrets set deepseek_api_key      # 粘贴 Key，不回显
uv run twin secrets set smtp_password         # 可选
uv run twin secrets list                      # 哪些秘密已设置（从不显示值）
copy config\config.example.yaml config\config.yaml   # 然后编辑；该文件不会被提交
uv run twin config show                       # 生效配置（秘密和个人 id 已打码）
uv run twin settings list                     # 存在数据库里的运行时设置
```

配置优先级：命令行 `--set 键.路径=值` > 环境变量（`TWIN_` 前缀，嵌套用 `__`）> `config\config.yaml` > 默认值。运行时可变的设置（时区、思考模式、后端、主动范围、暂停状态）存数据库，由指令或 `twin settings set` 修改，重启后保留。

### 1.4 体检与 DeepSeek 探针

```powershell
uv run twin doctor        # Python、依赖、时区库、凭据存储、磁盘、数据库、网络、计划任务、电源、显卡
uv run twin db status     # 数据库版本
uv run twin llm probe     # M0 探针：对真实 DeepSeek 做七项检查，写 docs\LLM_REPORT.md（约三十个很小的合成请求，费用远低于 0.1 美元）
uv run twin llm status    # 模型、探针学到的能力、预算级别与今日花费
```

- `twin doctor` 里 `keyring` 一行应是 `ok`，详情为 `WinVaultKeyring`；`database` 在首次 `twin db upgrade` 之前是 `warn`；`holiday-calendar` 是 `warn` 表示已安装的 `chinese-calendar` 还没收录下一年（不是故障，新版本发布后 `uv lock --upgrade-package chinese-calendar` 再 `uv sync`）。
- `twin llm probe` 通过的口径：思考开/关、缓存命中、JSON 输出（思考关闭与开启两种都要能解析）、JPEG 与 PNG 看图必须成功；其余是测量项。如果它输出 `STOP: JSON output with thinking enabled is not reliable`，不要继续往下做（主动消息规划依赖这一项），把 `docs\LLM_REPORT.md` 里第 4 项发给维护者。

### 1.5 服务管理

```powershell
uv run twin service install     # 注册计划任务（独占：先停止应用）；install.ps1 已经做过
uv run twin service status      # 计划任务、进程、最近一天的重启
uv run twin service start       # 立即启动
uv run twin service stop        # 优雅停止：先请 supervise 和 run 结束，正在发的那条回复发完才退出，宽限后才强制
uv run twin service uninstall   # 移除计划任务（独占；数据不动）
uv run twin health              # 通道、DeepSeek、风格模型、磁盘、任务队列、备份、预算
```

- 计划任务在你**登录 Windows 时**启动 `twin supervise`（登录类型 `InteractiveToken`：凭据管理器、通知和二维码窗口都要你的会话；不是 Windows 服务）。它以子进程运行 `twin run`，崩溃后按 5 秒到 5 分钟的指数退避重启，正常退出则结束。`twin service status` 应显示 `logon type: InteractiveToken`、`time limit: PT0S`、`second instance: IgnoreNew`。
- **断电或系统更新重启后想无人值守恢复**，需要开启 Windows 自动登录（`netplwiz` 或 Sysinternals Autologon）。风险：开机后任何接触这台电脑的人都直接进入你的账户，只在电脑放在可信的地方并开了 BitLocker 时才这么做；否则重启后要你手动登录一次。
- 调试时可以在窗口里前台运行：`uv run twin run`（Ctrl+C 优雅退出）。**不要和计划任务同时跑**：第二个会立刻报 `another 'run' instance is already running`（退出码 4）。
- 日志在 `data\logs\`：`twin.log`（应用）、`twin-cli.log`（命令）、`twin-supervise.log`（监督进程），每行一个 JSON，10 MB × 10 份轮转；INFO 及以上不含聊天正文。
- 告警（登录失效、DeepSeek 连续失败、预算、备份失败、磁盘不足、风格模型掉线……）同时走 Windows 通知和邮件，同类 1 小时最多 1 封；邮件里没有任何聊天内容，二维码只在本机弹窗里出现。

---

## 2. 微信扫码登录与绑定、掉线后重新登录

### 2.1 登录并绑定

```powershell
uv run twin channel login
```

终端会打印字符二维码和一个备用链接，同时把 PNG 存到 `data\tmp\` 并用默认图片查看器打开；用手机微信扫码并确认（遇到 “Enter the number shown on your phone” 时输入手机上显示的数字）。二维码约 5 分钟过期，会自动换新（最多 3 个）。登录成功后命令等你在 ClawBot 会话里给机器人发**一条消息**（最多等 10 分钟，`--wait-minutes` 可改），然后显示打码的发送者 id，问“是不是你”，输入 `y` 才绑定。这第一条消息本身不会被机器人处理。

- 如果终端警告 “NOT the account that scanned the login code”：给机器人发消息的账号和扫码的账号不是同一个，默认应该拒绝（输入 `bind` 以外的任何内容）。
- 绑定之后，机器人只处理这个账号的消息，出站接口传入其他收件人直接抛异常（R-CH-007）。名字和头像由你在 ClawBot 会话设置里自己改，代码不修改。

```powershell
uv run twin channel status               # 登录、绑定（id 打码）、context token、窗口剩余、条数、最近入站类型号
uv run twin channel send-test "你好"      # 手机上应收到“[测试]你好”（只会发给已绑定的你）
```

`twin channel listen`（应用没在运行时）把你发来的文字、图片、语音、视频、文件、引用、表情包各一条的**形态**打印出来（不显示内容），用来核对入站解析；Ctrl+C 结束。`twin run` 运行时它自己也会轮询并记录同样的信息，两者互斥。

### 2.2 掉线后重新登录

登录失效（协议错误码 `-14`）时：`twin channel status` 显示 `NEEDS RE-LOGIN`，`alerts` 表有一条 critical，同时你会收到邮件和 Windows 通知，应用在计划任务里运行时电脑上会弹出一个置顶的二维码小窗口（需要验证数字时窗口里有输入框）。手机扫码后自动恢复，并通知你“已恢复”。二维码不会出现在邮件或通知里。

窗口没弹出（或者你在远程）时手动重登，绑定保留：

```powershell
uv run twin channel login --force
```

换号（用另一个微信账号）要先解绑，会确认两次：`uv run twin channel unbind`，再 `uv run twin channel login`。

### 2.3 M0 通道探针（约 26 小时，需要手机配合）

主动消息能不能真的送达，取决于 ClawBot 在你发完消息之后还允许机器人推送多久、多少条。这些源码里读不出来，只能在你的会话里实测：

```powershell
uv run twin run                          # 窗口 A：探针由运行中的应用执行，保持开着，电脑不要睡眠
uv run twin channel probe start          # 窗口 B：读完说明后输入 y
uv run twin channel probe status         # 随时看进度和该做什么
uv run twin channel probe answer         # 回答“手机上实际收到几条”“GIF 在动吗”“看到正在输入吗”
uv run twin channel probe report         # 写 docs\CHANNEL_REPORT.md，并问你是否把建议值写入配置
uv run twin channel probe stop           # 中途放弃（已测到的保留）
```

每一步开始前探针会要你先给机器人发**一条**新消息；第 1 步每 2 分钟发一条直到失败或 15 条（以手机上实际收到的为准）；第 3 步约 25 小时内**不要给机器人发任何消息**，否则作废重做。探针发的消息都带 `[测试]` 前缀、图片是程序画的合成图。判定和读法见 [`CHANNEL_REPORT.md`](CHANNEL_REPORT.md)，第 13 章有 R-CH-010 的提醒。

---

## 3. 导入全部聊天记录与以后的增量导入

### 3.1 先看结构，再导入

```powershell
uv run twin import inspect "D:\你的导出目录"
uv run twin import "D:\你的导出目录"
uv run twin import status --watch
```

1. `twin import inspect` 只输出**结构**（文件树、每个 JSON 的键名与类型、`renderType` 取值计数、`_integrity/` 清单），**不含任何值**，写 `data\reports\inspect-<UTC>.md`。大导出可加 `--sample 20000`。目录名含中文没关系，加引号。
2. `twin import <目录>`（等同 `twin import start <目录>`）默认入队后立即返回，由运行中的应用执行；应用没在运行时加 `--foreground` 在本窗口里跑（带进度条）。第一次会列出非群聊会话让你选目标（`--target 2` 或 `--yes` 在只有一个候选时直接选），选择记入 `target.username`。群聊一律跳过，其他会话不读取内容。
3. `twin import status` 显示阶段、已处理条数、速度、预计剩余时间与各导入后钩子的进度。中断后 `twin import --resume` 从断点继续。结构化导入 100 万条目标 < 30 分钟（不含下载与 LLM 任务）。

导入报告（不含任何消息正文）在终端和 `data\reports\import-<UTC>.md`：按类型与发送方计数、日期范围、新增/重复/冲突、缺失媒体、表情包下载统计、风格指标变化。

### 3.2 导入之后自动排队的事，和单独的回填命令

导入完成会自动排队下面这些钩子；每个都有可以单独执行的命令（做过的不会重复花钱）：

| 钩子 | 回填命令 | 说明 |
| --- | --- | --- |
| 风格画像 + 作息模型（live 与 pre_holdout 两个范围） | `twin profile rebuild` | 结果与上一版相同时不写新版本 |
| 人设卡（统计规则立即刷新；描述增长 ≥ 10% 时排队重生成） | `twin persona regenerate` | `--stats-only` 只刷新统计规则（免费） |
| 表情包下载（导出里没有文件的） | `twin stickers download` | 并发 4、每秒 ≤ 4、重试 3 次，`--retry-failed` 重试不可用的 |
| 表情包打标签 | `twin stickers tag-all` | 一次性批任务，先估价 |
| 检索库（她在类似情况下的真实回复） | `twin retrieval rebuild` | 首次会下载向量模型到 `data\models\embeddings\`；可续跑 |
| 记忆回放（按日期顺序生成摘要与事实） | `twin memory replay estimate` 看估价，`twin memory replay start` 入队 | 一次性批任务，先估价；`twin memory replay status` 看进度 |
| 图片描述（默认最近 90 天） | `twin images caption-backfill` | 一次性批任务；用户新发来的图片在线同步描述 |
| 重训提醒检查 | `twin train retrain-check` | 她的新消息比上次训练数据多 10% 时告警 |

### 3.3 一次性批任务的费用确认

全量记忆回放、图片描述、表情包打标签、人设卡描述、hybrid 规划合成、评估生成都是**一次性批任务**（R-LLM-014）：入队时给出估算，**你确认之后才执行**，费用记入独立的 `one_time` 账目，单批上限 `budget.one_time_usd`（默认 30 美元），不计入日/月预算的降级判断；实际费用超过估算 20% 时该批次自己暂停并告警。

```powershell
uv run twin jobs list                          # 看队列；批次号在入队时打印，也在这里
uv run twin jobs approve <批次号>               # 确认提示；加 --yes 跳过
uv run twin jobs run --until-idle              # 应用没在运行时，在前台执行已批准的任务
uv run twin jobs show <任务号>                  # 一个任务的详情（加 --payload 看载荷）
uv run twin jobs retry <任务号>                 # 重新排队失败或取消的任务
uv run twin jobs cancel <任务号>
```

### 3.4 增量导入（以后）

以后想补充新记录：再导出一份，同样 `uv run twin import "D:\新的导出目录"`，或者在微信里发 `/导入 D:\新的导出目录`（先用 CLI 做过第一次导入，聊天里不能选目标会话）。按消息 id 幂等：重复的跳过，同 id 内容不同时以较新的导出为准并记录差异数。应用在运行时导入结束后她会用一条系统消息报一次条数。导入完成后画像、检索库、记忆回放等对新日期范围自动排队，并检查要不要提醒重训。

> 留出集切分点（评估和训练测试集用的最近 10%）由唯一函数 `holdout_cutoff()` 给出，**导入新数据不会自动移动它**。想把它移到今天数据的最新 10%：`uv run twin retrieval resplit`（评估结果前后不可比，会先问你）。

---

## 4. 人设卡审阅与手改

人设卡由统计规则 + DeepSeek 对分层抽样真实对话的归纳 + 你的手改 + 纠正规则组成，分四个区块：`[自动-统计规则]`、`[自动-描述]`、`[手动]`（你改的，重算时原样保留，分 `### 风格` 与 `### 事实` 两节）、`[不要这样]`（来自你的纠正，只描述说话方式）。

### 4.1 先核对画像和作息

```powershell
uv run twin profile show                 # 她与你的风格数字、数字风格规则、作息概览（当地时间）
uv run twin profile history              # 版本列表
uv run twin profile diff ~1 ~0           # 两版之间相差超过 10% 的指标
uv run twin profile phrases              # 她的高频整句、字组、称呼候选（真实文字，只在本机屏幕上）
uv run twin profile rollback <版本>       # 切回旧版本（连同当时的作息模型）
```

`profile show` 会请你**确认推断出的睡眠时段**。推断的睡眠核心如果落在当地白天（10:00–18:00），会有醒目警告和一条告警：多半是 `time.source_timezone`（导出里 `createTimeText` 的时区）设错了；她某段时间在国内时用 `time.source_timezone_ranges` 按日期区间指定，改完 `twin profile rebuild`。

作息不对时手动修正（手动修正优先于推断，持久化）：

```powershell
uv run twin routine add sleep 01:00 08:30                      # 睡眠区间；--days workday,weekend,holiday 限定日类型
uv run twin routine add busy 14:00 17:00 --weekdays mon-fri    # 每周忙碌时段
uv run twin routine add holiday 2026-10-01 2026-10-07          # 假期日期区间
uv run twin routine list                                       # 列出，带编号
uv run twin routine disable <编号>                              # 关掉但不删除；enable / remove 同理
uv run twin plan rebuild                                       # 按新作息重建今天剩下的计划
uv run twin plan show                                          # 今天的计划（起床、饭点、入睡、主动配额）
```

微信里等价的是 `/作息 睡 01:00-08:30`、`/作息 忙 周三 14:00-17:00`、`/作息 假期 10月1日..10月7日`、`/作息 查看`、`/作息 删除 1`。

### 4.2 人设卡

```powershell
uv run twin persona show                       # 生效中的 live 版本（Markdown）
uv run twin persona show --compact             # 风格模型用的精简版（≤ 400 token，只含风格，不含事实）
uv run twin persona show --full                # DeepSeek 后端用的完整版（≤ 1500 token）
uv run twin persona show --evidence            # 每条陈述背后的对话片段
uv run twin persona edit                       # 用系统默认编辑器打开 [手动] 区块，保存后加密入库
uv run twin persona history                    # 版本列表
uv run twin persona diff v2 v3                 # 两版的差别
uv run twin persona rollback v2                # 回到某一版（也可写 ~N 或 id 前缀）
uv run twin persona regenerate                 # 重新生成自动区块；描述部分排成一次性批任务，等你批准
uv run twin persona regenerate --stats-only    # 只刷新统计规则（免费）
uv run twin persona status                     # 哪个范围的自动描述该重写了
uv run twin persona templates                  # 提示词模板文件及生效版本
```

审阅时重点看：称呼、口头禅、不同情绪下怎么说、说话禁忌；**基本情况只应来自真实记录**，写得不对的在 `[手动]` 的 `### 事实` 里改；`### 风格` 里的内容会进入风格模型的精简提示词，事实只经记忆块进入。

### 4.3 纠正规则（`[不要这样]`）

`/不像`、`/重来` 之后每周会自己整理一次（也可以立刻做）；规则只描述说话方式，含日期或具体事件的会被校验丢弃（事实类纠正走 `/记住`）：

```powershell
uv run twin persona rules list                          # 带编号，最多 30 条
uv run twin persona rules delete 3                      # 删掉不对的
uv run twin persona rules consolidate --foreground      # 立刻整理（应用没在运行时；需要 Key，花一点费用）
```

### 4.4 版本回滚

画像、人设卡、提示词模板、风格模型都有版本，都可以回滚，回滚会写审计记录：

```powershell
uv run twin rollback profile <版本>
uv run twin rollback persona v2
uv run twin rollback prompt-template memory_extract 1
uv run twin rollback style-model <模型号>
```

---

## 5. 试聊、微信里的指令与日常观察

### 5.1 终端试聊

```powershell
uv run twin chat --local        # 与 twin run 互斥；需要 DeepSeek Key
```

她按真实节奏回复：你停下约 15 秒后她才开始，再等她的回复延迟（中位数十几秒，偶尔几分钟）、再一条条“打”出来——这不是卡住，没有秒回开关。想快一点试：`uv run twin --set engine.quiet_window_s=3 chat --local`（只缩短“等你说完”）。终端自带的 `/img <图片路径>`、`/quit` 由终端处理。没有微信、想跑完整应用时：`uv run twin --set channel.kind=console run`。

试这几件事：日常对话看气泡条数、长度、标点、表情包；问“你是不是 AI？”（她用自己的语气承认是模拟）；说“你给我打个电话吧”（自然带过，不答应）；发一句只有“好的”（偶尔不回）；危机表述（跳出角色、给出热线，不等她醒）。

### 5.2 微信里的指令

指令回复立刻出现，以 `⚙️` 开头，是系统口吻，不进入记忆、学习和训练。完整速查表在 [`README.md`](../README.md)，`/帮助 <指令>` 给出用法和例子。常用的：

- 看状态：`/状态`（时区与当地时间、她此刻的状态、后端、主动范围与今日已发、平台窗口剩余、今日费用与缓存命中率、最近告警、重训提醒）。
- 调节：`/思考 自动`、`/后端 deepseek`、`/暂停 2小时`、`/恢复`、`/主动 2-5`、`/主动 关`。
- 纠错与记忆：`/重来`、`/不像 那你早点睡吧`、`/记住 她下周三有考试`、`/忘掉 3`、`/记忆`。
- 反馈与费用：`/评分 4 晚安发得很自然`（M3 要求每周 ≥ 4/5，周末请务必评一次）、`/费用 本月`。
- 时区：`/时区 北京`（第 9 章）。

### 5.3 日常观察

```powershell
uv run twin health                       # 一屏看全
uv run twin proactive log --days 7       # 主动消息每次决定：类型、结果、原因、她的状态（不显示正文；--show-text 要确认两次）
uv run twin memory list                  # 她现在记得的事；--keyword 搜
uv run twin memory block "考试"          # 一个话题会取到的记忆块（判断召回得准不准）
uv run twin memory remember "她下周三有考试"
uv run twin memory forget 3
uv run twin plan lifeline                # 她今天的生活线
uv run twin cost report                  # 本月按日、用途、模型的费用、缓存命中率、高峰占比
uv run twin stickers list                # 表情包库，她最常用的在前
uv run twin settings history backend.fallback   # 某个设置谁在什么时候改过
```

观察期里每天至少给她发一条消息：平台只允许在你最近一条消息之后约 22 小时内主动发，过了窗口她就不发（日志里是“被窗口抑制”）。

---

## 6. 训练风格模型（AutoDL）

风格模型是在 AutoDL 单卡上用 LLaMA-Factory 对 Qwen3 做 LoRA 的**可切换生成后端**；训好之前默认只用 DeepSeek。流程：导出训练集 → 加密打包 → 上传 → 远程训练/评估/导出 → 下载 → 清理并释放实例。

### 6.1 前提

- 聊天记录已导入；`pre_holdout` 范围的画像、作息、人设卡已生成（`uv run twin profile rebuild`、`uv run twin persona regenerate`）；记忆回放覆盖要导出的日期范围（`uv run twin memory replay start` 并批准）。缺了哪个，`twin train export` 会直接告诉你先做什么。
- 想做 DPO：用 `/不像 <正确说法>` 攒偏好对，≥ 200 对才有意义（`training.dpo_min_pairs`）。

### 6.2 在 AutoDL 租卡并扩容数据盘

选 **RTX 5090（32 GB）** 或 **RTX PRO 6000（96 GB）**，镜像选 Python 3.11+、PyTorch 带 CUDA 12.8（脚本会检查 `torch >= 2.7` 且含 `sm_120`，不满足就从 cu128 索引重装；绝不接受 CUDA 13.x）。**默认 50 GB 数据盘不够，按档位扩容**：

| 档位 `--profile` | GPU | 基座 | 方法 | 数据盘至少 | 内存至少 |
| --- | --- | --- | --- | --- | --- |
| `5090-8b` | RTX 5090 | Qwen3-8B | LoRA bf16 | 70 GB | 17 GB |
| `5090-14b` | RTX 5090 | Qwen3-14B | QLoRA 4-bit | 110 GB | 43 GB（适配器在 CPU 上合并 bf16 基座） |
| `pro6000-14b` | RTX PRO 6000 | Qwen3-14B | LoRA bf16 | 110 GB | 23 GB |
| `pro6000-32b` | RTX PRO 6000 | Qwen3-32B | LoRA bf16 + 梯度检查点 | 230 GB | 41 GB |

不够时 `setup.sh` 会停下并提示到控制台扩容。**实例连续关机 15 天会被释放且数据清空**——训练前后的产物先下载到本地。按小时计费：做完立刻清理并释放。

### 6.3 填 `autodl.*` 与密码

在 `config\config.yaml` 里填（主机、端口来自控制台的 SSH 登录命令）：

```yaml
autodl:
  host: "connect.xxxx.seetacloud.com"
  port: 12345
  user: "root"
  auth: "password"          # 或 key（再填 key_path）
  workdir: "/root/autodl-tmp/twin"
```

```powershell
uv run twin secrets set autodl_password       # 实例密码进凭据管理器，不写任何文件
uv run twin train remote connect              # 登录、显示 GPU 和数据盘，并记住主机密钥指纹（对照控制台后输入 y）
```

实例重建后指纹会变，按提示删除 `data\training\known_hosts` 里那一行再连。SSH 用纯 Python 的 `asyncssh`，Windows 不需要装 OpenSSH。

### 6.4 导出训练集

```powershell
uv run twin train export --foreground        # 应用没在运行时；应用在运行就去掉 --foreground
uv run twin train export-status              # 样本数、丢弃原因、表情包占比、规划进度
uv run twin jobs approve <批次号>             # 第一次通常停在“规划合成”：确认估算费用后批准
uv run twin jobs run --until-idle            # 应用没在运行时执行已批准的批任务
uv run twin train export --foreground        # 再运行一次，这次写出数据集
```

每个样本以她的一个真实回复块为目标，输入是前面最多 8 个合并轮次，系统提示由与线上同一个 `StylePromptBuilder` 生成（精简人设卡 + 当时的时间与作息状态 + 当时已知的记忆），机器人自己的回复永远不进入。导出前脱敏；`--from 2026-01-01 --to 2026-06-30` 可限定日期范围；测试集是最近 10%（与留出集一致）。第一次导出会下载 Qwen3 的 `tokenizer.json`，两处都连不上时用 `--tokenizer <文件或目录>` 指定。

### 6.5 加密打包、上传与远程训练

```powershell
uv run twin train bundle --profile 5090-8b --dataset <数据集目录>     # 输入两次口令（≥ 12 个字符，不保存；丢了只能重新打包）
uv run twin train remote all --profile 5090-8b --dataset <数据集目录>   # 一条龙：上传、setup、训练、评估、导出、下载、清理
```

或逐步执行（每一步可单独执行与续跑；断网后重跑同一条命令即可，长步骤在实例上继续，日志接着拉，上传下载续传并校验 sha256）：

```powershell
uv run twin train remote upload --bundle <训练包>
uv run twin train remote setup          # 安装依赖、解密（要再输一次口令）、核对模板；模板一致性检查不过就停下，不要训练
uv run twin train remote train
uv run twin train remote dpo            # 偏好对够 200 对才有
uv run twin train remote eval           # 验证集 loss 与每个测试上下文的一条生成
uv run twin train remote export         # 合并 LoRA、转 GGUF、量化 Q4_K_M / Q5_K_M / Q8_0
uv run twin train remote download       # 下载到 data\models\<run_id>\ 并校验每个 sha256
uv run twin train remote cleanup        # 安全删除实例上的数据集、解密目录和日志里的样本
uv run twin train remote status         # 列出所有运行，并提醒还有数据留在实例上的
```

### 6.6 登记、清理、释放

```powershell
uv run twin model register data\models\<run_id>     # 校验产物并登记，锁定训练时的模板/人设卡/画像/数据集版本
uv run twin model list
```

`twin train remote cleanup` 之后（`twin train remote status` 不再提醒），**到 AutoDL 控制台释放实例**。`training_runs` 记录清理完成的时间；本地侧的 `twin purge --training-only` 会列出还没确认清理的远程训练。

### 6.7 继续训练与 DPO

她的新消息比上次训练数据多 10% 时会提醒（告警 + `/状态`），也可以 `uv run twin train retrain-check`。重训 = 重新 `twin train export`（新版本号）→ `twin train bundle` → `twin train remote all`，从基座全量训练，不在旧适配器上叠加。偏好对单独走 DPO：`uv run twin train export-dpo` 生成带 `dpo_train.jsonl` 的新数据集版本，再打包训练。

---

## 7. 评估与门槛

评估在**评估沙盒**里进行：与线上同一套提示词、后端和后处理，但用内存通道，不发微信、不写 `bot_turns` 等线上表；盲测与风格指标只通过 `AsOfView(t)` 读 pre_holdout 范围的派生数据，看不到未来。生成属于一次性批任务，先估价，你批准后才花钱。

### 7.1 盲测（M1 ≤ 70%，M4 ≤ 60%）

```powershell
uv run twin eval blind --backend deepseek --n 60       # 只抽样、估价、排队，不花钱；打印运行号与批次号
uv run twin jobs approve <批次号>
uv run twin eval blind --resume <运行号> --foreground   # 生成；全部生成后进入判分界面
```

`--n` 比 50 多一点：机器人没回出来、回退成其他后端的对子不会拿给你看，你跳过的也不计数，而门槛要求**最近一次盲测里有效判断 ≥ 50 对**。判分界面上面是同一段对话，下面左右两条候选（一条是她当时真实的回复），`1` 选左、`2` 选右、`s` 跳过、`q` 保存退出，随时 `--resume` 继续；只凭直觉，不要翻聊天记录。猜对率越接近 50% 越好；报告里有 Wilson 95% 区间、按时段和上下文长度分组的猜对率。多后端对比用 `--backend all`（登记了风格模型之后）。

### 7.2 风格指标

```powershell
uv run twin eval style --source live --days 7                              # 机器人最近 7 天真实输出 vs 她的 live 画像
uv run twin eval style --source eval_items --run <盲测运行号> --backend deepseek   # 留出集上下文的输出 vs pre_holdout 画像
```

六项核心指标（文字长度中位数、逗号率、连发中位数、表情包占比、表情代码率、引用率），**每项偏差在 ±30% 以内为通过**，有一项不通过退出码为 1。

### 7.3 记忆测试（M2 ≥ 80%）

```powershell
uv run twin eval memory                                       # 抽 20 题（10 题来自真实记录，10 题来自机器人对话），估价并排队
uv run twin jobs approve <批次号>
uv run twin eval memory --resume <运行号> --foreground         # DeepSeek 出题、机器人作答并初评；然后进入复核界面
```

复核时回车保持，`c` 正确、`p` 部分正确（0.5 分）、`w` 错误，`q` 保存退出；**你的判分覆盖 DeepSeek 的**。机器人对话来源的事实不足 10 条时记为“未通过（样本不足）”，不会拿真实记录凑数——先多聊几天，并用 `/记住` 记几件事。

### 7.4 主动消息审计（M3）

```powershell
uv run twin eval proactive --days 7        # 深睡时段 0 次、每天次数在范围内、间隔与追发零违规、边缘消息每周 ≤ 2、/评分 平均 ≥ 4
```

### 7.5 稳定性（M4）

```powershell
uv run twin eval stability --days 7        # 连续运行时长、重启次数与原因、长轮询中断、告警延迟
uv run twin ops drill network              # 打印断网演练的步骤（观察期内做一次：断网 15 分钟再恢复）
```

### 7.6 里程碑门槛

```powershell
uv run twin eval gate M1                   # 判定并把结果与证据写入 eval_runs；退出码 0 通过、1 未通过、2 尚未达到
uv run twin eval gate M1 --check           # 只读最近一次存下的结论，不重新判定
uv run twin eval runs --kind gate          # 评估与门槛记录
```

门槛的值来自 SPEC §26，**不得为了通过而降低或改统计口径**；没过就留在该阶段继续改进。M0–M5 的内容见第 13 章。

### 7.7 一致性、成本与汇总报告（第 15 轮）

下面三条命令是第 15 轮加的（每周一 04:20 当地时间还会自动排一次一致性审计，排在非高峰时段）：

```powershell
uv run twin eval consistency --days 7      # 每周一次的前后一致性审计：DeepSeek 找出生活线与机器人陈述的矛盾，你逐条确认；确认的明显矛盾 ≤ 1 次/周
#   --review 接着处理最近一次等你确认的审计（不再请求 DeepSeek）；--resume <运行号> 继续某一次；--queue 排到非高峰时段再跑
#   确认的矛盾会生成记忆修正建议，逐条确认后才写入；真实的、你说过的、/记住 的事实不会被改
uv run twin eval cost --month 2026-10      # 日常账目月费用 ≤ 15 美元；一次性批任务单列，不计入门槛
uv run twin eval report                    # 汇总 M0–M5 的门槛、各项度量和证据，写 data\reports\eval-<日期>.md（不含聊天正文）
```

---

## 8. 部署风格模型

训练好、登记了之后，要**先过上线门槛（M5）才允许设为默认**；通不过就保留 DeepSeek 后端（这不阻塞其他任何事）。

### 8.1 本机 llama.cpp（有 NVIDIA 显卡最好）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\get_llamacpp.ps1    # 下载固定版本的 llama.cpp，自动选 CUDA 或 CPU 构建、校验 sha256；-Build cpu 强制 CPU，-Force 重装
uv run twin doctor                          # 看 llamacpp 一行：装了没有、文件齐不齐、构建合不合这台电脑
uv run twin model recommend                 # 按显存推荐 Q8_0 / Q5_K_M / Q4_K_M（留 20% 余量）；--cpu 或 --vram 24GiB 可指定
uv run twin model list                      # 看模型号
uv run twin model show <模型号>              # 锁定的版本与评估数字
uv run twin model verify <模型号>            # 启动 llama-server 并做分词核对；不一致（常见是 GGUF 多了 BOS）就拒绝启用
uv run twin model serve <模型号>             # 前台运行服务器，打印首词元延迟和每秒词元数；Ctrl+C 结束（正式运行时应用自己启动和监视它）
```

`llama-server` 只监听本机（默认 `http://127.0.0.1:8081`），随主程序退出（作业对象保证应用被强杀后也不残留）。

### 8.2 远程推理（没有合适显卡时）

在 AutoDL 实例上 `bash training/autodl/serve_vllm.sh <档位>`（vLLM 加载基座 + LoRA，只监听实例本机）；本地在 `config\config.yaml` 里设 `style_model.mode: vllm_completion` 与 `style_model.model_id: twin-style`，再：

```powershell
uv run twin train remote connect          # 存好主机密钥
uv run twin model tunnel start            # 经 asyncssh 端口转发接入（没有应用时在本窗口保持隧道）
uv run twin model tunnel status           # 状态、重连次数、实例已运行多久、端口答不答
uv run twin model tunnel stop             # 用完关闭，然后到 AutoDL 控制台关机——实例按小时计费
```

隧道在线时每天（当地 `commands.morning_hour` 之后）会有一条系统消息提醒“实例在计费”，`/状态` 也会显示；不想要就把 `style_model.tunnel.remind_remote` 设为 `false`。

### 8.3 评估、激活门槛与回退

```powershell
uv run twin model evaluate <模型号> --n 80              # 从留出集抽 80 个新上下文，deepseek / style / hybrid 各生成一遍，估价
uv run twin jobs approve <批次号>
uv run twin model evaluate <模型号> --resume <运行号> --foreground   # 生成（这时才启动模型服务器）并让你判“哪条是她”
uv run twin eval gate M5                                # 判定上线门槛
uv run twin model activate <模型号>                      # 通过后：backend.active 设成猜对率更低的那种方式
uv run twin model disable <模型号>                       # 停用；运行中的应用几秒内跟上
```

上线门槛（R-SRV-005）：盲测里风格模型（style 或 hybrid 其中一种）的猜对率低于 deepseek 后端（单侧两比例检验 p < 0.1，每个后端各有 ≥ 50 对有效判断），**且**留出集六项风格指标都在 ±30% 以内。分词核对没过的模型任何情况下都不能激活。`--force` 能在门槛没过时强制启用，但会记审计、`/状态` 一直写“未通过门槛”、预算降级的最后一级不会切到它：

```powershell
uv run twin model activate <模型号> --force
```

微信里随时 `/后端 deepseek|style|hybrid` 切换。风格模型掉线时回复自动回退到 deepseek 并告警，连续健康满 10 分钟（`backend.recover_after_min`）自动切回并通知，每次切换在 `uv run twin settings history backend.fallback` 里有记录。

---

## 9. 回国切换时区（以及切回美国）

机器人时区默认 `America/Chicago`。你回国后切到 `Asia/Shanghai`；夏令时由 `zoneinfo` 自动处理（2026-11-01 芝加哥 UTC−5 → UTC−6 有测试）。**微信里发**：

```text
/时区 北京
```

或者在电脑上：

```powershell
uv run twin timezone set Asia/Shanghai
uv run twin timezone show              # 机器人时区、那里现在几点、她此刻的状态
uv run twin plan show                  # 新时区下今天剩下的计划
uv run twin timezone history           # 切换历史
```

切换时发生的事：从当前时刻起按新时区重建今天的日程；同样的钟点原样搬过去（芝加哥 8 点起床 → 北京 8 点起床）；**切换当天不重复起床问候**（距上次 ≥ 18 小时才允许）、**不跳过睡眠**（新时区若正处于睡眠区间则直接进入睡眠）。回复里会写明旧时区、新时区和当天剩余计划的变化。写错的时区名（`/时区 火星`）什么都不改，回用法。

要点：

- `time.source_timezone`（导出记录的时区，默认芝加哥）**不要**因为机器人换时区而改；它描述的是过去的记录。如果她的记录里有一段时间在国内，用 `time.source_timezone_ranges`（按日期区间）说明，然后 `uv run twin profile rebuild`。
- DeepSeek 的高峰/非高峰判定用北京时间的工作日，与机器人时区无关；`twin doctor` 的 `holiday-calendar` 提醒你 `chinese-calendar` 是否覆盖当年。
- 回美国时同样：`/时区 芝加哥`（或 `uv run twin timezone set America/Chicago`）。

---

## 10. 备份、校验、恢复

### 10.1 日常

每天当地 `ops.backup_hour_local`（默认 4 点）自动做一份一致性快照（SQLite 在线备份 API + 向量库 + 媒体清单），AES-GCM 加密写入 `data\backups\`；正在发送的会话最多顺延 `ops.backup_postpone_max_h`（2 小时）。保留 14 个日备份 + 8 个周备份。

```powershell
uv run twin backup now                  # 立即做一份
uv run twin backup list                 # 最新的在前
uv run twin backup verify <文件>         # 完整解密并核对内容（文件名或 data\backups 里的名字）
```

异地副本：把 `ops.backup_mirror_dir` 设成一个**已存在**的文件夹（外接盘或网络位置）。盘拔掉时本地备份照常成功，并告警“异地备份目录不可用”，插回后下一次备份恢复。备份超过 `ops.health.backup_stale_h`（36 小时）没有新的，`twin health` 会变色并告警。

### 10.2 恢复

```powershell
uv run twin service stop                # 恢复是独占命令：应用必须先停
uv run twin backup restore <文件>        # 先问一遍；会先把现有数据备份成 data\backups\pre-restore-<时间>.bak.enc（保留最新 3 个）
uv run twin service start
uv run twin retrieval rebuild           # 恢复后向量库可能和数据库差一点点，命令会列出问题和修复命令
```

密钥轮换后旧备份仍能恢复：`uv run twin secrets rotate-db-key`（独占，可中断续跑）把全部数据重加密并把旧密钥标为“已退役”，退役密钥保留在凭据管理器里，直到没有任何保留期内的备份引用它才删除。

> **一个必须知道的限制：** 数据库主密钥只存在这台电脑的 Windows 凭据管理器里（R-STO-003），代码里没有导出密钥的命令。重装 Windows、清空凭据或换电脑后，**旧备份无法解密**——这是加密设计的另一面。换电脑时请把原来的聊天导出重新导入新机器（机器人自己的对话记忆会丢失），或者在旧机器上保持凭据管理器完好。恢复演练见 [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)。

---

## 11. 一键删除（她要求时）

```powershell
uv run twin service stop
uv run twin purge --all
```

先列出将删除的内容与数量（数据库、媒体、向量库、备份与异地副本、训练集、本地模型与适配器、报告、日志），然后要你**键入**确认短语 `删除她的全部数据` 才执行；没有跳过确认的参数。同时删除凭据管理器里的全部数据库密钥（含已退役的）与备份密钥——加密粉碎，使任何残留备份都无法再解密。它不碰你的导出目录（`paths.export_dir`）和配置文件；AutoDL 上如果还有实例，请到控制台确认已释放（`twin purge` 会列出还没确认清理的远程训练）。

只想删训练相关的（训练集、训练包、本地风格模型文件），保留其余：`uv run twin purge --training-only`（确认短语是 `删除训练数据`）。

删除之后要重新使用必须从头来：`uv run twin db upgrade`、`uv run twin setup`、重新登录和导入。

---

## 12. 常见故障排查

先看 `uv run twin health` 和 `/状态`，再按症状找：

### 12.1 DeepSeek 余额不足（或 Key 失效）

- **症状**：收到“DeepSeek 连续出错”的邮件或通知；`twin health` 里 DeepSeek 一行变色；`twin doctor` 的 `deepseek-net` 显示余额不可用；微信里她用极短的自然回应兜底（永远不会把报错发给你）。HTTP 402 是余额耗尽，401/403 是 Key 被拒绝。
- **怎么办**：到 DeepSeek 控制台充值，或者重设 Key：`uv run twin secrets set deepseek_api_key`；`uv run twin llm status` 看预算级别和今日花费。恢复后下一次调用成功即自动继续；连续失败触发的熔断 5 分钟后自动解除。花费本身超预算时不是故障：会按顺序降级（关思考 → 减例子与记忆 → 暂停主动消息 → 最小上下文或风格后端），但永远不停止回复；`uv run twin cost report` 看钱花在哪。

### 12.2 微信掉线

- **症状**：收到“微信登录失效”（critical）或“微信长轮询中断”；`twin channel status` 显示 `NEEDS RE-LOGIN` 或轮询长时间没有成功。
- **怎么办**：登录失效 → 电脑上弹出二维码窗口，用手机扫码即自动恢复；没有弹窗就 `uv run twin channel login --force`（第 2.2 节）。轮询中断（多半是电脑断网）→ 检查网络，恢复后自动继续并通知你已恢复；掉线应在 10 分钟内收到提醒。`twin channel send-test "你好"` 验证发送。

### 12.3 主动消息被窗口抑制

- **症状**：她一整天没主动开口；`uv run twin proactive log --days 7` 里结果是“被拒”，原因写着窗口过期或条数不足；`/状态` 的“平台窗口”一行显示“已过期（等你下一条消息就会重新开始）”或剩余条数为 0。
- **原因与办法**：平台只允许在你最近一条消息之后约 `channel.proactive_window_safe_h`（22 小时）内、且剩余条数够时主动发；遇到会话过期的错误就停止主动直到下一条入站，**不会**循环重试，也不会去试“激活过期会话”之类的办法。**给她发一条消息窗口就重新开始。** 这一天算免责，但发不满条数。如果几乎每天都被压到下限以下，说明平台限制比预期严——回看 `docs\CHANNEL_REPORT.md` 的实测数字，并读第 13 章的 R-CH-010 提醒。

### 12.4 风格模型掉线

- **症状**：`/状态` 的后端一行写着“hybrid（风格模型不可用，眼下由 deepseek 回复：…）”；收到“风格模型不可用”告警；`twin health` 的 `style_serving` 一行变色。
- **怎么办**：不需要你立刻处理——回复已自动回退到 DeepSeek，模型恢复并连续健康满 10 分钟后自动切回并通知。本机推理：看 `llama-server` 是否在（应用会在几秒内自己重启它；连续崩溃时 `/状态` 会写“崩溃后等待重启”），`uv run twin model verify <模型号>` 复查分词；远程推理：`uv run twin model tunnel status`，实例是否还开着、`vLLM` 是否还在。急着用稳定的就 `/后端 deepseek`。分词核对不一致（`style_tokenize_mismatch`）说明模型与提示词模板对不上，不能启用。

### 12.5 磁盘满

- **症状**：“磁盘剩余空间不足”告警（剩余 < `ops.health.disk_min_gb`，默认 5 GB）；`twin health` 的 disk 一行变色；备份失败。SQLite 写入失败时应用会告警并以延迟回复降级，不会崩溃。
- **怎么办**：数据库、媒体、向量库、备份都在 `data\`。先腾出空间再重启：清理 `data\models\<run_id>\` 里不用的旧训练产物与 `tools\llama.cpp\` 里的旧版本，调小 `ops.backup_keep_daily` / `ops.backup_keep_weekly`，或把 `ops.backup_mirror_dir` 指到别的盘；**不要手动删除 `data\twin.db`、`data\media\` 或 `data\vectors\`**。腾出空间后 `uv run twin backup now` 做一份新备份，`uv run twin health` 确认恢复。

### 12.6 其他

| 症状 | 原因 / 办法 |
| --- | --- |
| `another 'run' instance is already running`（退出码 4） | 计划任务里的应用已经在跑；`uv run twin service status` 看，要手动运行先 `uv run twin service stop` |
| 命令说“needs exclusive access … stop the application first” | 独占命令（第 0 章）：`uv run twin service stop`，做完 `uv run twin service start` |
| 退出码 3 | 缺少 `consent.confirmed_at`，用 `uv run twin setup` 补上她同意的日期 |
| 退出码 5 / “schema outdated” | `uv run twin db upgrade`（先停应用） |
| 退出码 6 | 秘密缺失：`uv run twin secrets list` 看哪个没设，`uv run twin secrets set <名字>` |
| 电脑睡眠或重启之后 | 不补发错过的主动消息；重建当天计划；通道重连；睡眠期间你发来的消息会带延迟地回复。`twin doctor` 的 `power-plan` 提醒关掉自动睡眠 |
| 睡眠时段落在白天的警告 | 多半是 `time.source_timezone` 设错（第 4.1 节）；用 `/作息` 或 `uv run twin routine add sleep` 修正 |
| 邮件收不到 | `uv run twin setup` 重走 SMTP 一步并发测试邮件；Gmail 要应用专用密码，QQ/163 要授权码 |
| 导入很慢或卡住 | `uv run twin import status` 看阶段；`uv run twin import --resume` 续传；表情包下载失败不阻塞导入，`uv run twin stickers download --retry-failed` 重试 |
| 任务队列积压 | `uv run twin jobs list --status failed` 看失败的任务；`uv run twin jobs retry <任务号>`；DeepSeek 不可用时离线任务会等 |
| 一次性批任务暂停 | 实际费用超过估算 20%：检查后 `uv run twin jobs approve <批次号>` 重新批准，或取消 |
| 想让运行中的应用重读人设卡等 | 不用重启：轻量修改命令写库后应用 2 秒内感知 |

---

## 13. M0–M5 里程碑清单

每个里程碑过关后再进下一个；`uv run twin eval gate M<n>` 读取最新评估结果判定（0 通过、1 未通过、2 尚未达到），`--check` 只读已存结论。**沙箱里没有真实数据、微信、Key 和 GPU，下面每一项都还没有在真实环境里执行过，状态都是“待用户执行”**——没有任何探针结果、盲测结果或门槛记录是伪造的。每一项的详细做法按轮次写在 [`PENDING_USER_ACTIONS.md`](PENDING_USER_ACTIONS.md)。

> ## 注意：R-CH-010 风险——ClawBot 能不能在窗口外主动推送，要等真实的 M0 探针才知道
>
> 主动消息（起床问候、跟进、晚安）依赖“你发完消息之后，机器人还能在多长时间、多少条之内主动发”。**这是 ClawBot 平台的限制，源码里读不出来，社区报告也互相矛盾**（窗口过期、速率限制、每次回复条数上限）。通道探针（`twin channel probe start`）在你的会话里实测：窗口 < 12 小时，或一次入站后手机上收到的连发条数 < 3，报告会写“**未达标**”并停止（`probe report` 退出码 1）。
>
> 此时 M3（有作息、会主动）在 ClawBot 上**达不成**：按 R-CH-010 必须停下并向维护者报告，企业微信通道不在本规格范围、要你确认后另开一轮；**不要**自己去试“激活过期会话”之类的办法。在拿到真实探针结果之前，不要假设主动消息一定能送达。

| 里程碑 | 内容 | 过关条件 | 怎么做 | 状态 |
| --- | --- | --- | --- | --- |
| **M0 技术验证** | ClawBot 收发、主动窗口与条数、图片/GIF、正在输入；DeepSeek 思考开关与缓存命中 | 通道探针：主动消息在窗口内稳定送达（窗口 ≥ 12 小时、连发 ≥ 3 条）；DeepSeek 探针按 R-LLM-013 口径通过；否则停下评估企业微信通道 | `twin llm probe`（第 1.4 节）、`twin channel probe start`…`report`（第 2.3 节）、`twin eval gate M0`；见 `PENDING_USER_ACTIONS.md` 第 01、02 轮 | 待用户执行 |
| **M1 能聊、像她** | 导入、画像、生成与节奏、表情代码 | 盲测猜对率 ≤ 70%（点估计，有效判断 ≥ 50 对） | 第 3、4、5、7.1 节；`twin eval gate M1`；第 09b 轮 | 待用户执行 |
| **M2 记得住** | 表情包库、记忆、指令 | 记忆测试 ≥ 80%（20 题，10+10 构成完整） | 第 7.3 节；`twin eval gate M2`；第 09b、11 轮（指令全集完成后复判） | 待用户执行 |
| **M3 有作息、会主动** | 作息与真实模式、主动消息、思考开关 | **连续 7 天**真实运行：深睡时段主动 0 次、每天次数在范围内、间隔与追发零违规；该周 `/评分` ≥ 4/5 | 先确认 R-CH-010（上面的提示）；`twin run` 跑满 7 天；每天给她发一条；`twin eval proactive --days 7`、`twin eval gate M3`；第 10 轮 | 待用户执行 |
| **M4 持续成长** | 学习、增量导入、运维 | **无人值守 7 天**（全部由计划任务启动，期间做一次断网演练，学习和增量导入各成功一次）；**新的**盲测 ≤ 60% | 第 1.5、7.1、7.5 节；`twin eval stability --days 7`、`twin eval gate M4`；第 12 轮 | 待用户执行 |
| **M5 风格模型** | 训练、评估、量化部署 | R-SRV-005（第 8.3 节）；不通过则保留 DeepSeek 后端（不阻塞其他任何事） | 第 6、8 章；`twin model evaluate`、`twin eval gate M5`；第 13、14 轮 | 待用户执行（若训练） |

汇总所有门槛、各项度量和证据来源的报告：`twin eval report`（第 15 轮，见第 7.7 节）；打 `v1.0.0` 之前的完整检查在 [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md)。

---

## 附录 A：全部 CLI 命令与进程类别

类别含义见第 0 章；“说明”是命令 `--help` 的第一行。`tests/unit/test_docs_commands.py` 把本表与代码里的声明逐条比对（命令集合和类别都必须一致）。

| 命令 | 进程类别 | 说明 |
| --- | --- | --- |
| `twin backup list` | 只读 | The backups, newest first. |
| `twin backup now` | 轻量修改 | Make a backup now. |
| `twin backup restore` | 独占（先停止应用） | Replace the data with a backup (the application must be stopped). |
| `twin backup verify` | 只读 | Decrypt a backup completely and check what is in it. |
| `twin channel echo-test` | 独占（持有 `run` 锁） | Send back what you write, prefixed '[测试]' - a check of the channel itself. |
| `twin channel listen` | 独占（持有 `run` 锁） | Receive the bound user's messages and print what kind they are (never what they say). |
| `twin channel login` | 轻量修改 | Scan the QR code with WeChat, then bind the account that writes to the bot first. |
| `twin channel probe answer` | 轻量修改 | Answer the probe's questions (what arrived on the phone, whether the GIF moves). |
| `twin channel probe report` | 轻量修改 | Write docs/CHANNEL_REPORT.md from the probe result and offer the measured values. |
| `twin channel probe start` | 轻量修改 | Store a new probe plan; the running application carries it out (about 26 hours). |
| `twin channel probe status` | 只读 | Where the probe is, what it waits for, and what it has measured so far. |
| `twin channel probe stop` | 轻量修改 | End the running probe now (what was measured so far is kept). |
| `twin channel send-test` | 轻量修改 | Send one test message to the bound user (and nobody else). |
| `twin channel status` | 只读 | Login, binding, window and quota, and the item types seen recently (no content). |
| `twin channel unbind` | 轻量修改 | Stop talking to the bound user (asks twice). |
| `twin chat` | 独占（持有 `run` 锁） | Chat with her in the terminal: the whole application runs, with the terminal as the chat. |
| `twin config show` | 只读 | Print the effective configuration (secrets and personal ids masked). |
| `twin cost report` | 只读 | The month by day, purpose and model, cache hits, peak share, against the budget. |
| `twin db status` | 只读 | Show the applied and the latest schema revision. |
| `twin db upgrade` | 独占（先停止应用） | Create or migrate the database to the latest schema (stop the application first). |
| `twin doctor` | 只读 | Check the installation: Python, dependencies, time zones, keyring, disk, database. |
| `twin eval blind` | 轻量修改 | Blind test: pick which of two replies is hers (R-EVAL-001). |
| `twin eval consistency` | 轻量修改 | Consistency audit: DeepSeek lists contradictions, you decide which are real (R-EVAL-004). |
| `twin eval cost` | 轻量修改 | Month cost of the daily account against the 15 dollar line; one-time batches are listed apart (R-EVAL-007). |
| `twin eval gate` | 轻量修改 | Judge a milestone; exit code 0 passed, 1 not passed, 2 not reached yet (R-EVAL-010). |
| `twin eval memory` | 轻量修改 | Memory test: twenty questions from the fact store, judged and reviewed (R-EVAL-003). |
| `twin eval proactive` | 轻量修改 | Audit the proactive messages of the last days (R-EVAL-005); exit 1 if not compliant. |
| `twin eval report` | 轻量修改 | The summary report of the stored M0–M5 and measure results (R-EVAL-008). |
| `twin eval runs` | 只读 | List the recent evaluation runs. |
| `twin eval stability` | 轻量修改 | The stability report of the last days, from the health snapshots (R-EVAL-006). |
| `twin eval style` | 轻量修改 | Style metrics of the bot against her profile, within +-30 % (R-EVAL-002). |
| `twin health` | 只读 | How the bot is doing: channel, DeepSeek, style model, disk, queue, backup, budget. |
| `twin images caption-backfill` | 重任务（入队，由运行中的应用执行） | Queue picture descriptions as a one-time batch; approve it with `twin jobs approve`. |
| `twin import inspect` | 只读 | Describe the structure of an export without printing any value (R-IMP-014). |
| `twin import start` | 重任务（入队，由运行中的应用执行） | Import an export: queue the import job (default) or run it here (--foreground). |
| `twin import status` | 只读 | Show the phase, progress, speed, remaining time and hook states of the latest import. |
| `twin jobs approve` | 轻量修改 | Approve a one-time batch of jobs after reviewing its cost estimate (R-LLM-014). |
| `twin jobs cancel` | 轻量修改 | Cancel a pending or running job. |
| `twin jobs list` | 只读 | List jobs, newest first. |
| `twin jobs retry` | 轻量修改 | Re-queue a failed or cancelled job with a fresh attempt budget. |
| `twin jobs run` | 重任务（入队，由运行中的应用执行） | Execute queued jobs here (used when the application is not running). |
| `twin jobs show` | 只读 | Show one job (the payload is hidden unless --payload is given). |
| `twin llm probe` | 轻量修改 | Run the M0 checks against the real DeepSeek API and write docs/LLM_REPORT.md. |
| `twin llm status` | 只读 | Show models, learned capabilities, the last probe and the budget level. |
| `twin memory block` | 只读 | Show the memory block a reply about ``topic`` would get (to judge what is recalled). |
| `twin memory forget` | 轻量修改 | Forget a fact for good, with what was derived only from it. |
| `twin memory list` | 只读 | List what the bot remembers (current facts, newest first). |
| `twin memory reindex` | 轻量修改 | Encode every fact and summary again (after the embedding model changed). |
| `twin memory remember` | 轻量修改 | Remember something (source: user command, the highest rank). |
| `twin memory replay estimate` | 只读 | Show which days a replay would do and what it would cost (an upper bound; nothing runs). |
| `twin memory replay start` | 重任务（入队，由运行中的应用执行） | Queue the replay of the days that need it; it waits for `twin jobs approve <batch>`. |
| `twin memory replay status` | 只读 | Show how far the replay is: days done, batches, jobs and what the memory holds. |
| `twin memory summarize` | 重任务（入队，由运行中的应用执行） | Queue the summary of one day (off peak). |
| `twin model activate` | 轻量修改 | Make a model the one in use, if it passed the release gate (R-SRV-005). |
| `twin model disable` | 轻量修改 | Stop using a model; with no other active model the default is DeepSeek again. |
| `twin model evaluate` | 轻量修改 | Blind test of DeepSeek against a model (as style and as hybrid) on new contexts (R-SRV-005). |
| `twin model list` | 只读 | List the registered models, newest first. |
| `twin model recommend` | 只读 | Which quantisation (Q8_0, Q5_K_M, Q4_K_M) the graphics card can hold (R-SRV-002). |
| `twin model register` | 轻量修改 | Verify an artifact directory and register its models with their locked versions. |
| `twin model serve` | 轻量修改 | Run llama-server for a model in this window (the application does it by itself). |
| `twin model show` | 只读 | Show one registered model with its locked versions and evaluation numbers. |
| `twin model tunnel start` | 轻量修改 | Keep the SSH tunnel to the instance up (the application does it itself when it runs). |
| `twin model tunnel status` | 只读 | Whether the tunnel is wanted and up, how often it reconnected, how long the instance runs. |
| `twin model tunnel stop` | 轻量修改 | Close the tunnel and stop asking for it; then shut the instance down in the console. |
| `twin model verify` | 轻量修改 | Start the model's server and compare its tokens with the training tokenizer (R-TRN-011). |
| `twin ops drill network` | 只读 | Print the steps of the network drill. |
| `twin persona diff` | 只读 | Show what differs between two versions of the card. |
| `twin persona edit` | 轻量修改 | Edit the [手动] section of the live card in the system editor. |
| `twin persona history` | 只读 | List the stored versions of the card, newest first. |
| `twin persona regenerate` | 重任务（入队，由运行中的应用执行） | Regenerate the automatic part of the card; the description waits for your approval. |
| `twin persona rollback` | 轻量修改 | Make an older version of the card the one in force. |
| `twin persona rules consolidate` | 重任务（入队，由运行中的应用执行） | Queue the consolidation of the rules now (it also runs once a week by itself). |
| `twin persona rules delete` | 轻量修改 | Delete one rule (a new version of the live card; only [不要这样] changes). |
| `twin persona rules list` | 只读 | List the rules of [不要这样] with their numbers. |
| `twin persona show` | 只读 | Show a persona card (Markdown), its renderings, or the evidence of its statements. |
| `twin persona status` | 只读 | Show for each scope whether the automatic description is due to be written again. |
| `twin persona templates` | 只读 | List the prompt template files and which version of each is in force. |
| `twin plan lifeline` | 只读 | The life line of a day: what she does, where, in which mood (R-MEM-005). |
| `twin plan lifeline-generate` | 重任务（入队，由运行中的应用执行） | Queue the life line of a day (it replaces the planned entries of that day). |
| `twin plan rebuild` | 轻量修改 | Make today's plan again from now on (after a changed routine, holiday or range). |
| `twin plan show` | 只读 | The plan of a local day in the clock times of the bot's zone. |
| `twin proactive log` | 只读 | List the proactive scheduler's decisions of the last days (no words unless asked). |
| `twin profile diff` | 只读 | List the metrics that differ by more than 10 % between two versions. |
| `twin profile history` | 只读 | List the stored profile versions, newest first. |
| `twin profile phrases` | 只读 | Show her frequent sentences, n-grams and form-of-address candidates (real text, local). |
| `twin profile rebuild` | 重任务（入队，由运行中的应用执行） | Recompute the style profile and the routine model (queued as a job). |
| `twin profile rollback` | 轻量修改 | Make an older version (and the routine model computed with it) the active one. |
| `twin profile show` | 只读 | Show the style numbers and the routine overview (local time); confirm the sleep time. |
| `twin purge` | 独占（先停止应用） | Delete her data for good: database, media, vectors, backups, training, models, keys. |
| `twin retrieval rebuild` | 重任务（入队，由运行中的应用执行） | Build or repair the library of her real replies (queued as a job; resumes if stopped). |
| `twin retrieval resplit` | 重任务（入队，由运行中的应用执行） | Move the hold-out cutoff to the newest 10 % of today's data (evaluation changes). |
| `twin retrieval stats` | 只读 | Show the size of the library, the hold-out, the model and the progress of the last run. |
| `twin rollback persona` | 轻量修改 | Make an older persona card the one in force. |
| `twin rollback profile` | 轻量修改 | Make an older statistical profile (and its routine model) the active one. |
| `twin rollback prompt-template` | 轻量修改 | Make an older version of a prompt template the one in force. |
| `twin rollback style-model` | 轻量修改 | Make an earlier registered style model the one in use. |
| `twin routine add busy` | 轻量修改 | Mark a weekly busy period. |
| `twin routine add holiday` | 轻量修改 | Mark a range of dates as holidays (takes effect for learning at the next rebuild). |
| `twin routine add sleep` | 轻量修改 | Set the sleep interval (wins over the inferred one). |
| `twin routine disable` | 轻量修改 | Switch a correction off without deleting it. |
| `twin routine enable` | 轻量修改 | Switch a correction on again. |
| `twin routine list` | 只读 | List the manual routine corrections. |
| `twin routine remove` | 轻量修改 | Delete a correction. |
| `twin run` | 独占（持有 `run` 锁） | Start the application: channel, schedule, reply engine, job queue, state watcher. |
| `twin secrets check` | 只读 | Exit 0 if the secret is set, 1 if not (the value is never printed). |
| `twin secrets delete` | 轻量修改 | Remove a secret from the credential store. |
| `twin secrets list` | 只读 | List secret names and whether they are set (values are never shown). |
| `twin secrets rotate-db-key` | 独占（先停止应用） | Re-encrypt all data with a new database key (resumable; stop the application first). |
| `twin secrets set` | 轻量修改 | Store a secret in the credential store (never echoed, never logged). |
| `twin service install` | 独占（先停止应用） | Register the scheduled task that starts the bot when you log on. |
| `twin service start` | 轻量修改 | Start the scheduled task now. |
| `twin service status` | 只读 | The scheduled task, the processes, the restarts of the last day. |
| `twin service stop` | 轻量修改 | Stop the supervisor and `twin run`: the reply being sent is finished first. |
| `twin service uninstall` | 独占（先停止应用） | Remove the scheduled task (the data stays; `twin purge` deletes data). |
| `twin settings history` | 只读 | Show who changed a setting, when, and from what to what. |
| `twin settings list` | 只读 | List runtime settings with their current values. |
| `twin settings set` | 轻量修改 | Change a runtime setting; a running application picks it up within 2 seconds. |
| `twin setup` | 轻量修改 | First-run wizard: consent, DeepSeek key, e-mail alerts, conversation, time zone. |
| `twin stickers disable` | 轻量修改 | Never choose this sticker (it stays in the library). |
| `twin stickers download` | 重任务（入队，由运行中的应用执行） | Download the stickers whose files are not in the export (queued as a job). |
| `twin stickers enable` | 轻量修改 | Allow a disabled sticker again. |
| `twin stickers list` | 只读 | List the library, her most used stickers first, with the library statistics. |
| `twin stickers show` | 只读 | Show everything the library knows about one sticker. |
| `twin stickers tag` | 轻量修改 | Set the tags of a sticker by hand (they win over the automatic ones). |
| `twin stickers tag-all` | 重任务（入队，由运行中的应用执行） | Queue the tagging of every sticker that needs it; the price waits for your approval. |
| `twin stickers untag` | 轻量修改 | Remove the hand-set tags; the automatic ones apply again. |
| `twin supervise` | 独占（持有 `supervisor` 锁） | Keep ``twin run`` running: restart it when it crashes (the scheduled task runs this). |
| `twin timezone history` | 只读 | The switches of the bot's time zone, newest first. |
| `twin timezone set` | 轻量修改 | Switch the bot to another time zone and plan the rest of the day there (R-SCH-002). |
| `twin timezone show` | 只读 | The bot's time zone, the time there and her state now. |
| `twin train bundle` | 轻量修改 | Build the encrypted training package for a GPU profile (asks for a passphrase). |
| `twin train export` | 重任务（入队，由运行中的应用执行） | Export the training set from her real reply blocks (queued as a job, resumable). |
| `twin train export-dpo` | 轻量修改 | Write the preference pairs of /不像 as the DPO file of a new dataset version. |
| `twin train export-status` | 只读 | Show how the last export ended and how far the plans of the hybrid share are. |
| `twin train remote all` | 轻量修改 | Upload, set up, train, evaluate, export, download and clean up in one go. |
| `twin train remote cleanup` | 轻量修改 | Erase the data on the instance (after the download); remember the time. |
| `twin train remote connect` | 轻量修改 | Log in to the instance, show the GPU and the data disk, and remember its host key. |
| `twin train remote download` | 轻量修改 | Download the model files to data/models/<run_id>/ and verify every sha256. |
| `twin train remote dpo` | 轻量修改 | Preference training on the SFT adapter (only with enough pairs). |
| `twin train remote eval` | 轻量修改 | Validation loss and one generated reply per test context. |
| `twin train remote export` | 轻量修改 | Merge the adapter, convert to GGUF and quantise. |
| `twin train remote setup` | 轻量修改 | Install the packages, decrypt the package and verify the template. |
| `twin train remote status` | 只读 | List the training runs and remind of the ones that still have data on an instance. |
| `twin train remote train` | 轻量修改 | Fine-tune the style model (continues from the last checkpoint). |
| `twin train remote upload` | 轻量修改 | Upload the scripts and the encrypted package (resumes after a broken connection). |
| `twin train retrain-check` | 轻量修改 | Compare her messages now with the last training; raise the alert when it is time. |
