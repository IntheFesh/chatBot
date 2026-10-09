# 需要用户手动执行的步骤（PENDING_USER_ACTIONS）

> 云端沙箱里没有 Windows、GPU、DeepSeek Key、微信与真实聊天记录。凡是需要这些东西的验证，代码和测试已经写好，真实执行留给你。
> 每轮一节：**用户需要做什么**与**预期结果**。没有真实运行过的结果一律不在这里（也不在任何报告里）编造。

## 第 00 轮 —— 工程骨架、配置、加密存储、任务队列、质量门禁

在你的 Windows 10/11 x64 电脑上（PowerShell，仓库根目录）：

1. **安装依赖并跑全部检查**
   - 做什么：安装 [uv](https://docs.astral.sh/uv/) 后运行 `powershell -ExecutionPolicy Bypass -File scripts/check.ps1 -Round 00`。
   - 预期：每一步都显示 `==> ...`，最后输出 `All checks passed.`。其中 pytest 会真正执行 3 个 `windows` 标记的测试（真实命名互斥体、真实 `SetThreadExecutionState`、真实 `WindowsPowerManager`），在沙箱里这 3 个被跳过。
   - 如果 CI 在 GitHub 上启用，也可以只看 `windows-latest` 任务是否全绿（`.github/workflows/ci.yml`）。

2. **确认 Windows 凭据管理器被使用**
   - 做什么：`uv run twin doctor`。
   - 预期：`keyring` 一行是 `ok`，详情里是 `WinVaultKeyring: read/write ok`（**不是** `warn` / "encrypted file fallback"；那个回退只在 Linux 无桌面环境出现）。`database` 一行在首次运行前是 `warn`（"not initialised yet"），这是正常的。
   - 然后 `uv run twin secrets list`：应列出 `deepseek_api_key`、`smtp_password`、`autodl_password`（都是 `no`），并显示 `credential store: WinVaultKeyring`。

3. **初始化数据库并空跑应用**
   - 做什么：`uv run twin db upgrade`，然后 `uv run twin run`。
   - 预期：先打印生效配置（YAML，个人信息已打码），随后日志依次出现 `component_started`（state_watcher、heartbeat、job_worker）和 `application_running`。按 Ctrl+C，或直接关闭控制台窗口：日志出现 `application_stopping` → `component_stopped` ×3 → `application_stopped`，进程正常退出（退出码 0）。`data/logs/twin.log` 里是一行一个 JSON，且不含任何聊天正文。
   - 再开第二个 PowerShell 窗口，应用运行期间执行 `uv run twin run`：应立即报 `another 'run' instance is already running` 并退出（退出码 4）。执行 `uv run twin settings set time.bot_timezone Asia/Shanghai`：运行中的应用 2 秒内在日志里记录 `state_changed`。
   - 防休眠：应用运行期间 `powercfg /requests` 的 SYSTEM 一栏应出现 `python.exe`（`SetThreadExecutionState(ES_SYSTEM_REQUIRED)` 生效）；退出应用后消失。

4. **（可选）安装提交前隐私扫描**
   - 做什么：`uv run pre-commit install`。
   - 预期：之后每次 `git commit` 会先跑 `scripts/privacy_scan.py`、`ruff check`、`ruff format --check`。

5. **（可选）创建本地配置**
   - 做什么：`copy config\config.example.yaml config\config.yaml`，按需修改（`config/config.yaml` 已在 `.gitignore`，不会被提交）。`consent.confirmed_at` 默认已是 `2026-10-08`；删除或改成非日期时程序会拒绝启动并给出提示。

6. **（已确认）Windows CI 修复：第 00 轮首次 `windows-latest` 运行有 13 个失败，按根因修复后，提交 `eac181e`、`e3fac3e` 的 `windows-latest` 与 `ubuntu-latest` 任务均为绿色。以下仅作记录，无需再做。**
   - 做什么：看下一次推送后 GitHub Actions 里 `quality (windows-latest)` 任务，或本机 `uv run pytest -q`。
   - 预期：全绿。涉及的修复：`InstanceLock` 对象回收时释放命名互斥体（`src/twin/ops/instance_lock.py`）、`test_instance_lock.py` 里泄漏锁的用例改为显式释放、`test_cli.py` 里"运行期间持有锁"改用锁本身探测（Windows 上锁是互斥体而不是 `run.lock` 文件）、`test_secrets.py` 三个用例显式注入"无系统凭据库"（Windows 上真实存在 `WinVaultKeyring`）、`test_jobs.py::test_concurrency_is_bounded` 改为事件驱动（不再依赖 10 ms 的 `sleep` 与 Windows 上较慢的 SQLite 提交）。

## 第 01 轮 —— LLM 层（DeepSeek 客户端、费用与预算、脱敏、风格模型客户端、M0 探针）

沙箱里没有 DeepSeek Key，所以 `twin llm probe` 只在测试里用模拟接口验证过全部分支，**没有对真实接口运行过**，`docs/LLM_REPORT.md` 目前是"待实测"模板（没有任何实测数字）。

1. **存入 DeepSeek Key，运行 M0 探针（本轮的人工步骤）**
   - 做什么：`uv run twin secrets set deepseek_api_key`（粘贴 Key，不回显），然后 `uv run twin llm probe`（会先询问；约三十个很小的请求，合成内容，费用远低于 0.1 美元，记在 `one_time` 账目）。
   - 预期：依次打印七项检查名，最后一张表，第 1、2、3、4 项显示 `pass`，第 5、6、7 项是测量项，随后打印 `total cost: $0.0xxx` 和 `M0 (DeepSeek) passed`，退出码 0；`docs/LLM_REPORT.md` 被实测结果整体覆盖（文件头出现 `llm-report: measured`），其中"结论"一节写着 `M0 判定：通过`。
   - 如果输出 `STOP: JSON output with thinking enabled is not reliable`（退出码 1）：这是提示词要求的"停下来告诉我"——主动消息规划（第 11 轮）依赖思考开启时的 JSON，请把报告第 4 项的内容发给维护者，不要继续往下实现。
   - 如果某项缓存检查失败：官方说缓存是尽力而为且构建需要几秒，可以再运行一次；仍失败请把报告发给维护者。
   - 之后把更新后的 `docs/LLM_REPORT.md` 提交进仓库（只含技术结果，不含任何对话内容）。

2. **确认探针学到的能力**
   - 做什么：`uv run twin llm status`。
   - 预期：`API key (deepseek_api_key): set`；`capabilities (measured ...)` 一行显示 `detail=True gif=True json_in_thinking=True image_token_samples=5`（若接口拒绝了 `detail` 或 GIF，对应项显示 `False`，客户端会自动改变发送方式）；`last probe: ... M0 passed`；预算表显示今日与本月花费（探针费用在 `one-time spending today`，不计入降级）。

3. **（可选）核对非高峰计价对调休上班日的处理**（`docs/DECISIONS.md` D-004）
   - 做什么：在下一个"调休上班的周末"（例如官方日历里周六补班的日子）的 UTC 01:00–04:00 或 06:00–10:00 做一次很小的调用，随后在 DeepSeek 控制台的用量页看这些请求按全价还是半价计费。
   - 预期：SPEC 与代码按高峰（全价）处理。如果控制台显示是半价，把这些日期写进 `config/config.yaml` 的 `pricing.extra_offpeak_dates`（例如 `[2026-10-10]`），非高峰任务就会照半价规则安排。

4. **（可选）运行 live 测试**
   - 做什么：`$env:TWIN_LIVE=1; $env:TWIN_LIVE_DEEPSEEK_KEY="<你的 Key>"; uv run pytest tests/integration/test_llm_live.py -q`。
   - 预期：1 passed（和 `twin llm probe` 同样的七项检查，只是不写数据库与报告）。

5. **`twin doctor` 的新提示**
   - `holiday-calendar` 目前是 `warn`：已安装的 `chinese-calendar` 只收录到 2026 年，明年的节假日表官方还没发布。这不是故障；新版本发布后运行 `uv lock --upgrade-package chinese-calendar` 并 `uv sync`。
   - `deepseek-key` 在存入 Key 之前是 `warn`，存入后变 `ok`。

## 第 02 轮 —— 微信通道（openclaw-weixin / iLink 协议）、本地控制台通道、M0 通道实测

协议文档 `docs/ILINK_PROTOCOL.md`（步骤 02a）已根据官方源码写好（`@tencent-weixin/openclaw-weixin` 2.4.9，并对照 AstrBot 4.28.2 的 `weixin_oc`）。**官方源码已由沙箱拉取阅读，你不需要安装 Node.js**（除非想自己复核：`npm pack @tencent-weixin/openclaw-weixin`）。沙箱里没有微信账号，所以下面这些步骤**从未真实运行过**：`docs/CHANNEL_REPORT.md` 在拿到你的实测数据之前只能是"待实测"模板，源码读不出来的行为（主动发送窗口、可连发条数、`ret:-2` 的真实含义等，清单见协议文档第 13 节）都等这些步骤的结果。02b 步骤已提供 `twin channel login / status / send-test / unbind` 与 `twin doctor` 里的 WeChat 连通性检查（下面第 1–3 项现在可以做，**全部用合成响应测试过，从未对真实服务运行过**）；第 4、5 项（`twin channel probe …`、`twin chat --local`、`twin channel echo-test`）随 02c/02d 一并提供，同样只用合成响应和模拟平台测试过。

1. **准备手机微信，并检查这台电脑能不能连上微信服务器**
   - 做什么：把手机微信升级到支持 ClawBot 的版本（iOS ≥ 8.0.70 / Android ≥ 8.0.69），在"我 → 设置 → 插件"里确认能看到 ClawBot。然后在 Windows 电脑上运行 `uv run twin doctor`，看 `ilink-api` 与 `ilink-cdn` 两行。
   - 预期：能进入 ClawBot 的会话页面（名字和头像由你在会话设置里自己改，代码不会改）；`ilink-api`、`ilink-cdn` 显示 `ok` 和 `reachable (HTTP …)`。显示 `warn`（`cannot connect` / `timed out`）表示这台电脑从当前网络到微信服务器不通：检查网络、代理或 VPN；你人在美国，协议文档第 13 节第 15 项就是要确认这个，请把这两行的原文发给维护者。

2. **扫码登录并确认绑定（需要用户在手机上扫码）**
   - 做什么：`uv run twin channel login`。终端会打印字符二维码和一个备用链接，同时把 PNG 存到 `data/tmp/` 并用默认图片查看器打开；用手机微信扫码并确认；遇到"Enter the number shown on your phone"时输入手机上显示的数字。登录成功后命令会等你在 ClawBot 会话里给机器人发**一条消息**（最多等 10 分钟），然后显示打码的发送者 id（如 `o9cq****@im.wechat`）并问你"是不是你"，输入 `y` 才绑定。这第一条消息本身不会被机器人处理。
   - 预期：终端显示 `Logged in` 与 `Bound to …`，`uv run twin channel status` 显示已登录（bot id 打码）、已绑定、`context token: present`、窗口剩余约 22 小时、条数 8/8。二维码约 5 分钟过期会自动换新（最多 3 个码），PNG 登录结束后自动删除。
   - 如果终端警告"NOT the account that scanned the login code"：说明给机器人发消息的账号和扫码的账号不是同一个，默认应该拒绝（输入 `bind` 以外的任何内容）。已绑定后用另一个账号重新扫码会被拒绝，需要先 `uv run twin channel unbind`（要确认两次）。
   - 如果提示"需要重新登录"（协议错误码 `-14`，状态里显示 `NEEDS RE-LOGIN`，`alerts` 表有一条 critical 记录）：运行 `uv run twin channel login --force` 重新扫码，绑定保留。
   - 请把登录后 `uv run twin channel status` 的输出（不含内容，只有打码 id 与类型号）发给维护者，用来回填协议文档里标 † 的响应形状。

3. **收发冒烟与入站形态采集**
   - 做什么：先 `uv run twin channel send-test "你好"`，在手机上确认收到的是 `[测试]你好`。然后（`twin run` 没有在运行时）`uv run twin channel listen`，在 ClawBot 会话里依次发送：文字、图片、语音、视频、文件、引用一条旧消息（对文字和对图片各一次）、微信自带表情包各一条，终端每收到一条就打印一行（只有形态：`kind`、`item_type` 类型号、文字字数、媒体类型与大小、`quote=resolved/unresolved`、`flags`，**不显示内容**）。结束后 Ctrl+C，再 `uv run twin channel status` 看 `recent inbound item types` 与 `parse and handling counters`。`twin run` 运行时它自己也会轮询并记录同样的信息（`listen` 与 `run` 互斥，同一时间只有一个在轮询）。
   - 预期：手机收到测试消息；每条入站消息都出现一行。`send-test` 在窗口或条数用尽时会拒绝并提示"send the bot a message from your phone first"（先给机器人发一条消息）。如果 `send-test` 报 `code -2`，把 `errmsg` 原文发给维护者（协议文档第 13 节第 3 项要确认的 `-2` 含义）。
   - 请把 `listen` 的输出和 `status` 里出现的 `UNKNOWN` 类型号或解析失败计数（例如 `image.decrypt_failed`、`video_no_cover`、`voice_untranscribed`、`quote=unresolved`）原样发给维护者，用来补全协议文档里标"需 M0 实测确认"的入站形态（表情包、视频封面、语音转写覆盖率、只带 id 的引用）。不要贴任何聊天内容。

4. **运行 M0 通道探针（需要用户在手机上操作，约 26 小时）**
   - 前提：第 1–3 项已做完（已登录、已绑定，`send-test` 能在手机上收到 `[测试]` 消息）。探针由**运行中的应用**执行，所以要有一个窗口一直开着 `uv run twin run`，电脑不要睡眠；应用重启没关系，探针从数据库里存的计划继续。
   - 做什么：
     1. 窗口 A：`uv run twin run`，保持开着。
     2. 窗口 B：`uv run twin channel probe start`，读完说明后输入 `y`。之后随时 `uv run twin channel probe status` 看进度、当前该做什么、有没有问题在等你回答。（可选：加 `--empty-token-experiment`，多测一项"不带 context_token 的文字能不能送达"，默认不测。）
     3. **每一步开始前**，窗口 A 会弹出一个红框提示（探针同时往微信里发一条 `[测试]` 提示——但只有平台当时还接收消息才发得出去，第 2 步开始前通常发不出去，这时只有终端提示，`probe status` 会写明）。请给机器人发**一条**消息（内容随意，只发一条），然后等；探针看到这条消息、等 3 秒后才开始。
     4. 第 1 步（条数，约 30 分钟）：手机上每 2 分钟收到一条 `[测试] 条数测试 i/15（先别回复我）`，直到第一次失败或发满 15 条。**期间不要给机器人发消息。**结束后窗口 B 运行 `uv run twin channel probe answer`，回答"手机上实际收到几条 `[测试] 条数测试`"。
     5. 第 2 步（图片、GIF、正在输入，几分钟）：先发一条新消息；手机上收到三张画着 `[TEST]` 的合成图（JPG、PNG、GIF）。`probe answer` 会问每张是否收到、GIF 是否在动（`moving` / `still` / `missing`）；然后让你打开 ClawBot 聊天窗口盯着对话顶部，按回车后程序发送"正在输入"并保持 30 秒，再问你是否看到（`yes` / `no` / `unsure`）。如果第 1 步测得的条数 N 很小，图片会分成几次发，每次之前都要你再发一条新消息；源码不支持发送引用，所以没有引用子步骤。
     6. 第 3 步（窗口，约 25 小时）：先发一条新消息，**此后约 25 小时内不要给机器人发任何消息**，否则这一步作废，终端会告诉你并要求你再发一条消息重做。你发出那条消息之后约 1、6、12、20、23、25 小时，手机上各会收到一条 `[测试] 窗口测试 …`（条数不够时只保留靠后的测量点，报告会写明省略了哪些）。窗口 A 和电脑要一直开着。最后一条之后用 `probe answer` 回答共收到几条。
     7. 全部做完后：`uv run twin channel probe report`。
   - 对账（很重要）：程序提问时给出的是"服务器接受了几条"，你回答的是**你的手机聊天窗口里实际看到几条**，以你的回答为准。接口返回成功不一定等于手机收到（社区报告过缺 token 时接口"成功"但消息不推送，打开聊天窗口才出现）；两个数字不一致时报告里标"**不一致**"，并按手机上的数算。拿不准就打开 ClawBot 聊天窗口往上翻着数，只数以 `[测试] 条数测试`（或 `[测试] 窗口测试`）开头的消息；不要数 `[测试] 通道探针` 开头的提示消息。
   - 预期：`probe status` 最终显示 `completed`；`probe report` 写出 `docs/CHANNEL_REPORT.md`（只含测量结果与时间，文件头是 `channel-report: measured`），打印判定，并逐项显示"当前 → 建议"的 `channel.proactive_window_safe_h`（窗口下限 ×0.9）与 `channel.outbound_quota_safe`（N ×0.9），**问你是否写入 `config/config.yaml`**——回答 `n` 则不改任何配置，之后想改也可以重新运行 `probe report`。每次失败的 `ret`、`errcode`、`errmsg`（已脱敏）都在报告的失败表里，请原样保留。
   - 如果报告写"**未达标**"（窗口 < 12 小时，或一次入站后手机上收到的连发条数 < 3）：`probe report` 以退出码 1 结束并打印 `STOP`，这是提示词要求的"停下来告诉我"——主动消息（第 10 轮）依赖这两个条件。把报告发给维护者，由你决定下一步；企业微信通道不在本规格范围，需要你确认后另开一轮。"无法判定"表示某项没测到或测的点不足以下结论（报告写明缺什么），不算通过，补测即可。
   - 中途想放弃：`uv run twin channel probe stop`（已测到的保留，`probe report` 照样能出一份"未做完"的报告）。窗口测试被打断或网络断了，探针自己会作废那次尝试并要求你再发一条消息重做（每步最多 10 次）。
   - 提交 `docs/CHANNEL_REPORT.md`（只含技术结果，不含任何对话内容）；第 09b 轮的 `twin eval gate M0` 读取数据库里的同一份结果（`settings` 键 `m0.channel_probe`，格式见 `docs/DECISIONS.md` D-158）。

5. **（可选）本地控制台通道与回显诊断**
   - 做什么：`uv run twin chat --local`（与 `twin run` 互斥，要先停掉它）。输入一行文字回车就是发一条消息；`/img <图片路径>` 发送一张你电脑上的图片（路径可带引号）；`/help`、`/quit` 离开。再试 `uv run twin channel echo-test --local`：它把你写的话以 `[测试]回显:` 为前缀回给你，用来检查通道本身；对微信可以用 `uv run twin channel echo-test`（只回给已绑定的你，仍受安全窗口和条数限制；`twin run` 要先停掉）。
   - 预期：第 09 轮起 `twin chat --local` 接上了回复引擎（需要 DeepSeek Key，见第 09 轮的条目），你输入的话由她按真实节奏回复；`echo-test --local` 仍只回显；机器人发表情包时显示 `bot: [表情包：<标签>] <文件路径>`、发其他允许的图片时显示 `bot: [图片] <路径>`、发之前显示 `对方正在输入…`（表情包标签第 06 轮之前显示"未标注"）。

## 第 03 轮 —— 聊天记录导入（流式、可续传、媒体与表情包、图片描述、报告、导入后钩子）

沙箱里没有真实导出目录，整套导入只用 `tests/fixtures/synth_export.py` 生成的合成数据验证过。下面这些步骤需要你在自己的 Windows 电脑上、用真实导出做一遍，**顺序很重要**：先看结构，再导入。

1. **先看结构：对真实导出运行 `twin import inspect`（本轮最重要的一步）**
   - 做什么：`uv run twin import inspect "<你的导出目录>"`（目录名含中文也没关系，给引号）。大导出可加 `--sample 20000` 多看一些消息；默认每个 `messages.json` 看前 5000 条。
   - 预期：终端打印并保存 `data/reports/inspect-<UTC时间>.md`。**报告里只有结构，不含任何值**：文件树（会话文件夹显示成 `序号_昵称字数_哈希前4位`）、`report.json` / `meta.json` / `messages.json` 的键名与值类型和出现次数、`renderType` / `type` / `offlineMedia[].kind` 等枚举字段的取值计数、`_integrity/` 的文件清单和 JSON 键名。报告里不应出现任何消息文字、昵称、wxid、媒体文件名；如果你发现有，请不要贴出来，告诉我是哪一行。
   - 然后：把这份报告整个贴回给我。我据此核对四件事，有出入先告诉你再继续：
     1. SPEC R-IMP-002 的字段表和 `src/twin/ingest/schema.py`（是否有字段缺失、类型不同——例如 `createTime` 是秒还是毫秒，`isSent` 是布尔还是 0/1，`voiceLength` 的单位）；
     2. `offlineMedia[].kind` 的真实取值（`src/twin/ingest/normalize.py::classify_media` 目前按名字里含 image / emoji / avatar / thumb / cover / voice / video / file 归类，没见过的名字会被跳过并计数）；
     3. `renderType` 的真实取值（没见过的会保存为"未识别"类型、原始记录完整保留，并在导入报告里按名字列出）；
     4. **`_integrity/` 的真实格式**（见下一条）。

2. **`_integrity/` 的格式（决策 D-007）**
   - 代码现在能读两类常见形态：JSON（`files` / `entries` / `items` / `manifest` / `checksums` / `hashes` 成员，或顶层列表/映射，条目里有路径与 `sha256`/`sha1`/`md5`/`size`）和 `sha256sum` 风格的文本行。真实格式没见过，所以对不上时导入报告的"数据质量提示"里会写"完整性文件夹格式无法识别，没有校验任何文件"，导入照常进行。
   - 做什么：看 inspect 报告里 `_integrity/` 一节；若出现这条提示，把该节贴给我，我改 `src/twin/ingest/integrity.py::parse_manifest_text`。
   - 预期：格式被识别后，导入报告里会有"完整性校验（_integrity）：读到 N 条记录；messages.json ok；媒体通过 X，失败 0，未列入清单 Y"。校验失败的文件不会被导入（`messages.json` 失败则整次导入中止并报错）。

3. **配置并首次导入**
   - 做什么：在 `config/config.yaml` 里设置 `paths.export_dir`（全量导出目录，不要放进仓库），确认 `time.source_timezone` 是 `createTimeText` 所用的时区（芝加哥导出就保持默认；如果一段时间在国内导出，用 `time.source_timezone_ranges`，见 SPEC R-CFG-004）。然后：
     ```
     uv run twin db upgrade
     uv run twin import "<导出目录>" --foreground
     ```
     （应用在运行时去掉 `--foreground`，命令会入队后立即返回，用 `uv run twin import status --watch` 看进度。）
   - 预期：首次导入会列出**所有一对一会话**（昵称、消息数、打码的 wxid），让你输入编号选目标会话，选择写入运行时设置 `target.username`（`twin settings list` 里显示为打码）。群聊不会被列出。随后看到进度条（阶段 messages → media → stickers → finalize → hooks），最后打印导入报告并保存到 `data/reports/import-<UTC时间>.md`。报告不含任何消息文字和 wxid。
   - 请核对报告：①"按类型与发送方计数"里"她"和"用户"没有互换（`isSent=false` 是她）；②"日期范围"和你的记忆吻合；③"数据质量提示"里**没有**"createTime 与 createTimeText 相差超过一小时"（有的话说明 `time.source_timezone` 不对，改好后重新导入，幂等）；④"无法导入"为 0 或很小。把统计部分（不含正文）贴给我。

4. **续传与增量**
   - 做什么：在导入进行中按 Ctrl+C（或关掉窗口），再运行 `uv run twin import --resume --foreground`；以后有新导出时再次 `uv run twin import "<新导出目录>"` 即可（按消息 id 去重，新导出里改过的消息按 `exportedAt` 较新者为准）。
   - 预期：续传从上次提交的批次继续，最终条数与一次性导入完全一致；增量导入的报告里"新增 / 重复 / 冲突"三项各自有数。

5. **表情包下载**
   - 做什么：导入后 `uv run twin stickers download --foreground`（应用在运行时去掉 `--foreground`）。
   - 预期：并发 4、每秒不超过 4 个请求；结束后按状态列出表情包数量：`available`（MD5 一致）、`md5_mismatch`（已保存但不会被发送）、`unavailable`（带原因，例如 `http_404`、`timeout`）、`pending`。失败的可以稍后 `uv run twin stickers download --retry-failed`。这一步沙箱里只用模拟服务器验证过，没有访问过真实的表情包 CDN。

6. **图片描述（需要 DeepSeek Key，会花钱）**
   - 做什么：先 `uv run twin secrets set deepseek_api_key`（第 01 轮），然后 `uv run twin images caption-backfill`（导入结束时钩子已经排过一次，重复运行只会提示"已在队列中"）。命令会列出批次号、图片数和**估算费用**，但不会执行；确认金额后 `uv run twin jobs approve <批次号>`，任务在 DeepSeek 非高峰时段由运行中的应用执行（应用没运行时 `uv run twin jobs run --until-idle`）。
   - 预期：费用记在 `one_time` 账目（`uv run twin llm status` 可见），不计入日/月预算；实际花费超过估算 20% 时批次会自动暂停并告警。描述先经脱敏（电话、地址等被替换成 `[手机号]` 之类）再加密保存。沙箱里没有 Key，这条链路只在模拟接口上验证过，没有对真实 DeepSeek 跑过。

7. **性能（可选）**
   - 做什么：`uv run python scripts/bench_import.py --messages 1000000`（合成数据，不碰你的真实数据；需要约 1.5 GB 空闲磁盘）。
   - 预期：输出每秒条数、峰值内存和两个 PASS。沙箱里的实测数字见 `docs/PERFORMANCE.md`；你的电脑上的数字请告诉我，Windows 上的磁盘与杀毒软件可能让速度差一个数量级，但门槛是 30 分钟。

8. **（可选）Windows 长路径与中文路径**
   - 如果导出目录很深（媒体文件路径超过 260 字符），代码已对 Windows 使用 `\\?\` 前缀；若仍有"文件不存在"的缺失媒体，开启系统的长路径支持（`LongPathsEnabled`）后重新导入，缺失项会被补上。

## 第 04 轮 —— 风格统计画像、作息活动模型与留出切分点

沙箱里没有真实聊天记录，画像与作息只用 `tests/support/synth_chat.py` 生成的、带**已知规律**的合成数据（她 01:00–08:30 不发消息、工作日 13:00–17:00 回复慢、每天先开口 4 次、逗号率 3% 等）和 SPEC §0 的小时向量验证过。下面这些要在你自己的电脑上、用真实数据做一遍：

1. **回填一次画像与作息（验收命令）**
   - 做什么：确认真实数据已按第 03 轮导入，且 `time.source_timezone` 与导出里 `createTimeText` 所用的时区一致（第 03 轮导入报告里"createTime 与 createTimeText 相差超过一小时"一栏为空即可）。然后
     ```
     uv run twin profile rebuild --scope all --foreground
     ```
     （应用在运行时去掉 `--foreground`，命令只入队，运行中的应用执行；以后每次 `twin import` 有她的新消息时钩子会自动排队同一件事。）
   - 预期：打印 `live:` 与 `pre_holdout:` 两行（版本号、她的消息条数）。第一次计算同时**确定并保存留出切分点**（她的回复块里最近的 10%，之后导入新数据不会移动它）。
   - 请把耗时和消息总条数告诉我（沙箱里合成数据约 210 微秒/条，见 DECISIONS D-176），真实数据慢得多的话我再优化。

2. **确认推断出的睡眠时段（本轮最重要的一步）**
   - 做什么：`uv run twin profile show`，看"作息概览（当地时间）"：睡眠（工作日/周末分别）、忙碌时段、每天先开口次数、置信度。
   - 预期：睡眠时段是你熟悉的她的作息。**如果输出里有以 `!!!` 开头的红色警告**，说明推断出的睡眠核心落在当地白天 10:00–18:00，几乎一定是 `time.source_timezone`（或 `time.source_timezone_ranges`）与导出的时区不一致——先改配置再重新运行第 1 步，不要用 `routine add` 去掩盖。
   - 不对但时区没问题：用手动修正（优先于推断，立即生效，不需要重算；与第 11 轮的微信指令共用同一个存储）：
     ```
     uv run twin routine add sleep 01:00 08:30                      # 所有日子
     uv run twin routine add sleep 02:00 10:30 --days weekend,holiday
     uv run twin routine add busy 13:00 17:00 --weekdays mon-fri
     uv run twin routine add holiday 2026-10-01 2026-10-07          # 学习时按节假日算，需再运行第 1 步
     uv run twin routine list        # 以及 routine remove / disable / enable <id>
     ```
   - 置信度写"低"（可用于推断的夜晚少于 `activity.min_valid_days` = 14 个）时，睡眠时段取的是整体活跃曲线里最长的低谷，请特别核对。

3. **核对风格数字**
   - 做什么：同一条 `twin profile show` 的"风格指标"表，右边一列是 SPEC §0 那 7 天样本的数字。你的完整记录算出来的数字和样本有出入是正常的（样本只有 7 天），但如果某一项明显离谱（例如她的逗号率 40%、连发中位数 1），请把整张表贴给我——多半是我对某类消息的归类和你的数据不一致。
   - 另有 `uv run twin profile phrases`：在**本机屏幕上**列出她的高频整句、高频字组和"称呼候选"（句首/句尾反复出现的 2–3 字组合，附次数）。这些是真实消息原文，不会写进任何文件、日志或画像 JSON（它们加密存放在 `profile_versions.phrases` 列）。请看一眼称呼候选里有没有她叫你的名字、有没有明显不是称呼的词——第 06 轮的人设卡会用到。

4. **版本、对比与回滚**
   - 做什么：`uv run twin profile history`（`~0` 是最新）、`uv run twin profile diff ~1 ~0 --scope live`（列出变化超过 10% 的指标）、`uv run twin profile rollback ~1 --scope live`（连同当时的作息模型一起切回去）。
   - 预期：两个范围（`live` 线上用、`pre_holdout` 训练与评估用）各有自己的版本链；结果和上一版逐字节相同时不会产生新版本。

5. **导入后的"风格变化"（下次导入时看）**
   - 下次 `uv run twin import <新导出>` 之后，导入报告多一节"风格变化"：画像重算排队中时先写"还没有计算/首次画像"，任务跑完后这一节会被改写成"相对上一版变化超过 10% 的指标"。没有她的新消息的导入不会触发重算。

6. **（后续轮次）重切留出集**
   - 函数 `twin.profile.holdout.resplit_holdout()` 已完成并有测试；命令 `twin retrieval resplit` 在第 05 轮提供。重切会让评估结果前后不可比，并自动排队重算 pre_holdout 范围的派生数据。

## 第 05 轮 —— 向量模型与真实片段检索库

沙箱里没有真实聊天记录，检索库只用合成对话验证过（窗口边界、留出集、MMR、时段加分、`before` 过滤、增量与续跑、模型变更拒绝等），向量模型则**真的**下载并跑过一次：`BAAI/bge-small-zh-v1.5`（revision `7999e1d3…`，权重 SHA-256 `354763b9…`，512 维）在沙箱 CPU 上约 9 秒/千个窗口，数字与限制见 `docs/PERFORMANCE.md` 第 2 节。下面是只有你能在自己电脑上做的事。

1. **安装并建库（验收命令）**
   - 做什么：Windows 上 `uv sync` 会按 `uv.lock` 装 PyPI 的 torch（Windows 的 PyPI 轮子本来就是 CPU 版）和 LanceDB。第一次编码时会把模型下载到 `data/models/embeddings/`（约 100 MB，需要能访问 `huggingface.co` 一次；之后不再联网）。然后（应用没在运行时）：
     ```
     uv run twin retrieval rebuild --foreground
     uv run twin retrieval stats
     ```
     应用在运行时去掉 `--foreground`，命令只入队、运行中的应用执行，用 `twin retrieval stats` 看进度（“last index run”一行有已完成/总数、速度、预计剩余）。任务可随时 Ctrl+C，再次运行会从停下的地方继续。
   - 预期：`stats` 里有“windows (her reply blocks)”总数、“held out”（约占 10%）、“windows with a vector”（= 有上下文且未留出的窗口数）、“vectors in the index”与前者相同、模型名 `BAAI/bge-small-zh-v1.5`，且没有以 `problem:` 开头的行。
   - 请把这些数字告诉我：窗口总数、留出数、`last index run` 里的“每千个窗口多少秒”（沙箱 4 核 CPU 是 9 秒；你的 CPU 不同会有几倍差别）。另外：`uv run pytest tests/unit/test_retrieval_real_backend.py -q`（约 10 秒）确认 Windows 上 torch + sentence-transformers 能正常加载一个小模型；`$env:TWIN_LIVE=1; uv run pytest tests/integration/test_retrieval_live.py -s -q` 用真实模型再跑一遍（会下载模型，约一分钟）。
   - 如果访问不了 Hugging Face：在能访问的电脑上跑一次，再把整个 `data/models/embeddings/` 文件夹拷到本机同一位置；或设置环境变量 `HF_ENDPOINT` 指向你信任的镜像。

2. **想用 GPU 编码（可选；bge-small 在 CPU 上已经够用）**
   - 只有换成 `BAAI/bge-m3`（`retrieval.model`）才值得：它在 CPU 上约慢 20 倍（10 万个窗口约 5 小时）。有 NVIDIA 显卡时按 PyTorch 官网的命令装 CUDA 版 torch（例如 `uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall`），`retrieval.device: auto` 会自动用 GPU；注意之后每次 `uv sync` 会把 torch 换回锁定的 CPU 版，运行命令时用 `uv run --no-sync ...`。`retrieval.device: cuda` 但没有可用 GPU 时会明确报错，不会悄悄改用 CPU。
   - 换模型后必须 `uv run twin retrieval rebuild`（旧索引的向量与新模型不在同一个空间，增量导入会拒绝混用并发一条告警）。

3. **数据安全检查（一次就行）**
   - 向量库在 `data/vectors/`，只有窗口 id、向量、时间、15 分钟格、日类型，没有任何文字；`twin retrieval stats` 的输出里也只有计数。如果想自己确认：`data/vectors/` 下的文件用任意工具搜她的某句话，应当找不到。

4. **何时重切留出集（`twin retrieval resplit`）**
   - 背景：留出切分点在第一次计算后固定，之后导入的**更新的**消息全部落在切分点之后，算留出集、**不会进检索库**（评估集不能悄悄变化）。所以每次导入新记录后，最新的那一段对话要等你重切才会被她的“类似情况下怎么回”检索到。
   - 什么时候做：新记录积累到一定量（例如准备重训风格模型之前）。命令：
     ```
     uv run twin retrieval resplit            # 会先问一次；--yes 免问；--foreground 在应用没运行时就地执行排队的作业
     ```
     它把切分点移到“今天的数据的最新 10%”，立刻把新进入留出集的窗口从索引里删掉、把退出留出集的窗口排队编码，并排队重算 pre_holdout 的画像与作息。**重切后，重切之前的评估结果与之后的不可比**（留出集换了），第 09b 轮起请在重切之后重新评估。
   - 平时不需要重切。

5. **每次导入之后**
   - `uv run twin import <新导出>` 的导入后钩子会自动排队检索库的增量更新（只编码新增的、未留出的窗口；没有新消息时跳过）。导入报告“导入后钩子”一节里有 `retrieval` 一行。钩子的回填命令就是第 1 条的 `twin retrieval rebuild`。
   - 如果钩子一行显示 failed 且写着 `retrieval.model` 与索引不一致：你改过模型，运行 `uv run twin retrieval rebuild`。

6. **（后续轮次）**
   - 第 09 轮把 `render_example()` 的输出放进提示词、用 `BudgetLimits.examples_k` 限制条数；第 06 轮给表情包例子接上标签（`sticker_label`）；第 07 轮的记忆向量用同一个 `EmbeddingService` 与 `VectorStore`（各自独立的表）。这些都不需要你做什么。

## 第 06 轮 —— 人设卡与表情（表情代码、表情包库、打标签、选择与频率控制、识别）

沙箱里没有 DeepSeek Key、真实记录和真实表情包，所以人设卡的 Map-Reduce、证据校验、表情包视觉打标签与上下文修正都只用合成数据和固定响应验证过（`respx`）；一次性批任务做到“估算费用 → 等 `twin jobs approve` → 执行”的完整逻辑。**这一轮没有向微信发任何消息。** 下面是只有你能在自己电脑上做的事。

1. **前置：画像与第一版人设卡（只含统计规则，免费）**
   - 做什么（应用没在运行时；在运行时去掉 `--foreground`，命令只入队）：
     ```
     uv run twin profile rebuild --foreground
     uv run twin persona regenerate --stats-only --foreground
     uv run twin persona show
     ```
   - 预期：`live` 与 `pre_holdout` 各得到一张只有 `[自动-统计规则]` 的卡（`[自动-描述]` 先是空的），末尾一行显示完整版和精简版各多少 token（上限 1500 / 400）。
   - 导入之后这一步会自动排队（导入报告“导入后钩子”里的 `persona` 一行）；它要等同一次导入的画像重算完才会执行。

2. **生成第一版人设卡的描述（要花钱，先批准）**
   - 做什么：
     ```
     uv run twin persona regenerate                 # 先给费用估算（上限），live 与 pre_holdout 在同一个批次里
     uv run twin jobs approve persona-<批次号>        # 你同意这个价格
     uv run twin jobs run --until-idle              # 应用没在运行时；运行中的应用会自己执行
     ```
   - 作业只在 DeepSeek 非高峰时段执行（官方取消非高峰优惠时把 `pricing.offpeak_multiplier` 设为 1.0，就不再等待）。估算是上界（每段按最长算），实际一般更低；超出估算 20% 会暂停并告警。
   - 请通读：`uv run twin persona show`（`--full` / `--compact` 看两种渲染，`--evidence` 看每条陈述对应的会话段编号，`--scope pre_holdout` 看训练与评估用的那张）。每一条都应当能在你们的真实对话里找到；有“编出来”的事实就记下来告诉我（程序已经丢弃了没有合法证据编号的陈述，但证据编号只能证明“这段话在样本里”，不能证明模型理解对了）。
   - 请告诉我：两张卡的长度（token 数）、有没有明显不像她的描述。

3. **手改 `[手动]` 分区**
   - 做什么：`uv run twin persona edit`。Windows 上会用 `.md` 文件的默认程序打开解密后的临时文件（在 `data\tmp\` 里）；改好、**保存并关闭**，回到终端按回车。其他平台用 `$VISUAL` / `$EDITOR`。
   - 只有 `[手动]` 分区的修改会被接受；改了别的分区会被拒绝并说出是哪个分区。结束后临时文件（和编辑器留下的备份、交换文件）会被覆盖再删除。如果 `.md` 的默认程序不是文本编辑器，先在 Windows 里把它改成记事本。
   - `### 风格` 里写“称呼：宝宝”“口头禅：……”“开心时：……”这样带标签的行，裁剪时会按优先级保留；`### 事实` 是你们上线后才有的信息，只出现在 live 卡的完整版里。`### 风格` 的内容会自动复制到 pre_holdout 卡（训练与评估用）。
   - Windows 上请再确认一次：`uv run pytest tests/unit/test_persona_edit.py -q`（含只在 Windows 上运行的 `os.startfile` 测试），以及改完后 `data\tmp\` 里没有 `persona-*` 文件。

4. **给表情包打标签（要花钱，先批准）**
   - 做什么（表情包文件要先下载完：`uv run twin stickers download --foreground` 或等应用自己下载）：
     ```
     uv run twin stickers tag-all                   # 估算费用；列出批次号
     uv run twin jobs approve stickers-<批次号>
     uv run twin jobs run --until-idle
     uv run twin stickers list
     ```
   - 作业先看图（`detail: low`；是否真的发送 `detail` 取决于你跑过的 `twin llm probe` 结果，没跑过就按官方文档默认发送），给每张打 1–3 个标签（固定词表，见 `config/lists/sticker_tags.txt`）和一句描述，再对她在**留出切分点之前**用过 ≥ 3 次的表情包做上下文修正（上下文标签优先于纯视觉），最后用第 05 轮的向量服务给描述编码（第一次会下载嵌入模型，见第 05 轮第 1 条）。
   - 请告诉我：`twin stickers list` 第一行的统计（总数、可用文件数、她用过的、已打标签的、有描述向量的）。
   - 之后新增的表情包（导入、下载完成）会自动用日常账户打标签，不需要批准；只有“库里还没有任何标签”的第一次需要批准。

5. **检查并修正标签**
   - `uv run twin stickers list --limit 20` 看她最常用的 20 张；`uv run twin stickers show <md5 前缀>` 看画面描述、视觉标签、她的使用修正和最终标签。
   - 不对的用 `uv run twin stickers tag <md5 前缀> 委屈 撒娇` 手改（手动标签优先于一切自动结果），`uv run twin stickers untag <md5>` 取消；不想让它被选中的用 `uv run twin stickers disable <md5>`。
   - 请告诉我：前 20 张里错了几张（用来判断视觉打标签与上下文修正是否可靠）。

6. **导入新记录或重切留出集之后**
   - 导入：钩子会刷新统计规则；她的消息比上次生成描述时增加 ≥ 10% 时，会排一个新的描述生成批次（导入报告里有批次号和估算，等 `twin jobs approve`）。
   - `twin retrieval resplit`：pre_holdout 的画像重算、统计规则刷新自动排队；pre_holdout 描述的重新生成要你批准（报告里有批次号）；按旧切分点做的表情包上下文修正会立刻清掉并重做（日常账户，不需批准）。
   - `uv run twin persona status` 随时查看哪个范围的描述到期了。

7. **（后续轮次）** 第 09 轮把 `render_full()` / `render_compact()`、`EmojiCodePolicy`、`StickerSelector`、`StickerRateController`、`describe_incoming_sticker` 接进回复引擎；第 11 轮通过 `write_corrections()` 写入 `[不要这样]`；第 12 轮的 `twin rollback` 会覆盖人设卡与提示词模板的回滚。这些都不需要你现在做什么。

## 第 07 轮 —— 记忆系统（事实库、每日摘要、生活线与待跟进的数据层、检索组装、时间回放、AsOfView）

沙箱里没有 DeepSeek Key 和真实聊天记录，所以事实抽取、冲突判定、每日摘要和全量回放都只用合成对话和固定响应（`respx`）验证；一次性批任务做到“估算费用 → 等 `twin jobs approve` → 执行 → 花费超过估算 120% 暂停”的完整逻辑（有测试）。**这一轮没有向微信发任何消息，也没有改动聊天记录。** 下面是只有你能在自己电脑上做的事。

1. **前置**
   - 已导入聊天记录（第 03 轮）；已设置 Key：`uv run twin secrets set deepseek_api_key`（第 01 轮）；嵌入模型已下载（第 05 轮第 1 条的 `twin retrieval rebuild --foreground` 会下载；记忆的向量与检索库共用同一个模型和向量库，表各自独立）。
   - 确认 `time.source_timezone`（搬过家则还有 `time.source_timezone_ranges`）是对的：回放按**她当地的日历日**切分，“明天”“后天”也按那个时区换算成日期。

2. **估算（免费，只读）**
   - 做什么：`uv run twin memory replay estimate`（可加 `--from 2026-01-01 --to 2026-03-31` 只看一段）。
   - 预期：表里有“要回放的天数”“已做过的天数”“对话行数”“估算对话 token”“估算费用（美元，高峰价，上界）”“单批上限”。估算按最坏情况算（每天最多 12 条事实、每条最多 6 个旧条目要判定、不计缓存命中、不计非高峰折扣），实际一般明显更低。估算超过 `budget.one_time_usd`（默认 30）时会拆成多个批次，每批不超过上限；单独一天就超过上限时拒绝入队并说出是哪一天。
   - 请告诉我：天数、行数、估算费用。

3. **排队、批准、执行（要花钱）**
   - 做什么：
     ```
     uv run twin memory replay start                  # 排队；打印批次号和估算，此时还不花钱
     uv run twin jobs approve memory-<批次号>         # 你同意这个价格（每个批次一次）
     uv run twin jobs run --until-idle                # 应用没在运行时；运行中的应用会自己在 DeepSeek 非高峰时段执行
     uv run twin memory replay status                 # 随时看进度（天数、作业、批次花费、记忆里各来源的事实数）
     ```
   - 每个作业处理 `memory.replay_job_days`（7）个当地日，按日期顺序；每天先写摘要、再抽取事实与待跟进、再写入记忆（冲突判定、覆盖旧事实、让被真实事实推翻的生活线条目失效）。中断后重跑只做没做完的日子（按输入指纹判断，消息变了的日子会重做）。
   - 实际花费超过估算的 120% 时该批次自动暂停并告警（`status` 显示 paused）。用 `uv run twin llm status` 看花费，认可后再 `uv run twin jobs approve` 同一批次继续。
   - 请告诉我：实际花费 / 估算（`status` 里每批一行 `spent … of a cap of …`）、失败的天数（`uv run twin jobs list --status failed --type memory_replay`）。

4. **检查记忆质量（只读）**
   - `uv run twin memory list`（`--page N`、`--keyword 词`）：编号、文字、来源（`real_record` 来自真实记录），每页 10 条，末尾列出还没结束的待跟进。请通读前 20–30 条：每条都应当能在你们的真实对话里找到。
   - `uv run twin memory block "最近有什么打算"`：显示提示词里会得到的记忆块（`--at 2026-03-05T09:00:00-06:00` 看某个过去时刻的样子，必须带时区偏移；`--budget N` 改 token 预算）。留意两种情况：
     - 与话题无关的事实混进来：说明 `memory.recall_min_similarity`（默认 0.35）对真实向量模型偏低，请在 `config.yaml` 里调到 0.45–0.55 再试。默认值是在沙箱的玩具向量上调的，真实模型的相似度分布不同，这是本轮最需要你反馈的一项。
     - 该出现的纪念日、考试没出现。
   - 有错的或不想留的：`uv run twin memory forget <编号>`（硬删除，连同仅由它派生的待跟进和生活线条目；被它覆盖过的旧事实恢复生效）。注意：`replay start --force` 会重新抽取，可能把你删掉的事实再抽出来，删过东西的日子不要用 `--force`。
   - 请告诉我：事实总数、各来源的数量、前 20 条里错了几条、`block` 给出的内容像不像“她该记得的”。

5. **手动写入一条**
   - `uv run twin memory remember "她不吃香菜"`：来源 `user_command`（优先级最高，覆盖与它冲突的其他来源的事实）。有 Key 时模型会把它整理成带日期、周期的事实，失败则原样保存。

6. **每日摘要**
   - `uv run twin memory summarize 2026-03-05 --scope real`（排队，非高峰执行；日常账户，不用批准）。第 08 轮的调度器上线后会每天在她“起床”前自动排。摘要 ≤ 300 字；请抽几天对照原对话，看有没有漏掉重要的事、有没有编造。

7. **导入新记录之后**
   - 导入钩子会找出没回放过（或消息变了）的日子并排队：第一次回放必须你批准；之后的增量如果估算低于 `budget.one_time_usd` 的 10%（`memory.replay_auto_approve_ratio`）则自动批准。导入报告“导入后钩子”里有一行 `memory_replay`，单独补跑用 `uv run twin memory replay start`。
   - 换了嵌入模型（`retrieval.model`）之后：`uv run twin memory reindex`。

8. **（后续轮次）** 第 08 轮生成每日生活线并按日排每日摘要；第 09 轮把 `MemoryAssembler.build` 接进回复提示词、创建 `bot_turns` 并注册读取器（`register_bot_turn_reader`）、会话静默后调用 `queue_bot_extraction`；第 10 轮读写待跟进；第 11 轮的 `/记住`、`/忘掉`、`/记忆` 调用 `MemoryManager`；第 13 轮的训练集导出只经 `AsOfView(t)` 读记忆。这些都不需要你现在做什么。

## 第 08 轮 —— 时间服务、时区切换、每日计划与生活线生成

沙箱里没有 DeepSeek Key，生活线的生成和检查只用固定响应（`respx`）验证；Windows 的“隐藏窗口收 `WM_POWERBROADCAST`”在 Linux 上用记录型替身测逻辑，真实窗口的测试带 `windows` 标记，只会在 windows-latest 的 CI 上第一次真跑。**这一轮不会向微信发任何消息**（主动消息在第 10 轮）。下面是只有你能在自己电脑上做的事。

1. **前置**
   - 已导入聊天记录并重算画像（第 03、04 轮：`uv run twin profile rebuild`），已设置 Key（`uv run twin secrets set deepseek_api_key`，生活线要用）。
   - 升级数据库：`uv run twin db upgrade`（迁移 `0008_daily_plans_timezone`，新增 `daily_plans`、`timezone_history`）。
   - `uv run twin profile show` 的“作息概览”里，睡眠和忙碌时段必须是对的（计划从它抽样）；不对用 `uv run twin routine add sleep 01:00 08:30` 等修正。手动修正优先于推断。

2. **看时区和时间（只读）**
   - 做什么：`uv run twin timezone show`。
   - 预期：`bot time zone: America/Chicago (UTC-5, CDT)`、当地时间（和你手表上的芝加哥时间一致）、日类型（工作日/周末/节假日）、“routine learnt in: America/Chicago”、下一次拨钟（2026-11-01 07:00 UTC，CDT -> CST）、她此刻的状态（没有计划时写 `no plan yet`）。
   - 请告诉我：日类型对不对（美国联邦假日按 `holidays.US`）。

3. **看今天的计划（只读，先预览）**
   - 做什么：`uv run twin plan show`（或 `uv run twin plan show 2026-10-12`）。
   - 预期：当地时间的起床/入睡、忙碌段（带回复延迟中位）、饭点（标明“来自她的活跃峰”还是“常见饭点”）、主动配额（均值目标应接近 `uv run twin profile show` 里的“每天先开口 N 次”）、起床问候窗口、状态时间线。没有保存过时标“预览”，盐生成后种子才固定。
   - 请告诉我：起床/入睡时间、饭点、忙碌段像不像她；不像的用 `twin routine add …` 修正，然后 `uv run twin plan rebuild --force` 重做今天。

4. **启动应用，看每日作业**
   - 做什么：`uv run twin run`（保持运行）。在另一个终端：`uv run twin plan show`（这次是真正保存的计划，无“预览”）、`uv run twin jobs list`。
   - 预期：启动时今天的计划已经生成；到她起床的时刻出现 `lifeline_generate` 作业并很快完成；起床前一小时（高峰价格时顺延到非高峰，最晚起床后两小时）有 `memory_summary` 作业（前一天有真实记录行时 `real`，第 09 轮之后 `bot`）。
   - 想马上试：`uv run twin plan lifeline-generate`（排队今天的生活线，应用在运行时由它执行），再 `uv run twin plan lifeline` 看结果。大约两次模型调用，花费很少，走日常账户、不需要批准。

5. **看生活线**
   - 做什么：`uv run twin plan lifeline`。
   - 预期：5–10 段当天的安排（时间、做什么、在哪、心情）；睡觉时段没有活动；忙碌时段里是上课/上班之类的事；学校、工作、住处与事实库里的真实事实一致；昨天的事能接上。
   - 请告诉我：像不像她的一天、有没有和你知道的事实矛盾、日志里 `lifeline_generated` 那一行的 `drafts`（2 说明第一版被退回过）和 `corrected`（true 说明两次都没过、采用了规则修正的版本，同时会有 `lifeline_corrected` 告警）。

6. **切换时区（回国后）**
   - 做什么：`uv run twin timezone set Asia/Shanghai`，再 `uv run twin timezone show`、`uv run twin timezone history`。要切回：`uv run twin timezone set America/Chicago`。
   - 预期：输出新时区下“今天剩余部分”的计划；若上海此刻是夜里，她立即进入睡眠状态（`睡眠·早晨` 一行会写“时区切换时正处于夜里，立即去睡”），直到上海的起床时间；起床问候不会在距上一次不足 18 小时时重复；运行中的应用两秒内发现（日志有 `state_changed`），并发出 `TimezoneSwitched` 与 `CandidatesExpired` 事件（第 09、10 轮订阅）。切换前已排队的延迟回复保持原来的绝对时刻，主动候选作废并重抽（理由见 D-233）。
   - 注意：这会立刻改变她的状态（可能马上“睡着”），第 09 轮之前没有别的影响。

7. **夏令时**
   - 2026-11-01（芝加哥 02:00 回拨到 01:00，这一天 25 小时）前后：`uv run twin plan show 2026-11-01`。
   - 预期：起床/入睡的钟点不变（07:30 就是 07:30），那一夜实际睡 9 小时；若有钟点落在 01:00–02:00，会有一行“夏令时：…取第一次”。2027-03-14 春季那天同理（23 小时，落在 02:00–03:00 的钟点晚一小时发生）。

8. **睡眠唤醒（Windows）**
   - 做什么：应用运行时手动让电脑睡眠几分钟，再唤醒。
   - 预期：日志出现 `machine_resumed source=wm_powerbroadcast gap_s=…`，随后 `schedule_resumed kind=wake` 和 `channel_reconnected`；计划仍是当天的（跨过零点则生成新一天的）；没有补发任何东西。合盖/睡眠的检测在所有平台都有时钟跳变兜底。
   - 请告诉我：windows-latest CI 上 `tests/unit/test_power_events.py::test_a_real_hidden_window_receives_a_broadcast_message` 第一次真跑的结果；失败请把输出发给我。

9. **（后续轮次）** 第 09 轮的引擎读 `her_state()` 决定回复延迟（忙碌段用计划里的延迟分布引用）、睡着时不回，并处理中断期间到达的消息；第 10 轮订阅 `CandidatesExpired`（作废早于现在的候选）、按 `plan.quota.for_plan` 发主动消息、发起床问候前问 `planner.greeting_decision()`、发出后调 `record_wake_greeting`；第 11 轮的 `/时区`、`/主动` 调 `ScheduleComponent.switch_timezone` 和 `proactive.enabled`。这些都不需要你现在做什么。


## 第 13 轮（13a）—— 训练基础设施（档位模板、加密训练包、AutoDL 脚本、asyncssh 编排、训练表与模型登记）

> 13a 只做基础设施，数据集导出（`twin train export`）、防泄露导出测试、规划合成、重训提醒和 token 级模板一致性测试在 13b（第 09 轮之后）。下面是你自己环境里要做的事；沙箱没有 GPU、没有 AutoDL 实例，所以脚本里依赖真实硬件的部分（真的训练、合并、量化、vLLM）还没有跑过。

1. **租一台实例并扩容数据盘**
   - 在 AutoDL 租 RTX 5090（32GB）或 RTX PRO 6000（96GB）。镜像选 Python 3.11 或更新（建议 3.12）、PyTorch 带 CUDA 12.8 的（脚本会检查：Blackwell 上 `torch >= 2.7` 且为 CUDA 12.8 构建、`torch.cuda.get_arch_list()` 含 `sm_120`，不满足就从 cu128 索引重装；绝不接受 CUDA 13.x）。
   - **数据盘按档位扩容**（默认 50GB 不够）：`5090-8b` 至少 70GB（脚本要求 68）、`5090-14b` 与 `pro6000-14b` 至少 110GB、`pro6000-32b` 至少 230GB（脚本要求 225）。内存：`5090-14b` 要 43GB 以上（QLoRA 适配器在 CPU 上合并 bf16 基座），`5090-8b` 17GB，`pro6000-14b` 23GB，`pro6000-32b` 41GB。不够时 `setup.sh` 会停下并打印“请到 AutoDL 控制台扩容数据盘/换实例”。
   - 实例连续关机 15 天会被释放且数据全部清空（AutoDL 文档）；训练前后的产物先下载到本地。

2. **填配置和密码**
   - 在 `config/config.yaml` 里写 `autodl:`（主机、端口来自控制台的 SSH 登录命令；用户一般是 `root`；`auth: password` 或 `auth: key` 加 `key_path`；`workdir` 保持 `/root/autodl-tmp/twin`）。
   - `uv run twin secrets set autodl_password`（实例密码存进凭据管理器，不写任何文件）。
   - 第一次 `uv run twin train remote connect` 会显示主机密钥指纹并询问是否信任，对照控制台后输入 `y`；实例重建后指纹会变，按提示删除 `data/training/known_hosts` 里那一行再连。

3. **准备数据集**
   - `twin train export`（13b，见下一节）会写出数据集目录（`dataset_meta.json`、`sft_train.jsonl` 等，格式见 `training/README.md` 与 DECISIONS D-249）。不要拿真实数据手工拼数据集目录。

4. **打包与运行**（13b 完成后）
   - `uv run twin train bundle --profile 5090-8b --dataset <数据集目录>`：输入两次口令（至少 12 个字符，**不会保存**，丢了就只能重新打包）。输出在 `data/training/bundles/`，同目录有独立解密脚本。
   - `uv run twin train remote all --profile 5090-8b --dataset <数据集目录>`，或逐步：`upload`、`setup`（要再输一次口令）、`train`、`dpo`、`eval`、`export`、`download`、`cleanup`；`uv run twin train remote status` 列出所有运行，并提醒还有数据留在实例上的运行。断网后重新运行同一条命令即可：长步骤在实例上继续跑，日志从上次的位置接着拉；上传和下载都会续传并校验 sha256。
   - 下载完成后：`uv run twin model register data/models/<run_id>`；清理后到 AutoDL 控制台**释放实例**（脚本会提醒）。

5. **第一次真实运行时请留意（这些在沙箱里验证不了，请把输出贴给我）**
   - `setup.sh` 打印的 GPU 型号、驱动、计算能力、PyTorch 与 CUDA 版本和架构列表，以及它是否重装了 PyTorch。
   - bitsandbytes 4-bit（`5090-14b`）在 sm_120 上是否能训练；`setup.sh <档位> verify` 的 1 step 试跑最终用的批大小。
   - `setup.sh <档位> verify` 会运行 `twin.training.parity_check`（13b 已提供，随训练包进入实例）；它不通过或找不到时**明确报错并停下**（这是设计：模板一致性检查不能跳过）。
   - `export.sh`：llama.cpp（标签 b11177）能否在实例上编译 `llama-quantize`、转换 Qwen3 是否成功、三个量化文件的大小，以及每一步前的剩余空间。
   - `serve_vllm.sh`（第 14 轮才用到）：vLLM 0.26.0 的 cu128 轮子在 5090 / PRO 6000 上能否启动；沙箱没法下载 GitHub 发布页，所以脚本有 PyPI 回退，两条路径都没在真卡上试过（DECISIONS D-243）。

6. **费用与安全**
   - 按小时计费：训练、评估、导出连起来可能是几小时；做完立刻 `twin train remote cleanup` 并释放实例。
   - 不要把口令、实例密码写进任何文件或聊天；训练包本身经 AES-256-GCM 加密，实例上的明文数据只在解密后存在，`cleanup.sh` 会覆盖后删除，并清掉日志与数据集缓存。

## 第 13 轮（13b）—— 训练集导出、防未来泄露、规划合成、模板 token 级一致性、重训提醒

> 沙箱里没有真实聊天记录、DeepSeek Key 和 GPU：导出用合成对话 + respx 验证，模板一致性在装了固定版本 LLaMA-Factory 0.9.5 的独立环境里对真实 Qwen3 分词器跑过（205 个样本全部逐 token 一致，最长 1781 token；DECISIONS D-332）。下面是你自己环境里要做的事。

1. **前提**
   - 人设卡、画像与作息的 `pre_holdout` 版本已生成（`twin persona generate`、`twin profile rebuild`），记忆回放已覆盖要导出的日期范围（`twin memory replay start` 并批准）。缺了哪个，`twin train export` 会直接告诉你先做什么。
   - 分词器：第一次导出会下载 `tokenizer.json`（约 11MB，先 Hugging Face 再 ModelScope，核对哈希）。两处都连不上时，手动下载 `Qwen/Qwen3-8B` 仓库里的 `tokenizer.json`，用 `--tokenizer <文件或目录>` 指定。

2. **在真实数据上导出**
   - `uv run twin train export --foreground`（应用没运行时；应用在运行就去掉 `--foreground`，应用会执行）。可用 `--from 2026-01-01 --to 2026-06-30` 限定本地日期范围；切分点始终是 `holdout_cutoff()`。
   - 第一次通常停在 `waiting_for_plans`：它不写数据集，而是把“规划合成”排成一次性批任务并打印批次号与**估算费用**（上限，按峰时价）。你确认费用后：`uv run twin jobs approve <批次>`；应用不在运行时 `uv run twin jobs run --until-idle`；再运行一次 `uv run twin train export --foreground`，这次写出数据集。费用超过估算 20% 时批次会自己暂停并告警（R-LLM-014）；可以随时 `uv run twin train export-status` 看进度。
   - 完成后：`uv run twin train bundle --profile <档位> --dataset <打印出的目录>`，后面接 13a 的流程。

3. **检查导出统计**（`twin train export-status`，同样的数字在 `dataset_meta.json` 的 `stats` 里）
   - 各切分的样本数与丢弃原因。`no_user_turn_to_answer`（她主动开口的块，DECISIONS D-320）在合成数据里约占四分之一，真实数据按你们的聊天习惯会不同；`over_budget` 或 `target_too_long` 很多时请告诉我。
   - 表情包占比、表情代码占比应接近她的画像（统计里并排显示 `export` 与 `profile`），相差很大说明导出没有教到她的习惯。
   - 规划样本占训练+验证的约 30%；测试集不带规划（D-321）。
   - token 总数与各档位预计训练时长：**规划用的粗略数字**（D-327），不是测量；第一次真实训练后以 `training_runs` 为准。
   - 抽查几行 `sft_train.jsonl`：目标里没有 `[图片]`、`[语音…]` 等事件文字，上下文里有；占位符如 `[手机号#1]` 在整个数据集里指同一个号码。
   - 真实数据量大时（十万个样本量级）导出会花不少时间（每个样本一次“当时的记忆块”检索和一次向量编码）；沙箱里 440 个样本约 7 秒（用的是测试里的小向量模型），真实模型的耗时没有测过，请把你机器上的耗时告诉我。

4. **模板一致性检查**
   - 第一次 `setup.sh <档位> verify`（`twin train remote setup` 会跑）会在实例上运行 `python -m twin.training.parity_check --model-dir <基座目录> --cases <工作目录>/data/parity_cases.jsonl`，用的是实例上装的 LLaMA-Factory 0.9.5 和基座的分词器；退出码非 0 就**不要训练**，把输出贴给我（输出只有位置和 token 数，没有聊天文字）。
   - 想在自己机器上提前跑（可选）：`tests/integration/test_template_parity.py` 顶部写了怎么建独立环境（不要装进项目依赖）和需要的环境变量（`TWIN_LLAMAFACTORY_PYTHON`、`TWIN_QWEN_TOKENIZER`）。

5. **重训提醒**
   - 每次导入后自动检查，也可以手动 `uv run twin train retrain-check`：她的新消息达到上次训练数据的 10% 时发 `retrain_suggested` 告警，`/状态` 的“重训提醒”显示百分比。第一次训练完成之前没有基线，不会提醒。
   - 重训 = 重新 `twin train export`（新版本号）→ `bundle` → `remote all`，从基座全量训练，不在旧适配器上叠加。

6. **没有做、也不属于这一步的**
   - `twin train export-dpo` 和“偏好对 ≥ 200 的提示”在第 11 轮（读 `preference_pairs`）；服务、激活与上线门槛在第 14 轮。

## 第 09b 轮 —— 评估沙盒、盲测、风格指标、记忆测试与里程碑门槛框架

> 沙箱里没有真实聊天记录、DeepSeek Key 和真人：抽样、估价、批准、生成、判分界面、报告、门槛判定都用合成对话 + respx + 注入的按键流验证，沙盒的写入隔离用表级快照证明（DECISIONS D-360 至 D-373）。**因此 M0、M1、M2 没有在沙箱里通过，`eval_runs` 里没有任何一条是伪造的：下面三道门槛要靠你自己的环境和你自己的判断来过。**

1. **升级数据库**：`uv run twin db upgrade`（迁移 `0012_eval_tables`，新增 `eval_runs` 与 `eval_items`；条目正文加密）。本轮没有新增配置键。

2. **前提**
   - 已导入聊天记录；`uv run twin profile rebuild`（含 `pre_holdout` 画像）、`uv run twin persona generate`（含 `pre_holdout` 人设卡）、记忆回放覆盖到留出点之前；已设置 DeepSeek Key。缺了哪个，`twin eval blind` 在花钱之前就会一条条列出来。
   - **先和机器人聊几天**（`twin chat --local` 或微信）：风格指标的“线上”来源读它真实发出的话；记忆测试要从“她说过的、机器人编的”事实里出 10 题，事实库里这类来源的事实不到 10 条时测试直接记为“未通过(样本不足)”，**不会拿真实记录凑数**。

3. **盲测（M1 的条件，约 50 对）**
   - `uv run twin eval blind --backend deepseek --n 60`：只抽样、估价、排队，不花钱；打印运行号与批次号。建议 `--n` 比 50 多一点：机器人没回出来、回退成其他后端、回复为空的对子不会拿给你看，你跳过的也不计数，而 M1 要求**最近一次盲测**里有效判断不少于 50 个。
   - 看估算费用后批准：`uv run twin jobs approve <批次号>`（确认提示；加 `--yes` 跳过）。费用超过估算 20% 时批次自己暂停并告警（R-LLM-014）。
   - 生成：`uv run twin eval blind --resume <运行号> --foreground`（应用没在运行时在这里执行；应用在运行就去掉 `--foreground`，等它跑完）。
   - 判：全部生成后同一条命令进入判分界面。上面是同一段对话（最多 8 轮），下面左右两条候选，一条是她当时真实的回复，一条是机器人的，位置随机。`1` 选左、`2` 选右、`s` 跳过、`q` 保存退出；每判一对立刻保存，随时 `--resume` 继续。只凭直觉，不要翻聊天记录；界面不显示后端和生成方式。
   - 看报告：猜对率与 Wilson 95% 区间、按时段（凌晨 0-6、上午 6-12、下午 12-18、晚上 18-24 点）和上下文长度（1-2 轮、3-5 轮、6-8 轮）分组的猜对率。**猜对率越接近 50% 越好**（猜不出哪条是她）。M1 的线是 ≤70%（点估计）。
   - 请告诉我：猜对率、区间、哪些分组特别高，以及你凭什么认出机器人（我据此调提示词与后处理）。

4. **风格指标**
   - `uv run twin eval style --source live --days 7`：机器人最近 7 天真实发出的话 vs 她的当前（`live`）画像，六项指标逐项显示偏差，**每项 ±30% 以内为通过**，有一项不通过退出码为 1。线上来源的“引用率”显示 n/a（通道不支持引用，D-362）。
   - `uv run twin eval style --source eval_items --run <盲测运行号> --backend deepseek`：用盲测里同一批上下文上机器人的回复，对 `pre_holdout` 画像比较；同一批对子里她的真实回复的同样指标并排显示，作为参照。
   - 请告诉我：哪几项超出 ±30%，以及显示的偏差方向。

5. **记忆测试（M2 的条件，20 题）**
   - `uv run twin eval memory`：抽 20 题（10 题来自真实记录的事实，10 题来自“她说过的/机器人编的”事实），估价并排队，不花钱。
   - `uv run twin jobs approve <批次号>`，再 `uv run twin eval memory --resume <运行号> --foreground`：DeepSeek 把事实改写成日常问题，机器人（当前后端）在 live 模式下回答，DeepSeek 初评。
   - 全部出完后同一条命令进入复核界面：每题显示问题、应答出的要点、机器人的回答和 DeepSeek 的判分。回车或 `k` 保持，`c` 正确、`p` 部分正确（算 0.5 分）、`w` 错误，`q` 保存退出，随时 `--resume` 继续。**你的判分覆盖 DeepSeek 的**，只有复核完的运行才算数。
   - 总分 ≥ 80% 且 10 + 10 的构成完整才通过。
   - 请告诉我：DeepSeek 的初评与你的复核差多少、哪类题（真实记录 / 机器人对话）错得多。

6. **门槛判定**
   - `uv run twin eval gate M0`：DeepSeek 探针（`twin llm probe`）与通道探针（`twin channel probe`）的存档结果，按 SPEC 第 26 节的规则判；没做过探针会告诉你先做哪个。
   - `uv run twin eval gate M1`：最近一次包含当前默认后端的盲测，有效判断 ≥ 50 且猜对率点估计 ≤ 70%。
   - `uv run twin eval gate M2`：最近一次完成复核的记忆测试，总分 ≥ 80% 且构成完整。
   - `M3`、`M4`、`M5` 现在打印“该门槛在第 NN 轮接入”并退出码 2（后续轮次接入判定）。
   - 退出码：0 通过、1 未通过（含样本不足）、2 尚未接入或未判定；`--check` 只读最近一次存下的结论，不重新判定也不写入。每次判定（不含 `--check`）在 `eval_runs`（`kind=gate`）里留一条带证据的记录；后续轮次开工前都会先 `twin eval gate <上一个里程碑> --check`。
   - 请把 M0、M1、M2 三条命令的输出发给我。

7. **没有做、也不属于这一步的**
   - 风格模型 / hybrid 后端的盲测对比（`twin model evaluate`，赢了纯 DeepSeek 才上线，R-SRV-005）在第 14 轮；这一轮的盲测、风格指标和沙盒已经能对它们用（登记了风格模型之后 `--backend style|hybrid|all`），沙箱里是用一个登记进模型表的脚本化服务验证的，**真实模型上的结果没有看过**。
   - M3（第 10 轮）、M4（第 12 轮）、M5（第 14 轮）的判定。

## 第 09 轮 —— 回复引擎（分三步：09-1 无状态部分，09-2 后端与指令，09-3 状态机与发送）
## 第 09 轮 —— 回复引擎（状态机、真实模式决策、提示词、三种后端、后处理、拟人节奏、安全、指令）

沙箱里没有 DeepSeek Key、没有微信、没有 GPU 和风格模型：提示词、后处理、危机识别、拒绝处理、指令、状态机、后端回退与切回，都是用固定响应（`respx`）、终端通道、手动时钟和本地的“假风格模型服务”验证的。**回复质量（像不像她）、危机判定的准确度、真实的缓存命中率、真实风格模型上的表现，都没有在真实环境里看过**——下面的条目就是要你亲自看的部分。第 09 轮结束后，终端（`twin chat --local`）和微信（`twin run`）都已经能聊天、能发指令。

1. **准备（只做一次）**
   - 做什么：`uv run twin secrets set deepseek_api_key`（没有 Key 时 `twin run` 和 `twin chat --local` 会停在启动前并提示这条命令，退出码 6，不会崩溃）；`uv run twin db upgrade`（迁移 `0010_bot_turns_state_feedback`，新增 `bot_turns`、`conversation_state`、`feedback`）；`uv run twin profile rebuild`。
   - 为什么重算画像：画像里新增了两条指标。`closing_no_reply_rate`：对方说完“好的”“嗯嗯”“晚安”这类不需要回答的短话之后，她不再回复的比例，不重算则 `[不回]` 永远不会被接受。`typing_s_per_char`：她打一个字要几秒（由“连发块内两条消息之间的停顿 vs 后一条的字数”回归得到），不重算则气泡之间只有抽样的连发间隔和 1 秒下限，不再按字数变长。
   - 预期：`uv run twin profile show` 里有“对方说完结束性短句后她不回的比例”（应是小于 1 的合理比例，样本很少时可能是 0，此时机器人不会选择不回）和“打字速度（每个字的秒数）”（通常零点几秒；数据少时可能没有这一条，需要至少 3 个字数档、每档 20 个停顿）。
   - 请告诉我：不回的比例和你对她的感觉符不符（她是不是真的会对“嗯嗯”“好的”不回）；打字速度的数字，以及一条十几个字的气泡发出前她大约“打”多久像不像。

2. **检查三个词表（可编辑，随仓库提供）**
   - `config/lists/ai_phrases.txt`：**不要删除或改名里面的 `# --- …` 分组标题**——标题含 `self-identification` 的那一组是“机器人自称 AI”，含 `markdown` 的是格式符号，其余是客服腔；后处理按分组决定怎么处理。
   - `config/lists/commitment_patterns.txt`：承诺打电话、发语音、发照片、见面、转账等的句式（正则）。试聊时发现漏网的承诺句式，加一行即可。
   - `config/lists/crisis_keywords.txt`：危机关键词（故意偏宽）。命中后还要经过模型二次判定，所以多一点没关系；**没有设置 DeepSeek Key 时，命中关键词就按“危机”处理**（宁可多一句关心的话）。
   - 请告诉我：有没有你觉得会天天误触发的词（例如口头禅“笑死”“杀了你”之类），我来调整。

3. **终端试聊：`uv run twin chat --local`**
   - 做什么：几轮日常对话；看她回的气泡条数、长度、标点、表情包是否像她；试一句“你是不是 AI？”（应该用她的语气承认是模拟的，不会被删掉）；试“你给我打个电话吧”“发张照片给我”（应该自然带过，不答应）；故意发一句只有“好的”的话（偶尔应该不回）。`/quit` 随时离开，没回完的会在下一次 `twin chat --local` 或 `twin run` 时接着回。
   - **她是按真实节奏回的**：你停下约 15 秒后她才开始，再等她的回复延迟（中位数十几秒、偶尔几分钟）、再一条条“打”出来——这不是卡住。想快一点试可以临时 `--set engine.quiet_window_s=3`（只缩短“等你说完”，延迟本身没有加速开关）。没有画像（还没导入真实记录）时用 SPEC 里的参考数字（DECISIONS D-301），导入真实记录并 `twin profile rebuild` 之后才是她自己的节奏。
   - 注意：终端自带 `/help`、`/img <图片路径>`、`/quit`，由终端本身处理；机器人的指令是 `/帮助` 等（见第 4 条），英文别名（`/status`、`/think`…）可以用，`/help` 不行。
   - 请告诉我：节奏（等多久、几秒一条）像不像她、有没有不像人的地方。

4. **指令试聊（终端和微信里一样）** 依次发：`/帮助`、`/帮助 思考`、`/状态`、`/思考 自动`、`/思考：开`（中文冒号）、`／显示思考　开`（全角斜杠与全角空格）、`/后端 deepseek`、`/后端 hybrid`、`/重来`、`/不像`、`/记住 xx`。
   - 预期：每条指令的回复立刻出来（不等、不“正在输入”），以 `⚙️ ` 开头、是系统口吻、不会被改成她的语气，也不会进入记忆、检索和训练。`/帮助` 只列出 `/帮助 /状态 /思考 /显示思考 /后端 /重来` 六个（别的指令后续轮次才有，所以 `/不像`、`/记住` 现在得到“没有这个指令”加这份列表，而不是被当成聊天）；`/思考 也许` 得到用法与例子；`/后端 style`、`/后端 hybrid` 在没有登记并启用风格模型时被拒绝并说明原因；以 `/` 开头但不是指令的（微信旧文字表情 `/::)`、路径 `/home/x`）仍然是普通聊天。
   - `/重来`：先立刻回一句系统提示，然后她重新回上一轮（不再等“等你说完”，也不抽首条延迟，只有阅读时间和她的连发间隔）。上一轮回复被标成负例，不再出现在对话上下文、记忆和表情包占比里；她那轮回复里自己编的小细节会被撤销。她**还在一条条发**的时候发 `/重来`：没发出的气泡直接丢掉，再重新写一轮。`/重来` 不会撤销危机时跳出角色的回复（会告诉你不能重来）。刚重来完、新回复还没写出来时再发一次 `/重来`，得到“现在没有可以重来的回复”。
   - `/思考 开` 后她的回复会先思考一下再回；`/显示思考 开` 后每次回复后面会多一条 `⚙️ 思考：…` 系统消息（脱敏、最多 500 字，调试用）。
   - `/状态` 里：后端、思考模式、今日费用与缓存命中率（连续聊几轮后，从第二轮起应明显大于 0）、预算级别、风格模型、最近告警、“等你说完：15 秒；按你的连发习惯建议 N 秒”。“主动消息：未启用”“重训提醒：暂无”是如实的（对应的功能还没有上线），不是故障；“平台窗口”在终端里是模拟的数字。
   - 请告诉我：`/状态` 里有没有你觉得多余、缺少或看不懂的项；指令回复的措辞（在 `src/twin/commands/texts.py` 一处）有没有想改的。

5. **微信试聊：`uv run twin run`** 之后在手机上聊（先 `uv run twin channel login` 绑定）：睡着的时间段发消息，等她醒来（起床后 5–40 分钟）再回，回复像刚醒；忙碌时段发消息，回得慢；连发几条消息只回一轮；她正在一条条发的时候你又发了一条，未发出的气泡被取消、已发出的保留，后面接着上文说；回复过程中关掉 `twin run` 再启动：没发完的会接着发或重新抽延迟，不会在启动的一瞬间连发好几条。指令（第 4 条）在微信里同样立刻得到回复。请告诉我：哪一步和预期不一样。
   - 没有微信、只想跑完整应用时：`uv run twin --set channel.kind=console run` 把 `twin run` 的通道换成终端（读标准输入，输入结束即停止）。

6. **危机演练** 在测试环境里发一句明显的危机表述（例如“我真的不想活了”）：应该看到跳出角色的关心话和你所在时区对应的热线（芝加哥 988，上海 12356），**即使这时她“睡着”也不等到早晨**（等你说完那十几秒之后立刻回），并在日志里出现一行 `alert`（类别 `crisis`，不含聊天内容；告警表的命令行查看在第 12 轮）。发 `/重来` 不会撤销这条回复。**紧急联系人邮件**（`safety.emergency_contact`）要到第 12 轮接上 SMTP 才会真正发出；在那之前开启它只会得到一条“没有邮件服务”的告警。

7. **（有 Key 后）提示词的真实表现**
   - 历史以机器人的话开头时，我在最前面补了一条固定的开场 user 消息，避免“第一条不是 user”被 API 拒绝（`HISTORY_OPENER`）。真实 API 是否接受以 assistant 开头的历史我无法在沙箱里确认；第一次试聊时，如果日志里出现 `llm_attempt_failed` / 400 之类，把输出发给我。
   - 缓存命中率：连续聊几轮后，日志里 `llm_call` 一行的 `cache_hit_ratio`（以及 `/状态` 里的“缓存命中率”）从第二轮起应明显大于 0；窗口一次后移 10 轮的那一轮会短暂下降。请把前 10 轮的命中率告诉我。
   - hybrid 的规划：设置好 Key 和风格模型后 `/后端 hybrid`、`/思考 开`，聊几句，在日志里看规划的 `intent`、`tone` 是否合理（`bot_turns` 的规划字段是加密的，命令行查看在后续轮次）。规划提示词是 `src/twin/profile/templates/reply_plan.v1.md`，想改请新增 `reply_plan.v2.md`，不要改 v1（已存入数据库的版本不能被改）。

8. **（第 14 轮有风格模型后）风格模型的核对与实地演练**
   - 核对提示词和训练一致：`twin model serve` 启用前的分词核对会做（R-TRN-011.4）。如果核对报告说“渲染出的字符串与 HF 分词结果不一致”，把报告发给我；`StylePromptBuilder` 的格式只来自 `twin.training.lf_template`，不一致说明那边的模板版本或分词器设置有出入。
   - 回退与切回：`/后端 hybrid`（或 `style`）切到风格模型后，在电脑上停掉 `llama-server`（或断开 AutoDL 隧道），再给她发一条消息；然后重新启动，等待 10 分钟以上。预期：停掉之后的下一条回复由 deepseek 回（`/状态` 的“后端”一行显示“hybrid（风格模型不可用，眼下由 deepseek 回复：…）”，告警里多一条 `style_fallback`）；服务恢复并连续健康 10 分钟后自动切回，告警里多一条 `style_recovered`，`/状态` 恢复正常。每次切换在设置历史里有记录（`uv run twin settings history backend.fallback`，`by` 是 `auto`）。请告诉我：回退有没有太敏感（例如模型加载时的一次 503 就回退了——这是按 SPEC 做的，要放宽可以改 `backend.health_check_s`、`backend.recover_after_min`）。
   - 已知限制（DECISIONS D-355）：她正在一条条发时你又发了话，重新生成时 DeepSeek 后端会被告知“你刚才已经说了：…”，风格模型的提示词没有这一节（和训练格式一致），可能重复刚发出的气泡。试聊时留意风格模型有没有这种重复，告诉我。

9. **配置键与设置键（不需要做什么，有需要再调）**
   - 配置键：`backend.health_check_s`（30，看风格模型是否健康的间隔，秒）、`backend.fallback_violations`（3，风格模型的输出连续几次被判硬违规就回退）、`backend.recover_after_min`（10，健康多少分钟后切回）；`style_model.memory_tokens`（300，风格提示词里记忆块的预算，**训练集导出会用同一个值**，训练后不要再改）、`style_model.n_predict`（200）、`style_model.temperature`（0.7）、`style_model.top_p`（0.9）；`engine.quiet_window_adaptive`（默认关；`--set engine.quiet_window_adaptive=true` 让“等你说完”取你自己连发间隔的 75 分位，上限 `engine.quiet_window_max_s`，`/状态` 始终显示按画像算出的建议值）。回复节奏本身（首条延迟、忙碌段延迟、起床后抖动复用 `schedule.greeting_window_min`、连发间隔、打字速度）都来自画像和作息，没有配置键。
   - 运行时设置：`engine.paused_until`（带时区的 UTC 时间或空）由第 11 轮的 `/暂停 <时长>`、`/恢复` 写，引擎只读并遵守；现在可以手动试：`uv run twin settings set engine.paused_until 2026-10-09T20:00:00+00:00`（换成 `null` 就是恢复）。`show_thinking` 是 `/显示思考` 写的键，同样可以 `uv run twin settings set show_thinking true`；运行中的应用 2 秒内生效。`backend.fallback` 是后端回退的记录，由程序自己写，不要手改。
