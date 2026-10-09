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
   - 预期：`twin chat --local` 先打印"The persona engine is not connected yet (it arrives in round 09) …"，你输入的每条消息得到一行 `(received N character(s) of text; the engine is not connected, no reply)`——这是引擎（第 09 轮）接入前的真实行为，不是故障；机器人发表情包时显示 `bot: [表情包：<标签>] <文件路径>`、发其他允许的图片时显示 `bot: [图片] <路径>`、发之前显示 `对方正在输入…`（表情包标签第 06 轮之前显示"未标注"）。

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
