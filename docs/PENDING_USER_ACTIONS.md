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

协议文档 `docs/ILINK_PROTOCOL.md`（步骤 02a）已根据官方源码写好（`@tencent-weixin/openclaw-weixin` 2.4.9，并对照 AstrBot 4.28.2 的 `weixin_oc`）。**官方源码已由沙箱拉取阅读，你不需要安装 Node.js**（除非想自己复核：`npm pack @tencent-weixin/openclaw-weixin`）。沙箱里没有微信账号，所以下面这些步骤**从未真实运行过**：`docs/CHANNEL_REPORT.md` 在拿到你的实测数据之前只能是"待实测"模板，源码读不出来的行为（主动发送窗口、可连发条数、`ret:-2` 的真实含义等，清单见协议文档第 13 节）都等这些步骤的结果。02b 步骤已提供 `twin channel login / status / send-test / unbind` 与 `twin doctor` 里的 WeChat 连通性检查（下面第 1–3 项现在可以做，**全部用合成响应测试过，从未对真实服务运行过**）；`twin channel probe` 与 `report` 随 02c、`twin chat --local` 与 `channel echo-test` 随 02d 合入，合入之前第 4、5 项不可用。

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
   - 做什么：`uv run twin channel probe start`，按终端和微信里的提示依次操作。每一步开始前，终端和微信都会提示你先给机器人发一条新消息；完成后才开始该步。第 3 步（窗口）开始后约 25 小时内**不要给机器人发任何消息**，否则该步作废、需要重做。过程中探针会问你：GIF 在手机上是否在动、是否看到"对方正在输入"，并且每步结束后问"手机上实际收到几条 [测试] 消息"（接口返回成功不一定等于手机收到，以你手机上的数为准）。源码不支持发送引用，所以不会有引用子步骤。
   - 预期：`uv run twin channel probe status` 随时可看进度；全部完成后 `uv run twin channel probe report` 生成 `docs/CHANNEL_REPORT.md`（只含测量结果与时间），并给出 `channel.proactive_window_safe_h` 与 `channel.outbound_quota_safe` 的建议值（实测值留 10% 余量）。**建议值需要你确认后才会写入配置**。
   - 如果报告写"未达标"（窗口 < 12 小时或连发条数 < 3）：这是提示词要求的"停下来告诉我"——主动消息（第 10 轮）依赖这两个条件。把报告发给维护者，由你决定下一步（企业微信通道不在本规格范围，需要另开一轮）。
   - 提交 `docs/CHANNEL_REPORT.md`（只含技术结果，不含任何对话内容）。

5. **（可选）本地控制台通道**
   - 做什么：`uv run twin chat --local`（引擎在第 09 轮接入前只能用 `uv run twin channel echo-test` 验证通道本身），输入 `/img <路径>` 发送图片。
   - 预期：终端里双向聊天，机器人发表情包时显示 `[表情包：标签] <文件路径>`，显示"对方正在输入…"。
