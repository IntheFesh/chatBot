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
