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

6. **确认 Windows CI 修复（第 00 轮首次 `windows-latest` 运行有 13 个失败，已按根因修复，沙箱无法验证）**
   - 做什么：看下一次推送后 GitHub Actions 里 `quality (windows-latest)` 任务，或本机 `uv run pytest -q`。
   - 预期：全绿。涉及的修复：`InstanceLock` 对象回收时释放命名互斥体（`src/twin/ops/instance_lock.py`）、`test_instance_lock.py` 里泄漏锁的用例改为显式释放、`test_cli.py` 里"运行期间持有锁"改用锁本身探测（Windows 上锁是互斥体而不是 `run.lock` 文件）、`test_secrets.py` 三个用例显式注入"无系统凭据库"（Windows 上真实存在 `WinVaultKeyring`）、`test_jobs.py::test_concurrency_is_bounded` 改为事件驱动（不再依赖 10 ms 的 `sleep` 与 Windows 上较慢的 SQLite 提交）。
