# 第 12 轮：运维——Windows 常驻、健康检查、告警、费用报告、加密备份与恢复、一键删除、回滚

> 里程碑：M4（本轮完成后无人值守观察 7 天判定门槛）· 前置：第 00–11 轮全绿，`twin eval gate M2 --check`（第 11 轮复判）已通过 · 需要用户：配置 SMTP（发件邮箱与应用专用密码存 keyring）、运行安装脚本、7 天无人值守运行期间做一次断网演练、观察期末做 50 对新的盲测

## 先读
`CLAUDE.md`（§7 Windows）；`docs/SPEC.md` §20 运维全部、R-SCOPE-004、R-CH-003/008、R-LLM-005/008、R-PRIV-003/005、R-SRV-002（llama-server 子进程健康纳入检查）、R-TRN-012（重训提醒告警）。

## 本轮目标
让机器人在用户的 Windows 电脑上 7×24 稳定运行：开机登录自动启动、崩溃自动重启、防休眠、单实例；出问题第一时间通知用户；费用透明；数据每天加密备份、可恢复；她要求时能一键彻底删除。

## 必须实现的需求
R-OPS-001～010、R-SCOPE-004、R-SAFE-001（紧急联系人提醒邮件的发送通路）、R-ARCH-006（独占命令与 `twin service` 的接入）、R-STO-003（退役密钥随备份保留策略清理）、R-PRIV-003（清理状态记录的本地侧）、R-PRIV-005、R-EVAL-006（稳定性评估）、R-EVAL-010（注册 M4 判定器）、R-STO-006（本轮的表：`health_snapshots`、`backup_records`）。

## 详细要求

### A. 安装与常驻（R-OPS-001/002）
1. `scripts/windows/install.ps1`（PowerShell 5.1 兼容，`Set-StrictMode -Version Latest`，出错即停）：检查/安装 uv（官方安装方式）、`uv sync --frozen`、运行 `twin setup` 向导（DeepSeek Key、SMTP、目标会话、时区、consent 确认）、`twin db upgrade`，然后调用 `twin service install` 注册计划任务：
   - 触发器：当前用户登录时；登录类型 `InteractiveToken`（只在我登录时运行——keyring、Windows 通知与二维码窗口都需要用户会话；不要用 S4U/密码存储或 Windows 服务）；
   - 动作：虚拟环境里的 `.venv\Scripts\twin.exe supervise`（或 `uv run --frozen --no-sync twin supervise`，开机时不得改动环境），工作目录为仓库根，环境变量 `PYTHONUTF8=1`；
   - 设置：无执行时限；电池供电也运行；不允许多实例并行；任务自身"失败后每 1 分钟重启"作为第二道保险（它主要覆盖启动失败，进程运行中崩溃由 `supervise` 处理）；
   - 在输出中说明：若希望断电/系统更新重启后无人值守恢复，需要开启 Windows 自动登录（并说明风险）。
2. `twin supervise`：以子进程运行 `twin run`（同一个 Python 环境），子进程异常退出时按 5 秒→5 分钟指数退避重启（稳定运行 30 分钟后退避复位），每次重启写日志与 `alerts`（同类限频）；收到停止信号时先让子进程优雅退出再结束；`supervise` 持有名为 `supervisor` 的单实例锁，子进程 `twin run` 持有 `run` 锁，两者互不冲突；子进程处于退避等待时 `supervisor` 锁仍被持有，因此独占命令仍会被拒绝（有测试）。
3. `twin service install|uninstall|start|stop|status`（独占类别的 install/uninstall，其余轻量）：封装计划任务操作（`schtasks` 或 Task Scheduler COM 接口，二选一并说明理由）；`status` 显示任务状态、`supervise` 与 `run` 的进程状态、最近重启次数。R-ARCH-006 中"先停止应用"的提示从本轮起改为 `twin service stop`。
4. `uninstall.ps1`：调用 `twin service uninstall`（不删除数据，删除数据用 `twin purge`）。
5. 进程内：第 00 轮的单实例与防休眠在 `twin run` 中启用；优雅退出时完成正在发送的气泡、持久化状态、终止子进程（llama-server）并关闭进程内的 `asyncssh` 隧道——用 Windows Job Object 保证主进程意外退出时子进程也被结束。

### B. 健康检查（R-OPS-003）
- `HealthMonitor` 组件每分钟检查：通道登录状态与最近一次成功长轮询时间（> 5 分钟视为异常）、DeepSeek 最近错误率与熔断状态、风格模型端点（若后端需要）、磁盘剩余（< 5GB 告警）、任务队列积压（> 500 或最老任务 > 24 小时）、最近备份时间（> 36 小时）、预算级别；结果写 `health_snapshots`（保留 30 天）。
- 每周日当地 03:30 执行 SQLite `PRAGMA integrity_check` 与向量库一致性检查（向量 id 都能在数据库找到）。
- `twin health [--json]`。

### C. 告警（R-OPS-004）
1. `AlertService`：类别枚举（login_lost、deepseek_failure、circuit_open、budget_80、budget_level_n、one_time_overrun、style_model_down、style_tokenize_mismatch、backup_failed、backup_mirror_unavailable、disk_low、queue_backlog、retrain_suggested、crisis_detected、channel_window_unexpected、channel_poll_stale、calendar_out_of_range、sleep_timezone_suspect、process_restarted、system_resumed）；之前各轮写入 `alerts` 表的告警全部改由本服务发出；去重与限频（同类 1 小时最多 1 次，恢复时发一次"已恢复"）。
2. 渠道：邮件（SMTP over SSL/STARTTLS，凭据 keyring，HTML + 纯文本，正文不含聊天内容）+ Windows 通知（`windows-toasts` 或等价包）；邮件失败不影响主流程，记录并重试。
3. 登录失效流程：告警 → 本机弹出二维码窗口（复用第 02 轮登录）→ 扫码成功后自动恢复长轮询并发"已恢复"通知；二维码永不通过邮件发送。
4. 紧急联系人提醒（R-SAFE-001）：只有 `safety.emergency_contact.enabled=true` 且填写了邮箱时，`crisis_detected` 告警额外向该邮箱发送一封固定模板邮件（只有时间与"可能需要关心他"，模板对象没有任何正文字段，有测试）；同一事件只发一次；`twin setup` 中询问是否开启并说明会发送什么。

### D. 费用（R-OPS-005）
- `twin cost report [--month YYYY-MM]`：按日、按用途、按模型、缓存命中率、高峰/非高峰占比、与预算对比；每月 1 日当地 09:00 发上月月报邮件。

### E. 备份与恢复（R-OPS-006）
1. 每天当地 `backup_hour_local` 点（避开正在发送的会话，必要时顺延）：SQLite 在线备份 API 生成一致性快照 → 与向量库目录、媒体清单（媒体文件本身已加密且按 sha256 命名，增量拷贝到备份目录的媒体池）一起打包 → AES-256-GCM 加密（当前主密钥派生的备份密钥，文件头记录 `key_id`）→ `data/backups/twin-<当地日期>.bak.enc`；写 `backup_records`（大小、sha256、耗时、`key_id`）。
2. 保留策略：14 个日备份 + 8 个周备份（每周日的那份），超出删除；删除后检查已退役的数据库密钥（第 00 轮 `KeyRing`），没有任何保留备份引用的退役密钥才从 keyring 删除（R-STO-003）。
3. `twin backup now|list|verify <文件>|restore <文件>`：恢复前停止应用、备份当前数据到 `data/backups/pre-restore-<时间>`、恢复后跑迁移与完整性检查。
4. 异地副本：配置 `ops.backup_mirror_dir`（外接盘等）时同步拷贝加密备份并按同样策略清理；目录不可用时告警但不影响本地备份。

### F. 一键删除（R-OPS-008、R-PRIV-005）
1. `twin purge --all`（独占命令）：列出将删除的类别与数量（数据库各表行数、媒体文件数、向量表、备份数与异地副本、训练集、本地模型与适配器、报告、日志）→ 要求输入确认短语"删除她的全部数据" → 确认应用已停止（否则提示 `twin service stop`）→ 删除 → 从 keyring 删除全部数据库密钥（含已退役的）与备份密钥（加密粉碎）→ 输出删除报告（只含数量）。没有 `--confirm` 之类跳过确认的参数。
2. `twin purge --training-only`：只删训练集、训练包、远程清理记录核对（提示到 AutoDL 控制台确认实例已释放）、本地风格模型文件。
3. 有集成测试：purge 之后任何旧备份都无法解密。

### G. 自检与回滚（R-OPS-009/010）
1. `twin doctor` 增补：通道主机连通性、DeepSeek 连通性与余额错误识别、`chinese-calendar` 是否覆盖当前与下一年、显卡与驱动（`nvidia-smi`，用于本地风格模型）、计划任务是否已注册且登录类型为 `InteractiveToken`、Windows 电源计划是否会休眠（读取 `powercfg /query` 的睡眠超时并提醒）。
2. `twin rollback profile|persona|prompt-template|style-model <版本>`：统一入口调用各模块的回滚 API，记录回滚审计。

### H. 稳定性评估与 M4 判定器（R-EVAL-006、R-EVAL-010）
1. `twin eval stability --days 7`：基于 `health_snapshots`、`alerts` 与 `supervise` 重启记录：进程连续运行时长、重启次数与原因、长轮询中断时长、每次通道异常从发生到告警（Windows 通知）的延迟（必须 ≤ 10 分钟），结果写 `eval_runs(kind=stability)`。
2. 断网演练：`twin ops drill network`（只读类命令）打印演练步骤——拔网线或关闭 Wi-Fi 15 分钟后恢复；之后 `stability` 报告中应能看到一次"长轮询中断"及其告警延迟。
3. 注册 M4 判定器：最近连续 7 天由计划任务启动、无人值守运行（应用累计不可用时间 ≤ 10 分钟，重启后自动恢复）、观察期内至少一次断网演练且告警延迟 ≤ 10 分钟、观察期末一次新的盲测（第 09b 轮框架，新的上下文，有效判断 ≥ 50 对）当前默认后端猜对率 ≤ 60%；学习（第 11 轮）与增量导入（`/导入` 或 `twin import`）在观察期内各至少成功执行一次。

## 测试要求
- 计划任务注册脚本：用 PowerShell Pester 测试或 Python 侧生成的任务 XML 快照测试（仅 Windows 标记）。
- 健康检查每一项的阈值；告警去重、限频、恢复通知；邮件发送用本地 SMTP 测试服务器（`aiosmtpd`）。
- 备份：生成、加密、保留策略、恢复往返一致（行数与随机抽样内容）、篡改检测。
- purge：删除完整、密钥删除后旧备份无法解密、确认短语不符时不删除。
- 子进程清理：主进程被强制结束时子进程也结束（Windows 标记测试）。
- `supervise`：子进程崩溃后按退避重启、正常退出不重启、退避复位；计划任务 XML 中登录类型为 `InteractiveToken`、动作不是 `uv sync`。
- 备份：密钥轮换后旧备份仍可恢复；退役密钥在没有备份引用后才删除；异地副本目录不可用时告警。
- 稳定性统计与 M4 判定器（构造健康快照与告警记录：通过、中断过长、告警延迟超限、观察期未满、缺盲测）。

## 验收
```
uv run pytest -q
powershell -ExecutionPolicy Bypass -File scripts/windows/install.ps1
uv run twin doctor
uv run twin health
uv run twin backup now && uv run twin backup verify <文件>
uv run twin service status
uv run python scripts/trace_check.py --round 12
```

## 门槛检查（M4；安装后无人值守观察 7 天，未通过就停在本轮）
```
uv run twin ops drill network          # 观察期内做一次断网演练
uv run twin eval stability --days 7
uv run twin eval blind --backend deepseek --n 50   # 观察期末，新的上下文
uv run twin eval gate M4               # 7 天无人值守 + 掉线 10 分钟内提醒 + 盲测 ≤ 60%
```
未通过时：根据稳定性报告或盲测分析修正，然后重新开始 7 天观察（盲测未通过时按第 09b 轮的改进流程处理）。

## 不要做
- 不要把二维码、聊天内容或任何秘密放进邮件。
- 不要用 Windows 服务（LocalSystem 账户无法读取用户的凭据管理器）；按本轮要求用计划任务。

## 完成后汇报
按 CLAUDE.md 格式；附安装后计划任务状态截图说明（文字描述即可）、一次备份与校验的结果。
