# 执行约定(云端沙箱一次性实现全部轮次)

> 本文件记录"在 Claude Code 云端沙箱里把 00–16 轮一口气实现完"时,对提示词包原流程所做的适配。
> 它不改变 SPEC 与 CLAUDE.md 的任何需求;只说明哪些**人工/真实环境**环节被推迟到用户手里。

## 环境事实

- Linux 容器(不是 Windows),Python 3.12 由 `uv` 管理,无 GPU,无微信客户端,无 DeepSeek Key,无真实聊天记录,无 AutoDL 实例。
- 可联网(经代理):PyPI、npm、HuggingFace、官方文档站。DeepSeek 与 iLink 主机可达,但没有凭据。
- Windows 专属代码(命名互斥体、`SetThreadExecutionState`、Task Scheduler、`WM_POWERBROADCAST`、Job Object、PowerShell 脚本)照常完整实现,用 `sys.platform` 守卫,打 `windows` 标记的测试只在 `windows-latest` CI 上跑;同时提供跨平台的逻辑测试。

## 流程适配

| 提示词包原要求 | 本次做法 |
| --- | --- |
| 每轮先进计划模式、等用户确认 | 用户已授权整个项目,不进计划模式、不等确认;计划写在代码与 TRACEABILITY 里 |
| 第 02 轮协议文档写完先暂停 | 由编排者(主会话)复核协议文档后再继续,文档保留在 `docs/ILINK_PROTOCOL.md` |
| 需要真实数据/真实 Key/扫码/GPU/人工盲测/观察期的步骤 | **实现全部代码、工具、测试(合成数据)**;该步骤本身记入 `docs/PENDING_USER_ACTIONS.md`,由用户在自己的环境里执行 |
| 每轮开始前确认上一个里程碑门槛已过 | 门槛依赖真实数据与人工判断,沙箱里无法通过。门槛**判定器**照常实现并用合成评估数据做单测;`--check` 前置检查在沙箱里跳过,并在 `PENDING_USER_ACTIONS.md` 里写明 |
| 门槛未过就停在该轮 | 同上:判定器不放水(不降阈值、不改口径、不伪造 `eval_runs`),只是后续轮次的**代码**不因为"门槛需要人工"而被阻塞 |

## 硬性禁止(不因环境受限而放松)

- 不得伪造任何 live 探针结果、盲测结果、门槛通过记录;`docs/LLM_REPORT.md`、`docs/CHANNEL_REPORT.md` 在没有真实探针数据前只能是"待实测"模板,不得出现编造的数字。
- 测试里的 mock/fake 只放 `tests/`(见铁律 2)。
- SPEC 与现实冲突时,不静默降级:选最接近原意的做法,写入 `docs/DECISIONS.md` 的"SPEC 偏差"一节,附证据,并在汇报里标出。

## 测试编写约定(来自第 00 轮 Windows CI 与并发抖动的教训)

CI 同时在 `ubuntu-latest` 与 `windows-latest` 上跑,沙箱里只有 Linux,所以下面这些要在写测试时就避免:

1. **不依赖睡眠时长或固定的让出次数**。用 `asyncio.Event`/屏障或 `tests/support/waiting.wait_until` 等条件;对 `ManualClock`,启动组件后先 `await wait_until(lambda: clock.pending_sleepers >= n)` 再 `advance()`。
2. **句柄与锁要显式释放**。Windows 的命名互斥体、文件句柄不会随对象丢弃而关闭,会污染同一 pytest 进程里后面的测试;打开的文件在 Windows 上也不能被删除。用 `try/finally` 或上下文管理器。
3. **平台与后端别靠宿主机默认值**。例如 Windows 上存在真实的 `WinVaultKeyring`,测试要显式注入/monkeypatch 所需后端,而不是假定默认是 fail 后端。
4. **POSIX 专属语义**(信号、文件权限、`flock`、`fork`)的测试必须带平台跳过原因,并在 Windows 上有表达同一需求的对应测试。
5. **路径与编码**:一律 `pathlib`;读写文本显式 `encoding="utf-8"`;不要假设 `/` 或 `\n`;临时目录用 `tmp_path`。
6. 并发/时序类测试在本机用 `(for i in 1 2 3; do uv run pytest -q & done; wait)` 三路并发跑一遍,通过才算稳。
7. **"观察点"要是已写下的状态,不是"通道收到了"**。通道先拿到气泡,引擎随后才写行和状态;库里先有入站行,引擎随后才入队。Windows 上线程往返加 SQLite 提交要几十毫秒,在这个间隙里观察、`stop()`、`restart()`、读 `snapshot()` 的测试会稳定失败。等 `EngineComponent.handled`(消息已入队)、`tests/support/engine_harness.bubbles_written()`(气泡已记下)这类条件;推进手动时钟之前先 `waiting_for(engine, 时刻)`,等引擎真的在等(它的睡眠是相对注册那一刻的,时钟先动了就会晚醒同样久)。写完时序类测试后用 `SLOWDB_MS=30 uv run pytest -p tests.support.slow_db <文件>` 再跑一遍(每次数据库读写都慢 30 毫秒,在 Linux 上复现 Windows 的时序),0 毫秒与 30 毫秒都要通过。
