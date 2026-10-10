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
8. **进程级的全局注册只能借用并还原,不能“收尾为 None”**。整套测试在 CI 里是一个进程按文件名顺序跑完的,本机的单文件/分片运行看不到跨文件的污染(D-480:`test_memory_*` 注销了 `bot_turns` 读取器,后面的 `test_proactive_decide` 的规划器就没有历史)。要换掉读取器注册、off-peak 策略、密钥环、活动时钟这类“整个进程一个槽”的东西,用 `use_bot_turn_reader`/`use_keyring`/`use_clock` 这样的上下文管理器,或者在夹具里记下进入时的值、收尾时还原;不要在 `finally` 里写 `register_…(None)`。依赖这类全局状态的测试替身(如 `tests/support/proactive_world.py::build_world`)自己安装所需的那一份,不假设“导入时已经注册过”。改了这类状态的测试,除了单文件通过,还要和后面的文件放在一起跑一次(例如 `pytest tests/unit/test_memory_recent.py tests/unit/test_proactive_decide.py`)。
9. **测试不访问网络,忘了 mock 的测试会失败**(第 16 轮 A,D-559)。`tests/conftest.py` 的自动夹具 `no_network` 拒绝一切不是本机的连接(回环与 Unix 套接字放行),记录下来,测试结束时失败——客户端吞掉错误也逃不掉。新代码路径如果真的要连外面,用 `respx` mock;只有 `live` 标记、或设了 `TWIN_LIVE=1` 的 `integration` 测试可以出门。环境里的 `HTTPS_PROXY` 在守卫内被清掉(本地代理在回环上,否则请求会悄悄出去)。
10. **端到端场景用 `tests/support/life_world.py`**(第 16 轮 A,D-553、D-554):整个 `twin run`(`assemble()`)在 `LifeClock` 上跑几天,假 DeepSeek 按请求内容回答。写新场景:从 `tests/integration/life/test_life_*.py` 里挑一个最近的抄;每个场景末尾调 `tests/support/life_checks.py` 里对任何故事都成立的断言(其中 `assert_bot_text_not_in_her_data` 守铁律 7,`make_world` 夹具在建好世界时已记下"真实数据的样子",场景不用自己取快照);"进程被杀"用 `world.kill()` / `world.restart()`(取消全部任务,不调任何 `stop`);另一个进程跑命令行用 `world.cli(...)`。观察"消息被接走"等条件时等 `world.handled`,别等一个固定的虚拟时间。写完用 `SLOWDB_MS=30 uv run pytest -p tests.support.slow_db <文件>` 与三路并发各跑一遍(一个整天的场景在 `SLOWDB_MS=30` 下要几分钟)。
11. **时间相关的新模块要有测试**(第 16 轮 A,D-560):用了 `zoneinfo`、`twin.clock`、`twin.schedule.time_service` 的模块,必须有一个直接导入它、并且自己控制时间的测试文件,否则 `tests/unit/test_time_coverage_scan.py` 失败。控制时间 = 注入时钟(`clock` 夹具、`ManualClock`、`LifeClock`)或造固定的带时区时刻交给代码当 `now`。夏令时切换日(2026-03-08、2026-11-01)与时区切换日各至少有一个测试。
12. **测试的数据与被测命令必须读同一个时钟,不能有"只在写下它的那一天才对"的测试**。CLI 命令用 `build_services` 自己建一个 `SystemClock`,而测试夹具的数据是按 `ManualClock`(固定在 2026-10-09)写的;"最近 24 小时""本月"一类的断言于是在写下它的 24 小时后/下个月 1 日自己坏掉(`test_ops_service`、`test_ops_cost` 就是这样)。`CliContext.clock` 可以注入(根命令的回调会保留它,和 `secrets`、`http_transport` 一样),测试里的 CLI 辅助函数把夹具的时钟传进去。验证方法:用 `time-machine` 把真实时钟整体拨快 40 天后跑一遍全套(一个小的 pytest 插件在 `pytest_configure` 里 `time_machine.travel(now + 40 天, tick=True).start()`);**拨钟扫描里有两类误报,不用修**:C 库的时钟不受 `time-machine` 影响(OpenSSL 校验刚生成的自签证书会说"还没生效",文件的 mtime 是操作系统写的),所以 SMTP 往返类测试和用 `os.utime` 与真实 mtime 混比的测试会假失败。
