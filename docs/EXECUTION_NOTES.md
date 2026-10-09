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
