# 发布检查清单（打 `v1.0.0` 标签之前）

> 这份清单**不预先勾选任何一项**：每一项都要在打标签的那个提交上重新做一遍，做完的人在方框里打勾并把结果记在最后的“记录”里。
> “状态”一列说的是现在的情况：**可自动检查**的项目在开发环境和 CI 里每次都会跑，但打标签前仍要在目标提交上看一次结果；**待用户执行**的项目需要你的真实环境（Windows 电脑、微信、DeepSeek Key、AutoDL、7 天观察期），沙箱里没有做过，也不能代做——没有任何一条探针结果、盲测结果、门槛记录或演练结果是伪造的。
> 命令都在仓库根目录用 PowerShell 执行（CI 上是 bash，等价）。`tests/unit/test_docs_commands.py` 保证本文件里写出的 `twin` 命令都真实存在。

## A. 自动检查（开发机或 CI；打标签的那个提交上必须全绿）

| ☐ | 检查 | 命令 | 谁 / 在哪里做 | 状态 |
| --- | --- | --- | --- | --- |
| ☐ | A1 依赖锁定一致 | `uv lock --check` 然后 `uv sync --frozen` | 维护者 / 任意开发机 | 可自动检查 |
| ☐ | A2 ruff 零告警 | `uv run ruff check .` 与 `uv run ruff format --check .` | CI `lint` 作业（Linux 与 Windows） | 可自动检查 |
| ☐ | A3 类型检查：全部 `src/twin` 通过 `mypy --strict` | `uv run mypy src/twin`；Linux 上再 `uv run mypy src/twin --platform win32` | CI `lint` 作业 | 可自动检查 |
| ☐ | A4 全部测试通过（live 测试除外） | `uv run pytest -q -m "not live"`；CI 上是 Linux 3 片 + Windows 5 片的矩阵（[`CI.md`](CI.md)） | CI `tests` 作业（两个操作系统） | 可自动检查 |
| ☐ | A5 覆盖率：全部 `src/twin` ≥ 85%，每个子包 ≥ 75% | `uv run pytest -q --cov=src/twin --cov-report=json` 然后 `uv run python scripts/coverage_gate.py`；CI 的 `coverage` 作业合并各片后用同一道门槛 | CI `coverage` 作业 | 可自动检查 |
| ☐ | A6 集成测试 | `uv run pytest -q -m integration` | 开发机 | 可自动检查 |
| ☐ | A7 长时间运行（14 个模拟日，内存/队列/数据库大小/异常计数） | `uv run python scripts/soak.py --days 14 --accelerated`（第 16 轮 A 部分的开发工具） | 开发机 | 可自动检查 |
| ☐ | A8 **CI 全绿**：目标提交上 `ci gate` 检查通过 | GitHub → Actions → 该提交的 `ci` 运行；或 `gh run list --commit <SHA>` | 维护者 / GitHub | 可自动检查 |

## B. 需求与代码审计（全部可自动检查）

| ☐ | 检查 | 命令 | 谁 / 在哪里做 | 状态 |
| --- | --- | --- | --- | --- |
| ☐ | B1 **`trace_check` 全绿**：SPEC 里每个 `R-xxx` 都是“已实现”，有真实存在的实现位置和被 pytest 收集到的测试 | `uv run python scripts/trace_check.py`（不带 `--round`） | CI `lint` 作业 / 开发机 | 可自动检查 |
| ☐ | B2 隐私扫描绿：已跟踪文件里没有 wxid、手机号、身份证号、银行卡号、导出片段、密钥 | `uv run python scripts/privacy_scan.py`；总审计 `uv run pytest -q tests/unit/test_privacy_audit.py` | CI `lint` 作业 / 提交前钩子 | 可自动检查 |
| ☐ | B3 **`stub_scan` 绿**：`src/` 里没有 TODO/FIXME/stub/placeholder/mock/fake/dummy 等关键词、没有只有 `pass`/`...`/常量返回的函数、`src/` 不导入 `tests`；白名单里每一条都有理由 | `uv run python scripts/stub_scan.py` | CI `lint` 作业 / 开发机 | 可自动检查 |
| ☐ | B4 **`decisions_check` 绿**：`DECISIONS.md` 编号唯一、引用的 `D-xxx` 都存在、取舍表覆盖 SPEC 列出的全部取舍点、引用的测试都存在 | `uv run python scripts/decisions_check.py` | CI `lint` 作业 / 开发机 | 可自动检查 |
| ☐ | B5 对外联系审计（R-SAFE-004）与非目标审计（R-SCOPE-008） | `uv run pytest -q tests/unit/test_outbound_audit.py tests/unit/test_nongoals_audit.py` | CI `tests` 作业 | 可自动检查 |
| ☐ | B6 文档与代码一致（命令、指令、组件名、需求表） | `uv run pytest -q tests/unit/test_docs_commands.py tests/unit/test_docs_architecture.py tests/unit/test_traceability_table1.py` | CI `tests` 作业 | 可自动检查 |
| ☐ | B7 历史里从来没有提交过真实数据 | `git log --all --name-only --pretty=format: -- data exports models backups "*.db" "*.enc" "*.gguf"` 应没有任何输出 | 维护者 / 任意开发机 | 可自动检查 |

## C. 真实环境（**待用户执行**，在你要常驻运行的那台 Windows 电脑上）

| ☐ | 检查 | 命令 / 做法 | 谁 / 在哪里做 | 状态 |
| --- | --- | --- | --- | --- |
| ☐ | C1 **`twin doctor` 绿**：`keyring` 是 `ok`（`WinVaultKeyring`），`deepseek-net`、`ilink-api`、`ilink-cdn`、`scheduled-task`、`power-plan`、`gpu`/`llamacpp`（若用本机风格模型）都正常；`holiday-calendar` 的警告要理解是什么意思 | `uv run twin doctor` | 你 / 目标 Windows 电脑 | **待用户执行** |
| ☐ | C2 全新安装演练：`install.ps1` 在干净的账户或虚拟机上一路走完，`twin service status` 显示 `InteractiveToken`、`PT0S`、`IgnoreNew` | `powershell -ExecutionPolicy Bypass -File scripts\windows\install.ps1 -StartNow` 然后 `uv run twin service status` | 你 / Windows 电脑 | **待用户执行** |
| ☐ | C3 **备份恢复演练完成**：做备份 → 校验 → 停应用 → 恢复 → 启动 → 检索库与健康检查正常；再做一次“密钥轮换后恢复旧备份” | 见下面的“备份恢复演练” | 你 / Windows 电脑（先在无关紧要的数据上做） | **待用户执行** |
| ☐ | C4 一键删除演练（在演练数据上，不在真实数据上）：列出将删除的内容、键入短语、删完后备份无法解密 | `uv run twin service stop` 然后 `uv run twin purge --all`（**只在演练用的数据目录上做**） | 你 / Windows 电脑 | **待用户执行** |
| ☐ | C5 告警邮件与 Windows 通知真的送达；邮件里没有聊天内容、没有二维码 | `uv run twin setup`（发测试邮件）；让微信登录失效或断网一次，看通知、邮件和二维码窗口 | 你 / Windows 电脑 + 手机 | **待用户执行** |

### 备份恢复演练

先在一份无关紧要的数据上做（`PENDING_USER_ACTIONS.md` 第 12 轮第 5 项）：

```powershell
uv run twin backup now
uv run twin backup list
uv run twin backup verify <文件>            # 完整解密并核对
uv run twin service stop                    # 恢复是独占命令
uv run twin backup restore <文件>            # 会先把现有数据备份成 pre-restore-*.bak.enc
uv run twin service start
uv run twin health                          # 通道、数据库、备份新鲜度
uv run twin retrieval stats                 # 检索库与数据库对得上；对不上就 uv run twin retrieval rebuild
uv run twin secrets rotate-db-key           # 先 uv run twin service stop；轮换后再对旧备份做一次 backup verify
```

通过标准：`verify` 和 `restore` 无报错；恢复后 `twin health` 没有红色项；轮换密钥之后旧备份仍能 `verify`。

## D. 里程碑门槛报告（**待用户执行**；报告要是实测的，不是模板）

门槛由 `twin eval gate` 读取最新评估结果判定，**不得为了通过而降低阈值或改统计口径**。把每条命令的完整输出留作发布记录（`twin eval gate M<n> --check` 只读已存的结论）。

| ☐ | 里程碑 | 命令 | 过关条件 | 状态 |
| --- | --- | --- | --- | --- |
| ☐ | D1 **M0** 技术验证 | `uv run twin llm probe`、`uv run twin channel probe start`…`report`，然后 `uv run twin eval gate M0` | `docs/LLM_REPORT.md` 与 `docs/CHANNEL_REPORT.md` 的文件头是 `measured`（不是 `pending`）；主动消息在窗口内稳定送达（R-CH-010）；DeepSeek 探针通过 | **待用户执行** |
| ☐ | D2 **M1** 能聊、像她 | `uv run twin eval blind --backend deepseek --n 60` … `uv run twin eval gate M1` | 盲测猜对率 ≤ 70%，有效判断 ≥ 50 对 | **待用户执行** |
| ☐ | D3 **M2** 记得住 | `uv run twin eval memory` … `uv run twin eval gate M2` | 记忆测试 ≥ 80%（10+10 构成完整） | **待用户执行** |
| ☐ | D4 **M3** 有作息、会主动 | 连续真实运行 7 天后 `uv run twin eval proactive --days 7`、`uv run twin eval gate M3` | 深睡时段主动 0 次、每天次数在范围内、间隔与追发零违规、该周 `/评分` ≥ 4/5 | **待用户执行** |
| ☐ | D5 **M4** 持续成长 | 无人值守 7 天后 `uv run twin eval stability --days 7`、新的盲测、`uv run twin eval gate M4` | 无人值守 7 天（含一次断网演练）；盲测猜对率 ≤ 60% | **待用户执行** |
| ☐ | D6 **M5** 风格模型（**只在训练过时**） | `uv run twin model evaluate <模型号>` … `uv run twin eval gate M5` | R-SRV-005；不通过就保留 DeepSeek 后端并在发布说明里写明，这**不阻塞**发布 | **待用户执行**（若训练） |
| ☐ | D7 汇总报告（第 15 轮合并后） | `uv run twin eval report`，成本 `uv run twin eval cost --month <本月>`，一致性 `uv run twin eval consistency --days 7` | `data\reports\eval-<日期>.md` 列出 M0–M5 的门槛、当前值、样本数和证据来源，不含聊天正文；日常月费用 ≤ 15 美元 | **待用户执行** |

> **R-CH-010：** 如果 M0 的通道探针说 ClawBot 在窗口外不能稳定地主动推送（窗口 < 12 小时，或一次入站后连发 < 3 条），**不要发布**——停下来向维护者报告，企业微信通道要另开一轮（[`RUNBOOK.md`](RUNBOOK.md) 第 13 章）。

## E. 打标签

| ☐ | 步骤 | 命令 | 谁做 |
| --- | --- | --- | --- |
| ☐ | E1 版本号：`pyproject.toml` 的 `version`、`src/twin/__init__.py` 的 `__version__` 改成 `1.0.0`，`uv lock` 更新锁文件，提交 | `uv lock` ; `uv run twin --version` 应打印 `wechat-twin 1.0.0` | 维护者 |
| ☐ | E2 A、B 两节的检查在这个提交上全绿（CI 的 `ci gate`） | 见 A8 | 维护者 |
| ☐ | E3 C、D 两节已完成，结果记在下面的“记录” | — | 你 |
| ☐ | E4 打标签并推送 | `git tag -a v1.0.0 -m "wechat-twin 1.0.0"` ; `git push origin v1.0.0` | 维护者 |

## 记录（打标签时填写）

| 项 | 内容 |
| --- | --- |
| 提交 | `<SHA>` |
| CI 运行 | `<链接>` |
| `trace_check` / `stub_scan` / `decisions_check` / `privacy_scan` | `<输出摘要>` |
| `twin doctor`（目标电脑） | `<警告项及原因>` |
| 备份恢复演练 | `<日期、备份文件名、结果>` |
| M0–M4 门槛（`twin eval gate M<n> --check`） | `<每个里程碑的输出>` |
| M5 门槛（若已训练） | `<输出；未通过时写明保留 DeepSeek 后端>` |
| 已知问题 | `<发布说明里要写的>` |
