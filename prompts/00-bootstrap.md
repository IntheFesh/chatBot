# 第 00 轮：工程骨架、配置、加密存储、任务队列与质量门禁

> 里程碑：M0 前置 · 前置轮次：无 · 本轮结束后仓库可运行 `twin --help`、`twin doctor`、`twin run`（空跑）。

## 先读

1. `CLAUDE.md`（全部，尤其"铁律"）
2. `docs/SPEC.md`：§1 范围、§2 架构、§3 配置、§4 存储与加密、§20 中 R-OPS-002/007/009、§24 R-PRIV-001、§25 非功能
3. `docs/TRACEABILITY.md`

先进入计划模式，列出文件清单、依赖清单、迁移内容、测试清单与风险，等我确认后再写代码。

## 本轮目标

建立一个之后各轮（00–16 与 09b）都能直接往里加功能的工程底座：依赖与质量门禁、配置与秘密、加密存储、持久化任务队列、CLI 与常驻进程的进程模型、应用生命周期与崩溃隔离、日志、Windows 单实例与防休眠、需求追踪检查脚本。

## 必须实现的需求

R-SCOPE-003、R-SCOPE-004（Windows 运行与本地数据的底座部分）、R-PRIV-004、R-ARCH-001（骨架）、R-ARCH-002、R-ARCH-003、R-ARCH-004、R-ARCH-006（框架部分）、R-CFG-001～005、R-STO-001～005（R-STO-006 中本轮相关的表）、R-OPS-002、R-OPS-007、R-OPS-009（基础检查项）、R-PRIV-001、R-NFR-004。

## 详细要求

### A. 工程与依赖
1. 按 CLAUDE.md §4 建目录；`src/` 布局，包名 `twin`，入口 `twin = "twin.cli:app"`。
2. `pyproject.toml`：Python `>=3.12,<3.13`；运行依赖至少包含 `typer`、`rich`、`pydantic>=2`、`pydantic-settings`、`pyyaml`、`sqlalchemy>=2`、`alembic`、`cryptography`、`keyring`、`httpx`、`openai`、`ijson`、`tzdata`、`chinese-calendar`、`holidays`、`orjson`、`structlog`（或标准 logging + JSON formatter，二选一并说明理由）、`ulid-py` 或等价库（应用侧生成主键）。开发依赖：`pytest`、`pytest-asyncio`、`pytest-cov`、`respx`、`time-machine`、`hypothesis`、`ruff`、`mypy`、类型存根。之后各轮按需追加依赖，不在本轮预装用不到的大包（如 `sentence-transformers`）。
3. `ruff`（含 `DTZ` 规则以禁止 naive datetime、`S` 安全规则、`T20` 禁止 print 于库代码）、`mypy --strict` 覆盖**全部** `src/twin`、`pytest` 配置（`asyncio_mode=auto`；标记 `integration`、`live`（真实网络，默认跳过，需 `TWIN_LIVE=1`）、`windows`（仅 Windows 运行））、覆盖率配置（统计全部 `src/twin`，总体 ≥ 85% 且每个子包 ≥ 75%；写一个小脚本 `scripts/coverage_gate.py` 读取 coverage JSON 按子包检查，纳入 `scripts/check.ps1` 与 CI）。
4. `.gitignore` 覆盖 CLAUDE.md 铁律第 6 条列出的全部路径；`config/config.example.yaml` 与 SPEC R-CFG-004 完全一致（无秘密）；`config/lists/` 下提供 `ai_phrases.txt`、`commitment_patterns.txt`、`crisis_keywords.txt` 的初版词表（每行一条，`#` 注释，内容由你按 SPEC R-ENG-008、R-SAFE-001/002 编写完整，不是示例）。
5. 测试目录约定：`tests/unit`、`tests/integration`、`tests/fixtures`、`tests/support`（共享测试替身，如可快进时钟、本地 HTTP 测试服务）；写一个扫描测试保证 `src/` 中没有任何 `import tests`。
6. `.github/workflows/ci.yml`：在 `windows-latest` 与 `ubuntu-latest` 上跑 ruff、mypy、pytest（不含 live）、覆盖率门禁。另提供 `scripts/check.ps1` 本地一键检查。

### B. 配置与秘密（R-CFG）
1. `twin.config.Settings`：嵌套 pydantic 模型严格对应 R-CFG-004 的每个键；加载顺序 CLI > 环境变量（前缀 `TWIN_`，嵌套用 `__`）> YAML > 默认；未知键报错。
   - R-CFG-005 测试：解析 `docs/SPEC.md` 中 R-CFG-004 的 YAML 代码块、`config/config.example.yaml` 与 `Settings()` 的默认值，三者键集合与默认值逐一相等；之后每一轮新增参数都会被这个测试检查。
2. `twin secrets set|delete|list|check <name>`：写入/删除 keyring（服务名 `wechat-twin`），`list` 只显示名字与是否存在，永不打印值。
3. 运行时设置 `RuntimeSettings`：存 `settings` 表，带类型的读写接口与变更历史（谁、何时、旧值、新值，值加密）；首次启动从配置文件初始化。
4. 启动时打印生效配置（秘密打码）；`consent.confirmed_at` 缺失或不是合法日期时拒绝启动并给出明确提示（R-SCOPE-003）。

### C. 存储与加密（R-STO）
1. SQLAlchemy 2（声明式、类型标注）；SQLite 连接设置 `journal_mode=WAL`、`foreign_keys=ON`、`busy_timeout`；所有表 `created_at`/`updated_at` 为 UTC。
2. 主键在应用侧生成（ULID 或 UUIDv7），以便加密时把 `(表名, 主键, 列名)` 作为 AES-GCM 的 AAD。
3. `EncryptedText` / `EncryptedJSON` 自定义列类型：AES-256-GCM、每次写入随机 12 字节 nonce、密文格式带版本字节与密钥 id（为轮换做准备）。
4. 主密钥：首次运行生成 32 字节随机密钥存 keyring，带 `key_id`；读取失败时给出可操作的错误。`KeyRing` 管理"当前密钥 + 已退役密钥"，解密按密文中的 `key_id` 选择密钥。`twin secrets rotate-db-key`（独占命令）：生成新密钥，逐表逐行重加密，进度存 `settings`，可中断续跑；完成后旧密钥标记为已退役但**不删除**——由第 12 轮的备份保留策略在没有备份引用它时删除（R-STO-003）。
5. `MediaStore`（R-STO-004）：流式分块加密（例如 1MB 分块、每块独立 nonce、文件头含版本/密钥 id/块大小/明文 sha256），文件名用明文 sha256；`put(stream|bytes, kind)`、`open()`（流式解密）、`temp_file()` 上下文管理器（受控临时目录，用后覆盖并删除）、`verify(sha256)`。
6. Alembic：本轮迁移建 `settings`、`jobs`、`cost_ledger`、`alerts`、`channel_state` 五张表；之后每轮为自己的表新增迁移。迁移在启动时自动检查版本并提示执行 `alembic upgrade head`（或 `twin db upgrade`）。

### D. 持久化任务队列（R-ARCH-003）
1. `jobs` 表字段：`id, type, payload(EncryptedJSON), priority, status(pending|running|done|failed|cancelled), attempts, max_attempts, run_after, offpeak_only, deadline, last_error(已脱敏), created_at, updated_at`。
2. 处理器注册表：`@job_handler("type")`；未注册类型的任务保持 pending 并在 `twin jobs list` 中标注"无处理器"（这是正常排队，不是桩）。
3. Worker：并发上限可配；失败按指数退避重试；超过次数标记 failed 并触发告警钩子；进程重启后把 `running` 状态的任务恢复为 `pending`。
4. `offpeak_only` 任务的调度判断通过依赖注入的 `OffPeakPolicy` 接口完成；本轮提供"总是允许"的实现仅用于测试注入，生产实现在第 01 轮（R-LLM-007）注册——本轮在注释与 TRACEABILITY 中注明依赖关系。
5. CLI：`twin jobs list|show|retry|cancel|run --until-idle|approve <批次>`（`approve` 供 R-LLM-014 一次性批任务确认；批次概念与费用估算字段本轮建好，估算逻辑由各业务轮次提供）。

### D2. 进程模型（R-ARCH-006 框架）
1. `@command(kind=READ|LIGHT|HEAVY|EXCLUSIVE)` 装饰器包装全部 typer 命令；一个测试遍历 CLI 全部命令，断言每个都已声明类别（之后各轮新增命令同样被检查）。
2. 两把锁：`run` 锁（`twin run` 获取，G.1）与 `supervisor` 锁（第 12 轮的 `twin supervise` 获取），名字不同、互不冲突；`EXCLUSIVE` 命令检测到任何一把被占用时拒绝执行并提示先停止应用；`HEAVY` 命令只负责入队（`--foreground` 时若应用未运行则在前台用同一个 Worker 执行到完成）。
3. `state_version`：`LIGHT` 命令写库后在同一事务里递增 `settings.state_version`；应用中的 `StateWatcher` 组件每 2 秒读取，变化时广播"状态已变化"事件，各组件订阅并失效自己的缓存（本轮实现机制与测试，各业务缓存由后续轮次订阅）。
4. SQLite 多进程：`busy_timeout`、短事务；写一个"两个进程同时写 `jobs` 与 `settings`"的测试。

### E. 应用骨架（R-ARCH-001/004）
1. `twin.app.Application`：组件以 `Component` 协议注册（`start`、`stop`、`health`），按依赖顺序启动、逆序停止；Ctrl+C 与 Windows 关闭事件下优雅退出。
2. 组件内后台任务由 `Supervisor` 托管：任务异常时记录、调用告警钩子、指数退避后重启该任务，不影响其他组件；连续崩溃超过阈值则标记组件不健康。
3. `twin run` 启动应用（本轮只有任务队列组件与健康心跳组件）。

### F. 日志（R-OPS-007）
1. JSON 行日志，按 10MB × 10 份轮转，写 `data/logs/`；同时人类可读的控制台输出。
2. 日志 API 区分"内容字段"（如 `content=`、`text=`）：INFO 及以上自动丢弃这些字段；DEBUG 时先经过脱敏函数（本轮提供接口与基础正则，完整规则在第 01 轮 R-LLM-009 中实现并替换为同一函数）。有测试证明 INFO 级别日志中不会出现正文。

### G. Windows 相关（R-OPS-002）
1. 单实例：`InstanceLock(name)`，Windows 用命名互斥体（`CreateMutexW`），其他平台用文件锁；本轮 `twin run` 使用名为 `run` 的锁，第二个 `twin run` 启动时立即退出并提示；不同名字的锁互不影响（第 12 轮 `supervise` 用 `supervisor`）。
2. 防休眠：运行期间调用 `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`，退出时恢复；封装为可注入的 `PowerManager`，非 Windows 平台实现为记录"该平台不需要"的真实实现类（不是空函数），并有单测。
3. 启动时强制 UTF-8 控制台输出。

### H. 时钟
提供 `Clock` 协议（`now_utc()`、`monotonic()`、`sleep()`）与真实实现、测试实现（可快进）；全项目只能通过它获取时间（ruff 规则 + 一个扫描测试禁止直接调用 `datetime.now()`/`time.time()`，白名单仅限 `Clock` 实现文件）。

### I. 自检与追踪
1. `twin doctor`（R-OPS-009 基础项）：Python 版本、依赖可导入、`tzdata` 可用（`ZoneInfo("America/Chicago")` 与 `ZoneInfo("Asia/Shanghai")`）、keyring 可读写测试项、数据目录可写与剩余空间、数据库迁移版本。后续轮次在同一命令中追加检查项。
2. `scripts/trace_check.py`：解析 `docs/SPEC.md` 中全部 `R-xxx-nnn`，解析 `docs/TRACEABILITY.md` 的"需求实现表"，检查：每个编号都有一行；状态为"已实现"的行必须填写实现位置与测试名；测试名必须能在 `pytest --collect-only -q` 结果中找到；实现位置的文件与符号必须存在（用 AST 查找符号）。支持 `--round <轮次>` 只检查该轮负责的编号——轮次标识是两位数字加可选小写字母（如 `00`、`09b`），轮次列中逗号分隔；对多轮共同负责的编号，非最后负责轮次只要求"实现位置"中有本轮的部分记录。每轮结束都要跑。
3. 隐私扫描（R-PRIV-001）：`scripts/privacy_scan.py` 扫描仓库已跟踪文件中的 wxid 模式、手机号、身份证号、`messages.json` 片段特征等；作为 pre-commit 钩子与 CI 步骤。

## 测试要求
- 配置：优先级、未知键报错、秘密打码、consent 缺失拒绝启动。
- 加密：往返、篡改密文/替换到别的行（AAD 不匹配）必须失败、密钥轮换中断后续跑成功、`MediaStore` 大文件（>50MB 合成数据）流式往返与 sha256 校验。
- 任务队列：重试退避、重启恢复 running→pending、无处理器任务保持 pending、`offpeak_only` 受策略控制。
- 进程模型：命令类别全覆盖、独占命令在锁被占用时拒绝、`state_version` 变化被运行中的应用在 2 秒内感知、两进程并发写。
- 配置三处一致（R-CFG-005）。
- 密钥：轮换后旧密文仍可用退役密钥解密。
- 生命周期：组件异常被隔离并重启，其余组件不受影响。
- 日志：INFO 不含正文。
- 单实例：第二实例退出（Windows 标记测试 + 跨平台逻辑测试）。
- 时钟扫描测试。
- `trace_check.py` 自身的单测（构造小型 SPEC/TRACEABILITY 样例）。

## 验收
```
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src/twin
uv run pytest -q --cov=src/twin --cov-report=json
uv run python scripts/coverage_gate.py
uv run twin --help
uv run twin doctor
uv run python scripts/trace_check.py --round 00
```

## 不要做
- 不要实现任何聊天、通道、LLM 逻辑；不要预建后续轮次的表。
- 不要把任何秘密写进配置文件或测试快照。

## 完成后汇报
按 CLAUDE.md §6 第 6 条格式汇报，并在 `docs/TRACEABILITY.md` 中更新本轮负责的全部编号。
