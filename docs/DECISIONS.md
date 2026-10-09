# 决策记录（DECISIONS）

> 本文件记录三类内容：
> 1. **SPEC 偏差**：SPEC 与现实冲突、或实现与 SPEC 文字不完全一致之处（按 CLAUDE.md 铁律 9，选最接近原意的做法并附证据）；
> 2. **取舍记录（R-SCOPE-009）**：设计冲突时按"像她 > 记得住 > 像真人 > 可控 > 会成长"取舍的每个点，附采用的顺序、优先级理由与固定该行为的测试名（第 16 轮用脚本核对每条都指向真实存在的测试）；
> 3. **实现决策**：SPEC 留白处的工程选择及理由。
>
> 每轮把新增条目追加到对应小节，编号 `D-nnn` 递增，不改旧条目的编号。

## 1. SPEC 偏差

| 编号 | 轮次 | SPEC / 铁律原文 | 实际做法 | 证据与理由 |
| --- | --- | --- | --- | --- |
| D-001 | 00 | CLAUDE.md 铁律 10 与 R-CFG-002：秘密存 `keyring`（Windows 凭据管理器） | **Windows 上仍只用凭据管理器**（不可用时 `twin doctor` 报 FAIL，不回退）。**非 Windows 平台**在系统 keyring 不可用时回退到 `twin.config.secrets.EncryptedFileKeyring`：AES-256-GCM 加密文件 `secrets.enc`（位于 `$TWIN_SECRETS_DIR` 或 `~/.config/wechat-twin/`，**不在** `data/` 下），密钥来自 `TWIN_KEYRING_PASSPHRASE`（scrypt，N=2^15）或同目录权限 0600 的随机密钥文件。 | 开发/CI 沙箱（Linux 无桌面）里 `keyring.get_keyring()` 是 `keyring.backends.fail.Keyring`，任何读写都会抛错，`twin secrets`、数据库主密钥、`twin doctor` 均无法工作。回退是真实可用的加密实现（不是明文、不是桩），并由 `twin doctor` 明确标为 WARN 且给出诊断文字；`TWIN_KEYRING_BACKEND=system\|file` 可强制选择。未使用 `keyrings.alt`：它的加密文件后端每次要交互输入口令，且 `PlaintextKeyring` 违反铁律 10。测试：`tests/unit/test_secrets.py`、`tests/unit/test_doctor.py::test_keyring_check_variants`。 |
| D-002 | 00 | R-CFG-004 配置键清单 | 新增 `jobs: { concurrency: 2 }`（离线任务 Worker 并发上限，1–32），同时写入 `Settings`、`config/config.example.yaml` 与 SPEC R-CFG-004。 | 第 00 轮提示词要求"Worker 并发上限可配"，而 R-CFG-004 没有对应键；这是 CLAUDE.md 铁律 11 / R-CFG-005 允许的唯一无需事先征得同意的 SPEC 修改。三处一致性由 `tests/unit/test_config_consistency.py` 守护。 |
| D-003 | 00 | R-STO-002："列类型为自定义 `EncryptedText`/`EncryptedJSON`；AAD 绑定表名与主键" | 列类型 `EncryptedText`/`EncryptedJSON` 确实存在，但它们只存放并**拒收非 `SealedBlob` 的值**；加解密由模型上的描述符 `sealed_text()`/`sealed_json()` 完成（`row.payload` 读取时解密、赋值时以该行的 `(表名, 主键, 列名)` 作 AAD 加密）。 | SQLAlchemy 的 `TypeDecorator.process_bind_param` 看不到所属行，无法拿到主键，因此"纯列类型"做不到 AAD 绑定行。拆成"列类型守门 + 描述符加解密"后，明文无法被误写入加密列（`test_plaintext_cannot_be_written_to_an_encrypted_column`），换行/换列/换表的密文均解密失败（`test_ciphertext_copied_to_another_row_fails_to_decrypt`、`test_ciphertext_moved_between_columns_or_tables_fails`）。 |

## 2. 取舍记录（R-SCOPE-009）

> 优先级：像她 > 记得住 > 像真人 > 可控 > 会成长。下表是 SPEC 要求至少列出的取舍点，由对应轮次填写"采用的顺序 / 优先级理由 / 测试名"；第 00 轮只建骨架，**不填写尚未实现的行为**。

| 取舍点 | 负责轮次 | 采用的顺序 | 优先级理由 | 固定该行为的测试名 |
| --- | --- | --- | --- | --- |
| 预算降级顺序（R-LLM-008） | 01 | （待第 01 轮填写） | （待填写） | （待填写） |
| 平台条数不足时的气泡合并（R-ENG-009） | 09 | （待第 09 轮填写） | （待填写） | （待填写） |
| 睡眠与回复（R-SCOPE-006、R-ENG-003） | 09 | （待第 09 轮填写） | （待填写） | （待填写） |
| 表情包控频（R-STK） | 06 | （待第 06 轮填写） | （待填写） | （待填写） |
| 风格模型与 DeepSeek 后端的选择（R-SRV-005、R-ENG-006） | 14 | （待第 14 轮填写） | （待填写） | （待填写） |
| 记忆预算与例子预算的裁剪顺序（R-ENG-005） | 09 | （待第 09 轮填写） | （待填写） | （待填写） |

## 3. 实现决策

| 编号 | 轮次 | 主题 | 决策与理由 |
| --- | --- | --- | --- |
| D-101 | 00 | 日志库 | 使用标准库 `logging` + 自写 JSON 格式化器，不引入 `structlog`：只需要一个轮转文件处理器、一个控制台处理器和"INFO 及以上不含正文"这一条规则，标准库足够，第三方库本来就走标准库日志。内容字段（`content`/`text`/`body`/`prompt`/…）经 `StructLogger` 在 INFO 及以上直接丢弃，DEBUG 先过 `twin.llm.redaction.redact`；异常只记录类型、脱敏截断的消息和调用位置，不记录局部变量。CLI 与常驻进程写不同文件（`twin-cli.log` / `twin.log`），避免两个进程轮转同一个文件（Windows 上会失败）。 |
| D-102 | 00 | 主键 | 自己实现 ULID（`twin.storage.ids`，时间来自当前 `Clock`，同毫秒内单调递增），不引入 `ulid-py`：测试里用可快进时钟时 ID 有确定的排序；主键必须在应用侧先于写入存在（AAD 依赖主键）。 |
| D-103 | 00 | 同步 SQLAlchemy | 数据库访问用同步 SQLAlchemy 2 + `asyncio.to_thread`（`Database.arun`），不用 `aiosqlite`：CLI 命令是同步的、Alembic 是同步的，一套代码两边复用；SQLite 本地事务极短。读用延迟 `BEGIN`，写一律 `BEGIN IMMEDIATE`（通过连接执行选项实现），避免两个进程互相升级锁造成死锁，`busy_timeout` 让写者排队等待。 |
| D-104 | 00 | `settings` 的变更历史 | 本轮只允许建五张表（`settings`/`jobs`/`cost_ledger`/`alerts`/`channel_state`），因此变更历史存放在 `settings.history`（加密 JSON 列，保留最近 100 条 `{at, by, old, new}`），而不是另建历史表。内部键（`state_version`、`heartbeat`、轮换进度）不记录历史。 |
| D-105 | 00 | 非高峰策略 | `OffPeakPolicy` 协议在 `twin.ops.jobs`；"总是允许"的实现只在 `tests/`。生产默认是 `DeferredOffPeakPolicy`（不允许 `offpeak_only` 任务运行，直到其 `deadline`），第 01 轮用 `set_offpeak_policy()` 注册真正的高峰判定（R-LLM-007）。本轮没有任何代码入队 `offpeak_only` 任务，所以不存在被卡住的任务。该依赖关系已写入 TRACEABILITY 的 R-ARCH-003 行。 |
| D-106 | 00 | `cost_ledger` 列 | 只建第 01 轮提示词第 34 行列出的列（模型、purpose、各类 token、费用、是否高峰、耗时、是否思考、请求 id）；`account`、`batch_id` 由第 01 轮按其提示词用迁移追加，避免重复建列。 |
| D-107 | 00 | 一次性批任务字段 | `jobs` 表已有 `batch_id`、`estimated_cost_usd`、`requires_approval`、`approved_at`、`approved_usd`。`requires_approval` 且未批准的任务永远不会被 Worker 领取；`twin jobs approve <批次>` 校验批次估算不超过 `budget.one_time_usd`（超出要求拆批）后记录批准。估算逻辑由各业务轮次提供。 |
| D-108 | 00 | `twin run` 的进程类别 | 声明为 EXCLUSIVE，并 `acquires=("run",)`、`tolerates=("supervisor",)`：它自己持有 `run` 锁，其他实例拒绝启动；它由 `twin supervise`（持有 `supervisor` 锁）拉起，所以不把 `supervisor` 锁当作冲突。`twin jobs run --until-idle` 是 HEAVY 命令的前台执行者，应用运行中默认拒绝（`--force` 可强制）。 |
| D-109 | 00 | READ 命令的只读保证 | READ 类命令通过 `WritePolicy(read_only=True)` 使任何 `Database.transaction()` 抛出 `ReadOnlyViolationError`，LIGHT 类命令使写事务默认在同一事务内递增 `settings.state_version`；`RuntimeSettings.set` 仅在值真的改变时递增。 |
| D-110 | 00 | 密钥轮换范围 | `twin secrets rotate-db-key` 除数据库所有加密列（从 schema 自动发现，以后新表自动纳入）外，也重加密 `data/media` 下的媒体文件，这样旧密钥标记为"已退役"后，除备份外没有现存数据再依赖它。进度存 `settings['rotation.progress']`，但正确性不依赖进度：每个值按自身头部的 `key_id` 判断是否已迁移。 |
| D-111 | 00 | `channel.kind` | `Settings.channel.kind` 接受 `ilink` 与 `console`（R-ARCH-005 的 `LocalConsoleChannel`）；SPEC 默认值 `ilink` 不变。第 02 轮实现通道时可再收紧。 |
| D-112 | 00 | R-STO-005 | LanceDB 在第 05 轮才引入；本轮提供向量库记录的**模式守卫** `twin.storage.vector_schema`（只允许 `id`/`vector`/`at`/`kind` 四列，拒收任何文本列），第 05 轮的索引写入必须经过它。 |
| D-113 | 00 | 覆盖率口径 | 覆盖率统计全部 `src/twin`（行覆盖）；仅 Windows 才会执行的 ctypes 代码块用 `# pragma: win32-only` 排除（`exclude_also`），Windows 逻辑通过可注入的 `Win32` 协议与测试替身在所有平台上都有测试，真实绑定由 `@pytest.mark.windows` 测试在 `windows-latest` 上覆盖。`mypy --platform win32 src/twin` 在本地也做过一遍检查。 |
| D-114 | 00 | 单实例锁的生命周期与 Windows 互斥体作用域 | `InstanceLock` 的锁持续到 `release()` 或对象被回收（`weakref.finalize` 调用后端的幂等 `release()`），两个平台一致；此前被丢弃而未释放的对象会让 Windows 命名互斥体句柄（POSIX 上是原始 fd）一直开到进程结束。互斥体名 `Local\wechat-twin-<name>` 仍是整个登录会话范围、**不**随 `locks_dir` 变化（同一台电脑同一登录会话只允许一个机器人实例，与 R-OPS-002 一致）；因此测试若在 Windows 上忘记释放锁，会污染之后所有取同名锁的测试（第 00 轮 Windows CI 的 8 个 `test_process_model` 失败即由此而来，见 `test_instance_lock.py::test_a_lock_whose_object_is_dropped_is_freed`）。 |
