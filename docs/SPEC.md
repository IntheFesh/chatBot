# wechat-twin 技术规格（SPEC）

> 版本：1.1（2026-10-08）。来源：产品文档《微信拟人聊天机器人 · 产品文档》。1.1 依据独立审阅修订：里程碑门槛前移到对应轮次、训练与推理逐 token 一致、训练与评估防未来信息泄露、事件文字与照片不外发、CLI 与常驻进程的进程模型、平台条数与一次性费用预算、配置三处一致。
> 本文件是实现的唯一事实源。每条需求有编号 `R-<模块>-<序号>`。"必须"= 不实现即该轮未完成；"应当"= 默认实现，若有充分理由可在汇报中说明并经用户同意后调整。
> 文中"她"= 用户的女朋友（被模仿对象），"用户"= 仓库主人，"机器人"= 本系统扮演的她。

---

## 0. 参考数据（来自 7 天样本，全部记录导入后重算，仅作测试基准与默认值来源）

| 指标 | 她 | 用户 |
| --- | --- | --- |
| 文字消息长度中位数 / 90 分位 | 5 / 10 字 | 11 / 47 字 |
| 文字里带逗号比例 | 2.9% | 44.5% |
| 以句号结尾比例 | 0.2% | 0.4% |
| 连发条数中位数 / 90 分位 / 最大 | 2 / 6 / 56 | 1 / 4 / 37 |
| 连发条间隔中位数 | 6 秒 | 12 秒（75 分位 33 秒） |
| 回复延迟中位数 / 90 分位 | 17 秒 / 240 秒 | 17 秒 / 103 秒 |
| 表情包占全部消息 | 10.5%（86 种，前 3 种占 42%） | 16.3% |
| 带微信表情代码的文字比例 | 3.5% | 6.8% |
| 引用回复占文字类消息 | 6.4% | — |
| 超过 1 小时沉默后先开口 | 7 天 25 次（约 3.6 次/天） | 7 天 12 次 |

样本中她每小时消息数（芝加哥时间 0–23 点，7 天合计）：`244,62,75,44,27,28,13,64,118,81,245,352,14,4,14,137,94,0,7,59,129,180,144,78`。这 7 天里有多个低谷（04–07 点、12–15 点、17–19 点），单凭样本无法确定她的睡眠时段——作息推断必须稳健并请用户确认（R-ACT-003）。

---

## 1. 范围与约束（R-SCOPE）

- **R-SCOPE-001** 单用户系统：只服务仓库主人一人；机器人只在用户微信的 ClawBot 会话中出现。
- **R-SCOPE-002** 模仿对象是导出记录中目标会话的对方；`isSent=false` 为她，`isSent=true` 为用户。
- **R-SCOPE-003** 她已知情并授权（2026-10-08）。配置中记录 `consent.confirmed_at`，缺失时程序拒绝启动。
- **R-SCOPE-004** 运行环境：用户的 Windows 10/11 x64 常开电脑（不休眠）。全部聊天记录只在本地导入和保存。
- **R-SCOPE-005** 机器人时区可切换：默认 `America/Chicago`，用户回国后切到 `Asia/Shanghai`；夏令时自动处理。
- **R-SCOPE-006** 只有"真实模式"：回复可以慢，睡觉时不回、醒来再回；不实现"随叫随到"模式。
- **R-SCOPE-007** 两个模型分工：DeepSeek API 负责理解、记忆、规划、看图与默认生成；AutoDL 微调的风格模型作为可切换的生成后端。
- **R-SCOPE-008** 非目标（v1 不做，代码中也不得出现半成品）：语音/视频通话、发送语音、生成她的照片、给除用户外任何人发消息、群聊。
- **R-SCOPE-009** 体验目标按优先级：像她 > 记得住 > 像真人 > 可控 > 会成长。设计冲突时按此顺序取舍。可验证的落实方式：`docs/DECISIONS.md` 逐条列出每个取舍点（至少：预算降级顺序、平台条数不足时的气泡合并、睡眠与回复、表情包控频、风格模型与 DeepSeek 后端的选择、记忆预算与例子预算的裁剪顺序），写明采用的顺序、对应的优先级理由和固定该行为的测试名；第 16 轮用脚本核对每条都指向真实存在的测试。

## 2. 架构（R-ARCH）

- **R-ARCH-001** 单进程 asyncio 应用 `twin.app`，组件：通道层、人设引擎（回复编排、主动调度、指令与学习）、LLM 层（DeepSeek、风格模型）、本地存储（原始记录、画像、检索库、记忆库、表情包库）、离线任务队列。
- **R-ARCH-002** 包划分见 CLAUDE.md §4；模块之间通过明确的接口（`Protocol` / 抽象基类）依赖，禁止循环导入。
- **R-ARCH-003** 离线任务（摘要、画像重算、图片描述回填、训练集计划合成等）进入持久化任务队列（SQLite 表 `jobs`），按优先级与 DeepSeek 非高峰时段执行，可重试、可查看、重启后继续。
- **R-ARCH-004** 任何单次对话处理中的异常不得使主循环退出；异常记录、告警（按 R-OPS）并以"延迟回复"方式降级。
- **R-ARCH-005** 提供完整的本地控制台通道 `LocalConsoleChannel`（与微信通道同接口），用于开发、评估和离线演练。
- **R-ARCH-006** 进程模型（CLI 命令与常驻的 `twin run` 并存）：
  1. 两把单实例锁：`run` 锁由 `twin run` 持有，`supervisor` 锁由监督进程 `twin supervise`（R-OPS-001）持有，互不冲突；其他 CLI 命令在应用运行时也可以执行。
  2. 每个 CLI 命令必须声明类别（装饰器；有测试遍历全部命令确认都已声明）：**只读**（直接读库）；**轻量修改**（写库后递增 `settings.state_version`；运行中的应用每 2 秒检查该值，变化时使相关缓存失效并调和——例如重读人设卡、重建今日计划、按新后端启停 llama-server）；**重任务**（导入、画像与作息重算、人设卡生成、记忆回放、检索重建、表情包打标签、训练集导出等）一律进入 `jobs` 队列，由运行中的应用执行，应用未运行时由 `twin jobs run --until-idle` 在前台执行；**独占**（恢复备份、一键删除、密钥轮换、数据库迁移）检测到两把锁中任何一把被持有时拒绝执行并提示先停止应用（第 12 轮之后为 `twin service stop`）。评估命令只写评估表与费用账目，按轻量修改处理。
  3. 首次全量导入在后台跑、进度可查：`twin import` 默认入队后立即返回；`twin import status` 显示阶段、已处理条数、速度、预计剩余时间与各导入后钩子的进度。
  4. SQLite 多进程访问：WAL + `busy_timeout`，写事务短小；有"两个进程同时写"的测试。

## 3. 配置（R-CFG）

- **R-CFG-001** 配置来源优先级：命令行 > 环境变量 > `config/config.yaml` > 默认值；`pydantic-settings` 校验，启动时打印生效配置（秘密打码）。
- **R-CFG-002** 秘密（DeepSeek Key、SMTP 密码、数据库主密钥、iLink 登录凭据）只存 `keyring`（Windows 凭据管理器），`twin secrets set <name>` 交互写入。
- **R-CFG-003** 运行时可变设置（时区、思考模式、后端、主动范围、暂停状态、作息修正等）存数据库 `settings` 表，由指令修改，重启后保留；配置文件只提供初始值。
- **R-CFG-004** 必须包含的配置项与默认值：

```yaml
consent: { confirmed_at: "2026-10-08" }
paths: { data_dir: "./data", export_dir: null }       # export_dir 指向用户本地的全部导出
target: { username: null }                            # 目标会话 wxid，首次导入时确认
time:
  bot_timezone: "America/Chicago"
  source_timezone: "America/Chicago"
  source_timezone_ranges: []          # 例 [{from: "2025-01-01", to: "2025-08-31", tz: "Asia/Shanghai"}]；未覆盖的日期用 source_timezone
deepseek:
  base_url: "https://api.deepseek.com"
  chat_model: "deepseek-flash"
  offline_model: "deepseek-flash"      # 可改 deepseek-v4-pro
  vision_model: "deepseek-flash"
  timeout_s: { non_thinking: 60, thinking: 180 }
  max_concurrency: 4
  reasoning_effort: "high"             # low|high|max，只在思考开启时发送
pricing_usd_per_mtok:                  # 2026-10 官方价格页，高峰价
  deepseek-flash: { cache_hit: 0.006, cache_miss: 0.30, output: 1.20 }
  deepseek-v4-pro: { cache_hit: 0.044, cache_miss: 1.32, output: 3.96 }
pricing:
  offpeak_multiplier: 0.5              # 官方取消非高峰优惠时改为 1.0
  extra_offpeak_dates: []              # 额外按非工作日处理的北京日期
  extra_peak_dates: []                 # 额外按工作日处理的北京日期
budget: { daily_usd: 1.00, monthly_usd: 15.00, alert_ratio: 0.8, one_time_usd: 30.00,
          degrade_ratios: [1.0, 1.25, 1.5, 2.0] }   # 进入降级级别 1/2/3/4 时的花费占预算比例（R-LLM-008）
thinking: { chat: "off", proactive_planner: "on", auto_rules: true }   # off|on|auto
backend: { active: "deepseek", health_check_s: 30, fallback_violations: 3,
           recover_after_min: 10 }   # deepseek|style|hybrid；第 09 轮新增：后端回退的 3 个键（R-SRV-004）
jobs: { concurrency: 2 }                # 离线任务 Worker 并发上限（第 00 轮新增，R-ARCH-003）
ingest:                                 # 第 03 轮新增，R-IMP-004/008/012
  batch_size: 2000                     # 导入时每批事务写入的消息条数
  caption_recent_days: 90              # 图片描述只为最近 N 天的图片排队
  caption_wait_timeout_s: 30           # 用户刚发来的图片在线同步描述的超时（秒）
  sticker_download: { concurrency: 4, per_second: 4, retries: 3, timeout_s: 30 }   # 表情包下载
engine:
  quiet_window_s: 15
  quiet_window_adaptive: false         # true 时按用户连发间隔 p75 自适应，上限 quiet_window_max_s
  quiet_window_max_s: 45
  max_wait_s: 90
  history_turns_min: 30
  history_turns_max: 40                # 超过时一次后移 10 轮（缓存友好）
  examples_k: 8
  max_bubbles: 8
  ai_phrases_file: "config/lists/ai_phrases.txt"
  commitment_patterns_file: "config/lists/commitment_patterns.txt"
profile: { burst_gap_s: 120, segment_gap_min: 60, recent_days: 90, recency_weight: 0.6,
           rules_file: "config/lists/style_rules.yaml", emoji_codes_file: "config/lists/wechat_emoji_codes.txt" }
activity: { sleep_min_hours: 3.0, edge_minutes: 30, smoothing_sigma_slots: 2, min_valid_days: 14,
            sleep_rate_ratio: 0.25, busy_rate_ratio: 0.5, busy_latency_ratio: 3.0 }   # 第 04 轮新增：profile.rules_file、profile.emoji_codes_file 与 activity 的三个阈值
proactive: { daily_min: 1, daily_max: 6, min_spacing_min: 60, max_chase: 1,
             edge_of_sleep_weekly_max: 2, tick_minutes: 5, user_active_min: 10,
             unanswered_after_min: 30, meal_window_min: 30,
             bedtime_lead_min: [15, 60] }   # 第 10 轮新增：proactive 的 4 个键
schedule: { plan_minute: 5, tick_s: 30, greeting_min_gap_h: 18, greeting_window_min: [5, 40],
            min_awake_h: 6, meal_jitter_min: 15, summary_lead_min: 60,
            summary_latest_after_wake_h: 2, power_tick_s: 5, power_jump_ticks: 2 }   # 第 08 轮新增：schedule 的 10 个键
channel: { kind: "ilink", proactive_window_safe_h: 22, outbound_quota_safe: 8, proactive_reserve: 2 }
retrieval: { model: "BAAI/bge-small-zh-v1.5", device: "auto", holdout_ratio: 0.10,
             context_turns: 6, candidates: 50, mmr_lambda: 0.7, slot_weight: 0.06,
             slot_sigma_slots: 8, recency_weight: 0.04, recency_half_life_days: 180,
             dedup_similarity: 0.9, event_only_factor: 0.6, batch_size: 64 }   # 第 05 轮新增：除 model/device/holdout_ratio 外的 10 个键
persona: { sample_segments: 60, batch_segments: 10, segment_max_messages: 40, regen_ratio: 0.10,
           full_max_tokens: 1500, compact_max_tokens: 400 }   # 第 06 轮新增：persona 的 6 个键与 stickers 的 11 个键
stickers: { tags_file: "config/lists/sticker_tags.txt", neighbors_file: "config/lists/sticker_tag_neighbors.yaml",
            tag_job_size: 20, context_min_uses: 3, context_max_samples: 5, no_repeat_window: 10,
            repeat_rate_threshold: 0.30, recency_half_life_days: 90, rate_window: 200,
            rate_tolerance: 0.20, describe_timeout_s: 15 }
memory: { daily_summary: true, fact_extraction: true, block_tokens: 800, recall_facts: 12,
          recent_summary_days: 3, recall_summaries: 2, followup_lookahead_h: 36,
          recall_min_similarity: 0.35, conflict_candidates: 6, conflict_min_similarity: 0.45,
          recency_half_life_days: 90,
          quiet_minutes: 30, replay_job_days: 7, replay_chunk_lines: 400,
          replay_auto_approve_ratio: 0.10,
          weights: { similarity: 0.45, importance: 0.15, recency: 0.10, source: 0.10, date: 0.20 } }   # 第 07 轮新增：memory 的 13 个键与 weights 的 5 个子键
ops:
  backup_hour_local: 4
  backup_keep_daily: 14
  backup_keep_weekly: 8
  backup_mirror_dir: null              # 可选：外接盘等异地副本目录
  backup_postpone_max_h: 2             # 第 12 轮新增：备份遇到正在发送的会话时最多顺延多久
  smtp: { host: null, port: 465, user: null, to: null, security: "auto" }   # auto|ssl|starttls；auto：465 端口用 SSL，其余端口用 STARTTLS
  alert_cooldown_min: 60               # 第 12 轮新增：同类告警最短间隔（分钟）
  health: { interval_s: 60, poll_stale_min: 5, disk_min_gb: 5, queue_max: 500, queue_oldest_h: 24,
            backup_stale_h: 36, keep_days: 30, llm_window_min: 15, llm_error_rate: 0.5,
            llm_min_calls: 10, remind_h: 6 }   # 第 12 轮新增：健康检查的阈值
  supervise: { backoff_start_s: 5, backoff_max_s: 300, stable_after_min: 30, stop_grace_s: 40 }   # 第 12 轮新增：twin supervise 的重启退避
style_model:
  mode: "llamacpp_completion"          # llamacpp_completion|vllm_completion
  endpoint: "http://127.0.0.1:8081"
  model_id: null
  tunnel: { local_port: 8082, remote_port: 8000, backoff_start_s: 2.0, backoff_max_s: 60.0,
            remind_remote: true }      # 第 14 轮新增：隧道断线重连的退避与每天一次的计费提醒
  serve: { binary: null, context: 4096, gpu_layers: 999, parallel: 1, start_timeout_s: 180.0,
           backoff_start_s: 2.0, backoff_max_s: 60.0, stable_after_s: 120.0, warm_standby: true,
           eval_port: 8083 }           # 第 14 轮新增：本机 llama-server（只监听本机）
  memory_tokens: 300                   # 第 09 轮新增：风格模型提示词里记忆块的 token 预算，与训练集导出共用
  n_predict: 200                       # 一次生成最多多少 token
  temperature: 0.7
  top_p: 0.9
autodl: { host: null, port: null, user: "root", auth: "password", key_path: null, workdir: "/root/autodl-tmp/twin" }
training: { hybrid_plan_ratio: 0.30, dpo_min_pairs: 200, retrain_new_ratio: 0.10 }
eval: { blind_n: 50, memory_questions: 20 }
commands: { confirm_window_min: 60, pause_max_h: 168, morning_hour: 8, import_poll_s: 5.0 }
learning: { rules_max: 30, rules_interval_days: 7, rule_max_chars: 40, detect_corrections: true,
            check_interval_s: 3600.0 }   # 第 11 轮新增：commands 的 4 个键与 learning 的 5 个键
safety:
  crisis_keywords_file: "config/lists/crisis_keywords.txt"
  hotlines: { US: "988（美国心理危机热线，电话或短信）", CN: "12356（全国心理援助热线）" }
  timezone_country: { "America/Chicago": "US", "America/New_York": "US", "America/Denver": "US",
                      "America/Los_Angeles": "US", "Asia/Shanghai": "CN" }
  emergency_contact: { enabled: false, email: null }   # 显式开启才生效；邮件不含任何聊天内容
```

- **R-CFG-005** 配置三处一致：`twin.config.Settings` 的键与默认值、`config/config.example.yaml`、本节 R-CFG-004 的 YAML 必须完全一致，有测试解析本文件该 YAML 代码块逐键比对。任何一轮新增可调参数，必须在同一轮把它同时加入这三处并在汇报中列出（这是唯一无需事先征得同意的 SPEC 修改）。`config/lists/` 下的词表文件（AI 腔短语、承诺句式、危机关键词）随仓库提供、可编辑。

## 4. 存储与加密（R-STO）

- **R-STO-001** SQLite（WAL、`foreign_keys=ON`），SQLAlchemy 2 模型 + Alembic 迁移；所有表有 `created_at`/`updated_at`（UTC）。
- **R-STO-002** 敏感文本字段（消息正文、原始 JSON、记忆文本、摘要、人设卡、机器人对话、反馈）以 AES-256-GCM 加密存储，列类型为自定义 `EncryptedText`/`EncryptedJSON`；每条记录独立随机 nonce；附加数据（AAD）绑定表名与主键防止串列替换。
- **R-STO-003** 主密钥 256 位，首次运行生成并存入 `keyring`；每个密钥有 `key_id`，密文与备份都记录所用 `key_id`。`twin secrets rotate-db-key` 支持密钥轮换（重加密全部字段，可中断续跑）；轮换后旧密钥标记为"已退役"但保留在 keyring 中，直到没有任何保留期内的备份引用它才删除（保证旧备份仍可恢复）；一键删除（R-OPS-008）删除全部密钥。
- **R-STO-004** 媒体文件（她的照片、语音、视频、头像）加密存储于 `data/media/<sha256>.enc`；表情包图片同样加密（统一处理，避免分支）；需要时解密到进程内存或受控临时目录并即用即删。
- **R-STO-005** 向量库（LanceDB）只存向量、行 id、非敏感元数据（时间戳、类型），不存明文。
- **R-STO-006** 必须的表（字段在实现中细化，字段名用英文）：`conversations`、`messages`、`media_assets`、`stickers`、`sticker_uses`、`import_runs`、`jobs`、`profile_versions`、`activity_models`、`routine_overrides`、`persona_cards`、`prompt_templates`、`example_windows`、`facts`、`daily_summaries`、`lifeline_events`、`followups`、`daily_plans`、`timezone_history`、`bot_turns`（机器人会话每条进出消息）、`conversation_state`、`proactive_candidates`、`proactive_log`、`ratings`、`feedback`、`preference_pairs`、`settings`、`cost_ledger`、`channel_state`、`alerts`、`health_snapshots`、`backup_records`、`training_runs`、`dataset_versions`、`model_registry`、`eval_runs`、`eval_items`。各表由首次使用它的轮次建立（见 TRACEABILITY）。
- **R-STO-007** `messages` 与 `bot_turns` 是两张物理隔离的表；任何"风格样本/检索/训练"查询只能读 `messages` 中 `is_sent=false` 的她的真实消息（R-RET-004、R-TRN-004 有测试守护）。

## 5. 微信通道（R-CH）

- **R-CH-001** 通道接口 `Channel`：`start()`、`stop()`、`incoming()`（异步迭代入站消息）、`send_text(text, quote=None)`、`send_image(path_or_bytes, mime)`、`send_typing(active)`、`capabilities()`（是否支持引用、正在输入、GIF 动图、主动发送窗口参数）、`session_state()`。
- **R-CH-002** `IlinkChannel`：对接微信官方 ClawBot 所用的 openclaw-weixin（iLink）协议。实现前必须获取官方源码并以其为准：`npm pack @tencent-weixin/openclaw-weixin`（及 `@tencent-weixin/openclaw-weixin-cli`）解包阅读，同时对照 AstrBot 仓库中 `weixin_oc` 个人微信适配器的 Python 实现。已知线索（以源码为准）：主机 `https://ilinkai.weixin.qq.com`；接口 `/ilink/bot/get_bot_qrcode`、`/ilink/bot/get_qrcode_status`、`/ilink/bot/getupdates`（长轮询）、`/ilink/bot/sendmessage`（需 `context_token`）、`sendtyping`、`getconfig`、媒体上传下载（CDN，AES-128-ECB）。
- **R-CH-003** 登录：二维码登录（PNG 保存并用默认图片查看器打开，同时在终端打印），登录凭据加密存储；登录失效时发出告警并进入重新登录流程（R-OPS-004）。
- **R-CH-004** 长轮询：游标（如 `get_updates_buf`）持久化，重启不丢消息、不重复处理（按消息 id 去重）；网络错误指数退避（1s 起、上限 60s、带抖动）。
- **R-CH-005** 入站解析：文字；图片（下载、解密、加密入库）；语音（微信云端已转文字则取文字，否则标注未转写）；视频（取封面）；文件（取文件名）；表情包（若可识别 md5）。统一成 `InboundMessage{id, at, kind, text, media_ref, quote}`。
- **R-CH-006** 出站：文字、图片（只允许表情包库中的表情包，GIF/PNG/JPEG/WebP 原样，以及第 02 轮探针程序生成的测试图；见 R-SAFE-006）、正在输入、引用（若协议支持）。媒体发送需要非空 `context_token` 时按源码规则获取。
- **R-CH-007** 收件人绑定：登录后第一个入站消息的发送者经 CLI 确认后绑定为唯一用户；之后只处理该用户的入站消息；出站 API 只接受该用户，传入其他 id 直接抛异常（有测试）。
- **R-CH-008** 会话窗口跟踪：记录 `last_inbound_at`、`outbound_since_inbound`，提供 `remaining_quota() = outbound_quota_safe − outbound_since_inbound`；主动发送前检查 `now - last_inbound_at < proactive_window_safe_h` 且剩余配额足够；回复与主动消息的气泡数都受剩余配额约束（R-ENG-009、R-PRO-003）；收到协议表示会话过期的错误（社区报告为 `ret:-2`，以源码与实测为准）时标记过期，停止主动直到下一条入站；不得对该错误循环重试。
- **R-CH-009** 实测探针 `twin channel probe`（M0）：在用户自己的 ClawBot 会话中，以"[测试]"前缀按顺序实测并记录（每一步开始前都要求用户先发一条新的入站，使窗口与计数复位）：① **条数**：入站后每 2 分钟主动发一条，直到首次失败或 15 条，得到 N；② **媒体与交互**：新入站后发送 JPG、PNG、GIF（程序生成的合成图片），CLI 询问用户 GIF 是否为动图；发送正在输入并询问是否可见；若协议支持则发送一条引用并让用户确认显示效果（本步总条数 ≤ N−1，不够时分两次入站）；③ **窗口**：新入站后约 25 小时内用户不要给机器人发任何消息（开始前和过程中在终端与微信里提醒；期间收到入站则本步作废重做），探针在该入站后 1/6/12/20/23/25 小时各发一条文字（总条数 ≤ N−1，不够时减少测量点并在报告中说明）；④ 回复类出站与主动出站是否消耗同一配额（以源码与①的结果判断）。探针发送绕过 `proactive_window_safe_h` 与 `outbound_quota_safe` 两个安全阈值：只允许在探针计划激活期间、只允许带"[测试]"前缀的消息、每次绕过写审计；遇到协议失败（含会话过期）立即停止该步，不重试。结果写入 `docs/CHANNEL_REPORT.md` 和数据库，并据此更新 `channel.*` 默认值。探针是可长期运行的计划任务，不是一次性脚本。
- **R-CH-010** 若实测证明主动发送不可行（窗口过短或条数不足以满足 R-PRO），必须停止并向用户报告；企业微信通道不在本规格范围，需用户确认后另开一轮。
- **R-CH-011** `LocalConsoleChannel`：终端双向聊天，支持文字、发送图片路径、显示表情包为文件路径与标签、模拟正在输入；可配置模拟窗口与条数限制以便测试主动逻辑。

## 6. LLM 层（R-LLM）

- **R-LLM-001** `DeepSeekClient`：`openai.AsyncOpenAI(base_url=deepseek.base_url)`；方法 `chat(messages, *, purpose, thinking, reasoning_effort, temperature, max_tokens, json_schema=None, images=None)`，返回内容、`reasoning_content`、usage、费用、耗时。
- **R-LLM-002** 思考开关：通过 `extra_body={"thinking": {"type": "enabled"|"disabled"}}`；开启时 `temperature`/`presence_penalty`/`frequency_penalty` 无效，不发送；`reasoning_effort` 可配（`low`/`high`/`max`）。`reasoning_content` 不回填对话上下文、不发给用户，仅在 `/显示思考 开` 时以系统消息形式展示给用户并写调试日志（脱敏）。
- **R-LLM-003** JSON 输出：需要结构化结果的调用（规划、抽取、打标签、评审）使用 JSON 输出并用 pydantic 校验；校验失败带错误信息重试一次，仍失败则记为任务失败（离线任务进重试队列；在线路径降级）。
- **R-LLM-004** 看图：`deepseek-flash` 视觉；图片以 `image_url` 的 base64 data URL 放在 `user` 消息中（system/assistant 中放图会 400）；支持 JPEG/PNG/GIF/WebP；表情包用 `detail: low`，照片用 `auto`（`detail` 是否被接受以 R-LLM-013 实测为准，不接受则不发送该参数）；每张图计费上限 1024 tokens（实际计费以实测为准，用于费用估算）。
- **R-LLM-005** 可靠性：429/5xx/超时指数退避重试（最多 4 次，带抖动）；全局并发信号量；连续 10 次失败触发熔断 5 分钟并告警。
- **R-LLM-006** 费用记账：每次调用把 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`、`completion_tokens`（含推理 token）按价格表与是否高峰计入 `cost_ledger`，带 `purpose` 标签（reply / plan / proactive / extract / summary / persona / caption / sticker_tag / eval / train_plan）。
- **R-LLM-007** 高峰判定：北京日期为工作日（用 `chinese_calendar.is_workday()` 判定，含调休上班的周末，不含法定节假日）时，UTC 01:00–04:00 与 06:00–10:00 为高峰，其余为非高峰。`chinese-calendar` 对未收录年份会抛异常——此时回退为"周一至周五为工作日"并合并配置 `pricing.extra_offpeak_dates`/`extra_peak_dates`，同时告警、在 `twin doctor` 中提示升级该依赖。非高峰价格倍率为 `pricing.offpeak_multiplier`（默认 0.5；官方取消优惠时设为 1.0，此时离线任务不再等待非高峰）。提供 `next_offpeak_window(now)`，离线任务默认只在非高峰执行（可设截止时间强制执行）。
- **R-LLM-008** 预算：日预算与月预算；达到 80% 告警；超出按顺序降级并告警：① 关闭聊天思考 ② 检索例子 8→3、记忆上下文预算减半 ③ 暂停主动消息 ④ 若已激活的风格模型通过了上线门槛（R-SRV-005）且健康，则切到该风格后端，否则保持最小上下文的非思考 DeepSeek。任何情况下都不停止回复用户。一次性批任务的费用不参与本条判断（R-LLM-014）。进入各级的花费占预算比例由 `budget.degrade_ratios` 给出（默认 1.0 / 1.25 / 1.5 / 2.0，取日预算与月预算中比例较大者）。
- **R-LLM-009** 脱敏 `twin.llm.redaction`：发往 DeepSeek/AutoDL 前替换手机号（中国 `1[3-9]\d{9}` 与美国格式）、邮箱、身份证号（18 位含校验位）、银行卡号（16–19 位且通过 Luhn）、详细地址（省/市/区/路/号/栋/单元/室等启发式）、wxid；替换为类型占位符（如 `[手机号]`）。输出后处理检测占位符外泄（R-ENG-012）。脱敏有属性测试（hypothesis）。
- **R-LLM-010** 提示词缓存布局：固定前缀（规则 + 人设卡）→ 近期对话（批量窗口，R-MEM-001）→ 可变上下文与本轮用户消息（R-ENG-005）。可验证的性质：除窗口批量后移的那一轮外，第 n+1 次请求的消息序列以第 n 次请求"截至其最后一条历史消息"的内容为前缀（有测试）；记录每次缓存命中率，`/费用` 与 `/状态` 中可见。
- **R-LLM-011** `StyleModelClient`：两种模式，**都只发送由 `StylePromptBuilder` 渲染好的提示词字符串，绝不使用服务端聊天模板**（训练与推理一致，R-TRN-011）。`llamacpp_completion`：llama.cpp `llama-server` 的 `/completion` 接口；`vllm_completion`：vLLM 的 `/v1/completions` 接口（`prompt` 为渲染字符串，`model` 为 LoRA 名称）。都有健康检查、超时、停止词（`<|im_end|>`），以及启用前的分词核对（R-TRN-011 第 4 点）。
- **R-LLM-012** token 估算：本地估算器按字符类别计数，并用实际 usage 自校准系数（指数滑动平均），用于上下文预算裁剪。
- **R-LLM-013** DeepSeek 实测探针 `twin llm probe`（M0）：验证并记录 ① 思考开/关两种请求都成功，开启时返回 `reasoning_content`，关闭时不返回；② 同一长前缀连续请求两次，第二次 `prompt_cache_hit_tokens` > 0；③ 视觉请求（程序生成的 JPEG、PNG 与多帧 GIF 各一张）成功，记录 GIF 是否被接受；④ 思考关闭与开启两种情况下 JSON 输出都可被解析（主动消息规划依赖思考开启时的 JSON）；⑤ `detail` 参数是否被接受（带与不带各一次）；⑥ 每张图的实际计费 token 数（不同尺寸各一张，从 usage 差值推算）；⑦ 各请求耗时与费用。结果写入 `docs/LLM_REPORT.md`（并结构化写入数据库），据此调整 R-LLM-004 的参数与费用估算系数。M0 判定口径：①②④ 必须成功，③ 中 JPEG 与 PNG 必须成功（GIF 结果只记录，不被接受时用首帧兜底）；⑤⑥⑦ 是测量项，只要求记录。
- **R-LLM-014** 一次性任务预算：全量记忆回放、首次批量图片描述、首次表情包打标签、人设卡生成、hybrid 规划合成、评估生成等一次性批任务，入队前给出费用估算，经用户 `twin jobs approve <批次>` 确认后执行，费用记入独立的 `one_time` 账目（单批上限 `budget.one_time_usd`）；不计入日/月预算的降级判断，也不计入 R-EVAL-007 的月费用门槛（报告中单列）。实际费用超过估算 20% 时暂停该批次并告警。

## 7. 数据导入（R-IMP）

- **R-IMP-001** 输入为一个导出根目录：`report.json`、`conversations/<序号_昵称_wxid_hash>/{meta.json,messages.json}`、`media/{images,emojis,avatars,...}`、`_integrity/`（若存在则读取并校验其中的完整性信息）。目录名含中文与特殊符号，必须正确处理。
- **R-IMP-002** 按 `schemaVersion=1` 定义 pydantic 模型：`report.json{schemaVersion, exportId, account, createdAt, missingMedia[{kind,id,conversation,messageId}], errors[]}`；`meta.json{schemaVersion, username, displayName, avatarPath, isGroup, exportedAt, messageCount}`；`messages.json{schemaVersion, exportedAt, account, conversation{username,displayName,avatarPath,isGroup}, filters{startTime,endTime,messageTypes[]}, messages[]}`。消息字段：`id, localId, serverId, createTime, createTimeText, sortSeq, type, renderType, isSent, senderUsername, conversationUsername, isGroup, content, title, url, from, fromUsername, linkType, linkStyle, objectId, objectNonceId, recordItem, thumbUrl, imageMd5, imageFileId, imageMd5Candidates, imageFileIdCandidates, imageUrl, emojiMd5, emojiUrl, videoMd5, videoThumbMd5, videoFileId, videoThumbFileId, videoUrl, videoThumbUrl, voiceLength, voiceTranscript, voiceTranscriptStatus, voiceTranscriptError, voiceTranscriptLanguage, voiceTranscriptModel, quoteUsername, quoteServerId, quoteType, quoteThumbUrl, quoteVoiceLength, quoteTitle, quoteContent, amount, coverUrl, fileSize, fileMd5, paySubType, transferStatus, transferId, voipType, locationLat, locationLng, locationPoiname, locationLabel, senderDisplayName, senderAvatarPath, offlineMedia[{kind,path,md5,fileId}]`（可选）。未知 `schemaVersion` 直接报错退出；未知字段保留在原始 JSON 中并记录。
- **R-IMP-003** 只导入目标会话（`target.username`）；未配置时列出非群聊会话让用户在 CLI 中确认一次并写入设置。其他会话不读取消息内容（隐私最小化）。群聊一律跳过。
- **R-IMP-004** `messages.json` 用 `ijson` 流式解析（`messages.item`），每 2,000 条批量写入；内存占用与文件大小无关（10GB 文件峰值内存 < 500MB）。
- **R-IMP-005** 以消息 `id` 幂等 upsert；同 id 内容不同时以 `exportedAt` 较新的为准并记录差异计数。
- **R-IMP-006** 可续传：`import_runs` 记录状态、文件、已处理条数、最后 `id`/`sortSeq`；中断后 `twin import --resume` 从断点继续。
- **R-IMP-007** 归一化：`renderType` → `kind ∈ {text, sticker, image, quote, voice, video, file, link, call, transfer, redpacket, system, location, chathistory, unknown}`；每类生成用于上下文的"事件文字"，例如 `[通话 37 分钟]`、`[未接通话]`、`[她撤回了一条消息]`、`[转账]`、`[红包]`、`[位置：xxx]`、`[聊天记录]`、`[图片：<描述>]`、`[语音 5 秒：<转写>]`。`voip` 内容解析"通话时长 mm:ss / 对方已取消 / 已拒绝 / 未应答 / 已在其它设备接听"。
- **R-IMP-008** 媒体：`offlineMedia` 指向的文件复制进加密媒体库；`report.missingMedia` 记录缺失；表情包按 `emojiUrl` 下载（并发 4、每秒不超过 4 个请求、重试 3 次），按文件头识别 GIF/PNG/JPEG/WebP，校验 `emojiMd5`（不一致标记 `md5_mismatch` 仍保留）；下载可续传，失败标记不可用，不阻塞导入。
- **R-IMP-009** 语音无转写时记为 `[语音 N 秒，未转写]`；不在本地做语音识别（v1）。
- **R-IMP-010** 导入报告（不含任何消息正文）：按类型与发送方计数、日期范围、新增/重复/冲突、缺失媒体、表情包下载统计、风格指标变化（R-PROF 重算后）；输出到终端与 `data/reports/import-<UTC时间>.md`。
- **R-IMP-011** 导入完成后自动排队：画像重算（R-PROF）、作息模型（R-ACT）、人设卡（立即生成新版本以刷新 `[自动-统计规则]`；她的新消息自上次 `[自动-描述]` 生成以来增加 ≥ 10% 时，排队非高峰的 `[自动-描述]` 重生成；live 与 pre_holdout 两个范围同理，R-TRN-013）、新表情包打标签（R-STK）、检索库增量更新（R-RET）、新日期范围的记忆回放（R-MEM-010）、重训提醒检查（R-TRN-012）、新图片描述（R-IMP-012）。每个导入后钩子都有可单独执行的回填命令（如 `twin profile rebuild`、`twin persona regenerate`、`twin stickers tag-all`、`twin retrieval rebuild`、`twin memory replay`、`twin images caption-backfill`）；注册钩子的轮次在验收中对已导入的数据执行一次回填。
- **R-IMP-012** 图片描述：默认只为最近 90 天的图片与被检索命中的图片生成一句描述（DeepSeek 视觉，非高峰），其余在首次被用到时按需生成并缓存；描述文本先经 R-LLM-009 脱敏再存储与使用（照片里可能有电话、地址）；在线回复路径从不等待描述生成：检索例子里的图片尚无描述时以 `[图片]` 呈现并排队描述任务（R-RET-005）；用户发来的图片按 R-ENG-013 同步描述（带超时）。
- **R-IMP-013** 性能：100 万条消息的结构化导入（不含下载与 LLM 任务）在普通 PC 上 < 30 分钟；有基准测试脚本（合成数据生成器 + 计时）。
- **R-IMP-014** 结构探查 `twin import inspect <目录>`：只输出结构（文件树、每个 JSON 的键名、类型与出现次数、`renderType`/`type` 取值计数、`_integrity/` 的文件清单与键名），不输出任何值；用于核对合成数据生成器与 `_integrity` 解析是否与真实导出一致；结果写 `data/reports/inspect-<UTC>.md`。

## 8. 统计画像（R-PROF）

- **R-PROF-001** 只用她（以及用于对比的用户）的真实消息计算，以"合并连发块"（同一发送者、条间隔 ≤ 120 秒视为同一块）为基本单位之一。
- **R-PROF-002** 必须计算的指标（她与用户各一份）：文字长度分布（中位数、p75、p90、p95）；每类标点使用率（。，？！～…、空格）与句末标点率；连发块条数分布；连发条间隔分布；回复延迟分布（总体与按小时桶）；表情代码使用率、词表与频率、连用次数分布；Unicode emoji 使用率与词表；表情包占比、每个 md5 的使用频率与最近使用时间；引用回复率；句末语气词分布；高频整句与 n-gram（本地保留）；对用户的称呼（句首/句尾高频称谓）；笑声模式（"哈"连写长度分布）；问句率；消息类型构成；先开口次数与时段。
- **R-PROF-003** 双窗口：全量与最近 90 天，按 `recency_weight=0.6` 混合得到"当前画像"；所有分布保存为可抽样的经验分布（直方图或分位点）。
- **R-PROF-004** 版本化：每次重算写入 `profile_versions`（含数据范围、范围类型 `scope ∈ {live, pre_holdout}`（R-TRN-013）与指标 JSON），生成与上一版的差异摘要；支持回滚到任意版本。
- **R-PROF-005** 从统计层自动生成"数字风格规则"文本（例如"几乎不用逗号和句号，用分条代替；单条通常 3–8 字"），写入人设卡的自动区块（R-PERS-002）。

## 9. 作息活动模型（R-ACT）

- **R-ACT-001** 时间语义：消息时间是绝对时刻（UTC 存储）；学习作息时把每条消息换算为她当时所在地的"当地钟点"：默认用 `time.source_timezone`（芝加哥），若配置了 `time.source_timezone_ranges`（按日期区间指定时区，例如某段时间她在国内）则按区间选择；运行时把当地钟点放到当前 `bot_timezone` 上。切换时区后同样钟点原样搬过去（芝加哥 8 点起床 → 北京 8 点起床）。
- **R-ACT-002** 按"当地钟点（15 分钟粒度）× 日类型（工作日/周末/节假日）"计算她的发消息率、先开口率、回复延迟条件分布；平滑（高斯核或滑动平均）。节假日按她当时所在地区（R-ACT-001 的区间时区）：中国时区用 `chinese-calendar` 的 `is_workday()`（含调休；未收录年份按 R-LLM-007 回退），美国时区用美国联邦假日（`holidays` 包）。
- **R-ACT-003** 睡眠推断：先由全部数据的当地钟点活跃曲线找到最低活跃点，以它作为每个"活动日"的分界（而不是固定的午夜或正午）；在每个活动日内找最长的、≥ `activity.sleep_min_hours`（默认 3 小时）、发消息率低于阈值的连续区间作为睡眠，统计入睡/起床时刻分布（环形统计：均值、方差、分位数），并区分"深睡核心"（入睡后 `edge_minutes` 到起床前 `edge_minutes`）与"入睡/将醒边缘"。有效数据少于 `activity.min_valid_days` 天时标记低置信，用全部数据整体活跃曲线中最长的低谷作为默认睡眠（不使用写死的钟点）。合理性检查：推断的睡眠核心若落在当地 10:00–18:00 内，告警"可能是 source_timezone 设置不对"。推断结果在 `twin profile show` 中请用户确认，不对时用作息修正（R-ACT-005）。用 §0 小时向量生成的合成数据做测试：推断不报错，输出的睡眠区间是该向量中持续 ≥ 3 小时的低活跃区间，落在白天时产生合理性告警。
- **R-ACT-004** 忙碌推断：工作日中稳定低活跃但非睡眠的时段标为"可能忙碌"，统计该时段的回复延迟分布。
- **R-ACT-005** 手动修正（指令 R-CMD）：设定睡眠区间、忙碌时段（按星期几与时间）、节假日；手动修正优先于推断，持久化，可列出与删除。
- **R-ACT-006** 版本化（含 live 与 pre_holdout 两个范围，R-TRN-013）并随 R-PROF 一起重算；`twin profile show` 输出作息概览（当地时间），并提示用户确认推断出的睡眠时段。提供纯函数 `typical_state(当地时间, 日类型) -> deep_sleep|sleep_edge|busy|free`（按睡眠与忙碌分布的中位数判定，不抽样），供训练集导出与评估沙盒推断历史时刻她的状态；线上状态仍以每日计划为准（R-SCH-001）。

## 10. 人设卡（R-PERS）

- **R-PERS-001** 由 DeepSeek（`offline_model`）从分层抽样的真实对话（按月份、话题、情绪、时段分层，脱敏后）生成；分多次调用归纳再合并，覆盖：基本情况（只写真实记录中出现的）、语气、口头禅、对用户的称呼、常聊话题、开心/生气/撒娇/难过/拒绝/道歉时怎么说、对用户的态度、说话禁忌。按 R-TRN-013 分别生成 live（全部数据）与 pre_holdout（只用 `holdout_cutoff()` 之前的数据）两个范围的版本。
- **R-PERS-002** 结构化存储（Markdown 分区）：`[自动-统计规则]`（R-PROF-005）、`[自动-描述]`（内分"风格"与"基本情况"两小节）、`[手动]`（用户手改，重算时原样保留；内分 `### 风格` 与 `### 事实` 两小节）、`[不要这样]`（来自纠正，R-LRN-003）。
- **R-PERS-003** 版本化、差异展示、回滚：`twin persona show|diff|rollback <ver>|edit`（`edit` 用系统默认编辑器打开手动区块，保存后加密入库）。
- **R-PERS-004** 两个渲染版本：完整版（DeepSeek 后端与 hybrid 规划，≤ 1,500 tokens，含全部分区）与精简版（风格模型后端与训练集共用，≤ 400 tokens）。精简版**只含风格内容**：`[自动-统计规则]`、`[自动-描述]` 的风格小节、`[手动]` 的风格小节；不含任何事实（事实只经 as-of 记忆块进入，R-TRN-013），也不含 `[不要这样]`（风格模型经 hybrid 规划与 DPO 接受纠正）。
- **R-PERS-005** 人设卡、提示词模板文件都有版本号；`settings` 中记录当前生效版本；改坏了可回滚（R-OPS-010）。

## 11. 表情与表情包（R-STK）

- **R-STK-001** 微信表情代码：从她的消息中统计词表（如 `[拥抱]`、`[亲亲]`、`[流泪]`）与频率、连用分布；生成时只允许词表内代码，频率按画像控制。
- **R-STK-002** 表情包库：收录她发过的全部表情包（并收录用户发过的表情包用于识别）；每个包含 md5、文件、格式、尺寸、她的使用次数、最近使用时间、她使用时的上下文窗口 id。
- **R-STK-003** 打标签：DeepSeek 视觉（`detail: low`）输出 JSON：情绪标签（固定词表：开心、大笑、撒娇、委屈、难过、生气、无语、震惊、困、晚安、早安、亲亲、抱抱、加油、好的、拒绝、调皮、害羞、疑问、饿、其他）、一句画面描述、适用场景；再结合她实际使用时的上下文（DeepSeek 归纳）修正标签——上下文修正只使用 `holdout_cutoff()` 之前的使用记录（R-TRN-013）。用户可 `twin stickers tag <md5> <标签...>` 手动改。
- **R-STK-004** 发送选择：模型输出独立一行 `[表情包:<标签>]`；候选 = 标签匹配的她的表情包；得分 = 她的使用频率（平滑）× 与当前上下文的语义相似度（描述向量 vs 上下文向量）× 新近度；最近 10 个气泡内不重复同一张（除非她的重复率高于阈值）；无匹配时退化为同情绪相近标签，仍无则删除该行。
- **R-STK-005** 频率控制：滚动窗口（最近 200 个机器人气泡）表情包占比保持在她占比的 ±20% 区间（给 R-EVAL-002 的 ±30% 留余量），超出时后处理删除表情包行，不足时不强行插入。
- **R-STK-006** 识别用户发来的表情包：md5 命中库直接得到描述与标签；未命中用视觉模型描述，结果缓存。
- **R-STK-007** 发送通过通道图片接口；GIF 动图效果以 R-CH-009 实测为准，若不动则改用静态首帧还是原样发送由用户决定（报告中给出对比）。

## 12. 检索库（R-RET）

- **R-RET-001** 例子窗口：对她每个真实回复块，取其前面同一会话、间隔 ≤ 60 分钟内的最多 6 个合并轮次作为上下文；保存 `(context_ids, reply_ids, local_time, day_type)`。
- **R-RET-002** 向量：本地 `sentence-transformers`（默认 `BAAI/bge-small-zh-v1.5`，可配 `BAAI/bge-m3`；有 CUDA 用 GPU，否则 CPU），对脱敏后的上下文（最后两轮加权）编码；批量、可续跑；模型首次运行下载并缓存。
- **R-RET-003** 留出集：按时间最近 10% 的窗口不进入检索库，专供评估（R-EVAL）与训练测试集（R-TRN-006）；切分时刻只由唯一函数 `holdout_cutoff()` 提供（持久化；导入新数据不自动移动；`twin retrieval resplit` 显式重切），检索、训练、评估与 pre_holdout 范围的派生数据都调用它（R-TRN-013）。
- **R-RET-004** 只收她的真实回复；索引构建函数的输入类型只接受 `messages` 表记录，机器人回复无法进入（有测试）。
- **R-RET-005** 查询：用当前对话最近几轮构造查询，取 top-k（默认 8），MMR 去冗余（λ=0.7），加"时段相近"和"新近度"加分，过滤与当前回复近乎相同的例子。渲染例子时，她的回复里机器人无法复现的行（图片、语音、通话、转账等事件文字）以"（此处她发了：…）"的上下文注记呈现，不作为可模仿的回复行（R-SAFE-006）。
- **R-RET-006** 增量更新：导入新记录后只编码新增窗口；模型或参数变化时全量重建。

## 13. 记忆（R-MEM）

- **R-MEM-001** 近期对话：机器人会话最近 30–40 轮原文（合并块为单位）直接进入提示词；窗口起点只在轮数超过 `engine.history_turns_max`（40）时一次后移 10 轮（回到 30 轮），使相邻请求共享前缀、命中缓存（R-LLM-010）。
- **R-MEM-002** 每日摘要：每天在她"起床"前（作为离线任务，避开高峰），分别为当天的真实记录与机器人对话写带日期的摘要；日期按事件发生时的时区当地日期；摘要加密存储并向量化。
- **R-MEM-003** 事实库：字段 `subject(她/用户/双方/他人)、category(生活/偏好/计划/纪念日/昵称/关系/工作学习/其他)、text、source(real_record/user_said/bot_invented/user_command)、known_at(最早可知时间)、valid_from/valid_to、event_date(可选，事件的当地日期，如生日、纪念日、考试日)、recurrence(none/yearly/monthly)、confidence、superseded_by`；日期相关性检索（R-MEM-008）用 `event_date` 与 `recurrence` 判断"今天是纪念日"。
- **R-MEM-004** 冲突规则：优先级 真实记录 > 用户说的 > 机器人编的；新事实与旧事实冲突时由 DeepSeek 判定并设置 `superseded_by`；低优先级来源不得覆盖高优先级来源。
- **R-MEM-005** 生活线：每天在她"起床"时为她生成当天的虚构日程（时段、做什么、在哪、心情），必须与作息、真实事实、近几天生活线一致；机器人在对话中编出的新细节即时写入生活线；第二天能接上。
- **R-MEM-006** 待跟进：从用户消息中抽取有时间点的事（考试、面试、出门、看病等），记录 `due_at`、内容、状态；到时由主动消息跟进（R-PRO-004），用户主动提到后关闭。
- **R-MEM-007** 抽取：每段对话结束（静默 30 分钟）后异步调用 DeepSeek 抽取事实与待跟进；真实记录在导入回放时抽取。
- **R-MEM-008** 检索：按当前话题（向量相似）、关键词、日期相关性（今天是纪念日、昨天说今天考试）、重要度、新近度综合打分，在 token 预算内组装"记忆块"。
- **R-MEM-009** 指令：`/记住 <内容>`（来源 user_command，最高置信）、`/忘掉 <内容或编号>`（硬删除该条及仅由其派生的条目）、`/记忆 [页码|关键词]`（分页列出）。
- **R-MEM-010** 时间回放：按日期顺序处理真实历史，生成摘要与事实并记录 `known_at`（= 证据中最晚一条消息的时间）；提供 `memory_view(as_of=t)`：只返回 `known_at < t` 且在 t 时有效的事实、当地日期早于 t 当地日期的摘要、t 之前创建且在 t 时仍未关闭的待跟进；t 早于机器人上线时，机器人来源的记忆与生活线为空。训练集导出（R-TRN-002）与评估沙盒（R-EVAL-009）都只经此视图读取记忆（R-TRN-013）。
- **R-MEM-011** 机器人编的事实与生活线永远标记 `bot_invented`，导入新真实记录后被真实事实覆盖。

## 14. 回复引擎（R-ENG）

- **R-ENG-001** 会话状态机：`IDLE → COLLECTING → DECIDING → GENERATING → SENDING → IDLE`，状态持久化以便重启恢复。GENERATING 中收到用户新消息：取消进行中的生成（已产生的费用照常记账），把新消息并入后回到 COLLECTING；SENDING 中按 R-ENG-009 处理。
- **R-ENG-002** 等用户说完：收到用户消息后进入 COLLECTING，静默 `quiet_window_s`（默认 15 秒，产品文档默认值）后再处理，期间新消息重置计时；最长等待 `max_wait_s`（90 秒）。`engine.quiet_window_adaptive=true` 时按用户连发间隔 p75 自适应（上限 `quiet_window_max_s`），默认关闭，`/状态` 中显示按画像算出的建议值。
- **R-ENG-003** DECIDING：按当前她的状态（R-SCH）决定：醒着且空闲 → 正常回复；忙碌 → 从"忙碌时段回复延迟分布"抽样延迟；睡觉 → 排队到她起床后（加抖动）再回复，回复要像刚醒（提示词中注明"你刚醒，看到了这些消息"）；`/暂停` 期间不回复。
- **R-ENG-004** 首条延迟：从她真实回复延迟的经验分布（按当前小时桶）抽样，保留几分钟到几十分钟的长尾，不追求秒回；加阅读时间。延迟期间若用户再发消息，合并处理。
- **R-ENG-005** 提示词组装顺序（为缓存）：① 固定规则 + 人设卡（完整版）（system）② 机器人会话近期对话（R-MEM-001 的批量窗口，user/assistant 交替，用户消息只存原文）③ 最后一条 user 消息 = 本轮可变上下文块（当前当地时间与星期、她此刻状态与生活线片段、记忆块、待跟进、检索到的真实例子（以"她过去在类似情况下的真实回复"呈现））+ 用户本轮消息（含图片描述、语音转写、表情包描述、引用内容）。可变块只出现在最后一条 user 消息里；写入历史时只保留用户原文。
  > 与产品文档列出的顺序（规则 → 记忆 → 例子 → 最近对话 → 新消息）的差别：把"最近对话"提到可变块之前。原因：记忆与例子每轮都变，放在最近对话前面会让后面的对话历史全部失去缓存命中；产品文档该顺序的目的本来就是命中缓存。内容项完全相同，没有删减。
- **R-ENG-006** 生成后端：`deepseek`（DeepSeek 直接生成）、`style`（风格模型生成，提示词由 `StylePromptBuilder` 构造，与训练一致）、`hybrid`（DeepSeek 先产出 JSON 规划 `{reply: bool, intent, facts_to_use[], tone, bubble_hint, sticker_hint}`，风格模型按规划用她的语气生成）。思考开关在 deepseek 后端作用于生成本身，在 hybrid 后端作用于规划步骤；style 后端开思考时自动走 hybrid 规划。style/hybrid 用当前激活模型在 `model_registry` 中锁定的人设卡与模板版本渲染（R-SRV-001）。
- **R-ENG-007** 输出约定：每行一个气泡；允许词表内表情代码；表情包单独一行 `[表情包:<标签>]`；可选首行 `[引用:<被引用片段>]`（通道支持引用时生效，否则去掉）；`[不回]` 表示选择不回复（仅当用户最后一条是结束性的短回应且她的历史中有类似"不回"的概率时允许，概率按画像）。
- **R-ENG-008** 后处理：去 AI 腔（禁用短语表：作为AI、人工智能、语言模型、希望对你有帮助、有什么可以帮你、请注意、总之等，可配置）；标点按画像归一（她几乎不用句号逗号时去掉句末句号、把逗号句拆成多条）；单条长度上限 = 她的 p95 × 1.5；气泡数上限 = min(她的 p90 × 1.5, `max_bubbles`)；去除连续重复气泡；校验表情代码与表情包标签；去除脱敏占位符外泄；删除事件文字与媒体占位行（R-SAFE-006）；删除风格模型输出中的 `<think>…</think>` 片段（记为违规）；按平台剩余配额合并气泡（R-ENG-009）；违反硬规则时重新生成一次，仍违反则降级为 DeepSeek 非思考重试。
- **R-ENG-009** 发送节奏：气泡间隔从她的连发间隔分布抽样 + 打字时间（按她的打字速度估计，字数相关）；支持时发送"正在输入"；SENDING 期间用户又发消息：已发出的不撤回，剩余未发气泡取消，带"已经说过的话"重新生成后续。发送前按 `remaining_quota()` 约束气泡数：回复可用条数 = 剩余配额 − `channel.proactive_reserve`（至少 1），超出时先合并相邻文字行（空格连接），仍超出再删表情包行，动作记入 `bot_turns`。
- **R-ENG-010** 失败兜底：模型超时或出错时延后（2–10 分钟随机）重试，最多 3 次；永不把报错、堆栈或系统提示发给用户；超过重试时发一条极短的自然回应（由画像高频短句中抽取，例如她常用的应答）并记录告警。
- **R-ENG-011** 机器人会话记录：每条进出消息写入 `bot_turns`（加密），含后端、是否思考、规划、费用、延迟、后处理动作。
- **R-ENG-012** 占位符外泄检测：输出中出现任何脱敏占位符即视为违规（R-ENG-008 处理）。
- **R-ENG-013** 用户发图片：交给视觉模型生成描述放入上下文；视频用封面描述；文件只给文件名；语音用微信转写文字。

## 15. 安全与边界（R-SAFE）

- **R-SAFE-001** 危机识别：用户消息先过本地关键词筛查（自伤、自杀、极端绝望、他伤等，词表文件可配），命中后用 DeepSeek 二次判断；确认后机器人跳出角色，用关心的语气回应并给出求助渠道（当前 `bot_timezone` 经 `safety.timezone_country` 映射到国家：美国 988，中国 12356；映射不到时两者都给），不继续扮演；写告警通知用户本人。只有用户显式开启 `safety.emergency_contact` 并填写邮箱时，才另发一封**不含任何聊天内容**的提醒邮件（只有时间与"可能需要关心他"），默认关闭。
- **R-SAFE-002** 做不到的事：禁止机器人主动承诺或声称会打电话、视频、发语音、发照片、见面、转账、发红包、寄东西等现实行为；用户提出时用她的语气自然带过（规则写入固定提示词，后处理检测典型承诺句式并重写）。
- **R-SAFE-003** 用户真诚地问"你是不是 AI/机器人"时，不否认自己是模拟出来的（可以用她的语气说）。
- **R-SAFE-004** 机器人说的话不代表她本人；不得以她的名义对外联系任何人。可测试的落实：① 通道收件人守卫（R-CH-007）；② 扫描测试：代码中全部出站途径（通道发送、SMTP）的收件人只能来自已绑定用户、用户在配置中填写的本人邮箱或显式开启的紧急联系人，邮件模板中没有聊天正文字段。
- **R-SAFE-005** 内容遵守 DeepSeek 使用政策；生成被拒绝时按 R-ENG-010 兜底，不向用户暴露拒绝原因原文。
- **R-SAFE-006** 不发她的照片、不输出事件文字：① 出站图片只允许表情包库中可用的表情包——引擎侧发送接口的参数类型是 `Sticker`；通道层 `send_image` 再校验字节的 sha256 属于表情包库或第 02 轮探针程序生成的测试图，否则抛 `MediaNotAllowed`；`media_assets` 中的照片、视频、语音无法传入。② 生成输出中出现事件文字或媒体占位（`[图片…]`、`[语音…]`、`[视频…]`、`[通话…]`、`[未接通话]`、`[转账…]`、`[红包…]`、`[位置…]`、`[文件…]`、`[链接…]`、`[聊天记录]`、撤回与拍一拍等；检测规则由 `render_event_text` 的模板表自动生成）的行在后处理中删除，全部被删时重新生成。③ 训练目标中删除这类行（R-TRN-003），检索例子中以上下文注记呈现（R-RET-005）。④ 盲测排除真实回复含这类行的样本（R-EVAL-001）。每一点都有测试。

## 16. 时间与作息运行（R-SCH）

- **R-SCH-001** `TimeService`：唯一的"现在几点"来源；提供当前 `bot_timezone`、当地时间、日类型、她的当前状态（睡眠/边缘/忙碌/空闲）。测试可注入时钟。
- **R-SCH-002** 时区切换：`/时区 <IANA 名称>` 校验后持久化；立即从当前时刻起按新时区重建今天的日程；切换当天不重复起床问候（距上次 ≥ 18 小时才允许）、不跳过睡眠（新时区若正处于睡眠区间则直接进入睡眠）；记录切换历史。
- **R-SCH-003** 夏令时：一律用 `zoneinfo`；跨越夏令时的日子日程按当地钟点生成（2026-11-01 芝加哥 UTC−5→UTC−6 有测试）。
- **R-SCH-004** 每日计划：当地 00:05（或启动时缺失当天计划）生成：起床、入睡时刻（从分布抽样）、忙碌时段、饭点、当天主动次数配额（R-PRO-002）、生活线（R-MEM-005）；用可复现的随机种子（日期 + 安装随机盐）便于调试。
- **R-SCH-005** 电脑重启、进程中断或系统睡眠唤醒后：不补发错过的主动消息（过期候选作废）；重建当天计划；通道重连；对中断期间用户发来的消息按 R-ENG 正常（带延迟）回复。睡眠/唤醒检测：Windows 上监听 `WM_POWERBROADCAST`（挂起/恢复），并以"墙钟与单调时钟之差跳变超过 2 个 tick"作为跨平台兜底。

## 17. 主动消息（R-PRO）

- **R-PRO-001** 调度器每 `tick_minutes`（5 分钟）评估一次，采用非齐次泊松稀释：按她的"先开口率"曲线 λ(当地钟点) 缩放到当天配额，决定本 tick 是否产生候选。
- **R-PRO-002** 每天主动次数在 `[daily_min, daily_max]`（默认 1–6，均值贴近她真实的约 3.6 次/天，均值随全量数据重算）内随机；`/主动 2-5` 修改范围，`/主动 关|开` 开关。
- **R-PRO-003** 硬约束：深睡核心时段 0 次；两次主动间隔 ≥ `min_spacing_min`；用户没回上一条主动消息时最多再追 `max_chase` 条；`/暂停` 期间不发；平台窗口与条数（R-CH-008）不满足时不发并记录"被窗口抑制"；规划的多条超过剩余配额时截短，剩余配额为 0 时不发并记录。
- **R-PRO-004** 触发类型与优先级：跟进型（待跟进到期）> 作息型（起床问候在起床后 5–40 分钟、饭点、睡前晚安）> 沉默型（距上次互动超过她真实"沉默后先开口"间隔的抽样值）> 分享型（生活线片段）。
- **R-PRO-005** 入睡/将醒边缘：允许偶发"睡不着""刚醒"类消息，概率按她真实深夜消息频率，每周不超过 `edge_of_sleep_weekly_max`（默认 2）。
- **R-PRO-006** 两步决定：规则与概率命中后，调用 DeepSeek 规划（`thinking.proactive_planner`，默认开）返回 JSON `{send: bool, kind, messages[], sticker_hint, reason}`；`send=false` 时记录原因并按策略重新安排。内容必须与生活线、记忆一致，发出后写入生活线与 `bot_turns`。
- **R-PRO-007** 发送节奏同 R-ENG-009（可多条连发，按她的连发习惯）。
- **R-PRO-008** 审计日志 `proactive_log`：候选时间、触发类型、是否发送、抑制原因、内容（加密）、当地时间、她的状态；供 R-EVAL-005 使用。

## 18. 指令（R-CMD）

- **R-CMD-001** 以 `/` 开头且命中指令表的消息为指令：不进入记忆、学习、检索、训练；回复以系统口吻、带前缀"⚙️"，与人设回复明显区分。未知指令回复帮助摘要。
- **R-CMD-002** 指令表（全部必须实现）：

| 指令 | 作用 |
| --- | --- |
| `/帮助` | 列出全部指令 |
| `/状态` | 当前时区与当地时间、她此刻状态、后端、思考模式、主动范围与今日已发、平台窗口剩余、今日费用、最近告警、待重训提醒 |
| `/思考 开\|关\|自动` | 聊天生成的思考模式（R-LLM-002、R-ENG-006）；自动 = 用户提问、情绪话题或长消息时开 |
| `/显示思考 开\|关` | 是否把思考内容以系统消息展示（调试） |
| `/时区 <IANA>\|查看` | 切换或查看机器人时区（R-SCH-002） |
| `/暂停 <时长>`、`/恢复` | 暂停回复与主动 |
| `/主动 <最少>-<最多>\|开\|关` | 主动次数范围与开关 |
| `/作息 睡 <HH:MM>-<HH:MM>`、`/作息 忙 <星期> <HH:MM>-<HH:MM>`、`/作息 假期 <日期>[..<日期>]`、`/作息 查看`、`/作息 删除 <编号>` | 作息手动修正（R-ACT-005） |
| `/后端 deepseek\|style\|hybrid` | 切换生成后端（未部署风格模型时拒绝 style/hybrid） |
| `/重来` | 撤销机器人上一轮（标记为负例）并重新生成 |
| `/不像 [正确说法]` | 标记上一轮不像她；附正确说法时生成偏好对（R-LRN-002） |
| `/记住 <内容>`、`/忘掉 <内容或编号>`、`/记忆 [页码\|关键词]` | 记忆管理（R-MEM-009） |
| `/评分 <1-5> [备注]` | 对最近一周主动消息与整体体验打分（R-EVAL-005） |
| `/费用 [今天\|本月]` | 费用与缓存命中率 |
| `/导入 <路径>` | 触发增量导入（也可用 CLI） |

- **R-CMD-003** 指令解析容错：全角斜杠、多余空格、中英文冒号都能识别；参数错误时回复用法示例。

## 19. 学习（R-LRN）

- **R-LRN-001** 用户在机器人会话中说的事实与待跟进按 R-MEM-007 进入记忆。
- **R-LRN-002** 反馈：`/重来`、`/不像` 标记负例；`/不像 <正确说法>` 生成偏好对 `(prompt, chosen=用户给的说法, rejected=机器人原回复)` 存 `preference_pairs`，其中 prompt 存为结构化样本（系统段 + 对话轮次，与 SFT 导出同一结构），不存渲染后的字符串，避免 DPO 训练时被模板再包一层；自然语言纠正（如"她不会这么说"）由 DeepSeek 识别后询问用户是否记为纠正（系统消息，回复"是"即记录）。
- **R-LRN-003** 纠正汇总进人设卡 `[不要这样]` 区块：去重、合并，上限 30 条；每周离线整理一次。规则只描述说话方式，不得含日期、具体事件或新事实（生成后校验，不合格的丢弃；事实类纠正走记忆）。
- **R-LRN-004** 机器人回复永远不进入风格样本、检索库与 SFT 训练集（R-STO-007、R-RET-004、R-TRN-004）。偏好对中的 `rejected` 只用于 DPO 负例。

## 20. 运维（R-OPS）

- **R-OPS-001** Windows 安装：`scripts/windows/install.ps1` 完成 uv 安装检查、依赖同步（`uv sync --frozen`）、首次配置向导（`twin setup`：写入 Key、SMTP、目标会话、时区）、注册计划任务：触发器为当前用户登录；登录类型 `InteractiveToken`（只在用户登录时运行——keyring、Windows 通知与二维码窗口都需要用户会话）；动作为虚拟环境中的 `twin.exe supervise`（或 `uv run --frozen --no-sync twin supervise`，避免开机时改动环境）；无执行时限；不允许并行实例；任务自身的"失败后重启"作为第二道保险。`twin supervise` 以子进程运行 `twin run`，异常退出时按 5 秒到 5 分钟的指数退避重启并记录，正常退出则结束。提供 `twin service install|uninstall|start|stop|status`。说明如需断电或系统更新重启后无人值守恢复需开启自动登录（并说明风险）。`uninstall.ps1` 反向操作。
- **R-OPS-002** 运行期间调用 `SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)` 防休眠；单实例锁；优雅退出（Ctrl+C、计划任务停止）时完成正在发送的气泡并持久化状态。
- **R-OPS-003** 健康检查（每分钟）：通道登录与最近一次成功长轮询、DeepSeek 错误率与熔断、风格模型端点、磁盘剩余、任务队列积压、备份新鲜度；`twin health` 输出；每周一次 SQLite `integrity_check`。
- **R-OPS-004** 告警：微信登录失效（立即，提示重新扫码；二维码只在本机显示，不通过邮件发送）、DeepSeek 连续失败或熔断、预算 80%/超限与降级、风格模型掉线、备份失败、磁盘不足、重训提醒；渠道为邮件（SMTP，去重与限频：同类 1 小时最多 1 封）+ Windows 通知。
- **R-OPS-005** 费用：`cost_ledger` 日/月汇总，`/费用` 与 `twin cost report`；月报（邮件）。
- **R-OPS-006** 备份：每天当地 `backup_hour_local` 点用 SQLite 在线备份 API 生成一致性快照，连同向量库与媒体清单打包后 AES-GCM 加密写入 `data/backups/`（记录所用密钥 `key_id`，R-STO-003）；保留 14 个日备份 + 8 个周备份；配置 `ops.backup_mirror_dir` 时同步一份异地副本；`twin backup now|list|verify|restore <文件>`；恢复有集成测试（含密钥轮换后恢复旧备份）。
- **R-OPS-007** 日志：INFO 及以上不含正文；DEBUG 正文需脱敏；按 10MB × 10 份轮转；结构化（JSON 行）。
- **R-OPS-008** 一键删除：`twin purge --all` 先列出将删除的内容与数量，要求输入确认短语"删除她的全部数据"后执行：删除全部与她相关的数据（数据库、媒体、向量库、备份与异地副本、训练集、本地模型与适配器、报告、日志），并删除 `keyring` 中的全部数据库密钥与备份密钥（加密粉碎，使残留备份不可解密）；也提供 `--training-only`。属于独占命令（R-ARCH-006）。
- **R-OPS-009** 首次运行与升级：`twin doctor` 检查 Python、依赖、`tzdata`、keyring、磁盘、网络（DeepSeek、iLink 主机）、`chinese-calendar` 是否覆盖当前与下一年（R-LLM-007）、显卡与 llama.cpp CUDA 运行库（若用本地风格模型）、计划任务与电源设置。
- **R-OPS-010** 版本与回滚：画像、人设卡、提示词模板、风格模型都有版本；`twin rollback <对象> <版本>`。

## 21. 风格模型训练（R-TRN，在 AutoDL 上）

- **R-TRN-001** 基座模型（可配置，默认按 GPU 档位）：

| 档位 | GPU | 默认基座 | 方法 | 备注 |
| --- | --- | --- | --- | --- |
| `5090-8b` | RTX 5090 32GB | Qwen/Qwen3-8B | LoRA bf16（r=32, α=64） | 推荐，本地推理门槛低 |
| `5090-14b` | RTX 5090 32GB | Qwen/Qwen3-14B | QLoRA 4-bit（bitsandbytes） | |
| `pro6000-14b` | RTX PRO 6000 96GB | Qwen/Qwen3-14B | LoRA bf16（r=32） | |
| `pro6000-32b` | RTX PRO 6000 96GB | Qwen/Qwen3-32B | LoRA bf16（r=16），梯度检查点 | 本地推理需大显存，否则走远程 |

模板 `qwen3_nothink`（纯 ChatML、不含 think 标记；训练与推理逐 token 一致，见 R-TRN-011），`mask_history: true`（只在最后一轮、即她的目标回复上计算 loss），`cutoff_len` 2048，`lora_target: all`，学习率 1e-4，2 个 epoch（数据 < 20k 样本时 3），cosine，warmup 0.05，验证集 loss 早停（patience 2 次评估）。`5090-14b` 的 QLoRA 适配器合并时加载 bf16 基座（不设 `quantization_bit`，`export_device: cpu`）。
- **R-TRN-002** 训练集导出 `twin train export`：每个样本以她的一个真实回复块为学习目标，输入为其前面最多 8 个合并轮次；系统提示由 `StylePromptBuilder` 生成（pre_holdout 范围的精简人设卡 + 当时当地时间与星期 + 她当时的作息状态 + `memory_view(as_of=样本时刻)` 的记忆块），与线上 style 后端完全同一代码；全部数据经 R-TRN-013 的防泄露视图读取。
- **R-TRN-003** 表示约定：目标（她的回复块）中连发用换行；表情包写成 `[表情包:<标签>]`；表情代码原样；引用写成首行 `[引用:...]`；机器人无法复现的行（图片、语音、视频、通话、转账、红包、位置、文件、链接、聊天记录、系统事件）从目标中删除，删除后为空则丢弃该样本（R-SAFE-006）。上下文（用户与她之前的轮次）中这些仍以事件文字呈现（R-IMP-007）。
- **R-TRN-004** 只用她的真实消息作目标；机器人回复不得进入（导出函数只接受 `messages` 表；有测试）。
- **R-TRN-005** hybrid 训练数据：按比例（默认 30%）为样本合成"规划"（DeepSeek 读她的真实回复反推 `{intent, facts_to_use, tone, bubble_hint}`，非高峰执行），放入提示词的规划字段，使风格模型学会"按规划用她的语气说"；其余样本不带规划。
- **R-TRN-006** 切分：按时间，最后 10% 为测试集（与 R-RET-003 留出集一致），之前 5% 为验证集，其余训练；不得随机打乱跨越时间边界。
- **R-TRN-007** 脱敏：导出前过 R-LLM-009；同一实体在整个数据集中用一致的占位符。
- **R-TRN-008** 训练包 `twin train bundle --profile <档位>`：生成 `bundle.tar.zst`（数据集、`dataset_info.json`、由模板生成的 LLaMA-Factory YAML、AutoDL 脚本、清单与 sha256），用口令（scrypt + AES-GCM）加密；口令不落盘。
- **R-TRN-009** AutoDL 脚本（`training/autodl/`，bash，`set -euo pipefail`，可重复执行）：
  - `setup.sh`：检查 `nvidia-smi` 与计算能力；确认 `torch>=2.7` 且 CUDA 12.8 构建并包含 `sm_120`（Blackwell），不满足则安装 cu128 版 PyTorch；不使用 CUDA 13.x；`source /etc/network_turbo` 加速 GitHub/Hugging Face；安装固定版本的 LLaMA-Factory、bitsandbytes 与 `zstd`（解包训练包需要）；注意力实现用 `sdpa`（不编译 flash-attn）；`HF_HOME`、`MODELSCOPE_CACHE`、`PIP_CACHE_DIR`、`TMPDIR` 全部指向数据盘；模型经 ModelScope（`USE_MODELSCOPE_HUB=1`）下载到 `/root/autodl-tmp/models`；数据与输出放 `/root/autodl-tmp/twin`（数据盘，非系统盘）；按档位检查数据盘剩余空间（按基座实际大小计算，约为 8B ≥ 70GB、14B ≥ 110GB、32B ≥ 230GB；AutoDL 默认数据盘 50GB，不足时停止并提示到控制台扩容）与内存（QLoRA 合并需在 CPU 上加载 bf16 基座）。
  - `decrypt.sh`：解密训练包到数据盘。
  - `train.sh <profile>`：运行 SFT，日志与 loss 曲线落盘，支持断点续训（`resume_from_checkpoint`）。
  - `eval.sh`：在测试集上生成回复（固定采样参数与种子）输出 JSONL，并计算验证集 loss。
  - `dpo.sh`：偏好对 ≥ 200 时在 SFT 适配器上做 DPO（LLaMA-Factory `stage: dpo`）。
  - `export.sh`：合并 LoRA（QLoRA 档位在 bf16 基座上合并，`export_device: cpu`）→ 用 llama.cpp `convert_hf_to_gguf.py` 转 F16 GGUF → 删除合并后的 HF 权重 → `llama-quantize` 生成 Q4_K_M、Q5_K_M、Q8_0 → 删除 F16 GGUF → 计算 sha256 → 打包下载清单；同时保留 LoRA 适配器（远程 vLLM 用基座 + 适配器）。
  - `cleanup.sh`：安全删除数据集、解密目录、日志中的样本，打印"请在 AutoDL 控制台释放实例"。
- **R-TRN-010** 本地编排 `twin train remote ...`：用 `asyncssh`（纯 Python，支持 AutoDL 的密码登录，Windows 无需安装 OpenSSH）连接 AutoDL 实例（主机、端口、用户在 `autodl.*` 配置中，密码存 keyring 或使用密钥文件）：上传训练包、远程执行脚本、实时拉取日志、下载产物到 `data/models/<run_id>/`、调用清理；每一步可单独执行与续跑。`training_runs` 记录档位、数据版本、超参、耗时、指标、产物 sha256。
- **R-TRN-011** 模板一致性（训练 = 推理，逐 token）：
  1. 唯一的训练模板是固定版本 LLaMA-Factory 的 `qwen3_nothink`。经核对 LLaMA-Factory 0.9.x 源码，它是纯 ChatML：`<|im_start|>system\n…<|im_end|>\n`、`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`、回复 + `<|im_end|>\n`，不含 think 标记；实现时以所固定版本的源码再核对一次并在测试中注明版本。
  2. 一致性定义：对同一样本，`StylePromptBuilder` 渲染的提示词字符串经基座 tokenizer 编码后的 token id，必须等于该版本 LLaMA-Factory 模板编码器（`get_template_and_fix_tokenizer` + `encode_multiturn`）输出中最后一轮之前的全部 token；且最后一轮的标签（response_ids）恰好是推理时模型应在该提示词之后生成的内容（含结尾 `<|im_end|>`）。**不**与 transformers 的 `apply_chat_template` 比较：Qwen3 官方模板在最后一轮插入空 think 块，与训练格式本来就不同。
  3. 第 09 轮用模板源码中的格式字符串做字符级单测；第 13 轮在固定版本 LLaMA-Factory 上做 token 级集成测试（本地只需 tokenizer 文件；AutoDL `setup.sh` 末尾再跑一次）。
  4. 部署时（第 14 轮）做服务端分词核对：把同一渲染字符串交给 llama-server `/tokenize`（或 vLLM `/tokenize`），与 HF tokenizer 的结果逐 id 比对，检测 BOS 与特殊 token 处理差异；不一致则拒绝启用该模型。
  5. 多轮样本只在最后一轮计算 loss（`mask_history: true`）。ShareGPT 要求先 human 后 gpt 交替：上下文以她的消息开头时，开头连续的她的轮次放入系统段的"前文"小节，训练与推理用同一函数处理。
  6. 风格模型输出出现 `<think>` 片段视为硬违规（R-ENG-008）。
- **R-TRN-012** 继续训练：真实消息自上次训练增加 ≥ 10% 时提醒（告警 + `/状态`）；默认从基座用全量数据重训（不在旧适配器上叠加），偏好对单独走 DPO。
- **R-TRN-013** 防止未来信息泄露（训练与评估共用）：
  1. 唯一切分函数 `holdout_cutoff()`（R-RET-003）。派生数据分两个范围：`live`（全部数据，线上用）与 `pre_holdout`（只用切分点之前的数据，训练与评估用）；画像（R-PROF-004）、作息模型（R-ACT-006）、人设卡（R-PERS-001）都按两个范围各存版本；表情包上下文修正只用切分点之前的使用记录（R-STK-003）。
  2. 精简人设卡只含风格内容（R-PERS-004）；事实只经 `memory_view(as_of=t)` 进入（R-MEM-010），从目标回复块本身及之后的消息抽出的事实不可见。`[不要这样]` 纠正规则只允许描述说话方式（生成时校验不含日期与具体事件，R-LRN-003），因此评估中 DeepSeek 后端的 pre_holdout 完整版人设卡附加当前的 `[不要这样]`；精简版仍不含它。
  3. 训练集导出（R-TRN-002）与评估沙盒（R-EVAL-009）通过同一个 `AsOfView(t)` 读取全部上下文数据：记忆、摘要、待跟进、生活线（为空）、检索例子（只返回 `reply_at < t` 的窗口）、表情包与表情代码（只含她在 t 之前用过的表情包，使用次数与新近度按 t 之前计算，表情代码词表与频率、表情包占比来自 pre_holdout 画像）、她当时的状态（由 pre_holdout 作息模型的纯函数 `typical_state(当地时间, 日类型)` 推断，R-ACT-006）。
  4. 必须有注入测试：合成数据中放一个只在样本时刻之后（含目标块本身）出现的事实，断言该样本的训练提示词与评估沙盒提示词都不含它；断言 pre_holdout 范围的画像、作息与人设卡生成输入中没有切分点之后的消息 id。

## 22. 风格模型部署与后端（R-SRV）

- **R-SRV-001** 模型登记 `model_registry`：版本、档位、量化、sha256、评估分数、是否启用；并锁定该模型训练时使用的 `template_version`、`persona_version`（pre_holdout 精简人设卡版本）、`profile_version`、`dataset_version`——该模型作为后端时，`StylePromptBuilder` 用这些锁定版本渲染。表与 `twin model register|list|show` 在第 13 轮（产物下载后登记），启用、切换与服务在第 14 轮。
- **R-SRV-002** 本地推理：`scripts/windows/get_llamacpp.ps1` 下载固定版本的 llama.cpp Windows 发行包（有 NVIDIA 显卡选 CUDA 构建，并同时下载同版本的 `cudart` 运行库包；否则 CPU 构建），校验哈希；`twin model serve` 以子进程启动 `llama-server.exe`（端口 8081、上下文 4096、全部层上 GPU、并发 1），启用前做分词核对（R-TRN-011），健康检查，异常重启，随主程序退出；按显存自动推荐量化档（Q4_K_M/Q5_K_M/Q8_0）。
- **R-SRV-003** 远程推理（无合适本地显卡时）：在 AutoDL 上用 vLLM 加载基座 + LoRA（只监听实例本机），本地用 `asyncssh` 端口转发隧道接入（`style_model.tunnel`）；`StyleModelClient` 用 `vllm_completion` 模式；`twin model tunnel` 管理隧道与重连；`/状态` 显示远程实例按小时计费的提醒。
- **R-SRV-004** 后端选择：`/后端` 指令；风格模型不可用时自动回退 deepseek 后端并告警；回退与恢复记录在案。
- **R-SRV-005** 上线门槛：新风格模型（以 style 或 hybrid 方式）同时满足以下条件才允许设为默认：① R-EVAL-001 盲测中猜对率低于 deepseek 后端，单侧两比例检验 p < 0.1，且参与比较的每个后端各有 ≥ 50 对有效判断；② 留出集风格指标（R-EVAL-002 的 eval_items 模式）每项偏差在 ±30% 内。否则保留 deepseek 后端。`--force` 可强制启用，但记审计、在 `/状态` 标注"未通过门槛"，且不参与 R-LLM-008 第 ④ 级降级。

## 23. 评估与验收（R-EVAL）

- **R-EVAL-001** 盲测 `twin eval blind --backend <名称> --n 50`：在评估沙盒（R-EVAL-009）中，从留出集抽取上下文（排除以前盲测用过的上下文，排除真实回复含事件文字或媒体占位行的样本），按样本时刻生成机器人回复，与她的真实回复配对；两条候选经同一个渲染函数统一呈现（气泡换行、表情包都显示为"[表情包：描述]"、引用与表情代码同样处理），随机左右顺序，用户在终端选"哪条是她"（可跳过，跳过不计入）；记录到 `eval_items`；输出猜对率（点估计）与 95% Wilson 置信区间；支持多后端对比与按后端分组统计。门槛按点估计判定，且有效判断 ≥ 50 对：M1 ≤ 70%，M4 ≤ 60%。
- **R-EVAL-002** 风格指标：两种数据源——`--source live`：机器人一周内的真实输出（`bot_turns` 非指令出站）；`--source eval_items --backend X`：评估沙盒对留出集上下文生成的回复（M5 门槛用）。计算 R-PROF-002 的核心指标（文字长度中位数、逗号率、连发中位数、表情包占比、表情代码率、引用率），与她的画像对比（live 用 live 画像，留出集用 pre_holdout 画像），每项偏差在 ±30% 内为通过；`twin eval style` 输出报告。
- **R-EVAL-003** 记忆测试：自动从事实库生成 20 题（10 题来自真实记录、10 题来自机器人对话），在评估沙盒（R-EVAL-009 的 live 模式：当前时刻、不发微信、不写入真实会话）中提问，DeepSeek 判分、用户可复核；正确率 ≥ 80%。机器人对话来源的可出题事实不足 10 条时提示先多聊几天，此时门槛判定为"未通过（样本不足）"，不用真实记录替补。
- **R-EVAL-004** 前后一致：每周离线让 DeepSeek 审阅生活线与机器人对话，列出自相矛盾之处；明显矛盾 ≤ 1 次/周；报告供用户确认。
- **R-EVAL-005** 主动消息审计（第 10 轮实现，含 `/评分`）：基于 `proactive_log`，深睡核心时段主动 0 次；每天次数在设定范围内；间隔与追发规则无违规；边缘消息每周不超过上限；用户每周 `/评分` ≥ 4/5。
- **R-EVAL-006** 稳定性（第 12 轮实现）：无人值守连续运行 7 天不中断；掉线 10 分钟内收到提醒（以健康检查日志与告警记录核对；7 天内至少做一次断网演练）。
- **R-EVAL-007** 成本：月费用 ≤ 15 美元（`twin cost report`）；一次性批任务费用（R-LLM-014）单列，不计入该门槛。
- **R-EVAL-008** `twin eval report` 汇总以上全部，输出 Markdown 到 `data/reports/eval-<日期>.md`，并标明各里程碑门槛是否达成（M0–M5）。
- **R-EVAL-009** 评估沙盒 `EvalSandbox`：与线上同一套提示词、后端、后处理与表情包选择，但：通道为内存通道（实现 `Channel` 接口，绝不实例化 `IlinkChannel`，有测试）；跳过拟人延迟与发送节奏，只取生成与后处理结果；不写 `bot_turns`、`conversation_state`、`proactive_log`、`feedback`、`facts`、`lifeline_events`、`daily_plans`，不触发记忆抽取与学习，结果写 `eval_items`；允许的其他写入只有：`eval_runs`、`cost_ledger`、`jobs`（排队图片描述与批任务记录）、`alerts`、`settings` 中的 `state_version` 与 token 估算校准系数；时钟注入为样本时刻。两种模式：`holdout`（盲测与留出集风格指标：通过 `AsOfView(t)` 读取 pre_holdout 范围的人设卡、画像、作息与 as-of 记忆，检索只返回早于 t 的窗口，生活线与机器人对话为空，近期对话 = 样本上下文，各后端都用同样的最多 8 个合并轮次，与 style 后端和训练一致）与 `live`（记忆测试：当前时刻、live 范围）。费用记 `purpose=eval`（批量生成属于一次性批任务，R-LLM-014）。
- **R-EVAL-010** 里程碑门槛判定 `twin eval gate M0|M1|M2|M3|M4|M5`：按 §26 读取最新评估结果计算是否通过，把结果与证据（运行 id、数值、样本数）写入 `eval_runs(kind=gate)`，退出码 0/1；`--check` 只读取已存结果。各轮的"门槛检查"必须运行它；门槛未通过时停在该轮继续改进，不得开始下一轮；下一轮开始时先运行 `twin eval gate <上一门槛> --check` 确认。唯一例外是 M5：第 14 轮只要求已运行并记录 `twin eval gate M5`，未通过时保留 DeepSeek 后端并照常进入第 15 轮。

## 24. 隐私（R-PRIV）

- **R-PRIV-001** 全部真实聊天记录在本地导入和保存，不上传；仓库不含任何真实数据（CI 检查：`tests/fixtures` 只允许合成数据生成器产物，提交前扫描敏感模式）。
- **R-PRIV-002** 每次发往 DeepSeek 的只有当次需要的少量片段，且经过脱敏（R-LLM-009）；绝不整库上传。
- **R-PRIV-003** 训练数据上云：只上传脱敏、加密的训练包；训练完执行清理脚本并提醒释放实例；`training_runs` 记录清理完成时间。
- **R-PRIV-004** 本地加密：数据库敏感字段、媒体、备份均加密（R-STO）。
- **R-PRIV-005** 一键删除（R-OPS-008）覆盖原始记录、画像、表情包库、记忆、训练集与模型。
- **R-PRIV-006** 机器人只对用户说话（R-CH-007）；名字与头像由用户在 ClawBot 会话设置中自行设置，代码不修改。

## 25. 非功能（R-NFR）

- **R-NFR-001** 生成耗时（不含刻意延迟）p95 < 20 秒（deepseek 非思考）、< 60 秒（思考）；超时走 R-ENG-010。
- **R-NFR-002** 常驻内存 < 1.5GB（不含 llama-server）；CPU 空闲时 < 2%。
- **R-NFR-003** 启动到可收消息 < 30 秒（向量模型懒加载）。
- **R-NFR-004** 代码质量：`ruff` 无告警；全部 `src/twin` 通过 `mypy --strict`；全部 `src/twin` 行覆盖率 ≥ 85%（且每个子包 ≥ 75%）；单元测试不访问网络（`respx` 拦截）；集成测试可选地访问真实 DeepSeek（需显式环境变量开启）；`src/` 中不得有桩代码（第 16 轮用关键词与 AST 扫描：函数体只有 `pass`/`...`/常量返回、`NotImplementedError`、TODO/FIXME/stub/placeholder/mock/fake/dummy/简化/示例 等；`Protocol` 与抽象方法除外）。
- **R-NFR-005** 所有时间相关逻辑有基于注入时钟的测试，包括 DST 切换日与时区切换日。
- **R-NFR-006** 文档：`README.md`（安装与日常使用）、`docs/RUNBOOK.md`（首次部署、扫码登录、导入全量记录、训练与部署风格模型、回国切换时区、备份恢复、一键删除、故障排查）、`docs/CHANNEL_REPORT.md`（M0 实测）、`docs/LLM_REPORT.md`（M0 实测）、`docs/DECISIONS.md`（取舍记录，R-SCOPE-009）、`docs/ARCHITECTURE.md`。

## 26. 里程碑门槛（与产品文档一致；每个阶段过关后再进下一个）

| 里程碑 | 内容 | 过关条件 | 判定时机 |
| --- | --- | --- | --- |
| M0 技术验证 | ClawBot 收发、主动窗口与条数、图片/GIF、引用、正在输入；DeepSeek 思考开关与缓存命中 | 主动消息在窗口内稳定送达（R-CH-009/010 达标）；DeepSeek 探针按 R-LLM-013 的判定口径通过；否则停下评估企业微信通道 | 第 01、02 轮（报告）；第 09b 轮起可用 `twin eval gate M0` 复核 |
| M1 能聊、像她 | F1 导入、F2 画像、F3 生成与节奏、F4 表情代码 | 盲测猜对率 ≤ 70% | 第 09b 轮末 |
| M2 记得住 | F4 表情包库、F5 记忆、F10 指令 | 记忆测试 ≥ 80% | 第 09b 轮末首次判定；第 11 轮全部指令完成后复判 |
| M3 有作息、会主动 | F6 作息与真实模式、F7 主动、F9 思考开关 | 连续 7 天真实运行：深睡时段主动 0 次、每天次数在范围内、间隔与追发零违规；该周 `/评分` ≥ 4/5 | 第 10 轮完成后观察 7 天 |
| M4 持续成长 | F8 学习、增量导入、F11 运维 | 无人值守 7 天；盲测 ≤ 60% | 第 12 轮完成后观察 7 天 |
| M5 风格模型 | F12 训练、评估、量化部署 | R-SRV-005；不通过则保留 DeepSeek 后端（不阻塞第 15、16 轮） | 第 14 轮末 |
