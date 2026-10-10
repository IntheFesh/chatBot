# 第 07 轮：记忆系统（事实库、每日摘要、生活线与待跟进的数据层、检索组装、时间回放）

> 里程碑：M2 · 前置：第 00–06 轮全绿

## 先读
`CLAUDE.md`；`docs/SPEC.md` §13 记忆全部、R-LLM-003/007/008/009、R-STO-002/006、R-TRN-002（训练集要用 `memory_view(as_of)`）。

## 本轮目标
让机器人"记得住"：真实记录里的事、和机器人聊过的事、机器人自己编过的生活细节，都按来源与可信度管理，冲突可裁决，能按话题与日期取回，并支持"截至某时刻已知的记忆"视图（训练集防泄露）。生活线的**每日生成**与主动跟进在第 08、10 轮接入，本轮完成数据结构与写入/查询 API。

## 必须实现的需求
R-MEM-001～004、R-MEM-006～011、R-MEM-005（数据层：表、写入、查询、一致性校验 API；每日生成在第 08 轮）、R-STO-006（本轮的表：`facts`、`daily_summaries`、`lifeline_events`、`followups`）、R-IMP-011（注册"新日期范围记忆回放"钩子与回填命令）、R-TRN-013（`memory_view(as_of)` 与组合视图 `AsOfView(t)`）、R-LLM-014（全量回放作为一次性批任务）。

## 详细要求

### A. 数据模型
1. `facts`：SPEC R-MEM-003 全部字段（含 `event_date`、`recurrence`）+ `evidence_ref`（消息 id 列表或 bot_turn id）+ `importance(1-5)` + `embedding_id`；文本加密。抽取器对生日、纪念日、考试日等输出 `event_date` 与 `recurrence`。
2. `daily_summaries`：`(id, scope(real|bot), local_date, timezone, utc_start, utc_end, text(enc), embedding_id, version)`；同一 scope+日期唯一，重算产生新版本。
3. `lifeline_events`：`(id, local_date, timezone, start_local, end_local, activity, place, mood, detail(enc), source(plan|improvised), consistency_checked_at)`。
4. `followups`：`(id, text(enc), due_at_utc, window_minutes, source_turn_id, status(open|done|cancelled|expired), created_at, closed_at, close_reason)`。

### B. 抽取（R-MEM-006/007）
1. 抽取器 `FactExtractor`：输入一段对话（真实或机器人会话），DeepSeek JSON 输出新事实（subject、category、text、importance、evidence 消息编号）与待跟进（text、due 时间的自然语言 + 解析后的绝对时间、适合跟进的时间窗）；时间解析以对话发生时的当地时区为基准（"明天下午三点考试"）。
2. 触发：机器人会话静默 30 分钟后排队抽取（第 09 轮写入 `bot_turns` 后调用本 API）；真实记录在回放（D）中抽取。
3. 来源标记：真实记录 → `real_record`；用户在机器人会话中说的 → `user_said`；机器人说的关于"她自己"的新细节 → `bot_invented`（同时写入生活线 `improvised`）；`/记住` → `user_command`（第 11 轮调用）。

### C. 冲突裁决（R-MEM-004/011）
1. 新事实写入前，按向量相似度 + 同 subject/category 召回候选旧事实，交 DeepSeek 判断"相同 / 更新 / 冲突 / 无关"；
2. 规则：真实记录 > 用户说的 > 机器人编的；低优先级不得覆盖高优先级（仅记录为"被否决的候选"）；同优先级以较新者为准，旧的设 `superseded_by` 与 `valid_to`。
3. 导入新真实记录后，与之冲突的 `bot_invented` 事实和生活线被标记失效（R-MEM-011）。

### D. 时间回放（R-MEM-010）
1. `MemoryReplayer`：按当地日期顺序处理真实历史：每天生成 `real` 摘要、抽取事实；每条事实 `known_at` = 证据中最晚消息的时间。
2. 作为离线任务分批执行（非高峰、可续跑、可只跑指定日期范围），进度可查（`twin memory replay status`）；全量回放是一次性批任务（R-LLM-014）：先给出预计费用（基于 token 估算与价格表），我 `twin jobs approve` 后开始，费用记 `one_time` 账目、不触发日预算降级。**回放必须覆盖留出期**（第 09b 轮盲测需要留出样本时刻的 as-of 记忆）。
3. `memory_view(as_of: datetime)`（R-MEM-010）：只返回 `known_at < as_of`（严格小于）且在 as_of 时有效的事实、当地日期早于 as_of 当地日期的摘要、as_of 之前创建且在 as_of 时仍未关闭的待跟进（按 `closed_at` 判断）；as_of 早于机器人上线时，机器人来源的事实、摘要与生活线为空。
4. `AsOfView(t)`（R-TRN-013，第 09b 轮评估沙盒与第 13 轮训练集导出共用，全系统唯一入口）：组合 `memory_view(t)`、pre_holdout 范围的画像/作息/人设卡（第 04、06 轮；完整版附加当前 `[不要这样]`，精简版不含）、检索 `before=t`（第 05 轮）、表情包与表情代码的 as-of 数据视图（第 06 轮）、她当时的状态 `her_state()` = pre_holdout 作息模型的 `typical_state(t 的当地时间, 日类型)`（第 04 轮，时区按 R-ACT-001 取她当时所在地）；生活线与机器人对话为空。不提供任何"读全量"的方法。

### E. 每日摘要（R-MEM-002）
- `DailySummarizer`：`real` 与 `bot` 两个 scope 分别摘要，提示词要求保留人物、事件、情绪、约定与未完成的事，不超过 300 字；作为离线任务在她"起床"前执行（触发时间由第 08 轮日程提供，本轮提供"为指定日期生成"的 API 与任务处理器）。

### F. 检索与组装（R-MEM-001/008）
1. `RecentTurns`：取机器人会话近期对话（合并块，30–40 轮的批量窗口，窗口起点由第 09 轮 `HistoryWindow` 维护，R-MEM-001）——数据来源 `bot_turns`（第 09 轮建表）；本轮实现读取接口与对其的测试替身（测试替身放 `tests/`），在 TRACEABILITY 中标注第 09 轮接入。
2. `MemoryAssembler.build(query_context, now, budget_tokens)`：
   - 召回：事实（向量 + 关键词 + subject 过滤）、相关日期摘要（最近 3 天 + 语义相关的历史日）、今天的生活线、未完成且临近的待跟进、日期相关（纪念日、今天到期的事）。
   - 打分：相似度、重要度、新近度、来源优先级、日期相关性（`event_date` + `recurrence` 判断"今天/明天是纪念日"，待跟进的 `due_at`）加权（权重可配）。
   - 在 token 预算内组装成带小标题的文本块；预算受 R-LLM-008 降级影响。
3. 关键词检索：事实与摘要数量有限，在内存中建倒排索引（启动时从解密数据构建，增量维护），不使用明文 FTS 表。

### G. 记忆管理 API（供第 11 轮指令调用）
- `remember(text)`、`forget(query_or_id)`（硬删除该条及只由它派生的条目，例如由它生成的待跟进；返回删除清单）、`list(page, keyword)`；全部写审计日志（不含正文）。

### H. 导入后钩子
- 新增日期范围 → 自动排队回放这些日期（增量回放估算费用低于 `budget.one_time_usd` 的 10% 时自动批准，否则等我批准）；与已有 `bot_invented` 冲突的条目失效。回填命令 `twin memory replay start --from --to`。

## 测试要求
- 抽取 JSON 校验与时间解析（含跨时区："明天"在芝加哥与北京的差异）。
- 冲突裁决全部分支与优先级规则；真实记录导入后机器人编的事实失效。
- 回放：`known_at` 正确；中断续跑；一次性批任务的估算、批准与超支暂停。
- 防泄露注入测试：合成数据里放一个只在样本时刻 t 的目标回复块中才第一次出现的事实，以及 t 之后的摘要、t 之后才关闭的待跟进；断言 `memory_view(t)` 与 `AsOfView(t)` 渲染出的上下文不含该事实与 t 之后的摘要，待跟进在 t 时显示为未关闭。
- 纪念日：`recurrence=yearly` 的事实在每年当天进入记忆块。
- 组装：预算裁剪、日期相关性（今天到期的待跟进必须进块）、降级级别减半预算。
- forget 删除派生条目。

## 验收
```
uv run pytest -q
uv run twin memory replay estimate        # 对真实数据给出预计费用
uv run twin memory replay start --from <日期> --to <日期>   # 我确认后执行
uv run twin memory replay status
uv run python scripts/trace_check.py --round 07
```

## 不要做
- 不要把记忆正文写入日志或向量库元数据。
- 不要在本轮实现生活线的每日生成或主动跟进的发送（第 08、10 轮）。

## 完成后汇报
按 CLAUDE.md 格式；附回放预计费用与（若已执行）实际费用、事实数量按来源统计。
