# 第 04 轮：风格统计画像与作息活动模型

> 里程碑：M1（画像）/ M3（作息数据侧）· 前置：第 00–03 轮全绿

## 先读
`CLAUDE.md`；`docs/SPEC.md` §0 参考数据、§8 统计画像、§9 作息活动模型、R-SCOPE-005、R-STO-006。

## 本轮目标
从真实记录中算出"她怎么说话"的全部量化指标（可抽样的经验分布），以及"她什么时候醒、忙、睡、先开口、多久回"的作息模型；两者版本化、可对比、可回滚，并在导入后自动重算。本轮还确定全系统唯一的留出切分点 `holdout_cutoff()`，并让画像与作息各有 live 与 pre_holdout 两个范围（防止训练与评估看到未来，R-TRN-013）。

## 必须实现的需求
R-PROF-001～005、R-ACT-001～006、R-SCOPE-005（作息按当地钟点学习的部分）、R-RET-003（`holdout_cutoff()` 的定义与持久化）、R-TRN-013（画像与作息的 pre_holdout 范围）、R-STO-006（本轮的表：`profile_versions`、`activity_models`、`routine_overrides`）、R-IMP-011（注册"画像与作息重算"导入后钩子与回填命令）。

## 详细要求

### A. 基础单元与定义（必须写进模块文档字符串并有测试）
1. **合并连发块**：同一发送者、相邻两条间隔 ≤ `profile.burst_gap_s`（120 秒）视为同一块。
2. **会话段**：两条消息间隔 > `profile.segment_gap_min`（60 分钟）即新段（与 R-RET-001 共用同一配置项）。
3. **回复延迟**：对方块最后一条到本方块第一条的时间差，仅当二者在同一会话段内时计入；跨段的记为"未即时回复"单独计数。
4. **先开口**：间隔 ≥ 60 分钟后第一条消息的发送者。
5. 只用 `messages` 表；文字类指标只用 `kind=text`（以及 quote 的回复正文）。

### B. 经验分布工具
`EmpiricalDistribution`：从样本构建（保存分位点或直方图，可序列化为 JSON）、`sample(rng)`（逆 CDF + 线性插值，支持截断上下限）、`quantile(q)`、`mean()`；支持按条件分桶（如按小时桶的回复延迟）；样本过少的桶回退到上一级（全天）分布。

### C. 统计画像（R-PROF-002/003/005）
1. 实现 SPEC 列出的全部指标，分别对她和用户计算；表情代码用正则 `\[[一-龥A-Za-z]{1,6}\]` 并与微信表情代码表核对（表在仓库内维护，可扩展）；Unicode emoji 用 `emoji` 包或 Unicode 属性识别。
2. 对用户的称呼：统计她的消息中句首/句尾出现的高频称谓（基于 n-gram 频率与位置，输出候选与频率，不硬编码任何词）。
3. 双窗口（全量、最近 `profile.recent_days` 天）与 `profile.recency_weight` 混合；每个指标保存原始值与混合值。
4. "数字风格规则"生成器：根据指标阈值生成简短中文规则文本（例如逗号率 < 5% → "几乎不用逗号，用分条代替"；中位长度 → "单条通常 X–Y 字"；连发中位 → "常连发 2–3 条"；表情代码 → "常用 [拥抱][亲亲]，常连用 3 个"）；规则模板放配置表，可测试。
5. 两个范围（R-TRN-013）：`live` 用全部数据；`pre_holdout` 只用 `holdout_cutoff()`（见 H）之前的消息，其"最近 90 天"窗口以切分点为终点。两者都写入 `profile_versions`（字段 `scope`），线上只读 live，训练与评估只读 pre_holdout。

### D. 版本化（R-PROF-004）
- 每次重算写 `profile_versions(id, created_at, data_range, metrics_json(EncryptedJSON 中只含统计，不含原文), summary_rules, parent_id)`；`twin profile history|show [版本]|diff <a> <b>|rollback <版本>`；`settings` 记录生效版本。
- 差异摘要：列出变化超过 10% 的指标。

### E. 作息活动模型（R-ACT-001～004/006）
1. 时间换算：消息 UTC 时刻 → 她当时所在地时区（`time.source_timezone`，若配置了 `time.source_timezone_ranges` 则按日期区间选择）的当地钟点（15 分钟一格，0–95）与当地日期 → 日类型（工作日/周末/节假日；中国时区用 `chinese_calendar.is_workday()`（含调休，未收录年份按第 01 轮同一回退逻辑），美国时区用 `holidays.US`）。日历判定复用第 01 轮的同一个日历模块，不另写一份。
2. 对她计算：每格发消息率、每格先开口率、每格回复延迟条件分布（用 B 的分桶）；循环平滑（跨午夜连续，高斯核，σ 可配）。
3. 睡眠推断（R-ACT-003）：
   - 先用全部数据的当地钟点活跃曲线（平滑后）找到最低活跃点，以它作为每个"活动日"的分界（不用固定的午夜或正午，避免把睡眠切成两段）；
   - 每个活动日内找最长的、≥ `activity.sleep_min_hours`（3 小时）、发消息率低于阈值的连续区间作为当天睡眠；汇总入睡/起床时刻分布（环形统计处理跨午夜）；深睡核心与边缘按 `activity.edge_minutes`；
   - 有效天数少于 `activity.min_valid_days` 时标记低置信，默认睡眠取整体活跃曲线中最长的低谷（不使用任何写死的钟点）；
   - 合理性检查：睡眠核心落在当地 10:00–18:00 内时写告警"可能是 source_timezone 设置不对"，并在 `twin profile show` 醒目提示；
   - `twin profile show` 输出推断出的睡眠时段（工作日/周末分别）并请我确认，不对就用 F 的作息修正；
   - 注意：SPEC §0 的 7 天样本里有多个低谷（04–07、12–15、17–19 点），算法必须对这种数据稳健。
4. 忙碌推断：工作日中稳定低活跃、非睡眠、且该时段回复延迟显著变长（与全天中位相比 ≥ 3 倍）的时段，输出带置信度的忙碌区间及该时段延迟分布。
5. 输出 `ActivityModel`（版本化，存 `activity_models`，live 与 pre_holdout 两个范围），提供查询接口：`rate_at(local_slot, day_type)`、`initiation_rate_at(...)`、`latency_distribution(local_slot)`、`sleep_profile()`、`busy_windows(day_type)`、`typical_state(local_time, day_type) -> deep_sleep|sleep_edge|busy|free`（纯函数：按入睡/起床分布的中位数与忙碌窗口判定，不抽样；供第 07 轮 `AsOfView` 推断历史时刻她的状态，R-ACT-006）；这些接口只接受"当地钟点"，不关心当前时区（时区换算在第 08 轮 `TimeService` 完成）。

### F. 手动修正存储（R-ACT-005 的数据侧；指令在第 11 轮）
- `routine_overrides`：类型（睡眠区间 / 每周忙碌 / 节假日日期区间）、参数、创建时间、启用状态；查询时手动修正优先于推断。提供 Python API 与 `twin routine list|add|remove` CLI（与第 11 轮微信指令共用同一 API）。

### G. 导入后钩子与回填
- 注册"画像与作息重算"钩子：导入有新增她的消息时重算 live 与 pre_holdout 两个范围并写新版本；导入报告中输出"风格变化"摘要（来自 D 的差异）。
- 回填命令 `twin profile rebuild [--scope live|pre_holdout|all]`（重任务）；验收时对已导入的真实数据执行一次。

### H. 留出切分点（R-RET-003、R-TRN-013）
1. `holdout_cutoff()`：把她的全部真实回复块按第一条消息时间排序，取最后 `retrieval.holdout_ratio`（10%）的起始时刻作为切分点；首次计算后持久化到 `settings`，之后导入新数据**不自动移动**；`twin retrieval resplit`（第 05 轮提供命令，函数在本轮）显式重算并提示会影响评估可比性与 pre_holdout 版本（重切后自动排队重算所有 pre_holdout 派生数据）。
2. 这是全系统唯一的切分实现：第 05 轮检索、第 06 轮人设卡与表情包、第 09b 轮评估、第 13 轮训练都调用它（有一个扫描测试：仓库里没有第二处计算 10% 切分的代码）。

## 测试要求
- 用第 03 轮生成器扩展出"已知规律"的合成数据：例如她固定 01:00–08:30 不发消息、工作日 13:00–17:00 回复慢、每天先开口 4 次、逗号率 3% 等；断言推断结果在容差内（睡眠边界 ±30 分钟、先开口均值 ±15%、逗号率 ±0.5 个百分点）。
- 跨午夜睡眠、跨夏令时日期（芝加哥 2026-03-08 与 2026-11-01）的当地钟点换算正确。
- `source_timezone=Asia/Shanghai` 时节假日判定使用中国日历。
- 经验分布抽样的统计性质（大样本抽样的中位数与原分布中位数误差 < 5%）。
- 版本化与回滚；规则生成器各分支。
- 用 SPEC §0 小时向量（按该比例生成 28 天合成消息）测试睡眠推断：不报错；输出的睡眠区间是该向量中持续 ≥ 3 小时的低活跃区间；合理性告警是否产生与"睡眠核心是否落在 10:00–18:00"的判断一致。
- 另生成一组在 `Asia/Shanghai` 01:00–08:00 睡眠的合成数据，但以 `America/Chicago` 学习：睡眠核心落在芝加哥白天，必须产生合理性告警；以 `Asia/Shanghai` 学习则不告警。
- `typical_state()` 在一天中各时刻的取值与边界。
- `source_timezone_ranges`：两段不同时区的合成数据，当地钟点换算按区间正确切换。
- pre_holdout 范围：其输入中没有切分点之后的消息 id；`holdout_cutoff()` 持久化且新导入后不移动。

## 验收
```
uv run pytest -q
uv run twin profile rebuild --scope all   # 对已导入的真实数据回填
uv run twin profile show          # 输出指标与作息概览（当地时间）；我确认推断的睡眠时段，不对就用 twin routine add 修正
uv run python scripts/trace_check.py --round 04
```

## 不要做
- 不要在本轮生成人设卡（第 06 轮）或使用任何 LLM。
- 不要在画像 JSON 中保存消息原文；高频整句与 n-gram 单独加密保存，只在本地使用。

## 完成后汇报
按 CLAUDE.md 格式；若已导入真实数据，附 `twin profile show` 的数字摘要（不含原文），并与 SPEC §0 的样本指标对比。
