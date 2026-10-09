# 第 01 轮：LLM 层——DeepSeek 客户端、思考开关、看图、费用与预算、脱敏、风格模型客户端、M0 探针

> 里程碑：M0 · 前置：第 00 轮全绿 · 需要用户：`twin secrets set deepseek_api_key`

## 先读
`CLAUDE.md`；`docs/SPEC.md` §3（deepseek、pricing、budget、thinking、style_model 配置）、§6 LLM 层全部、§24 R-PRIV-002；DeepSeek 官方文档（实现前用浏览器/抓取确认最新内容并在代码注释中写明核对日期）：
- 模型与价格：https://api-docs.deepseek.com/quick_start/pricing
- 思考模式：https://api-docs.deepseek.com/guides/thinking_mode
- 上下文缓存：https://api-docs.deepseek.com/guides/kv_cache
- 看图：https://api-docs.deepseek.com/guides/vision

若官方文档与 SPEC 不一致（模型名、参数名、价格），以官方为准，先告诉我差异再改 SPEC 与实现。

## 本轮目标
所有与模型打交道的代码集中在 `twin.llm`：调用、重试、熔断、思考开关、JSON 输出、看图、费用记账、高峰判定、预算降级、脱敏、token 估算、风格模型客户端，以及 M0 的 DeepSeek 实测探针。

## 必须实现的需求
R-LLM-001～014、R-ARCH-003（注册生产版非高峰策略）、R-PRIV-002（本轮负责的发送前脱敏部分）、R-OPS-005（`cost_ledger` 写入与汇总查询接口；报告命令在第 12 轮）。

## 详细要求

### A. DeepSeekClient（R-LLM-001～005）
1. 基于 `openai.AsyncOpenAI(base_url=..., api_key=从 keyring 读)`；`chat()` 参数与返回见 SPEC；`purpose` 为必填枚举。
2. 思考开关（R-LLM-002）：`extra_body={"thinking": {"type": "enabled"|"disabled"}}`，可选 `reasoning_effort`；开启时不发送 `temperature`、`presence_penalty`、`frequency_penalty`（并在调用方传入时记录一次警告）。返回值中单独携带 `reasoning_content`；任何组装对话历史的代码都不得把它放回上下文（写一个测试：用历史消息再次调用时请求体里不包含推理内容）。
3. JSON 输出（R-LLM-003）：`chat_json(messages, schema: type[BaseModel], ...)`；失败时把校验错误附回去重试一次。
4. 看图（R-LLM-004）：`ImageInput` 支持 bytes/路径/`MediaStore` 引用；按文件头识别 JPEG/PNG/GIF/WebP；构造 `{"type":"image_url","image_url":{"url":"data:<mime>;base64,...","detail":...}}`，只允许放在 user 消息（system/assistant 中出现即抛异常）；超过官方尺寸/大小限制时先本地缩放（Pillow），GIF 保持原样不转码（除非超限）。
5. 可靠性（R-LLM-005）：区分可重试（429、5xx、连接错误、超时）与不可重试（400、401、402 余额不足、403）；指数退避 + 抖动，最多 4 次；`asyncio.Semaphore(max_concurrency)`；熔断器（连续 10 次失败 → 打开 5 分钟 → 半开探测）。余额不足与鉴权失败立即告警。
6. 超时：非思考 60 秒、思考 180 秒（配置）。

### B. 费用与高峰（R-LLM-006/007）
1. `Pricing`：从配置读取价格表；`cost(usage, model, at)` 计算 cache_hit/miss/output 三部分，非高峰乘以 `pricing.offpeak_multiplier`（默认 0.5；为 1.0 时 `OffPeakPolicy` 不再让离线任务等待）。
2. `PeakCalendar.is_peak(at_utc)`：把时刻换算到北京日期，用 `chinese_calendar.is_workday()` 判断是否工作日（含调休上班的周末、排除法定节假日），工作日的 UTC 01:00–04:00 与 06:00–10:00 为高峰；合并配置 `pricing.extra_offpeak_dates`/`extra_peak_dates`。`chinese-calendar` 对未收录年份抛异常时：回退为"周一至周五为工作日"+ 配置覆盖，写一次告警，并把该情况加入 `twin doctor` 检查项（检查当前与下一年是否覆盖）。`next_offpeak_window(now)` 返回下一个非高峰区间。
3. 实现生产版 `OffPeakPolicy` 并注册到第 00 轮的任务队列（替换测试注入实现）。
4. 每次调用写 `cost_ledger`（模型、purpose、各类 token、费用、是否高峰、耗时、是否思考、请求 id）。提供按日/月/purpose 汇总与缓存命中率查询接口。

### C. 预算与降级（R-LLM-008）
1. `BudgetManager`：读取日/月预算；根据当日与当月已花费（不含 `one_time` 账目）计算降级级别 0–4；级别变化时发事件并写告警（80% 提醒一次，进入每个级别各提醒一次）。第 ④ 级只有在"已激活的风格模型通过上线门槛且健康"时才切到风格后端（通过注入的 `StyleBackendStatus` 接口查询；第 14 轮提供实现，此前恒为"不可用"），否则保持最小上下文的非思考 DeepSeek。
2. 对外接口：`current_level()`、`allow(purpose)`（例如级别 3 时 `proactive` 被拒）、`limits()`（例子数、记忆预算、是否允许思考）；回复路径 `reply` 永远被允许（R-LLM-008 最后一句）。
3. 日期边界按 `bot_timezone` 的当地日期（通过注入的 `TimeService` 接口获取；第 08 轮提供完整实现，本轮使用 `Clock` + 配置时区的实现，并在 TRACEABILITY 注明）。

### C2. 一次性任务预算（R-LLM-014）
1. `OneTimeBatch`：`estimate()`（按 token 估算器、图片实测 token 数与价格表给出费用）、`approve()`（`twin jobs approve <批次>`，记录批准时间与金额）、执行中累计实际费用；实际超过估算 20% 时暂停该批次剩余任务并告警；单批上限 `budget.one_time_usd`，超出的估算需要拆批。
2. `cost_ledger` 增加 `account ∈ {daily, one_time}` 与 `batch_id`；一次性批任务的调用全部记入 `one_time`；`/费用` 与 `twin cost report` 分开显示。
3. 本轮提供通用机制与测试；具体批任务（记忆回放、图片描述回填、表情包打标签、人设卡生成、规划合成、评估生成）由各自轮次接入。

### D. 脱敏（R-LLM-009）
1. `redact(text) -> RedactionResult(text, spans)`：规则覆盖 SPEC 列出的全部类型；银行卡做 Luhn 校验、身份证做校验位校验以减少误伤；地址启发式要有误报控制（单独"路""号"不算）。
2. `ConsistentRedactor`：同一会话/数据集中同一实体映射为一致占位符（如 `[手机号#1]`），供训练集导出使用（R-TRN-007）。
3. 用 hypothesis 生成随机文本 + 插入随机 PII，断言全部被替换且非 PII 文本不变。
4. 第 00 轮日志的脱敏接口改为调用本模块。

### E. 缓存布局与 token 估算（R-LLM-010/012）
1. `PromptLayout`：把消息分为 `stable_prefix` 与 `variable_tail` 两段组装，并记录 `stable_prefix` 的哈希；每次调用后比较 `prompt_cache_hit_tokens` 与 `stable_prefix` 估算长度，统计命中率。
2. `TokenEstimator`：按汉字、ASCII 单词、标点、图片（每张按 1024 上限）估算；用实际 usage 做指数滑动平均校准，校准系数持久化到 `settings`。

### F. 风格模型客户端（R-LLM-011）
1. 接口 `StyleModelClient.generate(prompt: RenderedPrompt, params) -> StyleOutput`，`health()`。
2. `LlamaCppCompletionClient`：POST `/completion`，发送完整提示词字符串、`n_predict`、`temperature`、`top_p`、`stop`（至少含 `<|im_end|>`），解析 `content`；`/health` 健康检查。
3. `VllmCompletionClient`：面向 vLLM 的 `/v1/completions`，`prompt` 为完整渲染字符串、`model` 为 LoRA 名称、`stop` 含 `<|im_end|>`；**不使用** `/v1/chat/completions` 与任何服务端聊天模板（训练/推理逐 token 一致，R-TRN-011）。
4. 两个客户端都提供 `tokenize(text) -> list[int]`（llama-server 与 vLLM 都有 `/tokenize` 接口，以官方文档为准），供第 14 轮启用前的分词核对。
5. 测试用 `tests/support/` 下的本地假服务（例如用 `aiohttp`/`starlette` 起在随机端口，返回与真实接口相同结构的 JSON），覆盖超时、5xx、健康检查失败。
6. `RenderedPrompt` 的生成器（模板渲染器）在第 09 轮实现；本轮定义数据结构与接口。

### G. M0 探针（R-LLM-013）
`twin llm probe`（需要真实 Key，标记 live）：依次执行 SPEC R-LLM-013 列出的 7 项检查（看图用程序生成的合成 JPEG、PNG 与多帧 GIF，不用任何真实照片），把结果、耗时、费用写入 `docs/LLM_REPORT.md`（只含技术结果，不含对话内容），同时把每项的通过/失败与数值以结构化形式写入数据库（第 09b 轮的 M0 门槛判定读取）。根据结果：`detail` 不被接受则 `ImageInput` 不再发送该参数；用实测的每图 token 数校准费用估算；GIF 不被接受则看图前取首帧转 PNG；思考开启时 JSON 不可靠则在报告中写明并停下告诉我（主动消息规划依赖它）。

## 测试要求
- respx 拦截全部 HTTP；覆盖思考开/关请求体差异、温度参数被省略、`reasoning_content` 不回填、JSON 重试、看图消息位置校验、各类错误的重试/不重试、熔断状态机。
- 价格与高峰：工作日/周末/中国节假日（例如 2026-10-01）/调休上班的周末/UTC 边界时刻；非高峰倍率（0.5 与 1.0）；日历库未收录年份的回退与告警（用补丁模拟库抛异常）。
- 一次性批任务：估算、批准、超支 20% 暂停、不计入日/月降级。
- 预算：构造花费序列验证级别变化、事件与告警次数；`reply` 永远允许；第 ④ 级在风格模型"未通过门槛/不健康/通过且健康"三种状态下的行为。
- 脱敏属性测试。
- 风格模型客户端对假服务的全部分支。

## 验收
```
uv run pytest -q
uv run twin llm probe        # 需要已设置 Key；生成 docs/LLM_REPORT.md
uv run python scripts/trace_check.py --round 01
```
把 `docs/LLM_REPORT.md` 的结论贴进汇报。

## 不要做
- 不要在本轮写任何聊天提示词或人设内容。
- 不要把 Key 写进任何文件、日志或测试快照。

## 完成后汇报
按 CLAUDE.md 格式；额外列出与官方文档核对的结果（模型名、参数、价格、限制）。
