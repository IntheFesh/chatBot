# LLM 层实测报告（M0 · DeepSeek）

<!-- llm-report: pending -->

> 状态：**待实测**。
> 编写本轮代码的环境没有 DeepSeek API Key，`twin llm probe` 还没有对真实接口运行过，所以本文件
> 里没有任何实测数字。运行探针后，本文件会被实测结果整体覆盖。

## 怎样得到实测结果

```
uv run twin secrets set deepseek_api_key     # 粘贴 Key
uv run twin llm probe                        # 约几十个很小的请求，费用远低于 0.1 美元
```

探针把结果写入 `docs/LLM_REPORT.md`，同时以结构化形式存入数据库（`settings` 表，键 `m0.llm_probe`，
`schema_version` 1），第 09b 轮的 M0 判定读取它；探针学到的能力（`detail` 是否被接受、GIF 是否被
接受、思考开启时 JSON 是否可靠、各尺寸图片的计费 token）存入键 `llm.capabilities`，随后改变客户端
的行为。探针不发送任何对话内容，图片都是程序画出来的（红色圆形）。

## 七项检查与 M0 判定口径

| # | 标识 | 内容 | 计入 M0 |
| --- | --- | --- | --- |
| 1 | `thinking_toggle` | 思考开关：开启与关闭都成功；开启时返回 reasoning_content，关闭时不返回 | 是 |
| 2 | `cache_hit` | 上下文缓存：同一长前缀重复请求，prompt_cache_hit_tokens > 0 | 是 |
| 3 | `vision` | 看图：程序生成的 JPEG、PNG、多帧 GIF 各一张（GIF 只记录结果） | 是 |
| 4 | `json_output` | JSON 输出：思考关闭与开启两种情况下都能被解析 | 是 |
| 5 | `detail_param` | detail 参数：带与不带各请求一次，是否被接受 | 否（测量项） |
| 6 | `image_tokens` | 每张图的实际计费 token 数：不同尺寸各一张，由 usage 差值推算 | 否（测量项） |
| 7 | `latency_cost` | 各请求的耗时与费用 | 否（测量项） |

M0 通过 = 第 1、2、4 项成功，且第 3 项中 JPEG 与 PNG 成功（GIF 的结果只记录，不被接受时看图前取首帧
转 PNG）。第 5、6、7 项是测量项，只要求记录。

如果第 4 项显示思考开启时 JSON 不可靠，探针会在报告里写明并以非零退出码结束：主动消息规划依赖
这一点，需要先停下来商量。

## 官方文档核对（2026-10-09，非实测）

核对的页面：

- pricing: https://api-docs.deepseek.com/quick_start/pricing
- thinking_mode: https://api-docs.deepseek.com/guides/thinking_mode
- kv_cache: https://api-docs.deepseek.com/guides/kv_cache
- vision: https://api-docs.deepseek.com/guides/vision
- json_output: https://api-docs.deepseek.com/guides/json_mode
- error_codes: https://api-docs.deepseek.com/quick_start/error_codes
- rate_limit: https://api-docs.deepseek.com/quick_start/rate_limit

| 项目 | 官方文档的说法 | 实现中的处理 |
| --- | --- | --- |
| 模型名 | `deepseek-flash`（支持看图）、`deepseek-v4-pro`（不支持看图）；旧名 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 仍可用并按 Flash 计价 | 与 SPEC 一致；旧名按 Flash 计价；`vision_model` 配置为不能看图的模型时拒绝启动 |
| 思考开关 | 默认开启；`extra_body={"thinking": {"type": "enabled"|"disabled"}}`；`reasoning_effort` 取 low/high/max；开启时 `temperature`、`presence_penalty`、`frequency_penalty` 不报错但无效；没有 `tools` 参数时历史回合的 `reasoning_content` 会被 API 忽略 | 两种状态都显式发送；思考时不发送上述三个参数并记录一次警告；任何历史都不带 `reasoning_content` |
| 价格（美元/百万 token，高峰） | Flash：缓存命中 0.006、未命中 0.3、输出 1.2；Pro：0.044、1.32、3.96；非高峰为一半 | 与 SPEC R-CFG-004 的价格表逐项一致（有测试） |
| 高峰时段 | 周一至周五（不含中国法定节假日）UTC 01:00–04:00、06:00–10:00（北京时间 09:00–12:00、14:00–18:00）；周末与法定节假日全天为空闲时段 | 用 `chinese_calendar.is_workday()` 判定（含调休上班的周末，见 DECISIONS.md D-004） |
| 缓存字段 | usage 中有 `prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`；命中要求前缀已被持久化，构建需要数秒，尽力而为 | 逐次记账；探针对同一请求等待后重试 |
| 看图限制 | 仅 user 消息；JPEG/PNG/GIF/WebP 按文件头识别；`detail` 取 low/high/original/auto；每张图最多 1024 token；单图 32 MiB，请求体 48 MiB，每边 8192 px（15 张以上 4096 px），每次最多 600 张 | 超限时用 Pillow 本地缩放；system/assistant 消息里放图直接抛异常 |
| JSON 输出 | `response_format={"type": "json_object"}`；提示词里要有 "json"；要设置 `max_tokens`；偶尔返回空内容 | 自动补充指令；空内容当作无效并重试一次 |
| 错误码 | 400/422 请求有误，401 鉴权，402 余额不足，429 限速，500/503 服务端 | 429/5xx/超时/连接错误重试；其余不重试，401/402/403 立即告警 |

