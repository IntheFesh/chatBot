# 第 03 轮：聊天记录导入（全量 + 增量、流式、可续传、媒体与表情包、图片描述、导入报告）

> 里程碑：M1 · 前置：第 00–02 轮全绿 · 需要用户：在配置里填写本机全量导出目录 `paths.export_dir`（不要放进仓库），首次导入时确认目标会话。

## 先读
`CLAUDE.md`；`docs/SPEC.md` §0 参考数据、§7 导入全部、R-STO-002/004/006/007、R-LLM-004/007/009、R-PRIV-001。

## 本轮目标
把用户本地的全部导出（格式与样本相同，可能很大）可靠地导入加密数据库：只导入目标会话，流式解析，可续传，幂等增量，媒体加密入库，表情包下载入库，生成图片描述任务，输出不含正文的导入报告，并为后续轮次提供"导入后钩子"。

## 必须实现的需求
R-IMP-001～014、R-ARCH-006（导入入队与 `twin import status`）、R-STO-006（本轮的表：`conversations`、`messages`、`media_assets`、`stickers`、`sticker_uses`、`import_runs`）、R-STO-007（`messages` 与将来 `bot_turns` 的物理隔离）、R-SCOPE-002、R-SAFE-006（把第 02 轮的通道媒体白名单接到 `stickers` 表）、R-LLM-014（首次批量图片描述按一次性批任务处理）。

## 详细要求

### A0. 结构探查（R-IMP-014，先做，做完给我看）
1. `twin import inspect <目录>`：遍历导出目录，只输出结构——文件树（会话目录名打码为"序号_昵称长度_哈希前 4 位"）、`report.json`/`meta.json`/`messages.json` 的键名、值类型、出现次数，`renderType` 与 `type` 的取值计数，`offlineMedia[].kind` 取值计数，`_integrity/` 的文件清单与 JSON 键名；**不输出任何值**（有测试：对合成数据跑一遍，断言报告中不含任何消息文本与 wxid）。结果写 `data/reports/inspect-<UTC>.md`。
2. 我在真实导出上运行后把报告贴给你；你据此核对 SPEC R-IMP-002 的字段表、A 中生成器的结构和 `_integrity` 的解析方式，有出入先告诉我再继续。

### A. 合成数据生成器（测试与基准用）
1. `tests/fixtures/synth_export.py`：生成与真实导出**同结构**的目录：`report.json`（含 `missingMedia`、`errors`）、`conversations/<序号_含中文与特殊符号的昵称_wxid_hash>/{meta.json,messages.json}`、`media/{images,emojis,avatars}`、可选 `_integrity/`。
2. 覆盖全部 `renderType`：text、emoji（带 `emojiMd5`/`emojiUrl`）、quote（`quoteContent/quoteTitle/quoteType/...`）、image（`offlineMedia` 有的有、有的缺）、voice（有/无转写）、voip（"通话时长 mm:ss"、"对方已取消"、"已拒绝"、"未应答"、"已在其它设备接听"）、system（撤回、拍一拍）、transfer、redPacket、link、video、file、location、chathistory、以及一个未知类型；字段名与 SPEC R-IMP-002 列表逐一一致。
3. 可生成：一个目标单聊 + 若干其他单聊 + 一个群聊；可配置消息数（用于 100 万条性能基准）；内容全部为随机生成的中文短句，不得包含任何真实数据。

### B. 模型与解析（R-IMP-001/002）
1. 为 `report.json`、`meta.json`、`messages.json` 顶层与消息体写 pydantic 模型（`extra="allow"` 保存未知字段）；`schemaVersion != 1` 抛 `UnsupportedSchema`。
2. `_integrity/` 若存在：读取其中的完整性信息（先检查实际内容格式并在代码与文档中描述），校验失败的文件报告出来且不导入。
3. 路径处理：`pathlib`、Unicode 规范化（NFC）、Windows 长路径前缀处理。

### C. 目标会话（R-IMP-003）
- 只读取每个会话的 `meta.json` 来列出非群聊会话（显示昵称与消息数，wxid 打码）；首次导入且未配置 `target.username` 时在 CLI 中让我选择并写入运行时设置；其他会话的 `messages.json` 一律不打开。

### D. 流式导入（R-IMP-004～006）
1. `ijson` 按 `messages.item` 迭代，2,000 条一批事务写入；进度条（rich）。
2. 幂等 upsert：主键用导出的 `id`（另存 `serverId`、`localId`、`sortSeq`）；同 id 内容不同 → 以 `exportedAt` 较新者为准，计数"冲突更新"。
3. `import_runs`：记录导出根目录指纹（`exportId`、文件大小、修改时间）、状态、阶段、已处理条数、速度、最后 `sortSeq`、各钩子进度；中断（Ctrl+C、断电）后续跑不重复、不遗漏（有测试：中途杀进程后续跑，最终结果与一次性导入完全一致）。
4. 进程模型（R-ARCH-006）：`twin import <目录>` 是重任务——默认入队（`import` 任务）后立即返回并提示用 `twin import status` 查看；应用未运行时可 `--foreground` 在前台执行；`--resume` 续跑上次未完成的导入；`twin import status [--watch]` 显示阶段、已处理条数/总数、速度、预计剩余时间与各导入后钩子的状态。导入任务执行中定期写进度，查看命令只读。
5. 存储：`messages` 表列包括 `id, conversation_id, create_time_utc, sort_seq, is_sent, kind, render_type, text(EncryptedText), raw(EncryptedJSON), sticker_md5, media_sha256, quote(EncryptedJSON), call_status, call_duration_s, has_transcript, source_export_id`；索引 `(conversation_id, create_time_utc)`、`(is_sent, kind)`。

### E. 归一化与事件文字（R-IMP-007/009）
- 实现 SPEC 的 kind 映射与事件文字模板（放在一个可配置的模板表中，第 09 轮提示词与第 13 轮训练集共用同一函数 `render_event_text(message)`）。
- 同一模板表还要生成"事件文字/媒体占位检测器" `EventTextDetector`（R-SAFE-006）：由模板自动推导出匹配规则（不是另写一份正则），判断一行文字是不是事件文字；第 05、09、09b、13 轮复用。有测试：每种模板渲染出的文字都能被检测到，普通聊天文字不会被误判（用 hypothesis 生成普通中文短句）。
- `is_reproducible(message)`：文字、表情包、表情代码、引用为可复现；图片、语音、视频、通话、转账、红包、位置、文件、链接、聊天记录、系统事件为不可复现（训练目标与盲测过滤用）。
- 语音：有转写 → `[语音 N 秒：转写]`；无 → `[语音 N 秒，未转写]`；`voiceLength` 为毫秒字符串，需容错。
- 视频：封面若在 `offlineMedia` 中则入库；文件：只记文件名与大小。

### F. 媒体与表情包（R-IMP-008）
1. `offlineMedia[].path` 相对导出根目录；存在则流式加密进 `MediaStore` 并在 `media_assets` 记录（kind、原 md5、sha256、mime、尺寸）；不存在或在 `missingMedia` 中 → 记录缺失。
2. 表情包：对每个不同的 `emojiMd5` 建 `stickers` 行（建表后把第 02 轮的 `OutboundMediaPolicy` 接到 `stickers`：只有 `available` 的表情包文件 sha256 可以发出）（她用过 / 用户用过 / 两者），`sticker_uses` 记录每次使用（谁、何时、消息 id）。本地 `media/emojis/` 已有的文件直接入库；其余创建下载任务（任务队列，类型 `sticker_download`）：httpx，并发 4、每秒 ≤ 4 次、重试 3 次、超时 30 秒；按文件头识别格式；校验 md5（不符标记 `md5_mismatch` 但保留）；失败标记 `unavailable` 与原因；可续跑；回填命令 `twin stickers download [--retry-failed]`（重任务）。
3. 头像：只导入目标会话双方头像（加密）。

### G. 图片描述（R-IMP-012）
- 类型 `image_caption` 的离线任务（`offpeak_only=True`）：只为最近 90 天的图片排队，作为一次性批任务（R-LLM-014：先估算费用，我 `twin jobs approve` 后执行）；调用 DeepSeek 看图（R-LLM-004，`purpose=caption`），提示词要求一句客观中文描述、不猜测身份；**描述结果先经 `redact()` 脱敏**再加密存 `media_assets.caption`。
- `get_caption(media, *, wait: bool)`：`wait=True`（用户刚发来的图片，在线路径，带超时）时同步生成并缓存；`wait=False`（检索例子里的历史图片）时有描述就返回，没有就返回 `None` 并排队一个描述任务——在线回复路径从不等待历史图片描述。
- 回填命令 `twin images caption-backfill [--days N]`（重任务）。

### H. 导入报告与钩子（R-IMP-010/011）
1. 报告（不含正文、不含 wxid 明文）：按 kind × 发送方计数、日期范围（按 `source_timezone` 当地日期）、新增/重复/冲突、缺失媒体、表情包下载统计、图片描述排队数；保存 `data/reports/import-<UTC>.md` 并在终端显示。
2. `PostImportHooks` 注册表：每个后续轮次把自己的"导入后动作"注册进来（画像、作息、人设卡、表情包打标签、检索库、记忆回放、重训检查），注册时必须同时给出对应的回填命令名（注册 API 强制要求该参数，测试检查命令存在）。本轮注册：图片描述排队、表情包下载排队。导入完成后按注册顺序执行并在报告中列出各钩子结果。

### I. 性能（R-IMP-013）
- `scripts/bench_import.py`：用生成器造 100 万条目标会话消息，计时并输出每秒条数与峰值内存（`tracemalloc` + 进程 RSS）；目标 < 30 分钟、峰值 < 500MB。结果写进汇报。

## 测试要求
- 生成器产物通过模型校验；每种 renderType 的归一化与事件文字。
- 目标会话选择：只打开目标会话的 `messages.json`（用打开文件的计数/补丁断言）。
- 幂等：同一数据导入两次结果不变；修改后的同 id 消息按 `exportedAt` 处理。
- 续传：在第 N 批之后模拟崩溃，续跑后与一次性导入逐行一致。
- 媒体：存在/缺失/`missingMedia`；表情包下载成功、md5 不符、404、超时与续跑（respx）。
- 图片描述任务：只排最近 90 天、非高峰执行、按需同步生成。
- `is_sent` 与发送方识别正确；群聊与其他会话被忽略。
- 报告中不出现任何消息正文（扫描报告文件断言）。
- `inspect` 不输出任何值；`EventTextDetector` 与 `is_reproducible`；图片描述经过脱敏、`wait=False` 不阻塞。
- 导入入队后立即返回、`import status` 显示进度、应用运行时导入由应用执行、未运行时 `--foreground` 执行。

## 验收
```
uv run pytest -q
uv run twin import inspect "<我的导出目录>"   # 我执行并把报告给你核对（报告不含任何值）
uv run python scripts/bench_import.py --messages 1000000
uv run twin import "<我的导出目录>" --foreground   # 我执行，首次会让我选目标会话
uv run twin import status
uv run twin import "<我的导出目录>" --resume
uv run python scripts/trace_check.py --round 03
```

## 不要做
- 不要把任何真实导出复制进仓库或测试目录。
- 不要在本轮做画像/检索/记忆计算（它们由后续轮次通过钩子接入）。
- 不要在本地做语音识别。

## 完成后汇报
按 CLAUDE.md 格式；附基准测试结果；若我已用真实数据导入，附导入报告的统计部分（不含正文）。
