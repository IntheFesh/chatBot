# 第 05 轮：向量模型与真实片段检索库

> 里程碑：M1 · 前置：第 00–04 轮全绿

## 先读
`CLAUDE.md`；`docs/SPEC.md` §12 检索库全部、R-STO-005/007、R-LLM-009、R-TRN-006（留出集与测试集必须一致）。

## 本轮目标
建立只包含她真实回复的"类似情况下她怎么回"检索库：本地向量模型（Windows CPU/GPU 都能跑）、LanceDB 只存向量、按时间留出 10% 作评估与训练测试集、MMR 去冗余、增量更新；同时提供通用的文本向量服务供记忆与表情包使用。

## 必须实现的需求
R-RET-001～006、R-STO-005、R-NFR-003（向量模型懒加载）、R-STO-006（本轮的表：`example_windows`）、R-IMP-011（注册检索库增量更新钩子与回填命令）、R-TRN-013（检索的 as-of 过滤）、R-SAFE-006（例子中事件文字的注记呈现）。

## 详细要求

### A. 向量服务 `twin.retrieval.embedder`
1. `sentence-transformers` 加载 `retrieval.model`（默认 `BAAI/bge-small-zh-v1.5`，可配 `BAAI/bge-m3`）；`device: auto` 时有 CUDA 用 GPU 否则 CPU；首次运行下载到 `data/models/embeddings/` 并记录版本与哈希；懒加载（启动不阻塞，R-NFR-003）。
2. 批量编码、归一化；`encode(texts, kind)`；向量维度与模型名写入索引元数据，模型变更时拒绝混用并提示重建。
3. 编码前一律脱敏（R-LLM-009，同一函数），因为向量本身也可能泄露信息。

### B. 例子窗口（R-RET-001）
1. 以第 04 轮定义的"合并块"和"会话段"（`profile.burst_gap_s`、`profile.segment_gap_min`）为准：对她的每个回复块，取同一会话段内前面最多 6 个合并轮次为上下文。
2. `example_windows` 存 `(id, reply_block_ids, context_block_ids, reply_at_utc, local_slot, day_type, holdout(bool), embed_version)`；不存正文（正文按 id 从 `messages` 解密取）。
3. 上下文编码文本：最后两轮权重更高（重复一次或加权拼接，选择一种并在文档说明理由）；包含事件文字（`render_event_text`）。

### C. 留出集（R-RET-003）
- 调用第 04 轮的 `holdout_cutoff()`（唯一实现，不在本轮另算）：`reply_at_utc ≥ 切分点` 的窗口 `holdout=True`，不写入向量索引；第 09b 轮评估与第 13 轮训练集测试集使用同一切分点。
- 增量导入后切分点不自动移动（避免评估集变化）；提供 `twin retrieval resplit`（重任务）显式重切：调用第 04 轮的重算函数、重建索引中受影响的窗口，并排队重算所有 pre_holdout 派生数据，提示会影响历史评估可比性。

### D. 守护机器人回复不入库（R-RET-004、R-STO-007）
- 构建函数签名只接受 `messages` 的 ORM 实体或其 id，且断言 `is_sent=False`；从 `bot_turns`（第 09 轮建表）来的任何对象类型不兼容；写一个测试：伪造 bot 回复对象传入时抛类型错误/断言失败。

### E. 向量索引（R-STO-005）
- LanceDB 表只存 `window_id`、向量、`reply_at_utc`、`local_slot`、`day_type`；不存任何文本；数据目录在 `data/vectors/`。
- 同一套索引封装也供记忆（第 07 轮）与表情包描述（第 06 轮）使用，各自独立的表。

### F. 查询（R-RET-005）
1. 输入：当前对话最近几轮（机器人会话中的用户消息与机器人回复都可作为查询上下文，但结果只来自她的真实窗口）、当前当地钟点与日类型、可选 `before: datetime`（只返回 `reply_at_utc < before` 的窗口，供第 09b 轮评估沙盒与第 13 轮的 `AsOfView(t)` 使用，R-TRN-013）。
2. 召回 top-50 → 计算综合分：向量相似度 + 时段相近加分（环形距离）+ 新近度加分（指数衰减，半衰期可配）→ MMR（λ=0.7）选出 k 个（默认 8，受 R-LLM-008 降级级别限制）。
3. 过滤：与其他已选例子回复文本高度相似的（字符级相似度 > 0.9）去重；回复为纯事件（只有通话/转账等）的窗口降权。
4. 返回结构化例子：上下文轮次（谁说的、文本）与她的真实回复块（多行），供第 09 轮渲染。回复块中每行标注 `reproducible`（第 03 轮 `is_reproducible`）：不可复现的行（图片、语音、通话、转账等）在渲染时以"（此处她发了：<事件文字>）"的上下文注记呈现，不作为可模仿的回复行（R-SAFE-006）；例子中的图片用 `get_caption(wait=False)`，没有描述时显示 `[图片]` 并已自动排队描述任务（R-IMP-012）。
5. 例子渲染函数 `render_example()` 放在本轮，第 09 轮提示词与第 09b 轮评估复用。

### G. 增量与重建（R-RET-006）
- 导入后钩子：只为新增的非留出窗口编码；`twin retrieval rebuild` 全量重建（模型/参数变化时）；`twin retrieval stats`；编码任务进任务队列，可续跑，显示进度与预计时间。
- CPU 性能：记录每千条编码耗时；对 bge-m3 在 CPU 上过慢的情况给出提示。

## 测试要求
- 窗口构建：合成对话中上下文边界（会话段、6 轮上限）正确。
- 留出集：切分点稳定、索引中不含留出窗口、共享函数被第 13/15 轮复用（本轮先测函数本身）。
- 机器人回复无法入库。
- 查询：构造语义明显的合成数据（例如"吃饭"相关上下文），检索结果包含相应窗口；MMR 减少重复；时段加分生效。
- 增量：二次导入只编码新增；模型名变更时拒绝混用。
- `before` 过滤：不返回 `reply_at_utc ≥ before` 的窗口。
- 例子渲染：不可复现的行变成上下文注记；无描述的历史图片不阻塞并排队描述。
- 测试中的向量模型：用一个极小的、离线可用的嵌入实现注入（放 `tests/`），生产代码路径不包含假模型。

## 验收
```
uv run pytest -q
uv run twin retrieval rebuild      # 回填：对已导入的真实数据建库，观察耗时
uv run twin retrieval stats
uv run python scripts/trace_check.py --round 05
```

## 不要做
- 不要把机器人会话、她的照片或任何原文写入向量库元数据。

## 完成后汇报
按 CLAUDE.md 格式；附真实数据上的窗口数、留出数、编码耗时（若已运行）。
