# 性能记录（PERFORMANCE）

> 每一节只写**真的跑过**的数字：机器、条数、命令、结果。没有测到的部分如实写"没有测"。新增轮次的基准追加为新的小节。

## 1. 聊天记录导入（R-IMP-013，第 03 轮）

**测的是什么**：`scripts/bench_import.py` 用 `tests/fixtures/synth_export.py` 生成合成导出（一个目标会话，消息类型按默认比例：文字约 62%、表情包 12%、其余类型若干；不含媒体文件，所以不含图片入库、表情包下载与模型调用——和 R-IMP-013 "结构化导入（不含下载与 LLM 任务）"一致），再在**一个全新的子进程**里完成完整的导入运行：`messages → media → stickers → finalize → hooks → report`，每条消息都过 pydantic 校验、归一化、AES-256-GCM 加密 `text` / `raw` 并写入有两个二级索引的 SQLite 表（WAL，每批 2,000 条一个事务）。数据库是全新的。

**机器**（云端沙箱，不是用户的 Windows 电脑）：Intel Xeon @ 2.80 GHz，4 个逻辑核（导入是单线程），16 GB 内存，Linux，Python 3.12（uv）、SQLite 3.53、`ijson` 3.5.1 的 `yajl2_c` 后端，本地临时目录所在的普通磁盘。

### 实测结果

| 条数 | 命令 | 导入耗时 | 速度 | 峰值内存（RSS） | 数据库大小 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 20,000 | `python scripts/bench_import.py --messages 20000` | 2.9 秒 | 6,922 条/秒 | 111 MB | 18 MB | 冒烟 |
| 200,000 | `python scripts/bench_import.py --messages 200000` | 25.1 秒 | 7,978 条/秒 | 120 MB | 180 MB | 推算 100 万条约 2.1 分钟，于是做了下一行 |
| **1,000,000** | `python scripts/bench_import.py --messages 1000000` | **141.9 秒（2.4 分钟）** | **7,049 条/秒** | **155 MB**（子进程峰值，含最后的报告阶段） | 906 MB（数据库文件与 WAL 合计） | `messages.json` 363.6 MB；生成数据另用 14.8 秒 |
| 200,000 | `python scripts/bench_import.py --messages 200000 --tracemalloc` | 119.7 秒 | 1,671 条/秒 | 155 MB | 180 MB | **只看内存，不看速度**（`tracemalloc` 让导入慢约 5 倍）；Python 自己分配的峰值 **31.9 MB** |

100 万条的逐段速度（条/秒）：7,639 → 7,892 → 7,625 → 7,284 → 7,227 → 7,513 → 7,386 → 7,216 → 6,871 → 6,133（最后一段含索引与 WAL 的增长）；峰值内存在 20 万条之后一直是 146 MB 上下，**不随条数增长**。

### 对照门槛

| 门槛（SPEC R-IMP-013 / R-IMP-004） | 结果 |
| --- | --- |
| 100 万条结构化导入 < 30 分钟 | **通过**：沙箱里实测 2.4 分钟（约为门槛的 8%） |
| 峰值内存 < 500 MB | **通过**：155 MB（`ru_maxrss`）；Python 堆峰值 31.9 MB（`tracemalloc`，20 万条） |
| "10 GB 文件峰值内存 < 500MB" | **没有直接测**：沙箱磁盘放不下 10 GB 文件。依据是实现方式——`ijson` 逐条解析、每批 2,000 条后释放，内存与文件大小无关；实测从 20 万条（72 MB 文件）到 100 万条（364 MB 文件）峰值内存不变（146 MB），这条曲线是平的。 |

### 没有测到的

- **Windows**：没有在 Windows 上跑过。磁盘（NTFS）、杀毒软件实时扫描会明显影响 SQLite 写入速度；即使慢一个数量级（约 700 条/秒）100 万条也只要约 24 分钟，仍在门槛内。用户的实测数字请用 `uv run python scripts/bench_import.py --messages 1000000` 取得后补记在这里（docs/PENDING_USER_ACTIONS.md 第 03 轮第 7 条）。
- **带媒体的导入**：基准不含图片 / 表情包文件的加密入库（它们是流式加密、每个文件一次读写，与消息条数无关，用时取决于磁盘与文件大小）。
- **图片描述与表情包下载**：受网络和 DeepSeek 限制，不属于 R-IMP-013 的范围。
- **真实导出**：消息体比合成数据更大（链接、聊天记录的 XML 等）会让 `raw` 更大、速度略慢；这只是常数倍。

### 复现

```
uv run python scripts/bench_import.py --messages 1000000 [--workdir D:/bench --keep] [--json result.json]
uv run python scripts/bench_import.py --messages 200000 --tracemalloc      # 只看内存
```

## 2. 检索库：向量编码与查询（R-RET-006，第 05 轮）

**测的是什么**：`scripts/bench_retrieval.py` 用 `tests/support/synth_chat.py` 写一段合成对话（随机汉字，时间节奏像真实聊天：她的连发块、对方的回复、作息），然后走和 `twin retrieval rebuild` 完全相同的代码路径建库——`sync_windows` → 用**真实的** sentence-transformers 模型（`BAAI/bge-small-zh-v1.5`，首次运行从 Hugging Face 下载到 `data/models/embeddings/`）把每个窗口的上下文脱敏、编码 → 写 LanceDB——再用 `ExampleRetriever.rank` 做若干次查询（编码问题 → 50 个最近邻 → 打分 → MMR）。窗口里有一部分是留出集（不编码）、一部分是她先开口的窗口（没有上下文，不编码）。

**机器**（同第 1 节的云端沙箱，不是用户的 Windows 电脑）：Intel Xeon 2.80 GHz，4 个逻辑核，16 GB 内存，**只用 CPU**（torch 2.14.1+cpu，`retrieval.device: cpu`），Linux，Python 3.12。

### 实测结果

| 模型 | 合成对话 | 窗口数（编码 / 留出 / 无上下文且未留出） | 模型加载 | 编码耗时（含读消息、写索引） | **每千条编码耗时** | 速度 | 单次查询 | 峰值内存（RSS） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-small-zh-v1.5 | 120 天，9,568 条消息 | 1,800（1,188 / 180 / 432） | 6.7 / 7.4 秒（两次） | 10.6 / 10.8 秒 | **8.9 / 9.1 秒** | 约 111 个窗口/秒 | 平均 50 / 46 毫秒（各 10 次） | 1,202 MB |
| bge-m3 | 20 天，1,593 条消息 | 300（198 / 30 / 72） | 9.9 秒（磁盘缓存热；第一次冷启动 37.6 秒） | 35.9 秒 | **181 秒** | 5.5 个窗口/秒 | 平均 149 毫秒（10 次） | 2,917 MB |

命令：`uv run python scripts/bench_retrieval.py --days 120 --device cpu`；`uv run python scripts/bench_retrieval.py --days 20 --model BAAI/bge-m3 --device cpu`。“编码耗时”是 `run_index` 里编码循环的墙钟时间（每 256 个窗口：从数据库取消息并解密、拼文本、模型编码、写 LanceDB、在 SQLite 里标记），不含窗口表的同步（`sync_windows`，这批数据约 2 秒）。

**关于第一次测量**：我最先跑的一轮（bge-small 12.0 秒/千，bge-m3 346.6 秒/千）是在沙箱里同时跑着测试套件的时候测的，CPU 被抢占，数字偏高约 35%–90%，已作废；上表是 CPU 空闲时的重测（bge-small 重复两次，结果一致）。

- **外推（仅供估计，没有跑这么大的库）**：bge-small 10 万个窗口约 15 分钟；本次合成数据每条消息约产生 0.19 个窗口（1,800 / 9,568），按这个比例 100 万条消息约 19 万个窗口、约 29 分钟（真实对话的比例会不同）。bge-m3 约是 bge-small 的 20 倍（181 / 9.0），10 万个窗口约 5 小时——所以默认不用它，`twin retrieval rebuild` 在 bge-m3 + CPU 时先给出提示。有 NVIDIA 显卡并装了 CUDA 版 torch 时应当明显更快（没有测）。
- 只有约一百个窗口的小库（例如 `TWIN_LIVE=1` 的集成测试用的 14 天对话）每千条的数字会偏高（约 29 秒；推测是模型预热占了总时间的大头），**不能**用它外推。
- 单独测编码（没有窗口构造、数据库与 LanceDB）：1,000 段 30–240 字的随机汉字文本，bge-small 16.6 秒；1,000 段 40 字的文本 3.6 秒（这一组也是在 CPU 被其他任务占用时测的，偏慢，仅说明文字越长越慢）。窗口的编码文本（最近两轮出现两次）通常 100–300 字。
- 查询是在线路径上的开销：这里的毫秒数包含问题的编码、LanceDB 的 50 近邻搜索、SQLite 取窗口与消息、打分与 MMR，没有拆开测；与一次 DeepSeek 调用（秒级）相比可以忽略，且在线程里执行、不阻塞事件循环。
- 内存峰值主要是 torch 与模型本身（含合成数据生成与数据库），与窗口数基本无关；`twin run` 启动时**不**加载模型（R-NFR-003），第一次需要编码时才加载。

### 没有测到的

- **Windows**：没有在 Windows 上跑过。CPU 型号不同数字会差几倍；请在用户电脑上运行 `uv run twin retrieval rebuild --foreground` 后看 `twin retrieval stats` 的 “last index run” 一行（每千个窗口多少秒），或运行上面的 `bench_retrieval.py` 命令（docs/PENDING_USER_ACTIONS.md 第 05 轮第 1 条）。
- **GPU**：沙箱没有 GPU。
- **真实对话**：真实消息的长度分布与合成数据不同（表情、长句、引用）；窗口的文字更长时编码更慢，常数倍。
- **超过 10 万个窗口的查询延迟**：LanceDB 是精确搜索（D-183），线性于窗口数；10 万是实测过的（下面），更大的库没有测。

### 向量库本身（LanceDB，不含模型）

`uv run python scripts/bench_retrieval.py --vectors 100000`：10 万个随机单位向量（512 维）、窗口表的列，按索引器的方式每 256 个一批 `upsert`（含已有 id 的合并）：

| 向量数 | 写入（400 批） | `optimize()` 合并小文件 | 50 近邻搜索（带时间预过滤，5 次） | 
| --- | --- | --- | --- |
| 100,000 | 32.4 秒 | 2.0 秒 | 100–155 毫秒（暖后约 100 毫秒） |

即：向量库的写入时间相对编码（10 万窗口约 15 分钟）可以忽略；10 万窗口时一次精确搜索约 0.1 秒，仍远小于一次 DeepSeek 调用，所以不建近似索引。几十万个窗口以后搜索会线性变慢，那时再考虑 IVF 索引。机器同上。
