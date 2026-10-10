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

## 3. 运行时数字：回复耗时、内存、静止 CPU、启动（R-NFR-001..003，第 16 轮 A）

**测的是什么**：`scripts/bench_runtime.py`（`--only engine|idle|startup`，`--json` 写出数字）。

- **R-NFR-001 的引擎部分**：生产的 `ReplyPipeline`（取材、提示词、**假模型即时回答**、后处理）处理 100 条回复，每条用 `perf_counter` 量墙钟时间，不含刻意延迟。SPEC 的 20 秒是整个生成（模型加引擎）；引擎部分的门槛取十分之一，即 2 秒（D-562）。模型部分只能用真实 API 测，见下面的"没有测到的"。
- **R-NFR-003 启动**：起一个**真实的子进程** `twin run`（console 通道，空安装，系统时钟），从启动到日志出现 `application_running` 的秒数。向量模型懒加载，所以启动时不加载 torch（第 05 轮已有测试）。
- **R-NFR-002 静止**：启动后再等 10 秒，读 30 秒窗口里进程的 CPU 时间占一个核的百分比，以及常驻内存（RSS，不是峰值）。快进时钟下的"每个模拟日的 CPU"不能代替它：快进少了几个数量级的轮询。

**机器**（云端沙箱，不是用户的 Windows 电脑；和第 1、2 节不是同一台虚拟机）：Linux 6.18.44，Intel Xeon 2.10 GHz，4 个逻辑核，Python 3.12.13。

| 项目 | SPEC 门槛 | 沙箱实测（2026-10-10） | Windows 实测 |
| --- | --- | --- | --- |
| 引擎处理一条回复（100 条，假模型）R-NFR-001 | p95 < 2 秒（20 秒的十分之一） | 平均 84 毫秒，p50 78 毫秒，**p95 96 毫秒**，最大 336 毫秒 | 待实测 |
| 启动到可收消息 R-NFR-003 | < 30 秒 | **3.1 秒** | 待实测 |
| 静止 CPU（30 秒窗口）R-NFR-002 | < 2 % | **0.63 %** 个核 | 待实测 |
| 常驻内存（启动后静止）R-NFR-002 | < 1.5 GB | **204 MB** | 待实测 |
| 真实 DeepSeek 一次回复（不思考，20 条）R-NFR-001 | p95 < 20 秒 | 待实测（沙箱没有 Key，也不该有） | 待实测 |
| 真实 DeepSeek 一次回复（思考，5 条）R-NFR-001 | p95 < 60 秒 | 待实测 | 待实测 |

命令：`uv run python scripts/bench_runtime.py --json bench.json`（约 2 分钟）。同一份判定逻辑有单元测试（`tests/unit/test_bench_runtime.py`：门槛是"低于"不是"不高于"；读不到数字的平台不会因此判失败，也不会编数字），真实子进程的一次完整测量在 `tests/integration/test_nfr_runtime.py`。

### 没有测到的

- **真实 DeepSeek 的耗时**：`tests/integration/test_nfr_live.py`（`live` 标记，需要 `TWIN_LIVE=1` 和 `TWIN_LIVE_DEEPSEEK_KEY`，没有就跳过）向真实 API 要 20 条不思考和 5 条思考的回复，用形状像回复请求的**合成**提示词（没有真实聊天），判 p95，并用 `TWIN_LIVE_REPORT=<文件>` 写出 JSON。沙箱里没有 Key，**没有跑过**，上表对应行写"待实测"，不放估计值。耗时取决于网络位置，要在实际运行的那台电脑上测（`docs/PENDING_USER_ACTIONS.md` 第 16 轮 A 第 3 条）。
- **Windows**：没有在 Windows 上跑过。`tests/support/process_metrics.py` 里读常驻内存和 CPU 的 Windows 分支（`GetProcessMemoryInfo`、`GetProcessTimes`，ctypes）在沙箱里**无法执行**，只做了类型检查（`mypy --platform win32`）。杀毒软件的实时扫描可能让启动慢几倍。用户的数字用上面的命令取得后补在表里（`docs/PENDING_USER_ACTIONS.md` 第 16 轮 A 第 1 条）。
- **有 llama-server 子进程时的内存**：SPEC 的 1.5 GB 不含 `llama-server`（它在自己的进程里）；这里只测 `twin run` 本身。
- **有真实向量模型时的内存**：第一次需要编码时才加载 torch 和模型，峰值见第 2 节（bge-small 约 1.2 GB 含合成数据生成）；soak 用的是离线哈希编码器，所以第 4 节的内存**不含**模型。真实运行里 `twin run` 会在需要检索时加载一次 bge-small，常驻内存要加上它。这一项没有在长时间运行里测过。

## 4. 长时间运行：`scripts/soak.py`（R-NFR-002、R-ARCH-004，第 16 轮 A）

**测的是什么**：开发工具 `scripts/soak.py --days 14 --accelerated`。它用**真实的 `twin run` 装配**（`twin.assembly.assemble()`，D-551）、真实的 SQLite、真实的调度器和引擎，只把三样东西换成测试替身：时钟（`LifeClock` 快进，一个模拟日约十几秒真实时间）、DeepSeek（`respx` 上按请求内容回答的确定性替身）、用户（一个固定种子的"人"：有话多的日子、几乎不说话的日子、周末，8 天以上的运行里有 3 天一句话不说）。每天结束测一次：常驻内存、Python 堆、打开的文件、asyncio 任务数、线程数、任务队列（等待多少、最老的多久、失败几个）、数据库大小、ERROR 日志数、丢失的任务异常、`engine_error` 告警；并断言每个场景的承诺（屏幕=`bot_turns`、深睡不说话、平台条数、主动审计、机器人的话没进她的数据）。报告写到标准输出或 `--output`，**不**写 `data/reports/`。判定的口径在 D-558。

### 14 天验收运行（`python scripts/soak.py --days 14 --accelerated`，沙箱，161 秒，种子 1，console 通道）

| 天 | 常驻内存 | Python 堆 | 任务 / 线程 | 等待的任务 | 数据库 | 累计收 / 发 | 主动消息 | ERROR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 380 MB | 249 MB | 27 / 10 | 0 | 5.8 MB | 11 / 15 | 5 | 0 |
| 7 | 412 MB | 280 MB | 27 / 10 | 0 | 6.3 MB | 107 / 88 | 18 | 0 |
| 14 | 434 MB | 302 MB | 27 / 10 | 0 | 7.1 MB | 313 / 196 | 35 | 0 |

- **内存**：启动时 341 MB，14 天后 434 MB，最高 434 MB，远低于 1.5 GB（判定通过）。这 14 天里每天涨约 3–4 MB，是进程的预热（下面），**不判趋势**：报告里写 `not judged: the trend needs 26 days (21 of warm-up), this run has 14`，只判绝对门槛。
- **队列**：每天午夜没有等待的任务（等待批准的一次性任务单独计，0 个）；64 个任务完成，0 个失败。
- **数据库**：5.8 → 7.1 MB，约每天 0.1 MB（上限 10 MB/天）。
- **任务与线程**：第 2 天到第 14 天都是 27 个任务、10 个线程，没有累积。
- **异常**：0 条 ERROR 日志，0 个丢失的任务异常，0 个 `engine_error` 告警。告警只有 4 条 `lifeline_corrected`（生活线被纠正，正常的运营信息）。
- 工作量：收到 297 条、发出 196 个气泡、16 条命令，主动消息 35 条（bedtime 7、greeting 10、meal 13、share 2、edge 2、silence 1）；模型请求：回复 143、记忆 50、主动 35、生活线 28、图片描述 17、表情包打标签 15。

### 28 天趋势（`python scripts/soak.py --days 28 --accelerated`，沙箱，331 秒）

| 天 | 1 | 7 | 14 | 21 | 28 |
| --- | --- | --- | --- | --- | --- |
| 常驻内存 | 381 MB | 414 MB | 431 MB | 450 MB | 457 MB |
| 比前一档涨 | （启动时 341） | +33 | +17 | +19 | +7 |

最后一周（第 22–28 天）的斜率 **+0.77 MB/天**（上限 3 MB/天），判定通过；数据库 5.8 → 8.5 MB；任务 27、线程 10 不变；0 个错误。即：内存的增长是预热（缓存、分配器、懒加载的模块和表），三周后趋于平；只靠 14 天的运行看不出这条曲线，所以验收的 14 天运行不判趋势，趋势由这次 28 天运行给出。

- **补充一次探针**（一次性脚本，没有留在仓库里）：同样的世界 30 天无人说话，每周内存 +30、+15、+6、+2 MB，最终 414 MB。没有人说话时也在涨，说明预热与对话量无关。
- **这轮发现并修掉的一个真增长源**（D-557）：LanceDB 每次写入留下一个小数据文件，永不合并；用单独的脚本把同一张表 `merge_insert` 400 次，常驻内存 219 MB，修后（写入层看到 16 个小文件就 `optimize()`）134 MB。修的是真实的增长，但没有让 14 天曲线变平——那条曲线是预热。
- **没有测到的**：Windows（`docs/PENDING_USER_ACTIONS.md` 第 16 轮 A 第 2 条）；真实时钟下连续跑 14 天（快进时钟跳过了真实时间里的休眠、网络超时与重连）；真实向量模型与真实 `llama-server`（见第 3 节）；比 28 天更长的运行。

### 复现

```
uv run python scripts/soak.py --days 14 --accelerated [--output soak.txt] [--json soak.json] [--seed 1]
uv run python scripts/soak.py --days 28 --accelerated        # 需要 26 天以上才判内存趋势
```

## 5. 测试、覆盖率与网络守卫（R-NFR-004、R-NFR-005，第 16 轮 A）

**覆盖率**（`src/twin` 的行覆盖率，沙箱，2026-10-10）。整套测试分三片并发运行（每片约 20 分钟，三片同时跑），各片写自己的覆盖率数据，合并后交给门槛脚本（6,417 个测试通过，21 个跳过；`live` 标记的 5 个没有选中）：

```
# 三个终端（或后台）各跑一片：
COVERAGE_FILE=cov.1 uv run pytest -q -m "not live" --cov=src/twin $(uv run python scripts/shard_tests.py --shard 1 --of 3)
COVERAGE_FILE=cov.2 uv run pytest -q -m "not live" --cov=src/twin $(uv run python scripts/shard_tests.py --shard 2 --of 3)
COVERAGE_FILE=cov.3 uv run pytest -q -m "not live" --cov=src/twin $(uv run python scripts/shard_tests.py --shard 3 --of 3)
# 合并并判门槛：
uv run coverage combine cov.1 cov.2 cov.3 && uv run coverage json && uv run python scripts/coverage_gate.py
```

| 子包 | 覆盖率 | 已覆盖 / 语句数 | 门槛 75 % |
| --- | --- | --- | --- |
| (top-level) | 98.9 % | 562 / 568 | 通过 |
| channel | 99.7 % | 4,281 / 4,296 | 通过 |
| commands | 98.5 % | 1,485 / 1,508 | 通过 |
| config | 99.1 % | 1,027 / 1,036 | 通过 |
| engine | 98.6 % | 4,392 / 4,454 | 通过 |
| eval | 98.6 % | 2,691 / 2,728 | 通过 |
| ingest | 95.7 % | 2,770 / 2,894 | 通过 |
| learning | 97.5 % | 658 / 675 | 通过 |
| llm | 99.8 % | 2,786 / 2,792 | 通过 |
| memory | 98.7 % | 3,836 / 3,886 | 通过 |
| ops | 97.9 % | 5,692 / 5,814 | 通过 |
| profile | 98.7 % | 4,335 / 4,391 | 通过 |
| retrieval | 97.7 % | 1,462 / 1,497 | 通过 |
| schedule | 97.8 % | 3,357 / 3,434 | 通过 |
| serving | 96.2 % | 2,094 / 2,176 | 通过 |
| stickers | 98.0 % | 1,480 / 1,510 | 通过 |
| storage | 99.0 % | 2,509 / 2,534 | 通过 |
| training | 95.3 % | 3,641 / 3,820 | 通过 |
| **合计** | **98.1 %** | **49,058 / 50,013** | 门槛 85 %，通过 |

没有覆盖到的行分散在错误处理分支里，最多的三个模块是 `training/parity_check.py`（52 行，78.8 %）、`engine/machine.py`（36 行，96.0 %）、`ingest/importer.py`（32 行，91.9 %）。只在 Windows 上执行的分支沙箱里无法跑，列在 `docs/PENDING_USER_ACTIONS.md`。`live` 标记的测试（真实 DeepSeek、真实 llama-server 等）不在这个数字里。

**其他质量检查**：`ruff format --check .` 与 `ruff check .` 通过；`mypy src/twin` 与 `mypy --platform win32 src/twin`（469 个文件）通过；`scripts/trace_check.py`、`scripts/privacy_scan.py` 通过。

**网络守卫**（`tests/conftest.py::no_network`，D-559）：整套测试不访问网络，忘了 mock 的测试会失败。第一次全套运行就抓到两个老测试悄悄连了外网（沙箱有出站代理，所以此前只是"能通"，在 CI 上同样会去下载或联系真实服务）：

- `tests/unit/test_ingest_cli.py::test_a_queued_import_is_executed_by_the_job_runner_when_the_app_is_stopped`：导入后的钩子会为示例库编码，试图从 Hugging Face 下载 bge 模型（连 `13.226.251.x:443`）。改为用离线的哈希编码器（`embedder` 夹具）。
- `tests/unit/test_ilink_channel.py::test_the_application_runs_the_channel_and_reports_the_login_in_its_health`：测试里存了登录凭据，应用停止时通道向真实的 iLink 服务器发"停止"通知。改为 `respx` 回答。

**时间覆盖扫描**（`scripts/time_coverage_scan.py`，D-560）：469 个模块里 132 个用到时区、时钟或时间服务；每一个都有直接导入它并且自己控制时间的测试，缺口 0；夏令时切换日的测试 19 个，时区切换的测试 5 个。命令：`uv run python scripts/time_coverage_scan.py`。
