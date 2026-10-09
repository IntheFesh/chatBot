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
