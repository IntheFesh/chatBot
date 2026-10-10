# CI：矩阵分片

`.github/workflows/ci.yml` 把原来的"每个 OS 一个作业顺序跑完全部"拆成四类作业。所有质量门槛的语义不变（ruff、mypy --strict、全部测试、覆盖率 总计 ≥ 85 % 且每个子包 ≥ 75 %、隐私扫描、追溯检查、CLI 冒烟），只是分到不同的虚拟机上并行执行。取舍与理由见 `docs/DECISIONS.md` D-490～D-497。

## 作业

| 作业（检查名） | 个数 | 做什么 |
| --- | --- | --- |
| `lint (ubuntu-latest)`、`lint (windows-latest)` | 2 | `uv sync --frozen` → `ruff check` → `ruff format --check` → `mypy src/twin` →（仅 Linux）`mypy src/twin --platform win32` → `privacy_scan.py` → `trace_check.py`（全局一致性）→ `stub_scan.py`（`src/` 里没有桩、占位和玩具实现，白名单 `scripts/stub_scan_allowlist.toml` 逐条写理由）→ `decisions_check.py`（`DECISIONS.md` 的编号唯一、引用的 D-xxx 与测试都存在、取舍表覆盖 SPEC 列出的全部取舍点）→ `shard_tests.py --check-collection` → `twin --help`、`twin doctor`。Windows 上的 mypy 与 `twin doctor` 是原生运行的，不再在每个测试分片里重复。 |
| `tests (ubuntu-latest, i/3)`、`tests (windows-latest, i/5)` | 3 + 5 | 每片一台独立虚拟机：`uv sync --frozen` → `shard_tests.py` 算出本片的测试文件 → `pytest -q -m "not live" --cov=src/twin -p no:cacheprovider --junitxml=junit.xml <文件…>`。片内顺序执行。上传 `.coverage.<os>.<i>-of-<n>` 与 junit 报告，并在作业摘要里列出本片每个测试文件的耗时。 |
| `coverage (ubuntu-latest)`、`coverage (windows-latest)` | 2 | `needs: tests`：下载该 OS 全部分片的覆盖率数据 → `shard_tests.py --check-shard-set`（`1-of-n … n-of-n` 一片不缺）→ `coverage combine` → `coverage json -o coverage.json` → `scripts/coverage_gate.py`（门槛与参数没有改）。只装 dev 依赖组，不装 torch 等项目依赖。 |
| `ci gate` | 1 | `if: always()`，仅当 `lint`、`tests`、`coverage` 全部成功才通过；需要"必需检查"时只设这一项。 |

并发控制不变（同一分支的新推送取消旧运行）。`windows-subset.yml`（只在 `ci/windows-subset.txt` 变化时跑指定测试）保留，用来在几分钟内复现某个 Windows 问题。

为什么每片内部不并行（`pytest-xdist`）：有的测试用 Windows 命名互斥体、固定的本地 ssh/http 端口，同一台机器上的多个 worker 会互相干扰。分片是把工作分到**不同机器**上，所以没有这个问题。

## 分片器 `scripts/shard_tests.py`

```
uv run python scripts/shard_tests.py --shard 2 --of 4                # 第 2/4 片的测试文件，一行一个
uv run python scripts/shard_tests.py --of 4 --plan                   # 所有分片的文件数、测试数、权重
uv run python scripts/shard_tests.py --timings junit.xml             # 每个测试文件的耗时
uv run python scripts/shard_tests.py --update-weights junit.xml --platform windows
uv run python scripts/shard_tests.py --check-collection              # pytest 收集到的文件都在分片范围内
```

- **全集来自磁盘。** `tests/` 下所有 `test_*.py` 与 `*_test.py`（pytest 默认规则），不是手写清单。新增测试文件下一次运行就自动被某一片包含；每个文件恰好属于一个分片；各片的并集等于全集；`--of 1` 等于全集。
- **按文件分。** 模块级和类级的 fixture 保持完整。
- **LPT 贪心均衡。** 文件按权重从大到小，依次放进当前最轻的分片；权重相同按路径、分片号决定，结果是权重与文件树的纯函数（确定）。
- **权重** `ci/test_weights.json`：每个平台（`linux`、`windows`）每个文件的秒数。没登记的文件按"该文件定义的测试数 × 已登记文件的平均秒/测试"估算，所以新文件不会因为单位不一致而被排到极端位置；某个平台还没有任何数据时借用另一个平台的。权重只影响均衡，不影响正确性：权重缺失、过期、被删文件都不会让任何测试漏跑（过期条目只在日志里提示）。
- **多一层保险。** `lint` 作业的 `--check-collection` 让 pytest 自己收集一遍，如果它收集到分片器看不到的文件（例如有人改了 `python_files`）就失败；`coverage` 作业的 `--check-shard-set` 保证 `1-of-n … n-of-n` 全部到齐（有人改矩阵少了一片就失败）。

## 修改分片数

`ci.yml` 的 `tests` 矩阵里每个 OS 的每一片是一行 `{ os, shard, shards }`，同一 OS 的 `shards` 必须相同。加一片：给该 OS 的所有行把 `shards` 改成 n+1，再加一行 `shard: n+1`。其余（覆盖率合并、完整性检查、检查名）自动跟随。

## 更新权重

每个测试分片作业的摘要（以及日志末尾）有一张"每个测试文件耗时"表，junit 报告也作为产物保留 14 天。想重新均衡时：

```
# 方式一：下载某次运行的各片 junit-<os>-*.zip，解压到一个目录
uv run python scripts/shard_tests.py --update-weights path/to/junit-dir --platform windows
# 方式二：把各片日志里的耗时表复制进一个文本文件
uv run python scripts/shard_tests.py --update-weights timings-windows.log --platform windows
```

只更新被测量的文件和该平台，其余保持；磁盘上已不存在的文件会被清除。提交 `ci/test_weights.json` 即可。每次大改测试后顺手更新一次；偏差不大时不必更新。

同一个文件在不同虚拟机上的耗时相差可达两成（第一个用到某个重依赖的测试还要承担进程里一次性的导入成本），所以第二次起建议加 `--blend 0.5`：新权重 = 一半旧权重 + 一半新测量，相当于对多次运行取滑动平均。

`gh api repos/<owner>/<repo>/actions/artifacts/<id>/zip > junit.zip` 可以直接下载某个 junit 产物（`gh api repos/<owner>/<repo>/actions/runs/<run>/artifacts` 列出 id）。

## 本地用法

```
uv run pytest -q -m "not live" $(uv run python scripts/shard_tests.py --shard 1 --of 3)
```

（PowerShell：`uv run pytest -q -m "not live" (uv run python scripts/shard_tests.py --shard 1 --of 3)`。）

## 排查

- **某一片失败而全量顺序跑能过**：说明某个测试依赖了同文件之外的测试遗留的状态。修测试使其自洽（见 `docs/EXECUTION_NOTES.md` 的测试约定），不要把依赖的文件钉在同一片里。复现：`uv run pytest <失败的文件>` 单独跑。
- **覆盖率门槛失败而各片都绿**：看 `coverage (<os>)` 作业打印的各子包表，与旧的单进程结果等价；合并后的 `coverage.json` 作为产物 `coverage-report-<os>` 保留 7 天。
- **要在 Windows 上快速验证修复**：编辑 `ci/windows-subset.txt` 推送，`windows-subset` 作业只跑列出的测试。

## 实测（提交 5c02d9b 与 4fc485a 两次运行）

| 运行 | 分片依据 | Windows 各片 pytest 用时 | Linux 各片 pytest 用时 | 整条流水线 |
| --- | --- | --- | --- | --- |
| 第 1 次（run 89） | 测试数（权重文件为空） | 4 片：14:14、16:53、18:08、19:07 | 3 片：7:00、13:01、13:50 | 20 分 55 秒 |
| 第 2 次（run 91） | 第 1 次的每文件耗时 | 5 片：约 10.5、12.0、12.7、14.1、14.5 分 | 3 片：约 8.6、12.6、13.0 分 | 15 分 43 秒 |
| 第 3 次（run 38025492454，提交 10a6812，已含第 14 轮新增的测试） | 前两次的平均；新文件按测试数估算 | 5 片：14:48、15:10、17:09、13:31、17:33 | 3 片：8:16、16:23、15:24 | 18 分 27 秒 |

旧的单作业全量：Windows 约 58 分钟、Linux 约 25 分钟。固定开销很小：缓存命中时 Windows 的 `uv sync --frozen` 约 16 秒，检出、选择文件、上传产物合计不到 1 分钟，所以多分一片几乎不增加总运行分钟数；`lint (windows-latest)` 约 3 分钟，`coverage` 作业约 30 秒，都不在关键路径上。顺序跑的测试时间合计：Windows 约 65 分钟、Linux 约 32 分钟；理想情况下每片：Windows 4 片 16.4 分、5 片 13.1 分、6 片 10.9 分；Linux 3 片 10.8 分。同一文件在不同机器上的耗时波动是各片之间仍有 ±15 % 差别的主要原因。

第 3 次比第 2 次慢，原因是新合入的第 14 轮测试文件（真实子进程、真实 SSH 的 `test_serving_*`）还没有登记权重，按测试数估算得太轻，以及 Windows 虚拟机的波动；用该次运行的 junit 以 `--blend 0.5` 刷新权重后，Windows 5 片的预计权重各 843.6 秒（14.1 分），Linux 3 片各 728.7 秒（12.1 分）。**新增测试文件后的第一次运行总是按估算分片，下一次运行前记得刷新权重。**

