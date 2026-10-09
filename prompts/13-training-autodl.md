# 第 13 轮：风格模型训练——训练集导出、加密训练包、AutoDL 单卡（RTX 5090 / RTX PRO 6000）训练、评估生成、DPO、GGUF 导出、清理

> 里程碑：M5 · 前置：第 00–12 轮全绿，`twin eval gate M4 --check` 已通过；记忆回放（第 07 轮）已覆盖要导出的日期范围 · 需要用户：在 AutoDL 租一台 RTX 5090（32GB）或 RTX PRO 6000（96GB）实例，**数据盘按档位扩容**（8B ≥ 70GB、14B ≥ 110GB、32B ≥ 230GB；默认 50GB 不够），把 SSH 主机、端口、用户名填进 `autodl.*` 配置，密码用 `twin secrets set autodl_password` 存 keyring。

## 先读
`CLAUDE.md`（铁律 8、13）；`docs/SPEC.md` §21 训练全部（重点 R-TRN-011、R-TRN-013）、§22 中 R-SRV-001、R-RET-003、R-MEM-010、R-LLM-009（`ConsistentRedactor`）、R-LLM-014、R-LRN-002/004、R-PRIV-003、R-PERS-004、R-SAFE-006；第 07 轮 `AsOfView`、第 09 轮 `StylePromptBuilder`。

在动手前，用联网检索确认并在 `training/README.md` 中写明核对日期与版本：
- LLaMA-Factory 版本：沿用第 09 轮已固定并写入 `training/README.md` 的版本，核对其仍可用（如需升级，先告诉我，并重跑第 09 轮的字符级模板测试）；`qwen3_nothink` 模板源码（确认仍是纯 ChatML、无 think 标记）、`get_template_and_fix_tokenizer` 与 `encode_multiturn` 接口、`mask_history` 参数、ShareGPT 格式对角色顺序的要求、`flash_attn: sdpa` 参数名、`stage: dpo` 用法、`llamafactory-cli export` 用法与 QLoRA 适配器合并要求（在 bf16 基座上合并、不设 `quantization_bit`）；
- Blackwell（sm_120）训练要求：PyTorch ≥ 2.7 且为 CUDA 12.8 构建、不使用 CUDA 13.x、bitsandbytes 4-bit 支持情况；
- AutoDL：数据盘 `/root/autodl-tmp`、学术加速 `source /etc/network_turbo`（只加速 GitHub/Hugging Face，用完 `unset http_proxy https_proxy`）、ModelScope 上的 `Qwen/Qwen3-8B`、`Qwen/Qwen3-14B`、`Qwen/Qwen3-32B`；
- llama.cpp `convert_hf_to_gguf.py` 与 `llama-quantize` 对 Qwen3 的支持；
- vLLM 在 sm_120 上可用的版本、`--enable-lora` 与 `/v1/completions`、`/tokenize` 接口。
与 SPEC 不一致时先告诉我。

## 本轮目标
把用户本地的真实聊天变成"只教风格模型学她怎么说话"的训练数据（格式与线上推理完全一致、无未来信息泄露、已脱敏），加密打包后在 AutoDL 单卡上可重复地完成 SFT（可选 DPO）、评估生成、导出 GGUF，并把产物取回本地、清理云端数据。

## 必须实现的需求
R-TRN-001～013、R-IMP-011（注册重训检查钩子）、R-SRV-001（`model_registry` 表、登记训练产物、锁定版本）、R-SAFE-006（训练目标删除不可复现的行）、R-LLM-014（规划合成作为一次性批任务）、R-PRIV-003、R-STO-006（本轮的表：`training_runs`、`dataset_versions`、`model_registry`）。

## 详细要求

### A. 训练集导出（R-TRN-002～007）
1. `twin train export [--from 日期] [--to 日期]`：
   - 遍历她的每个真实回复块（只读 `messages`，`is_sent=False`；函数签名与运行时断言双重保证，R-TRN-004）；
   - 全部上下文只经第 07 轮 `AsOfView(t)` 读取（t = 该回复块第一条消息时刻；R-TRN-013，铁律 13）：前面同一会话段最多 8 个合并轮次；当时当地时间与星期（按她当时所在地时区，R-ACT-001）；她当时的作息状态（`AsOfView(t).her_state()`，即 pre_holdout 作息模型的 `typical_state`：睡/边缘/忙/空闲，与评估沙盒同一函数）；表情包标签与表情代码按 as-of 数据视图；as-of 记忆块（预算与线上 style 后端一致）；pre_holdout 范围的精简人设卡（只含风格，记录版本号）；
   - 用第 09 轮 `StylePromptBuilder` 渲染成与线上完全相同的结构（含"她开头的轮次进前文"规则），再转为 LLaMA-Factory ShareGPT 格式（`system` + 以 human 开头、交替的 `conversations`），以及一份同内容的"渲染后纯文本"用于一致性校验。
2. 表示约定（R-TRN-003、R-SAFE-006）：**目标**（她的回复块）中：连发 → 换行；表情包 → `[表情包:<标签>]`（用第 06 轮标签，未打标签的表情包写 `[表情包]`）；表情代码原样；她用引用 → 首行 `[引用:<被引用片段，截断 30 字>]`；不可复现的行（第 03 轮 `is_reproducible` 为假：图片、语音、视频、通话、转账、红包、位置、文件、链接、聊天记录、系统事件）**从目标中删除**。**上下文**（用户与她之前的轮次）中这些仍用 `render_event_text()` 呈现。
3. 过滤：目标删除不可复现行后为空的样本丢弃；目标超过 cutoff 的样本截断上下文而非目标；统计各丢弃原因；导出后用 `EventTextDetector` 扫描全部目标，出现任何事件文字即导出失败。
4. hybrid 规划样本（R-TRN-005）：抽取 `training.hybrid_plan_ratio`（30%）样本，作为非高峰离线任务调用 DeepSeek 读她的真实回复反推 `{intent, facts_to_use, tone, bubble_hint}`，写入提示词的规划字段；规划合成是一次性批任务（R-LLM-014：估算费用、我批准、可续跑）；`facts_to_use` 只能引用该样本 as-of 记忆块中已有的事实（校验，不在其中的丢弃）。
5. 切分（R-TRN-006）：测试集 = 第 05 轮 `holdout_cutoff()` 之后的全部样本；验证集 = 切分点之前最近 5%；其余训练；切分信息写入 `dataset_versions`。
6. 脱敏（R-TRN-007）：用 `ConsistentRedactor` 对整个数据集做一致占位；导出后再跑一遍扫描，发现残留 PII 则失败。
7. 统计报告：样本数、各切分数量、平均轮数、目标长度分布、表情包与表情代码占比（应接近她的画像）、规划样本比例、估算 token 数与各档位预计训练时长。

### B. 守护
- 单测：把 `bot_turns`、`preference_pairs` 的对象传入 SFT 导出入口必须失败；导出结果中任何目标文本都能在 `messages` 中找到对应的她的消息（抽样逐条核对）。
- 防泄露（R-TRN-013）：导出代码只依赖 `AsOfView`（结构测试：导出模块不导入任何直接查询全量数据的仓储类）；注入测试——合成数据里一个只在目标块及之后出现的事实不得出现在该样本的提示词里；所用人设卡、画像、作息都是 pre_holdout 版本。

### C. 训练包（R-TRN-008）
- `twin train bundle --profile 5090-8b|5090-14b|pro6000-14b|pro6000-32b`：包含数据集（JSONL）、`dataset_info.json`、由模板生成的 LLaMA-Factory YAML（SFT、可选 DPO、eval、export）、`training/autodl/*.sh`、`manifest.json`（文件列表与 sha256、数据集版本、档位、人设卡版本、提示词模板版本）；`tar` + `zstd` 压缩后用口令（scrypt 派生 + AES-256-GCM）加密；口令只在交互输入，不落盘；附带一个仅依赖 Python 标准库 + `cryptography` 的解密脚本。

### D. 档位配置（R-TRN-001）
为 SPEC 表中的四个档位各生成一份 YAML 模板（`training/llamafactory/*.yaml.j2`）：
- 共同：`stage: sft`、`finetuning_type: lora`、`lora_target: all`、`template: qwen3_nothink`、`mask_history: true`、`cutoff_len: 2048`、`learning_rate: 1e-4`、`lr_scheduler_type: cosine`、`warmup_ratio: 0.05`、`bf16: true`、`flash_attn: sdpa`（以核对后的参数名为准）、`gradient_checkpointing: true`、`val_size` 改为使用独立验证集文件、`eval_strategy: steps`、早停（patience 2）、`save_steps` 与 `resume_from_checkpoint` 支持、`report_to: none`（日志写本地）。
- `5090-8b`：Qwen3-8B，LoRA r=32 α=64，`per_device_train_batch_size` 与 `gradient_accumulation_steps` 按 32GB 显存给出并在 `setup.sh` 末尾用 1 个 step 的试跑自动下调直到不 OOM。
- `5090-14b`：Qwen3-14B，`quantization_bit: 4`、`quantization_method: bnb`；对应的 export YAML 加载 bf16 基座、**不设** `quantization_bit`、`export_device: cpu`（QLoRA 适配器只能在未量化基座上合并）。
- `pro6000-14b`：Qwen3-14B，LoRA r=32。
- `pro6000-32b`：Qwen3-32B，LoRA r=16。
- epoch：样本 < 20k 时 3，否则 2。

### E. AutoDL 脚本（R-TRN-009，`training/autodl/`，全部 `set -euo pipefail`，可重复执行、幂等）
1. `setup.sh`：打印 GPU 型号、显存、驱动、计算能力；若为 sm_120 则确认 `torch>=2.7` 且 `torch.version.cuda` 为 12.8 系列并且 `torch.cuda.get_arch_list()` 含 `sm_120`，否则安装 cu128 版 PyTorch（禁止 CUDA 13.x）；先把 `HF_HOME`、`MODELSCOPE_CACHE`、`PIP_CACHE_DIR`、`TMPDIR` 设到 `/root/autodl-tmp/` 下（系统盘很小）；`source /etc/network_turbo` 后安装固定版本的 LLaMA-Factory、bitsandbytes、modelscope 与 `zstd`（apt 或 `zstandard`，解包训练包需要），结束 `unset http_proxy https_proxy`；**按档位检查数据盘剩余空间**（基座 bf16 大小 × 3 + 量化产物 + 20% 余量；约 8B ≥ 70GB、14B ≥ 110GB、32B ≥ 230GB）与内存（QLoRA 档位合并需在 CPU 加载 bf16 基座），不足时停止并打印"请到 AutoDL 控制台扩容数据盘/换实例"；`USE_MODELSCOPE_HUB=1` 下载基座到 `/root/autodl-tmp/models/`；工作目录 `/root/autodl-tmp/twin/`；最后运行模板一致性测试（R-TRN-011）与 1 step 试跑。全部输出 tee 到日志。
2. `decrypt.sh`：解密训练包到 `/root/autodl-tmp/twin/data/`，校验 manifest sha256。
3. `train.sh <profile>`：SFT；断点续训；结束时输出最佳 checkpoint、验证 loss 曲线（PNG）与指标 JSON。
4. `eval.sh`：用最佳适配器对测试集上下文生成回复（固定温度、top_p、种子；每个上下文生成 1 条），输出 `eval_generations.jsonl`（只含样本 id 与生成文本），以及验证集 loss。
5. `dpo.sh`：若训练包中有 ≥ 200 条偏好对，在 SFT 适配器上做 DPO（β、学习率按 LLaMA-Factory 推荐值并写入配置），产出新适配器与指标。
6. `export.sh`：`llamafactory-cli export` 合并 LoRA 得到 bf16 HF 模型（QLoRA 档位用上面的 bf16 + `export_device: cpu` 配置）→ 克隆固定版本 llama.cpp、CPU 编译 `llama-quantize` → `convert_hf_to_gguf.py` 生成 F16 GGUF → **删除合并后的 HF 权重** → 量化 Q4_K_M、Q5_K_M、Q8_0 → **删除 F16 GGUF** → sha256 → `artifacts/manifest.json`（含模板版本、人设卡版本、画像版本、数据集版本，供 `model_registry` 锁定）；同时保留 LoRA 适配器目录（远程 vLLM 用基座 + 适配器）。每一步前检查剩余空间。
7. `serve_vllm.sh`（为第 14 轮远程推理准备）：在实例上启动 vLLM（基座 + `--enable-lora` 加载适配器），只监听 127.0.0.1；客户端只用 `/v1/completions`（传渲染好的字符串）与 `/tokenize`；记录 vLLM 在 sm_120 上的可用版本与注意事项。
8. `cleanup.sh`：用 `shred`（或多次覆盖后删除）清除 `data/`、解密目录、包含样本的日志与缓存；保留 `artifacts/` 直到本地确认下载完成；最后打印"请到 AutoDL 控制台释放实例（释放后数据盘才会被回收）"。

### F. 本地编排（R-TRN-010）
- `twin train remote connect|upload|setup|train|eval|dpo|export|download|cleanup|status|all`：用 `asyncssh`（纯 Python，Windows 可用，支持 AutoDL 的密码登录与密钥文件；连接参数来自 `autodl.*` 配置，密码来自 keyring）连接；上传带断点续传与 sha256 校验；远程命令在 `tmux`/`nohup` 中运行，断网后可重新附着并继续拉日志；`download` 把 GGUF、适配器、评估生成与指标拉到 `data/models/<run_id>/` 并校验 sha256；`cleanup` 执行后在 `training_runs` 记录完成时间，未清理的运行在 `/状态` 中提醒。
- 每一步记录到 `training_runs`：档位、数据集版本、超参、开始结束时间、GPU 型号、峰值显存、最佳验证 loss、产物 sha256、清理时间。

### G. 模板一致性（R-TRN-011）
- `tests/integration/test_template_parity.py`（R-TRN-011）：对一组合成样本（含多行回复、表情包标记、规划字段、她开头的上下文、单轮与 8 轮上下文、超长截断等边界）：
  ① 用导出器生成 ShareGPT 样本，交给固定版本 LLaMA-Factory 的 `get_template_and_fix_tokenizer(tokenizer, data_args)` + `template.encode_multiturn(...)`（与训练时相同的入口）得到每轮的 (prompt_ids, response_ids)；
  ② 用 `StylePromptBuilder` 渲染同一样本的推理提示词，经基座 tokenizer 编码；
  ③ 断言 ② 等于 ① 中最后一轮之前全部 token 加上最后一轮的 prompt_ids；最后一轮 response_ids 解码后等于目标文本加 `<|im_end|>`（及模板规定的换行）；
  ④ 断言 `mask_history: true` 下只有最后一轮计算 loss（检查 LLaMA-Factory 处理器产出的 labels）。
  **不**与 `apply_chat_template` 比较（Qwen3 官方模板会在最后一轮插入空 think 块，与训练格式不同）。本地运行只需 tokenizer 文件与固定版本的 LLaMA-Factory（标记 `integration`）；AutoDL `setup.sh` 末尾再跑一次。

### H. 模型登记（R-SRV-001 的登记部分）
- 本轮建 `model_registry` 表：`id, run_id, profile, base_model, quant, path, sha256, size, template_version, persona_version, profile_version, dataset_version, created_at, eval(json), enabled, active, gate_passed`；`twin model register <产物目录>`（校验 manifest 与 sha256，从 manifest 读取并锁定各版本）、`twin model list|show <id>`。激活、服务与门槛在第 14 轮。
- `StylePromptBuilder` 增加"按锁定版本渲染"：读取对应的模板版本与人设卡版本（第 06 轮的版本化存储），有测试。

### I. 重训与 DPO 管理（R-TRN-012）
- 每次导入后检查"自上次训练以来新增她的消息占比"，≥ 10% 时发 `retrain_suggested` 告警并在 `/状态` 显示；`preference_pairs` ≥ 200 时提示可做 DPO；DPO 导出（`twin train export-dpo`）只读 `preference_pairs`：把结构化的 `prompt_sample`（系统段 + 交替对话）原样转成 LLaMA-Factory 偏好数据格式（ShareGPT + `chosen`/`rejected`），由 LLaMA-Factory 套一次 `qwen3_nothink` 模板——不得写入任何已渲染的模板字符串（有测试：导出文件中不含 `<|im_start|>`）；样本记录的模板与人设卡版本与 SFT 适配器锁定的版本不一致时，用锁定版本重新生成系统段。

## 测试要求
- 导出：无未来泄露（注入测试 + pre_holdout 版本）、只用她的消息、表示约定（目标里没有任何事件文字、上下文里保留）、她开头的上下文进前文、过滤与截断、切分边界与 `holdout_cutoff()` 一致、脱敏一致性与残留扫描。
- `setup.sh` 的磁盘与内存检查函数（不同档位与剩余空间组合）；`export.sh` 的删除顺序（dry-run 下打印的步骤顺序快照）。
- 模型登记：manifest 校验、版本锁定、按锁定版本渲染。
- 规划合成：JSON 校验、可续跑、费用估算。
- 训练包：加密往返、manifest 校验、口令错误失败。
- YAML 模板渲染快照测试（四个档位）。
- AutoDL 脚本：`shellcheck` 通过；用 Docker（若可用）或在 CI 中以 dry-run 模式验证参数解析与幂等逻辑；`setup.sh` 的版本检查函数单测（bats 或 Python 包装）。
- 远程编排：用本地 SSH 测试服务器（`asyncssh` 自带服务端）测试上传续传、命令执行、断线重连、下载校验。
- 模板一致性（integration）。

## 验收
```
uv run pytest -q
uv run twin train export
uv run twin train bundle --profile 5090-8b      # 或 pro6000-14b
uv run twin train remote all --profile 5090-8b  # 我提供实例信息后执行
uv run twin train remote cleanup
uv run twin model register data/models/<run_id>
uv run python scripts/trace_check.py --round 13
```

## 不要做
- 不要上传未脱敏或未加密的数据；不要把口令或 AutoDL 密码写进任何文件。
- 不要在旧适配器上无限叠加训练（默认从基座全量重训）。
- 不要在训练集中放入任何机器人回复。

## 完成后汇报
按 CLAUDE.md 格式；附导出统计、所用档位、训练耗时与峰值显存、验证 loss、产物 sha256、云端清理确认。
