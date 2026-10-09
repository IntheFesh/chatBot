# 第 14 轮：风格模型部署与后端接入（本地 llama.cpp / 远程 vLLM、模型登记、上线门槛、自动回退）

> 里程碑：M5（本轮末判定）· 前置：第 00–13 轮全绿，至少完成一次训练、已下载并 `twin model register` 登记产物 · 需要用户：告诉我这台 Windows 电脑的显卡型号与显存（`twin doctor` 会检测）；做盲测判断

## 先读
`CLAUDE.md`（§7 子进程）；`docs/SPEC.md` §22 部署全部、R-LLM-008/011、R-ENG-006、R-TRN-009/010/011、R-EVAL-001/002/009/010、R-OPS-003/004、§26 M5。

在动手前联网确认并记录：llama.cpp 当前发行版本中 Windows CUDA 构建、配套 `cudart` 运行库包与 CPU 构建的发行包命名、`llama-server` 的 `/completion`、`/tokenize`、`/health` 接口、对 Qwen3 GGUF 的支持（是否默认加 BOS）；vLLM 在 sm_120 上可用的版本组合与 `/v1/completions`、`/tokenize` 接口。

## 本轮目标
让训练好的风格模型以可靠、可观测的方式接入回复引擎：有合适显卡就在本机跑（llama.cpp），否则在 AutoDL 远程跑（vLLM + SSH 隧道）；模型有登记与版本；只有在盲测胜过纯 DeepSeek 时才允许设为默认；任何故障自动回退 DeepSeek。

## 必须实现的需求
R-SRV-001～005、R-SCOPE-007（风格模型后端上线）、R-ENG-006（style/hybrid 在真实模型上的端到端验证）、R-LLM-011（真实端点验证与分词核对）、R-TRN-011（服务端分词核对）、R-EVAL-001（多后端盲测）、R-EVAL-010（注册 M5 判定器）。

## 详细要求

### A. 模型启用（R-SRV-001 的启用部分；表与登记在第 13 轮）
- `twin model activate <id> [--force]|disable <id>`（轻量修改，运行中的应用经 `state_version` 感知并启停服务）；激活规则见 D。
- 实现第 01 轮预留的 `StyleBackendStatus`：当前激活模型是否通过门槛（`gate_passed`）、是否健康——供 R-LLM-008 第 ④ 级降级与 `/后端` 指令使用。

### B. 本地推理（R-SRV-002）
1. `scripts/windows/get_llamacpp.ps1`：下载固定版本 llama.cpp Windows 发行包（检测到 NVIDIA 显卡且驱动支持 → CUDA 构建，**同时下载同版本的 `cudart` 运行库包**并解压到同一目录；否则 CPU 构建），校验 sha256，解压到 `tools/llama.cpp/<版本>/`（gitignore）；`twin doctor` 检查 CUDA 构建所需的 DLL 齐全。
2. 量化推荐：根据显存（`nvidia-smi --query-gpu=memory.total`）与模型大小推荐 Q8_0 / Q5_K_M / Q4_K_M（留 20% 余量与上下文所需显存）；CPU 时推荐 Q4_K_M 并提示延迟。
3. `twin model serve`（也由 `twin run` 在后端为 style/hybrid 时自动管理）：以子进程启动 `llama-server.exe`（`--host 127.0.0.1 --port 8081 -c 4096 -ngl 999 --parallel 1`，以及与模板相关的参数——我们走原始补全接口，提示词由 `StylePromptBuilder` 渲染，不依赖服务端聊天模板）；标准输出与错误写日志；健康检查 `/health`；崩溃自动重启（指数退避）；随主程序退出（Job Object）。
4. 分词核对（R-TRN-011 第 4 点）：启动后把一组渲染好的提示词（含特殊 token 与中文、表情包标记）交给 `/tokenize`，与 HF tokenizer（基座 tokenizer 文件，本地缓存）的结果逐 id 比对；检查是否多了 BOS 或特殊 token 被拆开；不一致则拒绝启用并告警，报告差异位置。
5. 预热：发送一个短提示，记录首 token 延迟与生成速度；写入 `/状态`。

### C. 远程推理（R-SRV-003）
1. 依赖第 13 轮 `serve_vllm.sh`；`twin model tunnel start|stop|status`：`asyncssh` 建立本地端口转发（本机 127.0.0.1:`style_model.tunnel.local_port` → 实例 127.0.0.1:`style_model.tunnel.remote_port`），断线自动重连；`StyleModelClient` 切到 `vllm_completion` 模式指向隧道（`/v1/completions`，传渲染好的字符串，`model` 为 LoRA 名称）；同样做分词核对（vLLM `/tokenize`）。
2. `/状态` 与每天一次的系统消息提醒"远程风格模型按小时计费，实例运行中"，并显示已运行时长；用户可 `/后端 deepseek` 后由 `twin model tunnel stop` + 提示到控制台关机。

### D. 上线门槛（R-SRV-005）
1. `twin model evaluate <id>`：使用第 09b 轮的评估沙盒与盲测框架——对同一批**新的**留出上下文分别用 deepseek 后端与该风格模型（style 与 hybrid 各一份）在 `holdout` 模式下生成（风格模型用 `model_registry` 锁定的版本渲染），然后进入盲测打分界面；再对同一批生成跑留出集风格指标（`twin eval style --source eval_items`）。
2. 注册 M5 判定器并在 `twin model activate <id>` 前检查（R-SRV-005）：style 或 hybrid 的猜对率低于 deepseek，单侧两比例检验 p < 0.1，每个后端有效判断 ≥ 50 对；且该方式的留出集风格指标每项偏差在 ±30% 内。不满足则拒绝并给出数据；样本不足时说明还需多少对。`--force` 允许强制激活，但记录审计、`gate_passed=false`、在 `/状态` 中标注"未通过门槛"，且不参与 R-LLM-008 第 ④ 级降级。
3. 通过后把运行时设置 `backend.active` 设为评估更优的 style 或 hybrid，`gate_passed=true`。不通过时保留 deepseek 后端——这不阻塞第 15、16 轮。

### E. 回退（R-SRV-004）
- 引擎在风格模型健康检查失败、超时或连续 3 次输出被后处理判为硬违规时，切回 deepseek 后端并告警（`style_model_down`）；恢复健康 10 分钟后自动切回原后端并通知；所有切换写审计。

## 测试要求
- 激活门槛与 M5 判定器（样本不足、未胜出、盲测胜出但风格指标超差、全部通过、强制激活）；`StyleBackendStatus` 各状态。
- 本地服务管理：用一个模拟 `llama-server` 的测试可执行文件（`tests/support/` 下的小 Python 脚本，提供相同的 HTTP 接口与可控的崩溃）验证启动、健康检查、崩溃重启、随主进程退出（Windows 标记）。
- 分词核对：模拟服务返回多一个 BOS、特殊 token 被拆开时拒绝启用。
- 量化推荐：不同显存与模型大小组合。
- 隧道：本地 SSH 测试服务器上的端口转发、断线重连。
- 回退与自动恢复。
- 端到端（integration，可选在真实模型上运行）：本地控制台通道 + 真实 GGUF（若存在）+ hybrid 后端，跑 10 轮对话，记录延迟与后处理违规率。

## 验收
```
uv run pytest -q
powershell -ExecutionPolicy Bypass -File scripts/windows/get_llamacpp.ps1
uv run twin model serve            # 或 twin model tunnel start（远程）
uv run python scripts/trace_check.py --round 14
```

## 门槛检查（M5；不通过则保留 DeepSeek 后端，不阻塞后续轮次）
```
uv run twin model evaluate <id>    # 我做盲测判断
uv run twin eval gate M5
uv run twin model activate <id>    # 只有门槛通过才会成功
```

## 不要做
- 不要让 llama-server 监听非本机地址。
- 不要在未通过门槛时默默切换默认后端。

## 完成后汇报
按 CLAUDE.md 格式；附本机推理速度（tokens/s、首 token 延迟）或远程隧道延迟，以及门槛检查结果。
