# 风格模型部署与服务笔记（第 14 轮）

> 这份笔记记录第 14 轮做了什么、联网核对到了什么、**没能核对什么**。沙箱是 Linux，没有 Windows、没有 NVIDIA 显卡、没有 AutoDL 实例，所以凡是要真机才能确认的事都写在第 3 节，由你按 `docs/PENDING_USER_ACTIONS.md` 去做。
> 检查日期 2026-10-10。决策编号 D-500 起见 `docs/DECISIONS.md`。

## 1. 联网核对到的事实

### llama.cpp（固定在 b11177）

| 事项 | 结果 |
| --- | --- |
| 为什么是 b11177 | 第 13 轮的 `export.sh` 用这个标签的转换器和量化器（`versions.env` 的 `TWIN_LLAMA_CPP_TAG`）；本机运行的 `llama-server` 与转换器同一个标签，避免格式对不上 |
| Windows 发行包 | `llama-b11177-bin-win-cpu-x64.zip`、`llama-b11177-bin-win-cuda-12.4-x64.zip`、`llama-b11177-bin-win-cuda-13.4-x64.zip`，以及与 CUDA 构建配套的运行库包 `cudart-llama-bin-win-cuda-12.4-x64.zip`、`cudart-llama-bin-win-cuda-13.4-x64.zip`（运行库包的名字里没有标签） |
| 哈希和大小 | 已按 2026-10-10 下载到的资产算出，写在 `scripts/windows/llamacpp.lock.json`（五个文件各自的 sha256 与字节数）。`get_llamacpp.ps1` 没有这个文件就拒绝运行，下载后逐个比对；发行 API 若同时给出 digest，则两者必须一致 |
| 最新标签 | 查到的最新标签是 b11539，本轮用到的参数和接口在两个标签的 `tools/server/README.md` 里相同 |
| 启动参数 | `-m <gguf> --host 127.0.0.1 --port 8081 -c 4096 -ngl 999 --parallel 1 --chat-template chatml --no-webui`。`--chat-template chatml` 只是让服务端启动时不去解析模型自带的 Jinja 模板，我们只用原始的 `/completion`，服务端的聊天模板不起作用 |
| `/health` | 模型还在加载时返回 503（`Loading model`），好了返回 200 `{"status":"ok"}`；`StyleHealth.loading` 据此区分“加载中”和“挂了” |
| `/props` | 返回 `model_path`（与 `-m` 相同），“同一模型文件的服务已在运行”的判断据此 |
| `/tokenize` | 参数 `content`、`add_special`、`parse_special`，返回 `tokens`。**`/completion` 对字符串提示词按 `add_special=true`、`parse_special=true` 分词**（读 b11177 与 b11539 的 `tools/server/server-context.cpp`），所以 GGUF 若声明 `tokenizer.ggml.add_bos_token` 就会在最前面多一个 BOS。分词核对因此也用这两个参数问 `/tokenize`（偏差 D-503） |
| Windows 包的文件 | 基础：`llama-server.exe`、`llama-server-impl.dll`、`llama.dll`、`ggml.dll`、`ggml-base.dll`、`llama-common.dll`；CPU 后端 `ggml-cpu*.dll`；CUDA 后端 `ggml-cuda.dll`，并且要有同一 CUDA 大版本的 `cudart64_*.dll`、`cublas64_*.dll`、`cublasLt64_*.dll`（CUDA 12 与 13 的文件名不同）。只有一半的目录能启动，但第一次请求就失败，所以 `twin doctor` 逐个检查 |
| 构建选择 | 驱动支持 CUDA ≥ 13.0 且算力 ≥ 7.5：`cuda-13.4`（对 RTX 50 系有原生内核）；否则驱动支持 CUDA ≥ 12.4：`cuda-12.4`；否则 `cpu`。`hardware.choose_build` 与脚本写了同一条规则，测试逐项比对数字 |
| Qwen3 的 KV 缓存 | 读了各模型 Hugging Face 上的 `config.json`：8B 36 层 × 8 个 KV 头 × 128 = 147,456 字节/词元（16 位）；14B 40 层，163,840；32B 64 层，262,144。量化推荐用它估上下文的显存 |

### vLLM（远程，固定 0.26.0）

| 事项 | 结果 |
| --- | --- |
| 发行包里有哪些 wheel | GitHub 发行页 `v0.26.0` 里 `vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` 能下载（HTTP 200）；`+cu128`、`+cu130` 都是 404 |
| 默认（PyPI）wheel | `vllm-0.26.0-cp38-abi3-manylinux_2_28_x86_64.whl`（303,698,761 字节）要求 `torch==2.11.0` 和 `nvidia-cutlass-dsl[cu13]`，是 **CUDA 13 构建**；训练环境固定在 CUDA 12.8（`versions.env`），AutoDL 的驱动不一定支持 13，所以不用它 |
| `+cu129` wheel 的依赖 | 用 HTTP 范围请求读了这个 wheel（518,935,080 字节）里的 `METADATA`：要求 `torch==2.11.0`、`torchvision==0.26.0`、`torchaudio==2.11.0`、`nvidia-cutlass-dsl==4.6.0`（没有 `[cu13]`）、`humming-kernels[cu12]==0.1.10`、`flashinfer-python==0.6.14`、`transformers>=5.5.3`。vLLM 住在自己的虚拟环境里，所以 `transformers` 5.x 不影响训练环境固定的 4.57.6 |
| 对应的 torch | `download.pytorch.org/whl/cu129` 上有 `torch-2.11.0+cu129`、`torchvision-0.26.0+cu129`、`torchaudio-2.11.0+cu129` 的 `cp312-cp312-manylinux_2_28_x86_64.whl`。`serve_vllm.sh` 因此改为先从 `cu129` 索引装这三个，再装 `+cu129` 的 vLLM wheel，去掉原来错的 `+cu128` 文件名和回退到 PyPI 的那一步（偏差 D-512） |
| 接口 | `POST /v1/completions`（`model` 是 LoRA 名 `twin-style`、`prompt`、`max_tokens`、`stop`、`add_special_tokens`）、`GET /health`、`POST /tokenize`（`prompt`、`add_special_tokens`，返回 `tokens`、`count`、`max_model_len`） |
| CUDA 12.9 对 Blackwell | CUDA 12.8 起支持 `sm_120`，12.9 同样；torch 2.11.0+cu129 的 `get_arch_list()` 应含 `sm_120`，`serve_vllm.sh` 第一次启动会打印出来 |

## 2. 做了什么

* **本机推理**：`get_llamacpp.ps1`（PowerShell 5.1、StrictMode、遇错即停；检测显卡选构建；下载、核对、解到 `tools/llama.cpp/<标签>/`，先解到临时目录再移动，写 `build.json`）；`twin model recommend`（`--vram`、`--cpu`）；`twin model serve`（前台，Ctrl+C 结束）；应用里的 `StyleServingComponent` 在需要时自己启动、监视、重启 `llama-server`，随主进程结束（Windows 作业对象）。
* **远程推理**：`twin model tunnel start|stop|status`；应用里同一个组件维持隧道，断线按 2 秒起翻倍到 60 秒重连，登录被拒或主机密钥变了不重试（等你修好再 `retry`），别的进程已经有一条健康的隧道就只看着它。
* **分词核对**：每次服务器起来都比对（7 个固定的合成提示词，逐个词元比），不一致就停服务、报 `style_tokenize_mismatch`、写出第一个差异的位置；结果记在登记表里，选择器拒绝上一次核对没通过的模型。
* **激活与门槛**：`twin model activate <id> [--force] [--backend style|hybrid]`、`disable`；M5 判定器；`--force` 记审计、`gate_passed=false`、`/状态` 标注、不参与预算第 ④ 级。
* **评估**：`twin model evaluate <id>`：同一批新的留出集上下文，经 deepseek、style、hybrid 三种方式生成，盲测，再按方式算风格指标。
* **回退**：沿用第 09 轮的 `BackendSelector`，新增“加载中不算故障”和“分词没通过不能用”两个状态。

## 3. 没能核对、需要你在真机上做的事

1. **`llama-server.exe` 真的在你的显卡上加载 `export.sh` 做出的 GGUF**：沙箱里只有模拟服务器（`tests/support/llama_server_sim.py`，同样的命令行、同样的 HTTP）。运行 `scripts\windows\get_llamacpp.ps1`，再 `twin model serve`，看分词核对和预热。
2. **速度**：首词元延迟、每秒词元数、三个量化档在你的卡上的实际显存占用——`twin model serve` 的预热会写进 `/状态`，真实数字还是空的。
3. **GGUF 是否带 BOS**：`twin model verify <id>` 会告诉你；若有差异，报告会指出第一个差异的位置和两边的词元。
4. **Windows 作业对象**：`tests/unit/test_serving_windows.py`（`windows` 标记）在 Windows 上才跑；沙箱里没有跑过。
5. **vLLM 0.26.0 + torch 2.11.0+cu129 在 RTX 5090 / RTX PRO 6000 上**能不能带 LoRA 和 bitsandbytes 4 位量化起来：第一次 `serve_vllm.sh` 会打印实际版本；起不来时先看 `vllm serve` 的报错，不要换版本号凑合。
6. **AutoDL 的 sshd 是否允许端口转发**，以及隧道的延迟：隧道测试用本地 SSH 服务器，行为是 asyncssh 的 `forward_local_port`；真实实例要你试 `twin model tunnel start` 和 `twin model tunnel status`。
7. **按小时计费的提醒**：每天一次的系统消息在 `commands.morning_hour` 之后发，条件和内容有测试，但没有在真实实例上看过。
8. **集成测试的真实 GGUF 版本**（`tests/integration/test_style_serving_e2e.py::test_ten_turns_through_the_hybrid_backend_with_a_real_gguf`）：设置环境变量 `TWIN_LLAMA_SERVER` 和 `TWIN_LLAMA_GGUF` 后才会跑，结果（首词元延迟中位数、违规率）写进测试目录的 `serving_report.json` 与测试报告属性。

## 4. 在哪里看数字

| 数字 | 在哪里 |
| --- | --- |
| 预热的首词元延迟、速度 | `/状态` 的“风格模型速度”一行；`twin model show <id>` 的 `eval.warmup` |
| 分词核对的结果 | `twin model show <id>` 的 `eval.tokenize_check`；`twin model verify <id>` |
| 激活记录 | `twin model show <id>` 的 `eval.activation_log` |
| 门槛的判定 | `twin eval gate M5` 与 `twin eval runs --kind gate` |
| 隧道状态、重连次数、实例已运行多久 | `twin model tunnel status`；`/状态` |
| 服务器日志 | `data/logs/llama-server.log`（启动时轮转）；评估用的服务器是 `llama-server-eval.log` |
