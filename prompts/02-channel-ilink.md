# 第 02 轮：微信通道（openclaw-weixin / iLink 协议）、本地控制台通道、M0 通道实测

> 里程碑：M0 · 前置：第 00、01 轮全绿 · 需要用户：手机微信升级到支持 ClawBot 的版本（iOS ≥ 8.0.70 / Android ≥ 8.0.69，"我 → 设置 → 插件"里有 ClawBot），在本机安装 Node.js（用于 `npm pack` 获取官方源码），扫码登录，确认绑定，启动实测探针。

## 先读
`CLAUDE.md`；`docs/SPEC.md` §5 通道全部、R-PRO-003、R-OPS-004、R-PRIV-006。

## 本轮目标
实现一个可长期运行、可恢复、只对用户说话的微信 ClawBot 通道，并用实测数据回答 M0 的问题：主动发消息的窗口有多长、无回复时能连发几条、图片与 GIF 能否发送且是否会动、正在输入与引用是否可用。同时实现功能完整的本地控制台通道。

## 必须实现的需求
R-CH-001～011、R-SCOPE-001、R-ARCH-005、R-OPS-004（登录失效告警与重新登录流程的通道侧部分）、R-PRIV-006、R-SAFE-006（通道侧媒体白名单）。

## 第一步：以官方源码为准写协议文档（先做，做完给我看）
1. 在临时目录执行 `npm pack @tencent-weixin/openclaw-weixin` 与 `npm pack @tencent-weixin/openclaw-weixin-cli`，解包阅读源码；同时阅读 AstrBot 仓库中个人微信适配器（`weixin_oc`）的 Python 实现作对照。不要把第三方源码复制进本仓库，只记录协议事实并标注出处（包名、版本、文件路径、行号）。
2. 输出 `docs/ILINK_PROTOCOL.md`：主机与全部接口（二维码登录、扫码状态、长轮询 getupdates、sendmessage、sendtyping、getconfig、媒体上传/下载）的请求与响应 JSON 结构、鉴权方式与各类 token（bot token、context_token、游标如 `get_updates_buf`）的来源与生命周期、消息项类型（文字、图片、语音、文件、视频、引用）的结构、媒体加密方式（AES-128-ECB 的密钥来源与填充）、错误码含义（尤其会话过期，社区报告为 `ret:-2`）、已知限制。
3. 若源码显示某能力不存在（例如不支持发送引用），在文档中明确写出，并在实现中把对应 `capabilities()` 置为不支持——不得伪造该能力。

**协议文档完成后暂停，等我确认再进入实现。**

## 详细要求

### A. 接口与数据结构（R-CH-001）
- `Channel` 抽象、`InboundMessage`、`OutboundResult`（成功/失败、协议错误码、是否会话过期）、`ChannelCapabilities`（supports_quote、supports_typing、gif_animated（来自实测，未知时为 None）、proactive_window_h、outbound_quota）。

### B. IlinkChannel（R-CH-002～008）
1. 登录：`twin channel login` 获取二维码 → 保存 PNG 到受控临时目录并用默认查看器打开，同时在终端打印字符二维码 → 轮询扫码状态 → 保存登录凭据（keyring 或加密 `channel_state`）。二维码过期自动刷新。
2. 长轮询：后台任务持续 getupdates；游标持久化；按消息 id 去重（`channel_state` 中保存最近 N 个已处理 id）；网络错误退避 1s→60s 带抖动；协议明确的鉴权失效 → 进入"需要重新登录"状态并触发告警钩子（第 12 轮接邮件/Windows 通知；本轮告警写入 `alerts` 表并在控制台醒目输出）。
3. 入站解析（R-CH-005）：文字；图片下载并按协议解密后存入 `MediaStore`；语音取微信转写文字（无则标记未转写）；视频取封面图；文件取文件名；引用取被引用内容；输出 `InboundMessage`。
4. 出站（R-CH-006）：`send_text`、`send_image`（上传 + 必要的 context_token 规则）、`send_typing`、`send_text(quote=...)`（仅当协议支持）。每次出站更新会话窗口计数。
   - `send_image` 的媒体白名单（R-SAFE-006 的通道侧）：只接受 sha256 属于表情包库（`stickers` 表，第 03 轮建表后接入；本轮先定义 `OutboundMediaPolicy` 接口）或本轮探针程序生成的测试图（生成时登记到 `channel_state` 的探针图片清单）的字节；其他一律抛 `MediaNotAllowed`。
5. 收件人绑定（R-CH-007）：未绑定时，第一条入站消息的发送者 id 在 CLI 中显示（打码）并要求我输入确认；确认后写入 `channel_state`。之后：其他发送者的入站消息丢弃并记录（不含内容）；任何出站 API 的收件人不是绑定用户时抛 `RecipientNotAllowed`。提供 `twin channel unbind`（需二次确认）。
6. 会话窗口（R-CH-008）：`SessionWindow` 记录 `last_inbound_at`、`outbound_since_inbound`、`expired`；`remaining_quota()`、`can_send_proactive(now, n)`（窗口内且剩余配额 ≥ n）按配置的安全阈值判断；收到会话过期错误立即 `expired=True`，下一条入站时复位；对过期错误绝不循环重试。

### C. 实测探针（R-CH-009/010）
1. `twin channel probe start|status|stop|report`：探针是持久化的计划（存数据库），由应用中的 `ChannelProbe` 组件执行，重启后继续；所有探针消息以"[测试]"开头。
2. 测量项（按 SPEC R-CH-009 的顺序执行，探针状态机持久化每一步；**每一步开始前都在终端与微信里提示我先发一条新消息**，收到后才开始，使窗口与计数复位）：
   - ① 条数：连续主动发送（间隔 2 分钟），直到首次失败或达到 15 条，记录成功条数 N 与失败错误码。
   - ② 媒体与交互：分别发送程序生成的 JPG、PNG 和一个多帧 GIF，并在 CLI 中询问我"GIF 在手机上是否在动"；发送 typing 后询问我是否看到"对方正在输入"；若协议支持，发送一条引用消息并让我确认显示效果。本步总条数不超过 N−1，不够时拆成两次（中间再提示我发一条消息）。
   - ③ 窗口：开始前明确提醒"接下来约 25 小时请不要给机器人发任何消息，否则窗口测试作废"；探针在该入站后 1、6、12、20、23、25 小时各尝试发一条文字（总条数不超过 N−1，不够时从后往前保留测量点并在报告中说明），记录成功/失败与错误码；任一时刻失败后停止该项；期间若收到我的入站，标记本步作废并提示重做。
   - ④ 配额是否共用：根据源码（`context_token` 的使用方式）与①的结果，在报告中写明回复与主动出站是否消耗同一配额。
   - 探针发送使用 `ProbeSendPolicy`：绕过 `proactive_window_safe_h` 与 `outbound_quota_safe` 两个安全阈值（否则测不到边界），但只在探针计划激活期间有效、只允许带"[测试]"前缀的消息、每次绕过写审计；遇到协议失败（含会话过期）立即停止当前步骤，绝不重试；正常的引擎与主动消息路径不能使用它（有测试）。
3. `report` 生成 `docs/CHANNEL_REPORT.md`（只含测量结果与时间），并把实测的窗口与条数写回 `channel.proactive_window_safe_h`、`channel.outbound_quota_safe` 的建议值（留 10% 余量），需我确认后才改配置。
4. R-CH-010：若窗口 < 12 小时或连发条数 < 3（不足以支撑 R-PRO 的作息型与追发需求），报告中写"未达标"，并在汇报中明确停下，等待我决定（企业微信通道不在本轮范围）。

### D. LocalConsoleChannel（R-CH-011）
- 终端里我输入、机器人输出；支持 `/img <路径>` 发送图片；机器人发表情包时显示"[表情包：标签] <文件路径>"；显示"对方正在输入…"；可配置模拟窗口与条数限制（用于测试主动逻辑）；同样受收件人绑定约束（本地用户固定 id）。
- `twin chat --local` 启动只连本地通道的应用（引擎在第 09 轮接入；本轮可以把入站消息原样回显用于验证通道本身——回显逻辑放在 `tests/` 或作为 `twin channel echo-test` 诊断命令，不能成为生产回复路径）。

### E. 诊断命令
`twin channel status`（登录状态、绑定情况、窗口剩余、最近错误）、`twin channel send-test "文字"`（只允许发给绑定用户，内容自动加"[测试]"前缀）。

## 测试要求
- 依据 `docs/ILINK_PROTOCOL.md` 构造合成的协议响应（respx），覆盖：登录全流程与二维码过期、长轮询游标持久化与重启续传、重复消息去重、各类入站解析、媒体解密（用协议同款算法加密合成图片再解密比对）、出站成功/失败/会话过期、typing、引用能力开关。
- 收件人绑定：未确认前不处理；其他发送者被丢弃；向非绑定用户发送抛异常。
- 会话窗口：time-machine 模拟 0–26 小时与条数边界；`remaining_quota()`；过期后不重试、入站后复位。
- 媒体白名单：非表情包、非探针测试图的字节被拒绝。
- 探针：步骤顺序与每步前的新入站要求、计划持久化与重启续跑、窗口测试期间收到入站即作废、总条数不超过 N−1、`ProbeSendPolicy` 只在探针期间对"[测试]"消息生效且写审计、报告生成、阈值写回需确认。
- 本地通道：交互流程（用伪终端或注入输入流）。

## 验收
```
uv run pytest -q
uv run twin channel login            # 我扫码
uv run twin channel send-test "你好"  # 我在手机上确认收到
uv run twin channel probe start      # 运行约 26 小时
uv run twin channel probe report     # 生成 docs/CHANNEL_REPORT.md
uv run python scripts/trace_check.py --round 02
```

## 不要做
- 不要用任何模拟按键、读取微信客户端内存或数据库、PC 微信自动化等非官方手段。
- 不要给除绑定用户以外的任何人发消息，包括测试。
- 不要把第三方源码复制进仓库。

## 完成后汇报
按 CLAUDE.md 格式；附 `docs/ILINK_PROTOCOL.md` 摘要与（若已完成）`docs/CHANNEL_REPORT.md` 的结论；明确 M0 通道部分是否达标。
