# iLink（openclaw-weixin）协议文档

> 第 02 轮 · 步骤 02a。以**官方源码**为准整理的协议事实，供第 02 轮后续步骤（IlinkChannel、探针、测试）实现与核对。
> 本文只记录**协议事实**与出处，不复制第三方源码；样例中的 token、id、URL 全部是占位值。
> 沙箱里没有微信账号，因此**所有"源码读不出来"的行为都标为"需 M0 实测确认"**，不做推测性断言。

## 0. 来源、证据等级与引用约定

### 0.1 取得的来源

| 代号 | 来源 | 版本 / 取得方式 | 校验 | 说明 |
| --- | --- | --- | --- | --- |
| **W** | npm `@tencent-weixin/openclaw-weixin`（腾讯官方，MIT） | **2.4.9**（`latest`，npm 修改时间 2026-09-17）；`npm pack` 到临时目录后只读解包，**未运行其中任何脚本、未安装依赖** | tarball sha256 `467e8047f7114e45944961fcd3eda9421843c9c65db61ea24176e252ab800ee4` | 包内含 TypeScript 源码 `src/**`（带行号引用）与编译产物 `dist/**`，本文引用 `src/**` |
| **W-CLI** | npm `@tencent-weixin/openclaw-weixin-cli` | **2.1.4**（`latest`） | sha256 `ac63cc9a5ae73f149b5314882e6f0227682b850b1e04e55b36b48931303ce28e` | 只是安装器（调用 `openclaw plugins install` 与 `openclaw channels login`），**不含协议实现**，仅记入兼容矩阵 |
| **P** | 腾讯官方仓库 `Tencent/openclaw-weixin` 的 `docs/protocol.md`（`main` 分支，2026-10-09 经 raw 域名取回，490 行） | 示例版本号写 2.4.8 | sha256 `8d778054436f6a0d93b4b38abf2dfd09c16e4a961bed044634cac2d3b8f155dd` | 官方给"兼容后端实现者"的协议说明，**本身声明"客户端类型不等于完整服务端契约"**；不在 npm 包里 |
| **A** | AstrBot **4.28.2**（PyPI wheel，`pip download astrbot --no-deps`）中的个人微信适配器 `astrbot/core/platform/sources/weixin_oc/{weixin_oc_adapter,weixin_oc_client,weixin_oc_event,login_registration}.py` | `api.github.com` 与 `codeload.github.com` 在沙箱里返回 403；`raw.githubusercontent.com`（`master` 分支同名文件可取到）与 PyPI 可达，选用 PyPI wheel 以便固定版本 | wheel sha256 `2d3b5db74da5b5c1aa7e1b5990b3ab4cb5569bc8e6bb331b54ce3d154b1b3530` | **第二来源**（Python，独立实现，与 W 互相印证） |
| **C** | 社区第三方文档（见 §9.2 列表） | 2026-10-09 经网页检索取得 | — | **未经验证、互相矛盾**，只作线索，**不作为实现依据** |

两个官方来源（W、P）和一个独立实现（A）在接口路径、请求/响应字段、加密方式上**完全一致**；下文不一致之处会明确指出。

### 0.2 证据标记与出处格式

- **【源码】** W 的源码直接证明；**【对照】** A 也独立证明；**【文档】** P 证明；**【社区】** 仅 C 声称（未验证）；**【M0】** 需用户在真机上实测确认（R-CH-009）；**【推断】** 由源码行为合理推出，但未被直接证明。
- 出处写法：`W:src/api/api.ts:211`＝W 包内 `src/api/api.ts` 第 211 行；`A:weixin_oc_adapter.py:886-893`；`P:protocol.md:48`。W 的 `dist/` 与 `src/` 一一对应（CHANGELOG 2.4.1），不另引。
- JSON 里 `<...>` 一律是占位值。

---

## 1. 结论速览（先看这里）

1. **主机**：API `https://ilinkai.weixin.qq.com`；CDN `https://novac2c.cdn.weixin.qq.com/c2c`（W:src/auth/accounts.ts:12-13）。登录成功响应里的 `baseurl` 可能改写 API 主机，需持久化（W:src/auth/accounts.ts:395）。
2. **凭据链**：二维码登录 → `bot_token`（Bearer）+ `ilink_bot_id` + `ilink_user_id`（扫码者，即用户自己）+ `baseurl`。之后所有 Bot 接口带 `Authorization: Bearer <bot_token>`。
3. **收消息**：`POST /ilink/bot/getupdates` 长轮询，游标 `get_updates_buf` 由服务端给、客户端原样回传；默认客户端超时 35 s，服务端可通过 `longpolling_timeout_ms` 调整。
4. **发消息**：`POST /ilink/bot/sendmessage`，**每个请求只放 1 个 item**；必须把"最近一条入站消息的 `context_token`"原样带上。官方客户端把 token 缓存并复用（含非回复场景的定时投递），**源码里没有任何条数/时长限制逻辑**。
5. **错误码**：**源码里不存在 `ret:-2`**。源码唯一处理的凭据/会话类错误是 **`-14`（getupdates 的 `ret` 或 `errcode`）**，官方 2.4.6 起明确命名为"bot token 失效"，**不是会话过期**。`-2` 只出现在社区文档中，且各家解释互相矛盾（见 §9）。
6. **媒体**：AES-128-ECB + PKCS7；密钥由客户端随机生成（上传）或来自消息体（下载）；整文件一次性加密、一次 POST 上传，**无分块**。上传前必须调用 `getuploadurl`（SPEC 清单里漏了这个接口）。
7. **能力**：**发送引用 = 源码不支持**（`ref_msg` 只在入站解析，所有发送构造器都不设）；**发送正在输入 = 支持**（`getconfig` 取 `typing_ticket` → `sendtyping`）；**GIF 与普通图片走同一条 IMAGE 通道、字节原样上传**，是否会动取决于服务端/手机，源码读不出来 → 【M0】。
8. **主动发送窗口真实长度、单次入站后可发条数、回复与主动是否共用配额、`-2` 的真实含义**：源码无任何线索 → 全部【M0】。社区另有"媒体报道称 ClawBot 不支持 AI 主动推送"（§9.2），与 R-CH-010 的停机条件直接相关，**M0 必须用真机回答**。

---

## 2. 传输、请求头、`base_info`

### 2.1 传输

| 项 | 值 | 出处 |
| --- | --- | --- |
| 协议 | HTTPS + JSON；字节字段为 base64 字符串 | W:src/api/types.ts:1-4；P:protocol.md:23-26 |
| 方法 | Bot 接口一律 `POST`；扫码状态查询是 `GET`；`get_bot_qrcode` 在官方 2.4.9 是 `POST`（A 用 `GET`，旧版） | W:src/auth/login-qr.ts:99-104,139；P:protocol.md:91；A:weixin_oc_adapter.py:1062-1068 |
| URL 拼接 | `new URL(endpoint, baseUrl + "/")`，endpoint 不带前导斜杠 | W:src/api/api.ts:217-219,318-319,398-399 |
| 64 位 id | `message_id` / `msg_id` / `svr_id` 在线上是 uint64 数字；官方先把它们加引号再解析以免精度丢失。**Python 的 `json` 原生支持大整数，直接 `str(value)` 即可** | W:src/api/api.ts:518-577 |
| HTTP 状态 | 非 2xx 一律当异常（消息体带上）；业务失败走 `ret/errcode`，随 HTTP 200 返回 | W:src/api/api.ts:422-424；P:protocol.md:82 |

### 2.2 请求头

Bot 接口（POST，需鉴权）的头（W:src/api/api.ts:240-254，P:protocol.md:34-48）：

| 头 | 值 | 备注 |
| --- | --- | --- |
| `Content-Type` | `application/json` | 不要手动设置 `Content-Length`（Node 24 报错，CHANGELOG 2.4.2，W:CHANGELOG.md:78；httpx 自动处理） |
| `AuthorizationType` | `ilink_bot_token` | 固定值 |
| `Authorization` | `Bearer <bot_token>` | token 非空才加 |
| `X-WECHAT-UIN` | `base64(str(随机uint32))` | **每个请求重新生成**（W:src/api/api.ts:222-225；A:weixin_oc_client.py:48-50 同） |
| `iLink-App-Id` | `bot` | 取自 W 的 `package.json` 顶层 `ilink_appid`（W:package.json:73；src/api/api.ts:92） |
| `iLink-App-ClientVersion` | `(major<<16)\|(minor<<8)\|patch` 的十进制字符串；W 2.4.9 → **`132105`** | W:src/api/api.ts:99-107 |
| `SKRouteTag` | 部署方配置的路由标签，可选 | 我们不使用 |

- `get_qrcode_status`（GET）只带 `iLink-App-Id`、`iLink-App-ClientVersion`（及可选 `SKRouteTag`），不带 `Authorization*`、`X-WECHAT-UIN`（W:src/api/api.ts:228-238,320；P:protocol.md:48）。
- `get_bot_qrcode`（POST）带 JSON 头（含 `AuthorizationType`、`X-WECHAT-UIN`），**无 `Authorization`，无 `base_info`**（W:src/auth/login-qr.ts:99-104；P:protocol.md:48）。
- **对照**：A 只发 `Content-Type / AuthorizationType / X-WECHAT-UIN / Authorization` 四个头，且只在二维码状态查询里发 `iLink-App-ClientVersion: 1`，**不发 `iLink-App-Id`**，仍然工作（A:weixin_oc_client.py:44-54；weixin_oc_adapter.py:1119）。说明 `iLink-App-*` 头**可能不是必需**，但官方都带。**建议**：照官方发全（`iLink-App-Id: bot`、`iLink-App-ClientVersion: 132105`），最大化兼容；"是否可省"【M0】。

### 2.3 `base_info`

所有鉴权 POST（除 `get_bot_qrcode`）的 JSON 体里都带（W:src/api/api.ts:203-208；P:protocol.md:50-65）：

```json
{ "base_info": { "channel_version": "<版本字符串>", "bot_agent": "<UA 风格标识>" } }
```

- `channel_version`：W 填插件版本（如 `2.4.9`）；A 填字面量 `"astrbot"`（A:weixin_oc_adapter.py:899）——**服务端接受任意字符串**。
- `bot_agent`：自报家门，类似 `User-Agent`，**仅用于观测，不参与鉴权/路由**；语法 `Name/Version`（可带 ` (comment)`，多个用空格分隔），ASCII，总长 ≤ 256 字节，非法片段被丢弃，空则回落到 `"OpenClaw"`（W:src/api/types.ts:9-21；src/api/api.ts:113-200；W:README.md:87-121）。
- **建议**：`bot_agent` 如实写 `WechatTwin/<版本>`；`channel_version` 写同一版本字符串。不要冒充 OpenClaw。

### 2.4 超时（客户端侧默认值）

| 请求 | 官方客户端超时 | 出处 | 对照 A |
| --- | --- | --- | --- |
| `getupdates`（长轮询） | **35 000 ms**；之后改用响应里的 `longpolling_timeout_ms`；**超时视为"无新消息"，返回空结果继续轮询** | W:src/api/api.ts:211,456,475-484；src/monitor/monitor.ts:13,107-110 | 35 000 ms（A:weixin_oc_adapter.py:135） |
| `sendmessage`、`getuploadurl` | 15 000 ms | W:src/api/api.ts:213,511,588 | 全部 API 120 000 ms（A:138） |
| `getconfig`、`sendtyping`、`notifystart/stop` | 10 000 ms | W:src/api/api.ts:215,611,627,643,658 | 同上 |
| `get_bot_qrcode` | **无客户端超时**（曾为 5 s→10 s，后移除） | W-CHANGELOG:104,143,164 | 15 000 ms（A:1067） |
| `get_qrcode_status`（长轮询） | 35 000 ms；超时/网关错误（如 524）**一律当 `wait`** | W:src/auth/login-qr.ts:33,128-158 | 35 000 ms（A:1118） |
| CDN 上传 / 下载 | 官方未设超时（用 `fetch` 默认）；上传最多重试 3 次 | W:src/cdn/cdn-upload.ts:7,40-83 | 同 `api_timeout_ms`（A:weixin_oc_client.py:147,183） |

---

## 3. 响应约定与错误码

### 3.1 通用约定

- 响应体可能带 `ret`（number）、`errcode`（number）、`errmsg`（string）；**不是每个接口都有**（W:src/api/types.ts:220-232,239-244,260-271；P:protocol.md:67-84）。
- **成功**＝`ret` 缺失或为 0（且 `errcode` 缺失或为 0）。`sendmessage` 成功时响应体可能就是 `{}`（W-CHANGELOG:58 提到成功用例的响应为 `"{}"`；A 用 `payload.get("ret", 0)`，A:weixin_oc_adapter.py:960-964）。
- 各接口官方客户端对业务码的处理并不统一（P:protocol.md:71-80）：

| 接口 | 官方客户端怎么判 | 出处 |
| --- | --- | --- |
| `getupdates` | `ret≠0` 或 `errcode≠0` 即失败；任一字段 `== -14` → 暂停该账号所有请求 1 小时 | W:src/monitor/monitor.ts:111-127；src/api/session-guard.ts:3-17 |
| `sendmessage` | `ret` 存在且 ≠0 → 抛错（`sendMessage ret=… errmsg=…`）；`ret` 缺失不报错 | W:src/api/api.ts:592-594 |
| `getconfig` | 仅 `ret === 0` 才接受结果，否则按退避重试 | W:src/api/config-cache.ts:45 |
| `sendtyping` | **不解析响应**，失败只靠 HTTP 异常 | W:src/api/api.ts:619-630（A 则检查 ret/errcode：A:weixin_oc_adapter.py:299-303） |
| `getuploadurl` | 不检查 `ret`，只要求 `upload_full_url` 或 `upload_param` 之一非空 | W:src/cdn/upload.ts:95-102 |
| `notifystart/stop` | `ret≠0` 或异常仅记警告，不阻塞 | W:src/channel.ts:519-529,566-576 |

### 3.2 错误码表

| 码 | 出现位置 | 含义（源码依据） | 官方行为 | A 的行为 |
| --- | --- | --- | --- | --- |
| `0`（或缺失） | 全部 | 成功 | — | — |
| **`-14`** | `getupdates` 的 `ret` 或 `errcode`（类型注释："e.g. -14 = session timeout"） | **bot token 失效**。官方 2.4.6 把常量 `SESSION_EXPIRED_ERRCODE` 改名 `STALE_TOKEN_ERRCODE`，并写明"-14 表示 token 失效，而非 session 过期" | `pauseSession`：暂停该账号所有收发 **1 小时**，期间 `assertSessionActive` 对出站直接抛错；1 小时后用同一 token 继续试 | 只看 `errcode`；清空 token、游标、context_token，回到二维码登录（A:weixin_oc_adapter.py:104,977-989,1599-1601） |
| **`-2`** | **源码中不存在** | 只见于社区文档，含义互相矛盾（§9.2） | — | — |
| 其它非 0 | 各接口 | 源码无枚举 | `getupdates`：连续 3 次失败后退避 30 s，否则 2 s 重试；`sendmessage`：抛错 | `getupdates`：睡 5 s（A:1602） |

出处：W:src/api/types.ts:222；src/api/session-guard.ts:3-6；W-CHANGELOG:42；src/monitor/monitor.ts:111-149。

**结论**：SPEC/提示词里"会话过期，社区报告为 `ret:-2`"在源码中**没有对应物**。`-14` 是**凭据**层面的失效，应走"需要重新登录 + 告警"（R-OPS-004）；"主动发送窗口过期"这一概念在源码中根本不存在，其服务端表现（很可能是 `sendmessage` 返回非 0 的 `ret`，社区称 `-2`/`prepare failed`）只能靠 M0 实测。见 §10、§12 D-007。

### 3.3 官方的重试 / 退避（仅供参考，SPEC 另有更严格要求）

- 长轮询循环：异常或业务失败计数；连续 3 次 → 睡 30 s 并清零，否则睡 2 s（W:src/monitor/monitor.ts:14-16,129-149,184-206）。
- `getconfig`：失败时退避 2 s 起、翻倍、封顶 1 h；成功后缓存并在 24 h 内**随机时刻**刷新（W:src/api/config-cache.ts:8-10,49,60-74）。
- CDN 上传：4xx 立刻放弃，其它失败最多总共 3 次（W:src/cdn/cdn-upload.ts:47-83）。
- 对 `sendmessage` **没有任何自动重试**（W:src/messaging/send.ts:120-136）。

---

## 4. 鉴权与 token 生命周期

| 名称 | 来源 | 用途 | 生命周期（源码依据） | 持久化（官方） |
| --- | --- | --- | --- | --- |
| **`qrcode`** | `get_bot_qrcode` 响应 | 标识一次扫码登录；作为 `get_qrcode_status?qrcode=` 参数 | 5 分钟（客户端 `ACTIVE_LOGIN_TTL_MS`，A 同为 5 分钟）；过期后服务端返回 `expired`，客户端换新码 | 内存 |
| **`qrcode_img_content`** | 同上 | **要编码成二维码图案的内容（一个 URL）**，不是图片：官方交给 `qrcode-terminal` 生成，A 交给 `qrcode` 库生成 | 同 `qrcode` | 内存 |
| **`bot_token`** | `get_qrcode_status` 的 `confirmed` 响应 | Bearer 令牌，所有 Bot 接口的鉴权 | 直到服务端返回 `-14`；真实寿命**源码未给出**【M0 长期观察】 | `accounts/<id>.json`，权限 0600（W:src/auth/accounts.ts:189-218）；A 写配置 |
| **`ilink_bot_id`** | 同上 | 机器人账号 id，形如 `<hex>@im.bot`；官方用作账号 id（文件名里把 `@`、`.` 规整成 `-`） | 与 token 绑定 | 同上 |
| **`ilink_user_id`** | 同上 | **扫码用户的 id**，形如 `<hex>@im.wechat`（W:src/channel.ts:243 注释"用户 id 都以 `@im.wechat` 结尾"）；官方将它作为授权名单兜底（"add to allowFrom"） | 稳定 | 同上（`userId`） |
| **`baseurl`** | 同上 | 后续 API 主机；缺省回落默认主机 | — | 同上（`baseUrl`） |
| **`context_token`** | **每条入站消息**里的 `context_token` | 发送时必须原样回传（官方注释"issued per-message … must be echoed verbatim in every outbound send"）；`getconfig` 也可带 | 每来一条入站就覆盖；官方**按（账号, 用户）缓存最近一个并重复使用**，包括非回复的定时投递；**服务端有效期源码未给出**【M0】 | 内存 + `accounts/<id>.context-tokens.json`，重启后恢复（W:src/messaging/inbound.ts:17-21,41-81,101-116；src/channel.ts:270-280） |
| **`get_updates_buf`** | `getupdates` 响应 | 轮询游标；首次或重置时发 `""` | 仅当响应里非空才更新；**官方先落盘游标再处理消息**（见 §10.5 对此的改进） | `accounts/<id>.sync.json`（W:src/monitor/monitor.ts:152-156；src/storage/sync-buf.ts:77-80） |
| **`typing_ticket`** | `getconfig` 响应 | `sendtyping` 的凭证 | 官方每用户缓存，24 h 内随机时刻刷新；A 每 60 s 或 context_token 变化时重取 | 内存 |
| `filekey` | 客户端生成（16 随机字节的 hex） | 一次上传的标识，进 `getuploadurl` 与上传 URL | 一次性 | — |
| `aeskey` | 客户端生成（16 随机字节） | 该文件的 AES 密钥 | 一次性 | — |
| `x-encrypted-param`（响应头） | CDN 上传响应 | 下载参数，填进发送 item 的 `media.encrypt_query_param` | 与媒体绑定 | — |

补充事实：

- **重复登录**：`get_bot_qrcode` 的 `local_token_list`（最多 10 个本地已有 `bot_token`）让服务端识别"这个机器人已绑定过本实例"，此时扫码状态返回 **`binded_redirect`**（不发新凭据）（W-CHANGELOG:97,99；W:src/auth/login-qr.ts:81-93,423-434）。**因此 `-14` 之后的重新登录应传空的 `local_token_list`**（旧 token 已失效），否则可能拿不到新凭据。
- **同一用户重新登录后**，官方会删除本地同 `ilink_user_id` 的旧账号及其 context_token，理由是"避免 context_token 匹配歧义"（W:src/auth/accounts.ts:82-108）。
- 官方日志脱敏把 `context_token / bot_token / token / authorization` 视为敏感字段，URL 的 query 一律遮蔽（W:src/util/redact.ts:28-57）。**这些值在我们的日志里同样不得出现**。
- 登录凭据**不**含"到期时间"字段（`token` + `savedAt` + `baseUrl` + `userId`）（W:src/auth/accounts.ts:115-121）。

---

## 5. 接口逐个说明

下列"请求/响应"均来自 W 的构造器与类型，并与 P、A 交叉核对；"响应"中只列源码读取到的字段，服务端可能多发其它字段（解析时必须容忍未知字段）。

### 5.1 `POST /ilink/bot/get_bot_qrcode?bot_type=3` —— 取登录二维码

- 查询参数 `bot_type=3`（W:src/auth/login-qr.ts:36；A:weixin_oc_adapter.py:129）。
- 请求体：`{"local_token_list": ["<已有 bot_token>", …]}`，最多 10 个、最新的在前；首次登录传 `[]`（W:src/auth/login-qr.ts:81-102）。无 `Authorization`、无 `base_info`。
- 响应：

```json
{ "qrcode": "<QRCODE>", "qrcode_img_content": "<要编码为二维码的 URL>" }
```

- A 用 `GET` 同一路径（旧版），两者响应相同（A:weixin_oc_adapter.py:1058-1072）。**2.4.9 用 POST**；实现用 POST，若遇 405/404 可回退 GET（【M0】首次联调记录）。

### 5.2 `GET /ilink/bot/get_qrcode_status?qrcode=<urlencoded>[&verify_code=<urlencoded>]` —— 轮询扫码状态

- 长轮询：服务端挂起至状态变化；客户端 35 s 超时视为 `wait`，之间睡 1 s（W:src/auth/login-qr.ts:33,128-158,486）。
- 响应字段：`status`（必有）、`bot_token`、`ilink_bot_id`、`baseurl`、`ilink_user_id`、`redirect_host`（W:src/auth/login-qr.ts:48-65）。
- `status` 取值与客户端动作（W:src/auth/login-qr.ts:334-476；P:protocol.md:137-148）：

| `status` | 含义 | 客户端动作 |
| --- | --- | --- |
| `wait` | 等待扫码 | 继续轮询 |
| `scaned` | 已扫码，等手机确认（"scaned"为官方拼写） | 提示"正在验证"；若之前提交过验证码，说明验证码正确，清除暂存 |
| `need_verifycode` | 服务端要求输入手机上显示的**数字验证码（配对码）** | 向用户索取，下一次轮询加 `&verify_code=<码>`；再次出现该状态＝上次输错 |
| `verify_code_blocked` | 验证码输错次数过多 | 清除暂存，刷新二维码（计入刷新次数），超限放弃 |
| `expired` | 二维码过期 | 重新 `get_bot_qrcode`，**最多刷新到第 3 个码**（官方 `MAX_QR_REFRESH_COUNT=3`）；超限放弃 |
| `scaned_but_redirect` | 需要换主机（IDC 重定向） | 之后改用 `https://<redirect_host>` 轮询（缺失则沿用当前主机） |
| `binded_redirect` | 该机器人已绑定到本实例（因 `local_token_list`） | 视为"已完成"，不发新凭据 |
| `confirmed` | 登录成功 | 取 `bot_token`、`ilink_bot_id`（缺失则登录失败）、`baseurl`、`ilink_user_id`，保存 |

- 整体等待上限：官方登录命令 480 s（8 分钟）（W:src/auth/login-qr.ts:311；src/channel.ts:431）。
- 终端展示：用 `qrcode_img_content` 生成字符二维码并**同时打印链接**作为兜底（W:src/auth/login-qr.ts:164-174）；A 用同一内容生成 ASCII 二维码并给出第三方在线二维码生成链接（A:1073-1092——**我们不使用在线服务，避免把登录内容发给第三方**，改本地生成 PNG）。

确认后的响应示例（占位值）：

```json
{ "status": "confirmed", "bot_token": "<BOT_TOKEN>", "ilink_bot_id": "<HEX>@im.bot",
  "baseurl": "https://<api-host>", "ilink_user_id": "<HEX>@im.wechat" }
```

### 5.3 `POST /ilink/bot/getupdates` —— 长轮询收消息

请求（W:src/api/api.ts:458-469；P:protocol.md:172-182）：

```json
{ "get_updates_buf": "", "base_info": { "channel_version": "<v>", "bot_agent": "<ua>" } }
```

`sync_buf` 已废弃，不要发（W:src/api/types.ts:213-215；P:protocol.md:206）。

响应：

```json
{ "ret": 0, "msgs": [ /* WeixinMessage… */ ], "get_updates_buf": "<下一个游标>", "longpolling_timeout_ms": 35000 }
```

| 字段 | 说明 |
| --- | --- |
| `ret` / `errcode` / `errmsg` | 见 §3；`-14` ＝ token 失效 |
| `msgs` | 入站消息数组，可为空或缺失 |
| `get_updates_buf` | 新游标；**仅当非空才保存并回传**（W:src/monitor/monitor.ts:152-156） |
| `longpolling_timeout_ms` | 服务端建议的**下一次**长轮询超时；>0 时官方把它设为下一次的**客户端**超时（W:src/monitor/monitor.ts:107-110） |

行为要点：

- 服务端挂起请求直到有消息或超时；客户端超时按"空结果"处理，**回传的是同一个游标**（W:src/api/api.ts:483）。
- 官方对 `msgs` **逐条处理且不按 `message_type` 过滤**（W:src/monitor/monitor.ts:158-183）；入站消息 `message_type=1`（USER），Bot 自己的消息 `=2`（W:src/api/types.ts:64-68）。服务端是否会回显 Bot 自己发的消息源码读不出来【M0 观察】；**建议实现里丢弃 `message_type==2`** 与 `from_user_id` 不是绑定用户的消息。
- 游标保存时机见 §10.5。

### 5.4 `POST /ilink/bot/sendmessage` —— 发送消息

请求（W:src/messaging/send.ts:57-80；P:protocol.md:214-236；A:weixin_oc_adapter.py:894-913）：

```json
{
  "msg": {
    "from_user_id": "",
    "to_user_id": "<USER_ID>@im.wechat",
    "client_id": "<客户端生成的 id>",
    "message_type": 2,
    "message_state": 2,
    "context_token": "<最近一条入站消息的 context_token>",
    "item_list": [ { "type": 1, "text_item": { "text": "你好" } } ]
  },
  "base_info": { "channel_version": "<v>", "bot_agent": "<ua>" }
}
```

| 字段 | 说明 |
| --- | --- |
| `from_user_id` | 固定 `""` |
| `to_user_id` | 收件人（用户的 `ilink_user_id`） |
| `client_id` | 客户端生成。W：`openclaw-weixin:<毫秒时间戳>-<8位hex>`（W:src/util/random.ts:7-9；src/messaging/send.ts:52-54）；A：`uuid4().hex`。**服务端是否按它去重源码不显示**【M0】 |
| `message_type` | `2`（BOT） |
| `message_state` | `2`（FINISH）；`0` NEW / `1` GENERATING 官方发送路径从不使用 |
| `context_token` | 见 §4；缺失时官方**只告警仍然发送**（W:src/messaging/send.ts:109-111），A 则**直接拒绝发送**并提示"先发一条消息刷新 context_token"（A:886-893） |
| `item_list` | **恰好 1 个 item**（W:src/messaging/send.ts:188-191：带文字说明的媒体拆成两次请求，**文字在前、媒体在后**，各自一个 `client_id`） |
| `run_id` | 可选，一次生成过程的标识；我们不用 |

- 文字为空时官方不带 `item_list`（W:src/messaging/send.ts:65-76）——**我们不得发空文字**。
- 响应：成功可能是 `{}`，也可能带 `message_id`（uint64，字符串化；官方用它缓存出站消息供引用还原，W:src/messaging/send.ts:127-128）。失败：HTTP 200 + `ret≠0` + `errmsg`。**真实的成功/失败响应体【M0】首次联调记录**。
- **单条文本长度**：框架侧 `textChunkLimit: 4000`（W:src/channel.ts:269），即官方在 4000 字符处分段；服务端硬上限未知【M0 可选】。
- 官方对出站文字做流式 Markdown 过滤（剥离部分 Markdown 记号），因为微信不渲染 Markdown（W:src/messaging/markdown-filter.ts:1-27；W-CHANGELOG:149-153）。**我们的输出本来就是拟人纯文本，无需处理，但不得依赖 Markdown 渲染。**

### 5.5 `POST /ilink/bot/getconfig` —— 取 `typing_ticket`

请求：`{"ilink_user_id": "<USER_ID>@im.wechat", "context_token": "<可选>", "base_info": {…}}`（W:src/api/api.ts:599-616）。
响应：`{"ret": 0, "typing_ticket": "<base64 票据>"}`（W:src/api/types.ts:266-271）。

- 官方在**收到每条入站消息时**按用户取一次（有缓存，见 §4）；`ret!==0` 视为失败。
- A 在没有该用户的 `context_token` 时认为"不支持 typing"，不调用（A:232-235,256-290）。**没有 context_token 时 getconfig 是否可用【M0】**。

### 5.6 `POST /ilink/bot/sendtyping` —— "对方正在输入"

请求：`{"ilink_user_id": "<USER_ID>@im.wechat", "typing_ticket": "<TICKET>", "status": 1, "base_info": {…}}`；`status`：`1` 开始输入，`2` 取消（W:src/api/types.ts:246-258；src/api/api.ts:619-630）。

- 官方用法：开始处理回复时发 `status=1`，**每 5 秒续发一次保活**，结束时发 `status=2`（W:src/messaging/process-message.ts:329-358，`keepaliveIntervalMs: 5000`）。A 同样 5 秒保活、票据 60 秒过期重取（A:179-186,305-378,413-495）。
- 没有 `typing_ticket` 时官方静默跳过（W:src/messaging/process-message.ts:329-343）。
- **手机上是否真的显示"对方正在输入"、`sendtyping` 是否消耗发送配额、票据在非回复场景是否有效：【M0】**（社区另称"`sendtyping` 返回 `ret:0` 但无效果"，仅在"试图用它重新激活会话"的语境下，参见 §9.2）。

### 5.7 `POST /ilink/bot/getuploadurl` —— 申请媒体上传地址

请求（W:src/cdn/upload.ts:83-93；src/api/types.ts:32-53）：

```json
{
  "filekey": "<16 字节随机数的 hex>",
  "media_type": 1,
  "to_user_id": "<USER_ID>@im.wechat",
  "rawsize": 12345,
  "rawfilemd5": "<明文 MD5 hex>",
  "filesize": 12352,
  "no_need_thumb": true,
  "aeskey": "<16 字节密钥的 32 位 hex>",
  "base_info": { "channel_version": "<v>", "bot_agent": "<ua>" }
}
```

| 字段 | 说明 |
| --- | --- |
| `media_type` | `1` 图片、`2` 视频、`3` 文件、`4` 语音（类型里有定义，**官方发送管线从不使用语音**，P:protocol.md:299） |
| `rawsize` / `rawfilemd5` | 明文大小 / MD5 |
| `filesize` | **密文大小** = `ceil((rawsize+1)/16)*16`（PKCS7；明文恰为 16 的倍数时多一整块）（W:src/cdn/aes-ecb.ts:18-21） |
| `no_need_thumb` | 官方**始终 `true`**：只传原件、不传缩略图（`thumb_*` 字段存在但未使用） |
| `aeskey` | 密钥的 **hex 字符串**（不是 base64） |

响应：`{"upload_param": "<加密参数>", "thumb_upload_param": "…", "upload_full_url": "<完整上传 URL，可选>"}`。`upload_full_url` 优先；没有则用 `upload_param` + `filekey` 自己拼 URL（§7.2）；两者都空视为失败（W:src/cdn/upload.ts:95-102）。

### 5.8 CDN 上传 / 下载

见 §7。

### 5.9 `POST /ilink/bot/msg/notifystart` 与 `/notifystop` —— 生命周期通知

请求体仅 `{"base_info": {…}}`，响应 `{"ret": 0, "errmsg": ""}`；通道启动时发 `notifystart`，停止时发 `notifystop`，**失败只记警告**，用于让服务端"对齐账号在线状态"（W:src/api/api.ts:637-662；src/channel.ts:519-529,566-576；W-CHANGELOG:100）。SPEC 清单里没有这两个接口；它们对收消息是否必要源码读不出来，**建议照官方做**（成本很低），失败不阻塞。

---

## 6. 消息模型

### 6.1 `WeixinMessage`（入站与出站同一结构）

| 字段 | 类型 | 说明 | 出处 |
| --- | --- | --- | --- |
| `seq` | number | 序号 | W:src/api/types.ts:193 |
| `message_id` | uint64（**转成字符串**） | 服务端消息 id；**去重主键** | :195 |
| `from_user_id` / `to_user_id` | string | 发送者 / 接收者（`…@im.wechat` 或 `…@im.bot`） | :196-197 |
| `client_id` | string | 客户端生成 id（出站用） | :198 |
| `create_time_ms` / `update_time_ms` / `delete_time_ms` | number（毫秒） | 时间戳（A 兼容"秒"：>1e12 视为毫秒，A:1536-1547） | :199-201 |
| `session_id`、`group_id` | string | 会话 / 群 id（群聊本项目不涉及；A 的 UI 文案"仅支持私聊"） | :202-203 |
| `message_type` | number | `1` 用户、`2` Bot | :204；:64-68 |
| `message_state` | number | `0` 新、`1` 生成中、`2` 完成 | :205；:81-85 |
| `item_list` | `MessageItem[]` | 内容项 | :206 |
| `context_token` | string | 回复用上下文凭据 | :207 |
| `run_id` | string | 生成过程 id（可选） | :208 |

**消息 id 与去重**：官方取 `message_id`，缺失则取 `item_list` 里第一个非空 `msg_id`（W:src/messaging/inbound.ts:202-210）；A 取 `message_id` 或 `msg_id`，都没有就用随机 uuid（A:1536）——**随机 id 会让去重失效**。我们的去重键：`message_id` → 首个 `item.msg_id` → 兜底用 `sha256(from_user_id|create_time_ms|seq|item 摘要)`。

### 6.2 `MessageItem`

`type` 取值（W:src/api/types.ts:70-79）：

| `type` | 含义 | 载荷字段 | 入站 | 出站（官方） |
| ---: | --- | --- | --- | --- |
| 1 | 文字 | `text_item.text` | 是 | 是 |
| 2 | 图片 | `image_item` | 是 | 是 |
| 3 | 语音 | `voice_item` | 是 | **否**（未实现） |
| 4 | 文件 | `file_item` | 是 | 是 |
| 5 | 视频 | `video_item` | 是 | 是 |
| 11 / 12 | 工具调用开始 / 结果 | `tool_call_start_item` / `tool_call_result_item` | — | 仅 OpenClaw 的"进度提示"用（W:src/messaging/reply-progress-sender.ts:165-221）；**我们不使用** |

通用字段：`create_time_ms`、`update_time_ms`、`is_completed`、`msg_id`、`ref_msg`（W:src/api/types.ts:175-189）。
**源码没有表情包/贴纸/位置/链接/名片等其它 item 类型**（`MessageItemType` 只有上表这些）。用户发来的微信表情包以何种形态到达【M0 采集】。

### 6.3 各类 item 的结构

**文字** `text_item`：`{ "text": "<string>" }`（W:src/api/types.ts:87-89）。

**CDN 媒体引用** `CDNMedia`（W:src/api/types.ts:92-99）：

```json
{ "encrypt_query_param": "<下载参数>", "aes_key": "<base64>", "encrypt_type": 1, "full_url": "<完整下载 URL，可选>" }
```

`encrypt_type`：`0` 只加密 fileid，`1` 打包缩略图/中图等信息（类型注释）；官方发送时恒为 `1`。

**图片** `image_item`（W:src/api/types.ts:101-114）：

| 字段 | 说明 |
| --- | --- |
| `media` | 原图 CDN 引用 |
| `thumb_media` | 缩略图 CDN 引用 |
| `aeskey` | **入站优先使用**：32 位 hex（16 字节原始密钥），优先于 `media.aes_key`（W:src/media/media-download.ts:43-45） |
| `mid_size` | 出站时填**密文大小**（W:src/messaging/send.ts:301） |
| `url`、`thumb_size`、`thumb_height`、`thumb_width`、`hd_size` | 类型里有定义，官方未使用 |

**语音** `voice_item`（W:src/api/types.ts:116-127）：`media`、`encode_type`（`1` pcm、`2` adpcm、`3` feature、`4` speex、`5` amr、`6` silk、`7` mp3、`8` ogg-speex）、`bits_per_sample`、`sample_rate`（Hz）、`playtime`（毫秒）、**`text`（微信云端语音转文字）**。官方：有 `text` 就直接当文字用且**不下载语音**；无 `text` 才下载并把 silk 转 wav（采样率 24 kHz）（W:src/messaging/inbound.ts:219-221；src/messaging/process-message.ts:141-146；src/media/silk-transcode.ts:3-4；src/media/media-download.ts:71-99）。

**文件** `file_item`：`media`、`file_name`、`md5`、`len`（**明文字节数，十进制字符串**）（W:src/api/types.ts:129-134；出站构造 W:src/messaging/send.ts:381-392）。

**视频** `video_item`：`media`、`video_size`（出站为密文大小）、`play_length`、`video_md5`、`thumb_media`（封面）、`thumb_size`、`thumb_height`、`thumb_width`（W:src/api/types.ts:136-145）。官方**只下载视频本体**（上限 100 MiB），不下载 `thumb_media`（W:src/media/media-download.ts:9,130-152）；出站视频不带封面、时长等字段（W:src/messaging/send.ts:337-347）。

### 6.4 引用 `ref_msg`

```json
{ "type": 1, "text_item": { "text": "<我的回复>" },
  "ref_msg": { "title": "<被引用消息的摘要>", "svr_id": "<被引用消息的服务端 id>",
               "message_item": { /* 完整的被引用 MessageItem，新版微信客户端可能省略 */ },
               "partial_text": { "start": "…", "end": "…", "startindex": 0, "endindex": 0, "quotemd5": "<md5>" } } }
```

字段（W:src/api/types.ts:147-162）：`message_item`（被引用内容）、`title`（摘要）、`svr_id`（新版客户端只给 id、不给内容时用）、`partial_text`（只引用了一段文字时的选区：`start`/`end` 是选区首尾文本，`startindex`/`endindex` 是它们是第几次出现，`quotemd5` 是选区文本的 MD5）。

- `ref_msg` 位于**用户发出的那条消息的 item 上**（通常是 TEXT item）；官方用 `item_list.find(i => i.ref_msg)` 找它（W:src/messaging/inbound.ts:300-302）。
- 摘要拼法：`[title] | [被引用 item 的正文]`（W:src/messaging/inbound.ts:304-314）。
- 只有 `svr_id` 时官方用本地 SQLite 旁路缓存（按账号+会话隔离，**入站与出站消息都缓存**，默认保留 30 天/1 万条；媒体 7 天/256 MiB/单个 25 MiB）按 id 还原正文；还原不到显示 `[引用消息内容未缓存]`（W:src/messaging/inbound.ts:338-403；src/messaging/quote-store.ts:11-15；W-CHANGELOG:19）。
- 局部引用解析：用 `start`/`startindex` 找起点（第 N 次出现），用 `end`/`endindex` 找终点——`endindex` 有"全局第 N 次"和"起点之后第 N 次"两种解读，两种都算候选，用 `quotemd5` 判定，没有 `quotemd5` 时取全局解读（W:src/messaging/partial-quote.ts:10-62）。
- 被引用的媒体：官方会下载被引用消息里的媒体（W:src/messaging/process-message.ts:147-170）；A 在被引用内容缺失时，用"60 秒内最近消息的 `create_time_ms`"去匹配本地缓存（A:113,1306-1345）。

### 6.5 映射到 `InboundMessage{id, at, kind, text, media_ref, quote}`（R-CH-005）

| 入站 item | `kind` | `text` | `media_ref` | 依据 |
| --- | --- | --- | --- | --- |
| TEXT | `text` | `text_item.text` | — | W:src/messaging/inbound.ts:212-217 |
| IMAGE | `image` | — | 下载→解密→`MediaStore`（`MediaKind.IMAGE`）。密钥取 `image_item.aeskey`(hex) → `media.aes_key`；**都没有则下载的字节本身就是明文** | W:src/media/media-download.ts:40-70 |
| VOICE，有 `voice_item.text` | `voice` | 转写文字 | 不下载 | W:src/messaging/inbound.ts:219-221 |
| VOICE，无转写 | `voice` | `None`（标"未转写"） | 可下载（silk）；本项目不转码，存原始字节或不存 | W:src/media/media-download.ts:71-99 |
| FILE | `file` | `file_item.file_name` | 下载→解密（`MediaKind.FILE`） | W:src/media/media-download.ts:100-129 |
| VIDEO | `video` | — | **取封面**：用 `video_item.thumb_media`（官方没做过，密钥规则不明，失败则标"无封面"）；视频本体默认不下载 | SPEC R-CH-005；W:src/api/types.ts:141【M0 采集】 |
| 带 `ref_msg` 的 item | 由主 item 决定 | — | — | `quote` 取被引用内容（还原规则见 §6.4） |
| 其它未知 `type` | `unknown` | — | — | 记录类型号（不记内容）到 DEBUG |

- `at`：`create_time_ms / 1000`（UTC）；缺失用接收时刻。
- 一条 `WeixinMessage` 的 `item_list` 可含多个 item（如文字 + 引用）；官方按"图片 > 视频 > 文件 > 语音"取**一个**主媒体，文字取**第一个**文字（W:src/messaging/process-message.ts:127-154）。我们的建议：一个媒体 item 对应一个 `InboundMessage`，id 后缀 `#<item 序号>`。

---

## 7. 媒体加密与 CDN

### 7.1 加密

| 项 | 值 | 出处 |
| --- | --- | --- |
| 算法 | **AES-128-ECB**，PKCS7 填充（块 16 字节） | W:src/cdn/aes-ecb.ts:6-16；A:weixin_oc_client.py:75-80,135-136 |
| 密钥来源（上传） | 客户端 `random 16 字节`；`getuploadurl.aeskey` 传其 **hex**；发送 item 的 `media.aes_key` 传 **`base64(hex 字符串的 ASCII)`**（44 字符），即"把 hex 当文本再 base64" | W:src/cdn/upload.ts:77,92；src/messaging/send.ts:298；A:weixin_oc_adapter.py:653,707 |
| 密钥来源（下载） | 来自入站消息：图片优先 `image_item.aeskey`（hex）；否则 `media.aes_key`（base64）。`media.aes_key` 解码后有两种形态，**都要支持**：16 字节原始密钥，或 32 字节且全为 hex 字符的 ASCII（再按 hex 还原成 16 字节）。其它长度报错 | W:src/cdn/pic-decrypt.ts:30-52；A:weixin_oc_client.py:93-107 |
| 填充 | PKCS7；密文长度 = `ceil((n+1)/16)*16`。A 的解填充**宽松**（填充不合法时原样返回） | W:src/cdn/aes-ecb.ts:18-21；A:weixin_oc_client.py:82-91 |
| 分块 | **无**。整文件一次加密，一次 `POST` | W:src/cdn/cdn-upload.ts:25,42-46 |
| 缩略图 | 官方不传（`no_need_thumb:true`） | W:src/cdn/upload.ts:91 |

沙箱里用 Python `cryptography`（`Cipher(AES, ECB)` + `padding.PKCS7(128)`）对 0/1/15/16/17/31/32/1000 字节做了加解密往返，密文长度与上式一致；两种密钥形态（`base64(16 字节)`=24 字符、`base64(32 位 hex)`=44 字符）的解析互相还原一致。因此第 02 轮"用协议同款算法加密合成图片再解密比对"的测试可以直接写。

### 7.2 URL 构造

- **上传**：优先 `getuploadurl` 返回的 `upload_full_url`；否则
  `<cdn>/upload?encrypted_query_param=<urlencode(upload_param)>&filekey=<urlencode(filekey)>`（W:src/cdn/cdn-url.ts:14-20；A:weixin_oc_client.py:59-63）。
- **下载**：优先媒体引用里的 `full_url`；否则
  `<cdn>/download?encrypted_query_param=<urlencode(encrypt_query_param)>`（W:src/cdn/cdn-url.ts:9-11；src/cdn/pic-decrypt.ts:66-73；A:weixin_oc_client.py:65-69）。官方留有开关 `ENABLE_CDN_URL_FALLBACK=true`（服务端以后可能只给 `full_url`）。
- 其中 `<cdn>` 默认 `https://novac2c.cdn.weixin.qq.com/c2c`。**下载不带任何自定义头**（W:src/cdn/pic-decrypt.ts:11）。

### 7.3 上传流程（出站图片）

1. 读明文，算 `rawsize`、`rawfilemd5`、`filesize`；生成 `filekey`（16 随机字节 hex）与 `aeskey`（16 随机字节）。
2. `getuploadurl`（`media_type=1`，`to_user_id`，`no_need_thumb=true`，`aeskey`=hex）。
3. 加密整文件；`POST <上传 URL>`，`Content-Type: application/octet-stream`，体＝密文。
4. 成功条件：HTTP 200 **且**响应头 `x-encrypted-param` 非空（＝下载参数）。HTTP 4xx 立刻放弃（错误文字在响应头 `x-error-message` 或体里）；其它失败最多尝试 3 次（总共）。
5. 用 `sendmessage` 发送 `image_item`：

```json
{ "type": 2, "image_item": { "media": { "encrypt_query_param": "<x-encrypted-param 的值>",
    "aes_key": "<base64(hex 密钥的 ASCII)>", "encrypt_type": 1 }, "mid_size": 12352 } }
```

出处：W:src/cdn/upload.ts:62-121；src/cdn/cdn-upload.ts:14-91；src/messaging/send.ts:275-315；A:weixin_oc_adapter.py:641-739。视频 `video_item{media, video_size=密文大小}`、文件 `file_item{media, file_name, len=明文大小字符串}` 同理（W:src/messaging/send.ts:337-347,381-392）。

### 7.4 大小限制

- 入站媒体官方保存上限 100 MiB（W:src/media/media-download.ts:9）。
- 出站大小上限、图片尺寸/格式限制：**源码无**【M0 可选】。出站类型靠扩展名分流：`video/*`→视频、`image/*`→图片、其它→文件（W:src/messaging/send-media.ts:32-60；扩展名表 src/media/mime.ts:19-37）。

---

## 8. 登录失效与重新登录

1. `getupdates`（以及任何接口）返回 `-14` → 凭据失效。官方只暂停 1 小时再试；A 直接清除登录状态进入扫码。
2. **我们的做法**（R-CH-003 / R-OPS-004）：进入 `NEEDS_RELOGIN`，停止一切发送，写 `alerts` 表并在控制台醒目输出；提示用户运行 `twin channel login`；登录时 `local_token_list` 传 `[]`。是否保留"每小时再试一次旧 token"的恢复探测由实现者决定（官方有，成本低，成功则自动解除告警）。
3. 重新登录成功后：**游标重置为 `""`，context_token 清空**（A 如此；官方在"同一用户重新登录"时清旧账号的 context_token）。

---

## 9. 会话窗口与条数：源码线索与社区说法

### 9.1 源码能证明的（全部）

1. 出站统一把"最近一条入站的 `context_token`"带上；官方按（账号,用户）缓存并**重复使用**，连"定时任务投递"（无入站触发）也走同一缓存，且重启后从磁盘恢复（W:src/messaging/inbound.ts:41-116；src/channel.ts:251,270-280,318,351）。→ 在**协议设计上**，"用上次的 context_token 主动发"是被预期的用法。
2. 源码里**没有**任何发送计数、窗口计时、配额检查；也没有为 `context_token` 设置过期时间（时间常量只有：getconfig 缓存 24 h、二维码 5 min、会话暂停 1 h、引用缓存保留期）。
3. 带文字说明的媒体 = 两次 `sendmessage`（计两条）（W:src/messaging/send.ts:188-264）。
4. 与"窗口"最接近的服务端信号只有 `-14`（凭据失效）。
5. 单条文本框架分段上限 4000 字符（W:src/channel.ts:269）。

→ **窗口长度、单次入站后可发条数、回复/主动是否共用配额、`sendtyping`/`getconfig` 是否计数：源码均无依据，全部【需 M0 实测确认（见 R-CH-009）】。**

### 9.2 社区说法（**未验证、互相矛盾，仅作探针设计的参考**）

| 编号 | 来源 | 说法 |
| --- | --- | --- |
| C1 | clawhub.ai/noaheleven/weixinclaw-proactive-push | `ret:-2` + `errmsg:"prepare failed"` ＝ 用户侧下行会话过期，"约 1 天无用户消息即过期"；会话存活时**文字可用空 `context_token`**，**媒体必须非空 token**；会话过期后连空 token 文字也 `-2`；`sendtyping`/`getconfig` 无法重新激活会话 |
| C2 | clawhub.ai/lroolle/wxclawbot-send | `-2` ＝ **频率限制，每 bot 约 7 条 / 5 分钟**（所有客户端共享）；`-14` ＝ 会话过期需重新登录；缺失/过期的 context_token 会让接口报 `ok:true` 但**消息不推送通知，留在服务器上直到用户打开聊天窗口** |
| C3 | openclawdir.com/plugins/weixin-bridge-8npli4 | "每 10 条需要用户回一条消息（微信限制）"；context token 会过期，长时间不活动后发送可能 `ret:-2` |
| C4 | chaincatcher.com/article/2254341（2026-03-25，媒体报道） | ClawBot "当前仅支持单聊，不支持群聊，不支持 AI 主动推送消息"；"平台会对传输内容进行安全审核"；"灰度测试阶段，腾讯保留随时调整、限制或终止服务的权利" |
| C5 | CSDN 排障文 | `{"ret":-2,"errmsg":"prepare failed"}` 被归因为"iLink 只能被动回复、不能向未建立活跃会话的用户主动推送"（作者推断） |

**这些说法至少有三种互不兼容的解释**：窗口/会话过期（C1、C3）、速率限制（C2）、每次回复 10 条上限（C3），外加"根本不能主动推"（C4）。

**对探针设计的直接含义**（给 02c）：

1. 记录每次失败的**完整 `ret`/`errcode`/`errmsg`**（脱敏），用 `errmsg` 区分上面几种解释；
2. ① 每 2 分钟一条＝每 5 分钟约 2.5 条，低于 C2 的 7 条/5 分钟，不会被误判为速率限制；
3. **`ret=0` 不等于手机收到**（C2 的"ok:true 却不推送"）。每步结束后在 CLI 里问用户"手机上实际收到几条 [测试]"，与 API 成功数**对账**，以手机为准；
4. 额外（可选，默认关）实验：空 `context_token` 的文字能否发送（C1 vs A 的拒绝策略）。

---

## 10. 给实现者的建议（02b 起）

### 10.1 能力矩阵

| 能力 | 源码结论 | 出处 | `capabilities()` 取值 | 待 M0 |
| --- | --- | --- | --- | --- |
| 发送文字 | **支持** | W:src/messaging/send.ts:103-137 | — | 单条长度上限（可选） |
| 发送**引用** | **不支持**。`ref_msg` 只存在于入站；所有发送构造器都不设置它（对 `src/**` 全文检索 `ref_msg`/`svr_id`/`partial_text`，只命中类型定义、入站解析、入站媒体选择与 64 位 id 精度处理，没有任何发送侧引用） | W:src/api/types.ts:147-154,181；src/messaging/send.ts（全文无） | `supports_quote = False`；`send_text(quote=…)` 抛 `CapabilityNotSupported` | 探针 ② 的引用子步骤**跳过**并在报告写"源码不支持"。是否值得做"手工构造 `ref_msg` 看服务端是否接受"的实验，由用户另行决定（未被任何来源验证，也违背"不伪造能力"） |
| 发送**正在输入** | **支持**（`getconfig`→`typing_ticket`→`sendtyping`，开始/保活 5 s/取消） | W:src/api/api.ts:599-630；src/messaging/process-message.ts:329-358 | `supports_typing = True`（协议层）；无票据时 `send_typing` 静默返回 | 手机是否可见；是否计入配额；无 context_token 时是否可用；票据有效期 |
| 发送**图片**（含 PNG/JPEG/WebP） | **支持**，字节原样加密上传（`image/*` 走 IMAGE 通道） | W:src/messaging/send-media.ts:46-58；src/cdn/upload.ts:123-135 | 支持（受 R-SAFE-006 白名单约束） | 大小上限（可选） |
| 发送 **GIF** | 与图片**同一条通道**：`.gif`→`image/gif`→IMAGE item，字节原样；无格式字段、无帧处理、无转码。**是否会动取决于服务端与手机** | W:src/media/mime.ts:27；src/messaging/send-media.ts:46-58 | `gif_animated = None`（未知） | **探针 ② 询问用户"GIF 是否在动"**；不动时由用户决定首帧/原样（R-STK-007） |
| 发送**文件** | **支持**（`media_type=3`，`file_item{file_name,len}`） | W:src/messaging/send.ts:366-404 | 通道 API 不开放（R-CH-006 只允许图片） | — |
| 发送**视频** | **支持**（`media_type=2`，`video_item{video_size}`，无封面/时长） | W:src/messaging/send.ts:322-359 | 通道 API 不开放 | — |
| 发送**语音** | **不支持**（`media_type=4` 仅类型定义，官方发送管线不使用） | P:protocol.md:299 | 不支持（也符合 R-SAFE-002） | — |
| 接收**文字** | 支持 | W:src/messaging/inbound.ts:212-217 | — | — |
| 接收**图片** | 支持（下载+解密） | W:src/media/media-download.ts:40-70 | — | — |
| 接收**语音** | 支持；**微信云端转写在 `voice_item.text`** | W:src/api/types.ts:125-126 | — | 无转写比例 |
| 接收**文件** | 支持（含 `file_name`） | W:src/media/media-download.ts:100-129 | — | — |
| 接收**视频** | 支持本体；**封面**有 `thumb_media` 字段，官方未使用 | W:src/api/types.ts:141 | — | 封面能否按同样规则解密【M0 采集】 |
| 接收**引用** | 支持，且新版客户端可能只给 `svr_id` → 必须本地缓存消息正文才能还原 | W:src/messaging/inbound.ts:300-403 | — | 实际形态【M0 采集】 |
| 接收**表情包** | **源码无此 item 类型**；到达形态未知 | W:src/api/types.ts:70-79 | — | 【M0 采集】 |
| 接收**群聊** | 类型里有 `group_id`，官方声明 `chatTypes:["direct"]`；本项目不涉及 | W:src/channel.ts:229-233 | — | — |
| 主动发送窗口时长 | **源码无线索** | §9.1 | `proactive_window_h = None`（配置值仅为保守猜测） | **【M0】R-CH-009 ③** |
| 单次入站后的发送条数上限 | **源码无线索** | §9.1 | `outbound_quota = None` | **【M0】R-CH-009 ①** |
| 回复与主动是否共用配额 | **源码无线索**（发送路径对两者一视同仁：同一接口、同一 token） | §9.1 | 先按"共用"保守处理 | **【M0】R-CH-009 ④** |

### 10.2 模块划分建议（`src/twin/channel/`）

| 模块 | 职责 |
| --- | --- |
| `base.py` | `Channel` 抽象、`InboundMessage`、`OutboundResult`（含 `kind` 枚举：OK / WINDOW_REJECTED / AUTH_EXPIRED / REJECTED / NETWORK / AMBIGUOUS）、`ChannelCapabilities`、`SessionState`；异常 `RecipientNotAllowed`、`MediaNotAllowed`、`CapabilityNotSupported` |
| `window.py` | `SessionWindow`（`last_inbound_at`、`outbound_since_inbound`、`expired`；`remaining_quota()`、`can_send_proactive(now, n)`）；LocalConsole 与 Ilink 共用，状态变更即持久化 |
| `policy.py` | `OutboundMediaPolicy`（sha256 白名单接口）、`ProbeSendPolicy`（探针期间、仅"[测试]"、写审计） |
| `ilink/wire.py` | 线上结构的宽松模型（pydantic，`extra="ignore"`）、常量（item 类型、状态、`-14`）、`message_id` 字符串化 |
| `ilink/http.py` | `IlinkHttp`：`httpx.AsyncClient` 封装——头构造、`base_info`、超时、脱敏日志、HTTP/业务错误统一成 `IlinkApiError(http_status, ret, errcode, errmsg)` |
| `ilink/media.py` | AES-128-ECB（`cryptography`）、密钥解析（16 字节/32 位 hex/base64 两形态）、`getuploadurl`+CDN 上传、CDN 下载+解密（纯函数 + 小的 I/O 层，便于 respx 测试） |
| `ilink/login.py` | 二维码登录状态机（刷新、验证码、重定向、`binded_redirect`）；`qr.py` 本地生成 PNG 与字符二维码（**新增依赖 `qrcode`**；Pillow 已有，不得使用在线二维码服务） |
| `ilink/poller.py` | 长轮询循环、游标、去重、退避、`NEEDS_RELOGIN` 状态机 |
| `ilink/inbound.py` | `WeixinMessage` → `InboundMessage`、引用还原、媒体落库 |
| `ilink/outbound.py` | item 构造、`send_text/send_image/send_typing`、`typing_ticket` 缓存与 5 s 保活 |
| `ilink/channel.py` | `IlinkChannel`：组合以上；收件人守卫（R-CH-007）、窗口记账、`start()/stop()`（含 `notifystart/stop`） |
| `local_console.py` | `LocalConsoleChannel`（R-CH-011） |
| `probe.py` | `ChannelProbe` 状态机（建议留给单独的后续步骤） |

**建议拆步**：02b＝`base/window/policy/ilink 全部/收件人绑定/登录与状态命令`+全部 respx 测试；02c＝`probe` + CLI + 报告；02d＝`LocalConsoleChannel` + `twin chat --local`/`echo-test`。理由：02b 之后即可真机联调登录与收发，探针所需的 `ProbeSendPolicy` 依赖 02b 的出站路径。

### 10.3 需要持久化的状态（`channel_state`，加密）

登录凭据放加密的 `channel_state` 即可（满足 R-CH-003；CLAUDE.md 铁律 10 的清单里没有 bot token，且 Windows 凭据管理器单条上限约 2.5 KB、token 长度未知）。

| 键 | 内容 | 说明 |
| --- | --- | --- |
| `ilink.credentials` | `bot_token`、`ilink_bot_id`、`ilink_user_id`(登录时的期望用户)、`api_base_url`、`saved_at` | `-14` 后标记失效而非删除，登录成功再覆盖 |
| `ilink.cursor` | `get_updates_buf` | 推进时机见 §10.5 |
| `ilink.bound_user` | 绑定的用户 id、绑定时间 | R-CH-007 |
| `ilink.context_token` | `{token, received_at}`（仅绑定用户） | 每条入站覆盖；重启后恢复；日志永不输出 |
| `ilink.seen_ids` | 最近 N 个已处理消息 id（环形，建议 N≥500）| 去重 |
| `ilink.window` | `last_inbound_at`、`outbound_since_inbound`、`expired`、`expired_at`、`last_error`(ret/errcode/errmsg) | 每次变化即写 |
| `ilink.auth_state` | `ok / needs_relogin(+since,+code)` | 重启后仍告警 |
| `ilink.quote_index` | 近期入站/出站消息的 `message_id → 短正文`（有上限与保留期） | 用于只给 `svr_id` 的引用还原；内容属聊天内容，**必须加密且不进日志** |
| `ilink.probe_images` | 探针生成图片的 sha256 清单 | R-SAFE-006 |
| 探针计划 | 由探针自己的表/键持久化 | R-CH-009 |

不必持久化：`typing_ticket`（24 h 缓存，重取即可）、二维码会话。

### 10.4 重连 / 退避

- 长轮询网络类错误（连接、DNS、TLS、5xx、网关 524、读超时以外的异常）：`delay = min(60, 1 × 2^(n-1)) × U(0.5, 1.5)`，**一次成功后清零**（SPEC R-CH-004）。
- **长轮询读超时是正常结果，不退避**。客户端超时建议取 `longpolling_timeout_ms`（默认 35 000）**再加 5 s 余量**——官方让客户端超时恰等于服务端建议值（W:src/monitor/monitor.ts:107-110），二者可能同时到期【推断】；超过余量仍无响应才当作空结果重发。服务端真实挂起时长【M0 记录】。
- 业务错误 `-14`：不退避，直接进入 `NEEDS_RELOGIN`。
- 其它非 0 的 `ret/errcode`：按网络类错误退避；连续 ≥3 次写一次 `alerts`（warning，去重）。
- `sendmessage` 失败：**不自动重试**（SPEC：对会话过期绝不循环重试）；网络层超时属于"结果未知"，返回 `AMBIGUOUS`，由上层决定，不要盲目重发（重复气泡对拟人体验的伤害大于漏发一条；`client_id` 是否让服务端去重未知）。
- CDN 上传：4xx 立刻失败，其它最多 3 次（同官方）。
- 单实例：只允许一个轮询者使用同一 `bot_token`（R-OPS-002 的运行锁已保证），两个客户端共用一个游标的语义源码未说明。

### 10.5 游标与去重的正确顺序（对官方行为的改进）

官方在收到 `getupdates` 响应后**先把游标写盘，再处理消息**（W:src/monitor/monitor.ts:152-183），崩溃发生在两者之间就会**永久丢消息**（至多一次）。R-CH-004 要求"重启不丢消息、不重复处理"，所以：

1. 把本批消息全部去重后**持久化到数据库/入队**（含媒体落盘）；
2. 在**同一个事务里**更新 `ilink.cursor` 与 `ilink.seen_ids`；
3. 之后才把消息交给上层。崩溃后重放的消息被 `seen_ids` 吸收。

### 10.6 错误处理表

| 场景 | 判定 | 通道动作 | 对上层 |
| --- | --- | --- | --- |
| 任意接口 `ret`/`errcode == -14` | `AUTH_EXPIRED` | 状态→`NEEDS_RELOGIN`；`alerts`(critical, 去重键 `channel.auth_expired`)+控制台醒目输出；停止收发；可选每 60 min 探测恢复 | `OutboundResult(ok=False, kind=AUTH_EXPIRED)` |
| `sendmessage` 返回 `ret=-2`（社区称会话/配额类） | `WINDOW_REJECTED` | `window.expired=True`，记录 `ret/errmsg`；**不重试**；下一条入站复位 | `is_session_expired=True` |
| `sendmessage` 其它 `ret≠0` | `REJECTED(code)` | 记录；**不改**窗口状态（除非 M0 证明该码也属窗口类）；不重试 | `ok=False, code` |
| `sendmessage` HTTP 5xx / 读超时 | `AMBIGUOUS` | 不重试；记录 | `ok=False, ambiguous=True` |
| `sendmessage` 连接失败（请求未发出） | `NETWORK` | 可由上层稍后重发 | `ok=False` |
| 无 `context_token` | `NO_CONTEXT` | 拒绝发送（同 A 的做法）；视为窗口未建立 | `ok=False, kind=WINDOW_REJECTED` |
| 收件人不是绑定用户 | — | 抛 `RecipientNotAllowed`（铁律 6） | 异常 |
| `send_image` 字节不在白名单 | — | 抛 `MediaNotAllowed`（R-SAFE-006） | 异常 |
| `getuploadurl` 无上传地址 / CDN 4xx / 缺 `x-encrypted-param` | `UPLOAD_FAILED` | 不发 `sendmessage`；4xx 不重试，其它最多 3 次 | `ok=False` |
| 入站媒体解密失败/下载失败 | — | 消息仍投递，`media_ref=None`，记录 `kind` 与原因（不含内容） | — |
| 入站 `from_user_id` ≠ 绑定用户 | — | 丢弃并记录（不含内容）（R-CH-007） | — |
| `get_qrcode_status` 网络/网关错误 | — | 当 `wait` 继续 | — |
| 二维码连续过期超限 / `verify_code_blocked` 超限 | — | 登录命令失败退出并提示重试 | CLI 退出码非 0 |

### 10.7 可用 respx 构造的合成响应样例清单

全部字段为占位值；结构取自 W/A/P。标 † 的响应**形状在源码里只能推断**（见 §12 的 M0 项），测试里要同时覆盖"字段缺失"的情形。

**登录**

| 编号 | 请求 | 响应 |
| --- | --- | --- |
| S-01 | `POST /ilink/bot/get_bot_qrcode?bot_type=3` | `{"qrcode":"<QRCODE_1>","qrcode_img_content":"https://<qr-host>/<PLACEHOLDER_1>"}` |
| S-02 | `GET /ilink/bot/get_qrcode_status?qrcode=<QRCODE_1>` | `{"status":"wait"}` → `{"status":"scaned"}` → `{"status":"confirmed","bot_token":"<BOT_TOKEN>","ilink_bot_id":"<HEX>@im.bot","baseurl":"https://<api-host>","ilink_user_id":"<HEX>@im.wechat"}` |
| S-03 | 同上 | `{"status":"expired"}`（触发换码，第 2、3 个 `qrcode`），第 3 次过期后失败 |
| S-04 | 同上 | `{"status":"need_verifycode"}` → 带 `&verify_code=` 的下一次请求返回 `scaned`；错误码时再次 `need_verifycode`；`{"status":"verify_code_blocked"}` |
| S-05 | 同上 | `{"status":"scaned_but_redirect","redirect_host":"<other-host>"}` → 之后请求必须发往新主机；缺 `redirect_host` 时沿用旧主机 |
| S-06 | 同上 | `{"status":"binded_redirect"}`（无凭据） |
| S-07 | 同上 | `confirmed` 但缺 `ilink_bot_id` → 登录失败；缺 `bot_token` → 失败 |
| S-08 | 同上 | HTTP 524 / 超时 → 当 `wait` |

**长轮询**（请求体必须含 `get_updates_buf` 与 `base_info`，头含 `Authorization: Bearer <BOT_TOKEN>`）

| 编号 | 响应 |
| --- | --- |
| S-10 | `{"ret":0,"msgs":[],"get_updates_buf":"<CURSOR_1>","longpolling_timeout_ms":35000}`；以及 `{}`（全缺字段）† |
| S-11 | 文字：`{"ret":0,"msgs":[{"seq":1,"message_id":9223372036854775001,"from_user_id":"<USER>@im.wechat","to_user_id":"<BOT>@im.bot","create_time_ms":1790000000000,"message_type":1,"message_state":2,"context_token":"<CTX_1>","item_list":[{"type":1,"text_item":{"text":"<合成文字>"}}]}],"get_updates_buf":"<CURSOR_2>"}`（**message_id 取大于 2^53 的整数**以覆盖精度） |
| S-12 | 图片：`item_list:[{"type":2,"image_item":{"media":{"encrypt_query_param":"<DL_PARAM>","aes_key":"<B64_RAW16>","encrypt_type":1},"aeskey":"<HEX32>","mid_size":<n>}}]`；变体：只有 `media.aes_key`(base64 16 字节)；只有 `media.aes_key`(base64 的 32 位 hex)；无任何密钥（密文即明文）；带 `full_url` |
| S-13 | 语音有转写：`{"type":3,"voice_item":{"media":{…},"encode_type":6,"sample_rate":24000,"playtime":1500,"text":"<合成转写>"}}`；无转写：同上去掉 `text` |
| S-14 | 文件：`{"type":4,"file_item":{"media":{…},"file_name":"<合成名>.pdf","md5":"<MD5>","len":"1024"}}` |
| S-15 | 视频：`{"type":5,"video_item":{"media":{…},"video_size":<n>,"play_length":3000,"thumb_media":{…},"thumb_width":240,"thumb_height":135}}` |
| S-16 | 引用（完整）：`{"type":1,"text_item":{"text":"<回复>"},"ref_msg":{"title":"<摘要>","message_item":{"type":1,"msg_id":"<ID>","text_item":{"text":"<被引用正文>"}}}}` |
| S-17 | 引用（仅 id）：`"ref_msg":{"svr_id":"<ID>"}`；以及带 `partial_text:{"start":"…","end":"…","startindex":0,"endindex":0,"quotemd5":"<MD5>"}` |
| S-18 | 引用图片：`ref_msg.message_item` 是 `type=2` 的 image_item |
| S-19 | 未知 item `type` 99；`message_type` 为 2 的回显消息；`from_user_id` 为陌生人；重复的 `message_id`；缺 `message_id` 但 item 有 `msg_id` |
| S-20 | `{"ret":-14,"errmsg":"<ERRMSG>"}`、`{"errcode":-14}`、`{"ret":0,"errcode":-14}` ；`{"ret":1,"errmsg":"<ERRMSG>"}`；HTTP 500/502 |

**发送与输入**

| 编号 | 响应 |
| --- | --- |
| S-30 | `sendmessage` 成功：`{}` 与 `{"ret":0,"message_id":123}` †；请求体断言：`item_list` 恰 1 项、`message_type==2`、`message_state==2`、`from_user_id==""`、`context_token` 原样、`to_user_id` 为绑定用户 |
| S-31 | `sendmessage` 失败：`{"ret":-2,"errmsg":"prepare failed"}` †（社区报告的形状，**errmsg 文字未经验证**）；`{"ret":-14}`；`{"ret":1,"errmsg":"<ERRMSG>"}`；HTTP 500；读超时 |
| S-32 | `getconfig`：`{"ret":0,"typing_ticket":"<TICKET>"}`；`{"ret":1,"errmsg":"<ERRMSG>"}`；无票据字段 |
| S-33 | `sendtyping`：`{}` 与 `{"ret":0}`；断言 `status` 为 1/2、票据原样 |
| S-34 | `notifystart/notifystop`：`{"ret":0,"errmsg":""}`；HTTP 500（应被吞掉） |

**媒体**

| 编号 | 说明 |
| --- | --- |
| S-40 | `getuploadurl`：`{"upload_full_url":"https://<cdn>/c2c/upload?encrypted_query_param=<P>&filekey=<K>"}`；仅 `{"upload_param":"<P>"}`（断言 URL 拼接与百分号编码）；两者皆空 → 失败；断言请求体 `filesize==ceil((n+1)/16)*16`、`aeskey` 为 32 位 hex、`no_need_thumb==true`、`media_type==1` |
| S-41 | CDN 上传：`POST` → 200 + 响应头 `x-encrypted-param: <DL_PARAM>`（断言请求体是**正确加密**的密文、`Content-Type: application/octet-stream`）；200 但无该头 → 重试后失败；400 + `x-error-message` → 不重试；500 ×2 后 200 → 成功 |
| S-42 | `sendmessage` 的 `image_item`：断言 `media.aes_key == base64(hex(key).encode())`、`encrypt_type==1`、`mid_size==密文大小` |
| S-43 | CDN 下载：`GET <cdn>/download?encrypted_query_param=<DL_PARAM>` → 用同款算法加密的合成图片字节；解密后与原字节逐字节比对（PNG、JPEG、GIF、0 字节、整块边界 16/32 字节） |
| S-44 | 下载 404 / 错误密钥（填充非法）→ 入站消息仍投递、`media_ref=None` |

### 10.8 探针（R-CH-009）实现提示

- **出站只有一条路径**：`IlinkChannel.send_*` 统一经 `ProbeSendPolicy` 检查；探针使用的发送方法需要显式传入探针令牌，正常引擎路径拿不到。
- 每步前"先发一条新消息"：以**入站事件**为准（收到后才开始），并把该入站的 `context_token` 记入探针记录。
- 记录项：每次发送的时间、`client_id`、HTTP 状态、`ret/errcode/errmsg`（脱敏）、**用户在手机上的实收确认**。
- 报告写出的"建议值"＝实测值 × 0.9，**用户确认后才改配置**（提示词 C.3）。
- 探针图片：用 Pillow 生成 JPG/PNG/多帧 GIF（Pillow 已在依赖里），生成后登记 sha256。

---

## 11. 对接顺序与首次联调清单（02b 之后）

1. `twin channel login`：记录各接口**真实响应**（脱敏）并回填本文 §5/§10.7 中标 † 的项。
2. `twin channel send-test "你好"`：记录 `sendmessage` 成功响应体。
3. 让用户依次发送：文字、图片、语音、视频、文件、引用（对文字和对图片各一次）、微信自带表情包，记录**item 形态**（只记结构，不记内容）。
4. 再运行探针。

---

## 12. 与 SPEC 的差异

> 对应 `docs/DECISIONS.md` 的 D-007～D-016。"差异"＝SPEC / 提示词的文字与源码事实不一致，或 SPEC 的条件分支（"若协议支持"）已被源码决定。

| 编号 | SPEC / 提示词原文 | 源码事实 | 本项目的做法 |
| --- | --- | --- | --- |
| **D-007** | R-CH-008：收到会话过期错误（社区报告为 `ret:-2`）时标记过期；R-CH-002 提示"以源码与实测为准" | 源码中**没有 `-2`**；唯一的凭据/会话类码是 `-14`（getupdates，官方已更名为"token 失效，而非 session 过期"）。"窗口过期"概念在源码中不存在。社区对 `-2` 有速率限制/会话过期/每 10 条回复三种互斥解释 | 拆成两类：`-14`→`AUTH_EXPIRED`（需重新登录+告警）；`sendmessage ret=-2`→暂按"窗口/配额类拒绝"处理（标 expired、不重试）；所有 `ret/errmsg` 原样记录，由 M0 报告判定 `-2` 的真实含义 |
| **D-008** | R-CH-006："媒体发送需要非空 `context_token` 时按源码规则获取" | 源码**没有"媒体需非空 token"的规则**：所有出站统一带最近入站的 `context_token`，缺失仅告警仍发送；"媒体必须非空、文字可空"只见于社区 C1 | 所有出站带最近的 `context_token`；没有则拒绝发送（A 同）；"空 token 文字能否发送"列为可选 M0 实验 |
| **D-009** | R-CH-006/R-CH-009 ②：引用（若协议支持） | **出站引用不存在**（`ref_msg` 只在入站解析） | 条件分支落定为"不支持"：`supports_quote=False`；探针引用子步骤跳过 |
| **D-010** | R-CH-005：表情包（若可识别 md5） | `MessageItemType` 无表情/贴纸类型；md5 只存在于 `file_item.md5`、`video_item.video_md5`，不是表情 md5 | 02b 不识别表情包；未知 item 类型记为 `unknown`；到达形态由 M0 采集后再决定是否支持 |
| **D-011** | R-CH-005：视频取封面 | 官方只下载视频本体；`thumb_media` 有字段无用法，密钥/URL 规则无示例 | 尝试用 `thumb_media` 自带的 `aes_key` 解密；失败降级为"视频（无封面）"；形态由 M0 采集 |
| **D-012** | R-CH-007：以登录后第一条入站的发送者为准，CLI 确认 | 登录响应直接给出扫码者的 `ilink_user_id`（官方以它作授权兜底） | 保留 SPEC 流程；额外把 `ilink_user_id` 作为期望值——不一致时 CLI 明确警告并默认拒绝（只加强，不放宽） |
| **D-013** | R-CH-002 接口清单 | 实际还有 `getuploadurl`（上传必经）、`msg/notifystart`、`msg/notifystop`；`get_bot_qrcode` 官方 2.4.9 是 `POST`+`local_token_list`（A 为旧式 `GET`）；CDN 主机 `novac2c.cdn.weixin.qq.com/c2c` 不在 SPEC；登录响应的 `baseurl`/`redirect_host` 可改写 API 主机 | 全部实现；`baseurl` 持久化；CDN 主机写为常量 |
| **D-014** | R-CH-004：游标持久化，重启不丢消息 | 官方先存游标再处理消息（至多一次），照搬会丢消息 | 持久化消息之后、与去重表同一事务推进游标（§10.5） |
| **D-015** | R-CH-008 / 配置 `proactive_window_safe_h:22`、`outbound_quota_safe:8` | 窗口时长、条数、配额口径在源码中**无任何依据**；社区数字（≈24 h、10 条/回复、7 条/5 分钟）互相矛盾 | 在 M0 实测前，这两个默认值只是保守猜测，**不得在文档/报告里当作协议事实**；SPEC 数值不改 |
| **D-016** | R-CH-009 的成功判据（发送是否成功） | `ret=0` 是否等于手机收到源码读不出来；社区 C2 称缺 token 时接口"成功"但不推送 | 探针每步增加"手机实收条数"对账，以手机为准（补强，不改变既定步骤顺序） |

另：**提示词 C.1 的语义补充**——① 的"首次失败"以"API 失败**或**手机少收"中先发生者为准。

---

## 13. 需 M0 实测确认清单（汇总）

| # | 问题 | 为什么源码答不了 | 由谁/怎么测 |
| --- | --- | --- | --- |
| 1 | 主动发送窗口真实长度 | 源码无计时逻辑 | 探针 ③（1/6/12/20/23/25 h） |
| 2 | 单次入站后可发条数 N；失败的 `ret/errmsg` | 源码无计数 | 探针 ① |
| 3 | `-2` 的含义（窗口/速率/每 10 条） | 源码无 `-2` | 探针 ①③ 的 `errmsg` 对比 |
| 4 | 回复与主动是否共用配额 | 同一接口，无区分 | 探针 ④（源码+①） |
| 5 | GIF 是否会动 | 与图片同通道，无帧信息 | 探针 ② 询问用户 |
| 6 | "正在输入"是否可见；是否计数；无 `context_token` 时是否可用；票据有效期 | 源码只示范了"回复入站时" | 探针 ② 询问用户 |
| 7 | 出站引用 | 源码不支持（D-009） | 默认不测 |
| 8 | 空 `context_token` 文字/媒体是否可发 | 官方只告警；社区矛盾 | 可选实验 |
| 9 | ret=0 与手机实收是否一致 | 源码读不出 | 探针对账（D-016） |
| 10 | 入站各类型的真实形态：表情包、视频封面、语音转写覆盖率、仅 `svr_id` 的引用、`message_id` 稳定性/重放 | 源码只有类型 | 首次联调采集（§11） |
| 11 | `sendmessage`/`getupdates` 的真实响应体（含 `{}`、`message_id`） | 官方仅类型与 mock | 首次联调记录 |
| 12 | 长轮询服务端真实挂起时长；`longpolling_timeout_ms` 的取值 | 源码只有客户端侧 | 首次联调记录 |
| 13 | `bot_token` 的实际寿命、`-14` 出现条件 | 源码无到期时间 | 长期运行观察 |
| 14 | `iLink-App-*` 头是否可省 | A 不发仍可用 | 可选 |
| 15 | 用户在美国（`America/Chicago`）时本机到 `ilinkai.weixin.qq.com` 与 CDN 的可达性与延迟 | 沙箱不等于用户网络 | `twin doctor` 增加连通性检查（02b 建议）；首次登录时观察 |
| 16 | 出站单条文本/图片/文件大小上限 | 源码无 | 可选 |
| 17 | 媒体报道称"不支持 AI 主动推送"（C4）是否属实 | 非协议来源 | 由 1–4 的真机结果直接回答；若不可行按 R-CH-010 停止并报告 |

> 以上全部需要**用户在手机上扫码并运行探针**（已记入 `docs/PENDING_USER_ACTIONS.md` 第 02 轮一节）。在拿到真机数据之前，`docs/CHANNEL_REPORT.md` 只能是"待实测"模板。

---

## 14. 02b 的实现对照（实现者笔记，不是协议事实）

> 02b 按上文实现了通道；这里只记录"文档建议"与"实现"之间的**差别和补充**，协议事实以前面各节为准。所有行为都只用合成响应（respx）验证过，标 † 的响应形状仍待首次联调。

| 主题 | 文档建议 | 实现（`src/twin/channel/`） |
| --- | --- | --- |
| 模块 | §10.2 | 与建议一致，另加 `ilink/store.py`（`channel_state` 的类型化访问）、`ilink/auth.py`（登录失效处理）、`ilink/flows.py`（登录与绑定的交互）、`ilink/status.py`、`ilink/connectivity.py`、`component.py`（`twin run` 里的组件）、`console.py`（可注入的终端交互）。`ProbeSendPolicy` 留给 02c，通道侧只定义 `SendBypass` 协议（`base.py`） |
| 持久化键 | §10.3 | 在建议的键之外增加 `ilink.pending_bind`（未绑定时的第一个发送者）、`ilink.inbox`（已接受、未被消费的消息）、`ilink.item_stats`（最近入站 item 类型号与解析失败计数）、`ilink.poll_status`（连续失败次数与最近错误）；`ilink.probe_images` 由 `ProbeImageManifest` 读写 |
| 退避 | §10.4 | `min(60, 2^(n-1)) × U(0.5, 1.5)` 之后**再封顶 60 s**（抖动不会超过上限）；长轮询读超时不退避，但如果一次轮询在不足 1 s 内就返回空结果，补一个 1 s 的停顿，避免对"立刻返回"的服务器空转；客户端读超时 = `longpolling_timeout_ms`（默认 35 s）+ 5 s |
| 登录失效 | §8、§10.6 | 任何接口的 `ret`/`errcode == -14` → `NEEDS_RELOGIN`（`alerts` 一条 critical、控制台红色面板，一个周期只报一次）；每小时用旧 token 做一次恢复探测（成功或读超时视为恢复，写一条 info 告警）。**HTTP 401/403 不当作 -14**（源码没有这样的约定），按普通错误退避，连续 3 次写 warning |
| 发送 | §10.6 | 与表一致；补充：结果未知（读超时、5xx、坏响应）的发送**计入**条数，连接失败和被服务器拒绝的不计入；`ret=-2` 之后窗口保持关闭到下一条入站，不再发请求；文字 > 4000 字符或为空直接拒绝（不拆分、不发空串） |
| 入站 | §6.5 | 一个 `WeixinMessage` 去掉工具调用 item 后只有一个 item → `id = message_id`，多个 → `message_id#<序号>`；**文件不下载**，只取文件名（D-128）；窗口与 context_token 以"用户的任意消息"为准（即使消息里没有可交付的内容）；`create_time_ms` 同时接受毫秒与秒；引用的 `partial_text` 下标按 0 起算【推断：源码示例是 0，未被证实】，有 `quotemd5` 时用它在两种读法里选，没有时取"全局第 n 次" |
| 地址 | §7.2 | `baseurl`、`redirect_host`、`full_url`、`upload_full_url` 只接受 https |
| 绑定 | D-012 | 未绑定时消息不处理，只记录第一个发送者和它的 `context_token`；绑定时沿用该 token，所以 `send-test` 在绑定后立刻可用 |

### 14.1 02c/02d 的实现对照（实现者笔记，不是协议事实）

> 探针与本地通道按 §9.2、§10.8 实现；这里只记录和文档建议不同或补充的地方。设计决策见 `docs/DECISIONS.md` D-152～D-163。

| 主题 | 文档建议 | 实现（`src/twin/channel/probe/`、`local.py`） |
| --- | --- | --- |
| 探针令牌 | §10.8：探针发送显式传入探针令牌 | `ProbeSendPolicy` 实现 `SendBypass.authorize(BypassRequest)`；计划未运行、不在一次尝试中、文字不以 `[测试]` 开头时抛 `BypassRefused`，每次（允许或拒绝）写审计（`channel_state` 键 `probe.audit`，不记录非测试文字的内容）。可返回 `BypassGrant(empty_context_token=True)`，只供可选的空令牌实验 |
| 入站记录 | §10.8：把该入站的 `context_token` 记入探针记录 | 只记 token 的 SHA-256 前 8 位（`context_fp`），不存 token；入站时间取窗口状态里的 `last_inbound_at`（`create_time_ms` 与本机时间的较小者），测量点 = 该时间 + 小时数 |
| 失败记录 | §9.2：完整记录 `ret`/`errcode`/`errmsg` | `OutboundResult` 增加 `ret`、`errcode`（`code` 仍是文档 §3.1 优先级选出的那个）；`errmsg` 经 `redact_text`，截断到 200 字符；另记 HTTP 状态、`outcome`、距入站的小时数 |
| 手机对账 | §9.2：每步结束问"手机上实际收到几条" | 问题写在计划里，由 `twin channel probe answer` 回答并校验（数字不能超过接口接受的条数）；N、窗口下限都取手机的数 |
| 重试 | §9.2：失败即停 | 平台的回答（错误码、HTTP 错误）是测量结果，立即停该次尝试、不重试；没有得到回答的失败（网络、结果未知、登录失效、重启）使该次尝试作废并重做；"请求没发出去"的网络错误每分钟重试一次、最多 5 次（D-153） |
| 测试图片 | §10.8：Pillow 生成 JPG/PNG/多帧 GIF | 320×240；渐变底色 + 描边的 `[TEST] JPG/PNG/GIF` 字样；GIF 8 帧、一块黄色方块从左滑到右，循环播放；发送前按 SHA-256 登记到 `ilink.probe_images` |
| 正在输入 | §13 #6 | 先问用户是否在看手机，答复后发 `sendtyping` 并保持 30 秒再取消；`send_typing` 不返回成功与否，结论只看用户的回答 |
| 窗口测量点 | §9.2 的 1/6/12/20/23/25 小时 | 时刻 = 用户那条入站的时间 + 小时数，存在计划里，重启不漂移；机器睡过头时按实际经过的小时数判定，并记录迟到了多久 |
| 微信里的提示 | R-CH-009：终端和微信里都提示 | 尽力而为：平台还接收消息时才发得出去；第 1 步以平台拒绝结束后（`-2`，窗口被平台关闭，本地也不再发送），第 2 步的微信提示被本地拒绝，只有终端提示 |
| 本地控制台 | R-CH-011 | `LocalConsoleChannel`：同接口、同收件人守卫、同图片白名单，模拟窗口与条数用 `SessionWindow`；`twin chat --local` 启动只连本地通道的应用，引擎未接入时明说（D-162、D-163） |
