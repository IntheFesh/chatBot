# 需求追踪（TRACEABILITY）

> 两张表：
> 1. **产品需求 → SPEC**：产品文档中的每个需求点映射到 SPEC 编号与负责轮次（轮次 00–16，另有 09b）。第 16 轮逐条填写"验证方式"（测试名或演示命令）。
> 2. **SPEC 需求实现表**：每个 `R-xxx` 一行，由各轮在完成时填写"状态 / 实现位置 / 测试"。`scripts/trace_check.py` 解析本表；格式不得改动（列顺序固定，状态取值：待实现 / 已实现）。"轮次"列与各轮提示词"必须实现的需求"一一对应；多轮共同负责的编号，由最后一个负责轮次改为"已实现"。
>
> "实现位置"写 `路径::符号`，多个用 `; ` 分隔；"测试"写 pytest 节点 id（如 `tests/unit/test_x.py::test_y`），多个用 `; ` 分隔。

## 表 1：产品需求 → SPEC
| 编号 | 产品文档需求点（原文要点） | 产品文档位置 | SPEC 编号 | 轮次 | 验证方式（第 16 轮填写） |
| --- | --- | --- | --- | --- | --- |
| P-01 | 只在你的 ClawBot 会话里出现，使用者只有你一个人 | 背景与目标 | R-SCOPE-001, R-CH-007 | 02, 09 |  |
| P-02 | 模仿对象：isSent=false 是她，isSent=true 是你 | 背景与目标 | R-SCOPE-002, R-IMP-007 | 03 |  |
| P-03 | 她已知情并授权 | 背景与目标、隐私 | R-SCOPE-003 | 00 |  |
| P-04 | 时区现在芝加哥，回国切中国，夏令时自动处理 | 背景与目标 | R-SCOPE-005, R-ACT-001, R-SCH-002, R-SCH-003 | 04, 08, 11 |  |
| P-05 | 运行在你当前的 Windows 电脑（不休眠），全部记录本地导入保存 | 背景与目标 | R-SCOPE-004, R-OPS-001, R-OPS-002, R-PRIV-001 | 00, 12, 16 |  |
| P-06 | 两个模型分工：DeepSeek 理解/记忆/规划/看图，AutoDL 微调风格模型作可切换后端 | 背景与目标 | R-SCOPE-007, R-ENG-006, R-SRV-004 | 09, 14 |  |
| P-07 | 体验目标优先级：像她 > 记得住 > 像真人 > 可控 > 会成长 | 背景与目标 | R-SCOPE-009 | 16 |  |
| P-08 | 非目标：通话、发语音、生成她的照片、以她的名义给别人发消息、群聊 | 背景与目标 | R-SCOPE-008, R-CH-007, R-IMP-003 | 02, 03, 16 |  |
| P-09 | 导出格式 messages.json + meta.json + media/，按稳定 id 去重增量导入 | 数据现状 | R-IMP-001, R-IMP-002, R-IMP-005 | 03 |  |
| P-10 | 各消息类型用法：文字主要模仿；表情包建库；引用作上下文；图片转描述且不发她的照片；通话/系统/转账/红包/链接转事件只进记忆；语音有转写才用 | 数据现状 | R-IMP-007, R-IMP-009, R-IMP-012, R-STK-002, R-SCOPE-008, R-SAFE-006 | 02, 03, 05, 06, 09, 09b, 13, 16 |  |
| P-11 | 她与你的风格指标差异就是要学的东西 | 数据现状 | R-PROF-002, R-EVAL-002 | 04, 09b |  |
| P-12 | 表情包文件大多缺失，按 CDN 地址先下载入库 | 数据现状 | R-IMP-008 | 03 |  |
| P-13 | createTimeText 为芝加哥时间，与机器人时区一致不用换算 | 数据现状 | R-ACT-001, R-CFG-004 (time.source_timezone) | 00, 04, 08 |  |
| P-14 | 这是部分记录；全部记录本地导入后统计与作息曲线重算 | 数据现状 | R-IMP-004, R-IMP-006, R-IMP-011, R-PROF-004 | 03, 04, 05, 06, 07, 13 |  |
| P-15 | 4 条语音没有转写 | 数据现状 | R-IMP-009 | 03 |  |
| P-16 | 主模型 deepseek-flash，支持看图，上下文 1M | 关键技术决策 | R-LLM-001, R-LLM-004 | 01 |  |
| P-17 | 离线任务可选 deepseek-v4-pro | 关键技术决策 | R-CFG-004 (deepseek.offline_model), R-PERS-001 | 00, 06 |  |
| P-18 | 学习方式：画像 + 检索 + 记忆，再加风格微调 | 关键技术决策 | R-PROF-*, R-RET-*, R-MEM-*, R-TRN-* | 04, 05, 06, 07, 08, 09, 09b, 10, 11, 13, 14 |  |
| P-19 | 风格模型：Qwen3 + LoRA，AutoDL 单卡 RTX 5090 或 RTX PRO 6000 | 关键技术决策 | R-TRN-001 | 13 |  |
| P-20 | 生成后端三选一；训好前默认仅 DeepSeek；混合后端中思考作用在规划 | 关键技术决策 | R-ENG-006, R-SRV-004, R-SRV-005, R-CFG-004 | 00, 09, 14 |  |
| P-21 | 思考模式参数开关，聊天默认关；主动消息决策默认开思考 | 关键技术决策 | R-LLM-002, R-PRO-006, R-CMD-002 | 01, 09, 10, 11 |  |
| P-22 | 微信通道：官方 ClawBot，直接对接 openclaw-weixin 协议 | 关键技术决策 | R-CH-002 | 02 |  |
| P-23 | 成本控制：固定内容放提示词前面；离线任务放非高峰 | 关键技术决策 | R-LLM-007, R-LLM-010, R-ENG-005, R-ARCH-003 | 00, 01, 09 |  |
| P-24 | 风格模型可本机 llama.cpp 推理，或留在 AutoDL 远程推理 | 关键技术决策 | R-SRV-002, R-SRV-003 | 14 |  |
| P-25 | F1 本地读取导出格式，支持多会话、多次导出 | F1 | R-IMP-001, R-IMP-003 | 03 |  |
| P-26 | F1 大文件流式、分批入库、中断续导、后台进度 | F1 | R-IMP-004, R-IMP-006, R-ARCH-006 | 00, 03, 12 |  |
| P-27 | F1 按 id 去重；原始记录只读，唯一事实来源 | F1 | R-IMP-005, R-STO-007, R-MEM-004 | 03, 07, 09 |  |
| P-28 | F1 表情包下载校验 md5，失败标记不阻塞 | F1 | R-IMP-008 | 03 |  |
| P-29 | F1 图片一句描述，默认最近 3 个月和被检索到的；通话等转事件文字 | F1 | R-IMP-007, R-IMP-012 | 03 |  |
| P-30 | F1 导入后触发 F2/F5/F6 更新，输出导入报告 | F1 | R-IMP-010, R-IMP-011 | 03, 04, 05, 06, 07, 13 |  |
| P-31 | F2 统计层（句长、标点、连发、间隔、延迟、表情、短句、称呼、活跃度） | F2 | R-PROF-002 | 04 |  |
| P-32 | F2 描述层：模型生成人设卡 | F2 | R-PERS-001 | 06 |  |
| P-33 | F2 画像版本号、对比、回滚；人设卡可手改且重算时保留 | F2 | R-PROF-004, R-PERS-002, R-PERS-003 | 04, 06 |  |
| P-34 | F3 等你说完（默认静默 15 秒） | F3 | R-ENG-002 | 09 |  |
| P-35 | F3 提示词顺序为命中缓存（SPEC 调整了顺序，见 R-ENG-005 说明） | F3 | R-ENG-005 | 09 |  |
| P-36 | F3 每行一条气泡；表情代码与表情包占位；去 AI 腔与书面标点；限长 | F3 | R-ENG-007, R-ENG-008 | 09 |  |
| P-37 | F3 不追求秒回；延迟长尾；忙时隔一阵或攒几条；连发间隔加打字时间；等待中你又发就取消重生成 | F3 | R-ENG-003, R-ENG-004, R-ENG-009 | 09 |  |
| P-38 | F3 兜底：超时出错晚点回，绝不发报错 | F3 | R-ENG-010 | 09 |  |
| P-39 | F4 微信表情代码直接发，频率与连用按画像 | F4 | R-STK-001 | 06 |  |
| P-40 | F4 表情包库：打标签、记录使用次数与语境、按标签与频率挑选 | F4 | R-STK-002, R-STK-003, R-STK-004, R-STK-005 | 06 |  |
| P-41 | F4 表情包以图片发送，GIF 是否会动在 M0 验证 | F4 | R-STK-007, R-CH-009 | 02, 09 |  |
| P-42 | F4 识别你发来的表情包 | F4 | R-STK-006 | 06 |  |
| P-43 | F5 近期对话最近 30 轮原文 | F5 | R-MEM-001 | 07, 09 |  |
| P-44 | F5 每日摘要（她起床前），真实与机器人分开 | F5 | R-MEM-002 | 07, 08 |  |
| P-45 | F5 事实库带来源、时间、置信度 | F5 | R-MEM-003 | 07 |  |
| P-46 | F5 冲突规则（真实 > 你说的 > 机器人编的）；编的生活细节也记 | F5 | R-MEM-004, R-MEM-005, R-MEM-011 | 07, 08 |  |
| P-47 | F5 按话题和日期检索相关记忆 | F5 | R-MEM-008 | 07 |  |
| P-48 | F6 按当地钟点存储发消息概率，运行时放到当前时区 | F6 | R-ACT-001, R-ACT-002, R-SCH-001 | 04, 08 |  |
| P-49 | F6 推断睡眠与忙碌，可手动修正，节假日单独设置 | F6 | R-ACT-003, R-ACT-004, R-ACT-005, R-CMD-002 | 04, 09, 10, 11 |  |
| P-50 | F6 只有真实模式，不做随叫随到 | F6 | R-SCOPE-006, R-ENG-003 | 09 |  |
| P-51 | F7 调度器每 5 分钟检查 | F7 | R-PRO-001 | 10 |  |
| P-52 | F7 触发类型：作息型、跟进型、沉默型、分享型 | F7 | R-PRO-004 | 10 |  |
| P-53 | F7 两步决定：规则概率决定发不发，模型决定说什么，也可决定不发 | F7 | R-PRO-006 | 10 |  |
| P-54 | F7 每天次数在范围内随机（默认 1–6）；间隔 ≥ 1 小时；没回最多追一条；睡眠默认不发，偶尔"睡不着/刚醒" | F7 | R-PRO-002, R-PRO-003, R-PRO-005 | 08, 10 |  |
| P-55 | F7 平台约束：约 24 小时窗口、约 10 条，超窗停止等你发消息 | F7 | R-CH-008, R-CH-009, R-PRO-003, R-ENG-009 | 02, 09, 10 |  |
| P-56 | F8 你在机器人里说的事实进入事实库和待跟进 | F8 | R-LRN-001, R-MEM-006, R-MEM-007 | 07, 09, 10, 11 |  |
| P-57 | F8 你的纠正记成反例和规则，写进"不要这样" | F8 | R-LRN-002, R-LRN-003 | 11 |  |
| P-58 | F8 机器人的回复永远不进入风格样本库 | F8 | R-LRN-004, R-RET-004, R-STO-007 | 03, 05, 09, 11 |  |
| P-59 | F9 /思考 开/关/自动，默认关 | F9 | R-CMD-002, R-LLM-002 | 01, 09, 10, 11 |  |
| P-60 | F9 思考内容不发给你，调试可 /显示思考 | F9 | R-LLM-002, R-CMD-002 | 01, 09, 10, 11 |  |
| P-61 | F10 指令：/时区 /暂停 /主动 /状态 /重来 /不像 /记住 /忘掉 /记忆 | F10 | R-CMD-002 | 09, 10, 11 |  |
| P-62 | F10 指令消息不进入记忆和学习 | F10 | R-CMD-001 | 09, 11 |  |
| P-63 | F10 名字和头像由你在 ClawBot 会话设置里自己改 | F10 | R-PRIV-006 | 02 |  |
| P-64 | F11 登录失效、DeepSeek 连续报错时邮件提醒 | F11 | R-OPS-004 | 02, 12 |  |
| P-65 | F11 每日费用统计和上限，超出自动降级 | F11 | R-LLM-008, R-OPS-005 | 01, 12 |  |
| P-66 | F11 每日加密备份；日志不记录完整聊天内容 | F11 | R-OPS-006, R-OPS-007 | 00, 12 |  |
| P-67 | F12 训练集：她的真实回复为目标；与线上同一套代码；带当时时间与当时已知记忆；表情包标记；连发换行 | F12 | R-TRN-002, R-TRN-003, R-TRN-011, R-TRN-013, R-MEM-010 | 04, 05, 06, 07, 09, 09b, 13, 14 |  |
| P-68 | F12 脱敏后上传 AutoDL，训完删除数据、释放实例 | F12 | R-TRN-007, R-TRN-008, R-TRN-009, R-PRIV-003 | 12, 13 |  |
| P-69 | F12 LLaMA-Factory LoRA；5090 训 8B LoRA 或 14B QLoRA；PRO 6000 训 14B–32B LoRA；按时间留出最近 10%（SPEC：最近 10% 为测试集，其前 5% 为验证集） | F12 | R-TRN-001, R-TRN-006, R-RET-003 | 04, 05, 13 |  |
| P-70 | F12 评估：留出集盲测与风格指标，和纯 DeepSeek 后端对比，赢了才上线 | F12 | R-EVAL-001, R-EVAL-002, R-EVAL-009, R-SRV-005 | 09b, 14 |  |
| P-71 | F12 部署：合并 LoRA 转 GGUF 量化，本机 llama.cpp；无合适显卡时 AutoDL 远程 | F12 | R-TRN-009, R-SRV-002, R-SRV-003 | 13, 14 |  |
| P-72 | F12 继续训练：新增超 10% 提醒重训；纠正攒够做 DPO | F12 | R-TRN-012, R-TRN-009 | 13 |  |
| P-73 | F12 机器人的回复永远不进训练集 | F12 | R-TRN-004, R-LRN-004 | 11, 13 |  |
| P-74 | 架构：人设引擎为核心，通道只收发，两个模型只负责生成，本地加密存储五类数据 | 系统架构 | R-ARCH-001, R-STO-006 | 00, 03, 04, 05, 06, 07, 08, 09, 09b, 10, 11, 12, 13 |  |
| P-75 | 收到消息流程：通道 → 回复编排 → 检索拼提示词 → 生成后处理 → 按节奏发出并写记忆 | 系统架构 | R-ENG-001～013 | 09, 14 |  |
| P-76 | 主动流程：每 5 分钟检查 → 命中后开思考决定 → 发出并记录 | 系统架构 | R-PRO-001～008 | 08, 10 |  |
| P-77 | 通道可替换（ClawBot 不能主动时换企业微信不影响其他部分） | 系统架构 | R-CH-001, R-CH-010 | 02 |  |
| P-78 | 时区是设置项、/时区 切换、切换当天不重复问候不跳过睡眠；离线任务按 UTC 避开高峰 | 补充考虑 | R-SCH-002, R-SCH-003, R-LLM-007 | 01, 08, 11 |  |
| P-79 | 越学越不像：风格样本只来自真实记录 | 补充考虑 | R-LRN-004, R-RET-004, R-TRN-004 | 05, 11, 13 |  |
| P-80 | 前后一致的生活线 | 补充考虑 | R-MEM-005, R-EVAL-004 | 07, 08, 15 |  |
| P-81 | 和真实的她冲突：导入后真实覆盖机器人编的；可用 /记住 补充 | 补充考虑 | R-MEM-011, R-MEM-009 | 07, 11 |  |
| P-82 | 做不到的事（电话、见面、转账、语音）不承诺，自然带过 | 补充考虑 | R-SAFE-002 | 09 |  |
| P-83 | 引用回复：M0 验证能否发送，不能就文字复述 | 补充考虑 | R-CH-009, R-ENG-007 | 02, 09 |  |
| P-84 | 你发的图片看图、语音用微信转写、视频看封面、文件看文件名 | 补充考虑 | R-ENG-013, R-CH-005 | 02, 09 |  |
| P-85 | 主动消息的平台窗口；不可行时备选企业微信 | 补充考虑 | R-CH-008, R-CH-010 | 02 |  |
| P-86 | 电脑要一直开着：关自动休眠；醒来不补发；登录持久化，掉线提醒 | 补充考虑 | R-OPS-002, R-SCH-005, R-CH-003, R-OPS-004 | 00, 02, 08, 09, 10, 12 |  |
| P-87 | 故障时延后回复，像"刚才没看手机"，不发报错 | 补充考虑 | R-ENG-010 | 09 |  |
| P-88 | 情绪安全：跳出角色，给出求助渠道 | 补充考虑 | R-SAFE-001 | 09, 12 |  |
| P-89 | 人设卡、画像、提示词模板都有版本，可回滚 | 补充考虑 | R-PROF-004, R-PERS-005, R-OPS-010 | 04, 06, 12 |  |
| P-90 | 她要求时一键删除全部数据（含训练集和模型） | 隐私 | R-OPS-008, R-PRIV-005 | 12 |  |
| P-91 | 数据留在你电脑上；只发少量片段给 DeepSeek，发前过滤手机号/地址/证件号/银行卡 | 隐私 | R-PRIV-001, R-PRIV-002, R-LLM-009 | 00, 01, 16 |  |
| P-92 | 训练数据上云：脱敏、训完删除、释放实例，只下载模型 | 隐私 | R-PRIV-003, R-TRN-008, R-TRN-009 | 12, 13 |  |
| P-93 | 本地加密：数据库和备份 | 隐私 | R-STO-002, R-STO-004, R-OPS-006, R-PRIV-004 | 00, 12 |  |
| P-94 | 只对你说话，不给任何其他人发消息，不以她的名义对外说话 | 隐私 | R-CH-007, R-SAFE-004 | 02, 09, 16 |  |
| P-95 | 机器人的话不代表她本人 | 隐私 | R-SAFE-004 | 09, 16 |  |
| P-96 | 成本：每月约 6–9 美元；全量导入约每 10 万条 1 美元；风格模型按小时租卡 | 成本估算 | R-LLM-006, R-OPS-005, R-MEM-010, R-TRN-005, R-EVAL-007 | 01, 07, 12, 13, 15 |  |
| P-97 | 每日费用上限防止意外超支 | 成本估算 | R-LLM-008 | 01 |  |
| P-98 | 盲测猜对率 ≤ 60% | 评估 | R-EVAL-001, R-EVAL-009, R-EVAL-010 | 09b, 10, 12, 14 |  |
| P-99 | 风格指标每项偏差 ±30% 以内 | 评估 | R-EVAL-002 | 09b |  |
| P-100 | 记忆测试 20 题正确率 ≥ 80% | 评估 | R-EVAL-003 | 09b, 11 |  |
| P-101 | 前后一致：每周明显矛盾 ≤ 1 次 | 评估 | R-EVAL-004 | 15 |  |
| P-102 | 主动消息：深睡时段 0 次；不超过每日范围；评分 ≥ 4/5 | 评估 | R-EVAL-005, R-CMD-002 (/评分) | 09, 10, 11 |  |
| P-103 | 稳定性：7 天不中断；掉线 10 分钟内提醒 | 评估 | R-EVAL-006, R-OPS-003, R-EVAL-010 | 09b, 10, 12, 14 |  |
| P-104 | 成本：每月 ≤ 15 美元 | 评估 | R-EVAL-007, R-LLM-008 | 01, 15 |  |
| P-105 | 里程碑 M0–M5 及各自过关条件 | 里程碑 | §26, R-CH-009, R-LLM-013, R-EVAL-010, R-EVAL-008 | 01, 02, 09b, 10, 12, 14, 15 |  |

## 表 2：SPEC 需求实现表

| 编号 | 摘要 | 轮次 | 状态 | 实现位置 | 测试 |
| --- | --- | --- | --- | --- | --- |
| R-SCOPE-001 | 单用户系统 | 02,09 | 待实现 |  |  |
| R-SCOPE-002 | 模仿对象是导出记录中目标会话的对方；isSent… | 03 | 待实现 |  |  |
| R-SCOPE-003 | 她已知情并授权（2026-10-08）。配置中记… | 00 | 已实现 | src/twin/config/loader.py::ensure_consent; src/twin/config/loader.py::ConsentError; src/twin/ops/process_model.py::command | tests/unit/test_config_settings.py::test_valid_consent_returns_the_date; tests/unit/test_config_settings.py::test_unquoted_yaml_date_is_accepted; tests/unit/test_config_settings.py::test_missing_consent_refuses_to_start; tests/unit/test_config_settings.py::test_invalid_consent_date_refuses_to_start; tests/unit/test_process_model.py::test_commands_refuse_to_run_without_valid_consent; tests/unit/test_doctor.py::test_config_check_reports_load_errors_and_missing_consent |
| R-SCOPE-004 | 运行环境 | 00,12 | 待实现 | src/twin/ops/instance_lock.py::InstanceLock; src/twin/ops/power.py::WindowsPowerManager; src/twin/ops/console.py::ensure_utf8; src/twin/config/loader.py::resolve_paths（部分：第 00 轮） | tests/unit/test_instance_lock.py::test_windows_backend_uses_named_mutexes_and_detects_existing_ones; tests/unit/test_power_console.py::test_windows_manager_sets_and_restores_the_execution_state; tests/unit/test_power_console.py::test_ensure_utf8_sets_the_windows_console_code_page; tests/unit/test_config_settings.py::test_relative_paths_are_anchored_at_the_project_root |
| R-SCOPE-005 | 机器人时区可切换 | 04,08,11 | 待实现 |  |  |
| R-SCOPE-006 | 只有"真实模式" | 09 | 待实现 |  |  |
| R-SCOPE-007 | 两个模型分工 | 09,14 | 待实现 |  |  |
| R-SCOPE-008 | 非目标（v1 不做，代码中也不得出现半成品） | 16 | 待实现 |  |  |
| R-SCOPE-009 | 体验目标按优先级 | 16 | 待实现 |  |  |
| R-ARCH-001 | 单进程 asyncio 应用 twin.app，… | 00 | 已实现 | src/twin/app.py::Application; src/twin/app.py::Component; src/twin/ops/components.py::build_application; src/twin/cli.py::run | tests/unit/test_lifecycle.py::test_components_start_in_dependency_order_and_stop_in_reverse; tests/unit/test_state_and_components.py::test_build_application_assembles_the_round_00_components; tests/unit/test_cli.py::test_serve_runs_the_application_until_a_termination_signal; tests/unit/test_cli.py::test_run_command_wires_logging_masked_config_runtime_settings_and_the_lock |
| R-ARCH-002 | 包划分见 CLAUDE.md §4；模块之间通过… | 00 | 已实现 | src/twin/__init__.py; src/twin/clock.py::Clock; src/twin/app.py::Component; src/twin/ops/jobs.py::OffPeakPolicy; src/twin/ops/alerts.py::AlertSink; src/twin/ops/power.py::PowerManager; src/twin/ops/winapi.py::Win32; src/twin/config/secrets.py::CredentialBackend | tests/unit/test_architecture.py::test_package_layout_matches_claude_md_section_4; tests/unit/test_architecture.py::test_module_boundaries_are_explicit_protocols; tests/unit/test_architecture.py::test_modules_form_an_acyclic_import_graph; tests/unit/test_architecture.py::test_low_level_modules_do_not_depend_on_the_application_layers; tests/unit/test_scan_rules.py::test_src_never_imports_the_tests_package |
| R-ARCH-003 | 离线任务（摘要、画像重算、图片描述回填、训练集计… | 00,01 | 待实现 | src/twin/ops/jobs.py::JobQueue; src/twin/ops/jobs.py::Worker; src/twin/ops/jobs.py::job_handler; src/twin/ops/jobs.py::OffPeakPolicy; src/twin/ops/jobs.py::DeferredOffPeakPolicy; src/twin/storage/models.py::Job; src/twin/ops/cli.py::jobs_list（部分：第 00 轮）（生产非高峰策略由第 01 轮 R-LLM-007 注册） | tests/unit/test_jobs.py::test_failed_jobs_retry_with_exponential_backoff_then_fail_with_an_alert; tests/unit/test_jobs.py::test_running_jobs_return_to_pending_after_a_restart; tests/unit/test_jobs.py::test_jobs_without_a_handler_stay_pending_and_are_not_touched; tests/unit/test_jobs.py::test_offpeak_only_jobs_follow_the_policy_and_the_deadline; tests/unit/test_jobs.py::test_unapproved_batch_jobs_are_never_claimed; tests/unit/test_cli.py::test_jobs_list_marks_jobs_without_a_handler |
| R-ARCH-004 | 任何单次对话处理中的异常不得使主循环退出；异常记… | 00,16 | 待实现 | src/twin/app.py::TaskSupervisor; src/twin/app.py::Application.stop; src/twin/ops/jobs.py::Worker._record_failure（部分：第 00 轮） | tests/unit/test_lifecycle.py::test_a_crash_in_one_component_does_not_disturb_another; tests/unit/test_lifecycle.py::test_a_crashing_task_is_restarted_with_exponential_backoff_and_alerts; tests/unit/test_lifecycle.py::test_a_failing_stop_does_not_prevent_the_others_from_stopping; tests/unit/test_jobs.py::test_handler_exception_does_not_stop_other_jobs |
| R-ARCH-005 | 提供完整的本地控制台通道 LocalConsol… | 02 | 待实现 |  |  |
| R-ARCH-006 | 进程模型（CLI 命令与常驻 twin run 并存） | 00,03,12 | 待实现 | src/twin/ops/process_model.py::command; src/twin/ops/process_model.py::CommandKind; src/twin/ops/process_model.py::undeclared_commands; src/twin/ops/process_model.py::enqueue_heavy; src/twin/ops/state_watch.py::StateWatcher; src/twin/storage/state.py::bump_state_version; src/twin/storage/db.py::Database.transaction（部分：第 00 轮） | tests/unit/test_process_model.py::test_every_cli_command_declares_its_process_model_class; tests/unit/test_process_model.py::test_exclusive_commands_refuse_while_either_instance_lock_is_held; tests/unit/test_process_model.py::test_twin_run_tolerates_the_supervisor_lock_but_not_a_second_run; tests/unit/test_process_model.py::test_light_commands_bump_state_version_with_their_writes; tests/unit/test_process_model.py::test_a_read_command_cannot_write; tests/unit/test_process_model.py::test_foreground_heavy_work_runs_with_the_same_worker_when_the_app_is_stopped; tests/unit/test_state_and_components.py::test_the_running_watcher_notices_a_change_within_two_seconds; tests/integration/test_multiprocess.py::test_two_processes_writing_jobs_and_settings_do_not_lose_updates; tests/integration/test_multiprocess.py::test_running_application_notices_a_cli_change_within_two_seconds_and_stops_gracefully |
| R-CFG-001 | 配置来源优先级 | 00 | 已实现 | src/twin/config/settings.py::Settings; src/twin/config/loader.py::load_settings; src/twin/config/loader.py::parse_overrides; src/twin/config/mask.py::masked_settings; src/twin/cli.py::run | tests/unit/test_config_settings.py::test_priority_cli_over_env_over_yaml_over_default; tests/unit/test_config_settings.py::test_unknown_yaml_keys_are_errors_at_every_level; tests/unit/test_config_settings.py::test_unknown_keys_via_env_and_command_line_are_errors; tests/unit/test_config_settings.py::test_effective_configuration_is_masked; tests/unit/test_cli.py::test_config_show_masks_personal_values_and_honours_overrides; tests/unit/test_cli.py::test_run_command_wires_logging_masked_config_runtime_settings_and_the_lock |
| R-CFG-002 | 秘密（DeepSeek Key、SMTP 密码、… | 00 | 已实现 | src/twin/config/secrets.py::SecretStore; src/twin/config/secrets.py::EncryptedFileKeyring; src/twin/config/secrets.py::select_backend; src/twin/config/cli.py::secrets_set; src/twin/config/cli.py::secrets_list | tests/unit/test_secrets.py::test_secret_store_set_get_exists_delete; tests/unit/test_secrets.py::test_file_backend_never_stores_plaintext; tests/unit/test_secrets.py::test_windows_never_falls_back_to_a_file; tests/unit/test_secrets.py::test_unusable_system_keyring_falls_back_on_linux_with_a_diagnosis; tests/unit/test_cli.py::test_secrets_set_list_check_delete_never_reveal_the_value; tests/unit/test_cli.py::test_secrets_set_prompts_with_hidden_input |
| R-CFG-003 | 运行时可变设置（时区、思考模式、后端、主动范围、… | 00 | 已实现 | src/twin/config/runtime.py::RuntimeSettings; src/twin/config/runtime.py::SettingSpec; src/twin/storage/settings_store.py::put_setting; src/twin/config/cli.py::settings_set | tests/unit/test_runtime_settings.py::test_initialize_seeds_from_config_once; tests/unit/test_runtime_settings.py::test_set_validates_and_persists_across_instances; tests/unit/test_runtime_settings.py::test_every_change_keeps_who_when_old_and_new; tests/unit/test_runtime_settings.py::test_history_values_are_encrypted_on_disk; tests/unit/test_runtime_settings.py::test_set_bumps_the_state_version_in_the_same_transaction; tests/unit/test_cli.py::test_settings_set_list_and_history |
| R-CFG-004 | 必须包含的配置项与默认值 | 00 | 已实现 | src/twin/config/settings.py::Settings; config/config.example.yaml | tests/unit/test_config_consistency.py::test_spec_block_example_file_and_settings_defaults_have_identical_keys_and_values; tests/unit/test_config_settings.py::test_defaults_without_any_source; tests/unit/test_config_settings.py::test_source_timezone_ranges_use_the_from_keyword |
| R-CFG-005 | 配置三处一致（Settings、示例配置、SPEC） | 00 | 已实现 | tests/unit/test_config_consistency.py::spec_block; config/config.example.yaml; config/lists/ai_phrases.txt; config/lists/commitment_patterns.txt; config/lists/crisis_keywords.txt; src/twin/config/lists.py::load_word_list | tests/unit/test_config_consistency.py::test_spec_block_example_file_and_settings_defaults_have_identical_keys_and_values; tests/unit/test_config_consistency.py::test_every_top_level_section_of_settings_is_documented_in_the_spec; tests/unit/test_config_consistency.py::test_configured_word_list_files_ship_with_the_repository; tests/unit/test_wordlists.py::test_ai_phrases_cover_the_spec_examples |
| R-STO-001 | SQLite（WAL、foreign_keys=… | 00 | 已实现 | src/twin/storage/db.py::Database; src/twin/storage/db.py::create_sqlite_engine; src/twin/storage/models.py::TimestampMixin; src/twin/storage/migrate.py::upgrade; src/twin/storage/migrations/versions/0001_core_tables.py | tests/unit/test_encrypted_models.py::test_sqlite_pragmas; tests/unit/test_encrypted_models.py::test_timestamps_are_utc_and_updated_at_moves; tests/unit/test_encrypted_models.py::test_utc_datetime_converts_offsets_and_rejects_naive; tests/unit/test_migrations.py::test_migration_and_models_are_in_sync; tests/unit/test_migrations.py::test_every_table_has_utc_timestamps_and_documented_columns |
| R-STO-002 | 敏感文本字段（消息正文、原始 JSON、记忆文本… | 00 | 已实现 | src/twin/storage/types.py::EncryptedText; src/twin/storage/types.py::EncryptedJSON; src/twin/storage/types.py::SealedAttr; src/twin/storage/crypto.py::KeyRing; src/twin/storage/crypto.py::build_aad | tests/unit/test_crypto.py::test_seal_open_round_trip_and_format; tests/unit/test_crypto.py::test_every_seal_uses_a_fresh_nonce; tests/unit/test_crypto.py::test_ciphertext_cannot_move_to_another_table_row_or_column; tests/unit/test_crypto.py::test_round_trip_property; tests/unit/test_encrypted_models.py::test_json_round_trip_and_ciphertext_on_disk; tests/unit/test_encrypted_models.py::test_ciphertext_copied_to_another_row_fails_to_decrypt; tests/unit/test_encrypted_models.py::test_ciphertext_moved_between_columns_or_tables_fails; tests/unit/test_encrypted_models.py::test_plaintext_cannot_be_written_to_an_encrypted_column |
| R-STO-003 | 主密钥 256 位，首次运行生成并存入 keyr… | 00,12 | 待实现 | src/twin/storage/keystore.py::KeyStore; src/twin/storage/crypto.py::KeyRing; src/twin/storage/rotate.py::rotate_db_key; src/twin/config/cli.py::secrets_rotate_db_key（部分：第 00 轮） | tests/unit/test_keystore_rotation.py::test_first_run_generates_a_256_bit_key_in_the_credential_store; tests/unit/test_keystore_rotation.py::test_a_missing_key_with_existing_data_is_never_silently_replaced; tests/unit/test_keystore_rotation.py::test_rotation_reencrypts_everything_and_retires_the_old_key; tests/unit/test_keystore_rotation.py::test_old_ciphertext_and_backups_remain_readable_with_the_retired_key; tests/unit/test_keystore_rotation.py::test_an_interrupted_rotation_can_be_resumed; tests/unit/test_cli.py::test_rotate_db_key_command |
| R-STO-004 | 媒体文件（她的照片、语音、视频、头像）加密存储于… | 00 | 已实现 | src/twin/storage/media.py::MediaStore; src/twin/storage/media.py::MediaReader | tests/unit/test_media_store.py::test_put_and_read_round_trip_with_sha256_file_name; tests/unit/test_media_store.py::test_stored_file_contains_no_plaintext; tests/unit/test_media_store.py::test_large_file_streams_without_holding_it_in_memory; tests/unit/test_media_store.py::test_tampering_is_detected; tests/unit/test_media_store.py::test_temp_file_decrypts_then_wipes_and_deletes; tests/unit/test_media_store.py::test_rekey_moves_files_to_the_current_key_keeping_name_and_content |
| R-STO-005 | 向量库（LanceDB）只存向量、行 id、非敏… | 00,05 | 待实现 | src/twin/storage/vector_schema.py::validate_record; src/twin/storage/vector_schema.py::VectorRow（部分：第 00 轮） | tests/unit/test_vector_schema.py::test_any_extra_column_such_as_plaintext_is_refused; tests/unit/test_vector_schema.py::test_a_row_converts_to_a_valid_record; tests/unit/test_vector_schema.py::test_malformed_fields_are_refused |
| R-STO-006 | 必须的表（字段在实现中细化，字段名用英文） | 00,03,04,05,06,07,08,09,09b,10,11,12,13 | 待实现 | src/twin/storage/models.py::Setting; src/twin/storage/models.py::Job; src/twin/storage/models.py::CostLedger; src/twin/storage/models.py::Alert; src/twin/storage/models.py::ChannelState（部分：第 00 轮） | tests/unit/test_migrations.py::test_round_00_migration_creates_exactly_the_five_tables; tests/unit/test_migrations.py::test_every_table_has_utc_timestamps_and_documented_columns |
| R-STO-007 | messages 与 bot_turns 是两张… | 03,09 | 待实现 |  |  |
| R-CH-001 | 通道接口 Channel | 02 | 待实现 |  |  |
| R-CH-002 | IlinkChannel | 02 | 待实现 |  |  |
| R-CH-003 | 登录 | 02 | 待实现 |  |  |
| R-CH-004 | 长轮询 | 02 | 待实现 |  |  |
| R-CH-005 | 入站解析 | 02 | 待实现 |  |  |
| R-CH-006 | 出站 | 02 | 待实现 |  |  |
| R-CH-007 | 收件人绑定 | 02 | 待实现 |  |  |
| R-CH-008 | 会话窗口跟踪 | 02 | 待实现 |  |  |
| R-CH-009 | 实测探针 twin channel probe（… | 02 | 待实现 |  |  |
| R-CH-010 | 若实测证明主动发送不可行（窗口过短或条数不足以满… | 02 | 待实现 |  |  |
| R-CH-011 | LocalConsoleChannel | 02 | 待实现 |  |  |
| R-LLM-001 | DeepSeekClient | 01 | 待实现 |  |  |
| R-LLM-002 | 思考开关 | 01 | 待实现 |  |  |
| R-LLM-003 | JSON 输出 | 01 | 待实现 |  |  |
| R-LLM-004 | 看图 | 01 | 待实现 |  |  |
| R-LLM-005 | 可靠性 | 01 | 待实现 |  |  |
| R-LLM-006 | 费用记账 | 01 | 待实现 |  |  |
| R-LLM-007 | 高峰判定 | 01 | 待实现 |  |  |
| R-LLM-008 | 预算 | 01 | 待实现 |  |  |
| R-LLM-009 | 脱敏 twin.llm.redaction | 01 | 待实现 |  |  |
| R-LLM-010 | 提示词缓存布局 | 01,09 | 待实现 |  |  |
| R-LLM-011 | StyleModelClient | 01,09,14 | 待实现 |  |  |
| R-LLM-012 | token 估算 | 01 | 待实现 |  |  |
| R-LLM-013 | DeepSeek 实测探针 twin llm p… | 01 | 待实现 |  |  |
| R-LLM-014 | 一次性任务预算 | 01,03,06,07,09b,13 | 待实现 |  |  |
| R-IMP-001 | 输入为一个导出根目录 | 03 | 待实现 |  |  |
| R-IMP-002 | 按 schemaVersion=1 定义 pyd… | 03 | 待实现 |  |  |
| R-IMP-003 | 只导入目标会话（target.username）… | 03 | 待实现 |  |  |
| R-IMP-004 | messages.json 用 ijson 流式… | 03 | 待实现 |  |  |
| R-IMP-005 | 以消息 id 幂等 upsert；同 id 内容… | 03 | 待实现 |  |  |
| R-IMP-006 | 可续传 | 03 | 待实现 |  |  |
| R-IMP-007 | 归一化 | 03 | 待实现 |  |  |
| R-IMP-008 | 媒体 | 03 | 待实现 |  |  |
| R-IMP-009 | 语音无转写时记为 [语音 N 秒，未转写]；不在… | 03 | 待实现 |  |  |
| R-IMP-010 | 导入报告（不含任何消息正文） | 03 | 待实现 |  |  |
| R-IMP-011 | 导入完成后自动排队 | 03,04,05,06,07,13 | 待实现 |  |  |
| R-IMP-012 | 图片描述 | 03 | 待实现 |  |  |
| R-IMP-013 | 性能 | 03 | 待实现 |  |  |
| R-IMP-014 | 结构探查 twin import inspect | 03 | 待实现 |  |  |
| R-PROF-001 | 只用她（以及用于对比的用户）的真实消息计算，以"… | 04 | 待实现 |  |  |
| R-PROF-002 | 必须计算的指标（她与用户各一份） | 04 | 待实现 |  |  |
| R-PROF-003 | 双窗口 | 04 | 待实现 |  |  |
| R-PROF-004 | 版本化 | 04 | 待实现 |  |  |
| R-PROF-005 | 从统计层自动生成"数字风格规则"文本（例如"几乎… | 04 | 待实现 |  |  |
| R-ACT-001 | 时间语义 | 04,08 | 待实现 |  |  |
| R-ACT-002 | 按"当地钟点（15 分钟粒度）× 日类型（工作日… | 04 | 待实现 |  |  |
| R-ACT-003 | 睡眠推断 | 04 | 待实现 |  |  |
| R-ACT-004 | 忙碌推断 | 04 | 待实现 |  |  |
| R-ACT-005 | 手动修正（指令 R-CMD） | 04,11 | 待实现 |  |  |
| R-ACT-006 | 版本化并随 R-PROF 一起重算；twin p… | 04 | 待实现 |  |  |
| R-PERS-001 | 由 DeepSeek（offline_model… | 06 | 待实现 |  |  |
| R-PERS-002 | 结构化存储（Markdown 分区） | 06 | 待实现 |  |  |
| R-PERS-003 | 版本化、差异展示、回滚 | 06 | 待实现 |  |  |
| R-PERS-004 | 两个渲染版本 | 06 | 待实现 |  |  |
| R-PERS-005 | 人设卡、提示词模板文件都有版本号；setting… | 06 | 待实现 |  |  |
| R-STK-001 | 微信表情代码 | 06 | 待实现 |  |  |
| R-STK-002 | 表情包库 | 06 | 待实现 |  |  |
| R-STK-003 | 打标签 | 06 | 待实现 |  |  |
| R-STK-004 | 发送选择 | 06 | 待实现 |  |  |
| R-STK-005 | 频率控制 | 06 | 待实现 |  |  |
| R-STK-006 | 识别用户发来的表情包 | 06 | 待实现 |  |  |
| R-STK-007 | 发送通过通道图片接口；GIF 动图效果以 R-C… | 09 | 待实现 |  |  |
| R-RET-001 | 例子窗口 | 05 | 待实现 |  |  |
| R-RET-002 | 向量 | 05 | 待实现 |  |  |
| R-RET-003 | 留出集 | 04,05 | 待实现 |  |  |
| R-RET-004 | 只收她的真实回复；索引构建函数的输入类型只接受 … | 05 | 待实现 |  |  |
| R-RET-005 | 查询 | 05 | 待实现 |  |  |
| R-RET-006 | 增量更新 | 05 | 待实现 |  |  |
| R-MEM-001 | 近期对话 | 07,09 | 待实现 |  |  |
| R-MEM-002 | 每日摘要 | 07,08 | 待实现 |  |  |
| R-MEM-003 | 事实库 | 07 | 待实现 |  |  |
| R-MEM-004 | 冲突规则 | 07 | 待实现 |  |  |
| R-MEM-005 | 生活线 | 07,08 | 待实现 |  |  |
| R-MEM-006 | 待跟进 | 07,10 | 待实现 |  |  |
| R-MEM-007 | 抽取 | 07,09 | 待实现 |  |  |
| R-MEM-008 | 检索 | 07 | 待实现 |  |  |
| R-MEM-009 | 指令 | 07,11 | 待实现 |  |  |
| R-MEM-010 | 时间回放 | 07 | 待实现 |  |  |
| R-MEM-011 | 机器人编的事实与生活线永远标记 bot_inve… | 07 | 待实现 |  |  |
| R-ENG-001 | 会话状态机 | 09 | 待实现 |  |  |
| R-ENG-002 | 等用户说完 | 09 | 待实现 |  |  |
| R-ENG-003 | DECIDING | 09 | 待实现 |  |  |
| R-ENG-004 | 首条延迟 | 09 | 待实现 |  |  |
| R-ENG-005 | 提示词组装顺序（为缓存） | 09 | 待实现 |  |  |
| R-ENG-006 | 生成后端 | 09,14 | 待实现 |  |  |
| R-ENG-007 | 输出约定 | 09 | 待实现 |  |  |
| R-ENG-008 | 后处理 | 09 | 待实现 |  |  |
| R-ENG-009 | 发送节奏 | 09 | 待实现 |  |  |
| R-ENG-010 | 失败兜底 | 09 | 待实现 |  |  |
| R-ENG-011 | 机器人会话记录 | 09 | 待实现 |  |  |
| R-ENG-012 | 占位符外泄检测 | 09 | 待实现 |  |  |
| R-ENG-013 | 用户发图片 | 09 | 待实现 |  |  |
| R-SAFE-001 | 危机识别 | 09,12 | 待实现 |  |  |
| R-SAFE-002 | 做不到的事 | 09 | 待实现 |  |  |
| R-SAFE-003 | 用户真诚地问"你是不是 AI/机器人"时，不否认… | 09 | 待实现 |  |  |
| R-SAFE-004 | 机器人说的话不代表她本人；不得以她的名义对外联系… | 09,16 | 待实现 |  |  |
| R-SAFE-005 | 内容遵守 DeepSeek 使用政策；生成被拒绝… | 09 | 待实现 |  |  |
| R-SAFE-006 | 不发她的照片、不输出事件文字 | 02,03,05,09,09b,13 | 待实现 |  |  |
| R-SCH-001 | TimeService | 08 | 待实现 |  |  |
| R-SCH-002 | 时区切换 | 08,11 | 待实现 |  |  |
| R-SCH-003 | 夏令时 | 08 | 待实现 |  |  |
| R-SCH-004 | 每日计划 | 08 | 待实现 |  |  |
| R-SCH-005 | 电脑重启或进程中断后 | 08,09,10 | 待实现 |  |  |
| R-PRO-001 | 调度器每 tick_minutes（5 分钟）评… | 10 | 待实现 |  |  |
| R-PRO-002 | 每天主动次数在 [daily_min, dail… | 08,10 | 待实现 |  |  |
| R-PRO-003 | 硬约束 | 10 | 待实现 |  |  |
| R-PRO-004 | 触发类型与优先级 | 10 | 待实现 |  |  |
| R-PRO-005 | 入睡/将醒边缘 | 10 | 待实现 |  |  |
| R-PRO-006 | 两步决定 | 10 | 待实现 |  |  |
| R-PRO-007 | 发送节奏同 R-ENG-009（可多条连发，按她… | 10 | 待实现 |  |  |
| R-PRO-008 | 审计日志 proactive_log | 10 | 待实现 |  |  |
| R-CMD-001 | 以 / 开头且命中指令表的消息为指令 | 09,11 | 待实现 |  |  |
| R-CMD-002 | 指令表（全部必须实现） | 09,10,11 | 待实现 |  |  |
| R-CMD-003 | 指令解析容错 | 11 | 待实现 |  |  |
| R-LRN-001 | 用户在机器人会话中说的事实与待跟进按 R-MEM… | 11 | 待实现 |  |  |
| R-LRN-002 | 反馈 | 11 | 待实现 |  |  |
| R-LRN-003 | 纠正汇总进人设卡 [不要这样] 区块 | 11 | 待实现 |  |  |
| R-LRN-004 | 机器人回复永远不进入风格样本、检索库与 SFT … | 11 | 待实现 |  |  |
| R-OPS-001 | Windows 安装 | 12 | 待实现 |  |  |
| R-OPS-002 | 运行期间调用 SetThreadExecutio… | 00,12 | 待实现 | src/twin/ops/power.py::WindowsPowerManager; src/twin/ops/power.py::NotNeededPowerManager; src/twin/ops/instance_lock.py::InstanceLock; src/twin/app.py::ShutdownSignals; src/twin/app.py::Application.run（部分：第 00 轮） | tests/unit/test_power_console.py::test_windows_manager_sets_and_restores_the_execution_state; tests/unit/test_power_console.py::test_non_windows_manager_is_a_real_implementation_that_records_why; tests/unit/test_instance_lock.py::test_second_instance_with_the_same_name_cannot_start; tests/unit/test_instance_lock.py::test_windows_backend_uses_named_mutexes_and_detects_existing_ones; tests/unit/test_instance_lock.py::test_a_lock_whose_object_is_dropped_is_freed; tests/unit/test_instance_lock.py::test_dropping_a_windows_lock_closes_its_mutex_handle_exactly_once; tests/unit/test_lifecycle.py::test_posix_signals_set_the_stop_event; tests/unit/test_lifecycle.py::test_windows_close_events_block_until_shutdown_finished; tests/integration/test_multiprocess.py::test_running_application_notices_a_cli_change_within_two_seconds_and_stops_gracefully |
| R-OPS-003 | 健康检查（每分钟） | 12 | 待实现 |  |  |
| R-OPS-004 | 告警 | 02,12 | 待实现 |  |  |
| R-OPS-005 | 费用 | 01,12 | 待实现 |  |  |
| R-OPS-006 | 备份 | 12 | 待实现 |  |  |
| R-OPS-007 | 日志 | 00,12 | 待实现 | src/twin/ops/logging.py::configure_logging; src/twin/ops/logging.py::StructLogger; src/twin/ops/logging.py::JsonFormatter; src/twin/llm/redaction.py::redact（部分：第 00 轮） | tests/unit/test_logging.py::test_info_logs_never_contain_content; tests/unit/test_logging.py::test_debug_logs_keep_content_but_redacted; tests/unit/test_logging.py::test_rotation_is_ten_megabytes_times_ten_files; tests/unit/test_logging.py::test_file_rotation_actually_happens; tests/unit/test_scan_rules.py::test_application_code_logs_through_the_structured_logger_only |
| R-OPS-008 | 一键删除 | 12 | 待实现 |  |  |
| R-OPS-009 | 首次运行与升级 | 00,12 | 待实现 | src/twin/ops/doctor.py::run_checks; src/twin/ops/doctor.py::doctor_check; src/twin/cli.py::doctor（部分：第 00 轮） | tests/unit/test_doctor.py::test_all_checks_pass_on_a_healthy_setup; tests/unit/test_doctor.py::test_keyring_check_variants; tests/unit/test_doctor.py::test_data_dir_free_space_thresholds; tests/unit/test_doctor.py::test_database_check_states; tests/unit/test_cli.py::test_doctor_passes_on_a_fresh_machine_and_diagnoses_the_keyring |
| R-OPS-010 | 版本与回滚 | 12 | 待实现 |  |  |
| R-TRN-001 | 基座模型（可配置，默认按 GPU 档位） | 13 | 待实现 |  |  |
| R-TRN-002 | 训练集导出 twin train export | 13 | 待实现 |  |  |
| R-TRN-003 | 表示约定 | 13 | 待实现 |  |  |
| R-TRN-004 | 只用她的真实消息作目标；机器人回复不得进入（导出… | 13 | 待实现 |  |  |
| R-TRN-005 | hybrid 训练数据 | 13 | 待实现 |  |  |
| R-TRN-006 | 切分 | 13 | 待实现 |  |  |
| R-TRN-007 | 脱敏 | 13 | 待实现 |  |  |
| R-TRN-008 | 训练包 twin train bundle --… | 13 | 待实现 |  |  |
| R-TRN-009 | AutoDL 脚本（training/autod… | 13 | 待实现 |  |  |
| R-TRN-010 | 本地编排 twin train remote .… | 13 | 待实现 |  |  |
| R-TRN-011 | 模板一致性测试 | 09,13,14 | 待实现 |  |  |
| R-TRN-012 | 继续训练 | 13 | 待实现 |  |  |
| R-TRN-013 | 防止未来信息泄露（训练与评估共用） | 04,05,06,07,09b,13 | 待实现 |  |  |
| R-SRV-001 | 模型登记 model_registry | 13,14 | 待实现 |  |  |
| R-SRV-002 | 本地推理 | 14 | 待实现 |  |  |
| R-SRV-003 | 远程推理（无合适本地显卡时） | 14 | 待实现 |  |  |
| R-SRV-004 | 后端选择 | 14 | 待实现 |  |  |
| R-SRV-005 | 上线门槛 | 14 | 待实现 |  |  |
| R-EVAL-001 | 盲测 twin eval blind --bac… | 09b,14 | 待实现 |  |  |
| R-EVAL-002 | 风格指标 | 09b | 待实现 |  |  |
| R-EVAL-003 | 记忆测试 | 09b,11 | 待实现 |  |  |
| R-EVAL-004 | 前后一致 | 15 | 待实现 |  |  |
| R-EVAL-005 | 主动消息审计 | 10 | 待实现 |  |  |
| R-EVAL-006 | 稳定性 | 12 | 待实现 |  |  |
| R-EVAL-007 | 成本 | 15 | 待实现 |  |  |
| R-EVAL-008 | twin eval report 汇总以上全部，… | 15 | 待实现 |  |  |
| R-EVAL-009 | 评估沙盒 EvalSandbox | 09b | 待实现 |  |  |
| R-EVAL-010 | 里程碑门槛判定 twin eval gate | 09b,10,12,14 | 待实现 |  |  |
| R-PRIV-001 | 全部真实聊天记录在本地导入和保存，不上传；仓库不… | 00,16 | 待实现 | scripts/privacy_scan.py::scan_text; scripts/privacy_scan.py::main; .pre-commit-config.yaml; .github/workflows/ci.yml; .gitignore（部分：第 00 轮） | tests/unit/test_gate_scripts.py::test_sensitive_patterns_are_found_without_echoing_them; tests/unit/test_gate_scripts.py::test_the_repository_itself_is_clean; tests/unit/test_scan_rules.py::test_gitignore_covers_the_private_paths |
| R-PRIV-002 | 每次发往 DeepSeek 的只有当次需要的少量… | 01 | 待实现 |  |  |
| R-PRIV-003 | 训练数据上云 | 12,13 | 待实现 |  |  |
| R-PRIV-004 | 本地加密 | 00 | 已实现 | src/twin/storage/types.py::EncryptedJSON; src/twin/storage/crypto.py::KeyRing; src/twin/storage/media.py::MediaStore | tests/unit/test_encrypted_models.py::test_json_round_trip_and_ciphertext_on_disk; tests/unit/test_media_store.py::test_stored_file_contains_no_plaintext; tests/unit/test_secrets.py::test_file_backend_never_stores_plaintext |
| R-PRIV-005 | 一键删除（R-OPS-008）覆盖原始记录、画像… | 12 | 待实现 |  |  |
| R-PRIV-006 | 机器人只对用户说话（R-CH-007）；名字与头… | 02 | 待实现 |  |  |
| R-NFR-001 | 生成耗时（不含刻意延迟）p95 < 20 秒（d… | 16 | 待实现 |  |  |
| R-NFR-002 | 常驻内存 < 1.5GB（不含 llama-se… | 16 | 待实现 |  |  |
| R-NFR-003 | 启动到可收消息 < 30 秒（向量模型懒加载）。 | 05,16 | 待实现 |  |  |
| R-NFR-004 | 代码质量 | 00,16 | 待实现 | scripts/coverage_gate.py::evaluate; scripts/check.ps1; .github/workflows/ci.yml; pyproject.toml（部分：第 00 轮） | tests/unit/test_gate_scripts.py::test_gate_fails_for_a_weak_package_even_if_the_total_is_high; tests/unit/test_gate_scripts.py::test_gate_passes_when_total_and_every_package_meet_the_thresholds; tests/unit/test_scan_rules.py::test_no_stub_or_toy_markers_in_production_code; tests/unit/test_scan_rules.py::test_no_function_has_an_empty_body; tests/unit/test_scan_rules.py::test_no_direct_clock_access_outside_the_clock_module |
| R-NFR-005 | 所有时间相关逻辑有基于注入时钟的测试，包括 DS… | 16 | 待实现 |  |  |
| R-NFR-006 | 文档 | 16 | 待实现 |  |  |
