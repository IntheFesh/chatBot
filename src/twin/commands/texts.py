"""The words of the commands: every success, failure and usage text in one table (R-CMD-001).

A command answers in the system's voice, not in hers: every reply starts with ``PREFIX`` ("⚙️ ")
so it cannot be taken for a message of the bot's persona, and it is sent as it is - the persona's
post-processing never touches it.  The texts are here and nowhere else; the handlers fill the
``{fields}``, the tests read the table.
"""

from __future__ import annotations

PREFIX = "⚙️ "  # the gear sign, as a symbol, and a space

# ---------------------------------------------------------------------------- shared
USAGE = "用法：{syntax}\n例如：{example}"
USAGE_WITH_REASON = "参数不对：{reason}\n" + USAGE
FAILED = "这个指令没有执行成功，请稍后再试（详情在日志里）。"
UNKNOWN = "没有这个指令：/{name}"
UNKNOWN_HINT = "可用的指令：{names}\n发 /帮助 看每个指令的用法和例子。"

# ------------------------------------------------------------------------------ /帮助
HELP_HEADER = "可用的指令（发 /帮助 <指令名> 看用法和例子）："
HELP_GROUP = "【{group}】"
HELP_LINE = "{syntax} — {summary}"
HELP_ONE = "{syntax}\n作用：{summary}\n例如：{example}"
HELP_ALIASES = "也可以写成：{aliases}"

# ------------------------------------------------------------------------------ /状态
STATUS_NONE = "暂无"
STATUS_HEADER = "当前状态"
STATUS_TIME = "时区：{zone}，当地时间 {local}"
STATUS_HER = "她此刻：{state}，到 {until}（还有约 {left}）"
STATUS_HER_STATES = {
    "free": "空闲",
    "busy": "在忙",
    "sleep_edge": "快睡着或刚醒",
    "deep_sleep": "睡着了",
}
STATUS_BACKEND = "后端：{backend}"
STATUS_BACKEND_FALLBACK = "后端：{requested}（风格模型不可用，眼下由 deepseek 回复：{reason}）"
STATUS_BACKEND_BUDGET = "后端：deepseek（预算已到最后一级，眼下由风格模型回复）"
STATUS_THINKING = "思考模式：{mode}；显示思考：{show}"
STATUS_PROACTIVE_OFF = "主动消息：未启用（主动消息还没有上线）"
STATUS_PROACTIVE_QUOTA = "；今天计划 {quota} 条"
STATUS_PROACTIVE_NEXT = "；下一个已排的是{kind}，约 {at}"
STATUS_PROACTIVE_BLOCKED = "；现在发不出去：{reason}"
STATUS_PROACTIVE = "主动消息：每天 {low}-{high} 条（{on}），今天已发 {sent} 条"
STATUS_WINDOW = "平台窗口：剩余 {left}，还能发 {quota} 条；可主动 {proactive} 条"
STATUS_WINDOW_EXPIRED = "平台窗口：已过期（等你下一条消息就会重新开始）"
STATUS_COST = "今日费用：${spent:.4f} / ${budget:.2f}，缓存命中率 {cache}"
STATUS_COST_NO_CALLS = "今日费用：${spent:.4f} / ${budget:.2f}，今天还没有调用"
STATUS_BUDGET = "预算级别：{level}（{meaning}）"
STATUS_BUDGET_LEVELS = (
    "正常",
    "已关闭聊天思考",
    "例子减到 3 条、记忆预算减半",
    "已暂停主动消息",
    "最后一级：风格模型或最小上下文",
)
STATUS_ALERTS_HEADER = "最近告警："
STATUS_ALERT = "- {when} {title}"
STATUS_NO_ALERTS = "最近告警：没有"
STATUS_STYLE_NONE = "风格模型：还没有登记"
STATUS_STYLE_NOT_ACTIVE = "风格模型：已登记 {count} 个文件，还没有启用"
STATUS_STYLE_ACTIVE = "风格模型：{label}，{gate}，{health}"
STATUS_GATE_PASSED = "已通过上线门槛"
STATUS_GATE_FORCED = "未通过门槛（强制启用）"
STATUS_HEALTH_OK = "运行正常"
STATUS_HEALTH_BAD = "不可用（{detail}）"
STATUS_REMOTE_HOURLY = "提醒：风格模型在 AutoDL 远程实例上，按小时计费，不用时记得在控制台关机。"
STATUS_UNCLEANED = (
    "提醒：有 {count} 次训练还没有清理 AutoDL 上的数据，请在电脑上运行 twin train remote cleanup。"
)
STATUS_RETRAIN = "重训提醒：{text}"
STATUS_QUIET_WINDOW = (
    "等你说完：{configured} 秒；按你的连发习惯建议 {suggested} 秒（自适应：{adaptive}）"
)

# ------------------------------------------------------------------------------ /思考
THINK_SET = "聊天的思考模式已设为「{mode}」。"
THINK_BUDGET = "（预算已超：思考暂时被关闭，预算恢复后才会生效。）"
THINK_MODES = {"on": "开", "off": "关", "auto": "自动"}
THINK_AUTO_HINT = "自动 = 你提问、聊情绪或消息比较长时才思考。"

# ------------------------------------------------------------------------ /显示思考
SHOW_SET_ON = "之后的回复会附上思考内容（系统消息，调试用）。"
SHOW_SET_OFF = "不再展示思考内容。"

# ------------------------------------------------------------------------------ /后端
BACKEND_SET = "生成后端已切到「{name}」。"
BACKEND_STYLE_HINT = "风格模型出问题时会自动回到 deepseek 并通知你。"
BACKEND_REFUSED = "没有切换：{reason}"
BACKEND_REASONS = {
    "not_registered": "还没有登记风格模型，不能用 style 或 hybrid。",
    "not_active": "已登记的风格模型还没有启用，先在电脑上运行 twin model activate。",
    "gate_not_passed": (
        "当前启用的风格模型（{detail}）没有通过上线门槛，不能从手机上切换。"
        "确实要用，只能在电脑上运行 twin model activate --force 强制启用（会留审计记录）。"
    ),
    "unhealthy": "风格模型现在连不上或不健康（{detail}），恢复后再试。",
    "template": "风格模型绑定的提示词模板这个版本的程序渲染不了（{detail}），不能使用。",
    "no_model": "没有启用的风格模型。",
}

# ------------------------------------------------------------------------------ /重来
REDO_DONE = "好的，上一轮回复已撤销，重新来一次。"
REDO_UNDONE = "撤销了她那轮回复里编出来的 {count} 条小细节。"
REDO_UNSHARED = "她那条消息里说过的 {count} 个生活片段恢复成还没说过。"
REDO_SAFETY = "这一条不能重来：那是你提到难受的事时，跳出角色的关心回复，不会撤销。"
REDO_NOTHING = "现在没有可以重来的回复（还没有回复过，或者上一轮已经撤销、正在重来）。"

# ----------------------------------------------------------------------------- /时区
TZ_SHOW = "当前时区：{zone}，当地时间 {local}"
TZ_HER = "她现在：{state}，到 {until}"
TZ_SWITCHED = "时区已从 {old} 切到 {new}，当地时间 {local}。今天剩下的日程已按新时区重排。"
TZ_SAME = "已经是 {zone} 了，当地时间 {local}。"
TZ_UNKNOWN = (
    "不认识这个时区：{name}。可以写 Asia/Shanghai、America/Chicago，或者 北京、上海、芝加哥。"
)
TZ_SHOW_WORDS = ("查看", "show", "now", "当前")

# ----------------------------------------------------------------------- /暂停 和 /恢复
PAUSE_SET = (
    "好的，暂停到 {until}（{zone}，还有约 {left}）。期间不回复也不主动发消息，"
    "到点后她会像刚看到你的消息一样回复。发 /恢复 可以提前结束。"
)
PAUSE_UNREADABLE = "没看懂这个时长：{text}。可以写 2小时、30分钟、1天，或者 到明早、到22:00。"
PAUSE_TOO_LONG = "最长只能暂停 {hours} 小时。"
PAUSE_NOT_FUTURE = "这个时长不对：暂停的结束时间必须在现在之后。"
RESUME_DONE = "已恢复，她会正常回复，也会照常主动发消息。"
RESUME_NOT_PAUSED = "现在没有在暂停。"

# ------------------------------------------------------------------------------ /主动
PROACTIVE_RANGE = "主动消息每天 {low}-{high} 条，已设好（明天起按这个抽，今天剩下的日程已重排）。"
PROACTIVE_RANGE_OFF = "主动消息现在是关着的，发 /主动 开 才会发。"
PROACTIVE_ON = "主动消息已开启。"
PROACTIVE_OFF = "主动消息已关闭，不会再主动发消息；发 /主动 开 打开。"
PROACTIVE_BAD_RANGE = "范围要满足 0 ≤ 最少 ≤ 最多 ≤ 12，收到的是：{text}"
PROACTIVE_UNREADABLE = "没看懂：{text}。写成 最少-最多（例如 2-5），或者 开、关。"
PROACTIVE_ON_WORDS = ("开", "开启", "打开", "on")
PROACTIVE_OFF_WORDS = ("关", "关闭", "off")

# ------------------------------------------------------------------------------ /作息
ROUTINE_SLEEP_SET = "睡眠时间已设为每天 {start}–{end}（{zone}本地时间），记为第 {number} 条。"
ROUTINE_BUSY_SET = "已记下：{days} {start}–{end} 她在忙，记为第 {number} 条。"
ROUTINE_HOLIDAY_SET = "已记下：{first} 至 {last} 按假期算，记为第 {number} 条。"
ROUTINE_REBUILT = "今天剩下的日程已按新的作息重排。"
ROUTINE_REBUILD_LATER = "今天的日程这次没能立刻重排，稍后会自动重排。"
ROUTINE_LIST_HEADER = "作息修正（编号用于 /作息 删除 <编号>）："
ROUTINE_LIST_LINE = "{number}. {text}{state}"
ROUTINE_LIST_DISABLED = "（已停用）"
ROUTINE_EMPTY = "没有手动修正，她的作息完全来自聊天记录。"
ROUTINE_REMOVED = "已删除第 {number} 条：{text}。"
ROUTINE_NO_SUCH = "没有第 {number} 条，发 /作息 查看 看编号。"
ROUTINE_BAD_SLEEP = "睡眠时间没看懂：{reason}"
ROUTINE_BAD_BUSY = "忙碌时段没看懂：{reason}"
ROUTINE_BAD_HOLIDAY = "假期日期没看懂：{reason}"
ROUTINE_VIEW_WORDS = ("查看", "列表", "list", "show")
ROUTINE_DELETE_WORDS = ("删除", "删", "remove", "delete")
ROUTINE_SLEEP_WORDS = ("睡", "睡眠", "sleep")
ROUTINE_BUSY_WORDS = ("忙", "忙碌", "busy")
ROUTINE_HOLIDAY_WORDS = ("假期", "放假", "节假日", "holiday")

# ------------------------------------------------------------------- /记住 /忘掉 /记忆
REMEMBER_DONE = "记住了（第 {number} 条）：{text}"
REMEMBER_FOLLOWUPS = "，另外记下了 {count} 条待跟进"
REMEMBER_REPLACED = "，它取代了 {count} 条旧的"
REMEMBER_AS_WRITTEN = "（没能分析这句话，按原话存下）"
FORGET_DONE = "已删除："
FORGET_FACT = "第 {number} 条：{text}"
FORGET_FOLLOWUP = "待跟进：{text}"
FORGET_LIFELINE = "生活安排：{text}"
FORGET_RESTORED = "原来被它取代的第 {numbers} 条重新生效。"
FORGET_AMBIGUOUS = "有 {count} 条都符合，没有删除。请用编号再发一次，例如 /忘掉 {example}："
FORGET_FOLLOWUPS_AMBIGUOUS = "有多条待跟进都符合，没有删除。请说得更具体一些："
FORGET_NONE = "没有找到符合的记忆。发 /记忆 看看现在有哪些。"
MEMORY_EMPTY = "还没有记住什么。"
MEMORY_NO_MATCH = "没有包含「{word}」的记忆。"
MEMORY_LINE = "{number}. {text}{when}"
MEMORY_WHEN = "（{date}）"
MEMORY_FOLLOWUP = "待跟进：{text}"
MEMORY_FOOTER = "第 {page}/{pages} 页，共 {total} 条"
MEMORY_NEXT = "；发 /记忆 {page} 看下一页"
MEMORY_SNIPPET_CHARS = 40

# ------------------------------------------------------------------------------ /不像
NOT_LIKE_DONE = "记下了：上一轮不像她。"
NOT_LIKE_PAIR = "你给的说法已存为一对对照（现在共 {total} 对），以后用来教她改。"
NOT_LIKE_PAIR_AGAIN = "这个说法之前已经记过了。"
NOT_LIKE_PAIR_SKIPPED = "这一轮没有可以对照的上文（不是在回你的话），所以只记成了负例。"
NOT_LIKE_HINT = "想教她怎么说，可以发 /不像 她会怎么说。"
NOT_LIKE_WEEKLY = "这类反馈每周会被整理成“不要这样”的说话规则。"
NOT_LIKE_NOTHING = "现在没有可以标记的回复。"
NOT_LIKE_NOT_HERS = "这一条不是她平时的回复（是跳出角色的关心或系统消息），不记为“不像”。"
NOT_LIKE_NO_TEXT = "上一轮她没有说话，没有可以标记的内容。"

# ---------------------------------------------------------- the correction in plain words
CORRECTION_ASK = "要把这次记为“不像”吗？回复“是”确认。"
CORRECTION_DONE = "好的，上一轮已记为“不像”。"
CORRECTION_CONFIRM_WORDS = ("是", "是的", "确认")

# ------------------------------------------------------------------------------ /费用
COST_TITLE_DAY = "今天（{date}）的费用"
COST_TITLE_MONTH = "{month} 的费用"
COST_TOTAL = "合计 ${spent:.4f}，预算 ${budget:.2f}，还剩 ${left:.4f}（已用 {used}）"
COST_NO_CALLS = "这段时间还没有调用。"
COST_PURPOSES = "按用途：{items}"
COST_PURPOSE_ITEM = "{purpose} ${cost:.4f}（{share}）"
COST_CACHE = "缓存命中率：{ratio}（命中 {hit:,} / 共 {total:,} 个输入 token）"
COST_ONE_TIME = "一次性批任务另计：${spent:.4f}（不占预算）"
COST_LEVEL = "预算级别：{level}（{meaning}）"
COST_DAY_WORDS = ("今天", "今日", "today", "day")
COST_MONTH_WORDS = ("本月", "这个月", "当月", "month")
PURPOSE_NAMES = {
    "reply": "回复",
    "plan": "规划",
    "proactive": "主动消息",
    "extract": "抽取",
    "summary": "摘要",
    "persona": "人设",
    "caption": "看图",
    "sticker_tag": "表情包标签",
    "eval": "评估",
    "train_plan": "训练规划",
    "probe": "探针",
}

# ------------------------------------------------------------------------------ /导入
IMPORT_STARTED = (
    "已开始导入 {name}（任务 {run}，约 {total} 条消息）。完成后我会告诉你统计；"
    "进度发 /状态 就能看到。"
)
IMPORT_RESUMED = (
    "这批文件之前导入到一半，现在接着导入（任务 {run}）。完成后我会告诉你；发 /状态 看进度。"
)
IMPORT_NO_FOLDER = "找不到这个文件夹：{path}。请给聊天记录导出所在的目录。"
IMPORT_NEEDS_TARGET = (
    "还没有选定要模仿的会话（第一次导入要在电脑上选）。请在电脑上运行 twin import {path}。"
)
IMPORT_BAD_EXPORT = "这个文件夹不是可以导入的导出：{reason}"
IMPORT_NOT_IN_EXPORT = "已选定的会话不在这个导出里：{reason}"
IMPORT_BUSY = "已经有一次导入在进行（任务 {run}），等它结束再开始新的。"
IMPORT_DONE = (
    "导入完成：新增 {inserted} 条，重复 {duplicates} 条，更新 {updated} 条，"
    "保留原有 {kept} 条，无效 {invalid} 条，用时 {took}。画像、检索库和记忆回放等已排队处理。"
)
IMPORT_FAILED = (
    "导入没有完成（{status}）：已处理 {processed}/{total}。"
    "请在电脑上运行 twin import status 看原因，twin import --resume 接着导入。"
)
IMPORT_SUPERSEDED = "这次导入被新的一次取代了，不再继续。"
STATUS_IMPORT = "导入：任务 {run} {status}，{phase}，已处理 {processed}/{total}"
STATUS_REPLAY = "记忆回放：{text}"
STATUS_DPO = "{text}"

# --------------------------------------------------------------------- the command table
SUMMARIES = {
    "帮助": "列出全部指令，或某个指令的用法",
    "状态": "时区、她此刻的状态、后端、费用、告警等一览",
    "思考": "聊天生成的思考模式；自动 = 提问、聊情绪或长消息时开",
    "显示思考": "是否把思考内容以系统消息发给你（调试用）",
    "后端": "切换生成后端（风格模型未启用或不健康时不能切到 style / hybrid）",
    "重来": "撤销她上一轮回复（记为负例）并重新生成",
    "时区": "查看或切换机器人所在的时区（可以写 北京、芝加哥）",
    "暂停": "暂停回复和主动消息一段时间（几小时、到明早）",
    "恢复": "结束暂停",
    "主动": "主动消息每天的条数范围，或者开关",
    "作息": "手动修正她的睡眠、忙碌时段和假期",
    "记住": "让她记住一件事（来源是你说的，优先级最高）",
    "忘掉": "删除一条记忆，和只由它产生的待跟进、生活安排",
    "记忆": "分页看她现在记得的事，或者按关键词找",
    "不像": "标记她上一轮不像她；附上她会怎么说就存成一对对照",
    "费用": "今天或本月的费用、按用途拆分、缓存命中率、预算剩余",
    "导入": "导入新的聊天记录导出（给出电脑上的文件夹路径）",
}
SYNTAXES = {
    "帮助": "/帮助 [指令名]",
    "状态": "/状态",
    "思考": "/思考 开|关|自动",
    "显示思考": "/显示思考 开|关",
    "后端": "/后端 deepseek|style|hybrid",
    "重来": "/重来",
    "时区": "/时区 <IANA 名称>|查看",
    "暂停": "/暂停 <时长>",
    "恢复": "/恢复",
    "主动": "/主动 <最少>-<最多>|开|关",
    "作息": (
        "/作息 睡 <HH:MM>-<HH:MM> | 忙 <星期> <HH:MM>-<HH:MM> | 假期 <日期>[..<日期>] "
        "| 查看 | 删除 <编号>"
    ),
    "记住": "/记住 <内容>",
    "忘掉": "/忘掉 <内容或编号>",
    "记忆": "/记忆 [页码|关键词]",
    "不像": "/不像 [正确说法]",
    "费用": "/费用 [今天|本月]",
    "导入": "/导入 <路径>",
}
EXAMPLES = {
    "帮助": "/帮助 思考",
    "状态": "/状态",
    "思考": "/思考 自动",
    "显示思考": "/显示思考 开",
    "后端": "/后端 deepseek",
    "重来": "/重来",
    "时区": "/时区 北京",
    "暂停": "/暂停 2小时",
    "恢复": "/恢复",
    "主动": "/主动 2-5",
    "作息": "/作息 睡 01:00-08:30",
    "记住": "/记住 她下周三有考试",
    "忘掉": "/忘掉 3",
    "记忆": "/记忆 2",
    "不像": "/不像 那你早点睡吧",
    "费用": "/费用 本月",
    "导入": "/导入 D:\\聊天记录\\导出",
}

# ------------------------------------------------------------------------------ /评分
RATING_SUMMARY = "给她最近一周的主动消息和整体体验打分（1 到 5），可以附一句备注"
RATING_SYNTAX = "/评分 <1-5> [备注]"
RATING_EXAMPLE = "/评分 4 晚安发得很自然"
RATING_DONE = "记下了：{score}/5。"
RATING_NOTE_KEPT = "备注也存好了。"
RATING_WEEK = "最近 7 天一共评过 {count} 次，平均 {mean} 分。"
RATING_NO_NUMBER = "没有看到分数"
RATING_NOT_WHOLE = "分数要是整数"
RATING_RANGE = "分数要在 1 到 5 之间"
