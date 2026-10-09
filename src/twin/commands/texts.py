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

# --------------------------------------------------------------------- the command table
SUMMARIES = {
    "帮助": "列出全部指令，或某个指令的用法",
    "状态": "时区、她此刻的状态、后端、费用、告警等一览",
    "思考": "聊天生成的思考模式；自动 = 提问、聊情绪或长消息时开",
    "显示思考": "是否把思考内容以系统消息发给你（调试用）",
    "后端": "切换生成后端（风格模型未启用或不健康时不能切到 style / hybrid）",
    "重来": "撤销她上一轮回复（记为负例）并重新生成",
}
SYNTAXES = {
    "帮助": "/帮助 [指令名]",
    "状态": "/状态",
    "思考": "/思考 开|关|自动",
    "显示思考": "/显示思考 开|关",
    "后端": "/后端 deepseek|style|hybrid",
    "重来": "/重来",
}
EXAMPLES = {
    "帮助": "/帮助 思考",
    "状态": "/状态",
    "思考": "/思考 自动",
    "显示思考": "/显示思考 开",
    "后端": "/后端 deepseek",
    "重来": "/重来",
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
