"""When a chat reply thinks first (R-LLM-002, R-ENG-006, R-LLM-008).

The runtime setting ``thinking.chat`` (``/思考``) is ``off``, ``on`` or ``auto``.  ``auto`` thinks
when the round looks like it deserves it:

* the user **asks** something (a question mark, a question word),
* the message is **emotional** (a word of distress or affection, or stacked exclamation marks),
* the user's messages are **long** (more than :data:`LONG_TEXT_CHARS` characters together).

``thinking.auto_rules`` can switch those rules off, which makes ``auto`` the same as ``off``.
The budget has the last word: from degradation level 1 the DeepSeek client switches thinking off
whatever is asked (R-LLM-008), and the reply that comes back says what was really used.
"""

from __future__ import annotations

import re

from twin.config.runtime import ThinkingMode
from twin.profile.closing import QUESTION_WORDS
from twin.profile.textstats import is_question

LONG_TEXT_CHARS = 60
EMOTION_WORDS = (
    "难过",
    "伤心",
    "想哭",
    "哭了",
    "崩溃",
    "生气",
    "气死",
    "烦死",
    "烦人",
    "委屈",
    "讨厌",
    "受不了",
    "想你",
    "好想",
    "害怕",
    "担心",
    "焦虑",
    "压力",
    "失眠",
    "累死",
    "分手",
    "吵架",
    "对不起",
    "难受",
    "郁闷",
    "失望",
    "不舒服",
    "生病",
    "孤单",
    "孤独",
    "寂寞",
    "后悔",
    "爱你",
    "喜欢你",
    "谢谢你",
)
_STACKED_MARKS = re.compile(r"[！!]{2,}")


def wants_thinking(text: str) -> bool:
    """Does this round of the user's messages deserve a thinking reply (``auto`` rules)?"""
    if len(text) > LONG_TEXT_CHARS:
        return True
    if is_question(text) or any(word in text for word in QUESTION_WORDS):
        return True
    return bool(_STACKED_MARKS.search(text)) or any(word in text for word in EMOTION_WORDS)


def resolve_thinking(mode: ThinkingMode, text: str, *, auto_rules: bool = True) -> bool:
    """Whether to ask for thinking in this round under the setting ``mode``."""
    if mode == "on":
        return True
    if mode == "auto" and auto_rules:
        return wants_thinking(text)
    return False
