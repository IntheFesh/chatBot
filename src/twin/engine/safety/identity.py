"""Does the user ask whether she is an AI? (R-SAFE-003).

When the user sincerely asks "你是不是 AI / 机器人 / 真人", the bot must not deny that it is a
simulation - it may admit it in her voice.  The fixed rules of the prompt say so; the
post-processing needs to know the question was asked, because an honest answer contains words the
AI-phrase filter would otherwise delete ("我是模拟她聊天的样子做出来的").  This module only
recognises the *question*; the sincerity is the model's to judge, guided by the rules.
"""

from __future__ import annotations

import re

_AI = r"(?:AI|ai|Ai|人工智能|机器人|机器|程序|bot|Bot|chatbot|ChatGPT|聊天机器人|真人|真的人|人类)"
_GAP = r"[^，。,.！？?!\n]{0,4}"
PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rf"你(?:到底|究竟|其实|真的)?(?:是不是|是否|算不算|算是){_GAP}{_AI}",
        rf"你是{_GAP}{_AI}\s*(?:吗|么|嘛|吧|还是)",
        rf"是{_AI}(?:吗|么|嘛)",
        r"你是(?:人|人类|真人)还是(?:机器|AI|ai|程序|机器人|人工智能)",
        r"你(?:到底|究竟)?是(?:机器|AI|ai|程序|机器人|人工智能)还是(?:人|人类|真人)",
        rf"(?:是|有)(?:不是)?(?:{_AI})(?:在|替|帮|代)(?:你)?(?:回|聊|说|跟我)",
        r"(?:你|她)(?:本人)?是真的(?:她|本人)(?:吗|么|嘛)",
    )
)


def asks_if_ai(text: str) -> bool:
    """Does ``text`` ask whether the one answering is an AI, a robot, or not a real person?"""
    return any(pattern.search(text) for pattern in PATTERNS)
