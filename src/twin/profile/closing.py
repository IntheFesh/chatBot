"""What counts as a closing reply, and how often she lets it be the last word (R-ENG-007).

A *closing* message is a short answer that asks nothing and opens nothing: "好的", "嗯嗯", "晚安",
"哈哈哈", a bare emoji code, a sticker.  After one of those she often does not answer, and the
bot is allowed to stay silent in the same situations (``[不回]``).  Two places need the same
definition, so it lives in one function:

* the profile counts, over the real conversation, how often a closing message of the other side
  was *not* answered within the same conversation segment (``closing_no_reply_rate`` of her
  metrics, :mod:`twin.profile.metrics`);
* the reply post-processing accepts ``[不回]`` only when the user's last message is closing and
  the profile says she does this at all (:mod:`twin.engine.postprocess`).

The test is on the shape of the message, not its words: nothing that asks a question, at most
:data:`CLOSING_MAX_CHARS` characters once emoji codes, emoji and punctuation are taken away.
"""

from __future__ import annotations

import re

from twin.profile.textstats import EMOJI_CODE_RE, EMOJI_RE, is_question

CLOSING_MAX_CHARS = 6
QUESTION_WORDS = (
    "吗",
    "呢",
    "什么",
    "怎么",
    "为什么",
    "为啥",
    "咋",
    "哪",
    "谁",
    "几点",
    "多少",
    "干嘛",
    "干啥",
    "怎样",
    "如何",
    "是否",
    "是不是",
    "能不能",
    "要不要",
    "可不可以",
    "好不好",
    "行不行",
    "对不对",
)
_NOT_WORDS = re.compile(r"[\W_]+", re.UNICODE)
TEXT_KINDS = frozenset({"text", "quote"})


def is_closing_message(kind: str, text: str | None) -> bool:
    """Is this message (of the user) a short reply that asks nothing?"""
    if kind == "sticker":
        return True
    if kind not in TEXT_KINDS:
        return False
    body = (text or "").strip()
    if not body or is_question(body) or any(word in body for word in QUESTION_WORDS):
        return False
    core = _NOT_WORDS.sub("", EMOJI_RE.sub("", EMOJI_CODE_RE.sub("", body)))
    return len(core) <= CLOSING_MAX_CHARS
