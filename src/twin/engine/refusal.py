"""Recognising a refused generation (R-SAFE-005).

DeepSeek refuses in three ways: a ``content_filter`` finish, an HTTP 400 "Content Exists Risk"
before any text exists, or - rarely - a reply that is the refusal itself ("抱歉，我无法继续这个
话题", "I can't assist with that").  The first two are the backend's to see; this module reads the
third.  A refusal is never shown to the user and never retried in the same breath: the engine
treats the round like any other failure and goes to the fallback (R-ENG-010), so the user sees
a late, short, natural answer and not a policy notice.

The patterns are deliberately about the *refusal* wording (policy, "cannot assist"), not about the
everyday "我不能说话了，要睡了": a girlfriend declining to continue a chat is not a refusal.
"""

from __future__ import annotations

import re

PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:抱歉|对不起|很遗憾)[，,]?\s*我?(?:无法|不能|没办法)(?:继续|满足|回答|提供|协助|帮助|讨论)"
        r"(?:这个|该|此|这类)(?:话题|请求|要求|问题|内容|对话)",
        r"我(?:无法|不能|没办法)(?:协助|帮助|满足)(?:你|您)?(?:完成|做|处理)?(?:这个|该|此|这类)",
        r"(?:违反|不符合|超出)(?:了)?(?:相关)?(?:的)?(?:使用)?(?:政策|规定|准则|规范)",
        r"i (?:can(?:'|’)?t|cannot|can not|am unable to|am not able to) "
        r"(?:help|assist|comply|continue|provide)",
        r"as an ai",
        r"content (?:policy|exists risk)",
    )
)
FILTER_FINISHES = frozenset({"content_filter"})


def looks_like_refusal(text: str) -> bool:
    """Is the whole output a refusal rather than a reply?"""
    body = text.strip()
    return bool(body) and any(pattern.search(body) for pattern in PATTERNS)
