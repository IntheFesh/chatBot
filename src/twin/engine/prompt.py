"""The prompt of the DeepSeek backend, laid out for the context cache (R-ENG-005, R-LLM-010).

Order of the messages:

1. **system** - the fixed rules and the full persona card (template ``reply_rules``, versioned in
   ``twin/profile/templates``).  It is the same text for every request as long as the card and
   the vocabularies do not change;
2. **the recent conversation** - the window of :mod:`twin.memory.recent` as ``user`` / ``assistant``
   messages.  A user message holds only what the user wrote (R-ENG-005: history keeps the
   original); consecutive turns of one side are one message;
3. **the last user message** - the only place that changes from request to request: the context of
   the moment (local time and weekday, her state and what she is doing, memory with its
   follow-ups, examples of how she really answered in similar situations, a hint when the user's
   message ends a conversation, what was already said in this round, what a previous attempt got
   wrong) followed by the user's messages of this round.

Because everything that varies sits in the last message, request *n+1* begins with request *n*'s
system message and history (:meth:`PromptLayout.extends`) until the window moves forward in its
batch step - which is what lets the provider's prefix cache keep hitting.  The layout is kept
apart as ``stable_prefix`` (system and history) and ``variable_tail`` (the last message) so the
cache monitor can see it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from twin.engine.types import ReplyContext, ReplyMaterial
from twin.llm.layout import PromptLayout
from twin.llm.types import ChatMessage, Role
from twin.profile.prompt_templates import PromptText, TemplateStore

if TYPE_CHECKING:
    from twin.services import Services

TEMPLATE = "reply_rules"
EMOJI_CODES_SHOWN = 30
NO_PERSONA = "（还没有人设卡：就按一个普通年轻女生发微信的样子来，不要说得太书面。）"
NO_EMOJI_CODES = "（她几乎不用微信表情代码，不要用）"
HISTORY_OPENER = "（以下是你们之前聊过的内容）"
CONTEXT_OPEN = "（以下是此刻的情况，只用来决定怎么回，不要复述给对方）"
CONTEXT_CLOSE = "（以上是此刻的情况）"
DAY_TYPES = {"workday": "工作日", "weekend": "周末", "holiday": "节假日"}
STATE_LINES = {
    "free": "空闲，可以正常聊天。",
    "busy": "在忙别的事，回复会简短一些、慢一些。",
    "sleep_edge": "快睡着了或者刚醒，迷迷糊糊的，回得很简短。",
    "deep_sleep": "睡着了；你是刚醒过来看到消息的。",
}
VIOLATION_NOTES = {
    "ai_self_reference": "不要提到自己是 AI、人工智能或语言模型，也不要用客服的口吻，像她一样说话",
    "token_leak": "回复里出现了像 [手机号] 这样的方括号标记，不要输出这种标记",
    "event_text_only": "不要输出 [图片]、[语音]、[通话] 这类消息占位，直接用文字回复",
    "commitment": "不要答应打电话、视频、发语音、发照片、见面、转账、寄东西这类你做不到的事，"
    "用她的语气自然地带过",
    "think_tag": "不要输出思考过程，只输出要发给对方的气泡",
    "not_chinese": "用中文口语回复",
    "empty": "必须回复一些内容",
    "no_reply_not_allowed": "对方这句需要回应，不要只回 [不回]",
}


def day_part(hour: int) -> str:
    """How the hour is called in speech."""
    if hour < 6:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 12:
        return "上午"
    if hour < 14:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


@dataclass(frozen=True)
class BuiltPrompt:
    """A request ready to send, with the facts the audit trail keeps."""

    layout: PromptLayout
    template: str
    history_turns: int
    persona_ref: str | None
    notes: tuple[str, ...] = field(default=())

    @property
    def messages(self) -> list[ChatMessage]:
        return self.layout.messages()

    def meta(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "persona": self.persona_ref,
            "history_turns": self.history_turns,
            "prefix_hash": self.layout.prefix_hash[:16],
        }


def render_context(context: ReplyContext, material: ReplyMaterial, notes: Sequence[str]) -> str:
    """The context block of the last user message (see the module description)."""
    local = material.local
    day = DAY_TYPES.get(local.day_type, local.day_type)
    moment = (
        f"{local.local.year}年{local.local.month}月{local.local.day}日 {local.weekday_name}"
        f"（{day}） {day_part(local.local.hour)} {local.local:%H:%M}"
    )
    now = [f"当地时间：{moment}"]
    if material.state in STATE_LINES:
        now.append(f"她现在的状态：{STATE_LINES[material.state]}")
    if material.lifeline:
        now.append(f"她这会儿大概在：{material.lifeline}")
    if context.woke_up:
        now.append("你刚醒，刚看到对方发来的这些消息：语气像刚睡醒，不要像一直在线。")
    if context.already_said:
        said = "\n".join(f"- {line}" for line in context.already_said)
        now.append(f"这一轮里你已经对对方说过这些话，接着往下说，不要重复：\n{said}")
    sections = ["【此刻】\n" + "\n".join(now)]
    if material.memory_text.strip():
        sections.append(material.memory_text.strip())
    if material.examples_text.strip():
        sections.append(
            "【她过去在类似情况下的真实回复（仅供模仿语气，不要照抄内容）】\n"
            + material.examples_text.strip()
        )
    if material.closing is not None and material.closing.no_reply_rate > 0:
        sections.append(
            "【对方这句的情况】\n对方这句是不需要回答的结束性短话。她遇到这种情况大约有 "
            f"{material.closing.no_reply_rate:.0%} 的时候不再回复；决定不回，就只输出一行 [不回]，"
            "否则照常回复。"
        )
    if notes:
        sections.append(
            "【上一次的回复有这些问题，这次要避免】\n" + "\n".join(f"- {note}" for note in notes)
        )
    return CONTEXT_OPEN + "\n" + "\n\n".join(sections) + "\n" + CONTEXT_CLOSE


def history_messages(context: ReplyContext) -> list[ChatMessage]:
    """The recent turns as chat messages: one per run of one side, the user first."""
    messages: list[ChatMessage] = []
    for turn in context.history:
        role: Role = "user" if turn.role == "user" else "assistant"
        if messages and messages[-1]["role"] == role:
            messages[-1] = {"role": role, "content": f"{messages[-1]['content']}\n{turn.text}"}
        else:
            messages.append({"role": role, "content": turn.text})
    if messages and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": HISTORY_OPENER})
    return messages


class PromptBuilder:
    """Builds the request of the DeepSeek backend from a round and its material."""

    def __init__(self, template: PromptText, *, sticker_tags: Sequence[str]) -> None:
        self._template = template
        self._tags = tuple(sticker_tags)

    @classmethod
    def from_services(cls, services: Services, sticker_tags: Sequence[str]) -> PromptBuilder:
        return cls(
            TemplateStore(services.db, services.clock).active(TEMPLATE), sticker_tags=sticker_tags
        )

    @property
    def template(self) -> PromptText:
        return self._template

    def build(
        self,
        context: ReplyContext,
        material: ReplyMaterial,
        *,
        emoji_codes: Sequence[str] = (),
        notes: Sequence[str] = (),
    ) -> BuiltPrompt:
        history = history_messages(context)
        carried = ""
        if history and history[-1]["role"] == "user":
            carried = str(history.pop()["content"])
        rendered = self._template.render(
            persona=material.persona.strip() or NO_PERSONA,
            sticker_tags="、".join(self._tags) or "（没有可用的表情包标签，不要发表情包）",
            emoji_codes=" ".join(emoji_codes[:EMOJI_CODES_SHOWN]) or NO_EMOJI_CODES,
            context=render_context(context, material, notes),
            message=context.user_text,
        )
        system, last = rendered
        if carried:
            last = {"role": "user", "content": f"{carried}\n\n{last['content']}"}
        layout = PromptLayout.of([system, *history], [last])
        return BuiltPrompt(
            layout, self._template.ref, len(context.history), material.persona_ref, tuple(notes)
        )
