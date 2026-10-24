"""The chat template of the style model: LLaMA-Factory's ``qwen3_nothink`` (R-TRN-011).

Training and inference must see exactly the same token sequence.  The training side is the
template of the pinned LLaMA-Factory release; this module holds that template's format strings
as constants, so the other side (``StylePromptBuilder``, round 09) renders prompts with the same
pieces and its character-level tests have an expected value that does not depend on the
implementation under test.

Source of the strings (checked on 2026-10-09 against the ``llamafactory==0.9.5`` wheel, file
``src/llamafactory/data/template.py``)::

    register_template(
        name="qwen3_nothink",
        format_user=StringFormatter(slots=[
            "<|im_start|>user\\n{{content}}<|im_end|>\\n<|im_start|>assistant\\n"]),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\\n"]),
        format_system=StringFormatter(slots=["<|im_start|>system\\n{{content}}<|im_end|>\\n"]),
        ...
        stop_words=["<|im_end|>"],
        replace_eos=True,
    )

That is plain ChatML: there is no ``<think>`` marker anywhere, no ``default_system`` (an empty
system message produces no system block at all) and no prefix such as a BOS token.  It is not what
``tokenizer.apply_chat_template`` produces for Qwen3 (the official template inserts an empty think
block before the last answer), so nothing here may be compared with that template.

What the pieces mean for a sample ``system + [u1, a1, ..., un] -> reply``:

``render_prompt(system, turns)``
    the text the model gets at inference time: the system block, every earlier turn in full,
    and the last user message followed by the assistant opener.  ``turns`` ends with a user turn.
``render_response(reply)``
    what LLaMA-Factory trains on after that prompt: the reply, ``<|im_end|>`` and a newline.
    At inference time the model generates ``reply`` and then ``<|im_end|>`` (the stop token); the
    trailing newline is part of the training label but is never generated.
``render_training_text(system, turns, reply)``
    ``render_prompt`` + ``render_response`` - the whole sequence of one sample.

The LLaMA-Factory encoder tokenises every filled slot as one string with
``tokenizer.encode(slot, add_special_tokens=False)``, which has three consequences that the
exporter must respect (the helpers below detect them): text that contains a control token such as
``<|im_end|>`` is tokenised as that token; a user message that contains the text ``{{idx}}`` is
rewritten by the formatter (``idx`` is a second keyword of the user formatter); and a ShareGPT
sample must alternate human and gpt turns, start with human and end with gpt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

LLAMAFACTORY_VERSION: Final = "0.9.5"
LF_TEMPLATE_NAME: Final = "qwen3_nothink"
TEMPLATE_VERSION: Final = f"{LF_TEMPLATE_NAME}@llamafactory-{LLAMAFACTORY_VERSION}"
"""Recorded in dataset manifests, training bundles and ``model_registry.template_version``."""

IM_START: Final = "<|im_start|>"
IM_END: Final = "<|im_end|>"
STOP_TOKEN: Final = IM_END
"""The one stop word of the template (``replace_eos=True`` makes it the EOS token as well)."""

SYSTEM_OPEN: Final = f"{IM_START}system\n"
USER_OPEN: Final = f"{IM_START}user\n"
ASSISTANT_OPEN: Final = f"{IM_START}assistant\n"
TURN_CLOSE: Final = f"{IM_END}\n"
USER_CLOSE_AND_ASSISTANT_OPEN: Final = f"{IM_END}\n{ASSISTANT_OPEN}"

FORMAT_SYSTEM: Final = f"{SYSTEM_OPEN}{{content}}{TURN_CLOSE}"
FORMAT_USER: Final = f"{USER_OPEN}{{content}}{USER_CLOSE_AND_ASSISTANT_OPEN}"
FORMAT_ASSISTANT: Final = f"{{content}}{TURN_CLOSE}"

CONTROL_TOKENS: Final = (IM_START, IM_END, "<|endoftext|>")
LF_SLOTS: Final = ("{{content}}", "{{idx}}")

# ShareGPT role tags of LLaMA-Factory's ``sharegpt`` format (dataset_info.json ``tags``)
ROLE_SYSTEM: Final = "system"
ROLE_USER: Final = "human"
ROLE_ASSISTANT: Final = "gpt"

Role = Literal["user", "assistant"]


class TemplateError(ValueError):
    """A sample cannot be rendered the way LLaMA-Factory would encode it."""


@dataclass(frozen=True)
class Turn:
    """One message of a conversation: ``user`` (the person) or ``assistant`` (her)."""

    role: Role
    content: str


def contains_control_token(text: str) -> bool:
    """True when ``text`` holds ChatML control text that the tokenizer would turn into a token."""
    return any(token in text for token in CONTROL_TOKENS)


def contains_lf_slot(text: str) -> bool:
    """True when ``text`` holds a slot name that LLaMA-Factory's formatter would substitute."""
    return any(slot in text for slot in LF_SLOTS)


def check_content(text: str) -> None:
    """Reject text that would not survive the template unchanged."""
    if contains_control_token(text):
        raise TemplateError("text contains a ChatML control token")
    if contains_lf_slot(text):
        raise TemplateError("text contains a LLaMA-Factory slot name ({{content}} or {{idx}})")


def check_alternation(turns: Sequence[Turn]) -> None:
    """The context of a sample: user first, strictly alternating, ending with a user turn."""
    if not turns:
        raise TemplateError("a sample needs at least one user turn")
    for index, turn in enumerate(turns):
        expected: Role = "user" if index % 2 == 0 else "assistant"
        if turn.role != expected:
            raise TemplateError(
                f"turn {index} is {turn.role!r} but the template needs {expected!r} "
                "(user first, alternating)"
            )
    if turns[-1].role != "user":
        raise TemplateError("the context must end with a user turn")


def render_system(content: str) -> str:
    """The system block; an empty system message produces nothing (no default system)."""
    if not content:
        return ""
    check_content(content)
    return FORMAT_SYSTEM.replace("{content}", content, 1)


def render_user(content: str) -> str:
    """A user message followed by the assistant opener."""
    check_content(content)
    return FORMAT_USER.replace("{content}", content, 1)


def render_assistant(content: str) -> str:
    """An earlier assistant message of the context."""
    check_content(content)
    return FORMAT_ASSISTANT.replace("{content}", content, 1)


def render_prompt(system: str, turns: Sequence[Turn]) -> str:
    """The inference prompt of a sample: everything the model sees before it answers."""
    check_alternation(turns)
    parts = [render_system(system)]
    for turn in turns:
        rendered = render_user if turn.role == "user" else render_assistant
        parts.append(rendered(turn.content))
    return "".join(parts)


def render_response(reply: str) -> str:
    """The training label that follows the prompt: the reply, ``<|im_end|>`` and a newline."""
    if not reply:
        raise TemplateError("the reply of a sample cannot be empty")
    check_content(reply)
    return FORMAT_ASSISTANT.replace("{content}", reply, 1)


def render_training_text(system: str, turns: Sequence[Turn], reply: str) -> str:
    """The whole sequence LLaMA-Factory encodes for one sample (prompt then label)."""
    return render_prompt(system, turns) + render_response(reply)


def sharegpt_conversations(turns: Sequence[Turn], reply: str) -> list[dict[str, str]]:
    """The ``conversations`` list of a ShareGPT sample: human first, alternating, gpt last."""
    check_alternation(turns)
    conversations = [
        {"from": ROLE_USER if turn.role == "user" else ROLE_ASSISTANT, "value": turn.content}
        for turn in turns
    ]
    conversations.append({"from": ROLE_ASSISTANT, "value": reply})
    return conversations
