"""The tokenizer comparison that must pass before a model is used (R-TRN-011.4, R-LLM-011).

The model was trained on token ids that LLaMA-Factory made from the ChatML text with the Qwen3
tokenizer; at inference the server turns the very same text into ids itself.  If the two disagree
the model is fed something it never saw, and the replies degrade without any error.  Two things go
wrong in practice, and both are looked for here:

* a **BOS token in front** - llama.cpp inserts one into a string prompt when the GGUF says
  ``tokenizer.ggml.add_bos_token`` (``/completion`` tokenizes with ``add_special=true`` and
  ``parse_special=true``, verified in ``tools/server/server-context.cpp`` of b11177 and b11539).
  Qwen3 files do not ask for one, but a converter or a quantiser may change that;
* a **control token split into pieces** - ``<|im_start|>`` and ``<|im_end|>`` must each be one id;
  a server that does not parse special tokens turns them into several ordinary ones.

:func:`check_tokenization` sends a fixed set of rendered prompts (:func:`sample_prompts`: plain
Chinese chat, sticker markers, emoji codes, a quote line, English and digits, unusual spacing, a
long conversation) to the server's ``/tokenize`` (``StyleModelClient.tokenize``, which asks the way
the completion endpoint will) and compares the ids with the pinned tokenizer, id by id.  The prompts
are made of fixed synthetic text and the pieces of :mod:`twin.training.lf_template`: nothing of her
conversation is sent anywhere.

The outcome is a :class:`TokenizationReport`.  A model whose report is not ``ok`` is refused: the
caller raises the alert ``style_tokenize_mismatch`` and shows :meth:`TokenizationReport.lines`,
which name the first difference of each prompt (the token position, the ids on both sides, a few
tokens of context and what kind of difference it is).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from twin.llm.errors import StyleModelError
from twin.llm.style_client import StyleModelClient
from twin.training import lf_template
from twin.training.lf_template import Turn

if TYPE_CHECKING:
    from twin.training.registry import ModelView

CONTEXT_TOKENS: Final = 4
MAX_REPORTED: Final = 5

DifferenceKind = Literal["extra_prefix", "missing_prefix", "split_special", "length", "different"]


class TokenizerLike(Protocol):
    """What the comparison needs from the local tokenizer (``QwenTokenizer``)."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_id(self, token: str) -> int | None: ...


class TokenizeCheckError(RuntimeError):
    """The server could not tokenize at all: the comparison could not be made."""


@dataclass(frozen=True)
class SamplePrompt:
    """A rendered prompt to compare."""

    name: str
    text: str


@dataclass(frozen=True)
class Difference:
    """The first place where the server's ids differ from the tokenizer's, in one prompt."""

    sample: str
    kind: DifferenceKind
    position: int
    local_id: int | None
    server_id: int | None
    local_count: int
    server_count: int
    local_text: str  # a few tokens around the position, decoded
    server_text: str

    def line(self) -> str:
        what = {
            "extra_prefix": "the server put extra token(s) in front (a BOS?)",
            "missing_prefix": "the server left token(s) out at the start",
            "split_special": "a control token was split into ordinary tokens",
            "length": "one side stops earlier",
            "different": "different token",
        }[self.kind]
        return (
            f"{self.sample}: {what}; first difference at token {self.position} "
            f"(tokenizer {self.local_id!r} {self.local_text!r}, "
            f"server {self.server_id!r} {self.server_text!r}); "
            f"{self.local_count} tokens against {self.server_count}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "kind": self.kind,
            "position": self.position,
            "local_id": self.local_id,
            "server_id": self.server_id,
            "local_count": self.local_count,
            "server_count": self.server_count,
        }


@dataclass(frozen=True)
class TokenizationReport:
    """The result of comparing a server with the tokenizer."""

    samples: int
    tokens: int
    differences: tuple[Difference, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.differences

    def lines(self) -> list[str]:
        if self.ok:
            return [f"tokenization matches: {self.samples} prompts, {self.tokens} tokens"]
        shown = [d.line() for d in self.differences[:MAX_REPORTED]]
        rest = len(self.differences) - len(shown)
        head = f"tokenization differs in {len(self.differences)} of {self.samples} prompts:"
        return [head, *shown, *([f"... and {rest} more"] if rest > 0 else [])]


# ------------------------------------------------------------------- the prompts


def _turn(role: str, content: str) -> Turn:
    return Turn("user" if role == "user" else "assistant", content)


def sample_prompts() -> list[SamplePrompt]:
    """The fixed prompts the comparison sends (rendered with the training template)."""
    persona = "她说话很短，常用语气词。\n不用句号。\n喜欢发表情包。"
    now = "【此刻】\n当地时间：周四 晚上 22:35\n她现在：在家，刚洗完澡"
    plan = "【规划】\n想表达：关心他今天累不累\n语气：温柔\n气泡：两条"
    prelude = "【前文】\n她：我到家啦\n她：[捂脸] 今天好忙"
    long_turns: list[Turn] = []
    for number in range(1, 17):
        long_turns.append(_turn("user", f"第{number}条：今天天气不错，去吃饭了吗？"))
        long_turns.append(_turn("assistant", f"吃了呀\n你呢 [表情包:开心]\n{number}点半才下班"))
    long_turns.append(_turn("user", "晚上想吃什么"))
    prompts = [
        ("plain", persona, [_turn("user", "在吗")]),
        (
            "chat",
            f"{persona}\n\n{now}",
            [_turn("user", "今天好累"), _turn("assistant", "抱抱"), _turn("user", "你呢")],
        ),
        (
            "stickers_and_codes",
            f"{persona}\n\n{now}",
            [
                _turn("user", "[表情包:大笑]"),
                _turn("assistant", "哈哈哈 [旺柴]\n[表情包:捂脸]"),
                _turn("user", "[微笑][微笑] 晚安"),
            ],
        ),
        (
            "quote_and_plan",
            f"{persona}\n\n{now}\n\n{plan}\n\n{prelude}",
            [
                _turn("user", "[引用:明天几点？]\n九点"),
                _turn("assistant", "好"),
                _turn("user", "到时候叫你"),
            ],
        ),
        (
            "mixed_scripts",
            persona,
            [
                _turn("user", "OK, see you at 7:30pm! https://example.com/a?b=1&c=2 😀"),
                _turn("assistant", "好的 ~ 繁體字也行 \t tab 和  两个空格"),
                _turn("user", "price: ¥12.50 / $3, 100%\n\n\n三个换行"),
            ],
        ),
        (
            "whitespace",
            f"{persona} ",
            [
                _turn("user", "  前面有空格"),
                _turn("assistant", "后面有空格  "),
                _turn("user", "好"),
            ],
        ),
        ("long", f"{persona}\n\n{now}", long_turns),
    ]
    return [
        SamplePrompt(name, lf_template.render_prompt(system, turns))
        for name, system, turns in prompts
    ]


# ------------------------------------------------------------------ comparison


def _context(tokenizer: TokenizerLike, ids: list[int], position: int) -> str:
    start = max(0, position - 1)
    return tokenizer.decode(ids[start : position + CONTEXT_TOKENS])


def _kind(
    tokenizer: TokenizerLike, local: list[int], server: list[int], position: int
) -> DifferenceKind:
    extra = len(server) - len(local)
    if extra > 0 and server[extra:] == local:
        return "extra_prefix"
    if extra < 0 and local[-extra:] == server:
        return "missing_prefix"
    control = {
        found
        for token in lf_template.CONTROL_TOKENS
        if (found := tokenizer.token_id(token)) is not None
    }
    if position < len(local) and local[position] in control:
        return "split_special"
    if position >= len(local) or position >= len(server):
        return "length"
    return "different"


def compare_ids(
    tokenizer: TokenizerLike, name: str, local: list[int], server: list[int]
) -> Difference | None:
    """The first difference between two id lists (``None``: identical)."""
    if local == server:
        return None
    position = next(
        (i for i, (a, b) in enumerate(zip(local, server, strict=False)) if a != b),
        min(len(local), len(server)),
    )
    kind = _kind(tokenizer, local, server, position)
    if kind == "extra_prefix":
        position = 0
    return Difference(
        sample=name,
        kind=kind,
        position=position,
        local_id=local[position] if position < len(local) else None,
        server_id=server[position] if position < len(server) else None,
        local_count=len(local),
        server_count=len(server),
        local_text=_context(tokenizer, local, position),
        server_text=_context(tokenizer, server, position),
    )


async def check_tokenization(
    client: StyleModelClient,
    tokenizer: TokenizerLike,
    prompts: list[SamplePrompt] | None = None,
) -> TokenizationReport:
    """Compare the server's tokens with the pinned tokenizer's, prompt by prompt."""
    chosen = prompts if prompts is not None else sample_prompts()
    differences: list[Difference] = []
    total = 0
    for prompt in chosen:
        local = tokenizer.encode(prompt.text)
        try:
            server = await client.tokenize(prompt.text)
        except StyleModelError as exc:
            raise TokenizeCheckError(
                f"the server could not tokenize {prompt.name!r} ({exc.kind or 'error'}): {exc}"
            ) from exc
        total += len(local)
        found = compare_ids(tokenizer, prompt.name, local, server)
        if found is not None:
            differences.append(found)
    return TokenizationReport(len(chosen), total, tuple(differences))


def tokenize_record(report: TokenizationReport, model: ModelView, *, at: str) -> dict[str, Any]:
    """What is kept of a comparison in the registry: the verdict and the first differences."""
    return {
        "ok": report.ok,
        "at": at,
        "model_sha256": model.sha256,
        "samples": report.samples,
        "tokens": report.tokens,
        "differences": [d.to_json() for d in report.differences[:5]],
    }
