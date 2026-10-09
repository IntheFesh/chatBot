"""A small byte-level BPE tokenizer built like Qwen's, for tests that need token counts.

It has the same structure as ``tokenizer.json`` of Qwen3: an NFC normalizer, the same
pre-tokenizer regular expression followed by the byte-level mapping, a BPE model trained on a few
sentences, and the three special tokens of ChatML (``<|im_start|>``, ``<|im_end|>``,
``<|endoftext|>``) as added tokens.  It is not the vocabulary of Qwen - the tests that need that
one run in an environment that has the real file (``TWIN_QWEN_TOKENIZER``) - but text is split at
the same places, so everything that depends on *where* the pieces of a prompt meet behaves alike.
"""

from __future__ import annotations

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from twin.training.tokenizer import QwenTokenizer

# the split pattern of the Qwen2 / Qwen3 tokenizer.json
QWEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SPECIAL = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
CORPUS = (
    "你好呀 在吗 今天好累 哈哈哈 晚安 早安 吃饭了吗 好的 嗯嗯 想你了 [表情包:开心] [拥抱] [亲亲]",
    "user assistant system the quick brown fox jumps over the lazy dog 12345",
    "【此刻】 当地时间 她现在的状态 【规划】 想表达 会用到 语气 气泡 【前文】 她：",
    "hello world, how are you? fine, thanks. see you tomorrow!",
)


def build_tiny_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=520,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS * 3, trainer)
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL])
    return tokenizer


def tiny_qwen_tokenizer() -> QwenTokenizer:
    """The tokenizer as the export sees it (a :class:`~twin.training.tokenizer.QwenTokenizer`)."""
    return QwenTokenizer(build_tiny_tokenizer(), "0" * 64)


def build_merging_tokenizer() -> Tokenizer:
    """A tokenizer that does *not* split words at line breaks, so merges cross piece borders.

    ChatML pieces are tokenised one by one by the trainer and as one string by the inference
    server; with a pre-tokenizer like this one a merge such as "newline + a" exists, and the
    prompt tokenises differently from the pieces.  The template check must notice that.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=400,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(["assistant\nabc ok\nab abc\nab"] * 200, trainer)
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL])
    return tokenizer
