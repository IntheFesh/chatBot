"""A tiny random Qwen3 base model with a Qwen-like tokenizer, saved like a downloaded one.

For the tests of ``training/autodl/tools/pair_logprobs.py`` (R18-E6, D-865): the tool loads a base
from a directory with ``AutoModelForCausalLM`` / ``AutoTokenizer`` and applies PEFT adapters, as it
does on the instance.  Nothing is downloaded: the weights are random (a fixed seed) and the
tokenizer is :func:`tests.support.tiny_tokenizer.build_tiny_tokenizer` (the ChatML special tokens,
Qwen's split pattern).  ``hidden`` 6 with one attention head of 6 makes ``q_proj`` and ``v_proj``
6 x 6, the shape of the adapters the fake ``llamafactory-cli`` writes
(:func:`tests.support.autodl_scripts.write_tiny_adapter`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.support.tiny_tokenizer import build_tiny_tokenizer

VOCAB = 600  # the tiny tokenizer has 520 trained tokens and three special ones


def tiny_config(*, hidden: int = 32, heads: int = 4) -> Any:
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
    )


def tiny_base(*, hidden: int = 32, heads: int = 4, seed: int = 1234) -> Any:
    """The same random base for the same arguments (the 'downloaded' weights of a test)."""
    import torch
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config(hidden=hidden, heads=heads))
    model.eval()
    return model


def tiny_tokenizer() -> Any:
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=build_tiny_tokenizer(),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )


def write_tiny_base(directory: Path, *, hidden: int = 32, heads: int = 4) -> Path:
    """Save the base and its tokenizer where a model directory would be; returns it."""
    directory.mkdir(parents=True, exist_ok=True)
    tiny_base(hidden=hidden, heads=heads).save_pretrained(str(directory))
    tiny_tokenizer().save_pretrained(str(directory))
    # the class name of a fast tokenizer that every transformers release knows (4.57 on the
    # instance does not know the name a newer release writes)
    config_path = directory / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return directory
