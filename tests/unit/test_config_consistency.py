"""R-CFG-005: Settings defaults, config.example.yaml and the SPEC block are identical."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from twin.config.lists import load_regex_list, load_word_list
from twin.config.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "docs" / "SPEC.md"
EXAMPLE = ROOT / "config" / "config.example.yaml"


def spec_block() -> dict[str, Any]:
    text = SPEC.read_text(encoding="utf-8")
    match = re.search(r"R-CFG-004\*\*.*?```yaml\n(.*?)```", text, re.DOTALL)
    assert match, "R-CFG-004 YAML block not found in docs/SPEC.md"
    return yaml.safe_load(match.group(1))  # type: ignore[no-any-return]


def example_file() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict) and node:
        flat: dict[str, Any] = {}
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: node}


def normalise(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, list):
        return [normalise(item) for item in value]
    return value


def comparable(tree: dict[str, Any]) -> dict[str, Any]:
    return {key: normalise(value) for key, value in flatten(tree).items()}


def test_spec_block_example_file_and_settings_defaults_have_identical_keys_and_values() -> None:
    defaults = comparable(Settings().to_plain())
    spec = comparable(spec_block())
    example = comparable(example_file())

    assert set(spec) == set(defaults), f"keys differ between SPEC and Settings: {set(spec) ^ set(defaults)}"
    assert set(example) == set(defaults), f"keys differ between example and Settings: {set(example) ^ set(defaults)}"
    for key in sorted(defaults):
        assert spec[key] == defaults[key], f"SPEC vs Settings default differ at {key}"
        assert example[key] == defaults[key], f"config.example.yaml vs Settings default differ at {key}"


def test_example_file_is_loadable_and_contains_no_secrets() -> None:
    from twin.config.loader import load_settings

    settings = load_settings(EXAMPLE)
    assert settings == load_settings(None, {})
    lowered = EXAMPLE.read_text(encoding="utf-8").lower()
    for word in ("password:", "api_key", "secret:", "token:"):
        assert word not in lowered


def test_every_top_level_section_of_settings_is_documented_in_the_spec() -> None:
    assert set(Settings.model_fields) == set(spec_block())


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("engine", "ai_phrases_file"),
        ("engine", "commitment_patterns_file"),
        ("safety", "crisis_keywords_file"),
    ],
)
def test_configured_word_list_files_ship_with_the_repository(section: str, key: str) -> None:
    relative = getattr(getattr(Settings(), section), key)
    path = ROOT / relative
    assert path.is_file(), f"{relative} is configured but missing"
    assert load_word_list(path), f"{relative} is empty"


def test_regex_list_compiles() -> None:
    patterns = load_regex_list(ROOT / Settings().engine.commitment_patterns_file)
    assert len(patterns) >= 30
