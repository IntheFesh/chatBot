"""Changing the user's ``config.yaml`` after they confirmed (probe suggestions, R-CH-009)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from twin.config.loader import ConfigError, load_settings
from twin.config.writer import set_section_values

VALUES = {"proactive_window_safe_h": 20.7, "outbound_quota_safe": 7}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_a_missing_file_is_created_with_only_the_section(tmp_path: Path) -> None:
    path = tmp_path / "config" / "config.yaml"
    set_section_values(path, "channel", VALUES)
    assert yaml.safe_load(read(path)) == {"channel": VALUES}
    settings = load_settings(path)
    assert settings.channel.proactive_window_safe_h == 20.7
    assert settings.channel.outbound_quota_safe == 7


def test_a_flow_style_line_is_replaced_and_everything_else_is_kept_byte_for_byte(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        '# my notes\nconsent: { confirmed_at: "2026-10-08" }   # agreed\n'
        'channel: { kind: "ilink", proactive_window_safe_h: 22, outbound_quota_safe: 8,'
        ' proactive_reserve: 2 }\n\n# the next section\nretrieval: { device: "cpu" }\n',
        encoding="utf-8",
    )
    set_section_values(path, "channel", VALUES)
    text = read(path)
    assert text.startswith('# my notes\nconsent: { confirmed_at: "2026-10-08" }   # agreed\n')
    assert text.endswith('\n# the next section\nretrieval: { device: "cpu" }\n')
    data = yaml.safe_load(text)
    assert data["channel"] == {
        "kind": "ilink",
        "proactive_window_safe_h": 20.7,
        "outbound_quota_safe": 7,
        "proactive_reserve": 2,  # keys that were not touched stay
    }
    assert data["retrieval"] == {"device": "cpu"}


def test_a_block_style_section_is_replaced_in_place(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "channel:\n  kind: ilink\n  # keep it small\n  outbound_quota_safe: 8\n"
        "\nops:\n  backup_hour_local: 5\n",
        encoding="utf-8",
    )
    set_section_values(path, "channel", {"outbound_quota_safe": 6})
    data = yaml.safe_load(read(path))
    assert data["channel"] == {"kind": "ilink", "outbound_quota_safe": 6}
    assert data["ops"] == {"backup_hour_local": 5}
    assert read(path).count("channel:") == 1


def test_a_section_that_is_missing_is_appended(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("# only a comment\nretrieval: { device: cpu }", encoding="utf-8")
    set_section_values(path, "channel", VALUES)
    text = read(path)
    assert text.startswith("# only a comment\nretrieval: { device: cpu }\n\nchannel:")
    assert yaml.safe_load(text)["channel"] == VALUES


def test_an_invalid_result_leaves_the_original_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    original = "channel: { kind: ilink }\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigError, match="outbound_quota_safe"):
        set_section_values(path, "channel", {"outbound_quota_safe": -3})
    assert read(path) == original
    assert not (tmp_path / "config.yaml.tmp").exists()


def test_a_file_that_is_not_a_mapping_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a mapping"):
        set_section_values(path, "channel", VALUES)
    assert read(path) == "- just\n- a list\n"


def test_a_file_with_windows_line_endings_keeps_them(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes(b"# mine\r\nchannel: { kind: ilink }\r\nretrieval: { device: cpu }\r\n")
    set_section_values(path, "channel", VALUES)
    raw = path.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") >= 3  # every line break is still CRLF
    assert raw.startswith(b"# mine\r\n") and raw.endswith(b"retrieval: { device: cpu }\r\n")
    assert yaml.safe_load(raw.decode("utf-8"))["channel"]["outbound_quota_safe"] == 7


def test_lines_are_written_with_unix_endings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    set_section_values(path, "channel", VALUES)
    assert b"\r" not in path.read_bytes()


def test_a_file_with_only_comments_gets_the_section_after_them(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("# nothing here yet\n", encoding="utf-8")
    set_section_values(path, "channel", VALUES)
    text = read(path)
    assert text.startswith("# nothing here yet\n") and yaml.safe_load(text)["channel"] == VALUES
