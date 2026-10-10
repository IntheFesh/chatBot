"""Configuration loading, priority, strictness, masking and consent (R-CFG-001, R-SCOPE-003)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from twin.config.loader import (
    ConfigError,
    ConsentError,
    default_config_path,
    ensure_consent,
    load_settings,
    parse_overrides,
    project_root,
    resolve_paths,
)
from twin.config.mask import MASK, mask_value, masked_settings
from twin.config.settings import ConfigFileError, Settings


def write_yaml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_any_source() -> None:
    settings = load_settings()
    assert settings.budget.daily_usd == 1.0
    assert settings.time.bot_timezone == "America/Chicago"
    assert settings.backend.active == "deepseek"
    assert settings.consent.confirmed_at == "2026-10-08"
    assert settings.pricing_usd_per_mtok["deepseek-flash"].cache_miss == 0.30


def test_priority_cli_over_env_over_yaml_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_yaml(tmp_path / "config.yaml", "budget: { daily_usd: 2.0, monthly_usd: 20.0 }\n")
    assert load_settings(config).budget.daily_usd == 2.0  # yaml beats default
    assert load_settings(config).budget.alert_ratio == 0.8  # untouched keys keep defaults

    monkeypatch.setenv("TWIN_BUDGET__DAILY_USD", "3.0")
    layered = load_settings(config)
    assert layered.budget.daily_usd == 3.0  # env beats yaml
    assert layered.budget.monthly_usd == 20.0  # yaml still supplies the sibling key

    final = load_settings(config, {"budget": {"daily_usd": 4.0}})
    assert final.budget.daily_usd == 4.0  # command line beats env
    assert final.budget.monthly_usd == 20.0


def test_nested_env_variables_use_double_underscore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWIN_TIME__BOT_TIMEZONE", "Asia/Shanghai")
    monkeypatch.setenv("TWIN_DEEPSEEK__TIMEOUT_S__THINKING", "99")
    settings = load_settings()
    assert settings.time.bot_timezone == "Asia/Shanghai"
    assert settings.deepseek.timeout_s.thinking == 99


def test_non_config_twin_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWIN_LIVE", "1")
    monkeypatch.setenv("TWIN_KEYRING_PASSPHRASE", "irrelevant")
    assert load_settings().budget.daily_usd == 1.0


@pytest.mark.parametrize(
    "yaml_text",
    [
        "surprise: 1\n",
        "budget: { daily_usd: 1.0, surprise: 2 }\n",
        "deepseek: { timeout_s: { thinking: 1, nope: 2 } }\n",
        "ops: { smtp: { host: x, extra: 1 } }\n",
    ],
)
def test_unknown_yaml_keys_are_errors_at_every_level(tmp_path: Path, yaml_text: str) -> None:
    config = write_yaml(tmp_path / "config.yaml", yaml_text)
    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
        load_settings(config)


def test_unknown_keys_via_env_and_command_line_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_settings(None, {"budget": {"daily_dollars": 1}})
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_settings(None, {"nonsense": 1})
    monkeypatch.setenv("TWIN_BUDGET__DAILY_DOLLARS", "1")
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_settings()


def test_invalid_values_report_the_key_path(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "c.yaml", "backend: { active: telepathy }\nbudget: { daily_usd: -1 }\n"
    )
    with pytest.raises(ConfigError) as info:
        load_settings(config)
    message = str(info.value)
    assert "backend.active" in message and "budget.daily_usd" in message


def test_explicit_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_default_config_path_is_ignored_when_absent(isolated_environment: Path) -> None:
    assert default_config_path() == isolated_environment.resolve() / "config" / "config.yaml"
    assert load_settings().target.username is None  # no file: defaults


def test_malformed_yaml_and_non_mapping_files(tmp_path: Path) -> None:
    broken = write_yaml(tmp_path / "broken.yaml", "budget: [unclosed\n")
    with pytest.raises(ConfigFileError, match="cannot read configuration file"):
        load_settings(broken)
    listy = write_yaml(tmp_path / "list.yaml", "- a\n- b\n")
    with pytest.raises(ConfigFileError, match="YAML mapping"):
        load_settings(listy)
    assert load_settings(write_yaml(tmp_path / "empty.yaml", "")).budget.daily_usd == 1.0


def test_source_timezone_ranges_use_the_from_keyword(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "c.yaml",
        "time:\n  source_timezone_ranges:\n"
        "    - { from: '2025-01-01', to: '2025-08-31', tz: Asia/Shanghai }\n",
    )
    settings = load_settings(config)
    only = settings.time.source_timezone_ranges[0]
    assert only.from_ == date(2025, 1, 1) and only.to == date(2025, 8, 31)
    assert settings.to_plain()["time"]["source_timezone_ranges"][0]["from"] == "2025-01-01"


def test_parse_overrides_builds_nested_mappings_with_yaml_scalars() -> None:
    parsed = parse_overrides(
        [
            "budget.daily_usd=2",
            "target.username=null",
            "time.bot_timezone=Asia/Shanghai",
            "proactive.daily_max = 3",
            "engine.max_bubbles=[1, 2]",
        ]
    )
    assert parsed == {
        "budget": {"daily_usd": 2},
        "target": {"username": None},
        "time": {"bot_timezone": "Asia/Shanghai"},
        "proactive": {"daily_max": 3},
        "engine": {"max_bubbles": [1, 2]},
    }


@pytest.mark.parametrize("bad", ["novalue", "=x", "a.b=[unclosed"])
def test_parse_overrides_rejects_malformed_items(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_overrides([bad])


def test_parse_overrides_detects_conflicts() -> None:
    with pytest.raises(ConfigError, match="conflicts"):
        parse_overrides(["a=1", "a.b=2"])


# ------------------------------------------------------------------- masking


def wxid_like() -> str:
    return "wxid_" + "synthetic9"  # built at run time so the privacy scan stays clean


def test_effective_configuration_is_masked(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "c.yaml",
        f"target: {{ username: {wxid_like()} }}\n"
        "ops: { smtp: { host: smtp.example.org, user: someone@example.org, to: me@example.org } }\n"
        "autodl: { host: connect.example.org }\n"
        "safety: { emergency_contact: { enabled: true, email: friend@example.org } }\n",
    )
    masked = masked_settings(load_settings(config))
    assert masked["target"]["username"] != wxid_like()
    assert MASK in masked["target"]["username"]
    assert masked["ops"]["smtp"]["user"].startswith("so") and MASK in masked["ops"]["smtp"]["user"]
    assert MASK in masked["ops"]["smtp"]["to"]
    assert MASK in masked["autodl"]["host"]
    assert MASK in masked["safety"]["emergency_contact"]["email"]
    assert masked["ops"]["smtp"]["host"] == "smtp.example.org"  # not personal
    assert masked["budget"]["daily_usd"] == 1.0


def test_none_values_stay_none_and_secret_looking_keys_are_hidden() -> None:
    from twin.config.mask import _walk

    assert masked_settings(load_settings())["target"]["username"] is None
    walked = _walk({"smtp_password": "hunter2", "api_key": "k", "token": None, "n": 1}, "")
    assert walked == {"smtp_password": MASK, "api_key": MASK, "token": None, "n": 1}
    assert mask_value("abc") == MASK
    assert mask_value("abcdefghij") == "ab***ij"


# ------------------------------------------------------------------- consent


def test_valid_consent_returns_the_date() -> None:
    assert ensure_consent(load_settings()) == date(2026, 10, 8)


def test_unquoted_yaml_date_is_accepted(tmp_path: Path) -> None:
    config = write_yaml(tmp_path / "c.yaml", "consent: { confirmed_at: 2026-10-08 }\n")
    assert ensure_consent(load_settings(config)) == date(2026, 10, 8)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_consent_refuses_to_start(value: str | None) -> None:
    settings = load_settings(None, {"consent": {"confirmed_at": value}})
    with pytest.raises(ConsentError, match="missing"):
        ensure_consent(settings)


@pytest.mark.parametrize("value", ["yesterday", "2026-13-40", "08/10/2026", "2026-10"])
def test_invalid_consent_date_refuses_to_start(value: str) -> None:
    settings = load_settings(None, {"consent": {"confirmed_at": value}})
    with pytest.raises(ConsentError, match="not a valid date"):
        ensure_consent(settings)


# --------------------------------------------------------------------- paths


def test_project_root_prefers_twin_home_then_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert project_root(env={"TWIN_HOME": str(tmp_path)}) == tmp_path.resolve()
    monkeypatch.delenv("TWIN_HOME")
    marked = tmp_path / "proj"
    (marked / "config").mkdir(parents=True)
    (marked / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = marked / "src" / "pkg"
    nested.mkdir(parents=True)
    assert project_root(nested, env={}) == marked.resolve()
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    assert project_root(lonely, env={}) == lonely.resolve()


def test_config_path_env_override(tmp_path: Path) -> None:
    assert default_config_path(env={"TWIN_CONFIG": str(tmp_path / "x.yaml")}) == tmp_path / "x.yaml"


def test_relative_paths_are_anchored_at_the_project_root(tmp_path: Path) -> None:
    settings = load_settings(None, {"paths": {"data_dir": "./var", "export_dir": "../exports"}})
    paths = resolve_paths(settings, tmp_path)
    assert paths.data_dir == (tmp_path / "var").resolve()
    assert paths.export_dir == (tmp_path.parent / "exports").resolve()
    assert paths.db_path == paths.data_dir / "twin.db"
    absolute = load_settings(None, {"paths": {"data_dir": str(tmp_path / "abs")}})
    assert resolve_paths(absolute, tmp_path / "other").data_dir == tmp_path / "abs"
    assert resolve_paths(load_settings(), tmp_path).export_dir is None


def test_data_paths_ensure_creates_directories(tmp_path: Path) -> None:
    paths = resolve_paths(
        load_settings(None, {"paths": {"data_dir": str(tmp_path / "d")}}), tmp_path
    )
    paths.ensure()
    assert paths.logs_dir.is_dir() and paths.locks_dir.is_dir()
    assert {
        paths.media_dir.name,
        paths.tmp_dir.name,
        paths.reports_dir.name,
        paths.backups_dir.name,
    } == {
        "media",
        "tmp",
        "reports",
        "backups",
    }


def test_settings_class_is_importable_without_sources() -> None:
    assert Settings().jobs.concurrency == 2
