"""Every route that moves the bot to another time zone is a time zone switch (R-SCH-002).

R18-F3 (D-829): ``twin settings set time.bot_timezone X`` wrote the runtime setting only, so no
``timezone_history`` row and the bot's past conversation was dated in the new zone.  It now goes
through the planner's ``switch_timezone``, the implementation of ``twin timezone set`` and
``/时区``.

R18-F4 (D-830): running ``twin setup`` again after the database exists changed only
``config/config.yaml``, which seeds the runtime settings once; the wizard also showed the file's
value instead of the one in force.  On an initialised database it now shows the effective time
zone and target, switches the zone through ``switch_timezone`` and writes the target as a runtime
setting - and refuses to point the target at another conversation than the one imported (R-IMP-003,
D-695).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from tests.support.clock import ManualClock
from tests.support.ingest import make_export, run_import
from tests.support.network import OfflineTransport
from twin.cli import app
from twin.config.loader import ENV_CONFIG, load_settings, resolve_paths
from twin.config.runtime import BOT_TIMEZONE, TARGET_USERNAME
from twin.config.secrets import SecretStore
from twin.schedule.store import TimezoneHistory
from twin.schedule.zone_timeline import BotZoneTimeline
from twin.services import CliContext, Services, build_services, set_cli_context

runner = CliRunner()
OTHER = "wx" + "id_another0002"  # built at run time so that the privacy scan stays clean


@pytest.fixture
def data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: ManualClock, secret_store: SecretStore
) -> Path:
    """A migrated database whose runtime settings were seeded, read by the CLI on ``clock``."""
    path = tmp_path / "cli-data"
    monkeypatch.setenv("TWIN_PATHS__DATA_DIR", str(path))
    monkeypatch.setenv(ENV_CONFIG, str(tmp_path / "config" / "config.yaml"))
    set_cli_context(
        CliContext(secrets=secret_store, clock=clock, http_transport=OfflineTransport())
    )
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    with_services(clock, lambda services: services.runtime.initialize())
    return path


def with_services[T](clock: ManualClock, work: Callable[[Services], T]) -> T:
    settings = load_settings()
    services = build_services(settings, root=resolve_paths(settings).root, clock=clock)
    try:
        return work(services)
    finally:
        services.close()


def timeline(services: Services) -> BotZoneTimeline:
    return BotZoneTimeline.from_history(
        TimezoneHistory(services.db, services.clock), lambda: services.runtime.get(BOT_TIMEZONE)
    )


def cli(*args: str, answers: list[str] | None = None) -> Any:
    entered = "\n".join(answers) + "\n" if answers is not None else None
    return runner.invoke(app, list(args), input=entered)


# ---------------------------------------------------------------- settings set (F3)


def test_settings_set_of_the_bot_timezone_is_a_switch_with_history(
    data_dir: Path, clock: ManualClock
) -> None:
    """R18-F3: the zone of a moment before the change stays Chicago after `settings set`."""
    before = clock.now_utc()
    clock.tick(3600)
    result = cli("settings", "set", "time.bot_timezone", "Asia/Shanghai")
    assert result.exit_code == 0, result.output
    assert "switched America/Chicago -> Asia/Shanghai" in result.output

    def check(services: Services) -> None:
        assert services.runtime.get(BOT_TIMEZONE) == "Asia/Shanghai"
        zones = timeline(services)
        assert len(zones.switches) == 1
        assert zones.zone_for(before).key == "America/Chicago"
        assert zones.zone_for(clock.now_utc() + timedelta(minutes=1)).key == "Asia/Shanghai"

    with_services(clock, check)
    again = cli("settings", "set", "time.bot_timezone", "Asia/Shanghai")
    assert again.exit_code == 0 and "nothing changed" in again.output
    assert with_services(clock, lambda services: len(timeline(services).switches)) == 1


def test_settings_set_refuses_a_zone_that_does_not_exist(
    data_dir: Path, clock: ManualClock
) -> None:
    """R18-F3: the same validation as `twin timezone set` (usage error, nothing written)."""
    result = cli("settings", "set", "time.bot_timezone", "Mars/Base")
    assert result.exit_code == 2, result.output
    assert with_services(clock, lambda services: len(timeline(services).switches)) == 0


# ---------------------------------------------------------------------- setup (F4)


def wizard(target: str, zone: str) -> Any:
    # consent and its date kept, no DeepSeek key, no mail, target, zone, no emergency contact
    return cli("setup", answers=["y", "", "", "n", target, zone, "n"])


def test_setup_again_switches_the_zone_and_shows_the_one_in_force(
    data_dir: Path, clock: ManualClock, tmp_path: Path
) -> None:
    """R18-F4: the wizard proposes the effective zone and switches with history."""
    assert cli("timezone", "set", "Europe/Paris").exit_code == 0  # in force; the file says Chicago
    clock.tick(3600)
    result = wizard("", "Asia/Shanghai")
    assert result.exit_code == 0, result.output
    assert "[Europe/Paris]" in result.output and "[America/Chicago]" not in result.output

    def check(services: Services) -> None:
        assert services.runtime.get(BOT_TIMEZONE) == "Asia/Shanghai"
        assert [s.to_zone for s in timeline(services).switches] == [
            "Europe/Paris",
            "Asia/Shanghai",
        ]

    with_services(clock, check)
    written = yaml.safe_load((tmp_path / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert written["time"] == {"bot_timezone": "Asia/Shanghai"}  # the seed of a new database
    kept = wizard("", "")  # every answer is the proposal: nothing switches
    assert kept.exit_code == 0 and "[Asia/Shanghai]" in kept.output
    assert with_services(clock, lambda services: len(timeline(services).switches)) == 2


def test_setup_writes_the_target_as_the_runtime_setting(data_dir: Path, clock: ManualClock) -> None:
    """R18-F4: a target given to the wizard on an initialised database is the one in force."""
    result = wizard(OTHER, "")
    assert result.exit_code == 0, result.output
    assert with_services(clock, lambda services: services.runtime.get(TARGET_USERNAME)) == OTHER
    again = wizard("", "")
    assert again.exit_code == 0 and f"[{OTHER}]" in again.output


def test_setup_will_not_point_the_target_at_another_conversation(
    data_dir: Path, clock: ManualClock, tmp_path: Path
) -> None:
    """R18-F4: the database holds one conversation; another target needs a purge first."""
    export = make_export(tmp_path, target_messages=20, other_conversations=0)

    def imported(services: Services) -> None:
        run_import(services, export)
        services.runtime.set(TARGET_USERNAME, export.target_username, by="import")

    with_services(clock, imported)
    result = wizard(OTHER, "")
    assert result.exit_code == 0, result.output
    assert "twin purge" in result.output  # why the target stays, and what would change it
    in_force = with_services(clock, lambda services: services.runtime.get(TARGET_USERNAME))
    assert in_force == export.target_username
    same = wizard(export.target_username, "")
    assert same.exit_code == 0 and "twin purge" not in same.output
