"""Simulate days of the proactive scheduler and print the statistics (R-PRO-008).

A development tool, not a ``twin`` command: it runs the **production** scheduler
(``twin.schedule.proactive``) over whole local days on a manual clock, in a throw-away data
directory, with DeepSeek replaced by a scripted fake of ``tests/support`` and a synthetic user who
writes in sessions.  It imports ``tests.support``, which is why it lives here and not in the
package.  Nothing it prints comes from real chat data, and nothing it does touches the real data
directory, the real credentials or the network.

Usage::

    uv run python scripts/simulate_proactive.py --days 14
    uv run python scripts/simulate_proactive.py --days 30 --user all-day --seed 7
    uv run python scripts/simulate_proactive.py --days 10 --silent 3,4,5 --min 2 --max 4

The output is a table per day (messages, the range and the quota of the day, what was sent, what
was refused), the totals, the local hours the messages went out in, and the audit of
``twin eval proactive`` over the same days.  The exit code is 1 when the audit finds a message in
deep sleep, a spacing or chase violation, or a day outside its range.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `tests.support` lives next to `src`

USERS = ("evening", "all-day")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--days", type=int, default=14, help="local days to simulate (14)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the user and the dice (1)")
    parser.add_argument("--salt", default="simulation", help="the plan salt: another set of days")
    parser.add_argument("--first-day", default="2026-10-12", help="first local day (a Monday)")
    parser.add_argument("--zone", default=None, help="the bot's time zone (America/Chicago)")
    parser.add_argument(
        "--user", choices=USERS, default="evening", help="when the user writes (evening)"
    )
    parser.add_argument(
        "--silent", default="", help="comma-separated day numbers (0-based) without any message"
    )
    parser.add_argument("--min", type=int, default=1, help="messages a day, at least (1)")
    parser.add_argument("--max", type=int, default=6, help="messages a day, at most (6)")
    parser.add_argument(
        "--reply-probability", type=float, default=0.7, help="how often he answers her (0.7)"
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")
    if not 0 <= args.min <= args.max:
        parser.error("--min must not be above --max")
    return args


def isolate(home: Path) -> None:
    """Keep the run away from the real home, credentials and clock (as the tests do)."""
    for name in list(os.environ):
        if name.startswith("TWIN_"):
            del os.environ[name]
    os.environ["TWIN_HOME"] = str(home)
    os.environ["TWIN_SECRETS_DIR"] = str(home / "secrets")
    os.environ["TWIN_KEYRING_BACKEND"] = "file"
    os.environ["XDG_CONFIG_HOME"] = str(home / "xdg")
    os.environ["COLUMNS"] = "160"


async def run(args: argparse.Namespace, home: Path) -> int:
    import respx
    from rich.console import Console
    from tests.support.clock import ManualClock
    from tests.support.deepseek import API
    from tests.support.proactive_sim import ALL_DAY, EVENING, UserBehavior, render, simulate
    from tests.support.proactive_world import ProactiveScript, build_world

    from twin.clock import set_active_clock
    from twin.config.loader import load_settings
    from twin.config.runtime import PROACTIVE_DAILY_MAX, PROACTIVE_DAILY_MIN
    from twin.config.secrets import SecretStore, select_backend
    from twin.eval.proactive_audit import audit_days
    from twin.eval.proactive_report import print_audit
    from twin.schedule.plan_builder import QuotaRange
    from twin.services import build_services
    from twin.storage import migrate

    clock = ManualClock()
    set_active_clock(clock)
    settings = load_settings(None, {"paths": {"data_dir": str(home / "data")}})
    migrate.upgrade(Path(settings.paths.data_dir) / "twin.db")
    backend, info = select_backend()
    services = build_services(settings, root=home, secrets=SecretStore(backend, info), clock=clock)
    try:
        services.runtime.set(PROACTIVE_DAILY_MIN, args.min, by="simulation")
        services.runtime.set(PROACTIVE_DAILY_MAX, args.max, by="simulation")
        script = ProactiveScript()
        with respx.mock(assert_all_called=False) as router:
            router.post(API).mock(side_effect=script)
            world = build_world(
                services,
                clock,
                quota=QuotaRange(args.min, args.max),
                zone=args.zone,
                script=script,
                salt=args.salt,
            )
            try:
                silent = frozenset(int(n) for n in args.silent.split(",") if n.strip())
                behavior = UserBehavior(
                    reply_probability=args.reply_probability,
                    sessions=ALL_DAY if args.user == "all-day" else (EVENING,),
                    silent_days=silent,
                )
                first = date.fromisoformat(args.first_day)
                result = await simulate(
                    world, first_day=first, days=args.days, behavior=behavior, seed=args.seed
                )
                print(render(result))
                audit = audit_days(
                    world.log,
                    world.ratings,
                    services.settings.proactive,
                    first_day=first,
                    last_day=first + timedelta(days=args.days - 1),
                    now=clock.now_utc(),
                )
                print()
                print_audit(Console(highlight=False), audit)
                broken = (
                    audit.deep_sleep
                    or audit.spacing_violations
                    or audit.chase_violations
                    or not audit.count_ok
                )
                return 1 if broken else 0
            finally:
                await world.aclose()
    finally:
        services.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="twin-simulation-") as folder:
        home = Path(folder)
        isolate(home)
        return asyncio.run(run(args, home))


if __name__ == "__main__":
    sys.exit(main())
