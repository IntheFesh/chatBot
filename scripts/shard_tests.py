"""Deterministic file-level test sharding for the CI matrix (docs/CI.md, D-490..D-499).

``.github/workflows/ci.yml`` runs the test suite as one job per shard, each on its own VM and
each sequential inside (a few tests use Windows named mutexes or fixed local ports, so tests of
one machine never run in parallel).  This tool decides which test files belong to which shard:

* **Automatic.**  The universe is every ``test_*.py`` / ``*_test.py`` below ``tests/`` found on
  disk, never a hand-written list, so a new file is covered by the next run without anybody
  touching a list.  Every file belongs to exactly one shard; the union of the shards is the
  universe; ``--of 1`` selects the universe.
* **Whole files.**  Module-level and class-level fixtures stay intact.
* **Balanced.**  Longest-processing-time greedy over per-file weights.  The weights are the
  committed seconds per file and platform in ``ci/test_weights.json`` (refresh them with
  ``--update-weights``).  A file without a recorded weight gets the number of tests it defines,
  scaled by the seconds-per-test of the recorded files so that both kinds of numbers compare.
* **Deterministic.**  Ties break by path and shard index; the same tree and weights always give
  the same shards.

Usage::

    python scripts/shard_tests.py --shard 2 --of 4              # files of shard 2/4, one per line
    python scripts/shard_tests.py --of 4 --plan                 # all shards with their weights
    python scripts/shard_tests.py --timings junit.xml           # seconds per test file
    python scripts/shard_tests.py --update-weights junit.xml    # refresh ci/test_weights.json
    python scripts/shard_tests.py --check-collection            # pytest collects no unknown file
    python scripts/shard_tests.py --check-shard-set NAME-1-of-3 NAME-2-of-3 NAME-3-of-3

Shards are numbered from 1.  The tool uses only the standard library (``--check-collection``
starts pytest in a subprocess).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = "tests"
WEIGHTS_RELATIVE = Path("ci") / "test_weights.json"
WEIGHTS_FILE = ROOT / WEIGHTS_RELATIVE
PLATFORMS = ("linux", "windows")
WEIGHTS_VERSION = 1
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")  # pytest's default ``python_files``
SHARD_SUFFIX = re.compile(r"(\d+)-of-(\d+)$")
TIMING_LINE = re.compile(r"(\d+(?:\.\d+)?)s\s+(tests/\S+\.py)\b")


class ShardError(Exception):
    """A problem with the arguments, the weights file or the shard files."""


# ----------------------------------------------------------------------------- discovery


def discover_test_files(root: Path = ROOT) -> list[str]:
    """Every test file below ``<root>/tests`` as a sorted POSIX path relative to ``root``."""
    base = root / TEST_DIR
    found = {
        path.relative_to(root).as_posix()
        for pattern in TEST_FILE_PATTERNS
        for path in base.rglob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(found)


def _literal_length(node: ast.expr) -> int:
    """Number of cases a ``parametrize`` values expression yields; 1 when it is not literal."""
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return max(1, len(node.elts))
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return 1
    try:
        return max(1, len(value))
    except TypeError:
        return 1


def _cases_of(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    cases = 1
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and len(decorator.args) >= 2
        ):
            cases *= _literal_length(decorator.args[1])
    return cases


def _count_in(body: Iterable[ast.stmt]) -> int:
    total = 0
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test"
        ):
            total += _cases_of(node)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            total += _count_in(node.body)
    return total


def count_tests(path: Path) -> int:
    """How many tests ``path`` defines (at least 1; ``parametrize`` with literal values counts).

    A static estimate that needs no imports; it only prices files that have no recorded weight.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return 1
    return max(1, _count_in(tree.body))


# ------------------------------------------------------------------------------- weights


def current_platform() -> str:
    return "windows" if sys.platform == "win32" else "linux"


def load_weights(path: Path = WEIGHTS_FILE) -> dict[str, dict[str, float]]:
    """``{platform: {test file: seconds}}``; a missing file means no recorded weights."""
    if not path.is_file():
        return {name: {} for name in PLATFORMS}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ShardError(f"cannot read {path}: {error}") from error
    if not isinstance(data, dict) or data.get("version") != WEIGHTS_VERSION:
        raise ShardError(f"{path}: expected an object with version {WEIGHTS_VERSION}")
    platforms = data.get("platforms")
    if not isinstance(platforms, dict):
        raise ShardError(f"{path}: 'platforms' must be an object")
    table: dict[str, dict[str, float]] = {name: {} for name in PLATFORMS}
    for name, files in platforms.items():
        if name not in PLATFORMS:
            raise ShardError(f"{path}: unknown platform {name!r} (expected {PLATFORMS})")
        if not isinstance(files, dict):
            raise ShardError(f"{path}: platforms.{name} must map file to seconds")
        for file, seconds in files.items():
            if isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds < 0:
                raise ShardError(f"{path}: platforms.{name}[{file!r}] is not a number >= 0")
            table[name][str(file)] = float(seconds)
    return table


def save_weights(table: Mapping[str, Mapping[str, float]], path: Path = WEIGHTS_FILE) -> None:
    document = {
        "version": WEIGHTS_VERSION,
        "unit": "seconds of one sequential run of the file (relative weights; only ratios matter)",
        "platforms": {
            name: {
                file: round(table.get(name, {})[file], 2) for file in sorted(table.get(name, {}))
            }
            for name in PLATFORMS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


@dataclass(frozen=True)
class Weighing:
    """The weight of every file and how each one was priced."""

    weights: dict[str, float]
    recorded: int  # files priced from a recorded time
    estimated: int  # files priced from their number of tests
    stale: int  # recorded files that no longer exist
    borrowed_from: str | None  # platform whose times were used because this one has none


def weigh(
    files: Sequence[str],
    table: Mapping[str, Mapping[str, float]],
    platform: str,
    root: Path = ROOT,
) -> Weighing:
    """Price ``files``: the recorded seconds, else the test count scaled to seconds."""
    if platform not in PLATFORMS:
        raise ShardError(f"unknown platform {platform!r} (expected one of {PLATFORMS})")
    recorded = dict(table.get(platform, {}))
    borrowed: str | None = None
    if not recorded:
        for other in PLATFORMS:
            if table.get(other):
                recorded, borrowed = dict(table[other]), other
                break
    known = [file for file in files if file in recorded]
    unknown = [file for file in files if file not in recorded]
    counts = {file: count_tests(root / file) for file in files}
    known_seconds = sum(recorded[file] for file in known)
    known_tests = sum(counts[file] for file in known)
    rate = known_seconds / known_tests if known_seconds > 0 and known_tests > 0 else 1.0
    weights = {file: recorded[file] for file in known}
    weights.update({file: counts[file] * rate for file in unknown})
    present = set(files)
    stale = sum(1 for file in recorded if file not in present)
    return Weighing(weights, len(known), len(unknown), stale, borrowed)


# ------------------------------------------------------------------------------ sharding


def assign(weights: Mapping[str, float], count: int) -> list[list[str]]:
    """Longest-processing-time greedy: heaviest file first onto the lightest shard.

    Ties break by path and then by the lowest shard index, so the result is a pure function of
    the weights.  Each file lands in exactly one shard; each shard lists its files sorted by path.
    """
    if count < 1:
        raise ShardError("--of must be at least 1")
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for file in sorted(weights, key=lambda name: (-weights[name], name)):
        target = min(range(count), key=lambda index: (loads[index], index))
        shards[target].append(file)
        loads[target] += weights[file]
    return [sorted(files) for files in shards]


def select_shard(
    index: int,
    count: int,
    *,
    platform: str | None = None,
    root: Path = ROOT,
    weights_file: Path | None = None,
) -> tuple[list[str], Weighing]:
    """The files of shard ``index`` (1-based) of ``count`` and how the files were priced."""
    if not 1 <= index <= count:
        raise ShardError(f"--shard must be between 1 and --of ({count}), got {index}")
    files = discover_test_files(root)
    if not files:
        raise ShardError(f"no test files found below {root / TEST_DIR}")
    table = load_weights(weights_file if weights_file is not None else root / WEIGHTS_RELATIVE)
    weighing = weigh(files, table, platform or current_platform(), root)
    return assign(weighing.weights, count)[index - 1], weighing


# ------------------------------------------------------------------------------ timings


def _module_file(classname: str, root: Path) -> str | None:
    """``tests.unit.test_x.TestY`` -> ``tests/unit/test_x.py`` (the longest module prefix)."""
    parts = classname.split(".")
    for length in range(len(parts), 0, -1):
        candidate = Path(*parts[:length]).with_suffix(".py")
        if (root / candidate).is_file():
            return candidate.as_posix()
    return None


def timings_from_junit(path: Path, root: Path = ROOT) -> dict[str, tuple[float, int]]:
    """``{test file: (seconds, tests)}`` summed from a pytest ``--junitxml`` report."""
    try:
        tree = ElementTree.parse(path)  # noqa: S314 - our own pytest report, not untrusted input
    except (OSError, ElementTree.ParseError) as error:
        raise ShardError(f"cannot read the junit report {path}: {error}") from error
    result: dict[str, tuple[float, int]] = {}
    for case in tree.iter("testcase"):
        file = case.get("file")
        name = file.replace("\\", "/") if file else _module_file(case.get("classname", ""), root)
        if name is None:
            continue
        seconds, tests = result.get(name, (0.0, 0))
        result[name] = (seconds + float(case.get("time", "0") or 0), tests + 1)
    return result


def timings_from_log(text: str) -> dict[str, tuple[float, int]]:
    """Read the table this tool prints (``--timings``), also out of a CI log with prefixes."""
    result: dict[str, tuple[float, int]] = {}
    for line in text.splitlines():
        match = TIMING_LINE.search(line)
        if match:
            result[match.group(2)] = (float(match.group(1)), 0)
    return result


def read_timings(sources: Sequence[Path], root: Path = ROOT) -> dict[str, tuple[float, int]]:
    """Merge junit reports (files or directories of them) and ``--timings`` logs."""
    merged: dict[str, tuple[float, int]] = {}
    for source in sources:
        if source.is_dir():
            paths = sorted(source.rglob("*.xml"))
        elif source.is_file():
            paths = [source]
        else:
            raise ShardError(f"{source} does not exist")
        for path in paths:
            if path.suffix.lower() == ".xml":
                found = timings_from_junit(path, root)
            else:
                found = timings_from_log(path.read_text(encoding="utf-8", errors="replace"))
            merged.update(found)  # a later source for the same file replaces the earlier one
    if not merged:
        raise ShardError("the sources hold no per-file timings")
    return merged


def format_timings(timings: Mapping[str, tuple[float, int]], platform: str) -> str:
    rows = sorted(timings.items(), key=lambda item: (-item[1][0], item[0]))
    total = sum(seconds for seconds, _ in timings.values())
    lines = [f"seconds per test file ({platform}, {len(rows)} files, {total:.1f}s in total):"]
    lines += [f"{seconds:9.1f}s  {file}  {tests} tests" for file, (seconds, tests) in rows]
    return "\n".join(lines)


def updated_weights(
    table: Mapping[str, Mapping[str, float]],
    timings: Mapping[str, tuple[float, int]],
    platform: str,
    root: Path = ROOT,
) -> dict[str, dict[str, float]]:
    """``table`` with the measured files replaced for ``platform`` and vanished files dropped."""
    present = set(discover_test_files(root))
    merged = {name: dict(files) for name, files in table.items()}
    merged.setdefault(platform, {})
    for name in merged:
        merged[name] = {file: sec for file, sec in merged[name].items() if file in present}
    merged[platform].update({file: sec for file, (sec, _) in timings.items() if file in present})
    return merged


# --------------------------------------------------------------------------------- checks


def collected_files(output: str) -> set[str]:
    """Test files named by the node ids of ``pytest --collect-only -q``."""
    files: set[str] = set()
    for line in output.splitlines():
        head, separator, _ = line.partition("::")
        if separator and head.endswith(".py"):
            files.add(head.strip().replace("\\", "/"))
    return files


def check_collection(root: Path = ROOT, targets: Sequence[str] = ()) -> list[str]:
    """Problems if pytest collects a test file that sharding would not select (empty = fine).

    ``targets`` narrows what pytest is asked to collect (default: its configured ``testpaths``).
    """
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not live",
            "-p",
            "no:cacheprovider",
            *targets,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if done.returncode != 0:
        return [
            f"pytest --collect-only failed with exit code {done.returncode}:\n{done.stdout[-2000:]}"
        ]
    unknown = sorted(collected_files(done.stdout) - set(discover_test_files(root)))
    return [f"pytest collects {file}, which sharding would never select" for file in unknown]


def check_shard_set(names: Sequence[str]) -> list[str]:
    """Problems unless ``names`` end in ``<i>-of-<n>`` and are exactly the shards 1..n."""
    seen: dict[int, int] = {}
    sizes: set[int] = set()
    problems: list[str] = []
    for name in names:
        match = SHARD_SUFFIX.search(Path(name).name)
        if not match:
            problems.append(f"{name}: no '<i>-of-<n>' suffix")
            continue
        index, size = int(match.group(1)), int(match.group(2))
        sizes.add(size)
        seen[index] = seen.get(index, 0) + 1
    if len(sizes) > 1:
        problems.append(f"shard files disagree on the number of shards: {sorted(sizes)}")
    elif sizes:
        (size,) = sizes
        missing = [index for index in range(1, size + 1) if index not in seen]
        extra = sorted(index for index in seen if not 1 <= index <= size)
        duplicated = sorted(index for index, times in seen.items() if times > 1)
        if missing:
            problems.append(f"missing shards of {size}: {missing}")
        if extra:
            problems.append(f"shard numbers outside 1..{size}: {extra}")
        if duplicated:
            problems.append(f"shards present more than once: {duplicated}")
    if not names:
        problems.append("no shard files given")
    return problems


# -------------------------------------------------------------------------------- output


def emit(lines: Iterable[str]) -> None:
    """One ``\\n``-terminated line each, also on Windows (``mapfile`` in bash dislikes ``\\r``)."""
    sys.stdout.flush()
    sys.stdout.buffer.write("".join(f"{line}\n" for line in lines).encode("utf-8"))
    sys.stdout.buffer.flush()


def describe_plan(
    count: int, platform: str, root: Path = ROOT, weights_file: Path | None = None
) -> list[str]:
    files = discover_test_files(root)
    table = load_weights(weights_file if weights_file is not None else root / WEIGHTS_RELATIVE)
    weighing = weigh(files, table, platform, root)
    shards = assign(weighing.weights, count)
    total = sum(weighing.weights.values()) or 1.0
    lines = [
        f"{len(files)} test files in {count} shards ({platform}); "
        f"{weighing.recorded} priced from recorded seconds, {weighing.estimated} from test counts"
    ]
    for number, members in enumerate(shards, start=1):
        load = sum(weighing.weights[file] for file in members)
        tests = sum(count_tests(root / file) for file in members)
        lines.append(
            f"  shard {number}/{count}: {len(members):>3} files  ~{tests:>5} tests  "
            f"weight {load:9.1f}  ({100 * load / total:4.1f}%)"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--shard", type=int, help="shard number, from 1 (needs --of)")
    parser.add_argument("--of", dest="count", type=int, help="number of shards")
    parser.add_argument("--plan", action="store_true", help="describe all shards of --of")
    parser.add_argument("--platform", choices=PLATFORMS, help="weights to use (default: this OS)")
    parser.add_argument(
        "--weights", type=Path, default=None, help="weights file (<root>/ci/test_weights.json)"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: here)")
    parser.add_argument(
        "--timings", nargs="+", type=Path, metavar="SOURCE", help="print seconds per file"
    )
    parser.add_argument(
        "--update-weights",
        nargs="+",
        type=Path,
        metavar="SOURCE",
        help="merge junit reports / --timings logs into the weights file",
    )
    parser.add_argument(
        "--check-collection", action="store_true", help="pytest collects no unknown file"
    )
    parser.add_argument(
        "--check-shard-set", nargs="*", metavar="NAME", help="names end in <i>-of-<n>"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    platform = args.platform or current_platform()
    root: Path = args.root
    weights_file: Path = args.weights if args.weights is not None else root / WEIGHTS_RELATIVE
    modes = [
        args.shard is not None,
        args.plan,
        args.timings is not None,
        args.update_weights is not None,
        args.check_collection,
        args.check_shard_set is not None,
    ]
    if sum(modes) != 1:
        parser.error(
            "choose exactly one of --shard, --plan, --timings, --update-weights, "
            "--check-collection, --check-shard-set"
        )
    try:
        if args.shard is not None or args.plan:
            if args.count is None:
                parser.error("--of is required")
            if args.plan:
                emit(describe_plan(args.count, platform, root, weights_file))
                return 0
            files, weighing = select_shard(
                args.shard, args.count, platform=platform, root=root, weights_file=weights_file
            )
            sys.stderr.write(
                f"shard {args.shard}/{args.count} ({platform}): {len(files)} files; "
                f"{weighing.recorded} priced from recorded seconds, {weighing.estimated} from "
                f"test counts, {weighing.stale} stale entries in {weights_file.name}"
                + (
                    f", weights borrowed from {weighing.borrowed_from}"
                    if weighing.borrowed_from
                    else ""
                )
                + "\n"
            )
            emit(files)
            return 0
        if args.timings is not None:
            emit([format_timings(read_timings(args.timings, root), platform)])
            return 0
        if args.update_weights is not None:
            merged = updated_weights(
                load_weights(weights_file), read_timings(args.update_weights, root), platform, root
            )
            save_weights(merged, weights_file)
            emit([f"{weights_file}: {len(merged[platform])} {platform} weights"])
            return 0
        problems = (
            check_collection(root)
            if args.check_collection
            else check_shard_set(args.check_shard_set)
        )
    except ShardError as error:
        sys.stderr.write(f"shard_tests: {error}\n")
        return 2
    if problems:
        sys.stderr.write("\n".join(problems) + "\n")
        return 1
    emit(["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
