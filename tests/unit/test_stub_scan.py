"""scripts/stub_scan.py: the keyword, syntax and import rules (R-NFR-004, CLAUDE.md rules 1-2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.support.scripts import SCRIPTS, load_script

scan = load_script("stub_scan")


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def rules(findings: list[object]) -> list[str]:
    return [f.rule for f in findings]  # type: ignore[attr-defined]


# ------------------------------------------------------------------------- keyword rule


@pytest.mark.parametrize(
    "line",
    [
        "# TODO: later",
        "x = 1  # fixme",
        "label = 'XXX'",
        "raise NotImplementedError",
        "class StubBackend: ...",
        "stub = 1",
        "placeholder_text = ''",
        "from unittest import mock",
        "fake_clock = None",
        "class FakeClock: ...",
        "dummy = 0",
        "message = '这是简化版'",
        "message = '示例数据'",
        "message = '暂不支持'",
        "message = '以后再做'",
    ],
)
def test_every_forbidden_word_is_found_in_code_comments_and_strings(line: str) -> None:
    assert scan.keyword_hits(line), line


@pytest.mark.parametrize(
    "line",
    [
        "x = 1",
        "stubborn = True",  # a different word
        "the todolist of the day",  # not a word of the list
        "textual = 'xxxl'",
        "message = '请先停止应用'",
        "def clock(): ...",
    ],
)
def test_ordinary_words_are_not_reported(line: str) -> None:
    assert scan.keyword_hits(line) == [], line


def test_the_words_are_found_whatever_their_case_and_spelling_of_the_identifier() -> None:
    assert scan.keyword_hits("FAKE_CLOCK = 1") and scan.keyword_hits("fakeClock = 1")
    assert scan.keyword_hits("Todo") and scan.keyword_hits("a_todo_b")
    assert scan.keyword_hits("MockServer") and scan.keyword_hits("DUMMY")


def test_a_finding_names_the_file_the_line_and_what_was_found() -> None:
    findings = scan.scan_keywords("src/twin/x.py", "ok = 1\nvalue = fake_value()\n")
    assert [(f.path, f.line, f.rule) for f in findings] == [("src/twin/x.py", 2, "keyword")]
    assert "fake" in findings[0].detail and "value = fake_value()" in findings[0].excerpt


# ------------------------------------------------------------------------- syntax rules


def functions(text: str) -> list[tuple[str, str]]:
    found = scan.scan_functions("src/twin/m.py", src(text))
    return [(f.symbol, f.rule) for f in found]


def test_a_body_of_pass_dots_or_only_a_docstring_is_empty() -> None:
    assert functions(
        '''
        def a():
            pass

        def b():
            ...

        def c():
            """Only words."""

        def d():
            """Words."""
            pass
        '''
    ) == [(name, "empty-body") for name in "abcd"]


@pytest.mark.parametrize(
    "value", ["None", "True", "False", "0", "3", "-1", "1.5", "'text'", "b'x'", "[]", "{}", "()"]
)
def test_a_function_that_only_returns_a_constant_is_reported(value: str) -> None:
    assert functions(f"def f():\n    return {value}\n") == [("f", "constant-return")]


def test_bare_return_and_empty_builtin_calls_count_as_constants() -> None:
    assert functions("def f():\n    return\n") == [("f", "constant-return")]
    assert functions("def f():\n    return list()\n") == [("f", "constant-return")]
    assert functions("def f():\n    return dict()\n") == [("f", "constant-return")]
    assert functions("def f():\n    return set()\n") == [("f", "constant-return")]


def test_methods_async_functions_and_nested_functions_are_all_looked_at() -> None:
    found = functions(
        """
        class Box:
            def method(self):
                return False

            async def later(self):
                return None

        def outer():
            def inner():
                pass
            return inner
        """
    )
    assert found == [
        ("Box.method", "constant-return"),
        ("Box.later", "constant-return"),
        ("outer.inner", "empty-body"),
    ]


def test_a_body_that_does_any_work_is_not_reported() -> None:
    assert (
        functions(
            """
        def a(x):
            return x

        def b():
            return [1]

        def c():
            return {"k": 1}

        def d(x):
            if x:
                return True
            return False

        def e():
            log("x")
            return None

        def f():
            return f"{value}"

        def g():
            return compute()

        def h():
            raise ValueError("no")
        """
        )
        == []
    )


def test_protocol_members_abstract_methods_and_overloads_are_exempt() -> None:
    assert (
        functions(
            """
        from abc import ABC, abstractmethod
        from typing import Protocol, overload
        import typing

        class P(Protocol):
            def a(self) -> int: ...
            def b(self) -> bool:
                return True

        class Q(typing.Protocol):
            def a(self) -> int:
                pass

        class R(Protocol[int]):
            def a(self) -> int: ...

        class S(ABC):
            @abstractmethod
            def a(self) -> int: ...

            @abc.abstractmethod
            def b(self) -> int:
                pass

        @overload
        def f(x: int) -> int: ...
        @overload
        def f(x: str) -> str: ...
        def f(x):
            return x
        """
        )
        == []
    )


def test_a_method_of_an_ordinary_class_is_not_exempt_because_of_its_neighbours() -> None:
    assert functions(
        """
        from typing import Protocol

        class P(Protocol):
            def a(self) -> int: ...

        class Impl:
            def a(self) -> int:
                return 1
        """
    ) == [("Impl.a", "constant-return")]


def test_work_inside_if_try_and_match_blocks_is_found() -> None:
    assert functions(
        """
        import sys
        if sys.platform == "win32":
            def win():
                pass
        else:
            def other():
                return False
        try:
            def guarded():
                ...
        except ImportError:
            def fallback():
                return None
        """
    ) == [
        ("win", "empty-body"),
        ("other", "constant-return"),
        ("guarded", "empty-body"),
        ("fallback", "constant-return"),
    ]


def test_the_reported_line_is_the_return_or_the_def() -> None:
    found = scan.scan_functions(
        "src/twin/m.py",
        src(
            '''
            def a():
                """Doc."""
                return 5

            def b():
                pass
            '''
        ),
    )
    assert [(f.line, f.excerpt) for f in found] == [(3, "return 5"), (5, "def b():")]


# --------------------------------------------------------------------------- import rule


def modules(**texts: str) -> dict[str, tuple[str, str]]:
    return {
        name.replace("__", "."): (f"src/{name.replace('__', '/')}.py", src(text))
        for name, text in texts.items()
    }


def test_src_importing_tests_is_found_in_every_form() -> None:
    found = scan.scan_imports(
        modules(
            twin__a="import tests.support.x\n",
            twin__b="from tests.support import y\n",
            twin__c="import importlib\nimportlib.import_module('tests.support.z')\n",
            twin__d="__import__('scripts.privacy_scan')\n",
            twin__e="import os\nfrom twin import a\n",
        )
    )
    assert sorted({f.symbol for f in found}) == ["twin.a", "twin.b", "twin.c", "twin.d"]
    assert set(rules(found)) == {"tests-import"}


def test_an_import_of_tests_that_the_command_line_can_reach_is_reported_with_its_chain() -> None:
    found = scan.scan_imports(
        modules(
            twin__cli="from twin.engine import runner\n",
            twin__engine__runner="from twin import helper\n",
            twin__helper="from tests.support.deepseek import ok\n",
            twin__unused="import os\n",
        )
    )
    chains = [f.detail for f in found if f.detail.startswith("reachable")]
    assert chains == [
        "reachable from the command line: twin.cli -> twin.engine.runner -> twin.helper"
    ]


def test_relative_imports_are_resolved_for_the_chain() -> None:
    graph = modules(
        twin__cli="from . import part\n",
        twin__part="from tests import support\n",
    )
    found = scan.scan_imports(graph)
    assert any("reachable" in f.detail and "twin.part" in f.detail for f in found)


def test_the_package_init_stands_for_its_package() -> None:
    root = Path("/repo/src")
    assert scan.module_name(root / "twin" / "a" / "__init__.py", root) == "twin.a"
    assert scan.module_name(root / "twin" / "a" / "b.py", root) == "twin.a.b"


# ----------------------------------------------------------------------------- allowlist


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src(text), encoding="utf-8")
    return path


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        write(tmp_path, name, text)
    return tmp_path


def test_a_repository_with_a_stand_in_fails_and_names_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_repo(
        tmp_path,
        {
            "src/twin/__init__.py": "",
            "src/twin/a.py": "def f():\n    return True\n",
            "src/twin/b.py": "x = 1  # TODO later\n",
        },
    )
    assert scan.main(["--root", str(root), "--allowlist", str(tmp_path / "none.toml")]) == 1
    out = capsys.readouterr().out
    assert "stub scan FAILED" in out
    assert "src/twin/a.py:2: constant-return in f" in out and "src/twin/b.py:1: keyword" in out


def test_a_clean_repository_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_repo(
        tmp_path,
        {"src/twin/__init__.py": "", "src/twin/a.py": "def f(x):\n    return x + 1\n"},
    )
    assert scan.main(["--root", str(root), "--allowlist", str(tmp_path / "none.toml")]) == 0
    assert "stub scan passed" in capsys.readouterr().out


def test_templates_and_other_text_files_under_src_are_scanned_too(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {"src/twin/__init__.py": "", "src/twin/templates/p.md": "给出示例回答\n"},
    )
    findings, count = scan.scan_tree(root)
    assert count == 2 and [(f.path, f.rule) for f in findings] == [
        ("src/twin/templates/p.md", "keyword")
    ]


def test_an_allowlist_entry_with_a_reason_covers_the_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_repo(
        tmp_path,
        {
            "src/twin/__init__.py": "",
            "src/twin/a.py": "class Policy:\n    def allowed(self):\n        return False\n",
            "src/twin/b.py": "note = 'a stub'  # the word in a quoted sentence\n",
        },
    )
    allow = write(
        tmp_path,
        "allow.toml",
        """
        [[allow]]
        rule = "constant-return"
        file = "src/twin/a.py"
        symbol = "Policy.allowed"
        reason = "Fail-closed policy: it allows nothing by definition."

        [[allow]]
        rule = "keyword"
        file = "src/twin/b.py"
        contains = "a stub"
        reason = "A quoted sentence that talks about the word, not a stand-in."
        """,
    )
    assert scan.main(["--root", str(root), "--allowlist", str(allow), "--show-allowed"]) == 0
    out = capsys.readouterr().out
    assert "allowed  src/twin/a.py:3" in out and "2 findings covered by 2 allowlist" in out


def test_an_entry_covers_only_the_symbol_and_the_rule_it_names(tmp_path: Path) -> None:
    root = make_repo(
        tmp_path,
        {
            "src/twin/__init__.py": "",
            "src/twin/a.py": "class A:\n    def x(self):\n        return False\n"
            "    def y(self):\n        return False\n",
        },
    )
    allow = write(
        tmp_path,
        "allow.toml",
        """
        [[allow]]
        rule = "constant-return"
        file = "src/twin/a.py"
        symbol = "A.x"
        reason = "Only the first method is a legitimate constant."
        """,
    )
    findings, _ = scan.scan_tree(root)
    remaining, allowed, stale = scan.apply_allowlist(findings, scan.load_allowlist(allow))
    assert [f.symbol for f in allowed] == ["A.x"]
    assert [f.symbol for f in remaining] == ["A.y"] and stale == []


def test_an_entry_that_matches_nothing_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_repo(tmp_path, {"src/twin/__init__.py": "", "src/twin/a.py": "x = 1\n"})
    allow = write(
        tmp_path,
        "allow.toml",
        """
        [[allow]]
        rule = "constant-return"
        file = "src/twin/a.py"
        symbol = "gone"
        reason = "The function was removed long ago."
        """,
    )
    assert scan.main(["--root", str(root), "--allowlist", str(allow)]) == 1
    assert "matches nothing" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("entry", "complaint"),
    [
        ('rule = "constant-return"\nfile = "a.py"\nsymbol = "f"\nreason = "short"', "reason"),
        ('rule = "constant-return"\nfile = "a.py"\nreason = "A long enough reason."', "symbol"),
        ('rule = "keyword"\nfile = "a.py"\nreason = "A long enough reason."', "excerpt"),
        ('rule = "made-up"\nfile = "a.py"\nsymbol = "f"\nreason = "A long enough reason."', "rule"),
        ('rule = "empty-body"\nsymbol = "f"\nreason = "A long enough reason."', "no file"),
    ],
)
def test_an_incomplete_entry_is_reported(tmp_path: Path, entry: str, complaint: str) -> None:
    allow = write(tmp_path, "allow.toml", "[[allow]]\n" + entry + "\n")
    problems = scan.load_allowlist(allow).problems()
    assert problems and any(complaint in problem for problem in problems)


def test_a_missing_allowlist_is_an_empty_one(tmp_path: Path) -> None:
    assert scan.load_allowlist(tmp_path / "nothing.toml").entries == []


# ------------------------------------------------------------------- this repository


def test_the_repository_has_no_stand_ins() -> None:
    """The scan itself, run on the real tree with the real allowlist (CI runs it as well)."""
    root = SCRIPTS.parent
    allowlist = scan.load_allowlist(scan.DEFAULT_ALLOWLIST)
    findings, count = scan.scan_tree(root)
    remaining, allowed, stale = scan.apply_allowlist(findings, allowlist)
    assert count > 400
    assert remaining == [], "\n".join(f.render() for f in remaining)
    assert stale == [] and allowlist.problems() == []
    assert len(allowed) == len(allowlist.entries)


def test_every_allowlist_entry_gives_a_real_reason_in_full_sentences() -> None:
    for entry in scan.load_allowlist(scan.DEFAULT_ALLOWLIST).entries:
        assert len(entry.reason.split()) >= 12, entry
        assert entry.file.startswith("src/twin/") and Path(SCRIPTS.parent / entry.file).is_file()


def test_no_module_of_src_reaches_the_tests_from_the_command_line() -> None:
    root = SCRIPTS.parent
    findings, _ = scan.scan_tree(root)
    assert [f for f in findings if f.rule == "tests-import"] == []
