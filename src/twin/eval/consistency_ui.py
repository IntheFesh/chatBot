"""The terminal screen of the consistency audit: decide each contradiction, then its corrections.

The user is shown the contradictions the audit found, one at a time, and says whether it is a real
one (``y`` an obvious one, ``m`` a real but not obvious one, ``n`` no contradiction at all, ``s``
decide later, ``q`` save and leave).  For each contradiction he confirms, the corrections of the
memory it suggests (:mod:`twin.eval.consistency_fixes`) are shown one by one and each is applied
only if he presses ``y`` for it; ``n`` leaves the memory as it is and ``s`` keeps the proposal for
later.  Every decision is written the moment it is made, so a review that stops anywhere is
continued with ``twin eval consistency --review``.

Input and output are injected like those of the blind test (:mod:`twin.eval.ui`): keys come from a
:class:`~twin.eval.ui.KeySource`, output goes to a rich console, everything shown is a
:class:`~rich.text.Text` (a reply with brackets in it is never taken for markup).
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from twin.eval.consistency_fixes import ACTION_LABELS, FixApplier, propose_fixes
from twin.eval.consistency_model import KIND_LABELS, Statement
from twin.eval.consistency_store import ConsistencyStore, FindingView, FixView
from twin.eval.store import RunView
from twin.eval.ui import KEY_QUIT, KEY_SKIP, KeySource, ask
from twin.memory.api import Memory
from twin.memory.conflict import SOURCE_NAMES

KEY_OBVIOUS = "y"
KEY_MINOR = "m"
KEY_NOT = "n"
KEYS_FINDING = (KEY_OBVIOUS, KEY_MINOR, KEY_NOT, KEY_SKIP, KEY_QUIT)
KEY_APPLY = "y"
KEY_KEEP = "n"
KEYS_FIX = (KEY_APPLY, KEY_KEEP, KEY_SKIP, KEY_QUIT)
SEVERITY_NAMES = {"obvious": "明显", "minor": "不明显"}


@dataclass
class ReviewOutcome:
    """What one session decided."""

    obvious: int = 0
    minor: int = 0
    rejected: int = 0
    skipped: int = 0
    applied: int = 0
    declined: int = 0
    stale: int = 0
    quit: bool = False


def describe(statement: Statement, zone: ZoneInfo) -> Text:
    """A statement as the user recognises it: where it comes from, when, and what it says."""
    label = KIND_LABELS[statement.kind]
    if statement.kind == "fact":
        source = SOURCE_NAMES.get(statement.source or "", "未知来源")
        number = f" #{statement.number}" if statement.number is not None else ""
        known = (statement.known_at or statement.at).astimezone(zone)
        head = f"{label}{number}（{source}；{known:%Y-%m-%d} 知道）"
    elif statement.kind == "reply":
        head = f"{label}（{statement.at.astimezone(zone):%m-%d %H:%M}）"
    else:
        head = label
    return Text(f"{head}：{statement.text}")


class ConsistencyReview:
    """Walks through the undecided contradictions and the proposed corrections of a run."""

    def __init__(
        self,
        store: ConsistencyStore,
        applier: FixApplier,
        memory: Memory,
        run: RunView,
        console: Console,
        keys: KeySource,
    ) -> None:
        self._store = store
        self._applier = applier
        self._memory = memory
        self._run = run
        self._console = console
        self._keys = keys
        self._zone = ZoneInfo(str(run.params.get("zone", "UTC")))

    def waiting(self) -> tuple[int, int]:
        """``(contradictions undecided, corrections proposed)`` of the run."""
        undecided = len(self._store.findings(self._run.id, status="proposed"))
        proposed = len(self._store.run_fixes(self._run.id, status="proposed"))
        return undecided, proposed

    def run(self) -> ReviewOutcome:
        outcome = ReviewOutcome()
        findings = self._store.findings(self._run.id)
        total = sum(f.status == "proposed" for f in findings)
        position = 0
        for finding in findings:
            current = finding
            if finding.status == "proposed":
                position += 1
                decided = self._decide(finding, position, total, outcome)
                if decided is None:
                    return outcome
                current = decided
            if current.status == "confirmed" and not self._fixes_of(current, outcome):
                return outcome
        return outcome

    # --------------------------------------------------------------- a contradiction

    def _show(self, finding: FindingView, position: int, total: int) -> None:
        shown: list[Text] = [
            describe(finding.first, self._zone),
            describe(finding.second, self._zone),
        ]
        for extra in finding.related:
            shown.append(Text("相关：") + describe(extra, self._zone))
        shown.append(Text(f"理由：{finding.reason}", style="dim"))
        title = (
            f"第 {position}/{total} 条  {finding.time_text}  "
            f"模型认为{SEVERITY_NAMES[finding.model_severity]}"
        )
        self._console.print(Panel(Group(*shown), title=title))

    def _decide(
        self, finding: FindingView, position: int, total: int, outcome: ReviewOutcome
    ) -> FindingView | None:
        """Ask about one contradiction; ``None`` when the user leaves.  A skipped one is returned
        as it is, still undecided."""
        self._show(finding, position, total)
        key = ask(
            self._keys,
            self._console,
            "这是真矛盾吗？  [y] 是，明显  [m] 是，但不明显  [n] 不是  [s] 先跳过  [q] 保存退出",
            KEYS_FINDING,
        )
        if key is None or key == KEY_QUIT:
            outcome.quit = True
            return None
        if key == KEY_SKIP:
            outcome.skipped += 1
            return finding
        if key == KEY_NOT:
            outcome.rejected += 1
            return self._store.decide(finding.id, status="rejected")
        severity = "obvious" if key == KEY_OBVIOUS else "minor"
        if severity == "obvious":
            outcome.obvious += 1
        else:
            outcome.minor += 1
        return self._store.decide(finding.id, status="confirmed", severity=severity)

    # ----------------------------------------------------------------- corrections

    def _fixes_of(self, finding: FindingView, outcome: ReviewOutcome) -> bool:
        """Propose (once) and go through the corrections of a confirmed contradiction.

        Returns ``False`` when the user leaves.  A contradiction carried over from an earlier
        decision has its corrections in the run where it was decided.
        """
        if finding.inherited_from is None and not self._store.fixes(finding.id):
            proposals = propose_fixes(finding, self._memory)
            if proposals:
                self._store.add_fixes(finding.id, proposals)
        pending = self._store.fixes(finding.id, status="proposed")
        for number, fix in enumerate(pending, start=1):
            if not self._ask_fix(fix, number, len(pending), outcome):
                return False
        return True

    def _ask_fix(self, fix: FixView, number: int, total: int, outcome: ReviewOutcome) -> bool:
        lines = [Text(f"旧：{fix.old_text}")]
        if fix.new_text:
            lines.append(Text(f"新：{fix.new_text}"))
        title = f"修正建议 {number}/{total}：{ACTION_LABELS[fix.action]}"
        self._console.print(Panel(Group(*lines), title=title, border_style="yellow"))
        key = ask(
            self._keys,
            self._console,
            "改动线上记忆？  [y] 应用  [n] 不改  [s] 稍后再决定  [q] 保存退出",
            KEYS_FIX,
        )
        if key is None or key == KEY_QUIT:
            outcome.quit = True
            return False
        if key == KEY_SKIP:
            return True
        if key == KEY_KEEP:
            self._applier.decline(fix.id)
            outcome.declined += 1
            return True
        result = self._applier.apply(fix.id)
        if result.status == "applied":
            outcome.applied += 1
            self._console.print(Text("已应用。", style="green"))
        else:
            outcome.stale += 1
            self._console.print(Text("这条记忆在建议之后已经变了，没有改动。", style="yellow"))
        return True
