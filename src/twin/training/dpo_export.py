"""``twin train export-dpo``: the preference pairs as a DPO file for LLaMA-Factory (R-LRN-002/004).

This is the **only** reader of ``preference_pairs`` in the training package.  The SFT export
(:mod:`twin.training.export` and the modules it uses) never imports :mod:`twin.learning`; a test
reads their imports.  The ``rejected`` text of a pair is the bot's own reply, and it appears in
the DPO file as the dispreferred answer and nowhere else (R-LRN-004).

*What a line is.*  A pair is the structured prompt sample (the system segment and the turns,
opening with the user and alternating), the user's wording as ``chosen`` and the bot's reply as
``rejected``, written as ShareGPT with ``chosen`` / ``rejected`` (:class:`DpoSample`).  The sample
is handed over as it is: **no template string is ever written** - LLaMA-Factory puts the
``qwen3_nothink`` template around it once when it trains, and the dataset check refuses any text
that holds a ChatML marker.

*Locked versions.*  The adapter DPO continues from was trained with one persona card version and
one template version (the ones of its dataset).  A pair made with other versions - the live card of
the bot, say - gets its system segment made again with the locked ones (the compact pre-holdout
card of that version, the local time, her state and the memory as of the moment of the reply,
through ``AsOfView``), so the model is trained on prompts of the kind it saw.  A pair that cannot
be made again (the card version or the template cannot be rendered) is left out and counted.

*Desensitising.*  One ``ConsistentRedactor`` for the file, and a scan of what is left, like the
SFT set (R-TRN-007, R-PRIV-003): the file leaves the machine.

*The result* is a new dataset directory (a version names one content): the SFT files of the base
dataset unchanged plus ``dpo_train.jsonl``.  ``twin train bundle --dataset <it>`` packs it;
``dpo.sh`` runs on the instance when it holds at least ``training.dpo_min_pairs`` pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from twin.engine.style_backend import DEFAULT_BUDGET
from twin.engine.style_prompt import (
    LockedVersionError,
    StylePromptBuilder,
    StylePromptError,
    StyleTurn,
)
from twin.learning.pairs import PairError, PairRecord, PreferencePairStore, PromptSample
from twin.llm.redaction import ConsistentRedactor, redact
from twin.memory.asof import AsOfSource
from twin.profile.persona.render import count_tokens
from twin.training import lf_template
from twin.training.dataset_dir import (
    DatasetDir,
    DatasetError,
    DpoSample,
    derive_with_dpo,
    load_dataset_dir,
)
from twin.training.export import ExportError
from twin.training.export_text import event_lines
from twin.training.layout import FILE_DATASET_META
from twin.training.registry import LockedVersions
from twin.training.runs import ensure_dataset_version

if TYPE_CHECKING:
    from twin.services import Services

DATASETS_DIR = ("training", "datasets")
MAX_REPLY_TOKENS = 256  # the room the training cut-off leaves for an answer (style_backend)
SKIP_TEMPLATE = "template_unrenderable"
SKIP_PERSONA = "persona_version_missing"
SKIP_NO_PROMPT = "prompt_unusable"
SKIP_TOO_LONG = "reply_too_long"
SKIP_EVENT = "event_text_in_wording"
SKIP_SAME = "same_after_cleaning"

SystemMaker = Callable[[PairRecord], PromptSample]
"""Makes the prompt sample of a pair again for the locked versions (see the module text)."""


@dataclass
class DpoExportResult:
    """What an export did."""

    dataset: DatasetDir
    pairs: int
    regenerated: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    minimum: int = 0

    @property
    def enough(self) -> bool:
        return self.pairs >= self.minimum


def newest_dataset(services: Services) -> Path | None:
    """The directory of the newest exported dataset, or ``None``."""
    root = services.paths.data_dir.joinpath(*DATASETS_DIR)
    if not root.is_dir():
        return None
    found = sorted(
        (path for path in root.iterdir() if (path / FILE_DATASET_META).is_file()),
        key=lambda path: path.name,
    )
    return found[-1] if found else None


def locked_persona_matches(pair: PairRecord, locked: LockedVersions) -> bool:
    """Was the pair's system segment made with the card version the adapter is locked to?"""
    return pair.persona_version == f"pre_holdout:{locked.persona_version}"


def needs_new_system(pair: PairRecord, locked: LockedVersions) -> bool:
    """Does the pair's system segment differ in persona card or template from the locked ones?"""
    return pair.template_version != locked.template_version or not locked_persona_matches(
        pair, locked
    )


class LockedSystemMaker:
    """Makes the prompt sample of a pair again with the locked versions, as of its time."""

    def __init__(self, services: Services, locked: LockedVersions) -> None:
        self._services = services
        self._locked = locked
        self._builder: StylePromptBuilder | None = None
        self._source: AsOfSource | None = None

    def __call__(self, pair: PairRecord) -> PromptSample:
        if self._builder is None:  # refused here, per pair, when the template cannot be rendered
            self._builder = StylePromptBuilder.from_services(self._services, locked=self._locked)
        if self._source is None:
            self._source = AsOfSource(self._services)
        moment = pair.sample.at or pair.created_at
        turns = [StyleTurn("assistant", text) for text in pair.sample.prelude]
        turns.extend(StyleTurn(turn.role, turn.content) for turn in pair.sample.turns)
        parts = self._builder.compose(self._source.at(moment), turns, budget=DEFAULT_BUDGET)
        return PromptSample(parts.system, parts.turns, moment, pair.sample.prelude)


def _scan(texts: Sequence[str]) -> None:
    for text in texts:
        found = redact(text)
        if found.changed:
            kinds = ", ".join(sorted(found.counts()))
            raise ExportError(
                f"personal data ({kinds}) is left in a preference pair after desensitising"
            )


def export_dpo(
    services: Services,
    *,
    dataset: Path | None = None,
    out_dir: Path | None = None,
    system_maker: SystemMaker | None = None,
    now: datetime | None = None,
) -> DpoExportResult:
    """Write the preference pairs as ``dpo_train.jsonl`` of a new dataset directory.

    ``dataset`` is the exported SFT dataset the adapter is trained on (default: the newest);
    ``system_maker`` replaces how a system segment is made again (tests).
    """
    base_dir = dataset or newest_dataset(services)
    if base_dir is None:
        raise ExportError("there is no exported dataset yet; run `twin train export` first")
    try:
        base = load_dataset_dir(base_dir)
    except DatasetError as exc:
        raise ExportError(str(exc)) from exc
    pairs = PreferencePairStore(services.db, services.clock).all()
    if not pairs:
        raise ExportError("there are no preference pairs yet; they come from `/不像 <正确说法>`")
    locked = LockedVersions(
        template_version=base.meta.template_version,
        persona_version=base.meta.persona_version,
        profile_version=base.meta.profile_version,
        dataset_version=base.meta.dataset_version,
    )
    maker: SystemMaker = system_maker or LockedSystemMaker(services, locked)
    redactor = ConsistentRedactor()
    skipped: dict[str, int] = {}
    regenerated = 0
    samples: list[DpoSample] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for pair in pairs:
        sample = pair.sample
        if needs_new_system(pair, locked):
            try:
                sample = maker(pair)
            except LockedVersionError as exc:
                skip(SKIP_PERSONA if "persona" in str(exc) else SKIP_TEMPLATE)
                continue
            except (StylePromptError, PairError):
                skip(SKIP_NO_PROMPT)
                continue
            regenerated += 1
        chosen, rejected = redactor.redact_text(pair.chosen), redactor.redact_text(pair.rejected)
        if chosen == rejected:
            skip(SKIP_SAME)
            continue
        if max(count_tokens(chosen), count_tokens(rejected)) > MAX_REPLY_TOKENS:
            skip(SKIP_TOO_LONG)
            continue
        if event_lines(chosen.split("\n")):
            skip(SKIP_EVENT)
            continue
        turns = tuple(
            lf_template.Turn(turn.role, redactor.redact_text(turn.content)) for turn in sample.turns
        )
        samples.append(
            DpoSample(pair.id, redactor.redact_text(sample.system), turns, chosen, rejected)
        )
    if not samples:
        raise ExportError(f"none of the {len(pairs)} preference pairs could be exported: {skipped}")
    for written in samples:
        _scan(
            [
                written.system,
                *(turn.content for turn in written.turns),
                written.chosen,
                written.rejected,
            ]
        )
    moment = now or services.clock.now_utc()
    version = f"{base.meta.dataset_version}-dpo{moment:%Y%m%d%H%M%S}"
    directory = out_dir or base_dir.parent / version
    stats = {
        "dpo": {
            "pairs_in_store": len(pairs),
            "exported": len(samples),
            "regenerated_systems": regenerated,
            "skipped": dict(skipped),
            "base_dataset": base.meta.dataset_version,
            "redacted_entities": redactor.counts(),
        }
    }
    try:
        result = derive_with_dpo(
            base,
            directory,
            samples,
            dataset_version=version,
            created_at=moment.isoformat(),
            stats=stats,
        )
    except DatasetError as exc:
        raise ExportError(str(exc)) from exc
    try:
        relative = result.path.resolve().relative_to(services.paths.data_dir.resolve()).as_posix()
    except ValueError:
        relative = str(result.path.resolve())
    ensure_dataset_version(services.db, result, relative)
    return DpoExportResult(
        result,
        len(samples),
        regenerated,
        skipped,
        services.settings.training.dpo_min_pairs,
    )
