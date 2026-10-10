"""The training-set export (R-TRN-002 to R-TRN-007, R-TRN-013).

``twin train export`` turns the real conversation into ShareGPT samples for LLaMA-Factory: every
reply block of hers is one sample, the target is what she wrote, the input is what the style
model would have been given at that moment.

*One prompt, one builder.*  The system message and the conversation come from the same
:class:`~twin.engine.style_prompt.StylePromptBuilder` the running bot uses, fed with
``AsOfView(t)`` of the moment ``t`` of the first message of the block: the pre-holdout compact
persona card, the local time and weekday, her state at that time according to the pre-holdout
routine, the memory block as it was known then.  Nothing after ``t`` can reach a prompt
(R-TRN-013).  The messages of the conversation itself are read through
:mod:`twin.training.export_blocks`, i.e. through :mod:`twin.ingest.corpus` and only before ``t``.

*What becomes a sample.*  A block whose target has nothing left after the lines the bot could not
send are deleted, a block that does not follow a turn of the user (it opened the conversation, or
she wrote twice in a row) and a block that does not fit 2,048 tokens even after all older context
is dropped are not samples; each reason is counted.  A sample that is too long loses its oldest
context, never its target: LLaMA-Factory would otherwise cut the end of the prompt, the assistant
opener included (``training/README.md``).

*Splits.*  By time: the samples from :func:`~twin.profile.holdout.holdout_cutoff` on are the test
set, the last 5 % of the rest the validation set, the earlier ones train.

*Desensitising.*  One :class:`~twin.llm.redaction.ConsistentRedactor` for the whole set: the same
person, phone number or address is the same token everywhere.  Every sample is scanned for
what is left (personal data, event text in a target) before the set is declared desensitised, and
the written files are scanned again.

*Plans.*  ``training.hybrid_plan_ratio`` of the training and validation samples carry a plan
(:mod:`twin.training.plans`).  When some of them have none yet, the export stops after a first
pass, hands the missing inputs back, and writes nothing; once the plans exist the same command
completes.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from twin.engine.prompt import STATE_LINES, moment_text
from twin.engine.style_prompt import (
    PlanFields,
    StylePromptBuilder,
    StylePromptError,
    StyleTurn,
    TokenBudget,
    normalize_context,
)
from twin.ingest.corpus import her_message_count
from twin.llm.redaction import ConsistentRedactor, redact, redact_text
from twin.memory.asof import AsOfSource
from twin.profile.holdout import HoldoutError, holdout_cutoff
from twin.profile.persona.render import RenderedPersona
from twin.stickers.catalog import StickerCatalog
from twin.training import lf_template
from twin.training.dataset_dir import DatasetDir, ParityCase, SftSample, write_dataset_dir
from twin.training.export_blocks import (
    BlockRef,
    LoadedBlock,
    MessageLoader,
    iter_block_refs,
)
from twin.training.export_stats import SampleFacts, StatsCollector
from twin.training.export_text import Target, context_text, event_lines, render_target
from twin.training.parity_cases import (
    DEEP_TURNS,
    MULTILINE_LINES,
    TAG_CODE,
    TAG_DEEP,
    TAG_ENTITY,
    TAG_MULTILINE,
    TAG_PLAN,
    TAG_PRELUDE,
    TAG_QUOTE,
    TAG_SINGLE,
    TAG_STICKER,
    TAG_TEST,
    TAG_TRIMMED,
    ParitySelector,
)
from twin.training.plans import (
    DONE,
    PlanEntry,
    PlanInput,
    PlanStore,
    fact_lines,
    needs_plan,
    plan_selected,
    target_hash,
)
from twin.training.profiles import CUTOFF_LEN
from twin.training.tokenizer import TextTokenizer

if TYPE_CHECKING:
    from twin.services import Services

VAL_RATIO = 0.05
MIN_PROMPT_TOKENS = 64
BATCH_BLOCKS = 256
SCOPE = "pre_holdout"
WORK_DIR_PREFIX = ".export-"

DROP_NO_REPRODUCIBLE = "no_reproducible_line"
DROP_EMPTY_TARGET = "empty_target"
DROP_NO_USER_TURN = "no_user_turn_to_answer"
DROP_TARGET_TOO_LONG = "target_too_long"
DROP_OVER_BUDGET = "over_budget"
DROP_TEMPLATE = "template_unsafe"
DROP_MISSING_MESSAGE = "message_missing"

ExportState = Literal["done", "waiting_for_plans"]


class ExportError(RuntimeError):
    """The training set cannot be exported (the message names what to do, never message text)."""


@dataclass(frozen=True)
class ExportOptions:
    """What to export: a time range (default: everything), where to write, the plan share."""

    since: datetime | None = None
    until: datetime | None = None
    out_dir: Path | None = None
    plan_ratio: float | None = None
    batch_blocks: int = BATCH_BLOCKS


@dataclass
class ExportResult:
    """The outcome of :meth:`TrainingSetExporter.run`."""

    state: ExportState
    dataset: DatasetDir | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    missing_plans: dict[str, PlanInput] = field(default_factory=dict)
    plan_selected: int = 0


class ExportPromptBuilder(StylePromptBuilder):
    """The style prompt builder with the persona card desensitised like everything else."""

    def __init__(self, redactor: ConsistentRedactor, *, memory_tokens: int) -> None:
        super().__init__(memory_tokens=memory_tokens)
        self._redactor = redactor
        self._cards: dict[str, RenderedPersona] = {}

    def persona(self, data: Any) -> RenderedPersona | None:
        card = super().persona(data)
        if card is None:
            return None
        known = self._cards.get(card.version_id)
        if known is None:
            known = replace(card, text=self._redactor.redact_text(card.text))
            self._cards[card.version_id] = known
        return known


@dataclass
class _Prepared:
    """A reply block that can become a sample: its target and the conversation before it."""

    block: LoadedBlock
    target: Target
    turns: list[StyleTurn]
    sha: str
    pre_holdout: bool
    selected: bool

    @property
    def sample_id(self) -> str:
        return self.block.sample_id

    @property
    def at(self) -> datetime:
        return self.block.at


def val_count(pre_holdout_samples: int) -> int:
    """How many of the samples before the cut-off are validation: the latest 5 % (at least one)."""
    return max(1, round(pre_holdout_samples * VAL_RATIO))


class TrainingSetExporter:
    """Builds the dataset directory (see the module description)."""

    def __init__(
        self,
        services: Services,
        tokenizer: TextTokenizer,
        options: ExportOptions | None = None,
    ) -> None:
        self._services = services
        self._tokenizer = tokenizer
        self._options = options or ExportOptions()
        try:
            self._cutoff = holdout_cutoff(services)
        except HoldoutError as exc:
            raise ExportError(f"the hold-out cannot be determined yet: {exc}") from exc
        self._source = AsOfSource(services)
        self._check_derived_data()
        settings = services.settings
        self._ratio = (
            self._options.plan_ratio
            if self._options.plan_ratio is not None
            else settings.training.hybrid_plan_ratio
        )
        self._memory_tokens = settings.style_model.memory_tokens
        self._labeler = StickerCatalog(services).tag_lookup()
        self._loader = MessageLoader(services)
        self._memory_builder = StylePromptBuilder(memory_tokens=self._memory_tokens)

    # ----------------------------------------------------------------------- checks

    def _check_derived_data(self) -> None:
        if self._source.compact is None:
            raise ExportError(
                "there is no pre-holdout persona card; run `twin persona generate` first"
            )
        if self._source.profile is None or self._source.activity is None:
            raise ExportError(
                "there is no pre-holdout profile or routine; run `twin profile rebuild` first"
            )

    def _persona_card(self) -> RenderedPersona:
        card = self._source.compact
        if card is None:
            raise ExportError("there is no pre-holdout persona card; run `twin persona generate`")
        return card

    @property
    def cutoff(self) -> datetime:
        return self._cutoff

    # --------------------------------------------------------------------- preparing

    def _prepare(self, block: LoadedBlock, stats: StatsCollector) -> _Prepared | None:
        target = render_target([m.data for m in block.reply], self._labeler)
        stats.target_dropped.update(target.dropped)
        if not target.has_content:
            stats.drop(DROP_EMPTY_TARGET)
            return None
        turns: list[StyleTurn] = []
        for turn in block.context:
            text = context_text([m.data for m in turn], self._labeler)
            if text:
                turns.append(StyleTurn("user" if turn[0].data.is_sent else "assistant", text))
        context = normalize_context(turns)
        if not context.turns or not context.ends_with_user:
            stats.drop(DROP_NO_USER_TURN)
            return None
        pre_holdout = block.at < self._cutoff
        return _Prepared(
            block=block,
            target=target,
            turns=turns,
            sha=target_hash(target.text),
            pre_holdout=pre_holdout,
            selected=pre_holdout and plan_selected(block.sample_id, self._ratio),
        )

    def _prepared(self, stats: StatsCollector) -> Iterator[_Prepared]:
        options = self._options
        refs: list[BlockRef] = []

        def flush() -> Iterator[_Prepared]:
            try:
                blocks = self._loader.load(refs)
            except LookupError:
                stats.drop(DROP_MISSING_MESSAGE)
                blocks = []
            refs.clear()
            for block in blocks:
                prepared = self._prepare(block, stats)
                if prepared is not None:
                    yield prepared

        for ref in iter_block_refs(self._services, since=options.since, until=options.until):
            if ref.reproducible < 1:
                stats.drop(DROP_NO_REPRODUCIBLE)
                continue
            refs.append(ref)
            if len(refs) >= options.batch_blocks:
                yield from flush()
        yield from flush()

    # ------------------------------------------------------------------ first pass

    def scan(self) -> tuple[dict[str, PlanInput], int]:
        """The samples that carry a plan and have none: ``({id: inputs}, number selected)``."""
        entries = PlanStore(self._services.db).entries()
        stats = StatsCollector()
        missing: dict[str, PlanInput] = {}
        selected = 0
        for item in self._prepared(stats):
            if not item.selected:
                continue
            selected += 1
            if needs_plan(entries.get(item.sample_id), item.sha):
                missing[item.sample_id] = self._plan_input(item)
        return missing, selected

    def _plan_input(self, item: _Prepared) -> PlanInput:
        """What the plan of a sample is written from; all of it desensitised (it leaves)."""
        view = self._source.at(item.at)
        memory = self._memory_builder.memory_for(view, item.turns)
        context = normalize_context(item.turns)
        turns = [("assistant", redact_text(text)) for text in context.prelude]
        turns += [(turn.role, redact_text(turn.content)) for turn in context.turns]
        state = view.her_state()
        when = moment_text(view.local)
        if state in STATE_LINES:
            when += f"；她那时的状态：{STATE_LINES[state]}"
        return PlanInput(
            target_sha=item.sha,
            when=redact_text(when),
            turns=tuple(turns),
            reply=redact_text(item.target.text),
            facts=tuple(redact_text(line) for line in fact_lines(memory.text)),
        )

    # ----------------------------------------------------------------- second pass

    def run(self) -> ExportResult:
        """Export the set, or report which plans are missing (see the module description)."""
        if self._ratio <= 0:
            return self.build()
        missing, selected = self.scan()
        if missing:
            return ExportResult("waiting_for_plans", missing_plans=missing, plan_selected=selected)
        return self.build()

    def build(self) -> ExportResult:
        """The second pass: compose, desensitise, split, scan and write the dataset directory."""
        services = self._services
        now = services.clock.now_utc()
        version = f"ds-{now:%Y%m%d-%H%M%S}"
        directory = self._options.out_dir or (
            services.paths.data_dir / "training" / "datasets" / version
        )
        if directory.exists() and any(directory.iterdir()):
            raise ExportError(f"{directory} is not empty; export to a new directory")
        work = directory.parent / f"{WORK_DIR_PREFIX}{version}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            return self._build(version, directory, work, now)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _build(self, version: str, directory: Path, work: Path, now: datetime) -> ExportResult:
        redactor = ConsistentRedactor()
        builder = ExportPromptBuilder(redactor, memory_tokens=self._memory_tokens)
        entries = PlanStore(self._services.db).entries()
        stats = StatsCollector()
        parity = ParitySelector()
        pre_path, test_path = work / "pre.jsonl", work / "test.jsonl"
        with (
            pre_path.open("w", encoding="utf-8", newline="\n") as pre_file,
            test_path.open("w", encoding="utf-8", newline="\n") as test_file,
        ):
            for item in self._prepared(stats):
                sample = self._compose(item, builder, redactor, entries, stats, parity)
                if sample is not None:
                    row = json.dumps(sample.to_json(), ensure_ascii=False, separators=(",", ":"))
                    (pre_file if item.pre_holdout else test_file).write(row + "\n")
        pre_count, test_count = len(stats.pre), len(stats.test)
        validation = val_count(pre_count)
        if pre_count - validation < 1 or pre_count < 2 or test_count < 1:
            raise ExportError(
                f"too few samples for a training set: {pre_count} before the hold-out, "
                f"{test_count} after it ({sum(stats.dropped.values())} blocks dropped: "
                f"{dict(stats.dropped)})"
            )
        profile = self._source.profile
        if profile is None:
            raise ExportError("there is no pre-holdout profile; run `twin profile rebuild` first")
        summary = stats.summarise(
            val_count=validation,
            profile_sticker_share=profile.metrics.scalar("her", "sticker_share"),
            profile_code_rate=profile.metrics.scalar("her", "emoji_code_rate"),
        )
        summary.update(self._facts_about_the_run(redactor, parity))
        card = self._persona_card()
        split = pre_count - validation
        # one generator per file range, closed here: an open file cannot be deleted on Windows
        streams = [
            samples_from(pre_path, 0, split, scan=True),
            samples_from(pre_path, split, pre_count, scan=True),
            samples_from(test_path, 0, test_count, scan=True),
        ]
        try:
            dataset = write_dataset_dir(
                directory,
                dataset_version=version,
                created_at=now.isoformat(),
                template_version=lf_template.TEMPLATE_VERSION,
                persona_version=f"v{card.number}",
                profile_version=profile.version.id,
                holdout_cutoff=self._cutoff.isoformat(),
                train=streams[0],
                val=streams[1],
                test=streams[2],
                parity=parity.cases(),
                redacted=True,
                plan_ratio=self._ratio,
                range_from=self._options.since.isoformat() if self._options.since else None,
                range_to=self._options.until.isoformat() if self._options.until else None,
                stats=summary,
            )
        finally:
            for stream in streams:
                stream.close()
        verify_written(dataset)
        return ExportResult("done", dataset=dataset, stats=summary)

    def _facts_about_the_run(
        self, redactor: ConsistentRedactor, parity: ParitySelector
    ) -> dict[str, Any]:
        options = self._options
        with self._services.db.session() as session:
            until = int(session.scalar(her_message_count(options.until)) or 0)
            before = (
                int(session.scalar(her_message_count(options.since)) or 0) if options.since else 0
            )
        card = self._persona_card()
        return {
            "her_messages_covered": until - before,
            "tokenizer_sha256": getattr(self._tokenizer, "sha256", None),
            "cutoff_len": CUTOFF_LEN,
            "context_turns": 8,
            "persona_scope": SCOPE,
            "persona_card": card.version_id,
            "redacted_entities": redactor.counts(),
            "parity_cases": len(parity.cases()),
        }

    # -------------------------------------------------------------------- composing

    def _compose(
        self,
        item: _Prepared,
        builder: ExportPromptBuilder,
        redactor: ConsistentRedactor,
        entries: dict[str, PlanEntry],
        stats: StatsCollector,
        parity: ParitySelector,
    ) -> SftSample | None:
        entity = False

        def red(text: str) -> str:
            nonlocal entity
            result = redactor.redact(text)
            entity = entity or result.changed
            return result.text

        reply = red(item.target.text)
        try:
            response_tokens = self._tokenizer.count(lf_template.render_response(reply))
        except lf_template.TemplateError:
            stats.drop(DROP_TEMPLATE)
            return None
        room = CUTOFF_LEN - response_tokens
        if room < MIN_PROMPT_TOKENS:
            stats.drop(DROP_TARGET_TOO_LONG)
            return None
        view = self._source.at(item.at)
        memory = builder.memory_for(view, item.turns)
        plan = self._plan_fields(item, entries, fact_lines(memory.text), red)
        try:
            parts = builder.compose(
                view,
                [StyleTurn(turn.role, red(turn.text)) for turn in item.turns],
                plan=plan,
                memory=replace(memory, text=red(memory.text)),
                budget=TokenBudget(room, self._tokenizer.count),
            )
        except StylePromptError:
            stats.drop(DROP_NO_USER_TURN)
            return None
        except lf_template.TemplateError:
            stats.drop(DROP_TEMPLATE)
            return None
        if parts.meta.over_budget:
            stats.drop(DROP_OVER_BUDGET)
            return None
        prompt = parts.render().text
        tokens = self._tokenizer.count(prompt) + response_tokens
        sample = SftSample(item.sample_id, parts.system, parts.turns, reply)
        target = item.target
        stats.add(
            SampleFacts(
                pre_holdout=item.pre_holdout,
                planned=plan is not None,
                plan_selected=item.selected,
                turns=len(parts.turns),
                prelude=parts.meta.prelude_turns > 0,
                trimmed=parts.meta.trimmed_turns > 0,
                target_lines=len(target.lines),
                target_chars=len(reply),
                stickers=target.stickers,
                text_lines=target.text_lines,
                code_lines=target.code_lines,
                quoted=target.quoted,
                tokens=tokens,
            )
        )
        tags = self._tags(item, parts.meta.prelude_turns, parts.meta.trimmed_turns, parts.turns)
        if plan is not None:
            tags.append(TAG_PLAN)
        if entity:
            tags.append(TAG_ENTITY)
        if parity.wants(tags, tokens):
            parity.offer(
                lambda found: ParityCase(
                    item.sample_id, parts.system, parts.turns, reply, prompt, found
                ),
                tags,
                tokens,
            )
        return sample

    @staticmethod
    def _tags(
        item: _Prepared, prelude: int, trimmed: int, turns: tuple[lf_template.Turn, ...]
    ) -> list[str]:
        target = item.target
        tags: list[str] = []
        flags = (
            (prelude > 0, TAG_PRELUDE),
            (trimmed > 0, TAG_TRIMMED),
            (len(target.lines) >= MULTILINE_LINES, TAG_MULTILINE),
            (target.stickers > 0, TAG_STICKER),
            (target.code_lines > 0, TAG_CODE),
            (target.quoted, TAG_QUOTE),
            (len(turns) == 1, TAG_SINGLE),
            (len(turns) >= DEEP_TURNS, TAG_DEEP),
            (not item.pre_holdout, TAG_TEST),
        )
        tags.extend(tag for present, tag in flags if present)
        return tags

    @staticmethod
    def _plan_fields(
        item: _Prepared,
        entries: dict[str, PlanEntry],
        facts: list[str],
        red: Callable[[str], str],
    ) -> PlanFields | None:
        if not item.selected:
            return None
        entry = entries.get(item.sample_id)
        if entry is None or entry.status != DONE or entry.plan is None:
            return None
        if entry.target_sha != item.sha:
            return None
        fields = entry.plan.fields(facts)
        plan = PlanFields(
            intent=red(fields.intent),
            facts_to_use=tuple(red(fact) for fact in fields.facts_to_use),
            tone=red(fields.tone),
            bubble_hint=red(fields.bubble_hint),
        )
        return None if plan.empty else plan


# ------------------------------------------------------------------- files and scans


def samples_from(path: Path, start: int, stop: int, *, scan: bool = False) -> Generator[SftSample]:
    """The samples on lines ``start`` to ``stop - 1`` of a file, scanned on the way if asked."""
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream):
            if number >= stop:
                return
            if number >= start:
                sample = sample_of(json.loads(line))
                if scan:
                    scan_sample(sample)
                yield sample


def sample_of(row: dict[str, Any]) -> SftSample:
    """An :class:`SftSample` from its ShareGPT line."""
    messages = row["conversations"]
    turns = tuple(
        lf_template.Turn("user" if index % 2 == 0 else "assistant", str(message["value"]))
        for index, message in enumerate(messages[:-1])
    )
    return SftSample(str(row["id"]), str(row["system"]), turns, str(messages[-1]["value"]))


def scan_sample(sample: SftSample) -> None:
    """Fail when personal data is left in a sample or a target holds event text (R-TRN-007)."""
    texts = [sample.system, *(turn.content for turn in sample.turns), sample.reply]
    for text in texts:
        found = redact(text)
        if found.changed:
            kinds = ", ".join(sorted(found.counts()))
            raise ExportError(
                f"personal data ({kinds}) is left in sample {sample.id} after desensitising"
            )
    events = event_lines(sample.reply.split("\n"))
    if events:
        raise ExportError(f"sample {sample.id} has {len(events)} event text line(s) in its target")


def scanned(samples: Iterator[SftSample]) -> Iterator[SftSample]:
    """``samples`` unchanged, each scanned on the way (a finding raises :class:`ExportError`)."""
    for sample in samples:
        scan_sample(sample)
        yield sample


def verify_written(dataset: DatasetDir) -> None:
    """Scan the files that were written once more (the set must not be used otherwise)."""
    for name in dataset.data_files():
        if name.startswith("sft_"):
            path = dataset.file_path(name)
            for sample in samples_from(path, 0, 1 << 62):
                scan_sample(sample)
