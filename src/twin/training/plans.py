"""Plans for the hybrid share of the training set (R-TRN-005, R-LLM-014).

The hybrid backend asks DeepSeek for a plan (``intent``, ``facts_to_use``, ``tone``,
``bubble_hint``) and hands it to the style model in the ``【规划】`` field of its prompt.  For the
model to learn to follow such a plan, ``training.hybrid_plan_ratio`` (30 %) of the training
samples carry one, written the other way round: DeepSeek reads her real reply and works out what
she was probably thinking.

*Which samples.*  :func:`plan_selected` decides from the hash of the sample id alone, so the
choice does not change when the data grows and a plan once written is found by every later
export.  Only samples of the training and validation parts are chosen: a plan is a summary of the
answer, and a test sample with one would no longer measure how she answers.

*Where the plan comes from.*  A one-time batch (R-LLM-014).  The export registers the samples that
lack a plan (:meth:`PlanStore.register`), the batches are priced and queued waiting for
``twin jobs approve <batch>`` (:func:`queue_plan_batches`), the plan jobs write one row at a time
(:func:`handle_training_plan`; a stopped job continues where it was), and the next export finds
the plans in ``training_plans``.  Everything sent to DeepSeek is desensitised first.

*Facts.*  The model is given the fact lines of the sample's as-of memory block, numbered, and
names the numbers she used; it cannot introduce a fact that is not there.  An export checks the
stored lines against the block of the sample once more (:func:`checked_facts`) and drops the ones
that are no longer in it, so a plan never names a fact the prompt does not show.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from twin.engine.refusal import FILTER_FINISHES
from twin.engine.style_prompt import PlanFields
from twin.llm.errors import (
    BudgetDeniedError,
    CircuitOpenError,
    InvalidRequestError,
    StructuredOutputError,
)
from twin.llm.redaction import redact_text
from twin.llm.runtime import LlmRuntime, build_llm_runtime
from twin.llm.types import DAILY, ChatMessage, LedgerTag, Purpose
from twin.memory.render import SECTION_FACTS, SECTION_TODAY, heading
from twin.ops.jobs import BatchTooLargeError, JobContext, JobDeferred, JobQueue, job_handler
from twin.ops.logging import get_logger
from twin.profile.prompt_templates import TRAIN_PLAN, PromptText, TemplateStore
from twin.storage.db import Database
from twin.storage.ids import new_id
from twin.storage.training_plan_models import TrainingPlan

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.training.plans")

PLAN_JOB = "training_plan"
PLAN_PRIORITY = 135
PLAN_JOB_SIZE = 20
PLAN_COMPLETION_TOKENS = 220
PLAN_MAX_TOKENS = 400
PLAN_TEMPERATURE = 0.3
MAX_FACTS = 4
FIELD_CHARS = 80
MIN_ECHO_CHARS = 6
MIN_FACT_CHARS = 2
PENDING, DONE, REFUSED = "pending", "done", "refused"

FACT_SECTIONS = (SECTION_TODAY, SECTION_FACTS)
_HEADINGS = tuple(heading(section) for section in FACT_SECTIONS)
_BULLET = re.compile(r"^\s*(?:[-*•·]\s+)")
_SPACES = re.compile(r"\s+")


# --------------------------------------------------------------------------- selection


def plan_selected(sample_id: str, ratio: float) -> bool:
    """Does this sample carry a plan?  Decided by the id and the plan ratio alone."""
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    digest = hashlib.sha256(f"plan\x1f{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64) < ratio


def target_hash(target_text: str) -> str:
    """The fingerprint of the reply a plan was written for (a changed reply makes it stale)."""
    return hashlib.sha256(target_text.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------- fact lines


def normalize_line(text: str) -> str:
    """A fact line without bullet, width and spacing differences, for comparing lines."""
    return _SPACES.sub(" ", _BULLET.sub("", unicodedata.normalize("NFKC", text))).strip()


def fact_lines(memory_text: str) -> list[str]:
    """The lines of the memory block a plan may name as facts.

    That is what the block writes under ``【今天要留意的】`` and ``【相关的事】``; the summaries of
    earlier days are context, not facts.
    """
    lines: list[str] = []
    inside = False
    for raw in memory_text.split("\n"):
        line = raw.strip()
        if line.startswith("【"):
            inside = line in _HEADINGS
        elif inside and line:
            lines.append(_BULLET.sub("", line))
    return lines


def checked_facts(stored: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    """The lines of ``available`` that ``stored`` names; a stored line that is not there is dropped.

    Both sides are compared after desensitisation (what was stored is what DeepSeek read), and the
    line returned is the one of ``available`` - the memory block of the sample as it is now.
    """
    index: dict[str, str] = {}
    for line in available:
        index.setdefault(normalize_line(redact_text(line)), line)
    chosen: list[str] = []
    for fact in stored:
        match = index.get(normalize_line(fact))
        if match is not None and match not in chosen:
            chosen.append(match)
    return tuple(chosen[:MAX_FACTS])


# ------------------------------------------------------------------------------- data


@dataclass(frozen=True)
class PlanInput:
    """What a plan is written from; every text is desensitised (it is sent to DeepSeek)."""

    target_sha: str
    when: str
    turns: tuple[tuple[str, str], ...]  # (role, text): "user" or "assistant", oldest first
    reply: str
    facts: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "target_sha": self.target_sha,
            "when": self.when,
            "turns": [list(turn) for turn in self.turns],
            "reply": self.reply,
            "facts": list(self.facts),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PlanInput:
        return cls(
            target_sha=str(data["target_sha"]),
            when=str(data["when"]),
            turns=tuple((str(role), str(text)) for role, text in data["turns"]),
            reply=str(data["reply"]),
            facts=tuple(str(fact) for fact in data["facts"]),
        )


@dataclass(frozen=True)
class StoredPlan:
    """A plan as it is stored; ``facts_to_use`` are desensitised lines of the memory block."""

    intent: str
    facts_to_use: tuple[str, ...] = ()
    tone: str = ""
    bubble_hint: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "facts_to_use": list(self.facts_to_use),
            "tone": self.tone,
            "bubble_hint": self.bubble_hint,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> StoredPlan:
        return cls(
            intent=str(data.get("intent", "")),
            facts_to_use=tuple(str(fact) for fact in data.get("facts_to_use", [])),
            tone=str(data.get("tone", "")),
            bubble_hint=str(data.get("bubble_hint", "")),
        )

    def fields(self, available_facts: Sequence[str]) -> PlanFields:
        """The plan fields of the prompt, with the facts re-checked against the sample's block."""
        return PlanFields(
            intent=self.intent,
            facts_to_use=checked_facts(self.facts_to_use, available_facts),
            tone=self.tone,
            bubble_hint=self.bubble_hint,
        )


class PlanDraft(BaseModel):
    """The JSON DeepSeek answers with (validated, R-LLM-003); other fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    intent: str = ""
    fact_numbers: list[int] = Field(default_factory=list)
    tone: str = ""
    bubble_hint: str = ""

    @field_validator("intent", "tone", "bubble_hint", mode="before")
    @classmethod
    def _text(cls, value: object) -> str:
        return "" if value is None else _SPACES.sub(" ", str(value)).strip()

    @field_validator("fact_numbers", mode="before")
    @classmethod
    def _numbers(cls, value: object) -> list[int]:
        items = value if isinstance(value, list | tuple) else [] if value is None else [value]
        numbers: list[int] = []
        for item in items:
            try:
                numbers.append(int(str(item).strip()))
            except ValueError:
                continue
        return numbers


def _echoes(text: str, reply_lines: Sequence[str]) -> bool:
    """Does ``text`` repeat one of her lines word for word (a plan is a hint, not the answer)?"""
    return any(
        len(line) >= MIN_ECHO_CHARS and not line.startswith("[") and line in text
        for line in reply_lines
    )


def plan_from_draft(draft: PlanDraft, source: PlanInput) -> StoredPlan | None:
    """The plan to store, or ``None`` when the draft has no usable intent.

    Fact numbers outside the list are dropped, a field that quotes her reply is dropped, and
    every field is cut to :data:`FIELD_CHARS` characters.
    """
    reply_lines = [normalize_line(line) for line in source.reply.split("\n")]
    seen: list[str] = []
    for number in draft.fact_numbers:
        if 1 <= number <= len(source.facts) and source.facts[number - 1] not in seen:
            seen.append(source.facts[number - 1])

    def clean(text: str) -> str:
        return "" if _echoes(normalize_line(text), reply_lines) else text[:FIELD_CHARS]

    intent = clean(draft.intent)
    if not intent:
        return None
    return StoredPlan(intent, tuple(seen[:MAX_FACTS]), clean(draft.tone), clean(draft.bubble_hint))


def plan_messages(template: PromptText, source: PlanInput) -> list[ChatMessage]:
    """The request that asks DeepSeek for the plan of ``source``."""
    context = "\n".join(
        f"{'对方' if role == 'user' else '她'}：{text}" for role, text in source.turns
    )
    facts = "\n".join(f"{number}. {line}" for number, line in enumerate(source.facts, start=1))
    return template.render(
        time=source.when,
        context=context or "（这是对话的开头）",
        facts=facts or "（没有）",
        reply=source.reply,
    )


# --------------------------------------------------------------------------- the table


@dataclass(frozen=True)
class PlanEntry:
    """A row of ``training_plans`` as the export sees it."""

    sample_id: str
    status: str
    target_sha: str
    plan: StoredPlan | None
    batch_id: str | None


@dataclass(frozen=True)
class RegisterResult:
    new: int
    reset: int
    kept: int


class PlanStore:
    """Reads and writes ``training_plans``."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def entries(self) -> dict[str, PlanEntry]:
        """Every plan row, by sample id (the export reads them all once)."""
        with self._db.session() as session:
            return {row.id: self._entry(row) for row in session.scalars(select(TrainingPlan))}

    @staticmethod
    def _entry(row: TrainingPlan) -> PlanEntry:
        stored = row.plan
        return PlanEntry(
            sample_id=row.id,
            status=row.status,
            target_sha=str(row.input.get("target_sha", "")),
            plan=StoredPlan.from_json(stored) if row.status == DONE and stored else None,
            batch_id=row.batch_id,
        )

    def get_input(self, sample_id: str) -> tuple[str, PlanInput] | None:
        """``(status, input)`` of a row, or ``None``."""
        with self._db.session() as session:
            row = session.get(TrainingPlan, sample_id)
            return None if row is None else (row.status, PlanInput.from_json(row.input))

    def register(self, inputs: Mapping[str, PlanInput]) -> RegisterResult:
        """Add the samples that need a plan; a row whose reply changed starts again."""
        new = reset = kept = 0
        with self._db.transaction() as session:
            for sample_id, source in inputs.items():
                row = session.get(TrainingPlan, sample_id)
                if row is None:
                    session.add(TrainingPlan(id=sample_id, status=PENDING, input=source.to_json()))
                    new += 1
                elif str(row.input.get("target_sha", "")) != source.target_sha:
                    row.status, row.plan, row.batch_id = PENDING, None, None
                    row.input = source.to_json()
                    reset += 1
                else:
                    kept += 1
        return RegisterResult(new, reset, kept)

    def pending_ids(self) -> list[str]:
        with self._db.session() as session:
            return sorted(
                session.scalars(select(TrainingPlan.id).where(TrainingPlan.status == PENDING))
            )

    def assign_batch(self, sample_ids: Iterable[str], batch_id: str) -> None:
        with self._db.transaction() as session:
            for sample_id in sample_ids:
                row = session.get(TrainingPlan, sample_id)
                if row is not None and row.status == PENDING:
                    row.batch_id = batch_id

    def save_done(
        self, sample_id: str, plan: StoredPlan, *, model: str, template_ref: str, cost_usd: float
    ) -> None:
        with self._db.transaction() as session:
            row = session.get(TrainingPlan, sample_id)
            if row is not None:
                row.status, row.plan = DONE, plan.to_json()
                row.model, row.template_ref, row.cost_usd = model, template_ref, cost_usd

    def save_refused(self, sample_id: str, *, cost_usd: float = 0.0) -> None:
        with self._db.transaction() as session:
            row = session.get(TrainingPlan, sample_id)
            if row is not None:
                row.status, row.plan, row.cost_usd = REFUSED, None, cost_usd

    def counts(self) -> dict[str, int]:
        with self._db.session() as session:
            found = dict.fromkeys((PENDING, DONE, REFUSED), 0)
            for status in session.scalars(select(TrainingPlan.status)):
                found[status] = found.get(status, 0) + 1
            return found


def needs_plan(entry: PlanEntry | None, source_sha: str) -> bool:
    """Is there no usable verdict for this reply yet (no row, a stale row, or one still waiting)?"""
    return entry is None or entry.target_sha != source_sha or entry.status == PENDING


# --------------------------------------------------------------------- the batches


@dataclass
class PlanQueueResult:
    """What :func:`queue_plan_batches` queued."""

    samples: int = 0
    jobs: int = 0
    estimated_usd: float = 0.0
    batch_ids: list[str] = field(default_factory=list)
    already_queued: int = 0


def queued_plan_ids(queue: JobQueue) -> set[str]:
    """Samples that wait in a pending or running plan job."""
    found: set[str] = set()
    for status in ("pending", "running"):
        for job in queue.list_jobs(status=status, job_type=PLAN_JOB, limit=100_000):
            found.update(str(i) for i in job.payload.get("ids", []))
    return found


def queue_plan_batches(
    services: Services, runtime: LlmRuntime, sample_ids: Sequence[str]
) -> PlanQueueResult:
    """Price the plan jobs of ``sample_ids`` and queue them as one-time batches (R-LLM-014).

    Every batch waits for ``twin jobs approve <batch>``; none is larger than
    ``budget.one_time_usd``.  Samples that already wait in a queued plan job are left alone.
    """
    queue = JobQueue(services.db, services.clock)
    waiting = queued_plan_ids(queue)
    wanted = [i for i in sample_ids if i not in waiting]
    result = PlanQueueResult(already_queued=len(sample_ids) - len(wanted))
    if not wanted:
        return result
    store = PlanStore(services.db)
    template = TemplateStore(services.db, services.clock).active(TRAIN_PLAN)
    model = services.settings.deepseek.offline_model
    prices: dict[str, float] = {}
    for sample_id in wanted:
        found = store.get_input(sample_id)
        if found is None:
            raise LookupError("a plan was asked for that was never registered")
        item = runtime.batches.item_from_messages(
            model, plan_messages(template, found[1]), completion_tokens=PLAN_COMPLETION_TOKENS
        )
        prices[sample_id] = runtime.batches.estimate_item(item)[0]
    limit = runtime.batches.limit_usd
    chunks: list[list[str]] = [[]]
    chunk_usd = 0.0
    for sample_id in wanted:
        price = prices[sample_id]
        if price > limit:
            raise BatchTooLargeError("a single plan", price, limit)
        if chunks[-1] and (len(chunks[-1]) >= PLAN_JOB_SIZE or chunk_usd + price > limit):
            chunks.append([])
            chunk_usd = 0.0
        chunks[-1].append(sample_id)
        chunk_usd += price
    estimates = [sum(prices[i] for i in chunk) for chunk in chunks]
    groups: list[list[int]] = [[]]
    running = 0.0
    for index, estimate in enumerate(estimates):
        if groups[-1] and running + estimate > limit:
            groups.append([])
            running = 0.0
        groups[-1].append(index)
        running += estimate
    stamp = f"{services.clock.now_utc():%Y%m%d%H%M%S}-{new_id()[-4:].lower()}"
    for number, group in enumerate(groups, start=1):
        batch_id = f"trainplan-{stamp}-{number}"
        runtime.batches.enqueue(
            batch_id,
            PLAN_JOB,
            [{"batch_id": batch_id, "ids": chunks[i]} for i in group],
            [estimates[i] for i in group],
            priority=PLAN_PRIORITY,
            offpeak_only=True,
        )
        store.assign_batch((s for i in group for s in chunks[i]), batch_id)
        result.batch_ids.append(batch_id)
    result.samples = len(wanted)
    result.jobs = len(chunks)
    result.estimated_usd = sum(estimates)
    return result


# ---------------------------------------------------------------------- running


async def write_plans(
    services: Services, runtime: LlmRuntime, sample_ids: Sequence[str], tag: LedgerTag
) -> tuple[int, int]:
    """Ask DeepSeek for the plans of ``sample_ids``; returns ``(written, refused)``.

    Each plan is saved as soon as it arrives, so a job that stops goes on from the next sample.
    """
    store = PlanStore(services.db)
    template = TemplateStore(services.db, services.clock).active(TRAIN_PLAN)
    written = refused = 0
    for sample_id in sample_ids:
        found = store.get_input(sample_id)
        if found is None or found[0] != PENDING:
            continue
        source = found[1]
        try:
            reply = await runtime.client.chat_json(
                plan_messages(template, source),
                PlanDraft,
                purpose=Purpose.TRAIN_PLAN,
                tag=tag,
                temperature=PLAN_TEMPERATURE,
                max_tokens=PLAN_MAX_TOKENS,
            )
        except StructuredOutputError:
            store.save_refused(sample_id)
            refused += 1
            continue
        except InvalidRequestError as exc:
            if "risk" not in str(exc).lower():
                raise
            store.save_refused(sample_id)
            refused += 1
            continue
        plan = (
            None
            if reply.chat.finish_reason in FILTER_FINISHES
            else plan_from_draft(reply.value, source)
        )
        if plan is None:
            store.save_refused(sample_id, cost_usd=reply.total_cost_usd)
            refused += 1
            continue
        store.save_done(
            sample_id,
            plan,
            model=reply.chat.model,
            template_ref=template.ref,
            cost_usd=reply.total_cost_usd,
        )
        written += 1
    return written, refused


@job_handler(PLAN_JOB)
async def handle_training_plan(ctx: JobContext) -> None:
    services = ctx.services
    if services is None:
        raise RuntimeError("training_plan needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    tag = LedgerTag("one_time", str(batch_id)) if batch_id else DAILY
    ids = [str(i) for i in payload.get("ids", [])]
    runtime = build_llm_runtime(services)
    try:
        written, refused = await write_plans(services, runtime, ids, tag)
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    finally:
        await runtime.client.aclose()
    log.info("training_plans_written", written=written, refused=refused, samples=len(ids))


def plan_moment(moment: datetime) -> str:
    """The moment as the plan prompt writes it (ISO, minutes; the zone is already local)."""
    return f"{moment:%Y-%m-%d %H:%M}"
