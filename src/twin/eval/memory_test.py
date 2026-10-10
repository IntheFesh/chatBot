"""The memory test: twenty questions from the fact store (R-EVAL-003, R-LLM-014, R-MEM-004).

``twin eval memory`` asks the bot twenty questions whose answers are facts it should remember:
**ten from the real records** (``real_record``) and **ten from the conversation with the bot**
(``user_said``, ``bot_invented`` or ``user_command``, the last being what the user asked her to
remember with ``/记住``; a fact the user took back with ``/忘掉`` is no longer current and is never
asked).  A test needs both: a bot that recalls the old chat but not what it said last week, or the
other way round, has not passed.  When the conversation with the
bot holds fewer than ten facts to ask about, the test says so ("chat a few more days first") and is
recorded as **not passed - not enough samples**; real records are never used to make up the number.

Three steps, each of which can stop and continue:

1. **Plan** (:func:`plan_memory`).  The facts are drawn (seeded) from the memory as it is now and
   one ``eval_items`` row per question is stored; the generation is priced and queued as a
   one-time batch (R-LLM-014).
2. **Ask** (:func:`ask_items`, the job ``eval_memory``).  For each fact DeepSeek rewrites it as the
   question the user would naturally ask, with the points a correct answer must contain; the
   question is put to the bot in the sandbox's ``live`` mode (the present, all the data, no write
   outside the evaluation tables, nothing sent to WeChat); DeepSeek then judges the answer
   (``correct`` / ``partial`` / ``wrong`` and a reason).
3. **Review** (:class:`MemorySession`).  The user goes through the questions and may change any
   verdict.  Only a reviewed run counts.

The score is ``correct + 0.5 * partial`` over the twenty; **80 %** or more passes
(:func:`summarize`).  Every question keeps its evidence: the fact (its number, source, when it
became known) and the messages it was taken from.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from twin.config.runtime import BACKEND_ACTIVE
from twin.engine.types import InboundItem
from twin.eval.blind import check_backends, pack
from twin.eval.render import render_candidate
from twin.eval.samples import StickerDescriber
from twin.eval.sandbox import (
    SandboxKit,
    SandboxMode,
    SandboxRequest,
    backend_status,
    build_sandbox,
    stable_seed,
)
from twin.eval.store import EvalStore, ItemView, NewItem, RunView
from twin.eval.ui import KeySource, ask
from twin.llm.errors import (
    BudgetDeniedError,
    CircuitOpenError,
    InvalidRequestError,
    StructuredOutputError,
)
from twin.llm.runtime import LlmRuntime
from twin.llm.types import ChatMessage, LedgerTag, Purpose
from twin.memory.api import memory_view
from twin.memory.records import FactRecord
from twin.ops.jobs import BatchTooLargeError, JobContext, JobDeferred, job_handler
from twin.ops.logging import get_logger
from twin.stickers.catalog import StickerCatalog

if TYPE_CHECKING:
    from twin.services import Services

log = get_logger("twin.eval.memory_test")

EVAL_MEMORY_JOB = "eval_memory"
JOB_PRIORITY = 125
JOB_SIZE = 5
PER_SOURCE = 10
TOTAL = 2 * PER_SOURCE
PASS_RATIO_NUM, PASS_RATIO_DEN = 4, 5  # 80 %
PROMPT_VERSION = "eval-memory-1"
REAL_SOURCES = ("real_record",)
BOT_SOURCES = ("user_said", "bot_invented", "user_command")
GROUP_REAL, GROUP_BOT = "real", "bot"
MIN_FACT_CHARS = 4
QUESTION_COMPLETION_TOKENS = 200
JUDGE_COMPLETION_TOKENS = 200
ANSWER_COMPLETION_TOKENS = 600
LEAK_RETRIES = 1

Outcome = Literal["correct", "partial", "wrong"]
SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

QUESTION_SYSTEM = (
    "你在为一个聊天机器人的记忆测试出题。机器人扮演用户的女朋友，和用户在微信上聊天。"
    "下面给你一条机器人应该记得的事实，请把它改写成用户会很自然地问出口的一句话（口语、"
    "简短、像微信聊天，不要像考试），并写出正确回答必须包含的要点。"
    "硬性要求：问句里不能出现答案本身（不要把要点原样写进问句）；不要加问句以外的内容；"
    "要点是简短的词或短句，一到三条。只输出 JSON："
    '{"question": "...", "key_points": ["..."]}'
)
JUDGE_SYSTEM = (
    "你在给一个聊天机器人的记忆测试判分。给你：用户问的问题、这个问题背后的事实、"
    "正确回答必须包含的要点、机器人的回答。请只根据事实和要点判断："
    "回答包含全部要点且不与事实矛盾 = correct；只包含一部分要点、含糊但没有说错 = partial；"
    "没有回答问题、答错、与事实矛盾或编造了别的答案 = wrong。"
    '只输出 JSON：{"verdict": "correct|partial|wrong", "reason": "一句话理由"}'
)


class QuestionDraft(BaseModel):
    """DeepSeek's rewrite of a fact as a question."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=2, max_length=160)
    key_points: list[str] = Field(min_length=1, max_length=5)


class Judgement(BaseModel):
    """DeepSeek's verdict on one answer."""

    model_config = ConfigDict(extra="ignore")

    verdict: Outcome
    reason: str = Field(default="", max_length=400)


def question_messages(fact_text: str, source: str, note: str = "") -> list[ChatMessage]:
    origin = {
        "real_record": "来自过去真实的聊天记录",
        "user_said": "是用户和机器人聊天时自己说过的",
        "bot_invented": "是机器人聊天时说起的关于她自己的事",
        "user_command": "是用户用“记住”指令专门让机器人记下的",
    }.get(source, "")
    body = f"事实（{origin}）：{fact_text}"
    if note:
        body += f"\n上一次出的题不合格：{note}"
    return [
        {"role": "system", "content": QUESTION_SYSTEM},
        {"role": "user", "content": body},
    ]


def judge_messages(
    question: str, fact_text: str, key_points: Sequence[str], answer: str
) -> list[ChatMessage]:
    body = (
        f"问题：{question}\n事实：{fact_text}\n要点：{'；'.join(key_points)}\n"
        f"机器人的回答：{answer or '（没有回答）'}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": body},
    ]


def leaked(question: str, key_points: Sequence[str]) -> str | None:
    """A key point that the question already contains (the answer must not be in the question)."""
    for point in key_points:
        if len(point.strip()) >= 2 and point.strip() in question:
            return point.strip()
    return None


# ----------------------------------------------------------------------- facts


def group_of(source: str) -> str | None:
    if source in REAL_SOURCES:
        return GROUP_REAL
    if source in BOT_SOURCES:
        return GROUP_BOT
    return None


def eligible_facts(services: Services, now: datetime) -> tuple[list[FactRecord], list[FactRecord]]:
    """The facts a question can be asked about now: ``(from real records, from the bot's chat)``."""
    seen: set[str] = set()
    real: list[FactRecord] = []
    bot: list[FactRecord] = []
    for fact in memory_view(services, now).facts():
        group = group_of(fact.source)
        text = fact.text.strip()
        if group is None or not fact.current or len(text) < MIN_FACT_CHARS or text in seen:
            continue
        seen.add(text)
        (real if group == GROUP_REAL else bot).append(fact)
    return real, bot


@dataclass
class MemoryPlan:
    """The outcome of :func:`plan_memory`."""

    run: RunView
    real_available: int
    bot_available: int
    estimated_usd: float = 0.0
    batch_ids: list[str] = field(default_factory=list)

    @property
    def insufficient(self) -> bool:
        return self.run.verdict == "insufficient"


def fact_payload(fact: FactRecord) -> dict[str, Any]:
    return {
        "fact_id": fact.id,
        "fact_number": fact.number,
        "fact": fact.text,
        "source": fact.source,
        "subject": fact.subject,
        "category": fact.category,
        "known_at": fact.known_at.isoformat(),
        "evidence": fact.evidence or {},
    }


async def plan_memory(
    services: Services,
    *,
    seed: int | None = None,
    kit: SandboxKit | None = None,
    backend: str | None = None,
) -> MemoryPlan:
    """Choose the twenty facts and queue the questions (see the module description)."""
    store = EvalStore(services.db, services.clock)
    now = services.clock.now_utc()
    real, bot = eligible_facts(services, now)
    chosen_seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    if len(bot) < PER_SOURCE or len(real) < PER_SOURCE:
        reasons = []
        if len(bot) < PER_SOURCE:
            reasons.append(
                f"only {len(bot)} fact(s) from the conversation with the bot (need {PER_SOURCE}): "
                "chat a few more days first"
            )
        if len(real) < PER_SOURCE:
            reasons.append(
                f"only {len(real)} fact(s) from the real records (need {PER_SOURCE}): "
                "run `twin memory replay start` first"
            )
        run = store.create_run(
            "memory",
            mode="live",
            status="done",
            verdict="insufficient",
            params={"seed": chosen_seed, "prompt_version": PROMPT_VERSION},
            summary={
                "total": 0,
                "real_available": len(real),
                "bot_available": len(bot),
                "reasons": reasons,
                "passed": False,
            },
        )
        return MemoryPlan(run, len(real), len(bot))
    answer_backend = backend or str(services.runtime.get(BACKEND_ACTIVE))
    check_backends(services, [answer_backend])
    rng = random.Random(chosen_seed)  # noqa: S311 - a reproducible draw, not security
    picked = [*rng.sample(real, PER_SOURCE), *rng.sample(bot, PER_SOURCE)]
    rng.shuffle(picked)
    run = store.create_run(
        "memory",
        mode="live",
        backends=[answer_backend],
        params={
            "seed": chosen_seed,
            "prompt_version": PROMPT_VERSION,
            "answer_backend": answer_backend,
            "real_available": len(real),
            "bot_available": len(bot),
        },
    )
    store.add_items(
        run.id,
        [
            NewItem(
                sample_key=fact.id,
                backend=answer_backend,
                at=now,
                payload=fact_payload(fact),
                source=fact.source,
            )
            for fact in picked
        ],
    )
    own = kit is None
    sandbox_kit = kit or build_sandbox(services, mode=SandboxMode.LIVE)
    try:
        usd = await _estimate_items(services, sandbox_kit, picked, answer_backend, now)
        batches = sandbox_kit.llm.batches
        stored = store.items(run.id, with_payload=False)
        by_fact = dict(zip((f.id for f in picked), usd, strict=True))
        jobs = [stored[i : i + JOB_SIZE] for i in range(0, len(stored), JOB_SIZE)]
        job_usd = [sum(by_fact[item.sample_key] for item in job) for job in jobs]
        batch_ids: list[str] = []
        try:
            for number, group in enumerate(pack(job_usd, batches.limit_usd), start=1):
                batch_id = f"evalmem-{run.id[-8:].lower()}-{number}"
                batches.enqueue(
                    batch_id,
                    EVAL_MEMORY_JOB,
                    [
                        {
                            "run_id": run.id,
                            "batch_id": batch_id,
                            "item_ids": [item.id for item in jobs[i]],
                        }
                        for i in group
                    ],
                    [job_usd[i] for i in group],
                    priority=JOB_PRIORITY,
                    offpeak_only=False,
                )
                batch_ids.append(batch_id)
        except BatchTooLargeError:
            store.update_run(run.id, status="cancelled")
            raise
        total = sum(job_usd)
        run = store.update_run(
            run.id, batch_ids=batch_ids, params={"estimated_usd": round(total, 6)}
        )
        return MemoryPlan(run, len(real), len(bot), total, batch_ids)
    finally:
        if own:
            await sandbox_kit.aclose()


async def _estimate_items(
    services: Services,
    kit: SandboxKit,
    facts: Sequence[FactRecord],
    backend: str,
    now: datetime,
) -> list[float]:
    """The estimated cost of each question: the rewrite, the answer and the judgement."""
    batches = kit.llm.batches
    model = services.settings.deepseek.chat_model
    offline = services.settings.deepseek.offline_model
    prices: list[float] = []
    for fact in facts:
        rewrite = batches.item_from_messages(
            offline,
            question_messages(fact.text, fact.source),
            completion_tokens=QUESTION_COMPLETION_TOKENS,
        )
        judge = batches.item_from_messages(
            offline,
            judge_messages(fact.text, fact.text, [fact.text], fact.text),
            completion_tokens=JUDGE_COMPLETION_TOKENS,
        )
        usd = batches.estimate_item(rewrite)[0] + batches.estimate_item(judge)[0]
        if backend != "style":
            stand_in = InboundItem("estimate", now, "text", fact.text)
            messages = await kit.sandbox.preview(SandboxRequest((stand_in,), (), now, backend))
            answer = batches.item_from_messages(
                model, messages, completion_tokens=ANSWER_COMPLETION_TOKENS
            )
            usd += batches.estimate_item(answer)[0]
        prices.append(usd)
    return prices


# ------------------------------------------------------------------------- ask


@dataclass
class AskSummary:
    asked: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0


async def _one_question(
    llm: LlmRuntime, tag: LedgerTag, fact: str, source: str
) -> tuple[QuestionDraft, float]:
    """The question and its key points; a question that gives the answer away is asked again."""
    note = ""
    cost = 0.0
    last = ""
    for _ in range(LEAK_RETRIES + 1):
        result = await llm.client.chat_json(
            question_messages(fact, source, note),
            QuestionDraft,
            purpose=Purpose.EVAL,
            tag=tag,
            temperature=0.7,
            max_tokens=QUESTION_COMPLETION_TOKENS * 2,
        )
        cost += result.total_cost_usd
        giveaway = leaked(result.value.question, result.value.key_points)
        if giveaway is None:
            return result.value, cost
        last = giveaway
        note = f"问句里已经包含了答案“{giveaway}”，请换一种问法"
    raise StructuredOutputError(f"the question gives the answer away ({last!r})", attempts=2)


async def ask_item(
    services: Services, kit: SandboxKit, store: EvalStore, item: ItemView, tag: LedgerTag
) -> None:
    """Rewrite the fact, put the question to the bot, judge the answer; save the item."""
    llm = kit.llm
    payload = item.payload
    fact_text = str(payload["fact"])
    cost = 0.0
    try:
        draft, spent = await _one_question(llm, tag, fact_text, str(payload["source"]))
    except (StructuredOutputError, InvalidRequestError):
        store.mark_failed(item.id, "question_not_written")
        return
    cost += spent
    now = item.at
    request = SandboxRequest(
        inbound=(InboundItem(f"question-{item.id}", now, "text", draft.question),),
        history=(),
        at=now,
        backend=item.backend,
        thinking_mode="off",
        seed=stable_seed(item.id),
    )
    reply = await kit.sandbox.reply(request)
    cost += reply.draft.cost_usd
    describe = StickerDescriber(StickerCatalog(services))
    answer = render_candidate(reply.candidate, describe) if reply.usable else ""
    try:
        judged = await llm.client.chat_json(
            judge_messages(draft.question, fact_text, draft.key_points, answer),
            Judgement,
            purpose=Purpose.EVAL,
            tag=tag,
            temperature=0.0,
            max_tokens=JUDGE_COMPLETION_TOKENS * 2,
        )
    except (StructuredOutputError, InvalidRequestError):
        store.mark_failed(item.id, "judgement_not_written")
        return
    cost += judged.total_cost_usd
    verdict: Outcome = judged.value.verdict if answer else "wrong"
    store.save_auto(
        item.id,
        {
            "question": draft.question,
            "key_points": draft.key_points,
            "answer": answer,
            "judge_reason": judged.value.reason if answer else "没有回答",
            "answered_by": reply.draft.backend,
            "answered": bool(answer),
        },
        auto_outcome=verdict,
        cost_usd=cost,
    )


async def ask_items(
    services: Services, run_id: str, item_ids: Sequence[str], batch_id: str | None
) -> AskSummary:
    store = EvalStore(services.db, services.clock)
    summary = AskSummary()
    kit = build_sandbox(services, mode=SandboxMode.LIVE, batch_id=batch_id)
    tag = LedgerTag("one_time", batch_id) if batch_id else LedgerTag()
    try:
        for item_id in item_ids:
            item = store.item(item_id)
            if item.status != "pending":
                summary.skipped += 1
                continue
            if not backend_status(services, item.backend).available:
                store.mark_failed(item.id, "not_deployed")
                summary.failed += 1
                continue
            await ask_item(services, kit, store, item, tag)
            after = store.item(item.id)
            summary.cost_usd += after.cost_usd
            if after.status == "generated":
                summary.asked += 1
            else:
                summary.failed += 1
    finally:
        await kit.aclose()
    return summary


@job_handler(EVAL_MEMORY_JOB)
async def handle_eval_memory(ctx: JobContext) -> None:
    """The job of a batch: ask some questions of a memory run and judge the answers."""
    services = ctx.services
    if services is None:
        raise RuntimeError("eval_memory needs the services container")
    payload = ctx.job.payload
    batch_id = payload.get("batch_id")
    try:
        summary = await ask_items(
            services,
            str(payload["run_id"]),
            [str(i) for i in payload.get("item_ids", [])],
            str(batch_id) if batch_id else None,
        )
    except BudgetDeniedError as exc:
        raise JobDeferred(str(exc), retry_in_s=3600.0) from exc
    except CircuitOpenError as exc:
        raise JobDeferred("DeepSeek circuit breaker is open", retry_in_s=300.0) from exc
    log.info(
        "eval_questions_asked",
        asked=summary.asked,
        failed=summary.failed,
        cost_usd=round(summary.cost_usd, 6),
    )


# ---------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class MemorySummary:
    """The score of a memory run."""

    total: int  # questions
    reviewed: int  # decided by the user
    correct: int
    partial: int
    wrong: int
    real_items: int
    bot_items: int
    failed: int = 0

    @property
    def points(self) -> float:
        return self.correct + 0.5 * self.partial

    @property
    def accuracy(self) -> float | None:
        return self.points / self.total if self.total else None

    @property
    def composition_complete(self) -> bool:
        """Ten questions from each source and nothing else (R-EVAL-003)."""
        return self.real_items == PER_SOURCE and self.bot_items == PER_SOURCE

    @property
    def complete(self) -> bool:
        """Every one of the twenty questions has a verdict the user has seen."""
        return self.composition_complete and self.reviewed == TOTAL and self.total == TOTAL

    @property
    def meets_threshold(self) -> bool:
        """``correct + 0.5 partial`` is at least 80 % of the questions (integer arithmetic)."""
        half_points = 2 * self.correct + self.partial
        return bool(self.total) and half_points * PASS_RATIO_DEN >= PASS_RATIO_NUM * 2 * self.total

    @property
    def verdict(self) -> Literal["passed", "failed", "insufficient"]:
        if not self.complete:
            return "insufficient"
        return "passed" if self.meets_threshold else "failed"


def summarize(items: Sequence[ItemView]) -> MemorySummary:
    """Count a memory run's questions by what the user decided."""
    usable = [i for i in items if i.status != "failed"]
    reviewed = [i for i in usable if i.status == "judged" and i.outcome is not None]
    outcomes = Counter(i.outcome for i in reviewed)
    groups = Counter(group_of(i.source or "") for i in usable)
    return MemorySummary(
        total=len(items),
        reviewed=len(reviewed),
        correct=outcomes["correct"],
        partial=outcomes["partial"],
        wrong=outcomes["wrong"],
        real_items=groups[GROUP_REAL],
        bot_items=groups[GROUP_BOT],
        failed=len(items) - len(usable),
    )


def finish_run(store: EvalStore, run_id: str) -> RunView:
    """Close a reviewed run: the score and the verdict are stored on it."""
    items = store.items(run_id, with_payload=False)
    summary = summarize(items)
    return store.update_run(
        run_id,
        status="done",
        verdict=summary.verdict,
        summary={
            "total": summary.total,
            "reviewed": summary.reviewed,
            "correct": summary.correct,
            "partial": summary.partial,
            "wrong": summary.wrong,
            "failed": summary.failed,
            "real_items": summary.real_items,
            "bot_items": summary.bot_items,
            "points": summary.points,
            "accuracy": summary.accuracy,
            "composition_complete": summary.composition_complete,
            "passed": summary.verdict == "passed",
        },
    )


# ----------------------------------------------------------------------- review

KEY_KEEP = "k"
KEY_CORRECT = "c"
KEY_PARTIAL = "p"
KEY_WRONG = "w"
KEY_QUIT = "q"
REVIEW_KEYS = (KEY_KEEP, "", KEY_CORRECT, KEY_PARTIAL, KEY_WRONG, KEY_QUIT)
_BY_KEY: dict[str, Outcome] = {KEY_CORRECT: "correct", KEY_PARTIAL: "partial", KEY_WRONG: "wrong"}
_NAMES = {"correct": "正确", "partial": "部分正确", "wrong": "错误"}


@dataclass
class ReviewOutcome:
    reviewed: int = 0
    changed: int = 0
    quit: bool = False
    finished: bool = False


class MemorySession:
    """The user goes through the questions and confirms or changes each verdict."""

    def __init__(self, store: EvalStore, run: RunView, console: Console, keys: KeySource) -> None:
        self._store = store
        self._run = run
        self._console = console
        self._keys = keys

    def pending(self) -> list[ItemView]:
        return self._store.items(self._run.id, status="generated")

    def run(self) -> ReviewOutcome:
        outcome = ReviewOutcome()
        pending = self.pending()
        for position, item in enumerate(pending, start=1):
            self._show(item, position, len(pending))
            key = ask(
                self._keys,
                self._console,
                "回车/k 保持判分  [c] 正确  [p] 部分正确  [w] 错误  [q] 保存退出",
                REVIEW_KEYS,
            )
            if key is None or key == KEY_QUIT:
                outcome.quit = True
                return outcome
            auto: Outcome = item.auto_outcome or "wrong"  # type: ignore[assignment]
            chosen: Outcome = _BY_KEY.get(key, auto)
            self._store.judge(
                item.id,
                chosen,
                score=SCORES[chosen],
                payload={"reviewed_as": chosen, "changed": chosen != auto},
            )
            outcome.reviewed += 1
            outcome.changed += int(chosen != auto)
        run = finish_run(self._store, self._run.id)
        outcome.finished = run.status == "done"
        return outcome

    def _show(self, item: ItemView, position: int, total: int) -> None:
        payload = item.payload
        source = "真实记录" if group_of(item.source or "") == GROUP_REAL else "和机器人的对话"
        lines = [
            Text(f"问：{payload.get('question', '')}"),
            Text(f"事实 #{payload.get('fact_number', '?')}（{source}）：{payload.get('fact', '')}"),
            Text("要点：" + "；".join(str(p) for p in payload.get("key_points", []))),
            Text(f"机器人答：{payload.get('answer') or '（没有回答）'}"),
            Text(
                f"自动判分：{_NAMES.get(item.auto_outcome or '', '?')}"
                f" —— {payload.get('judge_reason', '')}",
                style="cyan",
            ),
        ]
        self._console.print(Panel(Group(*lines), title=f"第 {position}/{total} 题"))
