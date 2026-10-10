"""``twin model activate`` and ``disable``: when a model may become the default (R-SRV-001/005).

Activating is the one decision that puts the style model between the user and her replies, so it
has rules and leaves a trail:

1. the model must be **renderable and servable**: bound to the template version this program
   renders, and of the kind of file the configured mode serves (a GGUF for ``llamacpp_completion``,
   the LoRA adapter for ``vllm_completion``);
2. its **tokenizer comparison** must have passed for exactly this file (R-TRN-011.4); if there is
   no record for it the check is made now.  A model whose tokens differ from the training is
   refused even with ``--force``: that is a bug in the files, not a question of quality, and the
   alert ``style_tokenize_mismatch`` is raised;
3. the **release gate** (M5, :func:`~twin.serving.gate_m5.judge_model`) must pass.  Not passed means
   refused, with the numbers and, when the evidence is short, how many pairs are missing.
   ``--force`` activates anyway, but the model is then marked ``gate_passed = false`` for good (a
   later activation judges again), ``/状态`` says "未通过门槛", the budget's last level never hands
   over to it (R-LLM-008), and the activation is on record as forced.

A passed gate sets the runtime setting ``backend.active`` to the better way (``style`` or
``hybrid``); a forced activation sets it to the way asked for with ``--backend`` (``style`` by
default) - the user said so on the command line, it is never done silently.  Either way a stored
fall-back is cleared.  Activating a model of the adapter kind also asks for the SSH tunnel.

Every activation and deactivation is appended to the ``activation_log`` of the registry row (when,
what, forced or not, which gate run) and written to the audit log.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, TUNNEL_WANTED, BackendName
from twin.engine.style_models import SERVED_KINDS
from twin.eval.gates import GateContext, GateVerdict
from twin.eval.store import EvalStore
from twin.ops.logging import get_logger
from twin.serving.gate_m5 import WAYS, judge_model
from twin.serving.tokencheck import TokenizationReport, TokenizeCheckError, tokenize_record
from twin.training import lf_template
from twin.training.registry import (
    EVAL_GATE,
    EVAL_TOKENIZE_CHECK,
    ModelView,
    RegistryError,
    activate_row,
    deactivate_row,
    find_model,
    record_eval,
)

if TYPE_CHECKING:
    from twin.services import Services

audit = get_logger("twin.model.audit")

TokenCheck = Callable[[ModelView], Awaitable[TokenizationReport]]
Judge = Callable[[GateContext, str], GateVerdict]


class ActivationRefused(RuntimeError):
    """The model may not be activated: a closed ``code`` and the lines that say why."""

    def __init__(self, code: str, message: str, lines: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.lines = lines


@dataclass(frozen=True)
class ActivationResult:
    model: ModelView
    previous: ModelView | None
    verdict: GateVerdict | None
    forced: bool
    gate_passed: bool
    backend: str  # what backend.active was set to
    tokenizer: TokenizationReport | None  # None: a recorded check for this very file was used


@dataclass(frozen=True)
class DisableResult:
    model: ModelView
    was_active: bool
    backend_reset: bool  # backend.active went back to deepseek


def has_valid_check(model: ModelView) -> bool:
    """Is there a passed comparison for exactly this file?"""
    record = model.eval.get(EVAL_TOKENIZE_CHECK)
    return (
        isinstance(record, dict)
        and record.get("ok") is True
        and record.get("model_sha256") == model.sha256
    )


def _kind_for(services: Services) -> str:
    return SERVED_KINDS[services.settings.style_model.mode]


def require_servable(services: Services, model: ModelView) -> None:
    """Rule 1: the template this program renders and the kind of file the mode serves."""
    if model.versions.template_version != lf_template.TEMPLATE_VERSION:
        raise ActivationRefused(
            "template",
            f"the model is bound to the template {model.versions.template_version}, "
            f"this program renders {lf_template.TEMPLATE_VERSION}",
        )
    wanted = _kind_for(services)
    if model.kind != wanted:
        raise ActivationRefused(
            "kind",
            f"style_model.mode is {services.settings.style_model.mode}, which serves {wanted} "
            f"files, but {model.id} is a {model.kind} file",
        )


async def ensure_tokenizer_check(
    services: Services, model: ModelView, check: TokenCheck
) -> TokenizationReport | None:
    """Rule 2: a passed comparison for this file; made now if there is none."""
    if has_valid_check(model):
        return None
    try:
        report = await check(model)
    except TokenizeCheckError as exc:
        raise ActivationRefused("tokenizer_unavailable", str(exc)) from exc
    record_eval(
        services.db,
        model.id,
        {
            EVAL_TOKENIZE_CHECK: tokenize_record(
                report, model, at=services.clock.now_utc().isoformat()
            )
        },
    )
    if not report.ok:
        services.alerts.raise_alert(
            "style_tokenize_mismatch",
            f"the tokens of {model.id} differ from the training tokenizer",
            severity="warning",
            detail={"model": model.id, "differences": len(report.differences)},
            dedup_key=f"style_tokenize_mismatch:{model.id}",
        )
        raise ActivationRefused(
            "tokenizer",
            "the server's tokens differ from the training tokenizer",
            tuple(report.lines()),
        )
    return report


def _gate_lines(verdict: GateVerdict) -> tuple[str, ...]:
    return (verdict.summary, *(f"- {c.name}：{c.detail}" for c in verdict.checks))


def _choose_backend(verdict: GateVerdict, asked: str | None, *, forced: bool) -> str:
    if asked is not None and asked not in WAYS:
        raise ActivationRefused("backend", f"--backend is style or hybrid, not {asked!r}")
    if forced:
        return asked or "style"
    ways = verdict.values.get("ways", {})
    if asked is not None:
        if ways.get(asked, {}).get("verdict") != "passed":
            raise ActivationRefused(
                "backend",
                f"{asked} did not pass the gate; the winner is {verdict.values.get('winner')}",
            )
        return asked
    winner = verdict.values.get("winner")
    if winner not in WAYS:
        raise ActivationRefused("backend", "the gate passed without naming a winner")
    return str(winner)


async def activate_model(
    services: Services,
    reference: str,
    *,
    token_check: TokenCheck,
    force: bool = False,
    backend: str | None = None,
    judge: Judge = judge_model,
) -> ActivationResult:
    """Activate a model under the rules of the module description (raises ActivationRefused)."""
    try:
        model = find_model(services.db, reference)
    except RegistryError as exc:
        raise ActivationRefused("unknown", str(exc)) from exc
    require_servable(services, model)
    tokenizer = await ensure_tokenizer_check(services, model, token_check)

    store = EvalStore(services.db, services.clock)
    verdict = judge(GateContext(services, store), model.id)
    gate_run = store.create_run(
        "gate",
        status="done",
        milestone="M5",
        verdict=verdict.verdict,
        params={"milestone": "M5", "model_id": model.id},
        summary=verdict.to_json(),
    )
    passed = verdict.passed
    forced = force and not passed
    if not passed and not force:
        raise ActivationRefused(
            "gate",
            f"{model.id} did not pass the release gate (M5): DeepSeek stays the default backend",
            _gate_lines(verdict),
        )
    chosen = _choose_backend(verdict, backend, forced=forced)
    now = services.clock.now_utc().isoformat()
    entry = {
        "at": now,
        "action": "activate",
        "forced": forced,
        "gate": verdict.verdict,
        "gate_run": gate_run.id,
        "backend": chosen,
    }
    gate_record = {"run": gate_run.id, "verdict": verdict.verdict, "passed": passed, "at": now}
    previous, updated = activate_row(
        services.db,
        model.id,
        gate_passed=passed,
        entry=entry,
        eval_updates={EVAL_GATE: gate_record},
    )
    services.runtime.set(BACKEND_ACTIVE, cast(BackendName, chosen), by="command")
    services.runtime.set(BACKEND_FALLBACK, None, by="command")
    if model.kind == "adapter":
        services.runtime.set(TUNNEL_WANTED, True, by="command")
    audit.warning(
        "model_activated" if not forced else "model_activated_forced",
        model=model.id,
        previous=previous.id if previous is not None else None,
        gate=verdict.verdict,
        gate_run=gate_run.id,
        backend=chosen,
    )
    return ActivationResult(updated, previous, verdict, forced, passed, chosen, tokenizer)


def disable_model(services: Services, reference: str) -> DisableResult:
    """Stop using a model; with no other model active the default goes back to DeepSeek."""
    try:
        model = find_model(services.db, reference)
    except RegistryError as exc:
        raise ActivationRefused("unknown", str(exc)) from exc
    was_active = model.active
    updated = deactivate_row(
        services.db,
        model.id,
        entry={"at": services.clock.now_utc().isoformat(), "action": "disable"},
    )
    reset = False
    if was_active:
        if services.runtime.get(BACKEND_ACTIVE) != "deepseek":
            services.runtime.set(BACKEND_ACTIVE, "deepseek", by="command")
            reset = True
        services.runtime.set(BACKEND_FALLBACK, None, by="command")
        if model.kind == "adapter":
            services.runtime.set(TUNNEL_WANTED, False, by="command")
    audit.warning("model_disabled", model=model.id, was_active=was_active, backend_reset=reset)
    return DisableResult(updated, was_active, reset)
