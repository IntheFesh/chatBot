"""``twin model activate`` and ``disable``: the gate, the tokenizer check, the audit (R-SRV-005)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from sqlalchemy import select

from tests.support.model_eval import evaluation_run, style_report
from tests.support.style_models import register_model
from twin.config.runtime import BACKEND_ACTIVE, BACKEND_FALLBACK, TUNNEL_WANTED
from twin.engine.style_models import StyleModels
from twin.eval.gates import GateContext, GateVerdict
from twin.eval.store import EvalStore, RunView
from twin.eval.style_metrics import StyleReport
from twin.services import Services
from twin.serving.activation import (
    ActivationRefused,
    activate_model,
    disable_model,
    has_valid_check,
)
from twin.serving.gate_m5 import judge_model
from twin.serving.tokencheck import Difference, TokenizationReport, TokenizeCheckError
from twin.storage.models import Alert
from twin.training.registry import EVAL_ACTIVATION_LOG, EVAL_GATE, get_model

RUN = "r-1"
MODEL = f"{RUN}-Q5_K_M"


class Checker:
    """The tokenizer comparison of the tests: passes, fails or cannot be made."""

    def __init__(self, outcome: str = "ok") -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    async def __call__(self, model: object) -> TokenizationReport:
        self.calls.append(getattr(model, "id", "?"))
        if self.outcome == "unavailable":
            raise TokenizeCheckError("the server could not tokenize 'plain'")
        if self.outcome == "mismatch":
            difference = Difference("plain", "extra_prefix", 0, 5, 7, 20, 21, "<|im_start|>", "<s>")
            return TokenizationReport(7, 500, (difference,))
        return TokenizationReport(7, 500)


def good_style(
    services: Services, store: EvalStore, runs: Sequence[RunView], backend: str
) -> StyleReport:
    return style_report(backend)


def judge_with_good_style(ctx: GateContext, model_id: str) -> GateVerdict:
    return judge_model(ctx, model_id, style_of=good_style)


def register(services: Services, run: str = RUN, **options: object) -> str:
    options.setdefault("active", False)
    options.setdefault("gate_passed", None)
    return register_model(services, run_id=run, **options)  # type: ignore[arg-type]


async def activate(
    services: Services,
    reference: str = MODEL,
    *,
    checker: Checker | None = None,
    **options: object,
) -> object:
    return await activate_model(
        services,
        reference,
        token_check=checker or Checker(),
        judge=judge_with_good_style,
        **options,  # type: ignore[arg-type]
    )


def passing_evaluation(services: Services, model: str = MODEL) -> None:
    evaluation_run(services, model, {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 18)})


def failing_evaluation(services: Services, model: str = MODEL) -> None:
    evaluation_run(services, model, {"deepseek": (50, 30), "style": (50, 30), "hybrid": (50, 31)})


# ----------------------------------------------------------------- the gate


async def test_a_model_that_was_never_evaluated_is_refused_with_the_way_forward(
    services: Services,
) -> None:
    register(services)
    with pytest.raises(ActivationRefused) as refused:
        await activate(services)
    assert refused.value.code == "gate"
    assert any("twin model evaluate" in line for line in refused.value.lines)
    model = get_model(services.db, MODEL)
    assert not model.active and model.gate_passed is None
    assert services.runtime.get(BACKEND_ACTIVE) == "deepseek"  # nothing changed silently


async def test_a_model_that_failed_the_gate_is_refused_with_the_numbers(
    services: Services,
) -> None:
    register(services)
    failing_evaluation(services)
    with pytest.raises(ActivationRefused) as refused:
        await activate(services)
    text = "\n".join(refused.value.lines)
    assert "50" in text and "猜对" in text and "deepseek" in text
    assert not get_model(services.db, MODEL).active
    assert services.runtime.get(BACKEND_ACTIVE) == "deepseek"
    # the verdict was judged and stored, like `twin eval gate M5` does
    store = EvalStore(services.db, services.clock)
    gate = store.latest_run("gate", milestone="M5")
    assert gate is not None and gate.verdict == "failed" and gate.params["model_id"] == MODEL


async def test_a_model_that_passed_becomes_the_default_with_the_better_way(
    services: Services,
) -> None:
    register(services)
    passing_evaluation(services)
    services.runtime.set(BACKEND_FALLBACK, {"requested": "style", "reason": "error"}, by="auto")
    result = await activate(services)
    model = get_model(services.db, MODEL)
    assert model.active and model.enabled and model.gate_passed is True
    assert result.backend == "hybrid" and not result.forced and result.gate_passed  # type: ignore[attr-defined]
    assert services.runtime.get(BACKEND_ACTIVE) == "hybrid"
    assert services.runtime.get(BACKEND_FALLBACK) is None
    entry = model.eval[EVAL_ACTIVATION_LOG][-1]
    assert entry["action"] == "activate" and entry["forced"] is False and entry["gate"] == "passed"
    assert entry["backend"] == "hybrid" and model.eval[EVAL_GATE]["passed"] is True
    active = StyleModels(services.db).active()
    assert active is not None and active.id == MODEL and active.passed_gate
    assert services.runtime.get(TUNNEL_WANTED) is False  # a GGUF needs no tunnel


async def test_the_better_way_can_be_overridden_by_another_way_that_also_passed(
    services: Services,
) -> None:
    register(services)
    passing_evaluation(services)
    result = await activate(services, backend="style")
    assert result.backend == "style" and services.runtime.get(BACKEND_ACTIVE) == "style"  # type: ignore[attr-defined]


async def test_a_way_that_did_not_pass_cannot_be_chosen_on_a_passed_gate(
    services: Services,
) -> None:
    register(services)
    evaluation_run(services, MODEL, {"deepseek": (50, 40), "style": (50, 38), "hybrid": (50, 18)})
    with pytest.raises(ActivationRefused, match="style did not pass"):
        await activate(services, backend="style")
    assert not get_model(services.db, MODEL).active


async def test_activating_a_newer_model_deactivates_the_older_one_of_the_same_kind(
    services: Services,
) -> None:
    register(services, "r-0", active=True, gate_passed=True)
    register(services)
    passing_evaluation(services)
    result = await activate(services)
    assert result.previous is not None and result.previous.id == "r-0-Q5_K_M"  # type: ignore[attr-defined]
    assert not get_model(services.db, "r-0-Q5_K_M").active and get_model(services.db, MODEL).active


# ------------------------------------------------------------------- --force


async def test_a_forced_activation_is_marked_as_not_passed_and_is_on_record(
    services: Services,
) -> None:
    register(services)
    failing_evaluation(services)
    result = await activate(services, force=True, backend="hybrid")
    model = get_model(services.db, MODEL)
    assert result.forced and not result.gate_passed  # type: ignore[attr-defined]
    assert model.active and model.gate_passed is False
    assert services.runtime.get(BACKEND_ACTIVE) == "hybrid"  # the user said so on the command line
    entry = model.eval[EVAL_ACTIVATION_LOG][-1]
    assert entry["forced"] is True and entry["gate"] == "failed" and entry["gate_run"]
    active = StyleModels(services.db).active()
    assert active is not None and not active.passed_gate  # /状态 says so, level 4 never uses it


async def test_a_forced_activation_defaults_to_the_style_way(services: Services) -> None:
    register(services)
    result = await activate(services, force=True)
    assert result.backend == "style" and services.runtime.get(BACKEND_ACTIVE) == "style"  # type: ignore[attr-defined]


async def test_force_on_a_gate_that_passes_is_an_ordinary_activation(services: Services) -> None:
    register(services)
    passing_evaluation(services)
    result = await activate(services, force=True)
    assert not result.forced and result.gate_passed  # type: ignore[attr-defined]
    assert get_model(services.db, MODEL).gate_passed is True


async def test_a_forced_model_never_counts_for_the_last_budget_level(services: Services) -> None:
    from twin.engine.style_models import StyleModels as Models

    register(services)
    await activate(services, force=True)
    assert Models(services.db).active() is not None
    assert not Models(services.db).active().passed_gate  # type: ignore[union-attr]


async def test_a_bad_backend_name_is_refused(services: Services) -> None:
    register(services)
    with pytest.raises(ActivationRefused, match="style or hybrid"):
        await activate(services, force=True, backend="deepseek")


# ------------------------------------------------------ what even force cannot do


async def test_a_model_of_another_template_cannot_be_activated_even_by_force(
    services: Services,
) -> None:
    register(services, template_version="qwen3_think@1")
    with pytest.raises(ActivationRefused) as refused:
        await activate(services, force=True)
    assert refused.value.code == "template" and "qwen3_think@1" in refused.value.message


async def test_the_kind_of_file_must_be_the_one_the_mode_serves(services: Services) -> None:
    register(services, "r-lora", quant="lora", kind="adapter")
    with pytest.raises(ActivationRefused) as refused:
        await activate(services, "r-lora-lora", force=True)
    assert refused.value.code == "kind" and "llamacpp_completion" in refused.value.message


async def test_an_adapter_is_activated_in_vllm_mode_and_asks_for_the_tunnel(
    services: Services,
) -> None:
    services.settings.style_model.mode = "vllm_completion"
    register(services, "r-lora", quant="lora", kind="adapter")
    evaluation_run(
        services, "r-lora-lora", {"deepseek": (50, 40), "style": (50, 20), "hybrid": (50, 18)}
    )
    await activate(services, "r-lora-lora")
    assert services.runtime.get(TUNNEL_WANTED) is True
    assert get_model(services.db, "r-lora-lora").active


async def test_an_unknown_model_is_refused(services: Services) -> None:
    with pytest.raises(ActivationRefused) as refused:
        await activate(services, "nothing")
    assert refused.value.code == "unknown"


# ------------------------------------------------------------ the tokenizer


async def test_the_comparison_is_made_when_there_is_no_record_and_recorded(
    services: Services,
) -> None:
    register(services)
    passing_evaluation(services)
    checker = Checker()
    result = await activate(services, checker=checker)
    assert checker.calls == [MODEL] and result.tokenizer is not None  # type: ignore[attr-defined]
    model = get_model(services.db, MODEL)
    assert has_valid_check(model) and model.eval["tokenize_check"]["tokens"] == 500


async def test_a_recorded_comparison_of_the_same_file_is_not_made_again(
    services: Services,
) -> None:
    register(services)
    passing_evaluation(services)
    checker = Checker()
    await activate(services, checker=checker)
    await activate(services, checker=checker)
    assert checker.calls == [MODEL]  # the second activation used the record


async def test_a_changed_file_is_compared_again(services: Services) -> None:
    from twin.training.registry import record_eval

    register(services)
    passing_evaluation(services)
    record_eval(
        services.db,
        MODEL,
        {"tokenize_check": {"ok": True, "model_sha256": "f" * 64, "at": "x"}},  # another file
    )
    checker = Checker()
    await activate(services, checker=checker)
    assert checker.calls == [MODEL]


async def test_a_mismatch_is_refused_even_by_force_and_alerts(services: Services) -> None:
    register(services)
    passing_evaluation(services)
    with pytest.raises(ActivationRefused) as refused:
        await activate(services, checker=Checker("mismatch"), force=True)
    assert refused.value.code == "tokenizer"
    assert any("differs" in line for line in refused.value.lines)
    model = get_model(services.db, MODEL)
    assert not model.active and not has_valid_check(model)
    assert model.eval["tokenize_check"]["ok"] is False
    assert model.eval["tokenize_check"]["differences"][0]["kind"] == "extra_prefix"
    with services.db.session() as session:
        categories = [row.category for row in session.scalars(select(Alert))]
    assert categories == ["style_tokenize_mismatch"]
    # and the model is unusable from now on: the selector reads this record
    assert StyleModels(services.db, pinned=MODEL).active().tokenizer_ok is False  # type: ignore[union-attr]


async def test_a_server_that_cannot_tokenize_is_a_refusal_not_a_pass(services: Services) -> None:
    register(services)
    passing_evaluation(services)
    with pytest.raises(ActivationRefused) as refused:
        await activate(services, checker=Checker("unavailable"))
    assert refused.value.code == "tokenizer_unavailable"
    assert not get_model(services.db, MODEL).active


# ------------------------------------------------------------------ disable


async def test_disabling_the_active_model_sends_the_default_back_to_deepseek(
    services: Services,
) -> None:
    register(services)
    passing_evaluation(services)
    await activate(services)
    services.runtime.set(BACKEND_FALLBACK, {"requested": "hybrid", "reason": "error"}, by="auto")
    result = disable_model(services, MODEL)
    model = get_model(services.db, MODEL)
    assert result.was_active and result.backend_reset
    assert not model.active and not model.enabled and model.gate_passed is True  # the verdict stays
    assert services.runtime.get(BACKEND_ACTIVE) == "deepseek"
    assert services.runtime.get(BACKEND_FALLBACK) is None
    assert model.eval[EVAL_ACTIVATION_LOG][-1]["action"] == "disable"
    assert StyleModels(services.db).active() is None


async def test_disabling_a_model_that_is_not_active_changes_nothing_else(
    services: Services,
) -> None:
    register(services, "r-0", active=True, gate_passed=True)
    register(services)
    services.runtime.set(BACKEND_ACTIVE, "style", by="command")
    result = disable_model(services, MODEL)
    assert not result.was_active and not result.backend_reset
    assert services.runtime.get(BACKEND_ACTIVE) == "style"
    assert get_model(services.db, "r-0-Q5_K_M").active


async def test_disabling_an_adapter_stops_asking_for_the_tunnel(services: Services) -> None:
    services.settings.style_model.mode = "vllm_completion"
    register(services, "r-lora", quant="lora", kind="adapter", active=True, gate_passed=True)
    services.runtime.set(TUNNEL_WANTED, True, by="command")
    disable_model(services, "r-lora-lora")
    assert services.runtime.get(TUNNEL_WANTED) is False


def test_disabling_an_unknown_model_is_refused(services: Services) -> None:
    with pytest.raises(ActivationRefused) as refused:
        disable_model(services, "nothing")
    assert refused.value.code == "unknown"
