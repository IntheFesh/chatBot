"""The style prompt on real data: the compact card, locked versions and ``AsOfView`` (R-TRN-013).

Synthetic data only.  Markers stand for facts and corrections: a marker that lies in a place the
prompt must not read from must never appear in it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.support.embedding import HashingBackend
from tests.support.memory import add_fact, make_memory
from tests.support.style_models import register_model
from tests.support.synth_chat import ChatSpec, build_chat
from twin.engine.dataview import LiveDataSource
from twin.engine.style_models import StyleModels
from twin.engine.style_prompt import LockedVersionError, StylePromptBuilder, StyleTurn
from twin.memory.api import AsOfSource, AsOfView, Memory
from twin.profile.builder import rebuild
from twin.profile.persona import compose
from twin.profile.persona.store import PersonaStore
from twin.services import Services
from twin.training import lf_template
from twin.training.registry import LockedVersions

PAST = datetime(2026, 8, 25, 18, 0, tzinfo=UTC)
FACT_AUTO = "养了一只叫豆包的猫"  # [自动-描述] ### 基本情况
FACT_MANUAL = "她的生日在三月三日"  # [手动] ### 事实
CORRECTION = "不要说得太书面"  # [不要这样]
STYLE_AUTO = "口头禅是嘿嘿"  # [自动-描述] ### 风格
STYLE_MANUAL = "喜欢拖长音"  # [手动] ### 风格
RULE = "几乎不用逗号"  # [自动-统计规则]


def live_card() -> str:
    description = f"### 风格\n- {STYLE_AUTO}\n\n### 基本情况\n- {FACT_AUTO}\n\n"
    return (
        compose.stats_block([RULE])
        + compose.auto_block(description)
        + compose.manual_block([STYLE_MANUAL], [FACT_MANUAL])
        + compose.dont_block([CORRECTION])
    )


def past_card(marker: str) -> str:
    description = f"### 风格\n- 口头禅是{marker}\n\n### 基本情况\n- 过去的事实{marker}\n\n"
    return (
        compose.stats_block([RULE])
        + compose.auto_block(description)
        + compose.manual_block([STYLE_MANUAL], None)
    )


@pytest.fixture
def store(services: Services) -> PersonaStore:
    return PersonaStore(services.db, services.clock)


def turns() -> list[StyleTurn]:
    return [StyleTurn("user", "在吗"), StyleTurn("assistant", "在呢"), StyleTurn("user", "你好呀")]


def test_the_compact_card_in_the_prompt_has_style_and_no_fact_and_no_correction(
    services: Services, embedder: HashingBackend, store: PersonaStore
) -> None:
    store.add_version("live", live_card(), reason="generate")
    view = LiveDataSource(services, memory=Memory(services)).view()
    prompt = StylePromptBuilder.from_services(services, memory_tokens=0).build(view, turns())
    for style in (STYLE_AUTO, STYLE_MANUAL, RULE):
        assert style in prompt.text
    for forbidden in (FACT_AUTO, FACT_MANUAL, CORRECTION):
        assert forbidden not in prompt.text
    assert prompt.meta is not None and prompt.meta.persona_scope == "live"


def test_a_registered_model_is_served_with_the_card_version_it_was_trained_with(
    services: Services, embedder: HashingBackend, store: PersonaStore
) -> None:
    store.add_version("pre_holdout", past_card("版本一"), reason="generate")
    store.add_version("pre_holdout", past_card("版本二"), reason="generate")  # now the active one
    store.add_version("live", live_card(), reason="generate")
    register_model(services, persona_version="v1")
    active = StyleModels(services.db).active()
    assert active is not None and active.versions.persona_version == "v1"
    view = LiveDataSource(services, memory=Memory(services)).view()
    builder = StylePromptBuilder.from_services(services, locked=active.versions, memory_tokens=0)
    prompt = builder.build(view, turns())
    assert "口头禅是版本一" in prompt.text and "版本二" not in prompt.text
    assert STYLE_AUTO not in prompt.text  # not the live card either
    assert "过去的事实" not in prompt.text  # the compact card has no facts
    meta = prompt.meta
    assert meta is not None and meta.locked
    assert (meta.persona_scope, meta.persona_version) == ("pre_holdout", "v1")


def test_a_card_version_that_is_not_in_the_store_cannot_be_served(
    services: Services, embedder: HashingBackend, store: PersonaStore
) -> None:
    store.add_version("pre_holdout", past_card("版本一"), reason="generate")
    locked = LockedVersions(lf_template.TEMPLATE_VERSION, "v9", "p1", "ds-1")
    view = LiveDataSource(services, memory=Memory(services)).view()
    builder = StylePromptBuilder.from_services(services, locked=locked, memory_tokens=0)
    with pytest.raises(LockedVersionError, match="v9"):
        builder.build(view, turns())


def test_the_registry_reader_finds_the_active_model_of_the_served_kind(services: Services) -> None:
    models = StyleModels(services.db, mode="llamacpp_completion")
    assert models.registered() == 0 and models.active() is None
    register_model(services, run_id="r1", kind="gguf", active=False, gate_passed=None)
    assert models.registered() == 1 and models.active() is None
    register_model(services, run_id="r2", quant="lora", kind="adapter", active=True)
    register_model(services, run_id="r3", kind="gguf", active=True, gate_passed=False)
    found = models.active()
    assert found is not None and found.run_id == "r3" and found.kind == "gguf"
    assert not found.passed_gate and found.label == "r3 Q5_K_M"
    remote = StyleModels(services.db, mode="vllm_completion").active()
    assert remote is not None and remote.run_id == "r2" and remote.passed_gate


@pytest.fixture
def world(services: Services, embedder: HashingBackend, store: PersonaStore) -> Services:
    services.settings.retrieval.model = embedder.info.model
    build_chat(services, ChatSpec(days=40))
    rebuild(services, "all")
    store.add_version("pre_holdout", past_card("过去"), reason="generate")
    store.add_version("live", live_card(), reason="generate")
    memory = make_memory(services)
    add_fact(memory, "早就知道的事：对方住在二楼", datetime(2026, 8, 10, tzinfo=UTC))
    add_fact(memory, "之后才知道的事：对方换了工作", datetime(2026, 9, 20, tzinfo=UTC))
    return services


def test_the_prompt_of_a_past_moment_reads_only_what_was_known_then(world: Services) -> None:
    """The training export and the sandbox pass ``AsOfView(t)``: nothing from after ``t``."""
    view = AsOfView(AsOfSource(world, memory=Memory(world)), PAST)
    builder = StylePromptBuilder.from_services(world, memory_tokens=600)
    prompt = builder.build(view, [StyleTurn("user", "对方住在哪里呢")])
    assert "早就知道的事" in prompt.text
    assert "之后才知道的事" not in prompt.text  # known only after the moment
    assert "口头禅是过去" in prompt.text and STYLE_AUTO not in prompt.text  # the pre-holdout card
    assert "2026年8月25日" in prompt.text  # the clock of the moment, not today's
    assert FACT_MANUAL not in prompt.text and CORRECTION not in prompt.text
    assert prompt.meta is not None and prompt.meta.persona_scope == "pre_holdout"
    assert prompt.text.endswith("<|im_start|>assistant\n")
    assert "她现在的状态：" in prompt.text  # her typical state at that time, from the routine


def test_the_live_view_and_the_past_view_make_the_same_kind_of_prompt(world: Services) -> None:
    live = LiveDataSource(world, memory=Memory(world)).view()
    past = AsOfView(AsOfSource(world, memory=Memory(world)), PAST)
    builder = StylePromptBuilder.from_services(world, memory_tokens=600)
    context = [StyleTurn("user", "对方住在哪里呢")]
    now_prompt, then_prompt = builder.build(live, context), builder.build(past, context)
    shape = ("<|im_start|>system\n", "【此刻】\n当地时间：", "<|im_end|>\n<|im_start|>user\n")
    for fragment in shape:
        assert fragment in now_prompt.text and fragment in then_prompt.text
    assert now_prompt.text.endswith(then_prompt.text[-60:])  # the same conversation part
