"""R-TRN-001: the four profiles, their requirements and the rendered LLaMA-Factory YAML."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import yaml

from twin.training import lf_template, versions
from twin.training.layout import RemoteLayout
from twin.training.profiles import (
    DISK_OVERHEAD_GB,
    DISK_PEAK_FACTOR,
    PROFILES,
    QUANT_FACTORS,
    SPEC_DISK_USABLE_SHARE,
    ProfileError,
    epochs_for,
    get_profile,
    plan_steps,
    profile_env,
    profile_names,
)
from twin.training.yaml_render import (
    GENERATION_TOP_K,
    ConfigKind,
    dataset_info,
    render_all,
    render_config,
    training_assets_dir,
)

SNAPSHOTS = Path(__file__).resolve().parents[1] / "fixtures" / "training" / "yaml"
ROOT = Path(__file__).resolve().parents[2]


def test_there_are_four_profiles_with_the_models_of_the_spec_table() -> None:
    assert profile_names() == ("5090-8b", "5090-14b", "pro6000-14b", "pro6000-32b")
    assert {name: p.base_model for name, p in PROFILES.items()} == {
        "5090-8b": "Qwen/Qwen3-8B",
        "5090-14b": "Qwen/Qwen3-14B",
        "pro6000-14b": "Qwen/Qwen3-14B",
        "pro6000-32b": "Qwen/Qwen3-32B",
    }
    assert [(p.lora_rank, p.lora_alpha) for p in PROFILES.values()] == [
        (32, 64),
        (32, 64),
        (32, 64),
        (16, 32),
    ]
    assert [p.quantization_bit for p in PROFILES.values()] == [None, 4, None, None]
    assert [p.method for p in PROFILES.values()] == ["lora", "qlora", "lora", "lora"]
    assert all(p.effective_batch_size == 16 for p in PROFILES.values())


def test_an_unknown_profile_is_refused_with_the_valid_names() -> None:
    with pytest.raises(ProfileError, match="5090-8b"):
        get_profile("4090-7b")


def test_the_disk_requirement_follows_the_base_size_and_matches_the_round_numbers_of_the_spec() -> (
    None
):
    required = {name: p.required_disk_gb for name, p in PROFILES.items()}
    assert required == {"5090-8b": 68, "5090-14b": 110, "pro6000-14b": 110, "pro6000-32b": 225}
    for profile in PROFILES.values():
        # a disk expanded to the number in R-TRN-009 always passes, and the number is close
        assert profile.required_disk_gb <= profile.spec_disk_gb
        assert profile.required_disk_gb >= profile.spec_disk_gb * 0.9


def test_a_fresh_disk_of_the_size_the_user_is_told_covers_the_requirement() -> None:
    """R18-E4 (D-811): 98 % of a disk of ``spec_disk_gb`` decimal GB is still enough.

    The 14B profiles told the user to expand the disk to exactly the requirement (110 GB), which a
    file system never offers in full; they now say 115 GB - still "about 110" of R-TRN-009.
    """
    assert SPEC_DISK_USABLE_SHARE == 0.98
    for name, profile in PROFILES.items():
        assert profile.spec_disk_gb * SPEC_DISK_USABLE_SHARE >= profile.required_disk_gb, name
    spec = {name: profile.spec_disk_gb for name, profile in PROFILES.items()}
    assert spec == {"5090-8b": 70, "5090-14b": 115, "pro6000-14b": 115, "pro6000-32b": 230}


def test_the_cpu_merge_of_the_qlora_profile_needs_the_whole_bf16_model_in_memory() -> None:
    qlora = PROFILES["5090-14b"]
    assert qlora.export_device == "cpu"
    assert qlora.required_ram_gb > qlora.base_gb
    for name in ("5090-8b", "pro6000-14b", "pro6000-32b"):
        assert PROFILES[name].export_device == "auto"
        assert PROFILES[name].required_ram_gb < PROFILES[name].base_gb * 1.3 + 4


@pytest.mark.parametrize(
    ("samples", "epochs"), [(0, 3), (1, 3), (19_999, 3), (20_000, 2), (150_000, 2)]
)
def test_three_epochs_below_20000_samples_and_two_from_there(samples: int, epochs: int) -> None:
    assert epochs_for(samples) == epochs


def test_the_step_plan_evaluates_about_four_times_per_epoch() -> None:
    plan = plan_steps(12_000, PROFILES["5090-8b"], 3)
    assert plan.steps_per_epoch == 750
    assert plan.total_steps == 2250
    assert plan.eval_steps == 188
    tiny = plan_steps(10, PROFILES["5090-8b"], 3)
    assert tiny.steps_per_epoch == 1 and tiny.eval_steps == 1 and tiny.logging_steps == 1
    assert plan_steps(10_000_000, PROFILES["5090-8b"], 2).eval_steps == 500
    with pytest.raises(ProfileError):
        plan_steps(0, PROFILES["5090-8b"], 3)


@pytest.mark.parametrize("name", list(PROFILES))
def test_the_rendered_yaml_of_every_profile_matches_its_snapshot(name: str) -> None:
    profile = PROFILES[name]
    for kind in ConfigKind:
        rendered = render_config(kind, profile, train_samples=1200)
        expected = (SNAPSHOTS / f"{name}.{kind.value}.yaml").read_text(encoding="utf-8")
        assert rendered == expected, f"{name} {kind.value} changed; review it, then regenerate"


@pytest.mark.parametrize("name", list(PROFILES))
def test_the_evaluation_samples_like_the_live_clients_without_top_k(name: str) -> None:
    """R18-E8 (D-815): LLaMA-Factory 0.9.5 generates with ``top_k`` 50 unless told otherwise
    (``hparams/generating_args.py``); the two live clients switch top-k off (D-120), so the
    replies the evaluation compares were sampled from another distribution.  ``top_k: 0`` is
    "off" for transformers' ``generate`` as it is for llama.cpp."""
    config = yaml.safe_load(
        render_config(ConfigKind.EVAL_GENERATE, PROFILES[name], train_samples=1200)
    )
    assert config["do_sample"] is True and config["top_k"] == 0
    assert config["top_k"] == GENERATION_TOP_K == 0
    assert (config["temperature"], config["top_p"]) == (0.7, 0.9)
    assert "repetition_penalty" not in config  # LLaMA-Factory's default is 1.0: off


@pytest.mark.parametrize("name", list(PROFILES))
def test_the_sft_yaml_has_the_training_rules_of_the_spec(name: str) -> None:
    profile = PROFILES[name]
    config = yaml.safe_load(render_config(ConfigKind.SFT, profile, train_samples=1200))
    assert config["stage"] == "sft" and config["finetuning_type"] == "lora"
    assert config["lora_target"] == "all"
    assert config["template"] == "qwen3_nothink"
    assert config["mask_history"] is True
    assert config["cutoff_len"] == 2048
    assert config["learning_rate"] == 1e-4
    assert config["lr_scheduler_type"] == "cosine"
    assert config["warmup_ratio"] == 0.05
    assert config["bf16"] is True and config["gradient_checkpointing"] is True
    assert config["flash_attn"] == "sdpa"
    assert config["report_to"] == "none"
    assert config["num_train_epochs"] == 3.0
    assert config["lora_rank"] == profile.lora_rank and config["lora_alpha"] == profile.lora_alpha
    # the independent validation file instead of val_size, and early stopping after two evaluations
    assert config["eval_dataset"] == "twin_sft_val" and "val_size" not in config
    assert config["eval_strategy"] == "steps" and config["eval_steps"] == config["save_steps"]
    assert config["early_stopping_steps"] == 2 and config["load_best_model_at_end"] is True
    assert config["metric_for_best_model"] == "eval_loss" and config["greater_is_better"] is False
    assert config["overwrite_output_dir"] is False and config["resume_from_checkpoint"] is None
    assert config["model_name_or_path"] == f"/root/autodl-tmp/models/{profile.base_model}"
    if profile.quantization_bit:
        assert config["quantization_bit"] == 4 and config["quantization_method"] == "bnb"
    else:
        assert "quantization_bit" not in config


def test_the_sft_yaml_uses_two_epochs_from_20000_samples() -> None:
    config = yaml.safe_load(
        render_config(ConfigKind.SFT, PROFILES["5090-8b"], train_samples=20_000)
    )
    assert config["num_train_epochs"] == 2.0


@pytest.mark.parametrize("name", list(PROFILES))
def test_no_export_yaml_sets_quantization_and_the_qlora_profile_merges_on_the_cpu(
    name: str,
) -> None:
    config = yaml.safe_load(render_config(ConfigKind.EXPORT, PROFILES[name], train_samples=1200))
    assert "quantization_bit" not in config and "quantization_method" not in config
    assert config["model_name_or_path"].endswith(PROFILES[name].base_model)
    assert config["export_device"] == ("cpu" if name == "5090-14b" else "auto")
    assert config["template"] == "qwen3_nothink"
    assert config["export_dir"].endswith("/output/merged")


def test_the_qlora_profile_trains_on_a_4bit_base_in_dpo_and_evaluation_too() -> None:
    profile = PROFILES["5090-14b"]
    for kind in (ConfigKind.DPO, ConfigKind.EVAL_GENERATE, ConfigKind.EVAL_LOSS):
        config = yaml.safe_load(render_config(kind, profile, train_samples=1200))
        assert config["quantization_bit"] == 4, kind


def test_dpo_trains_with_the_recommended_values_on_the_sft_adapter() -> None:
    config = yaml.safe_load(render_config(ConfigKind.DPO, PROFILES["5090-8b"], train_samples=1200))
    assert config["stage"] == "dpo"
    assert config["adapter_name_or_path"].endswith("/output/sft")
    assert config["output_dir"].endswith("/output/dpo")
    assert config["pref_beta"] == 0.1 and config["pref_loss"] == "sigmoid"
    assert config["learning_rate"] == 5e-6
    assert config["dataset"] == "twin_dpo_train"


REF_KEYS = ("ref_model", "ref_model_adapters", "ref_model_quantization_bit")


@pytest.mark.parametrize("name", ["5090-8b", "pro6000-14b", "pro6000-32b"])
def test_dpo_of_a_bf16_profile_makes_a_new_adapter_on_the_merged_sft_model(name: str) -> None:
    """R17-49d (D-650): ``create_new_adapter`` merges SFT in memory; no ``ref_model`` is set.

    LLaMA-Factory 0.9.5 (``model/adapter.py``): with ``is_trainable`` and ``create_new_adapter``
    every adapter of ``adapter_name_or_path`` is merged and a new LoRA is created; without a
    ``ref_model`` the trainer switches that new adapter off for the reference pass
    (``train/dpo/trainer.py``), so the reference is the merged SFT model - the old setup continued
    the SFT adapter and the reference was the bare base.
    """
    profile = PROFILES[name]
    assert profile.dpo_mode == "layered" and profile.dpo_reference_quantization_bit is None
    config = yaml.safe_load(render_config(ConfigKind.DPO, profile, train_samples=1200))
    assert config["adapter_name_or_path"].endswith("/output/sft")
    assert "," not in config["adapter_name_or_path"]
    assert config["create_new_adapter"] is True
    assert not any(key in config for key in REF_KEYS)
    assert "quantization_bit" not in config
    assert config["output_dir"].endswith("/output/dpo")
    assert config["lora_rank"] == profile.lora_rank and config["lora_alpha"] == profile.lora_alpha
    assert (
        config["per_device_train_batch_size"] == 1 and config["gradient_accumulation_steps"] == 16
    )


def test_dpo_of_the_4bit_profile_holds_the_sft_model_as_an_explicit_4bit_reference() -> None:
    """R17-49d (D-650): a quantised model cannot take a new adapter, so SFT is continued.

    LLaMA-Factory 0.9.5 (``hparams/parser.py``) raises "Cannot create new adapter upon a quantized
    model" for ``create_new_adapter`` there, so the key must be absent; the reference is given by
    ``ref_model`` (the base), ``ref_model_adapters`` (the SFT adapter) and
    ``ref_model_quantization_bit`` (``train/trainer_utils.py::create_ref_model``).
    """
    profile = PROFILES["5090-14b"]
    assert profile.dpo_mode == "continued" and profile.dpo_reference_quantization_bit == 4
    config = yaml.safe_load(render_config(ConfigKind.DPO, profile, train_samples=1200))
    assert "create_new_adapter" not in config
    assert config["adapter_name_or_path"].endswith("/output/sft")
    assert config["quantization_bit"] == 4 and config["quantization_method"] == "bnb"
    assert config["ref_model"] == config["model_name_or_path"] == profile.model_dir
    assert config["ref_model_adapters"] == config["adapter_name_or_path"]
    assert config["ref_model_quantization_bit"] == 4
    assert config["output_dir"].endswith("/output/dpo")


@pytest.mark.parametrize("name", list(PROFILES))
def test_the_dpo_modes_follow_whether_the_base_is_quantised(name: str) -> None:
    profile = PROFILES[name]
    assert profile.dpo_mode == ("continued" if profile.quantization_bit else "layered")
    assert f"TWIN_DPO_MODE={profile.dpo_mode}\n" in profile_env(profile)


def test_evaluation_generates_one_reply_per_test_context_with_fixed_sampling() -> None:
    config = yaml.safe_load(
        render_config(ConfigKind.EVAL_GENERATE, PROFILES["5090-8b"], train_samples=1200)
    )
    assert config["do_predict"] is True and config["predict_with_generate"] is True
    assert config["eval_dataset"] == "twin_sft_test"
    assert config["temperature"] == 0.7 and config["top_p"] == 0.9 and config["seed"] == 42
    loss = yaml.safe_load(
        render_config(ConfigKind.EVAL_LOSS, PROFILES["5090-8b"], train_samples=1200)
    )
    assert loss["do_eval"] is True and loss["eval_dataset"] == "twin_sft_val"


def test_the_work_directory_can_be_changed() -> None:
    config = yaml.safe_load(
        render_config(
            ConfigKind.SFT,
            PROFILES["5090-8b"],
            train_samples=100,
            layout=RemoteLayout("/mnt/work/twin/"),
        )
    )
    assert config["dataset_dir"] == "/mnt/work/twin/data"
    assert config["output_dir"] == "/mnt/work/twin/output/sft"


def test_render_all_adds_the_dpo_file_only_when_asked() -> None:
    profile = PROFILES["5090-8b"]
    plain = render_all(profile, train_samples=100)
    assert sorted(plain) == ["eval_generate.yaml", "eval_loss.yaml", "export.yaml", "sft.yaml"]
    assert "dpo.yaml" in render_all(profile, train_samples=100, with_dpo=True)


def test_dataset_info_describes_sharegpt_files_with_the_roles_of_llamafactory() -> None:
    info = json.loads(dataset_info(with_dpo=True))
    assert set(info) == {"twin_sft_train", "twin_sft_val", "twin_sft_test", "twin_dpo_train"}
    train = info["twin_sft_train"]
    assert train["formatting"] == "sharegpt" and train["file_name"] == "sft_train.jsonl"
    assert train["columns"] == {"messages": "conversations", "system": "system"}
    assert train["tags"]["user_tag"] == "human" and train["tags"]["assistant_tag"] == "gpt"
    assert info["twin_dpo_train"]["ranking"] is True
    assert info["twin_dpo_train"]["columns"]["chosen"] == "chosen"
    assert "twin_dpo_train" not in json.loads(dataset_info(with_dpo=False))


def test_the_profile_env_holds_plain_words_and_the_computed_requirements() -> None:
    lines = dict(line.split("=", 1) for line in profile_env(PROFILES["5090-14b"]).splitlines())
    assert lines["TWIN_PROFILE"] == "5090-14b"
    # R18-E4 (D-811): the disk the user is told to expand to leaves room above the requirement
    assert lines["TWIN_REQUIRED_DISK_GB"] == "110" and lines["TWIN_SPEC_DISK_GB"] == "115"
    assert lines["TWIN_QUANT_BIT"] == "4" and lines["TWIN_EXPORT_DEVICE"] == "cpu"
    assert lines["TWIN_BASE_GB"] == "29.6" and lines["TWIN_BASE_GB_CEIL"] == "30"
    assert lines["TWIN_DPO_MIN_PAIRS"] == "200"
    assert all(" " not in value and "'" not in value for value in lines.values())
    custom = profile_env(PROFILES["5090-8b"], dpo_min_pairs=50)
    assert "TWIN_DPO_MIN_PAIRS=50\n" in custom


def test_the_committed_versions_file_is_the_one_python_generates() -> None:
    committed = (ROOT / "training" / "autodl" / "versions.env").read_text(encoding="utf-8")
    assert committed == versions.versions_env()
    assert f"TWIN_LLAMAFACTORY={lf_template.LLAMAFACTORY_VERSION}\n" in committed
    assert "cu128" in versions.TORCH_INDEX_URL and "cu13" not in versions.TORCH_INDEX_URL


def test_the_disk_peak_factor_is_the_sum_of_what_is_on_the_disk_at_the_last_quantization() -> None:
    """R17-8 (D-634): the base, the F16 GGUF and the three quantizations, and nothing else.

    The quantizer writes straight into ``artifacts/gguf``, so there is no copy of the quantized
    files in the sum.  The numbers the user is told stay what R-TRN-009 says: 8B 70 GB, 14B 110 GB,
    32B 230 GB (the computed need is not above them).
    """
    assert set(QUANT_FACTORS) == set(versions.GGUF_QUANTS)
    assert 1 + 1 + sum(QUANT_FACTORS.values()) == DISK_PEAK_FACTOR
    assert pytest.approx(3.20) == DISK_PEAK_FACTOR
    needs = {name: profile.required_disk_gb for name, profile in PROFILES.items()}
    assert needs == {"5090-8b": 68, "5090-14b": 110, "pro6000-14b": 110, "pro6000-32b": 225}
    for name, profile in PROFILES.items():
        peak = profile.base_gb * (1 + 1 + sum(QUANT_FACTORS.values())) + DISK_OVERHEAD_GB
        assert profile.required_disk_gb == math.ceil(peak), name
        assert profile.required_disk_gb <= profile.spec_disk_gb, name


def test_the_packages_llamafactory_checks_to_generate_replies_are_pinned() -> None:
    """R17-1 (D-631): jieba, nltk and rouge-chinese are mandatory for ``predict_with_generate``."""
    lines = dict(line.split("=", 1) for line in versions.versions_env().splitlines())
    pins = {
        "TWIN_JIEBA": versions.JIEBA,
        "TWIN_NLTK": versions.NLTK,
        "TWIN_ROUGE_CHINESE": versions.ROUGE_CHINESE,
    }
    for key, pin in pins.items():
        assert lines[key] == pin
        assert re.fullmatch(r"\d+(\.\d+)+", pin), f"{key} must be an exact version, not {pin!r}"


def test_the_assets_directory_is_found_next_to_the_sources() -> None:
    assert (training_assets_dir() / "llamafactory" / "sft.yaml.j2").is_file()
    assert (training_assets_dir() / "autodl" / "setup.sh").is_file()
