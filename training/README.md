# training/

AutoDL-side scripts and LLaMA-Factory configuration templates for the style model (round 13).
Nothing here runs on the Windows machine; the package that is uploaded is desensitised and
encrypted (R-PRIV-003), and everything is deleted from the instance after training.

```
training/
  llamafactory/*.yaml.j2   one template per kind of configuration (sft, dpo, eval_generate,
                           eval_loss, export); rendered per GPU profile by twin.training.yaml_render
  autodl/                  lib.sh and the scripts: setup, decrypt, train, eval, dpo, export,
                           serve_vllm, cleanup; versions.env (pinned versions, checked against
                           twin.training.versions by a test); tools/*.py (standard library only)
```

The encrypted package built by `twin train bundle` contains the scripts, and `twin train remote
upload` also puts them on the instance in plain (they are needed to decrypt and hold no data); the
instance compares the two after decrypting and stops if they differ.

## Pinned versions and what was checked on 2026-10-09

Everything the instance installs is pinned in `src/twin/training/versions.py`
(`training/autodl/versions.env` is the same list for the shell scripts).

| Item | Pinned | How it was checked |
| --- | --- | --- |
| LLaMA-Factory | **0.9.5** (PyPI, 2026-05-30, wheel sha256 `10776e9b...c6a`) | `pip download llamafactory==0.9.5 --no-deps`, wheel and sdist read |
| PyTorch | 2.9.1 from `https://download.pytorch.org/whl/cu128` (+ torchvision 0.24.1, torchaudio 2.9.1) | index listing; build script of v2.9.1 |
| transformers / peft / trl / accelerate / datasets | 4.57.6 / 0.18.1 / 0.24.0 / 1.11.0 / 4.0.0 | upper bounds LLaMA-Factory 0.9.5 accepts (`transformers<=5.6.0`, `peft<=0.18.1`, `trl<=0.24.0`, `accelerate<=1.11.0`, `datasets<=4.0.0`) |
| bitsandbytes | 0.49.2 | wheel contains `libbitsandbytes_cuda128.so`; CMake targets compute capability 120 for CUDA >= 12.8 |
| modelscope | 1.39.1 | wheel; its `download` command moved to the `modelscope-hub` package, so `setup.sh` uses `snapshot_download()` |
| llama.cpp | tag **b11177** (2026-09-25) | `conversion/qwen.py` registers `Qwen3ForCausalLM`; `llama-quantize` is a CMake target |
| vLLM | 0.26.0 in its own venv (requires `torch==2.11.0`) | PyPI metadata, its Docker build lists compute capability 12.0 |

### `qwen3_nothink` (R-TRN-011)

Source: `src/llamafactory/data/template.py` of the 0.9.5 wheel.  It is registered with the plain
`Template` class (the plain `qwen3` template is a `ReasoningTemplate`):

```
format_user      <|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n
format_assistant {{content}}<|im_end|>\n
format_system    <|im_start|>system\n{{content}}<|im_end|>\n
stop_words       ["<|im_end|>"], replace_eos=True, no default_system, no prefix
```

Pure ChatML, no `<think>` marker anywhere.  `src/twin/training/lf_template.py` holds exactly these
strings (module docstring explains the use for `StylePromptBuilder`, round 09).  Other facts read in
the same source:

* `get_template_and_fix_tokenizer(tokenizer, data_args)` returns the `Template`;
  `template.encode_multiturn(tokenizer, messages, system, tools, discarding_history_cot)` returns
  one `(prompt_ids, response_ids)` pair per turn.  Every filled slot is tokenised on its own with
  `tokenizer.encode(slot, add_special_tokens=False)`, so text that contains `<|im_end|>` becomes the
  special token, and the formatter also substitutes `{{idx}}` inside the user message.  The dataset
  loader (`dataset_dir.py`) rejects both.
* `mask_history: true`: the pairs are encoded newest first, only the last pair gets labels, and
  **a sample longer than `cutoff_len` loses its oldest turns first - but the prompt of the last turn
  is cut at its end**, i.e. the assistant opener (and the beginning of the reply) is lost.  The
  exporter (round 13b) must therefore fit every sample in 2048 tokens itself by dropping the oldest
  context turns; it must never leave that to LLaMA-Factory.
* ShareGPT (`dataset_info.json`, `formatting: sharegpt`): an optional `system` column (or a first
  message from `system`), then `human`/`gpt` strictly alternating from `human`, an even number of
  messages for SFT.  A ranking dataset (`ranking: true`) has an odd number of messages and the
  `chosen` / `rejected` replies as `{"from": "gpt", "value": ...}` objects.
* `flash_attn` accepts `auto | disabled | sdpa | fa2 | fa3`; the configs use `sdpa`.
* `stage: dpo` defaults: `pref_beta: 0.1`, `pref_loss: sigmoid`; the official example uses a
  learning rate of 5e-6.  With a LoRA adapter the reference model is the same model with the adapter
  switched off.  `adapter_name_or_path` + `do_train` continues that adapter.
* Early stopping is `early_stopping_steps` (a patience counted in evaluations; needs
  `load_best_model_at_end`).  Training resumes by itself from the last checkpoint of `output_dir`
  unless `overwrite_output_dir` is set.  `do_predict` + `predict_with_generate` writes
  `generated_predictions.jsonl` with `prompt`, `predict` and `label`.
* `llamafactory-cli export`: raises `Please merge adapters before quantizing the model` when an
  adapter and `export_quantization_bit` are given, and `Cannot merge adapters to a quantized model`
  when the base is loaded quantised.  The merge must therefore load the bf16 base without
  `quantization_bit`; `export_device: cpu` for the 4-bit profile.

### Blackwell (RTX 5090, RTX PRO 6000: compute capability 12.0, `sm_120`)

* PyTorch >= 2.7 with CUDA 12.8 wheels supports Blackwell (release blog of 2.7); the build script of
  v2.9.1 lists `12.0` for CUDA 12.8.  The `cu128` index carries torch up to 2.11.0; 2.12 and later are
  only on `cu129`/`cu130`.  The default PyPI wheel of torch 2.11.0 is a **CUDA 13** build (it depends on
  `nvidia-*-cu13`), which this project does not use - `setup.sh` installs torch from the `cu128`
  index first, so no later `pip install` replaces it.
* `setup.sh` checks `torch >= 2.7`, `torch.version.cuda` in the 12.8 series and `sm_120` in
  `torch.cuda.get_arch_list()` when the GPU reports compute capability 12.0, and refuses 13.x always.

### AutoDL

* Data disk `/root/autodl-tmp` (50 GB free, expandable on hosts that allow it; kept when the system is
  reset; the instance and its data are erased after 15 days of continuous shutdown).
* `source /etc/network_turbo` accelerates github.com, githubusercontent.com, githubassets.com and
  huggingface.co only; undo it with `unset http_proxy https_proxy`.  Provided without a stability
  promise, for academic use.
* SSH: `ssh -p <port> root@<host>`, password login; the page recommends tmux/screen for long jobs
  (`remote_job.py` detaches jobs itself, so tmux is not needed).
* ModelScope: `Qwen/Qwen3-8B` (16.4 GB), `Qwen/Qwen3-14B` (29.55 GB), `Qwen/Qwen3-32B` (65.54 GB) exist
  with `safetensors` shards (sizes from the repository listing).

### llama.cpp

The converter is a package now: clone the whole tree (`convert_hf_to_gguf.py` imports
`conversion/`).  `conversion/qwen.py` registers `Qwen3ForCausalLM`/`Qwen3Model`.
`python convert_hf_to_gguf.py <dir> --outfile x.gguf --outtype f16` and
`build/bin/llama-quantize x.gguf out.gguf Q4_K_M` (also `Q5_K_M`, `Q8_0`).  Its requirements pin
`torch==2.11.0` (CPU index) and `transformers==4.57.6`, so `export.sh` creates a separate virtual
environment for the converter and builds only the `llama-quantize` target on the CPU.

### vLLM

`vllm serve <model> --enable-lora --lora-modules name=<adapter dir> --max-lora-rank N`; a request
selects the adapter with its `model` field; `/v1/completions` and `/tokenize` exist.

### Could not be checked

* github.com web pages answer 403 in the development sandbox (`raw.githubusercontent.com` and `git`
  work), so the GitHub **release download** of the vLLM `+cu128` wheel could not be tried;
  `serve_vllm.sh` falls back to `pip install vllm==0.26.0` with the `cu128` extra index.
* Neither bitsandbytes 4-bit nor vLLM 0.26.0 was run on a real RTX 5090 or RTX PRO 6000 (no GPU in
  the sandbox); the CUDA 12.8 binaries and the `12.0` architecture entries were read from the sources
  above.  The first real run will show whether they work; `setup.sh` and `serve_vllm.sh` print the
  versions they ended up with.
* The request fields of vLLM's `/tokenize` are not listed on the documentation page; round 14 builds the
  tokenizer comparison against the actual responses.
* AutoDL's image list for the two GPUs and the size of the system disk (their pages did not state it).

## Contracts used by round 13b and later rounds

* **Dataset directory** (`twin.training.dataset_dir`): `dataset_meta.json`, `sft_train.jsonl`,
  `sft_val.jsonl`, `sft_test.jsonl`, optional `dpo_train.jsonl`; written with `write_dataset_dir`.
  The exporter must set `redacted=True` only after the residual PII scan.
* **Template check** (`setup.sh <profile> verify`): needs `python -m twin.training.parity_check
  --model-dir <base model dir> --cases <workdir>/data/parity_cases.jsonl` in the package (exit code 0 =
  token-identical).  The package carries `pylib/twin/training/<module>.py` for the modules listed in
  `twin.training.bundle.PYLIB_MODULES` (`lf_template` now; round 13b adds `parity_check`) and expects the
  cases file in `data/`.  If the module is missing, `verify` stops with an error; it does not skip.
* **Artifact manifest** (`artifacts/manifest.json`, schema 1): `run_id`, `profile`, `base_model`,
  `template`, `template_version`, `persona_version`, `profile_version`, `dataset_version`,
  `llamafactory_version`, `llama_cpp_tag`, `models[]` (`quant`, `kind`, `path`), `files[]` (`path`,
  `sha256`, `size`), `metrics`.  `twin model register` locks the four versions from it.
