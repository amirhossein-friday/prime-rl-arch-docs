# Config system — prime-rl @ b944873

> Scope: how configs are parsed (pydantic-config `@`-composition + CLI), how `RLConfig` resolves and splits into per-process configs, every cross-component propagation/validation rule, dry-run, and how to add a field end to end.
> Files read in full: `deps/pydantic-config/src/pydantic_config/cli.py` (1730), `deps/pydantic-config/src/pydantic_config/__init__.py` (5); `packages/prime-rl-configs/src/prime_rl/configs/{rl.py (869), shared.py (255), env_server.py (45), inference.py (664), orchestrator.py (803), trainer.py (828), sft.py (618), eval.py (167), monitors.py (71)}`, `configs/algorithm.py:40-99` (FrozenModelConfig / SamplingConfig); `packages/prime-rl-configs/src/prime_rl/utils/{config.py (56), validation.py (318), parsers.py (60)}`; `packages/prime-rl-configs/pyproject.toml`; entrypoints `src/prime_rl/entrypoints/*.py` (see `01`). Tests read: `tests/unit/test_configs.py` (958), selected `deps/pydantic-config/tests/test_cli.py` (683-1017). Docs cross-checked: `docs/configuration.md`, `skills/configs/SKILL.md`, `skills/training/start-run/SKILL.md`.
> Related docs: `01-deployment-topology-and-launch.md` (what the launcher does with the resolved configs), `03-orchestrator.md`, `05-trainer.md`, `06-inference-and-transports.md` (semantics of the fields themselves).

---

## 1. Mental model

Every entrypoint is `main(): run(cli(SomeConfig))`. `cli()` (from the vendored `pydantic-config`, `deps/pydantic-config/src/pydantic_config/cli.py:1592-1730`) turns `argv` into **one nested dict** — defaults ⊂ `@` config files ⊂ CLI flags — and calls `SomeConfig.model_validate(dict)` **exactly once**. All "smartness" (defaults derived from other fields, cross-component consistency, GPU/DP auto-fill, run naming) lives in pydantic `model_validator`s on the config classes, not in the parser. The parser is deliberately dumb: it maps kebab-case dotted flags to snake_case nested keys, and anything it does not recognize is passed through raw so validators (or `extra="forbid"`) can decide.

The configs live in a separate slim distribution, `prime-rl-configs` (`packages/prime-rl-configs/pyproject.toml`: depends only on pydantic, prime-pydantic-config, renderers, verifiers, tomli, tomli-w — "no GPU/ML deps"), sharing the implicit namespace packages `prime_rl` and `prime_rl.utils` with the main `prime-rl` wheel (neither has an `__init__.py`). So `prime_rl.utils.config`, `.validation`, `.parsers` come from the configs wheel while `prime_rl.utils.pathing`, `.process` come from the main one.

`RLConfig` is a **composite**: it embeds complete `TrainerConfig`, `OrchestratorConfig`, optional `InferenceConfig`, plus a set of *shared* top-level blocks (`model`, `tokenizer`, `ckpt`, `monitors`, `log`, `seq_len`, `max_steps`, `weight_broadcast`, `rollout_transport`, `deployment`, `slurm`, …). Resolution is two-phase: a `mode="before"` validator copies shared values into the raw sub-dicts (so each sub-config's own validators see them), then ~20 `mode="after"` validators cross-wire and cross-check the constructed sub-configs. The launcher then dumps each sub-config to **JSON** and every child process re-parses its JSON through the same `cli()` — so the resolved JSON is the inter-process contract, and every validator must be a fixed point on its own output.

## 2. Where it runs

Config parsing runs **in every process**: once in the launcher on the user's TOML+CLI, then again in each child on its resolved JSON (`inference @ inference.json`, `orchestrator @ orchestrator.json`, `torchrun ... -m prime_rl.trainer.rl.train @ trainer.json`, `env-server @ envs/<split>/<name>.json`), plus once more per multi-node rank with extra CLI overrides injected by the sbatch template (e.g. `--server.port`, `--model.client.base-url`, `--weight_broadcast.host`; see `01` §3.4). `default_factory` values that read the environment (`LogConfig.level ← $PRIME_LOG_LEVEL`, `vf_level ← $PRIME_VF_LOG_LEVEL`, `output_dir ← $PRL_OUTPUT_DIR`; `shared.py:200-204`, `utils/config.py:10-12`) are therefore evaluated in the launcher and frozen into the JSON; the child's environment no longer affects them.

Lifecycle: parse → validate → (launcher) mutate → dump JSON → (child) parse → validate. Errors are raised as `ConfigFileError` and, when parsing `sys.argv`, printed as a boxed message with CLI-flag-flavoured locations and "did you mean" hints, then `sys.exit(1)` (`cli.py:1716-1730`, `225-326`, `355-388`).

## 3. Mechanics

### 3.1 `cli()` pipeline (`cli.py:1592-1730`)

1. Strip reserved flags `--plain` / `--no-wide` (colour / width; also `$PYDANTIC_CONFIG_PLAIN`, `$PYDANTIC_CONFIG_WIDE`) (`:1643-1654`).
2. **`_process_args`** (`:438-496`) extracts config-file references:
   - root `@ path` (the `@` must be its own token) → loaded and deep-merged left-to-right into `root_config`;
   - nested `--key @ path` or `--key @path` → `nested_configs[key] = loaded` (a plain dict assignment — a second `--key @ other` for the same key **replaces** the first, it is not merged).
   Files are `.toml` (tomli), `.json`, `.yaml/.yml` (pyyaml, optional) by extension (`:391-424`). Then every nested file is `_nest_config`ed under its dotted key and deep-merged **after all root files**, regardless of its position on the command line (`:1658-1662`).
3. **`_expand_short_flags`**: single-letter `validation_alias` entries become `-x` synonyms (`:1094-1149`); e.g. `EvalConfig.model` accepts `-m`, `num_examples` `-n`, `group_size` `-r` (`configs/eval.py:54,65,68`).
4. **`_expand_bare_optional_flags`** (`:1160-1241`) for every "optional path" — a field typed `Optional[BaseModel]` or a multi-model union, found recursively **only through plain `BaseModel` fields** (`_find_optional_model_paths`, `:538-554`):
   - `--x` (bare) → `{x: {}}` (enable with defaults);
   - `--x None` / `--no-x` → `{x: "None"}` (disable);
   - `--no-x.field` → `{x: {field: False}}`;
   - `--x.a.b V` → `{x: {a: {b: V}}}` with `V` taken raw (JSON-parsed if it starts with `{`/`[`), or `True` if no value follows.
   Everything *below* an optional/union path is handled by this raw rule — no bool/list/type awareness.
5. **`_extract_json_value_args`** (`:1025-1091`): for fields typed `dict`/`list` (or `Optional[...]` of them) reachable through plain models, a value starting with `{`/`[` is `json.loads`ed.
6. `--help`/`-h` → render panels from `model_fields` (PEP-224 docstrings or `Field(description=)`, one panel per sub-model / optional / union variant) and `exit(0)` (`:984-1019,1684-1687`).
7. **`_parse_cli_to_dict`** (`:1349-1471`) for the remaining tokens against the leaf map built by `_build_field_meta_map` (`:1273-1325`; recursion again only through plain `BaseModel` fields, aliases included):
   - `--name value` ≡ `--name=value`;
   - bool leaves: bare → `True`, `--no-name` → `False`, or an explicit `true|True|1|false|False|0`;
   - list leaves: greedily consume following tokens until the next `--…` token (a single leading `-`, e.g. `-1e-3`, is a value); bare → `[]`;
   - interior (plain-model) path → error "is a config group, not a leaf field";
   - **unknown flag → stored raw under its snake_case path** (value = next token, or `True`) so `mode="before"` validators can remap legacy keys; otherwise `extra="forbid"` rejects it later (`:1411-1429`).
   Leftover non-flag tokens → `ConfigFileError("Unrecognized arguments: …")` (`:1696-1699`).
8. **Compose & validate once**: `merged = deep_merge(deep_merge(default.model_dump(exclude_unset=True), toml_dict), cli_overrides)`; `_normalize_alias_keys` collapses alias spellings (per union variant when a `type` tag is present) so TOML `seed` + CLI `random_seed` don't both survive (`:1474-1573,1709-1717`); `cls.model_validate(merged)`. Because validation happens once on the merged dict, `model_fields_set` faithfully records which fields the user wrote anywhere — validators rely on this for "explicit wins" logic.

`_deep_merge` merges dicts recursively and **replaces** everything else (lists, scalars) (`:427-435`). So TOML overlays replace lists wholesale; a CLI JSON dict merges into a TOML dict (test `test_dict_field_via_json_cli_with_toml`, `tests/test_cli.py:989-1000`).

### 3.2 `BaseConfig` (`cli.py:94-144`)

`model_config = ConfigDict(extra="forbid")` — no `validate_assignment`. Three `mode="before"` validators run on every model level:

| Validator | Effect |
|---|---|
| `_none_str_to_none` | any field value exactly `"None"` → `None` (TOML has no null) — applies to *any* field, including strings, but not to elements inside list/dict values |
| `_coerce_dict_str_values` | for `dict` fields whose values are **all** strings (CLI signature), coerce each to bool/int/float (`_coerce_str_value`, `:40-74`); skipped for `dict[K, str]` fields (e.g. `EnvVars`) |
| `_default_discriminator_types` | if a field's value is a dict without `type` and the field **default** is a `BaseModel` instance with a `type`, inject that default `type` |

pydantic runs `before` validators in reverse definition order, subclass before parent (so a subclass's `mode="before"` sees `"None"` strings not yet converted — `utils/validation.py:161-165` relies on this) and `after` validators in definition order (parent first). All prime-rl config classes subclass this `BaseConfig` (re-exported by `prime_rl/utils/config.py:6-7`), except `VllmConfig` which sets `extra="allow"` (`configs/inference.py:57`).

### 3.3 Semantics cheat-sheet

| Want | TOML | CLI |
|---|---|---|
| nested scalar | `[trainer.optim]` `lr = 1e-5` | `--trainer.optim.lr 1e-5` |
| compose | — | `@ base.toml @ overlay.toml` (L→R); `--trainer @ t.toml` (nested) |
| null | `"None"` | `--x None` |
| enable optional sub-config with defaults | empty `[ckpt]` | `--ckpt` |
| disable optional sub-config | `ckpt = "None"` | `--no-ckpt` / `--ckpt None`. Verified exceptions: Annotated-union optionals are leaves, so `--no-weight-broadcast` is rejected ("Extra inputs") while `--weight-broadcast None` works (= auto default); `--rollout-transport None` propagates the `"None"` string into both sub-configs and fails; under `inference.*` use `--inference.router None` (`--no-inference.router` becomes `router=False` → error) |
| enable + set | `[model.compile]` `fullgraph = true` | `--model.compile.fullgraph` |
| pick union variant | `type = "muon"` | `--trainer.optim.type muon` |
| list | `x = ["a","b"]` | `--x a b` or `--x '["a","b"]'` (leaf reachable via plain models only; below an optional/union use JSON) |
| dict | table | `--x '{"k": 1}'` (deep-merges) |
| list-of-tables item | `[[orchestrator.train.source]]` | only whole-list JSON `--orchestrator.train.source '[...]'` — no index paths |
| env var in a value | not supported — no `${VAR}` interpolation anywhere in the parser or configs; use the shell | shell expansion |

Env-var-*named* fields are resolved at runtime by consumers, not by the parser: `ClientConfig.api_key_var` (default `VLLM_API_KEY`) and `headers_from_env` (`shared.py:180-187`).

### 3.4 `RLConfig` object graph (`configs/rl.py:232-307`)

```
RLConfig
├─ trainer: TrainerConfig                        (configs/trainer.py:671-828)
│   model{name, seq_len=2048, lora?, cp, ep, dp_replicate, impl, ...} tokenizer{name?, trust_remote_code?, chat_template?}
│   data{fake?} loss(ipo|icepop|custom) optim(adamw|sgd|muon|sign_sgd) scheduler ckpt? resume?
│   weight_broadcast(filesystem*|nccl|nixl) rollout_transport(zmq*|filesystem) log: TrainerLogConfig{ranks_filter=[0]}
│   monitors{wandb?, file*} output_dir max_steps? enable_router_replay metrics_server? heartbeat? env_vars
├─ orchestrator: OrchestratorConfig              (configs/orchestrator.py:518-803)
│   algo(grpo*|...) model{name, lora?, client: ClientConfig{base_url, admin_base_url?, dynamo?, ...}}
│   train{source: [TrainSourceConfig{env, serve, name?, sampling, ratio, group_size, algo?, curriculum?}], sampling}
│   tokenizer renderer(auto*) eval?{source:[OnlineEvalSourceConfig], interval,...} log monitors{wandb?, file*, prime?}
│   ckpt? resume? weight_broadcast rollout_transport output_dir batch_size|token_batch_size concurrency group_size
│   seq_len=2048 num_train_workers pad_to_multiple_of max_steps? max_off_policy_steps=8 env_vars heartbeat?
├─ inference: InferenceConfig | None             (configs/inference.py:443-664)
│   server{host?, port=8000, liveness_timeout_seconds} router(vllm-router*|llm-d|None) backend_port=8100
│   vllm: VllmConfig(extra="allow"){model, tp, dp, dp_local?, api_server_count, rpc_port, enable_lora, ...}
│   log env_vars weight_broadcast{type} kv_cache_offload?(native|mooncake) enable_return_sampling_mask ...
│   [launcher-only] deployment(single_node*|multi_node|disaggregated) slurm? output_dir dry_run
└─ shared / launcher-only:
    env_vars run{name?, dir?} output_dir clean log{level?, json_logging} ckpt?{output_dir?, interval?, keep_last?, keep_interval?}
    resume?{step?|dir?} monitors{wandb?, file?, prime?} model?{name, vlm?} tokenizer? max_steps? seq_len?
    weight_broadcast?(filesystem|nccl|nixl) rollout_transport?(filesystem|zmq) deployment(single_node*|multi_node) slurm? dashboard=True dry_run=False
```
(`*` = default variant; `?` = Optional.)

### 3.5 Phase 1 — `propagate_shared_fields` (before) (`utils/validation.py:10-191`, called from `RLConfig.auto_setup_shared_configs`, `configs/rl.py:370-377`)

Operates on the **raw merged dict**, so only values the user actually wrote propagate (shared-block defaults such as `SharedModelConfig.name = "Qwen/Qwen3-0.6B"` never do). Rules: **fill-if-absent** into each target (only if the target's top-level section key, e.g. `trainer`, is present as a dict — `fill`, `:37-47`), and a **consistency mutex**: if a target already holds a *different* value, collect a conflict; equal values are accepted so dumped configs round-trip (`:51-65`). All conflicts are raised together (`:181-189`).

| Shared path | → targets |
|---|---|
| `model.name` | `trainer.model.name`, `inference.vllm.model`, `orchestrator.model.name` |
| `model.vlm` | `trainer.model.vlm`, `orchestrator.model.vlm` |
| `log.level`, `log.json_logging` | `trainer.log.*`, `orchestrator.log.*`, `inference.log.*` |
| `ckpt.output_dir` | `trainer.ckpt.output_dir` only |
| `ckpt.{interval,keep_last,keep_interval}` | `trainer.ckpt.*`, `orchestrator.ckpt.*` |
| `monitors.wandb.{project,entity,name,group,tags,offline}` | `trainer.monitors.wandb.*`, `orchestrator.monitors.wandb.*` |
| `monitors.file.path` | both sub-configs |
| `monitors.prime.name` | `orchestrator.monitors.prime.name` only |
| `tokenizer.{name,trust_remote_code}` | trainer, orchestrator |
| `tokenizer.chat_template` | trainer, orchestrator, `inference.vllm.chat_template` |
| `rollout_transport` (whole block) | `trainer.rollout_transport`, `orchestrator.rollout_transport` |
| `max_steps` | `trainer.max_steps`, `orchestrator.max_steps` |
| `seq_len` | `trainer.model.seq_len`, `orchestrator.seq_len` |
| `slurm` (whole block) | `inference.slurm` |
| (cascade) `trainer.tokenizer.chat_template` | `inference.vllm.chat_template` (fill-if-absent) |
| presence of bare `[ckpt]`, `[monitors.wandb]`, `[monitors.file]` / `[monitors.prime]` | `{}` into both / orchestrator only; `"None"` (from `--no-…`) propagates the disable |

Not propagated here (handled in phase 2): `weight_broadcast`, `resume`, `output_dir`/`run`, `deployment`, `env_vars` (RL-level `env_vars` never reach `inference.env_vars`; see `01` §3.2).

Two verified consequences. (a) `fill` inserts the **same** dict object into every target (no copy), so a sub-config's `mode="before"` validator that mutates it in place also mutates the shared block — e.g. `--rollout-transport.port 6000` with no `type` resolves because `TrainerConfig`'s `_default_discriminator_types` injects `type="zmq"` into the aliased dict, whereas `--weight-broadcast.port 29999` (not propagated in phase 1) fails with "Unable to extract tag using discriminator 'type'". (b) Presence propagation of `"None"` works: `--no-monitors.file`, `--monitors.file None` and `--no-monitors.wandb` disable the monitor on both trainer and orchestrator (checked by running `cli(RLConfig)`); only *leaf* propagation skips `None` values (`get(...) is None` → return, `:57-59`).

### 3.6 Phase 2 — `RLConfig` after-validators, in execution order (`configs/rl.py:311-867`)

Sub-configs are fully constructed (their own validators already ran) before these run. Mutations here are **not** re-validated in the launcher (`validate_assignment` is off) — they are re-validated only when the child re-parses its JSON.

| # | Validator (line) | What it does |
|---|---|---|
| 1 | `auto_setup_infer_nodes` (311) | multi-node: `deployment.num_infer_nodes ← inference.deployment.num_nodes` (or 1 for single-node inference, 0 without inference); must match for multi_node inference |
| 2 | `validate_deployment` (338) | multi-node requires `[slurm]`; infer nodes ⇔ `[inference]`; 0 infer nodes requires `trainer.data.fake` |
| 3 | `validate_enough_devices_for_nccl` (358) | single-node + `trainer.weight_broadcast.type == "nccl"` needs ≥2 GPUs total (checks the *pre-resolution* trainer value; see §7) |
| 4 | `auto_setup_run_dir` (379) | resolve `run.name` (`<envs>--<model>--<8hex>`, lowercased) and `run.dir`; set `trainer.output_dir = orchestrator.output_dir = output_dir/run.dir`, rejecting an explicit conflicting sub-config `output_dir` |
| 5 | `auto_setup_resume` (398) | copy top-level `resume` into trainer and orchestrator |
| 6 | `auto_setup_run_identity` (407) | unset W&B names (shared + both subs) and Prime monitor names ← `run.name` |
| 7 | `validate_shared_configs` (428) | `validate_shared_{model_name, tokenizer, max_steps, seq_len, ckpt_config, wandb_config}` (§3.7) |
| 8 | `auto_setup_weight_broadcast` (439) | default `nccl` (or `filesystem` if LoRA or no inference); LoRA ⇒ must be filesystem; build trainer/orchestrator broadcast configs with `host, port, timeout, inference_world_size = dp*tp` — the DP value *before* #19 fills it (+ NIXL `session_id`, `overlap_transfer_and_replay`); `inference.weight_broadcast.type` ← same; then `validate_shared_weight_broadcast` |
| 9 | `auto_setup_rollout_transport` (497) | trainer/orchestrator transport types must match; shared `rollout_transport ← trainer.rollout_transport` if unset (launcher uses it to decide ZMQ host injection) |
| 10 | `validate_eplb` (519) | `inference.vllm.enable_eplb` rejected |
| 11 | `auto_setup_lora` (525) | trainer LoRA ⇒ orchestrator `model.lora` (rank/alpha inherited, conflicts rejected), `inference.vllm.enable_lora = True`, `max_lora_rank = trainer rank` |
| 12 | `auto_setup_router_replay` (571) | `trainer.enable_router_replay` ⇒ `inference.vllm.enable_return_routed_experts = True` |
| 13 | `validate_llmd_no_routed_experts` (588) | llm-d router × routed-expert return rejected |
| 14 | `validate_multi_node_requires_router` (607) | multi-node needs `inference.router` |
| 15 | `auto_setup_sampling_mask_capture` (613) | any truncating policy train sampling ⇒ `inference.enable_return_sampling_mask = True` (+ eval warning) |
| 16 | `validate_disaggregated_combined_replay` (645) | P/D × (sampling replay + routed experts) rejected |
| 17 | `validate_router_replay_without_kv_offload` (660) | router replay × KV offload rejected |
| 18 | `validate_mooncake_offload_requires_slurm` (673) | mooncake offload needs `[slurm]` |
| 19 | `auto_setup_deployment` (687) | `orchestrator.pad_to_multiple_of = trainer.model.cp`. Single-node: `num_train_workers = num_train_gpus // cp` (if >1 GPU); `vllm.data_parallel_size = num_infer_gpus // tp` when `dp*tp != num_infer_gpus`; `api_server_count = dp` unless LoRA. Multi-node: `num_train_workers = num_train_nodes*gpus_per_node // cp`; `dp_replicate` from `nodes_per_fsdp_group`; EP: `dp*tp` must equal replica GPUs, `dp_local = gpus_per_node // tp`; non-EP: dp/dp_local/api_server_count ← `gpus_per_node // tp` when left at 1/None/1; NCCL/NIXL: `inference_world_size = total_infer_nodes*gpus_per_node` on both sides, trainer NCCL host `0.0.0.0` |
| 20 | `auto_setup_disaggregated_inference` (796) | multi-node + P/D: node count check, `orchestrator.inference_metrics_roles` (prefill…decode per replica, unless set), `inference_world_size = total infer GPUs` |
| 21 | `auto_setup_inference_client` (830) | no policy-sourced train env ⇒ `base_url = http://<server.host or localhost>:<server.port>/v1` (unless set); single-node + router ⇒ `admin_base_url = [http://<host>:<backend_port>/v1]` (unless set) |
| 22 | `auto_setup_slurm_template` (857) | `slurm.template_path ←` bundled `single_node_rl.sbatch.j2` / `multi_node_rl.sbatch.j2` via `find_package_resource("templates")` |

The "Warnings" section header at `configs/rl.py:869` is empty.

### 3.7 Cross-component checks (`utils/validation.py:194-318`)

| Function | Rule |
|---|---|
| `validate_shared_ckpt_config` | `trainer.ckpt` ⇔ `orchestrator.ckpt`; equal `interval`; `trainer.resume == orchestrator.resume` |
| `validate_shared_model_name` | with inference: `inference.vllm.model == orchestrator.model.name` (trainer may differ, e.g. FP8 inference variant); without: trainer == orchestrator (except `Jackmin108/*` models) |
| `validate_shared_wandb_config` | W&B on both or neither; same `project` |
| `validate_shared_max_steps` | equal |
| `validate_shared_seq_len` | `trainer.model.seq_len >= orchestrator.seq_len` |
| `validate_shared_tokenizer` | `chat_template` equal across trainer, orchestrator, inference (names may differ) |
| `validate_shared_weight_broadcast` | same `type` on trainer, orchestrator, inference |

### 3.8 Sub-config self-resolution worth knowing

- `InferenceConfig` (`configs/inference.py:494-576`): `backend_port = server.port + 100` if a router is set and `backend_port` unset; KV offload forces `enable_prefix_caching`; P/D sets `use_pd_kv_transfer`, EP on, EPLB off, and dp/dp_local/api_server_count from `gpus_per_node // tp`; multi-node/P/D requires `slurm` and a router; llm-d requires multi-node. `VllmConfig` (`:44-216`): unknown keys pass through (kebab→snake, JSON-decoded strings); `tool_call_parser`/`reasoning_parser = "auto"` resolved from the model name by regex tables (`utils/parsers.py:4-60`); `max_lora_rank` rounded up to `{8,16,32,64,128,256,320,512}`; `api_server_count ≥ dp_local or dp` unless set explicitly; LoRA forces 1; `headless` extra forces 0. `to_namespace()` builds vLLM args, dropping `None` for `_OMIT_IF_NONE` keys and all `None` extras, defaulting `logprobs_mode = "processed_logprobs"`, `enable_prompt_tokens_details = True`, fp32 lm-head/router `hf_overrides`, and the KV transfer config (`:607-664`).
- `OrchestratorConfig` (`configs/orchestrator.py:609-786`): tokenizer ← model; per-source `algo` ← top-level `algo`; truncating policy sampling ⇒ `top_k` defaults to `TRAIN_TOP_K_BOUND = 512` (warning), `temperature > 0`, `top_k ≤ 512`, no opd/opsd; `renderer.name = "auto"` must resolve via `MODEL_RENDERER_MAP` — keyed on `tokenizer.name or model.name` (`:715`), whereas the runtime renderer is built from `model.name` (`orchestrator/clients.py:85`), so a mapped `tokenizer.name` can pass the check while the renderer falls back; `batch_size` defaults to 128 if neither batch knob set, must divide by `group_size`; per-source `group_size` ← top-level; policy-sourced sources get `extra_body` defaults `top_k=-1, min_p=0.0, return_token_ids=True`; top-k capture mode must be uniform.
- `TrainerConfig` (`configs/trainer.py:735-828`): tokenizer ← model; DeepEP or full offload disable grad clipping (warning); scheduler vs `max_steps` checks; LoRA ⇒ filesystem broadcast; tracing caps.
- `SFTConfig` is monolithic (no split): `mode="before"` `normalize_deployment` drops single-node keys from multi-node deployments and `propagate_model_to_inference` fills `inference.vllm.model` (`configs/sft.py:299-324`); `auto_setup_online_eval` wires `weight_broadcast` (NCCL default when `[inference]` exists), DP fill, `eval.client.base_url/admin_base_url` (`:362-493`).
- `EvalConfig` flattens `ServedEvalConfig` to the top level with Prime Inference defaults (`configs/eval.py:48-118`); `SFTOnlineEvalConfig` is the online-eval process config (`:121-167`).
- `EnvServerConfig` narrows `env` to the selected env's verifiers config class (`vf.resolve_env_field`) and requires an env id (`configs/env_server.py:31-45`).

### 3.9 Serialization and the child re-parse

`dump_resolved_config(cfg, exclude) = cfg.model_dump(exclude=exclude, mode="json")` (`utils/config.py:15-22`). JSON (not TOML) is used because it keeps nulls, so explicit `None` overrides round-trip (test `test_resolved_json_roundtrips_explicit_none`, `tests/unit/test_configs.py:372-386`). `SerializeAsAny[vf.EnvConfig]` on `env` fields preserves subclass fields of the narrowed env config (`configs/orchestrator.py:160`). Launcher-only fields are excluded per component: inference drops `{deployment, slurm, output_dir, dry_run}` (and sets `router = None` for multi-node per-rank engines); SLURM single-node `rl.json` drops `{slurm, dry_run, clean}` (`entrypoints/rl.py:69-113,596`). Children receive shared values already materialized in their own sub-config (e.g. `trainer.json` contains `model.seq_len`, `output_dir = run_dir`, `resume`), so they need no knowledge of `RLConfig`. The one exception is SLURM single-node `rl.json`, which is re-parsed as a whole `RLConfig`: phase-1 sees every shared value equal to its sub-config copy (no conflict), and phase-2 runs again on already-resolved values — which is not always a fixed point (it *changes* the single-node `inference_world_size`, §7.7).

### 3.10 `--dry-run`

| Entrypoint | What still happens | What is skipped |
|---|---|---|
| `rl` local | full validation, `PRL_RUN_*` env, `validate_run_dir` (**incl. `--clean` rmtree**), `clean_future_steps`, attempt dirs + `latest` repoint, `command.txt`, launch TOML, all sub-config JSONs (all verified by running `rl()` on a scratch dir) | model pre-download, dashboard, GPU-availability check, process launch |
| `rl` SLURM | same + `rl.json` (single-node) or sub-configs (multi-node) + `launcher/rl.sbatch`; prints `sbatch <path>` | `sbatch`, dashboard, model download |
| `sft` | same pattern (`entrypoints/sft.py:274-276,305-307`); stale eval-artifact cleaning and data/model pre-download are skipped (`:541-551`) | |
| `inference` | local: returns right after parse (`entrypoints/inference.py:200-202`); SLURM: writes `inference.json` + sbatch | |
| `eval` | run dir guard/clean, attempt dirs, `eval.json`, env-server JSONs (`entrypoints/eval.py:133-160`) | env servers, eval |

`docs/configuration.md:62-65` recommends `--dry-run --output-dir /tmp --run.name check` to inspect `configs/latest/resolved/`.

## 4. Interfaces & contracts

### 4.1 `cli()` signature

`cli(cls, *, args: list[str] | None = None, default: T | None = None, prog=None, description=None, plain=None, wide=None) -> T` (`cli.py:1592-1601`). With `args=None` it reads `sys.argv[1:]` and prints+exits on error; with explicit `args` it raises `ConfigFileError` (tests use this, `tests/unit/test_configs.py:43-49`). Exported: `cli`, `BaseConfig`, `ConfigFileError` (`__init__.py:1-5`, version `0.3.0` in-tree while `pyproject.toml:9` requires `prime-pydantic-config>=0.4.3` from the editable path source `deps/pydantic-config`).

### 4.2 Resolved-config files (per launch attempt)

| File | Class that parses it | Written by |
|---|---|---|
| `configs/attempt_N/resolved/rl.json` | `RLConfig` | `rl` SLURM single-node |
| `.../trainer.json` | `TrainerConfig` | `rl` |
| `.../orchestrator.json` | `OrchestratorConfig` | `rl` |
| `.../inference.json` | `InferenceConfig` | `rl`, `sft`, `inference` SLURM |
| `.../envs/<split>/<name>.json` | `EnvServerConfig` | `rl`, `sft`, `eval` (`utils/pathing.py:230-246`) |
| `.../sft.json`, `.../eval.json` | `SFTConfig`, `SFTOnlineEvalConfig` / `EvalConfig` | `sft`, `eval` |
| `configs/attempt_N/command.txt`, `<name>.toml` | humans / tools | all launchers |

### 4.3 Fields every launcher-aware process relies on

`output_dir` (run dir after resolution), `run.name/dir`, `resume`, `ckpt`, `weight_broadcast.{type,host,port,timeout,inference_world_size}`, `rollout_transport.{type,host,port,hwm}`, `model.client.{base_url,admin_base_url}`, `num_train_workers`, `pad_to_multiple_of`, `env_vars`. Types/defaults are listed in `01` §4.4 and in the class files.

## 5. Invariants & assumptions

1. **Dump → re-parse is a fixed point.** Every validator must accept its own output: "explicit wins" checks use `model_fields_set`, which is *larger* in the child (every dumped field is "set"), so derivations must not re-trigger differently (e.g. `auto_setup_backend_port` skips because `backend_port` is now set; the propagation mutex accepts equal duplicates).
2. **The launcher's in-memory config is not fully validated after phase 2 mutations** — sub-validators re-run only in the child. Example: `auto_setup_lora` sets `inference.vllm.max_lora_rank = trainer rank` after `VllmConfig.auto_setup_max_lora_rank` ran, so the launcher holds e.g. 20 while the inference child rounds it to 32; `enable_lora` forcing `api_server_count = 1` also only takes effect in the child.
3. **After-validator order is semantic** (definition order in the class body). Several docstrings rely on it (e.g. `validate_llmd_no_routed_experts` "runs after `auto_setup_router_replay`", `configs/rl.py:588-594`; `auto_setup_run_identity` after the orchestrator's own `auto_setup_prime_monitor_name`). Reordering validators changes behaviour.
4. **Shared values are defaults, never stompers**; a value set in two places must agree.
5. Only fields reachable through plain `BaseModel` chains get typed CLI parsing (bool/list/aliases/short flags); everything below an `Optional[...]` or union is raw strings validated by pydantic's lax coercion.
6. `extra="forbid"` everywhere except `VllmConfig`; renamed fields get **no aliases** by policy (`skills/configs/SKILL.md:37`; `docs/configuration.md:58`) — old keys fail loudly.

## 6. Extension points — adding a config field end to end

1. **Declare** it on the right class in `packages/prime-rl-configs/src/prime_rl/configs/<component>.py` with a type, default and a PEP-224 docstring directly below (it becomes the `--help` text, `cli.py:657-705`). Use `X | None = None` for optional sub-blocks (bare-flag enable), a discriminated union with a `type: Literal[...]` for variants, `Field(ge=…)` for bounds. Put incompatibility checks in a `model_validator(mode="after")` on the *smallest* class that sees all inputs (configs skill: "must raise at resolve time, not at runtime").
2. **Component-local field**: that's it for config; read `config.<field>` in the consumer. It is automatically dumped into the component's JSON.
3. **Field that must agree across components**: add it to the shared block on `RLConfig` (`configs/rl.py:57-307`) and one `propagate("shared.path", "trainer.…", "orchestrator.…", …)` line in `propagate_shared_fields` (`utils/validation.py:67-141`); add a `validate_shared_*` check if sub-configs can still diverge (via explicit sub-config values); if it is presence-only (bare block), add it to `presence_targets` (`:166-179`). For `SFTConfig`, which is monolithic, read it directly or mirror a `mode="before"` fill like `propagate_model_to_inference`.
4. **Field derived from topology** (GPU counts, hosts, world sizes): add an after-validator on `RLConfig` placed *after* every validator whose output it reads (see §3.6 table), honour explicit user values via `"field" not in obj.model_fields_set`, and make it idempotent on re-parse.
5. **Launcher-only field** (only the launcher reads it): put it on `RLConfig` itself — children only ever receive sub-config dumps, so it never reaches them. If it must live on a sub-config class that a child also parses (the `InferenceConfig` "launcher-only" block: `deployment`, `slurm`, `output_dir`, `dry_run`), add it to the dump exclusion set (`entrypoints/rl.py:101`, `entrypoints/inference.py:141`, `entrypoints/sft.py:121`) so the child re-parses without it.
6. **Needed by a multi-node child at runtime** (host/port known only inside the allocation): pass a template variable in `write_slurm_script` (`entrypoints/rl.py:479-578`), and inject it in `multi_node_rl.sbatch.j2` as a CLI override on the child's command line (pattern: `--weight_broadcast.host $MASTER_ADDR`, `multi_node_rl.sbatch.j2:563-566`). Snake-case flags work because unknown paths fall through to raw nesting.
7. **Env-var knob for a child**: prefer `[<component>.env_vars]` over new fields; never use a name in `PROTECTED_ENV_VARS` (`shared.py:13-24`).
8. **vLLM argument**: nothing to add — `[inference.vllm] foo = …` passes through; type it in `VllmConfig` only if prime-rl reads it, and add it to `_OMIT_IF_NONE` if vLLM rejects `None` (`configs/inference.py:607-611`).
9. **Test**: extend `tests/unit/test_configs.py` (round-trip, propagation, conflict); every checked-in TOML under `configs/`, `examples/`, `k8s/` must parse with *some* config class (`test_load_configs`, `:52-70`). Run `uv run rl @ … --dry-run` and diff `configs/attempt_*/resolved/`.
10. **Rename/remove**: delete the old spelling (no alias); if a transitional remap is unavoidable, a `mode="before"` validator can rewrite the raw key (the parser keeps unknown flags raw for exactly this, `cli.py:1411-1417`; tests `test_before_validator_remaps_legacy_cli_key`).

## 7. Gotchas & limitations

1. **Nested `--x @ file` always overrides root `@` files**, regardless of order, and two nested files for the same key don't merge — the last wins (`cli.py:480,1658-1662`). Verified precedence with `cli(RLConfig)`: root `@ a @ b` → `b` wins; `--trainer @ t.toml` placed *before* the root file still wins; a second `--trainer @ t2.toml` drops `t.toml` entirely (the root file's value reappears where `t2` is silent); a CLI leaf flag beats the nested file.
2. **Under an optional/union parent, CLI values are untyped raw strings**: lists must be JSON (`--inference.vllm.lora-target-modules a b` → "Unrecognized arguments: b"; use `'["a","b"]'`), bare flags become `True`, and the `--flag=value` form is **not** split — the path including `=value` becomes a key set to `True` (`cli.py:1219-1236`). Verified: `--inference.vllm.max-model-len=4096` is **silently accepted** as the VllmConfig extra `{"max_model_len=4096": True}` (extra="allow") with `max_model_len` unset; `--trainer.optim.lr=5e-5` fails loudly ("Extra inputs", `optim` is a union). `--orchestrator.batch-size=32` (plain path) works. Always use the space form. The whole `inference.*` subtree and every union (`trainer.optim`, `trainer.loss`, `orchestrator.algo`, …) are in this mode.
3. **Optional unions defaulting to `None` or to a `default_factory` never get their `type` injected** (`_default_discriminator_types` only inspects `field_info.default`, `cli.py:132-144`): shared `[weight_broadcast]` and `[inference.router]` (`default_factory=VllmRouterConfig`) must spell `type` explicitly — verified `--weight-broadcast.port 29999` and `--inference.router.policy round_robin` both fail with "Unable to extract tag using discriminator 'type'". Shared `[rollout_transport]` happens to work without `type` via dict aliasing (§3.5).
4. `"None"` is converted to `None` for **every** field, including string fields (a W&B run literally named "None" cannot be expressed).
5. **Shared-block defaults don't propagate** — only written values do. `SharedWandbConfig.project` default `"prime-rl"` coincides with `WandbMonitorConfig.project`, but in general a shared default is not a sub-config default.
6. `validate_enough_devices_for_nccl` runs **before** `auto_setup_weight_broadcast` resolves the shared transport, so it only fires for an explicit `[trainer.weight_broadcast] type = "nccl"` (`configs/rl.py:358-366` vs `439-495`).
7. **Single-node NCCL/NIXL `inference_world_size` is computed before DP auto-fill** (`configs/rl.py:458-469` runs before `:696-703`, and the single-node branch never updates it). Verified by running the validators: `num_infer_gpus=2` (DP unset) → `data_parallel_size=2` but `inference_world_size=1` in `trainer.json`/`orchestrator.json`; explicit `dp=4` on 2 GPUs → DP silently rewritten to 2, world size 4. Re-parsing the dumped `rl.json` (SLURM single-node) yields the correct 2 — so local and SLURM runs of the same config differ. **Confirmed bug**: local runs with `num_infer_gpus > tp` and DP unset fail at the first weight sync; set `inference.vllm.data_parallel_size` explicitly (`01` §7.5).
8. `auto_setup_inference_client` only auto-sets `base_url` when *no* train source samples from the policy; the common case relies on the `ClientConfig.base_url` default `http://localhost:8000/v1` (the validator docstring's `["http://localhost:8000/v1"]` list form is stale — `base_url` is `str`, `shared.py:177`). Change `inference.server.port` ⇒ change `orchestrator.model.client.base_url` too (the local launcher checks the port, `entrypoints/rl.py:176-186`).
9. **Docs drift**: `docs/configuration.md:103-104` uses `--orchestrator.train.source.0.args`, which cannot parse (no index paths; sources have no `args`). `skills/configs/SKILL.md:76` states IPO defaults `eps = 0.1`, `kl_tau = 1e-3`; the class has `eps = 0.3`, `kl_tau = 0.0` (`configs/trainer.py:572-581`). `skills/training/start-run/SKILL.md:224` says "validation aliases let renamed fields keep working", contradicting the no-alias policy. The `RLConfig.seq_len` docstring says "explicit per-component values always win" (`configs/rl.py:290-291`), but `--seq-len 4096 --orchestrator.seq-len 1024` raises the phase-1 conflict error (verified).
10. `write_launch_toml` concatenates multiple root TOMLs into one file with comment headers — duplicate tables make it a record, not a re-loadable config; `resolved/*.json` is the reproducible artifact.
11. `find_package_resource("templates")` returns `None` on a configs-only install; then `slurm.template_path` stays `None` and `rl_slurm` asserts (`entrypoints/rl.py:430-431`).
12. The in-tree `pydantic_config.__version__` attribute is hard-coded `"0.3.0"` while the distribution version is derived from git tags (`deps/pydantic-config/pyproject.toml:3`, hatch-vcs) and `packages/prime-rl-configs/pyproject.toml:9` requires `>=0.4.3` — don't use `__version__` to reason about parser behaviour; the submodule pin (`65b15df`) is the truth.

## 8. For a custom framework

- **Keep**: one typed schema package importable without GPU deps; `@`-composed TOML + dotted CLI overrides; validate-once on the merged dict so `model_fields_set` means "user wrote it"; `extra="forbid"` + no-alias renames; resolved **JSON** per process as the handoff; the shared-block "fill-if-absent + disagree-is-error" rule — it is the right semantics for a composite config.
- **Fix**: (a) make the parser type-aware below `Optional`/union nodes (recurse into the inner model / selected variant) so lists, bools and `=` work uniformly; (b) inject discriminator defaults for `default_factory` and `None`-defaulted unions (or require explicit `type` everywhere and say so); (c) replace order-dependent after-validators with an explicit resolution pass (a function that takes the composite and returns fully derived sub-configs, then validates each sub-config once) — this removes the "launcher view ≠ child view" gap and the NCCL world-size ordering hazard; (d) compute topology-derived values (world sizes, ranks, hosts) in one place, ideally from the actual placement plan, not from config arithmetic in three validators and two bash templates.
- **Drop/avoid**: string sentinels (`"None"`), silent pass-through of unknown CLI keys into raw dicts (keep it only behind an explicit legacy-remap hook), and duplicated knobs that must be kept equal across sub-configs (`seq_len`, `max_steps`, ckpt intervals) — derive them from one source instead of validating equality.

## 9. Open questions

1. Whether any code path re-validates `RLConfig`-mutated sub-configs in the launcher (e.g. for `--dry-run` output fidelity) — none found in `entrypoints/rl.py`; the dumped JSON is the first place the child's view is materialized (diff `resolved/*.json` against a child re-parse to see divergences such as `max_lora_rank` 20 → 32, verified).
2. Whether the VllmConfig extra produced by `--inference.vllm.x=v` (§7.2) is passed to vLLM by `to_namespace()` and what vLLM does with an attribute named `x=v` (runtime).
