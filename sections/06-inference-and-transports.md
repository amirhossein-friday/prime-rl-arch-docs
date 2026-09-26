# Inference server, weight sync, and rollout transport — prime-rl @ b944873

> Scope: the vLLM inference server as prime-rl builds and patches it, every HTTP route the orchestrator uses on it (S2 server side + wire), the rollout/batch wire from orchestrator to trainer (S6 wire formats), and trainer→inference weight sync (S7 transport + receiver) for filesystem, NCCL and NIXL, plus routers, P/D, llm-d, Dynamo, and KV offload.
>
> Files read in full: `src/prime_rl/inference/server.py` (49), `inference/patches.py` (920), `inference/dynamo.py` (429), `inference/json_logging.py` (88), `inference/vllm/server.py` (241), `inference/vllm/serving_tokens.py` (87), `inference/vllm/routed_experts.py` (48), `inference/vllm/gpt_oss_weight_loading.py` (145), `inference/vllm/worker/{__init__,filesystem,nccl,nixl,weight_transfer}.py` (18/55/147/593/54), `entrypoints/inference.py` (255), `transports/batch/{__init__,base,filesystem,zmq,types}.py` (56/55/62/141/125), `transports/weights/{__init__,base,filesystem,nccl}.py` (51/169/75/213), `transports/weights/nixl/{__init__,nixl,agent,graph,tensor_routing,trainer_tensor_table,model_express,cuda_malloc_memory}.py` (5/502/210/346/83/60/97/67), `utils/{weights,nccl,metrics_server}.py` (252/42/176), `orchestrator/clients.py` (556), `orchestrator/watcher.py` (139), `packages/prime-rl-configs/src/prime_rl/configs/inference.py` (664), `configs/shared.py` (255), `templates/{_launch_router.sh.j2,_mooncake_store.sh.j2,_launch_rank.sh.j2,inference.sbatch.j2,multi_node_rl.sbatch.j2,llmd/*}`, `scripts/install_{nixl_from_source,modelexpress,llmd}.sh`, `docs/inference.md`, `docs/scaling.md`, `deps/renderers/renderers/client.py` (638), tests `tests/unit/{inference/test_serving_tokens.py,inference/test_dynamo.py,transports/test_nccl_broadcast.py,orchestrator/test_clients.py}`.
> Read in part (the call sites): `orchestrator/orchestrator.py` (1–640, 960–1077), `orchestrator/dispatcher.py` (545–614), `orchestrator/envs.py` (60–170), `trainer/rl/train.py` (170–284, 560–605), `trainer/rl/data.py` (175–279), `configs/rl.py` (full), `configs/trainer.py` (625–704), `configs/orchestrator.py` (20–180, 440–620), `entrypoints/rl.py` (190–250), `eval/online.py` (40–110), `utils/pathing.py` (261–425), `utils/process.py` (17–33), `deps/verifiers/verifiers/v1/clients/train.py` (120–190, 300–460), `deps/verifiers/verifiers/v1/types.py` (170–285).
>
> Related docs: 01-deployment-topology-and-launch.md (A: process spawn, placement), 02-config-system.md (A), 03-orchestrator.md (B: client side of S2, producer of S6), 04-algorithms-loss-data-path.md (C: consumer of S6 fields, S9 versioning), 05-trainer.md (D: trainer-side export/conversion for S7), 07-verifiers-core.md (F: token-in client, S4), 09-renderers.md (H).

---

## 1. Mental model

**Inference = stock vLLM, lightly extended in-process.** prime-rl never forks vLLM. It (a) builds the vLLM CLI `Namespace` from a typed pydantic config, (b) monkeypatches vLLM's app factory to mount a handful of extra FastAPI routes (`/pause`, `/resume`, `/update_weights`, `/load_lora_adapter`, `/liveness`, `/init_broadcaster`), (c) installs a **vLLM worker extension class** (one per weight transport) whose methods are invoked through `engine_client.collective_rpc(...)` on every GPU worker, and (d) registers a `vllm.general_plugins` entry point that applies ~12 model/scheduler monkeypatches inside every vLLM process including spawned workers (`pyproject.toml:50-51`, `src/prime_rl/inference/patches.py:4-25`). Rollouts use vLLM's upstream **token-in/token-out** endpoint `/inference/v1/generate`, with prime-rl only swapping in a subclass that re-encodes `routed_experts` (`inference/vllm/serving_tokens.py:58-87`).

**Every deployment is "router in front, engines behind".** A single global router (PrimeIntellect's `vllm-router` fork by default, or llm-d EPP+Envoy) listens on `inference.server.port` (default 8000) and is the only data-plane URL clients use; engines listen on `backend_port` (default `server.port + 100` = 8100) or `backend_port + k` per DP rank. **Admin traffic never goes through the router**: the orchestrator holds one httpx client per engine process (`admin_base_url` list) and fans pause/update/resume out to all of them (`orchestrator/clients.py:132-140`).

**Weight sync is a four-marker filesystem handshake wrapped around a transport.** For all three transports the trainer master creates `broadcasts/step_N/`, touches `.sender_ready`, blocks until the consumer (orchestrator, or the SFT online-eval process) touches `.receiver_ready`, touches `.started`, moves the weights, then touches `.finished` (`transports/weights/base.py:17-70`). The consumer side (`WeightReceiver`) pauses the engines (mode `keep`), makes the engines pull/receive, then resumes them. The transports differ only in how bytes move:
- **filesystem**: trainer writes an HF safetensors checkpoint (or PEFT adapter) into the step dir; each vLLM worker reloads it layerwise from disk.
- **nccl**: a one-off `StatelessProcessGroup` + `PyNcclCommunicator` with trainer rank 0 as rank 0 and every inference GPU as ranks `1..W`; the trainer master broadcasts per-layer, per-dtype concatenated HF-format tensors.
- **nixl**: receiver-driven one-sided RDMA READs. Trainer ranks expose FSDP-local shards (in *trainer* naming) in registered staging arenas; each vLLM worker traced vLLM's own `load_weights` once with lazy "recording" tensors to derive a static copy plan (source shard ranges → destination parameter views + replayable view ops), and replays it every step. Discovery goes through a ModelExpress gRPC metadata service; synchronization through NIXL notifications.

**Rollout/batch transport is a dumb per-DP-rank pipe of msgpack.** The orchestrator packs a `list[list[MicroBatch]]` grid (one list per trainer data rank) and ships it either over ZMQ PUB/SUB (default) with topic `data_rank|<r>|`, or as `batches/step_N/rank_<r>.bin` files (`transports/batch/*`). There is no step id in the ZMQ message; both ends count steps implicitly. Backpressure is not in the transport — it is the orchestrator's `TARGET_LAG = 1` dispatch gate (`orchestrator/orchestrator.py:95, 607-625`).

---

## 2. Where it runs

```
                 ┌──────────────────────── inference node(s) ─────────────────────────┐
 orchestrator ──►│ router :8000  (vllm-router | envoy:ROUTER_PORT + epp:9002)          │
  data plane     │    │  /v1/*  /inference/v1/*   (+ /finish_session on vllm-router)    │
  (renderer/     │    ▼                                                               │
   env server)   │ vLLM API server(s) :backend_port(+k)   ◄── admin plane (direct)    │◄── orchestrator AdminPlane
                 │    │  custom routes /pause /resume /update_weights ...             │    (one httpx client / engine)
                 │    ▼  collective_rpc                                               │
                 │ EngineCore(s) → GPU workers  + worker_extension_cls                │
                 │   (FileSystem|NCCL|NIXL)WeightUpdateWorker                         │
                 └────────────────────────────────────────────────────────────────────┘
                       ▲ NCCL broadcast (rank0=trainer master) │ NIXL READs from trainer arenas
 trainer (torchrun) ───┘  or safetensors in output_dir/broadcasts/step_N
 trainer ◄── ZMQ SUB(tcp://ORCH:5555) / PUSH READY(:5556) ── orchestrator PUB/PULL   (or batches/step_N/rank_r.bin)
```

**Process launch.** `uv run inference @ inference.json` → `entrypoints/inference.py:main` (`:249-251`). If `slurm` is set it renders `inference.sbatch.j2` and `sbatch`es it (`:132-162`); otherwise `inference_local` (`:194-239`):
1. `os.environ.update(DEFAULT_COMMON_ENV_VARS ∪ DEFAULT_INFERENCE_ENV_VARS ∪ config.env_vars)` (`:210`); defaults are `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, `VLLM_ENGINE_READY_TIMEOUT_S=4200`, `UCX_TLS=all` (`utils/process.py:28-33`).
2. `setup_vllm_env(config)` sets vLLM env before vLLM import (`inference/server.py:7-49`, see §3.1).
3. If `config.router` is not None, spawn `vllm-router` on `server.port` fronting `http://localhost:backend_port`, then rewrite `config.server.port = config.backend_port` so the engine binds behind it (`entrypoints/inference.py:216-220`). A daemon thread SIGTERMs the inference process if the router dies (`:222-228`).
4. `from prime_rl.inference.vllm.server import server` (importing it applies patches and route registration) and `server(config)` (`:232-235`), which runs `run_server` (single API server, in-process uvloop), `run_multi_api_server` (`api_server_count > 1`), or `run_headless` (`inference/vllm/server.py:234-241`).

Under the `rl` entrypoint single-node, the launcher `Popen`s `inference @ config_dir/inference.json` with `CUDA_VISIBLE_DEVICES` = the inference GPU slice (`entrypoints/rl.py:204-222`). Multi-node/disaggregated SLURM launches one `uv run inference` **per DP rank** (external load balancing) via `launch_inference_rank` with `--server.port <port>`, `--vllm.data-parallel-size <dp>`, `--vllm.data-parallel-size-local 1`, `--vllm.api-server-count 1`, `CUDA_VISIBLE_DEVICES=<tp slice>`, and a per-rank JSON overrides file (`templates/_launch_rank.sh.j2:16-39`). Per-rank configs are written with `router = None` (`entrypoints/inference.py:42-57, 141-142`) and the sbatch starts the single global router on inference node 0 (`templates/_launch_router.sh.j2:9-86`).

**Lifecycle.** Startup readiness: the orchestrator polls `GET /health` on each engine (and the router) until 200 (`orchestrator/clients.py:154-163, 332-367`), then `GET /v1/models` to confirm the model id (`:311-329`). Steady state: data-plane requests via router; admin ops per engine. Shutdown: the SLURM templates tear down on the first exiting background job (`templates/inference.sbatch.j2:357-363`); locally the router is cleaned up in `finally` (`entrypoints/inference.py:236-239`). There is no "drain" API on the server; the orchestrator's final `wait_for_final_broadcast` keeps the receiver alive for the trainer's last handshake (`orchestrator/orchestrator.py:487-493`).

**Crash semantics.** Any admin op failure propagates out of the watcher task and kills the orchestrator (`orchestrator/orchestrator.py:561-571`). `_resume_engines` runs in a `finally` so an update failure still tries to resume (`orchestrator/clients.py:216-232`). The trainer fails the run if no receiver acks within `weight_broadcast.timeout` (default 1200 s; `transports/weights/base.py:75-93`, `configs/shared.py:40-43`). That timeout bounds **only the ack wait**; the transfer itself (`_broadcast`) has no trainer-side deadline beyond NCCL/NIXL/store timeouts. The FS receiver's `wait_for_path(.finished)` has **no timeout** (`utils/pathing.py:414-425`): a trainer dying mid-write strands the watcher.

---

## 3. Mechanics

### 3.1 How the vLLM server is built

**Env before import** (`inference/server.py:7-49`):
| Env var | Value | Why |
|---|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` (setdefault) | fork deadlocks with multithreaded procs / Qwen3-VL (`:10-11`) |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` if `enable_return_sampling_mask`; else if routed experts: `0` for `disaggregated`, `1` otherwise (setdefault) | sampling-mask capture needs V2; NIXL routed-expert stitching is V1-only (`:13-21`) |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | `0` (setdefault) | vLLM 0.24 flipped default to grammar-constrained tool calls — a distribution the trainer never sees (`:23-27`) |
| `VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM` | `1`/`0` from `use_deep_gemm` (forced) | `:29-31` |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `True` if `vllm.enable_lora` | `:33-34` |
| `VLLM_CONFIGURE_LOGGING=1`, `VLLM_LOGGING_CONFIG_PATH=<tmp json>` | if `log.json_logging` | env var crosses the spawn boundary to workers (`:36-49`, `inference/json_logging.py:79-88`) |

**Namespace translation** — `InferenceConfig.to_namespace()` (`configs/inference.py:613-664`):
- `host`, `port`, `liveness_timeout_seconds` from `server`; every typed and **extra** (pass-through, `extra="allow"`) field of `[inference.vllm]`; `None` values are dropped for `_OMIT_IF_NONE` keys and for all extras so vLLM applies its default (`:609-627`).
- `enable_auto_tool_choice = hasattr(namespace, "tool_call_parser")` unless overridden (`:631-632`).
- `logprobs_mode = "processed_logprobs"` unless set (`:635-636`) — logprobs are of the post-temperature/top-p/top-k distribution (why sampling replay exists, §3.3).
- `enable_prompt_tokens_details = True` (router cache-discount billing counters parse `usage`) (`:640-641`).
- `hf_overrides.moe_router_dtype = "float32"` if `enable_fp32_router_logits` (default True) and `hf_overrides.head_dtype = "float32"` if `enable_fp32_lm_head` (default True) (`:645-655`).
- `return_sampling_mask = True` if `enable_return_sampling_mask` (`:657-658`).
- `kv_transfer_config` = NIXL connector (P/D), offload connector, or `MultiConnector` of both (`:578-605, 660-662`).

**`server(config)`** (`inference/vllm/server.py:212-241`): sets `PRIME_NO_MOE_LORA=1` when `lora_target_modules` contains no `"expert"` (`:218-221`); parses the namespace through vLLM's own `make_arg_parser` + `validate_parsed_serve_args` (`:225-229`); sets `args.worker_extension_cls` from `WORKER_EXTENSION_CLS[config.weight_broadcast.type]` (`:59-63, 232`).

**Route mounting.** At module import, prime-rl replaces three vLLM symbols (`:204-206`):
- `entry.build_app` → `custom_build_app` which calls the original and `app.include_router(router)` (`:185-191`).
- `entry.init_app_state` → `custom_init_app_state` (`:149-176`): runs the original, stores `state.liveness_timeout_seconds`, and **swaps `state.serving_tokens` for a `PrimeRlServingTokens` shell** by `object.__new__` + copying `__dict__` (`:170-176`).
- `vllm.v1.utils.run_api_server_worker_proc` → wrapper that re-imports `prime_rl.inference.vllm.server` in each child API-server process so the patches apply in multi-API-server mode (`:194-201`).

It also applies four patches at import (API-server process only) (`:28-43`): `monkey_patch_tokenize_params_validation`, `monkey_patch_nano_v3_reasoning_parser`, `monkey_patch_strip_routed_experts_from_chat`, `monkey_patch_dp_coordinator_startup_timeout`.

### 3.2 Every patch (what, where applied, why)

Applied in **every vLLM process** via the `vllm.general_plugins` entry point `apply_shared_vllm_patches` (`patches.py:4-25`; `pyproject.toml:50-51`), in the order of the table. Failure semantics (vLLM 0.24 `plugins/__init__.py:28-82`; pin 0.29): only `plugin.load()` (the entry-point import) is wrapped in `try/except → logger.exception`, so the docstring's "silently skips ALL" (`:8-10`) applies only if `import prime_rl.inference.patches` fails (it imports only `torch`). Patches import their vLLM targets lazily and `func()` is called **unguarded** ⇒ a broken patch import **aborts process startup**. `VLLM_PLUGINS`, if set, can exclude the plugin.

| Patch | Target | Why | Cite |
|---|---|---|---|
| `patch_gpt_oss_weight_loading` | `GptOssModel._load_weights_other`, `RoutedExperts.weight_loader` | route GPT-OSS BF16 expert (`w13/w2` weight+bias) and `sinks` tensors through vLLM weight loaders with TP/EP slicing; accept already-sharded combined expert tensors (`expert_id=None`) | `inference/vllm/gpt_oss_weight_loading.py:6-145` |
| `_patch_lora_key_prefix` | `LoRAModel.from_local_checkpoint` (full copy of vLLM 0.24) | accept both bare-suffix (`down_proj`) and qualified (`experts.N.down_proj`) expert module names in adapters; upstream PR closed unmerged | `patches.py:449-606` (patched block `:532-538`) |
| `_patch_qwen35_moe_lora_format` | `Qwen3_5MoeForConditionalGeneration.is_3d_moe_weight = False` | trainer emits 2D per-expert LoRA layout | `:426-446` |
| `monkey_patch_nano_v3_reasoning_parser` | registers `"nano_v3"` reasoning parser (DeepSeekR1 subclass swapping reasoning/content when `enable_thinking=False`) | also applied at API-server import | `:292-306` |
| `monkey_patch_minimax_m2_think_end_passthrough` | `minimax_m2.minimax_m2_config` | vLLM 0.24 parser swallows `</think>` and strips content when tools present; breaks think-tag round-tripping | `:309-340` |
| `monkey_patch_return_routed_experts_with_nixl_connector` | `VllmConfig.__post_init__` | vLLM rejects routed-expert capture with any KV connector; prime-rl temporarily clears the flag during post-init for NIXL P/D; requires PP=1 and V1 runner | `:343-385` |
| `monkey_patch_kv_xfer_finished_tolerate_freed` | `Scheduler._update_from_kv_xfer_finished` | aborted P/D request whose recv+send complete in the same step is freed twice → assertion kills EngineCore and cascades across DP; skip already-freed ids (vllm#46240) | `:127-185` |
| `monkey_patch_online_fp8_parameter_cast` | `fp8.per_block_cast_to_fp8` | layerwise reload restores `Parameter` subclasses that break Dynamo tracing; pass `.data` | `:103-124` |
| `monkey_patch_deepseek_v4_allowed_layer_types` | transformers `layer_types` validator | DSV4 config construct under transformers < 5.15 | `:28-42` |
| `monkey_patch_deepseek_v4_request_tools_placement` | DSV4 tokenizer factory | attach request tools to the first system message (match training renderer); remove at vLLM 0.31 | `:45-100` |
| `monkey_patch_deepseek_v4_per_layer_rope` | `build_deepseek_v4_rope` (both defining module and `attention` namespace) | per-layer RoPE: plain for sliding layers, YaRN at `compress_rope_theta` for compressed; remove at 0.29.1 | `:800-920` |
| `monkey_patch_deepseek_v4_bf16_o_proj` | `deep_gemm_fp8_o_proj` (+ two importers) | bf16 checkpoints have no FP8 scales; einsum fallback in bf16 with fp32 inverse-RoPE | `:188-289` |

Applied **only in the API-server process** at `inference/vllm/server.py` import:
| Patch | Why | Cite |
|---|---|---|
| `monkey_patch_tokenize_params_validation` | reject only `prompt_len > max_model_len` instead of `> max_model_len - max_tokens`; fix HF truncation `max_length` | `patches.py:610-676` |
| `monkey_patch_strip_routed_experts_from_chat` | chat-completions responses carry routed experts as a base64 `.npy` string the PD router rejects; strip them (evals use chat) | `:388-423` |
| `monkey_patch_dp_coordinator_startup_timeout` | vLLM's hard-coded 120 s DP coordinator startup wait; env `PRIME_DP_COORDINATOR_STARTUP_TIMEOUT` (default 300) | `:767-797` |

Applied **in the worker process** at worker-extension import (`inference/vllm/worker/__init__.py:11-18`):
- `monkey_patch_minimax_m2_for_lora` — always: rebuilds MiniMax-M2 MoE gate as bf16 `GateLinear` with fp32 output (LoRA Triton kernel asserts fp16/bf16) and sets an `hf_to_vllm_mapper` for adapter key names (`patches.py:679-744`).
- `monkey_patch_no_moe_lora` — only if `PRIME_NO_MOE_LORA=1`: sets `FusedMoEConfig.is_lora_enabled=False` post-init so faster MoE kernels are picked (`:747-764`).

### 3.3 Token-in generation: `/inference/v1/generate`

This is vLLM's upstream `ServingTokens` handler (`vllm.entrypoints.scale_out.token_in_token_out`), mounted at the **server root** (not under `/v1`). prime-rl's only server delta is `PrimeRlServingTokens.serve_tokens_full_generator` (`inference/vllm/serving_tokens.py:58-87`): when `model_config.enable_return_routed_experts`, it wraps the result generator in `_GenerateRoutedExpertsCapture(start=request.sampling_params.routed_experts_prompt_start)` which, for every streamed `RequestOutput`, encodes `output.routed_experts` into the compact form and then replaces `choices[i].routed_experts` in the final response (`:46-55, 72-85`). Everything else (schema, sampling params, validation, DP-rank header, lora, cache salt, multimodal, priority, usage) is upstream (`:1-14`).

**Request as sent by the training path** (built by `renderers.client.generate`, `deps/renderers/renderers/client.py:310-352`, called from verifiers `TrainClient.get_response`, `deps/verifiers/verifiers/v1/clients/train.py:367-443`):

```jsonc
POST {base_url without /v1}/inference/v1/generate      // client.py:338-339
headers: { "X-Session-ID": "<session id>" }             // train.py:440-442; SESSION_ID_HEADER (verifiers/v1/clients/client.py:16)
{
  "model": "<policy model name>",                       // served name; for LoRA the adapter shadows it
  "token_ids": [int, ...],                              // rendered (or bridged) prompt ids
  "sampling_params": {                                  // forwarded verbatim to vLLM SamplingParams
     "temperature": float, "top_p": float,              // TrainSamplingConfig.to_sampling_args (configs/orchestrator.py:91-108)
     "max_tokens": int?,                                // max_completion_tokens aliased (verifiers/v1/types.py:270-284)
     "top_k": int?, ...extra_body keys,                 // flattened extra_body
     "logprobs": 1,                                     // forced (client.py:312)
     "stop_token_ids": [int, ...],                      // forced from renderer (client.py:311)
     "skip_special_tokens": false,                      // default (client.py:313)
     "routed_experts_prompt_start": int?                // only on bridged multi-turn (train.py:409-412)
  },
  "cache_salt": "<policy_version_at_group_start>",      // top-level (client.py:330-331; dispatcher.py:562-565)
  "features": {mm_hashes, mm_placeholders, kwargs_data}?,   // multimodal, vLLM MultiModalFeatures (client.py:321-329, 572-637)
  "content_parts": [{type, url}]?,                      // process_multimodal=False path (client.py:320, 326-327)
  "priority": int?
}
```

**Response** (fields consumed by `client.py:353-422`):
| Field | Meaning |
|---|---|
| `request_id` | vLLM request id |
| `prompt_token_ids` | effective (possibly mm-expanded) prompt ids; required when `content_parts` used |
| `mm_placeholders` | required when `content_parts` used |
| `choices[0].token_ids` | completion token ids |
| `choices[0].logprobs.content[i]` | `{token: "token_id:<id>", logprob: float}`; length must equal completion; `-9999.0` is rejected as "no sampling evidence" (`client.py:137-188`) |
| `choices[0].finish_reason` | `stop`/`length`/…; client promotes `stop`→`tool_calls` if it parsed ≥1 OK tool call (`:389-394`) |
| `choices[0].routed_experts` | `{data: base64(raw uint8|uint16 bytes), shape: [tokens, layers, topk], start: int, dtype: "uint8"|"uint16"}` (`inference/vllm/routed_experts.py:11-33`); uint16 when any expert id > 255 (e.g. 512-expert NemotronH). The client splices the base64 string out of the raw JSON before `json.loads` to avoid decoding megabytes (`client.py:112-134`) |
| `choices[0].sampling_mask` | vLLM native (`--return-sampling-mask`): one list of surviving vocab ids per completion token; verifiers packs it into `SamplingMask{ids:int32 flat, counts:int32}` (`deps/verifiers/verifiers/v1/types.py:194-217`) |
| `usage` | includes `prompt_tokens_details` (cached tokens) because `enable_prompt_tokens_details=True` |

**Prefill scoring** uses the same endpoint: `prefill_logprobs` posts `{"model", "token_ids", "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1}}` and flattens `prompt_logprobs`: position 0 (`None`) → 0.0, else the **first dict entry** of each position (`orchestrator/clients.py:523-556`). The first entry is the prompt token itself — vLLM always inserts the actual (prompt/sampled) token first, then the top-k (vLLM 0.24 `v1/engine/logprobs.py:213-217`), and JSON preserves the order. Used by `InferenceClient.score` (`:103-106`). The request carries **no `cache_salt`**, but that is harmless: any request with `prompt_logprobs` sets `skip_reading_prefix_cache=True` (vLLM 0.24 `sampling_params.py:481-485`, honoured in `v1/core/kv_cache_manager.py:215-219`), so prefill scoring always recomputes the full prompt under the current weights (it still *writes* unsalted blocks). It also carries **no `top_k`**, so it would be rejected by an engine with sampling-mask capture on; the config prevents that pairing by rejecting `opd`/`opsd` with truncated train sampling (`configs/orchestrator.py:684-690`).

**Eval path** uses stock OpenAI `/v1/chat/completions` through the router (`InferenceClient.eval_client`, `orchestrator/clients.py:92`); routed experts are stripped there (§3.2).

**Why the knobs matter for RL correctness.**
- `logprobs_mode=processed_logprobs` + truncated sampling ⇒ inference logprobs are $\log \tilde\pi(y_t)$ with $\tilde\pi = \pi \cdot \mathbb 1[y\in M_t] / \sum_{y'\in M_t}\pi(y')$. The trainer must renormalize over the same kept set $M_t$ ("sampling replay", `docs/inference.md:304-318`). The `rl` entrypoint auto-enables `enable_return_sampling_mask` when any policy-sourced train env truncates (`configs/rl.py:613-643`); `top_k` is bounded to `TRAIN_TOP_K_BOUND = 512` because the trainer pads each micro batch's masks to the largest mask (`configs/orchestrator.py:509-515`; `trainer/rl/data.py:225-234`).
- Sampling-mask capture is **engine-wide**: vLLM rejects requests with `temperature <= 0` or without `top_k > 0` while it is on, including eval requests on the same server. The rejection is **inside vLLM** (native `--return-sampling-mask`, ≥0.28, not readable here); prime-rl documents it (`configs/inference.py:472-473`), rejects `temperature == 0` on truncated train sampling (`configs/orchestrator.py:658-663`), injects `top_k = 512` when only `top_p` truncates (`:673-681`), and only **warns** about evals (`configs/rl.py:636-642`) — an eval without `top_k` fails at request time.
- fp32 lm_head / fp32 router logits default on to shrink trainer↔inference mismatch (`configs/inference.py:475-479`).

### 3.4 The admin plane (orchestrator → engines)

`AdminPlane(client_config)` (`orchestrator/clients.py:132-236`):
- `clients = setup_admin_clients(client_config)`: one `httpx.AsyncClient` per URL in `admin_base_url` (else `[base_url]`), with `/v1` stripped (admin routes are at root), `Authorization: Bearer $<api_key_var>` if set, **`max_connections=4, max_keepalive_connections=1`**, `timeout=None` (`:283-308`). A dedicated pool so admin ops don't queue behind streaming data-plane requests (`:284-289`).
- `_router_clients`: when `admin_base_url` is set, also a client for the router URL, used only for `/health` (`:144-150`). Engines are always health-checked too because Envoy (llm-d) 404s `/health` and `check_health` treats 404 as "no health route, skip" (`:155-163, 349-352`).
- **Client order is load-bearing**: it must match the GPU rank order used by NCCL/NIXL init and the metrics collector's role list (`:138-140`).

`_admin_post(client, path, timeout_s=300, **kwargs)`: tenacity retry on `HTTPStatusError` ≥500 and any `httpx.TimeoutException`/`TransportError` (4xx never retried), `stop_after_delay(2·timeout_s) | stop_after_attempt(10)`, exponential wait 1–10 s, `reraise=True`, per-attempt `httpx.Timeout(connect=10, read=timeout_s, write=60, pool=10)` (`:370-412`). Constants: `ADMIN_TIMEOUT_S = 300.0` (`:389`), `UPDATE_WEIGHTS_TIMEOUT_S = 720.0` (`:392`) ⇒ `/update_weights` budget 720 s/attempt, 1440 s total.

`AdminPlane.update_weights(weight_dir, *, transport, step, on_paused)` (`:205-232`):
```
_pause_engines(all clients)                # POST /pause params={mode: keep, clear_cache: false}, gathered   (:415-422)
try:
    on_paused()                            # NCCL: touch .receiver_ready (the ack)                           (:218-219)
    gather(POST /update_weights {"weight_dir": <posix|null>}, timeout_s=720) on every client               (:220-230)
finally:
    _resume_engines(all clients)           # POST /resume, gathered, retried (idempotent)                    (:425-433)
```

`setup_admin_plane` returns `DynamoAdminPlane` when `client_config.dynamo.enabled` (`:239-245`; §3.10).

### 3.5 Server routes (engine side)

All in `inference/vllm/server.py`:

| Route | Body | Action | Response | Cite |
|---|---|---|---|---|
| `POST /pause` | ignored (query params ignored) | `engine_client.pause_generation(mode="keep", clear_cache=False)` — `keep` = freeze in place (§3.6.1c) | `{"status":"paused"}` | `:66-70` |
| `POST /resume` | — | `engine_client.resume_generation()` | `{"status":"resumed"}` | `:73-76` |
| `POST /update_weights` | `{"weight_dir": str|null}` | `collective_rpc("update_weights_from_path", args=(weight_dir,))` on every worker | `{"status":"ok"}` | `:79-83` |
| `POST /init_broadcaster` | `{"host", "port", "timeout", "rank_offset", "inference_world_size", "session_id"="default"}` | `collective_rpc("init_broadcaster", args=(host, port, rank_offset, inference_world_size, timeout, session_id))` | `{"status":"ok"}` | `:133-146` |
| `POST /load_lora_adapter` | vLLM `LoadLoRAAdapterRequest` (`lora_name`, `lora_path`, …) | forces `load_inplace=True`, calls `OpenAIServingModels.load_lora_adapter`, then resets the stored request's `load_inplace=False` (sticky flag would force a disk reload every scheduler step) | `{"status":"ok"}` or vLLM `ErrorResponse` with its code | `:86-117` |
| `GET /liveness` | — | `collective_rpc("liveness_probe")` bounded by `server.liveness_timeout_seconds` (default 30) | `{"status":"ok"}` / 503 `{"status":"engine_unresponsive"}` | `:120-130`; config `configs/inference.py:23-24` |
| `POST /inference/v1/generate` | §3.3 | `PrimeRlServingTokens` | §3.3 | `serving_tokens.py` |
| stock vLLM | `/health`, `/v1/models`, `/v1/chat/completions`, `/metrics`, … | — | — | — |

Router-side (vllm-router fork, not in this repo): `POST /finish_session?session_id=<id>` releases a sticky session; the orchestrator calls it (5 s timeout, errors logged at debug) via a client pointed at the **router** URL for every trace id of a finished/failed/cancelled episode (`orchestrator/clients.py:96-100, 113-129`; `orchestrator/dispatcher.py:581-610`). Only active when `admin_base_url` is set.

### 3.6 Weight-sync handshake (all transports)

Markers (`transports/weights/base.py:23-26`): `.sender_ready`, `.receiver_ready`, `.started`, `.finished`, in `output_dir/broadcasts/step_<N>/` (`utils/pathing.py:287-292`).

`WeightSender.broadcast(model, step)` — `@final`, runs on **all trainer ranks** (`base.py:53-70`):
```
if master: rm -rf step_dir; mkdir; touch .sender_ready; poll(.receiver_ready, 0.1 s, timeout) ; touch .started
self._broadcast(model, step, step_dir)          # transport; ALL ranks (collectives)
if master: touch .finished; _clean(step)        # keep step and step-1 only (:101-108)
```
Non-master ranks enter `_broadcast` immediately; each transport must hold them back until the master finished the handshake (`:95-99`) — NCCL and filesystem do it with `dist.barrier()`; NIXL with the `dist.broadcast_object_list`/notifications inside.

`WeightReceiver` (`base.py:111-169`) runs in the **consumer** process (orchestrator; or the SFT online-eval process, `eval/online.py:58-69, 86-88`):
- `is_published(step)` = `.sender_ready` exists; `next_version(current)` = newest published step (or `current`) (`:139-146`).
- `wait_published(step)` polls `.sender_ready` every 0.2 s (`:148-156`).
- `_ack(step)` touches `.receiver_ready` (`:158-160`).
- `sync_startup(step, timeout)` = `wait_for(wait_published)` then `receive` (`:166-169`).

**Trainer cadence** (`trainer/rl/train.py`): at the first loop iteration, the master prunes `broadcasts/step_* > start_step-1` and all ranks broadcast the **startup policy** `v{start_step-1}` (v0 from scratch, v{ckpt} on resume) *before* waiting for the first batch (`:263-276`). After each optimizer step, `torch.cuda.synchronize(); torch.cuda.empty_cache()` then `broadcast(model, step=progress.step)` (`:581-597`). Trainer steps are 1-indexed; the model after step $N$ is policy $v_N$.

**Orchestrator cadence** (`orchestrator/watcher.py`, `orchestrator/orchestrator.py`): setup builds the receiver, calls `receiver.initialize()` (NCCL group / NIXL session), then `watcher.sync_startup(sync_version)` with `sync_version = resume_step or 0` and timeout `ckpt.wait_for_weights_timeout or 1200` (`orchestrator.py:304-320, 400`). The `WeightWatcher.start()` loop polls `next_version` every 1 s (`watcher.py:56-65`) and `apply_policy_update(next_step)` (`:79-121`):
```
async with update_lock:
    await receiver.wait_published(next_step)
    ckpt_step = next_step
    for obs in observers: await obs.on_version_pending(next_step)   # dispatcher drops over-stale rollouts BEFORE pausing
    await receiver.receive(next_step)                                 # the transport-specific apply
    policy.version = next_step
    notify on_new_version + update hooks (dispatch gate, eval trigger)
```
The drain-before-pause ordering exists because aborting a request triggers NIXL-connector cleanup that only propagates while the engine steps; aborting after resume races with KV-transfer completion flushes and trips the scheduler assertion (`watcher.py:93-102`; see also patch `patches.py:127-185`).

**Lockstep invariant.** Because the trainer blocks in `_wait_for_receiver_ready` on every version and cannot produce $v_{N+1}$ before $v_N$ is acked, `next_version` in practice returns `ckpt_step + 1`; it never skips versions in steady state. The orchestrator ships batch $s$ only once `policy.version >= s - 1 - TARGET_LAG` (`orchestrator.py:607-625`); `TARGET_LAG = 1` (`:95`). So at most two batches are in flight ahead of the applied policy (see 03/04 for S9 accounting).

**`receive` completes only after every engine applied the version.** `AdminPlane.update_weights` `gather`s `/update_weights` over all admin clients (`clients.py:220-230`); each route returns only when its `collective_rpc` has returned from every worker (and, behind one URL with internal DP, from every engine — vLLM 0.24 `DPLBAsyncMPClient.call_utility_async` gathers over all `core_engines`, `v1/engine/core_client.py:1443-1452`); resume is gathered after that. LoRA's `load_lora_adapter` also gathers over all clients (`:488`). Any failure raises out of `receive`, so `policy.version` (`watcher.py:116`) never advances on a partial apply. Engines do resume at slightly different instants, which only makes the version stamp conservative.

### 3.6.1 What a weight update does to the prefix cache, to in-flight requests, and to retries

**(a) Nothing in prime-rl resets vLLM's prefix cache on a weight update, for any transport.**
- `/pause` is hard-wired to `pause_generation(mode="keep", clear_cache=False)` (`inference/vllm/server.py:66-70`); the orchestrator (and Dynamo) also *request* `clear_cache=false` (`orchestrator/clients.py:420`; `inference/dynamo.py:355`). The only reset knob prime-rl touches is thus explicitly turned **off**.
- `/update_weights` only runs `collective_rpc("update_weights_from_path")` (`vllm/server.py:79-83`). The three worker implementations reload parameters (`worker/filesystem.py:28-55`, `worker/nccl.py:132-147`, `worker/nixl.py:476-581`) and never call a cache reset; `/resume` only calls `resume_generation()` (`:73-76`). A repo-wide search finds no `reset_prefix_cache` call in `src/prime_rl`, `deps/verifiers`, or `deps/renderers`.
- The worker-side reload touches model parameters inside GPU worker processes; the prefix-cache block-hash table lives in the EngineCore scheduler/KV-cache manager, which the reload does not reach. In vLLM 0.24 (nearest local source; pin 0.29) the only cache reset on this path is `EngineCore.pause_scheduler(..., clear_cache)` → `_reset_caches()` *iff* `clear_cache` (`v1/engine/core.py:722-751`, `:1684-1687`), and `AsyncLLM.pause_generation` documents `clear_cache` as "clear KV cache and prefix cache". So **the prefix cache survives every update**, and prime-rl relies on `cache_salt` instead (stated intent: `orchestrator/clients.py:463-465`).
- LoRA: `/load_lora_adapter` likewise does no reset (`clients.py:463-465`); vLLM keys LoRA prefix blocks by adapter id, and the id is reused in place (`vllm/server.py:88-99`) `[UNVERIFIED: whether vLLM's block hash includes adapter weights version — it cannot, so the salt is the only guard]`.

**(b) Consequence: a group can straddle a swap and reuse old-weight KV.** `cache_salt = str(group.policy_version_at_start)` for live-policy groups, fixed when the group is created (`orchestrator/dispatcher.py:532, 559-565`). Episodes of a group are dispatched one at a time (`episodes_to_schedule -= 1`, `:567`), and `try_schedule` prefers continuing an existing group (`:496-499`). The update barrier only blocks *scheduling during* the update (`policy_update_pending`, `:413-417, 450-457`) and is lifted by `on_new_version` (`:439-441`); the dispatcher then resumes the same partially-dispatched group with its **old salt**. Those later members hit prefix blocks (at least the shared task prompt) computed under the old weights while decoding under the new ones. The same applies to every later turn of an in-flight multi-turn episode (same salt for the episode's lifetime). The group is still dropped if it later exceeds `max_off_policy_steps` (`:421-431`). Quantitatively, the KV of the reused prefix is $K,V$ from $\theta_{k}$ while the query/new tokens use $\theta_{k+1}$; the sampled tokens' recorded logprobs are those of this hybrid computation, not of $\pi_{\theta_{k+1}}$ — a mismatch neither importance ratio nor off-policy accounting (keyed on `policy_version_at_start`, `:575`) models. Whether this is intended (salt = "version the group's KV belongs to") or a bug is not stated in code; flag for B/C.

**(c) `/pause` with `mode="keep"`: frozen in place, not aborted, not drained.**
- vLLM source (0.24, nearest local; pin 0.29 — re-check there): `AsyncLLM.pause_generation` documents `"keep"` as "Freeze requests in queue; they resume on resume_generation" (`v1/engine/async_llm.py:750-793`). `EngineCoreProc.pause_scheduler` sets `PauseState.PAUSED_ALL` (`v1/engine/core.py:1695-1696`); under `PAUSED_ALL` the scheduler's token budget is 0 (`v1/core/sched/scheduler.py:409-410`) and `get_num_unfinished_requests()` returns 0 (`:2112-2113`), so the engine goes idle immediately with requests (and their KV blocks) parked; `resume_scheduler` flips back to `UNPAUSED`. `"wait"` is the draining mode; `"abort"` aborts.
- Evidence requests *survive* the pause: the watcher cancels stale rollouts *before* pausing precisely because KV transfers keep completing during the pause and are flushed on resume (`orchestrator/watcher.py:93-102`); the KV-transfer patch notes the crash trigger is aborts, "not weight-update pause/resume" (`inference/patches.py:141-146`); `max_off_policy_steps` documents that "a rollout can span several weight updates" (`configs/orchestrator.py:603-604`); a request aborted by pause would surface as a failed rollout, and nothing in the client path expects that.
- The comment at `orchestrator/clients.py:386-388` ("`/pause` drains in-flight requests (mode="keep")") and the docstring of `_pause_engines` (`:416`) are **inaccurate** under these semantics; `/pause` should return fast.
- Conclusion: in-flight requests are **not aborted**; they stay in the engine and continue after `/resume` under the new weights, keeping their existing KV. Any token sampled after `/resume` uses $\theta_{k+1}$ over a KV prefix partly computed with $\theta_k$.

**(d) Retrying `/update_weights` after a timeout is NOT safe under NCCL (nor NIXL); it is safe under filesystem.**
- `_admin_post` retries on `httpx.TimeoutException`, any `TransportError`, and 5xx, per engine independently, with a 720 s read timeout and a 1440 s total budget (`orchestrator/clients.py:370-412, 220-230`). A client read timeout does **not** cancel the server-side `collective_rpc` already running in the workers.
- NCCL: each `/update_weights` makes every worker of that engine enter a fresh `receive_integer` + per-layer broadcasts (`worker/nccl.py:77-85, 141`). The trainer sends once per version (`transports/weights/nccl.py:143-159`). vLLM 0.24 runs utility calls inline on the EngineCore loop (`v1/engine/core.py:1386-1399`), so a timeout retry **queues behind** the running first `collective_rpc`, then enters `receive_integer` waiting for a *next* broadcast that never starts (this orchestrator cannot ack v$_{N+1}$ while stuck); the queued `/resume` waits too, so the engine stays frozen until the 1440 s budget expires and the orchestrator dies. If the first attempt failed mid-stream (5xx), the collective is desynced and a retry misreads tensor bytes as metadata; per-engine retries also desync engines. Only a connect-error retry is benign. Trigger: any transfer slower than 720 s/attempt (`clients.py:392`); the trainer's 1200 s timeout bounds the *ack wait*, not the transfer. Dynamo's NCCL path deliberately does *not* retry `collective_rpc` (single POST under `asyncio.timeout(720+15)`) and marks the plane terminal on any failure (`inference/dynamo.py:309-327, 352-384`).
- NIXL: a retry re-runs `apply_transfer_plan`, which waits for group notifications at the *current* generations; notifications consumed by the first attempt were removed from the local buffer (`transports/weights/nixl/agent.py:107-111`) and generations already bumped are not rolled back (`worker/nixl.py:525`), so the retry waits for notifications the trainer will not send until the next version → timeout or cross-version mixing. Unsafe.
- Filesystem: `update_weights_from_path(step_dir)` re-reads the same immutable safetensors dir (`worker/filesystem.py:28-55`) → idempotent; safe (only wasted time).

### 3.7 Filesystem transport

**Sender** `FileSystemWeightSender._broadcast` (`transports/weights/filesystem.py:34-58`):
- LoRA (`lora_config` set): every rank materializes `get_lora_state().adapter_state_dict()` DTensors with `full_tensor()`; master copies to CPU and writes `adapter_model.safetensors` (unsharded, `ADAPTER_SAFE_WEIGHTS_NAME`) + `adapter_config.json` via `save_lora_config(rank, alpha, dropout)` (`:35-52`; `utils/weights.py:45-91`).
- Full model: `dist.barrier()` (holds non-masters until the handshake), `gather_weights_parallel(model)` (every rank does every DTensor `full_tensor()` after downcast to bf16, or fp32 when `keep_in_fp32_for_weight_transfer(key)`; each rank keeps only the keys it owns under a byte-balanced, layer-atomic partition, `utils/weights.py:131-199`), `convert_state_dict_to_hf` (prime → HF via the model's conversion chain, or `revert_weight_conversion` for plain HF models, `:94-112`), then `save_state_dict_parallel` (`:202-252`): each rank writes `tmp-rank{r}-*.safetensors` (≤5 GB shards), `all_gather_object` of shard maps, each rank renames its own files to `model-XXXXX-of-YYYYY.safetensors` (or `model.safetensors` if one shard), barrier, master writes `model.safetensors.index.json` last — **only when there is more than one shard** (`:248`); the completion signal is `.finished`, not the index. **No `config.json`, tokenizer or generation config is written** — the step dir is weights-only, which suffices because the vLLM worker streams the safetensors into its already-built model (below); it is not a loadable HF checkpoint on its own.
- LoRA keys are `"{module_fqn}.lora_{A,B}.weight"` (e.g. `model.layers.0.self_attn.q_proj.lora_A.weight`, after `convert_adapter_to_hf`), **without** PEFT's `base_model.model.` prefix (`trainer/lora.py:58-73`); `adapter_config.json` = `{peft_type: LORA, r, lora_alpha, lora_dropout, target_modules: sorted suffixes, modules_to_save, ...}` (`:314-360`). vLLM accepts bare names: `parse_fine_tuned_lora_name` strips the prefix only if present (vLLM 0.24 `lora/utils.py:171-200`), and prime-rl's patched loader uses the same parser (`inference/patches.py:530`).

**Receiver** `FileSystemWeightReceiver.receive(step)` (`filesystem.py:68-75`): **ack first** (the trainer then writes), `await wait_for_path(.finished)` (1 s poll), then:
- if `adapter_config.json` exists → `load_lora_adapter(admin_plane, model_name, step_dir)`: `POST /load_lora_adapter {"lora_name": <base model name>, "lora_path": <step_dir>}` to every engine, **no pause** (in-place adapter reload is vLLM-native), 30 s per attempt / 120 s total, retries on 404/500/transport (NFS propagation) (`orchestrator/clients.py:436-488`).
- else `admin_plane.update_weights(step_dir, transport="filesystem")` (pause → `/update_weights` → resume).
- Dispatch cost: the watcher calls `on_version_pending` *before* `receive` (`watcher.py:103-113`), which sets the dispatcher's `policy_update_pending` (`dispatcher.py:413`). New scheduling therefore stops for the **whole** trainer-side export (gather + safetensors write + `.finished`) plus the load, not just the engine pause. In-flight requests keep generating until the pause.

**Worker** `FileSystemWeightUpdateWorker.update_weights_from_path(weight_path)` (`inference/vllm/worker/filesystem.py:28-55`): unwrap `model_runner.model.runnable` if compiled; build a `DefaultModelLoader.Source(weight_path, revision=None, prefix="")` and its weights iterator; `load_weights_checkpoint_layerwise` = `initialize_layerwise_reload(model)` → `model.load_weights(iter)` → `finalize_layerwise_reload(model, model_config)` inside `torch.device(device)` + `set_current_vllm_config` (`worker/weight_transfer.py:12-23`). vLLM's layerwise reload re-runs `process_weights_after_loading` per layer (so online FP8 quantization etc. are redone) `[UNVERIFIED: vLLM-internal]`.

### 3.8 NCCL transport

**Group formation (once).**
- Trainer: `NCCLWeightSender.__init__` → `NCCLBroadcaster(host, port, rank=0, world_size=inference_world_size+1, device, timeout)`; **only the trainer master** calls `disable_nccl_p2p_if_unavailable()`, `StatelessProcessGroup.create(host, port, rank=0, world_size, store_timeout=timeout)`, `PyNcclCommunicator(pg, device)` (`transports/weights/nccl.py:117-140, 162-179`). vLLM's `StatelessProcessGroup.create` makes rank 0 bind a listen socket to exactly `(host, port)` and host the `TCPStore` (vLLM 0.24 `distributed/utils.py:473-486`), so `weight_broadcast.host` on the trainer is the bind address: `"localhost"` default = loopback-only; `"0.0.0.0"` in multi-node (`configs/rl.py:788-789`). The store also runs a `world_size` wait (`store_timeout = weight_broadcast.timeout`).
- Orchestrator: `NCCLWeightReceiver.initialize()` → `AdminPlane.initialize_nccl(host, port, timeout, inference_world_size)` → for admin client $i$, `POST /init_broadcaster {"host","port","rank_offset": i·(W/len(clients)),"inference_world_size": W,"timeout"}` (no `session_id`), all concurrently (`nccl.py:199-205`; `orchestrator/clients.py:165-203`). In multi-node the orchestrator is launched with `--weight_broadcast.host $MASTER_ADDR` (trainer node 0) (`templates/multi_node_rl.sbatch.j2:137, 563`).
- Each vLLM worker: `NCCLWeightUpdateWorker.init_broadcaster` computes `local_rank = self.device.index`, `global_rank_inference = rank_offset + local_rank`, and joins with **rank `global_rank_inference + 1`**, world `W + 1` (`inference/vllm/worker/nccl.py:91-126`). The HTTP call blocks until the group forms.

Rank layout:
$$\text{rank}(\text{trainer master})=0,\qquad \text{rank}(\text{server } i,\ \text{GPU } d)=1 + i\cdot\tfrac{W}{n_{\text{servers}}} + d,\qquad W = \texttt{inference\_world\_size}.$$
`W` is auto-set to `dp × tp` single-node (`configs/rl.py:458-469`), `total_infer_nodes × gpus_per_node` multi-node (`:778-792`) and for P/D (`:822-826`). The inference side **derives nothing**: `/init_broadcaster` forwards the orchestrator's `W` and `rank_offset`, and each worker uses `device.index` (`worker/nccl.py:111-123`). `device.index` is the worker's index *within its server's `CUDA_VISIBLE_DEVICES`*; with vLLM-internal DP it is `dp_local_rank·tp + tp_rank` (vLLM 0.24 `v1/worker/gpu_worker.py:255-313`), so one admin URL fronting `dp` engines maps to ranks $1..dp\cdot tp$. Each admin URL must own exactly `W/n` GPUs.

**Bug — single-node `W` undercount (local `rl`).** Pydantic runs `RLConfig`'s after-validators in definition order: `auto_setup_weight_broadcast` (`configs/rl.py:439-495`) computes `W = data_parallel_size × tensor_parallel_size` **before** `auto_setup_deployment` (`:687-709`) rewrites `data_parallel_size = num_infer_gpus // tp` whenever `dp·tp ≠ num_infer_gpus` (overriding even an explicit DP, so over-counts are possible too) and raises `api_server_count`; nothing recomputes `W` single-node. Executed against the pin: `configs/basic/wordle/rl.toml` with `num_infer_gpus=3` → `dp=3, W=1`; `num_infer_gpus=4, tp=2` → `dp=2, W=2`; the shipped **`examples/basic/hendrycks-sanity/rl.toml` (4 infer GPUs, default NCCL) → `dp=4, W=1`**. `rl_local` writes the sub-config JSONs from this object (`entrypoints/rl.py:89-126`). Single-node **SLURM** is unaffected: it re-runs `rl @ rl.json` inside the allocation (`entrypoints/rl.py:595-596`, `single_node_rl.sbatch.j2:126`), and the re-parse sees the filled `dp` (verified → `W=3`). Multi-node/P/D set `W` after DP fill. Consequence (vLLM 0.24): workers of DP engines ≥1 get rank $> W$ and hit `assert self.rank < self.world_size` (`distributed/utils.py:203-204`) → `/init_broadcaster` 500, **swallowed** by `initialize_nccl` (§7) → the first `/update_weights` fails on those engines → retries (§3.6.1d) → orchestrator dies at startup sync. Configs that set `inference.vllm.data_parallel_size` explicitly (wordle, nightly-fft) sidestep it. NIXL has the same `W` (used as the ModelExpress inference count, `nixl.py:384-388`) → the trainer connects only to inference ranks $0..W{-}1$; the other engines never receive group notifications and their `/update_weights` times out.

**Per-version transfer.**
```
orchestrator                     vLLM workers (all)                trainer master           trainer non-masters
receive(step):
 POST /pause ×n  ──────────────► pause(keep)
 on_paused: touch .receiver_ready ───────────────────────────────► sees ack; touch .started
 POST /update_weights ×n ───────► update_weights_from_path:        _broadcast: barrier ◄──── barrier
                                  receive_integer (bcast src 0) ◄─ broadcast_integer(L+1)
                                  for each of L+1 layer dicts:
                                    size, pickle meta, per-dtype  ◄─ resolve_dtensors (ALL ranks full_tensor)
                                    concatenated flat tensors         preprocess_layer_checkpoint (→HF)
                                  load_weights_checkpoint_layerwise   broadcast_state_dict (master only)
 ◄── 200 ×n                                                        touch .finished
 POST /resume ×n
```
Wire format per "layer state dict" (`nccl.py:30-64`, receiver `worker/nccl.py:31-57`):
1. `size_tensor`: `torch.long[1]` = len(pickle bytes).
2. `state_tensor`: `uint8[size]` = `pickle.dumps({dtype: [(key, shape, numel), ...]})` (dict in first-seen dtype order).
3. For each dtype group: one flat `concatenated` tensor of that dtype = `cat([v.flatten() for v in group])`.
The very first message of a version is `long[1] = num_layers + 1` (`nccl.py:24-27, 148-151`). Dict order: non-layer keys first (`layer_idx=-1`), then `layer_prefix{i}.*` for $i=0..L-1$ (`:67-81`), where `L = get_max_layer_num(...)` (count, `trainer/conversion_utils.py:4-13`) and `layer_prefix` from `get_layer_prefix(model.config)`.
- Names are **HF checkpoint names** (after `model.convert_layer_to_hf` for prime-format custom models, else `revert_weight_conversion`) (`nccl.py:103-114`).
- dtype: DTensors are cast to `bf16` (hard-coded `self.dtype`, `:129`) or fp32 for `keep_in_fp32_for_weight_transfer(key)` *before* `full_tensor()`; non-DTensor tensors (buffers) go in their own dtype (`:84-100`).
- Receiver yields zero-copy views into the concatenated buffer into vLLM `load_weights`; buffers are freed after each group (`worker/nccl.py:40-57`).

Non-master trainer ranks run `resolve_dtensors` + conversion in lockstep with the master (they participate in the `full_tensor` all-gathers); `dist.barrier()` in `_broadcast` ensures they don't enqueue those collectives before the receiver paused inference (otherwise NCCL watchdog kills them, `nccl.py:182-190`).

### 3.9 NIXL transport (receiver-driven RDMA)

**Bug — stale q/k/v under the default `qkv` fusion.** The trainer's `model.state_dict()` is read **once** (`initialize_transfer`, `nixl.py:326-338`, first broadcast only) and every later `copy_to_staging` copies from the captured `source_tensor`s. With runtime fusions on by default (`fusions.enabled = ["gate_up","qkv"]`, `configs/trainer.py:93-94`; generic attention supports `qkv`, `trainer/models/layers/attn.py`), `q/k/v_proj.weight` come from a state-dict post-hook splitting the packed `qkv_proj.weight` along dim 0 (`trainer/models/fusions.py:47-67, 86-111`). Under FSDP's default `Shard(0)` on ≥2 ranks the split redistributes and yields **Replicate copies** (docstring `fusions.py:17-20`; executed on 2 gloo ranks, torch 2.11: outputs are `Replicate()` and do not see an in-place update of the packed tensor; on 1 rank they alias). NIXL thus re-sends the startup q/k/v forever (rank 0 serves the copy, `nixl.py:159-171`, restaged each step `:423-424`). With `shard_fused_on_dim1=true` (opt-in, `configs/trainer.py:99`) the split is a live `Shard(1)` view, but `collect_local_tensor_shards` assumes dim-0 shards (offset $=0$ on every rank, `nixl.py:174-187`) and `route_sharded_tensor` fails at plan build (`no trainer shard owns element`, `tensor_routing.py:57-63`). No config check; NCCL/FS re-read `state_dict()` per broadcast and are fine. Unaffected: 1-GPU trainer, fusions off, LoRA (`trainer/model.py:1000-1001`). Workaround: `trainer.model.fusions.enabled = []`. [static + DTensor probe; no NIXL run]

Components: `NixlAgent` wrapper over `nixl_cu13`/`nixl` with the **UCX** backend only (`transports/weights/nixl/agent.py:34-43`); `ModelExpressSession` over the ModelExpress gRPC client for metadata publish/discover (`model_express.py:17-97`); `TrainerTensorTable` msgspec/msgpack schema (`trainer_tensor_table.py:8-60`); lazy-graph tracing (`graph.py`); shard routing (`tensor_routing.py`); a `cudaMalloc`-backed `torch.cuda.MemPool` for registered arenas (`cuda_malloc_memory.py:22-67`).

**Discovery (ModelExpress).** Every participant publishes `WorkerMetadata{worker_rank, nixl_metadata: bytes}` under identity `SourceIdentity{mx_version:"0.3.0", mx_source_type: WEIGHTS, model_name:"prime-rl-weights", extra_parameters:{role, session_id}}` (`model_express.py:25-62`). `wait_for(role, count)` polls `list_sources(role)` every 50 ms until ranks `0..count-1` are all present (`:68-91`). RPCs retry `UNAVAILABLE`/`DEADLINE_EXCEEDED` for up to 120 s (`:33-49`). Roles/ranks:
| Role | rank | worker_id | nixl_metadata |
|---|---|---|---|
| `trainer` | 0 | `trainer-table` | encoded `TrainerTensorTable` (includes every serving trainer rank's agent metadata) (`nixl.py:344-353`) |
| `inference` | `rank_offset + device.index` | `inference-<global_rank>` | its NIXL agent metadata (`worker/nixl.py:104-115`) |
| `orchestrator` | 0 | `orchestrator` | its NIXL agent metadata (`nixl.py:458-475`) |

**Trainer side — `NIXLWeightSender`** (`transports/weights/nixl/nixl.py:72-447`):
- *Serving ranks*: all ranks, or only `dp_replicate` local-rank 0 when replication is enabled (`:96-100`). Each creates `NixlAgent("trainer-<host>-r<rank>")` and sets UCX env (`:82-84`).
- `initialize_transfer` (first broadcast only, `:326-359`): groups = `["non_layer", "layer.<i>"...]` by regex `(?:^|\.)layers\.(\d+)(?=\.|$)` over floating tensors (`:45, 102-119`). `collect_local_tensor_shards` (`:121-188`): skip non-float; `wire_dtype = fp32 if keep_in_fp32(name) else bf16`; plain tensors and fully-replicated DTensors are served only by the master; FSDP DTensors contribute this rank's **contiguous dim-0 shard** at element offset `global_offset[0] · prod(shape[1:])`. Staging arenas: one per wire dtype, size `staging_buffer_count × largest group` where `staging_buffer_count = min(#groups, 2 if overlap_transfer_and_replay else 1)`; group $g$ uses slot $g \bmod \text{count}$; arenas allocated from the cudaMalloc pool and registered with NIXL (`:190-238`). Fragments (per-rank tables with `addr = staging_tensor.data_ptr()`) are `dist.gather_object`ed to rank 0, merged (agent index = order in the gathered list; shards sorted by offset) and published to ModelExpress (`:240-324`). **Names are trainer (prime) state-dict names, not HF names.**
- `_broadcast` (`:372-447`):
  1. First call only: master waits in ModelExpress for the orchestrator and `inference_world_size` inference workers, fetches their metadata; `dist.broadcast_object_list` shares the inference metadata to all ranks; serving ranks `add_remote_agent` + `make_connection` to every inference peer; master connects to the orchestrator peer; `group_generations = [0]*#groups` (`:377-408`).
  2. Master sends notification `policy:<step:016x>:ready` to the orchestrator (`:410-414`; format `agent.py:30-31`).
  3. For each group $g$: if $g \ge$ count, `finish_transfer_group(g - count)` = wait for the group notification `"<g:08x>:<gen:016x>"` from **every** inference peer (the READ-completion notification = buffer credit), then bump its generation (`:361-370, 416-420`). Serving ranks `copy_to_staging()` their shards (fp32 master → wire dtype cast happens in `copy_`), `cuda.synchronize()`, send the group notification to all inference peers (`:422-428`).
  4. Drain the last `count` groups' credits (`:435-437`); master waits for `policy:<step>:complete` from the orchestrator; `dist.barrier()` (`:439-447`).

**Orchestrator side — `NIXLWeightReceiver`** (`nixl.py:450-502`): creates `NixlAgent("orchestrator-<host>-r0")` with `set_ucx_env_defaults(0)` (`:455-456`). `initialize()` → `init_nixl_broadcast` = `POST /init_broadcaster {host, port, rank_offset: i·(W/n), inference_world_size, timeout, session_id}` via `_admin_post(timeout_s=max(300, timeout))` to every engine (`orchestrator/clients.py:491-520`), then publishes its own metadata. `receive(step)`: **ack first**; on first use fetch the trainer table from ModelExpress and connect to trainer agent 0; wait for `policy:<step>:ready` from the trainer; `admin_plane.update_weights(None, transport="nixl")` (pause → `/update_weights {"weight_dir": null}` → resume); send `policy:<step>:complete` to the trainer.

**Inference side — `NIXLWeightUpdateWorker`** (`inference/vllm/worker/nixl.py:86-593`):
- `init_broadcaster` creates `NixlAgent("inference-<host>-r<global_rank>")`, a `ModelExpressSession(role="inference")`, publishes metadata (`:94-123`). `inference_world_size` is ignored.
- `initialize_transfer` (first `/update_weights` only, `:125-145`): wait for the trainer table; **trace** (`:147-203`): under `initialize_layerwise_reload`, wrap every layer tensor's `weight_loader` so the recorder knows the active destination, then call vLLM's real `model.load_weights(make_hf_lazy_weights(table, ...))`. `make_hf_lazy_weights` builds a `LazyWeight` per trainer tensor and applies the trainer model class's **prime→HF `conversion_chain`** lazily (`graph.py:312-346`; it imports `prime_rl.trainer.models` inside the vLLM worker). `LazyWeight` is a `torch.Tensor` wrapper subclass that records only `SUPPORTED_OPS` (narrow/select/view/reshape/getitem/unsqueeze/squeeze/transpose/t/permute/flatten/contiguous/chunk/split/unbind/to/float/bfloat16, `graph.py:35-54`) and turns the terminal `copy_` into a `RecordedCopy(source_name, ops, destination_module, destination_name, destination_offset/shape/stride, is_persistent)` without touching memory. Any other op that reaches a `LazyWeight` — `torch.cat`, `torch.stack`, `new_empty`, arithmetic, in-place ops other than `copy_` — falls through to `__torch_dispatch__` and raises `UnsupportedOpError` at trace time (`:254-309`). Only the prime→HF direction of the conversion chain runs here (unbind/split-style ops), and a vLLM model loader that concatenates or allocates from the source weight is incompatible with NIXL. Whether specific upstream loaders do this (e.g. Qwen3.5-MoE, GPT-OSS) is vLLM-0.29-internal; prime-rl's own GPT-OSS loader patch only `copy_`s (`inference/vllm/gpt_oss_weight_loading.py:42`). Only bf16/fp32 on both sides (`:55, 228-232`). Layerwise state is restored afterwards (`worker/nixl.py:195-232`).
- `build_transfer_plan` (`:234-269`): `plan_tensor_replay` splits each op chain into the longest prefix that is still a contiguous, same-dtype view of the trainer root (transferred directly) and a replay suffix run locally on the receive arena (`graph.py:92-118`). Receive arenas per dtype sized `staging_buffer_count × max per-group elements` (`worker/nixl.py:287-321`). `route_sharded_tensor` maps each (possibly strided) source view onto trainer shards, emitting `(agent, source_addr, destination_addr, nbytes)` contiguous runs (`tensor_routing.py:21-83`). Per group, one prepared READ per trainer agent, with the agent order rotated by `rank % n_agents` to spread load across source rails (`worker/nixl.py:452-474`). A vLLM reload layer must read from a single trainer group (`:346-357`).
- `update_weights_from_path(None)` → `apply_transfer_plan` (`:476-581`): under `initialize_layerwise_reload`, for each group: wait for the group notification from **all** trainer serving agents (generation-versioned), post all READs with that notification string (NIXL delivers it to the trainer on completion = the credit), wait for DONE, bump generation; then `replay_group`: `materialize_layer`, zero the destination tensors, `destination.as_strided(...).copy_(apply_chain(staging, replay_ops))` for each copy, rerun `quant_method.process_weights_after_loading`, restore kernel tensors (`:532-559, 583-593`). With `receive_buffer_count > 1` a 1-thread executor prefetches group $g+1$ while replaying $g$ (`:563-579`). Finally `finalize_layerwise_reload`, `update_mla_absorbed_weights` (recompute `W_UV`/`W_UK_T` from `kv_b_proj`, `worker/weight_transfer.py:26-54`), `cuda.synchronize`.

Notification/ordering sequence for one version:
```
trainer master ──policy:S:ready──► orchestrator  ──/pause,/update_weights──► workers
serving ranks ──"g:gen"──► every worker ; worker READs (notif "g:gen") ──► every serving rank (credit)
workers done ─► orchestrator /resume ─► orchestrator ──policy:S:complete──► trainer master ─► barrier
```

**UCX defaults** (`agent.py:151-210`): if `UCX_NET_DEVICES` unset, pick the ACTIVE InfiniBand port with minimum sysfs PCI-path distance to this GPU (error if none), plus `UCX_MAX_RNDV_RAILS=1`, `UCX_MAX_RMA_RAILS=1`; setdefault `UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy`, `UCX_IB_GPU_DIRECT_RDMA=y`, `UCX_RNDV_SCHEME=get_zcopy`, `UCX_RNDV_THRESH=0`, `UCX_MEMTYPE_CACHE=n`.

### 3.10 Dynamo admin plane

`DynamoAdminPlane(AdminPlane)` (`inference/dynamo.py:154-429`) replaces static `admin_base_url` with discovery from an ai-dynamo frontend: `GET <discovery_url>/v1/rl/workers` returns `{"protocol_version": 1, "workers": [{model, instance_id, admin_base_url, world_size, error?}]}` (`:26-33, 119-151`). `discovery_url` defaults to `base_url` with port+1 (`:103-116`). Constraints: exactly one worker for the model with `world_size == 1`; `admin_base_url` must be a bare http(s) origin on the discovery host (`:40-82`). Readiness pins the topology only after **two identical consecutive snapshots** (`:199-244`); every mutation re-verifies (`ensure_topology_current`, `:272-307`). Non-NCCL updates delegate to the base class (`:337-346`); NCCL uses Dynamo's own `/pause` (with `mode=keep, clear_cache=false` params), `POST /collective_rpc {"method", "timeout", "args", "kwargs": {}}` expecting exactly `{"results": [None]}`, and `/resume`, with a state machine `uninitialized → initializing → ready ↔ terminal`: the state is set `terminal` *before* the pause and restored to `ready` only after a successful resume (`:352, 384`), so any failure mid-update is terminal ("restart required") (`:309-425`). `/pause` still goes through the retrying `_admin_post`; the `collective_rpc` is a single POST (no retry). `initialize_nccl` requires `inference_world_size == 1` and sends `init_broadcaster(host, port, 0, 1, timeout, "default")` (`:387-425`). `client.dynamo` defaults to `None`; a present `DynamoConfig` defaults `enabled=True` (`configs/shared.py:164-196`). Tests: `tests/unit/inference/test_dynamo.py`.

### 3.11 Batch (rollout) transport

Factory: `setup_batch_sender(output_dir, data_world_size, current_step, transport)` / `setup_batch_receiver(output_dir, data_rank, current_step, transport)` (`transports/batch/__init__.py:21-40`). Serialization: `msgspec.msgpack.Encoder()` / `Decoder(type=list[MicroBatch])` (`base.py:15, 34`). The payload per data rank is `list[MicroBatch]` (the rank's micro batches for one step); both implementations assert the grid has `data_world_size` rows of equal length (`filesystem.py:20-22`, `zmq.py:68-70`).

**ZMQ** (`transports/batch/zmq.py`):
| Socket | Side | Type | Endpoint | Opts |
|---|---|---|---|---|
| data | orchestrator | `PUB` (asyncio ctx) `bind` | `tcp://{host}:{port}` | `SNDHWM=hwm` |
| ready | orchestrator | `PULL` `bind` | `tcp://{host}:{port+1}` | `RCVHWM=hwm` |
| data | each trainer rank | `SUB` `connect`, `SUBSCRIBE b"data_rank|<r>|"` | same | `RCVHWM=hwm` |
| ready | each trainer rank | `PUSH` `connect`, sends `str(data_rank)` once at init | same | `SNDHWM=hwm` |
(`:24-34, 94-112`). Defaults `host="localhost"`, `port=5555`, `hwm=10` (`configs/shared.py:242-252`). Send: on first `send`, block until READY from all `data_world_size` distinct ranks (`:47-64`), then for each rank `send_multipart([b"data_rank|<r>|", msgpack], copy=False)` (`:66-80`). Receive: `poller.poll()` for `wait()`, `recv_multipart` → decode (`:121-134`). PUB never blocks; a message past a subscriber's HWM is silently dropped — safe only because the dispatch gate bounds in-flight steps to `TARGET_LAG + 1` (`:13-21`). The topic's trailing `|` matters: ZMQ `SUBSCRIBE` is a prefix match, so `data_rank|1|` does not also catch ranks 10–19. Multi-node: orchestrator and trainer are both launched with `--rollout_transport.host $ORCH_ADDR` (`templates/multi_node_rl.sbatch.j2:153-161, 519, 565`); `ORCH_ADDR` is a SLURM **hostname** (`scontrol show hostnames`, `:85, 160`), and binding a PUB to a hostname works with the pinned pyzmq 27.1.0 / libzmq 4.3.5 (bind-tested locally) — it binds only the interface that name resolves to.

**READY barrier and CP>1.** READY dedups by `data_rank`, so with CP>1 one peer's READY satisfies the rank. Masked in practice: every trainer rank builds its receiver (`trainer/rl/train.py:231-236`) before the collective startup broadcast (`:267-273`), which the orchestrator must receive (`sync_startup`) before it can ship anything; the residual window is subscription propagation (ms) vs rollout time (s).

**Wire encoding (executed at the pinned msgspec 0.21.1).** `array_like` + `omit_defaults` still encodes **all 19 `MicroBatch` positions** (unset optionals as `nil`). Old decoders ignore extra trailing elements; new decoders default missing trailing ones ⇒ appending a defaulted field is compatible both ways; inserting/reordering is not.

**Filesystem** (`transports/batch/filesystem.py`): writes `output_dir/batches/step_<N>/rank_<r>.bin` via `.tmp` + `rename` in a worker thread (`:17-35`); receiver `sync_wait_for_path` (1 s poll) then reads and decodes (`:46-62`). Step counters start at `progress.step` on both sides (`orchestrator.py:264-266`; `trainer/rl/train.py:231-236`). The `rl` launcher calls `clean_future_steps` (`utils/pathing.py:379-397`): on resume it deletes `batches/step_* > resume_step` and `broadcasts/step_* >= resume_step`; on a fresh run (`-1`) it wipes both (`entrypoints/rl.py:681-686`). Nothing garbage-collects `batches/` during a run.

**Trainer consumption** (`trainer/rl/data.py:178-267`): `dp_rank = rank // (world_size // dp_world_size)` — every non-DP rank (e.g. CP peers) of a DP group creates its own receiver on the same `dp_rank` (`:190-193`); the contiguous-block formula is correct only because `cp` is the innermost mesh dim (05). The orchestrator's `data_world_size` is `num_train_workers` (`orchestrator.py:264-266`), a separate config value that must equal the trainer's DP size; nothing cross-checks it at runtime. Too many rows makes the ZMQ sender wait forever for READY; too few leaves trainer ranks starving. `_micro_batch_to_tensor` rebuilds tensors: `mm_kwargs` via `torch.frombuffer(dtype, shape)`, `routed_experts` → `int32[1, seq, layers, topk]`, `sampling_mask` → `int32[1, T, max_count]` padded with −1, lists → `long`/`float`/`bool` tensors with a leading batch dim.

### 3.12 Routers, multi-replica, wide-EP, P/D, llm-d, KV offload

**vllm-router (default)** — PrimeIntellect fork, wheel `vllm_router-0.2.1` (`pyproject.toml:272-274`). Local: `vllm-router --policy <policy> --host --port server.port --worker-urls http://<host>:backend_port --intra-node-data-parallel-size <dp_local or dp> --request-id-headers x-session-id --request-timeout-secs <14400> --worker-startup-timeout-secs 4200 --prometheus-port server.port+21000` (`entrypoints/inference.py:165-191`). SLURM: same flags with `--worker-urls <per-rank urls>` (regular) or `--vllm-pd-disaggregation --prefill <url>… --decode <url>…` (P/D) (`templates/_launch_router.sh.j2:71-85`). Policies (`configs/inference.py:309-318`; `docs/inference.md:192-198`): `sticky_least_loaded` (default; first request of an `X-Session-ID` goes to the least-loaded worker, later turns stick; released via `/finish_session`), `consistent_hash` (hash of session id), `round_robin`. For router replay in P/D, the router merges prefill and decode `routed_experts` objects — this is why only the `{data, shape, start}` object form is allowed (`patches.py:388-399`). `[UNVERIFIED: router internals — merge semantics, how --intra-node-data-parallel-size addresses DP ranks]`.

**Multi-node "multi-replica"** (`inference.deployment.type="multi_node"`): every node runs `dp_per_node = gpus_per_node / tp` independent single-DP-rank servers on `BACKEND_PORT + d`; the router on inference node 0 fronts all of them (`templates/inference.sbatch.j2:315-344`; `multi_node_rl.sbatch.j2:459-488`).

**Wide-EP** (`enable_expert_parallel=true`, multi-node): each rank still is its own API server, but ranks form one **external-LB DP group** (`data_parallel_size = INFER_GROUP_DP`, `data_parallel_rank = global`, `data_parallel_address` = replica head IP/host, shared `data_parallel_rpc_port` default 13345) so EP all2all spans nodes (`multi_node_rl.sbatch.j2:470-480`; `configs/inference.py:95-96`; validation `configs/rl.py:726-759`). Note the standalone `inference.sbatch.j2` roots EP groups per node (`:331-336`) — see §7.

**P/D disaggregation** (`type="disaggregated"`): validator forces `use_pd_kv_transfer=True`, EP on, EPLB off, `data_parallel_size(_local) = gpus_per_node/tp`, `api_server_count = dp_per_node` (`configs/inference.py:550-568`). `kv_transfer_config = NixlConnector(kv_role=kv_both, num_threads=1)` (+ offload connector via `MultiConnector`) (`:578-605`). Each node is prefill or decode by its rank in the replica; prefill uses `all2all_backend=deepep_high_throughput`, decode `deepep_low_latency` + `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` and `UCX_NET_DEVICES=mlx5_0:1`; per-rank `VLLM_NIXL_SIDE_CHANNEL_PORT=5600+k`; `VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP`; `GLOO_SOCKET_IFNAME` from the IP's interface; role-specific env vars and vLLM overrides (`multi_node_rl.sbatch.j2:356-457`). `ADMIN_URLS` lists all prefill ranks then all decode ranks, per replica (`:97-115`), and `inference_metrics_roles` is auto-built in the same order (`configs/rl.py:813-821`). Weight updates therefore go to prefill and decode engines alike.

**llm-d router** (`router.type="llm-d"`, SLURM multi-node/P/D only, `configs/inference.py:321-365, 525-532`): `epp` (Endpoint Picker; gRPC 9002, health 9013 — avoids mooncake's 9003, metrics 9090) + `envoy` (listens on `ROUTER_PORT`, routes `/v1/` and `/inference/v1/` through `ext_proc` to EPP which sets `x-gateway-destination-endpoint`; `ORIGINAL_DST` cluster; circuit breakers raised to 100000) (`_launch_router.sh.j2:11-70`; `llmd/envoy.yaml.j2`). Endpoints are file-discovered (`llmd/endpoints.yaml.j2`, hostnames substituted by IPv4). Non-P/D profile: `scorers` weights (default `prefix-cache-scorer: 3.0, active-request-scorer: 2.0`) + `max-score-picker` (`llmd/epp_estimate.yaml.j2`). P/D: `prefix-based-pd-decider(nonCachedTokens=16)` → prefill profile (base + `queue-scorer: 2, kv-cache-utilization-scorer: 2`) or decode-only; decode endpoints point at a per-decode-node `pd-sidecar` on `decode_sidecar_port + rank` (default 8300) that orchestrates the remote prefill (`llmd/epp_pd.yaml.j2`; `multi_node_rl.sbatch.j2:426-445`). Binaries built by `scripts/install_llmd.sh` from a fork (`S1ro1/llm-d-router@1ca4243`) that adds P/D for `/inference/v1/generate` (`install_llmd.sh:8-16, 27-28`). llm-d rejects routed-expert return (`configs/inference.py:500-509`, `configs/rl.py:588-605`).

**KV cache offload** (`inference.kv_cache_offload`, `configs/inference.py:224-288`): `cpu` tier required; optional `disk`. `native` → `OffloadingConnector` with `cpu_bytes_to_use` (+ `TieringOffloadingSpec` with an `fs` secondary tier) (`:251-265`). `mooncake` (SLURM only, `configs/rl.py:673-685`) → `MooncakeStoreConnector`; the template starts one `mooncake_master` (rpc 50051, HTTP metadata 8080, `default_kv_lease_ttl=3600000`) on the head node and one `mooncake_client` per node (port 50052, `-protocol=rdma`, `-global_segment_size=<cpu bytes>`), writes `MOONCAKE_CONFIG_PATH` JSON (`mode: standalone-store`, `local_buffer_size` 4 GiB), sets `PYTHONHASHSEED=0` (blocks keyed by model + rank + content hash across nodes) (`templates/_mooncake_store.sh.j2:11-42`); per-rank `VLLM_RPC_BASE_PATH` avoids lookup-socket collisions (`_launch_rank.sh.j2:18-23`). Offload force-enables prefix caching (`configs/inference.py:540-548`). Router replay + KV offload is rejected ("external KV cache hits do not carry routed-expert decisions", `configs/rl.py:660-671`).

**Router replay / sampling replay — when used.** `trainer.enable_router_replay` forces `inference.vllm.enable_return_routed_experts=True` (`configs/rl.py:571-586`); routing arrays flow response → verifiers `RoutedExperts` → `TrainingSample.routed_experts`/`MicroBatch.routed_experts` (`{data: raw bytes, shape: [seq_len, layers, topk], dtype}`) → trainer `int32` tensor (`transports/batch/types.py:15-18`; `trainer/rl/data.py:213-224`). Sampling replay: auto when train sampling truncates (§3.3). Combined router+sampling replay is rejected for P/D (V1 vs V2 runner, `configs/inference.py:511-523`).

---

## 4. Interfaces & contracts

### 4.1 HTTP routes used across S2

| Caller | Target | Route | Method | Payload | Timeout / retry |
|---|---|---|---|---|---|
| orch AdminPlane | each engine | `/health` | GET | — | 1 s poll, `wait_for_ready_timeout` (3600) |
| orch AdminPlane | router | `/health` | GET | — | same (only if `admin_base_url`) |
| orch AdminPlane | each engine | `/v1/models` | GET | — | none; skipped by `skip_model_check` |
| orch | each engine | `/pause` | POST | params `mode=keep, clear_cache=false` (ignored by prime server) | 300 s/attempt, ≤10 attempts / 600 s |
| orch | each engine | `/update_weights` | POST | `{"weight_dir": str|null}` | 720 s/attempt |
| orch | each engine | `/resume` | POST | — | 300 s/attempt |
| orch | each engine | `/init_broadcaster` | POST | `{host, port, rank_offset, inference_world_size, timeout[, session_id]}` | NCCL: no timeout, no retry; NIXL: `_admin_post(max(300, timeout))` |
| orch | each engine | `/load_lora_adapter` | POST | `{"lora_name": <model>, "lora_path": <step dir>}` | 30 s/attempt, 120 s total, retry 404/500 |
| orch dispatcher | router | `/finish_session?session_id=` | POST | — | 5 s, best effort |
| orch InferenceMetricsCollector | each engine | `/metrics` | GET | Prometheus | (owned by B/J) |
| env server / renderer client | router | `/inference/v1/generate` | POST | §3.3 | client-side |
| eval / chat clients | router | `/v1/chat/completions` | POST | OpenAI | client-side |
| k8s / ops | engine | `/liveness` | GET | — | `liveness_timeout_seconds` |

### 4.2 Files and directories

| Path | Writer | Reader | When |
|---|---|---|---|
| `output_dir/broadcasts/step_N/.sender_ready` | trainer master | orchestrator watcher / online eval | start of each broadcast |
| `.../.receiver_ready` | consumer (`_ack`) | trainer master | FS/NIXL: at receive start; NCCL: after engines paused |
| `.../.started` | trainer master | nobody in-tree | after ack |
| `.../.finished` | trainer master | FS receiver | after transfer |
| `.../model*.safetensors`, `model.safetensors.index.json` (index only if >1 shard; no `config.json`) | all trainer ranks (FS) | vLLM workers | FS full-model |
| `.../adapter_model.safetensors`, `adapter_config.json` | trainer master (FS+LoRA) | vLLM LoRA loader | LoRA |
| `output_dir/batches/step_N/rank_r.bin` | orchestrator (FS batch transport) | trainer DP rank r (all its non-DP peers) | each step |
| broadcasts retention | trainer master `_clean` keeps `N` and `N-1` | — | after `.finished` |

### 4.3 Wire schemas (S6) — `transports/batch/types.py`

All structs: `msgspec.Struct, array_like=True, gc=False` (+ `omit_defaults=True` for most) ⇒ **positional msgpack arrays**, all positions always present (§3.11); new fields must be appended at the end with a default (`:79-81, 118-119`).

`EncodedTensor` (`:7-10`): `[dtype: str, shape: list[int], data: bytes]`.
`RoutedExperts` (`:15-18`): `[data: bytes, shape: list[int] /*[seq_len, layers, topk]*/, dtype: str]`.
`SamplingMask` (`:23-25`): `[ids: bytes /*int32*/, counts: bytes /*int32 per position; 0 = no mask*/]`, `len(ids) == 4·Σcounts`.

`MicroBatch` (orchestrator → trainer, `:91-125`), in wire order:
| # | Field | Type | Notes |
|---|---|---|---|
| 0 | `input_ids` | `list[int]` | packed tokens |
| 1 | `loss_mask` | `list[bool]` | |
| 2 | `advantages` | `list[float]` | per token |
| 3 | `inference_logprobs` | `list[float]` | |
| 4 | `position_ids` | `list[int]` | |
| 5 | `sequence_lengths` | `list[int]` | per packed sequence (see C for vs. `seq_lens`) |
| 6 | `temperatures` | `list[float]` | per token |
| 7 | `env_names` | `list[str]` | |
| 8 | `seq_lens` | `list[int]` | (see C) |
| 9 | `ref_logprobs` | `list[float] | None` | |
| 10 | `routed_experts` | `RoutedExperts | None` | |
| 11 | `mm_kwargs` | `dict[str, EncodedTensor] | None` | `**`-unpacked into forward |
| 12 | `mm_token_type_ids` | `list[int] | None` | 0 text / 1 image / 2 video |
| 13–15 | `rl_weights`, `ce_weights`, `ref_kl_weights` | `list[float] | None` | loss component weights |
| 16 | `sampling_mask` | `SamplingMask | None` | |
| 17–18 | `trace_ids`, `branch_indices` | `list[str] | None`, `list[int] | None` | `""`/`-1` unknown |

`TrainingSample` (`:30-87`) is the orchestrator-internal pre-packing form (not on the wire to the trainer).

### 4.4 NCCL / NIXL wire (S7)

- NCCL: §3.8 (long count; per layer: long size, uint8 pickle `{torch.dtype: [(name, torch.Size, numel)]}`, flat per-dtype tensors). **pickle of torch dtypes** ⇒ both sides need compatible torch.
- NIXL: `TrainerTensorTable` msgpack (`trainer_tensor_table.py:50-60`):
  `TrainerTensorTable{agents: [TrainerAgent{name, metadata: bytes, device_id}], staging_buffer_count, groups: [TrainerGroup{name, tensors: [TrainerTensor{name, wire_dtype: "bfloat16"|"float32", shape, shards: [TrainerShard{agent, offset, numel, addr}]}]}]}`. Notifications: group `f"{group:08x}:{generation:016x}"`, policy `f"policy:{step:016x}:{ready|complete}"` (`agent.py:26-31`). Agent names `f"{role}-{hostname}-r{rank}"` (`agent.py:147-148`).

### 4.5 Ports

| Port | Default | Who |
|---|---|---|
| `inference.server.port` | 8000 | router (client-facing) |
| `inference.backend_port` | `server.port + 100` = 8100 (+k per DP rank on SLURM) | vLLM API server(s) |
| router Prometheus | `server.port + 21000` = 29000 | vllm-router (`entrypoints/inference.py:188-189`; `_launch_router.sh.j2:77`) |
| trainer metrics/health server | 8000 (opt-in `trainer.metrics_server`, default `None`) | **clashes with the router's default 8000** when both run on one host (single-node) (`configs/shared.py:226-231`, `configs/trainer.py:729`, `trainer/rl/train.py:104-112`) |
| `vllm.data_parallel_rpc_port` | 13345 | vLLM external-LB DP RPC |
| P/D `prefill_port` / `decode_port` | 8100 / 8200 (+k) | vLLM per rank |
| llm-d `decode_sidecar_port` | 8300 (+k) | pd-sidecar |
| llm-d EPP | 9002 grpc / 9013 health / 9090 metrics; envoy admin 127.0.0.1:9901 | |
| NIXL side channel (KV P/D) | 5600 + k | vLLM NixlConnector |
| NCCL weight broadcast | 29501 | trainer master TCP store |
| ModelExpress | 8001 (gRPC); Redis 6379 (6380 if MX port is 6379) | trainer head node (SLURM) |
| ZMQ rollout | 5555 data, 5556 ready | orchestrator bind |
| mooncake | 50051 master rpc, 8080 metadata, 50052 client | head/every inference node |
| trainer torchrun | `MASTER_PORT=29500` | |

### 4.6 Config fields (transport-relevant)

`InferenceConfig` (`configs/inference.py:443-492`): `server.{host=None, port=8000, liveness_timeout_seconds=30.0}`; `router: VllmRouterConfig{request_timeout_secs=14400, policy="sticky_least_loaded"} | LlmdRouterConfig{scorers, prefill_scorer_overrides, decode_scorer_overrides={}, non_cached_tokens=16, decode_sidecar_port=8300} | None` (default vllm-router); `backend_port=8100` (auto `server.port+100`); `vllm: VllmConfig` (extra=allow; typed: `model="Qwen/Qwen3-0.6B"`, `dtype="auto"`, `max_model_len`, `enforce_eager=False`, `trust_remote_code`, `chat_template`, `tool_call_parser="auto"`, `reasoning_parser="auto"`, `rope_scaling`, `tensor_parallel_size=1`, `data_parallel_size=1`, `data_parallel_size_local`, `data_parallel_rpc_port=13345`, `api_server_count=1` (auto ≥ DP; 1 with LoRA; 0 if `headless`), `seed=0`, `gpu_memory_utilization=0.9`, `enable_prefix_caching`, `quantization: "fp8_per_block"|None`, `enable_lora=False`, `max_loras=1`, `max_lora_rank` (rounded up to {8,16,32,64,128,256,320,512}), `lora_target_modules`, `enable_expert_parallel=False`, `all2all_backend="allgather_reducescatter"`, `enable_eplb=False`, `enable_ep_weight_filter=True`, `enable_dbo=False`, `enable_return_routed_experts=False`) (`:44-216`); `log`; `env_vars`; `use_deep_gemm=False`; `weight_broadcast.type ∈ {nccl, filesystem, nixl}` (default filesystem, `:219-221`); `kv_cache_offload`; `use_pd_kv_transfer` (auto); `enable_return_sampling_mask=False`; `enable_fp32_lm_head=True`; `enable_fp32_router_logits=True`; `deployment` (single_node / multi_node{num_nodes=2} / disaggregated{prefill/decode nodes per replica=1, replicas=1, ports, env vars, vllm overrides}; `gpus_per_node=8`); `slurm`; `output_dir`; `dry_run`.

Weight broadcast (trainer `configs/trainer.py:629-668`, orchestrator `configs/orchestrator.py:444-482`, shared `configs/rl.py:131-172`): `timeout=1200`; NCCL `{host="localhost", port=29501, inference_world_size=1}`; NIXL `{host="localhost", port=8001, inference_world_size=1, session_id="default", overlap_transfer_and_replay=False}`. `rl` default is **NCCL** unless LoRA or no inference, then filesystem (`configs/rl.py:447-451`); standalone trainer/orchestrator default is filesystem. LoRA requires filesystem (`configs/rl.py:452-457`; `configs/trainer.py:796-798`). All three components must agree (`packages/prime-rl-configs/src/prime_rl/utils/validation.py:306-318`).

Rollout transport (`configs/shared.py:234-255`): `filesystem` | `zmq{host="localhost", port=5555, hwm=10}`; default `zmq` on both trainer and orchestrator; types must match (`configs/rl.py:497-517`). `orchestrator.num_train_workers` = trainer world / cp (auto) (`configs/orchestrator.py:594-595`; `configs/rl.py:687-714`).

Client (`configs/shared.py:173-196`): `base_url="http://localhost:8000/v1"`, `api_key_var="VLLM_API_KEY"`, `headers`, `headers_from_env`, `skip_model_check=False`, `admin_base_url: list[str] | None`, `wait_for_ready_timeout=3600`, `dynamo: DynamoConfig{enabled=True, discovery_url=None} | None`.

`SlurmConfig.launch_modelexpress=True` (`configs/shared.py:109-110`).

### 4.7 Env vars

`PRIME_NO_MOE_LORA` (server → workers), `PRIME_DP_COORDINATOR_STARTUP_TIMEOUT` (default 300), `VLLM_*` (§3.1), `UCX_*` (§3.9; note inference default `UCX_TLS=all`), `NCCL_P2P_DISABLE/NCCL_SHM_DISABLE` (set to 1 when no NVLink detected, `utils/nccl.py:8-42`), `NCCL_IB_HCA` (templates), `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`, `GLOO_SOCKET_IFNAME`, `MOONCAKE_CONFIG_PATH`, `PYTHONHASHSEED=0`, `VLLM_RPC_BASE_PATH`, SLURM-level `INFER_URLS`, `ADMIN_URLS`, `ROUTER_ARGS`, `MASTER_ADDR`, `WEIGHT_BROADCAST_HOST`, `MODEL_EXPRESS_PORT`, `ORCH_ADDR`.

### 4.8 Discovery summary (how everyone finds everyone)

| Link | Single-node `rl` | Multi-node `rl` (SLURM) |
|---|---|---|
| orchestrator → router (data) | `base_url` default `http://localhost:8000/v1` | `--model.client.base-url $INFER_URLS` = `http://<infer node 0>:ROUTER_PORT/v1` (`multi_node_rl.sbatch.j2:94, 534`) |
| orchestrator → engines (admin) | `admin_base_url=[http://host:backend_port/v1]` auto (`configs/rl.py:846-854`) | `--model.client.admin-base-url $ADMIN_URLS` (one per DP rank, fixed order) (`:95-128, 535`) |
| NCCL rendezvous | trainer binds `localhost:29501`; orchestrator forwards `localhost` | trainer `host=0.0.0.0`; orchestrator `--weight_broadcast.host $MASTER_ADDR` (`:563`) |
| NIXL/ModelExpress | **nothing launched locally** — an external MX server at `weight_broadcast.host:port` is required | MX + Redis started on trainer node 0 (`:268-312`), all components get `--weight-broadcast.host $WEIGHT_BROADCAST_HOST` (`:518, 564`); or external if `launch_modelexpress=false` |
| ZMQ rollout | `localhost:5555/5556` | `--rollout_transport.host $ORCH_ADDR` to both sides (`:158-161, 519, 565`) |
| `inference_world_size` | `dp × tp` **as configured before DP auto-fill** (undercount bug, §3.8; correct under single-node SLURM re-parse) | `total_infer_nodes × gpus_per_node` |

---

## 5. Invariants & assumptions

1. **Admin client order = rank order.** `rank_offset = i · W/len(clients)` assumes every server owns exactly `W/n` GPUs, listed in the same order as the NCCL/NIXL rank layout and metrics roles (`orchestrator/clients.py:138-140, 173, 200`; `:501, 519`).
2. **Worker `device.index` is local to its server's `CUDA_VISIBLE_DEVICES`** and covers `0..W/n-1` (`worker/nccl.py:107-112`; `worker/nixl.py:104`). With vLLM internal DP on a single node (one admin URL, several engines) vLLM sets `local_rank += dp_local_rank · tp·pp`, so indices are distinct across engines (vLLM 0.24 `v1/worker/gpu_worker.py:255-272`; re-check at 0.29).
3. **`collective_rpc` from any one API server reaches every worker behind it.** With `api_server_count > 1` (single-node DP), a `/pause`/`/update_weights` request lands on one of several API-server processes sharing a port; vLLM's internal-LB client fans every utility call (`pause_scheduler`, `resume_scheduler`, `collective_rpc`) to all `core_engines` and returns the first result (vLLM 0.24 `DPLBAsyncMPClient.call_utility_async`, `v1/engine/core_client.py:1443-1452`). The `api_server_count < dp` deadlock comment (`configs/rl.py:704-706`) is about *worker count* (too few workers exist), not fan-out.
4. **One version at a time, strictly in order.** Trainer blocks on every ack; receiver applies under `update_lock`; `policy.version` only advances after the engines applied it (`watcher.py:79-121`). Skipping a version would strand the trainer (it waits for the skipped `.receiver_ready`).
5. **Broadcast directory markers are per attempt**: the trainer master `rm -rf`s `step_N` before re-offering, and prunes `> startup_version` at startup (`prune_broadcasts_beyond`, `transports/weights/base.py:29-36`; `train.py:269-270`). A consumer must not cache marker state across a trainer restart.
6. **NCCL transfer is HF-named, full, bf16 (fp32 opt-in)**; NIXL transfer is **trainer-named, sharded**, with conversion replayed on the inference side — requires `PreTrainedModelPrimeRL` custom model with `conversion_chain`, `keep_in_fp32_for_weight_transfer`, FSDP dim-0 `Shard` placements (a `Shard(1)` tensor is mis-addressed as a dim-0 shard), state-dict entries that **alias** live parameters (violated by default `qkv` fusion, §3.9), and vLLM loaders expressible with `SUPPORTED_OPS` on bf16/fp32 (`nixl.py:121-188, 329`; `graph.py:35-55, 312-346`). The trainer is always FSDP-wrapped, even on one GPU (`trainer/model.py:1049`, unconditional `setup_fsdp`), so parameters are DTensors everywhere: NCCL casts them to bf16 (fp32 opt-in); only non-DTensor persistent buffers go in their native dtype.
7. **NIXL plan is static.** The trace + plan are built once per worker lifetime (`worker/nixl.py:127-128`) and the trainer's staging addresses/table once per trainer lifetime (`nixl.py:327-328`); the model's parameter storage and the trainer shard layout must not change between versions, and every served state-dict tensor must alias its live parameter (default `qkv` fusion breaks this → stale q/k/v, §3.9). A trainer restart without an inference restart leaves the workers with a stale plan (and MX session) `[inferred]`.
8. **Prefix cache is never cleared on weight update, by any transport** (`clear_cache=False` hard-wired, no reset call anywhere; §3.6.1a). The only guard is `cache_salt = str(group.policy_version_at_start)` (`dispatcher.py:532, 559-565`), which is fixed per **group**, not per dispatch. A group straddling a swap (members dispatched one at a time, `:496-499, 567`) sends the old salt to the new weights and reuses old-weight prefix KV; so do later turns of any in-flight episode (§3.6.1b). The recorded inference logprobs are what was actually sampled, so the trainer's $\pi_\theta/\mu$ ratio sees the true (larger) mismatch; what is unmodeled is the version tag: $\mu$ is a hybrid of old KV and new weights, not $\pi_{\text{start}}$, and nothing marks the affected tokens (03 §7.1, 04 §3.11).
9. **In-flight requests survive updates (frozen, not aborted or drained).** `pause(mode="keep")` parks requests with their KV (`PAUSED_ALL`, §3.6.1c); after `resume` they continue under the new weights over partly-old KV. A single completion can contain tokens from two policies; off-policy accounting uses the version at dispatch (see C/B).
9a. **`/update_weights` is only idempotent for filesystem.** Under NCCL/NIXL a client-side retry after a timeout or 5xx duplicates a collective/credit participation and can hang or corrupt the transfer (§3.6.1d), yet `_admin_post` retries it by default.
10. **ZMQ PUB/SUB relies on the dispatch gate** to stay below HWM=10 per rank, and on all ranks' READY arriving before the first send (`zmq.py:13-21, 47-64`).
11. **Rollout transport step counters are implicit.** Neither transport's ZMQ message carries a step; the FS path encodes it in the directory; both start at `progress.step`. Orchestrator and trainer resume steps must agree (S8).
12. **LoRA adapter name = base model name** so requests keep addressing one stable model id (`vllm/server.py:88-99`; `orchestrator.py:301-303`); requires `max_loras=1`, `api_server_count=1` (`configs/inference.py:116-118, 214-215`).

---

## 6. Extension points

**Add a weight transport** (lockstep changes):
1. Config: add a `type` literal to trainer, orchestrator, shared (`rl.py`) broadcast unions and `InferenceConfig.WeightBroadcastConfig.type` (`configs/trainer.py:665-668`, `configs/orchestrator.py:479-482`, `configs/rl.py:169-172, 458-491`, `configs/inference.py:219-221`).
2. Trainer: subclass `WeightSender`, implement `_broadcast` (hold non-masters until handshake); register in `setup_weight_sender` (`transports/weights/__init__.py:22-35`).
3. Consumer: subclass `WeightReceiver` with `initialize()` and `receive(step)` (ack exactly once; use `admin_plane.update_weights(..., on_paused=...)`); register in `setup_weight_receiver` (`:38-51`).
4. Server: a worker-extension class with `init_broadcaster(...)`, `update_weights_from_path(arg)`, `liveness_probe()`; add to `WORKER_EXTENSION_CLS` (`inference/vllm/server.py:59-63`). Reuse `load_weights_checkpoint_layerwise` for any iterator of `(hf_name, tensor)` (`worker/weight_transfer.py:12-23`).
5. Launch wiring: SLURM template host/port injection (`multi_node_rl.sbatch.j2`) and `entrypoints/rl.py` template vars.

**Add a server route**: append to `router` in `inference/vllm/server.py`; it is mounted in every API-server process automatically (`:185-206`). Worker-side logic goes through a new extension method + `collective_rpc`.

**Add a vLLM patch**: put it in `apply_shared_vllm_patches` (every process; plugin entry point) or at `vllm/server.py` import (API server only) or `vllm/worker/__init__.py` (worker extension import). Make it idempotent with a `_prime_rl_*` marker attribute like the existing ones (`patches.py:66-67, 115-116`).

**Change the generate response**: subclass logic in `PrimeRlServingTokens` (`serving_tokens.py`), and update the parser in `renderers/client.py` + verifiers `TrainClient` (F/H).

**Add a rollout transport**: implement `BatchSender.send(grid)` / `BatchReceiver.{wait,can_receive,receive}`, add config variant to `TransportConfig` (`configs/shared.py:255`) and the factory (`transports/batch/__init__.py:21-40`), plus multi-node host injection gated in `entrypoints/rl.py` (`use_zmq_transport`, `:531, 569`).

**Add a MicroBatch field**: append at the end of `MicroBatch`/`TrainingSample` with a default (positional encoding), populate in the packer (B/C), consume in `trainer/rl/data.py:_micro_batch_to_tensor` and `TensorMicroBatch`.

**Add a router backend**: new discriminated `RouterConfig` variant (`configs/inference.py:367-368`), launch branch in `_launch_router.sh.j2`, local launch in `entrypoints/inference.py:start_router` (only vllm-router today).

**Admin-plane discovery**: subclass `AdminPlane` like `DynamoAdminPlane` and select it in `setup_admin_plane` (`orchestrator/clients.py:239-245`).

---

## 7. Gotchas & limitations

- **Single-node local `rl` undercounts NCCL/NIXL `inference_world_size`** whenever DP is auto-filled (`num_infer_gpus > dp·tp` as configured) — including the shipped `examples/basic/hendrycks-sanity` config (`W=1` vs 4 engines). Validator order bug (`configs/rl.py:439-495` before `:687-709`); fails at the first weight sync. Workaround: set `inference.vllm.data_parallel_size = num_infer_gpus / tp` explicitly (§3.8; also 01 §7.5).
- **NIXL + default `qkv` fusion serves stale q/k/v** on multi-GPU trainers (state dict captured once, fused split yields copies); with `shard_fused_on_dim1` it fails at plan build instead. Disable fusions for NIXL (§3.9).
- **`AdminPlane.initialize_nccl` swallows non-404 HTTP errors.** The `except httpx.HTTPStatusError` only logs for 404 and otherwise falls through silently (`orchestrator/clients.py:192-196`) — a 500 from `/init_broadcaster` leaves the orchestrator believing NCCL is initialized, and the failure only surfaces at the first `/update_weights` (e.g. the `W` undercount above). Transport errors are not caught and do propagate. The request also has no timeout (client `timeout=None`) and no retry.
- **Retried `/update_weights` under NCCL/NIXL is a hang/corruption hazard**: `_admin_post` retries timeouts and 5xx (720 s per attempt, 1440 s total), but a server-side collective keeps running after a client timeout, and the retry queues behind it on the EngineCore (§3.6.1d). Any full-model transfer slower than 720 s trips it.
- **Old-salt reuse across a swap**: straddling groups and multi-turn episodes reuse prefix KV computed under the previous policy (§3.6.1b).
- **Plugin failures are fatal, not silent**: patch imports are lazy inside `apply_shared_vllm_patches`, which vLLM calls unguarded, so a vLLM rename breaks startup of every vLLM process. Only an import failure of the `prime_rl.inference.patches` module itself is logged and skipped (§3.2). The docstring at `patches.py:8-10` overstates the silent case.
- **`/pause` ignores its query params**; the prime server always uses `mode="keep", clear_cache=False` (`vllm/server.py:66-70`) even though the client sends `mode/clear_cache` (`clients.py:420`).
- **`UCX_TLS` conflict**: inference processes get `UCX_TLS=all` from `DEFAULT_INFERENCE_ENV_VARS` (`utils/process.py:32`), so `set_ucx_env_defaults`' `setdefault("UCX_TLS", "rc_x,rc,dc_x,dc,cuda_copy")` never applies on vLLM workers, only on trainer/orchestrator (`agent.py:205`).
- **NIXL needs an InfiniBand port** unless `UCX_NET_DEVICES` is preset (`agent.py:187-188`), a from-source UCX 1.19 with `rc_verbs` + `cuda_copy` (`multi_node_rl.sbatch.j2:252-267`; `scripts/install_nixl_from_source.sh`), and ModelExpress 0.3.0 + Redis 7.4.2 (`scripts/install_modelexpress.sh`). Local (non-SLURM) runs start no ModelExpress server.
- **NIXL is custom-model only** (casts to `PreTrainedModelPrimeRL`, `nixl.py:329`) and imports trainer model code inside vLLM workers (`graph.py:337-342`). Loaders using ops outside `SUPPORTED_OPS` (`cat`/`stack`/`new_empty`/arithmetic) or non-bf16/fp32 dtypes (e.g. FP8 checkpoints) fail at trace time. The plan is also static, so any trainer-side tensor that is a *copy* rather than a view of a live parameter is served stale (the `qkv` bug above).
- **NIXL arenas double memory** when `overlap_transfer_and_replay` (largest group per GPU per extra arena, `docs/scaling.md:257`); arenas come from a cudaMalloc pool, likely because the trainer's `expandable_segments:True` allocator is unsuitable for RDMA registration `[inferred]`.
- **NCCL group is one-shot**: no re-init path; an inference restart requires a trainer restart (and vice versa). Dynamo makes this explicit ("terminal").
- **NCCL memory**: broadcast materializes each layer's full bf16 tensors on the trainer master plus a concatenated copy per dtype, and on each receiver a concatenated per-dtype buffer; trainer calls `synchronize()` + `empty_cache()` first (`train.py:587-595`). That comment's "per-layer gather + fp8 conversion peaks ~50 GiB" has no fp8 step on the trainer send path — NCCL/FS/NIXL only cast to bf16/fp32 and apply the prime→HF chain (`nccl.py:143-159`, `utils/weights.py:162-199`, `trainer/models/base.py:111-125`). FP8 quantization happens **inference-side**, when layerwise reload re-runs `process_weights_after_loading` for `quantization="fp8_per_block"` (hence the patch `monkey_patch_online_fp8_parameter_cast`, `patches.py:103-124`).
- **`resolve_dtensors` never consults `keep_in_fp32` for non-DTensors** and NCCL hard-codes bf16 (`nccl.py:84-100, 129`); the NIXL path has its own inline copy of the dtype rule (TODOs at `nccl.py:93`, `utils/weights.py:157`).
- **Filesystem transport writes the full model every step** to shared storage and keeps two versions; LoRA reloads under live traffic (no pause).
- **LoRA**: filesystem only; one API server; `gather_weights_parallel` raises on LoRA keys (`utils/weights.py:196-197`).
- **ZMQ restart asymmetry**: READY is sent once at receiver construction (`zmq.py:111-112`) and the sender waits once (`_ready`); if only the orchestrator restarts, it waits forever; if only a trainer rank restarts, PUB may drop its first message (slow joiner). Components must restart together (the launchers do).
- **Multiple subscribers per DP rank** (CP peers) all receive the same topic; READY dedups by rank id and so does not cover every peer. This is masked by construction order: receivers exist before the startup broadcast, which gates any send (§3.11).
- **FS transport**: the receiver waits for `.finished` with no timeout, and dispatch stays blocked for the whole trainer export (§3.7).
- **Trainer metrics server port 8000** (opt-in) collides with the router's default `inference.server.port = 8000` on a single node (§4.5).
- **`batches/` is never garbage-collected** by the FS batch transport (only future steps are removed on resume).
- **Routed experts**: disallowed with llm-d, with KV offload, and (combined with sampling replay) with P/D; chat-completions responses never carry them; payload is large — docs recommend sizing up env-server pools (`docs/inference.md:300`).
- **Sampling-mask capture rejects greedy/`top_k`-less requests engine-wide**, including evals on the same server.
- **EPLB is rejected with RL weight updates** (`configs/rl.py:519-523`).
- **Standalone multi-node wide-EP template mismatch**: `inference.sbatch.j2` forms node-local EP groups (`DP_PER_NODE`, `LOCAL_IP`) (`:331-336`) while `multi_node_rl.sbatch.j2` spans the replica (`:470-480`) and docs describe EP spanning nodes (`docs/inference.md:108-124`) — confirm which is intended before relying on the standalone path.
- **vLLM version coupling**: several patches are copies of vLLM 0.24 internals (`_patch_lora_key_prefix`) or target private symbols (`Scheduler._update_from_kv_xfer_finished`, `DPCoordinator._wait_for_zmq_addrs`, layerwise reload internals `LAYERWISE_INFO`, `_place_kernel_tensors`); the dependency is declared `vllm>=0.29.0` (`pyproject.toml:59`), but `[tool.uv.sources]` pins the exact v0.29.0 release wheels (`pyproject.toml:276-278`).
- **Admin HTTP pool**: 4 connections per engine; many concurrent admin ops to one engine queue.

---

## 8. For a custom framework

**Essential design worth keeping.**
- *Router-fronted data plane + direct per-engine admin plane.* Clean separation; session-affine routing (`X-Session-ID` + explicit release) is what makes multi-turn prefix caching work at scale.
- *Token-in/token-out generation with engine-reported per-token logprobs, sampling masks and routed experts.* This is the correctness core of off-policy RL on LLMs: the trainer needs exactly the sampling distribution (processed logprobs + kept set) and, for MoE, the routing, to compute unbiased ratios $r_t = \pi_\theta(y_t)/\tilde\pi_{\text{old}}(y_t)$.
- *Version handshake decoupled from bytes.* The 4-marker protocol + `WeightReceiver/WeightSender` split lets transports be swapped without touching the orchestrator loop. Keep the "consumer pauses, acks, receives, resumes" shape and the lockstep guarantee; they make version accounting trivial.
- *Drain-before-pause ordering* and *per-version cache salt* are subtle but necessary.

**Incidental / would simplify.**
- The **filesystem marker bus** is simple but polls at 0.1–1 s and couples trainer and orchestrator via a shared FS even when bytes move over NCCL/NIXL. A small control-plane RPC (or reuse the ZMQ channel) would remove FS dependence for in-memory transports.
- **Monkeypatching vLLM app factories** (`build_app`, `init_app_state`, `run_api_server_worker_proc`) is fragile; a vLLM plugin/route registration API or a thin sidecar process would be sturdier. Most model patches are upstream bug workarounds with removal dates.
- **NCCL via `StatelessProcessGroup`** needs static world sizes and cannot survive restarts; for elasticity prefer NIXL/RDMA pull or a store-and-forward service.
- **NIXL lazy-trace replay** is the most sophisticated piece: it derives the trainer→vLLM parameter mapping *automatically* by running vLLM's own loader symbolically, avoiding a hand-maintained per-model conversion table, and ships only FSDP shards (no all-gather on the trainer). Worth copying if we target vLLM and RDMA; cost is a hard dependency on vLLM loader internals and custom-model conversion chains.
- **Rollout transport**: ZMQ PUB/SUB with HWM and implicit steps is minimal but lossy by design; a framework that allows restarts of either side independently should use PUSH/PULL or ROUTER/DEALER with step ids and acks, or a durable queue.

**Coupling points to watch.** vLLM version (private APIs), vllm-router fork (session release, routed-expert merge), renderer ↔ server response format (`routed_experts` object form, `logprobs.content[i].token == "token_id:<id>"`), trainer model conversion chains (NCCL/FS: `convert_to_hf`; NIXL: `conversion_chain` executed inside vLLM).

---

## 9. Open questions

1. Every vLLM-internal claim here (keep = freeze, `clear_cache`-only reset, DP utility fan-out, DP `device.index`, serial utility calls, `prompt_logprobs` skipping prefix-cache reads, `StatelessProcessGroup` rank assert, plugin loader) was read in the locally available **vLLM 0.24** source; the pin is **0.29.0**. Re-verify in 0.29 `v1/engine/{async_llm,core,core_client}.py`, `v1/worker/gpu_worker.py`, `sampling_params.py`, `plugins/__init__.py`. The native sampling-mask rejection exists only in ≥0.28 and could not be read at all.
1a. Is per-group (not per-dispatch) `cache_salt` intentional? (B owns `dispatcher.py:559-565`.)
2. Whether the vllm-router fork proxies `/v1/models`: the renderer's `max_model_len` preflight goes through the router URL and silently disables itself on any failure (`deps/renderers/renderers/client.py:71-100`).
3. `routed_experts_prompt_start` precise semantics (vLLM `SamplingParams`) and how vllm-router stitches prefill+decode routed experts in P/D.
4. How `--intra-node-data-parallel-size` makes vllm-router address DP ranks behind one engine URL (likely `X-data-parallel-rank`; the header name only appears in comments, `inference/vllm/server.py:168`).
5. Runtime confirmation of the two static bugs: single-node `W` undercount (exact failure mode at 0.29) and NIXL stale q/k/v (needs a ≥2-GPU NIXL run comparing served q/k/v with the trainer).
6. Whether a vLLM ≥0.29 LoRA prefix-block hash can distinguish in-place-reloaded adapter weights (§3.6.1a).
7. `MicroBatch.sequence_lengths` vs `seq_lens` — both on the wire; C owns the distinction.
8. Standalone `inference.sbatch.j2` vs `multi_node_rl.sbatch.j2` wide-EP grouping discrepancy (§7).
9. What happens to the NIXL static plan / ModelExpress session on trainer-only restart (resume) while inference stays up (under SLURM everything restarts together, so likely unexercised).
