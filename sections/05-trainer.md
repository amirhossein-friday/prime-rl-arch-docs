# Trainer engine (process model, model build, parallelism, optimizer, checkpoints, weight export) — prime-rl @ b944873

> Scope: the RL/SFT trainer *engine* — torchrun process model, device mesh, model construction and HF↔prime weight conversion, FSDP2/EP/CP/AC/offload, optimizers and schedulers, LoRA, low-precision and fused paths, the training-loop skeleton, DCP checkpointing/resume (S8 trainer side) and weight export for inference sync (S7 trainer side). The per-step loss/data math inside `rl/train.py` is C's; the wire/receiver side of weight sync is E's.
>
> Files read in full: `src/prime_rl/entrypoints/trainer.py` (26), `trainer/rl/train.py` (756), `trainer/model.py` (1138), `trainer/parallel_dims.py` (342), `trainer/ckpt.py` (333), `trainer/world.py` (44), `trainer/utils.py` (344), `trainer/perf.py` (245), `trainer/scheduler.py` (121), `trainer/lora.py` (358), `trainer/moe_runtime.py` (133), `trainer/activation_checkpointing.py` (134), `trainer/sign_sgd.py` (53), `trainer/conversion_utils.py` (13), `trainer/optim/{__init__ (249), base (28), state_offload (122), offload (1616), cpu_adam/__init__ (114)}`, `trainer/distributed/{__init__ (3), collectives (263), deepep (475), token_dispatcher (332), expert_parallel (14)}`, `trainer/models/{__init__ (125), base (159), conversion_ops (348), fusions (312), fp8 (39)}`, `trainer/models/layers/{attn (208), ring_attn (497), ulysses_attn (334), moe (480), mlp (42), activations (43), grouped_gemm (68), lm_head (357), lm_head_gemma (202), norms (59), rms_norm (3), rotary_emb (118), fp8_linear (200), fp8_grouped_gemm (276), mxfp8_linear (99), lora/{__init__ (20), base (177), multi_linear (164), multi_moe (841)}}`, `models/qwen3/*` (282), `models/qwen3_moe/*` (590), `__init__`+`configuration_*`+`converting_*` for deepseek_v4, glm4_moe, glm_moe_dsa, gpt_oss, laguna, minimax_m2, nemotron_h, qwen3_5, afmoe, llama/`__init__`; `utils/{cp (164), act_offloading (330), weights (252), vlm (135), transformers_compat (32)}`; `packages/prime-rl-configs/.../configs/trainer.py` (828) + `shared.py` 1–240; `transports/weights/{__init__ (51), base (169), filesystem (75), nccl (213), nixl/nixl (502), nixl/trainer_tensor_table (60), nixl/tensor_routing (83), nixl/graph (346)}`; `trainer/sft/train.py` (706); `tools/convert_{dcp_to_bf16 (200), dcp_to_fp8 (65), bf16_to_fp8 (153), fp8_to_bf16 (117)}.py`; `scripts/mini_moe.py` (327); `docs/{training,scaling,advanced,development}.md`; `skills/kernels/SKILL.md`; tests `unit/train/{test_model,test_state_offload}.py`, `unit/train/models/{test_checkpointing,test_fusions,test_moe_conversions,test_state_loading,test_qwen3_moe}.py`, `unit/utils/test_weights.py`, `unit/transports/test_nccl_broadcast.py`, `integration/test_reverse_text_moe.py`. Targeted reads (located by grep, then read): `utils/pathing.py` 249–425, `utils/process.py` 1–40, `utils/sequence.py` 38–52, `configs/rl.py` 340–495, `entrypoints/rl.py` 300–370, `rl/data.py` 175–214, family `modeling_*.py` override blocks (cp_support / keep_in_fp32 / is_*_state_dict).
>
> Related docs: 01-deployment-topology-and-launch.md (A: torchrun spawn, resume dirs), 02-config-system.md (A), 03-orchestrator.md (B: consumer of broadcasts, orchestrator ckpt), 04-algorithms-loss-data-path.md (C: loss, micro-batch schema, step accounting), 06-inference-and-transports.md (E: weight-sync wire + receiver), 11-observability-eval-ops.md (J: monitors, metrics server).

---

## 1. Mental model

The trainer is a **torchtitan-flavoured FSDP2 engine wrapped around HF-compatible model classes**. One Python process per GPU (torchrun), all ranks SPMD, no pipeline parallelism (`pp` is hard-wired to 1, `parallel_dims.py:329`). The world is carved into a named `DeviceMesh` (`dp_replicate × dp_shard × cp`, with `ep` *borrowed* out of `dp_shard × cp`). Every parameter is an FSDP2 `DTensor` sharded along the data-parallel dims; MoE expert weights are additionally sharded along the EP dim. Compute runs in bf16 (`MixedPrecisionPolicy(param_dtype=bf16)`, `model.py:503`) over fp32 sharded master params (`optimization_dtype="float32"` default, `configs/trainer.py:324`).

The model itself is either a **custom "prime" implementation** (`PreTrainedModelPrimeRL` subclasses registered in `trainer/models/__init__.py`, used when `model.impl` resolves to `custom`) or a **stock HF `AutoModelForCausalLM`** (fallback). Custom models own their packed-sequence (`seq_lens`/varlen flash) handling, MoE (grouped GEMM + token dispatchers + EP), CP-awareness, and — crucially for weight sync — a **declarative, invertible HF↔prime state-dict conversion chain** (`conversion_ops.py`). Training state lives in the *prime* layout (e.g. stacked `[E, H, D]` expert tensors); everything that leaves the trainer for vLLM is converted back to *HF* layout (except NIXL, where the receiver replays the conversion as view ops; §3.17.4).

The RL loop is deliberately simple and **synchronous**: wait for a pre-packed batch from the orchestrator → forward/backward over micro-batches → clip → optimizer step → scheduler step → **blocking weight broadcast of the new policy** (handshaked with the consumer) → optional blocking DCP checkpoint → metrics. The only overlaps are inside the step: FSDP prefetch, DeepEP chunk pipelining, and (with `full_offload`) a CPU optimizer that runs chunk-by-chunk during the last backward. Off-policy-ness is created *outside* the trainer, by the orchestrator generating with older policies while the trainer steps (B/C).

Think of it as five layers: **(1)** process/mesh (`world.py`, `parallel_dims.py`, `utils.setup_torch_distributed`); **(2)** model build (`model.setup_model`: meta init → fusions → LM-head injection → quantization → LoRA/freezing → MoE runtime/EP → AC → compile → FSDP → weight load); **(3)** optimizer stack (`optim/`: plain, state-offloaded, or full CPU offload with optimizer-in-backward); **(4)** loop (`rl/train.py`, `sft/train.py`); **(5)** persistence/export (`ckpt.py` DCP; `utils/weights.py` + `transports/weights/*` sender side).

## 2. Where it runs

### 2.1 Process model

```
rl launcher (entrypoints/rl.py)  ──Popen──►  torchrun --role=trainer --nproc-per-node=<num_train_gpus>
                                              --rdzv-endpoint=localhost:<free port> --local-ranks-filter=<log.ranks_filter>
                                              -m prime_rl.trainer.rl.train @ <run_dir>/configs/.../trainer.toml
                                                   │
                     ┌─────────────────────────────┼──────────────────────────────┐
                 rank 0 (master)               rank 1 … rank N-1  (identical SPMD code)
   - heartbeat, MetricsServer (/metrics)       - HealthServer on each other node's local_rank 0
   - broadcast dir markers (.sender_ready…)    - participate in every FSDP / DTensor collective
   - NCCL weight-broadcast communicator        - (NIXL) serve their own FSDP shards
   - ckpt dir mkdir + old-ckpt deletion        - write own DCP shard files
```
- Single-node launch cites: `entrypoints/rl.py:330-366` (torchrun command, `CUDA_VISIBLE_DEVICES` = trainer GPUs, env merges `DEFAULT_COMMON_ENV_VARS` + `DEFAULT_TRAINER_ENV_VARS` + `config.env_vars` + `config.trainer.env_vars`). Multi-node/SLURM placement is A's (S1).
- Standalone entrypoint `trainer` (`entrypoints/trainer.py:17-22`) is a thin CLI that defers heavy imports; `python -m prime_rl.trainer.rl.train` is what the launcher runs (`entrypoints/trainer.py:7-9`).
- Default env for trainer processes: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`utils/process.py:24-26`); common `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`, `GIT_LFS_SKIP_SMUDGE=1` (`utils/process.py:17-22`). `model.py:9` sets `USE_HUB_KERNELS=NO` (prevents transformers hub-kernel substitution of e.g. `RMSNorm`, which is decorated with `use_kernel_forward_from_hub`, `norms.py:29`).
- Topology is read from torchrun env vars `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE` into the `World` singleton (`world.py:7-13`); `is_master = rank == 0` (`world.py:22-24`). It asserts `world_size % local_world_size == 0` (uniform nodes, `world.py:20`).

### 2.2 Rank roles

| Role | Where decided |
|---|---|
| Heartbeat POST, full Prometheus server | master only (`rl/train.py:96-108`) |
| Health-only server on other nodes | `local_rank == 0 and not is_master` (`rl/train.py:109-112`) |
| Broadcast-dir reset/markers, `prune_broadcasts_beyond`, `_clean` | master (`transports/weights/base.py:58-70`, `rl/train.py:269-270`) |
| NCCL weight-broadcast communicator (rank 0 of trainer+inference group) | master only (`nccl.py:131-140`) |
| NIXL shard serving | every rank with `dp_replicate` local rank 0 (`nixl.py:96-100`); unsharded/replicated tensors served only by master (`nixl.py:137-172`) |
| Filesystem export | every rank writes its owned layers (`utils/weights.py:162-199, 202-252`); master writes the index |
| Checkpoint dir mkdir, old-ckpt deletion | master (`ckpt.py:283-285, 313-319`); DCP shard writes are per-rank |
| Batch receive stream | one stream per **DP rank**: `dp_rank = rank // (world_size // dp_size)` (`rl/data.py:190-193`) — only the CP-mates of a DP rank read the same batch (EP borrows `dp_shard`, so EP-mates are *different* DP ranks with different batches); valid only because `cp` is the innermost mesh dim in both mesh layouts |

### 2.3 Lifecycle

- **Start**: see §3.1 (ordered). Anything that raises propagates to `@clean_exit` (`rl/train.py:74`; `utils/utils.py:46-87`) which logs, `wandb.finish(exit_code=1)`, `sys.exit(1)`, and always `destroy_process_group()`.
- **Steady state**: `while True` loop (`rl/train.py:253-719`), one iteration per policy version.
- **Normal shutdown**: after `is_last_step` (`progress.step >= max_steps`, `rl/train.py:258`): export profiler trace, write **final checkpoint** (if `ckpt` set; `rl/train.py:730-733`), close gradient-offload workers, finalize monitors, stop servers (`rl/train.py:735-746`). `max_steps=None` → runs forever.
- **Crash modes worth knowing**: a consumer that never acknowledges a broadcast raises `TimeoutError` after `weight_broadcast.timeout` (default 1200 s, `configs/shared.py:41`; `base.py:83-86`); NCCL/Gloo collectives time out after `dist_timeout_seconds` (default 3600, `configs/trainer.py:723`) — the default PG timeout is patched so mesh sub-groups inherit it (`utils.py:185-192`); full-offload worker stalls raise after 120 s (`offload.py:1070, 692-701`).

## 3. Mechanics

### 3.1 Startup sequence (`rl/train.py:74-251`)

Order matters (several steps are only legal before/after others):

1. `get_world()`, `setup_logger` (`:77-82`); `monitors.setup(producer="trainer", …)` (`:85-93`).
2. Heartbeat (master) and metrics/health servers (`:96-112`).
3. `setup_torch_distributed(timeout, enable_gloo=fsdp_cpu_offload or full_offload)` (`:115-118`): `torch.cuda.set_device(local_rank)`, backend `"cpu:gloo,cuda:nccl"` when CPU-side collectives are needed else NCCL default, `init_process_group(device_id=local_rank)` (`utils.py:173-193`).
4. Full-offload host prep: NUMA-bind the process to its GPU's socket and set `torch.set_num_threads(cpu_count // local_world_size)` (`:119-120`; `utils.py:119-170`).
5. `torch.set_float32_matmul_precision(config.matmul_precision)` (default `"high"` = TF32; `:124`).
6. `resolve_ep(config.model)` — `"auto"` → `min(world_size // dp_replicate, 8)` for MoE configs (detected by `num_experts`/`n_routed_experts` attr on the HF config), 1 otherwise or for `impl="hf"` (`parallel_dims.py:287-315`).
7. `get_parallel_dims` → `ParallelDims(dp_replicate, dp_shard=-1, cp, pp=1, ep, world_size)`; mesh is built lazily on first `.world_mesh`/`get_mesh` access (`parallel_dims.py:212-224, 318-342`).
8. `setup_ckpt_manager(output_dir, ckpt, resume)` — always returns a manager (`ckpt.py:325-333`); resolves `checkpoint_step` from `resume.dir` (its `step_N` name), `resume.step`, or the latest `step_*` under the ckpt dir (`:137-143`).
9. `setup_model(config.model, parallel_dims, loading_from_ckpt_later=checkpoint_step is not None)` (§3.3).
10. VLM guard: `model.supports_packed_multimodal_training` required if `[model.vlm]` (`:152-153`).
11. `setup_rl_loss_fn(config.loss)` (C).
12. `setup_optimizer(...)` → `(optimizer, gradient_manager)` (§3.7–3.8); `setup_scheduler` (§3.7).
13. `setup_weight_sender(output_dir, weight_broadcast, parallel_dims, lora)` unless `data.fake` (`:179-191`) — NCCL sender builds its communicator *here* (rendezvous with inference; §3.17.3).
14. `setup_context_parallel` if `cp > 1` (§3.11) — after FSDP, monkeypatches attention and pushes a `CPContext` into every module with a `cp_context` attr.
15. LoRA adapters re-initialized after FSDP materialization (`get_lora_state().reset_adapter_parameters()`, `:201-202`).
16. Resume: `ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress, path=resume.dir/"trainer" if set)`, then `progress.step += 1` (`:205-221`).
17. `DataLoader(output_dir, progress.step, dp_size, rollout_transport)` or `FakeDataLoader` (`:228-236`; C/E own the transport).
18. `AnnotationWriter`, `GarbageCollection` (disables automatic GC; `gc.collect(1)` every `gc.interval`=50 steps on all ranks together, `utils.py:30-54`), optional torch profiler (`:239-250`).

### 3.2 Device mesh and process groups (`parallel_dims.py`)

**Degrees.** `dp_shard=-1` is resolved to `world_size // (dp_replicate·cp·pp)` (`:64-66`), and `dp_replicate·dp_shard·cp·pp == world_size` is asserted (`:68-71`). EP must satisfy `ep % cp == 0` and `(dp_shard·cp) % ep == 0` (`:73-75`): **EP borrows all of CP and a slice of dp_shard**, it is not an extra mesh factor.

**Mesh without EP** (`:157-210`): dims (only >1 kept, but `dp_shard` always kept) in order `[pp, dp_replicate, dp_shard, cp]` — row-major, so **cp is innermost** (CP peers are adjacent ranks).

**Mesh with EP** (`:83-155`): `dp_shard_mod_ep = dp_shard·cp/ep`, `dp_shard_in_ep = ep/cp`; dims `[pp, dp_replicate, dp_shard_mod_ep (always kept), dp_shard_in_ep, cp]`.

Flattened submeshes (all created eagerly so every PG exists on every rank):

| Name | Composition (no EP) | Composition (EP) | Used for |
|---|---|---|---|
| `dp` | dp_replicate × dp_shard | dp_replicate × dp_shard_mod_ep × dp_shard_in_ep | data loading size, token accounting (no comm) (`:115-116, 141`) |
| `dp_shard_cp` | dp_shard × cp | dp_shard_mod_ep × dp_shard_in_ep × cp | FSDP shard group; Muon distributed mesh (`optim/__init__.py:216-217`) |
| `dp_cp` | dp_replicate × dp_shard × cp | dp_replicate × … × cp | loss-denominator all-reduce, MoE stats (`rl/train.py:320-321`) |
| `ep` | — | dp_shard_in_ep × cp | token all-to-all, expert sharding (`:144`) |
| `hsdp` | `dp_shard_cp`, or 2-D `(dp_replicate, dp_shard_cp)` when `dp_replicate>1` | same | the mesh `fully_shard` is called with (`model.py:514`) |
| `dp_replicate`, `dp_shard_mod_ep`, `cp` | raw dims | raw dims | LoRA seed broadcast, expert FSDP, CP groups |

```
EP example: world=16, dp_replicate=1, cp=2, ep=8  →  dp_shard=8, dp_shard_mod_ep=2, dp_shard_in_ep=4
mesh dims [dp_shard_mod_ep=2, dp_shard_in_ep=4, cp=2]   rank = a*8 + b*2 + c
ep group  = {b,c} for fixed a  → 8 contiguous ranks (one node)       experts Shard(0) over these
expert FSDP mesh (dp_mod_ep) = {a} for fixed (b,c) → ranks r, r+8    experts' FSDP Shard(0) across nodes
dense FSDP (hsdp=dp_shard_cp) = all 16 ranks
```

Derived quantities: `fsdp_gradient_divide_factor = dp_replicate·dp_shard·cp` (`:258-260`); `seq_len_divisor = 2·cp` — only validated when a `seq_len` is passed, which SFT does and RL does not (`:334-340`; `rl/train.py:130` vs `sft/train.py:114`).

### 3.3 Model build (`model.setup_model`, `model.py:968-1078`)

Ordered pipeline (the comment "AC -> Compile -> FSDP" at `:1043` is load-bearing):

1. **Attention resolve** — `attn="auto"` → FA4 on SM10x/11x, FA3 on SM90, else FA2 (`:947-965`); FA3 needs `flash_attn_3`, FA4 validates `flash_attn.cute` isn't shadowed (`:975-981, 927-944`).
2. **Meta-device build** `get_model(config, device=meta, dtype=optimization_dtype)` (`:986`):
   - `AutoConfig.from_pretrained(name, attn_implementation=attn, trust_remote_code)`, force `use_cache=False` on config + sub-configs (`:317-332`), IndexShare/`index_cache` wiring for DSA models (`:333-348`), fill `pad_token_id` from generation config / eos, unwrap list-valued ids (`:350-373`), `debug.num_layers` truncation (`:381-388`).
   - **Impl dispatch** (`:390-443`): `impl="auto"` → `custom` iff `type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING` (`models/__init__.py:91-100`) or, for VLM archs, iff `_CUSTOM_VLM_MAPPING` has the `model_type` (`models/__init__.py:106-114`). Errors: FA3/FA4 with HF impl (`:401-405`); `cp>1` with auto→hf (`:407-411`); unsupported `cp_style` per the model's `cp_support()` (`:415-423`); `[model.vlm]` without a custom VLM class (`:425-429`).
   - Model class: custom → `AutoModelForCausalLMPrimeRL` (a `_BaseAutoModelClass` over the private mapping, `models/__init__.py:79-83`); HF text → `AutoModelForCausalLM`; HF VLM → `AutoModelForImageTextToText` (`:431-443`). On meta: `model_cls.from_config(...)`; otherwise `from_pretrained` on CPU (`:449-459`). Asserts `lm_head.weight.dtype == optimization dtype` (`:462-464`).
3. **Meta-loadability check** `can_reinit_empty_buffers` (`:768-806`): custom models always OK (they implement `init_buffers_post_meta`); HF models only if their buffers are within a known whitelist (rotary `inv_freq`, GPT-OSS, Gemma3). Otherwise **every rank loads the full model on CPU** (`:996-998`).
4. **Runtime fusions** (`apply_model_fusions`, skipped with LoRA; `:1000-1004`) — §3.14.
5. **LM-head injection** `inject_prime_lm_head(model, chunk_size)` (`:1006-1010`) — §3.12. Note this *replaces `model.forward`* for every model (custom or HF) (`lm_head.py:316, 319-357`).
6. **Dense quantization** FP8 blockwise / MXFP8 (`apply_quantization`, `:876-893`; MXFP8 requires SM100+) — §3.13.
7. **Trainable-parameter configuration** (`:1014-1028`): `configure_trainable_parameters` *identifies* the vision encoder to freeze (always for VLM ckpts trained text-only; configurable with `[model.vlm]`) and applies LoRA (`:896-908`); optional `freeze_moe_router` (`:121-146`); fp32 routers (`moe_router_dtype="float32"` default → `router.to(fp32)` + `fp32_gate=True`, `:149-169`); **always freeze sparse-attention indexers** (they run under `no_grad`; leaving them trainable breaks strict DCP resume, `:192-214`); optional forced round-robin routing (`:217-235`).
8. **MoE runtime** `configure_moe_runtime(model, config, parallel_dims)` — grouped GEMM backend, token dispatcher, and **EP sharding of expert params** (`moe_runtime.py:59-133`) — §3.10. Re-freeze LoRA params afterwards because EP re-creates params as DTensors with `requires_grad=True` (`model.py:1031-1035`); only *then* is the vision encoder actually frozen (`:1037-1041`).
9. **AC** wrap every `freq`-th decoder block in place (`apply_ac`, `:848-862`) — §3.9.
10. **Compile** each decoder block in place with `layer.compile(fullgraph, mode)` (keeps FQNs stable for checkpoints; raises dynamo `recompile_limit=16`, `cache_size_limit=64`; `:103-106, 865-873`). On by default (`CompileConfig()`, `configs/trainer.py:282`).
11. **FSDP2** `setup_fsdp` (§3.4).
12. **Weight materialization** (`:1054-1075`):
    - If resuming: `model.to_empty(device)`, barrier, `init_buffers_post_meta()` (custom) or `fix_model_post_empty` (HF: recompute rotary `inv_freq`, Gemma `embed_scale`) + `tie_weights()`. Weights come from DCP later.
    - Else `load_dcp_from_hf` (below).
13. Zero MoE runtime stats buffers (`:920-924, 1077`).

**`load_dcp_from_hf` (`model.py:657-765`)** — how HF safetensors land in sharded DTensors:
1. `to_empty(cuda|cpu)`, barrier, re-init buffers *before* loading (persistent buffers such as `selection_bias` must be overwritten by the checkpoint, tested in `test_state_loading.py:33-49`).
2. `debug.random_init` → return (random-init debugging).
3. Resolve `snapshot_path` (local path or `snapshot_download`).
4. **Format auto-conversion (custom models only)**: every rank reads only safetensors *key names* (`load_state_dict_keys`, `utils/weights.py:25-31`); if the snapshot is HF-format and the model's own keys are prime-format, master alone loads the **whole** state dict to CPU, runs `model.convert_to_prime`, and writes a converted copy to `(conversion_dir or snapshot)/prime` plus a `.prime-v1` marker (`:697-709`); the reverse (prime snapshot → HF-keyed model) writes `…/hf` (`:712-724`). Everyone barriers; a `prime` dir without marker is fatal (`:691-692, 728-733`).
5. `dcp_load(model.state_dict() minus LoRA keys minus tied lm_head, storage_reader=HuggingFaceStorageReader(snapshot_path))` — each rank reads only the slices of its DTensor shards (`:735-744`).
6. `write_back_loaded_packed_parameters` — packs logical q/k/v or gate/up entries back into fused params when the state-dict entries were copies (`fusions.py:194-204`).
7. Re-tie HF models; move buffers to CUDA for FSDP CPU offload (`:746-750`). The following LoRA block (seed broadcast across `dp_replicate`, `:752-764`) is **dead code**: it selects modules with a `_init_lora_parameters` attribute, which no class defines (grep), so the list is always empty. Actual adapter init is `get_lora_state().reset_adapter_parameters()` on the already-sharded DTensors (`rl/train.py:199-202`; §3.15).

**`PreTrainedModelPrimeRL` interface** (`models/base.py:21-156`) — the contract a custom family implements:

| Member | Default | Purpose |
|---|---|---|
| `cp_context: CPContext` (class attr) | `CPContext()` | replaced per module by `setup_context_parallel` (`utils/cp.py:59-68`) |
| `cp_support(config) -> CPSupport` | both `ring`,`ulysses` | which CP styles work; validated at build (`model.py:415-423`) |
| `keep_in_fp32_for_weight_transfer(name) -> bool` | `False` | per-tensor wire dtype override (§3.17.1) |
| `is_hf_state_dict(sd)`, `is_prime_state_dict(sd)` | raise | format detection for load + export |
| `conversion_chain(config) -> list[ConvOp]` | `[]` | declarative HF↔prime mapping |
| `convert_to_hf(sd)` / `convert_to_prime(sd)` | play chain backward/forward in place (`:111-121`) | full-dict conversion |
| `convert_layer_to_hf(sd, idx)` / `convert_layer_to_prime` | call full-dict version (`:123-129`) | NCCL per-layer path |
| `convert_adapter_to_hf(sd)` (classmethod) | identity | LoRA adapter key rename (NemotronH overrides) |
| `init_buffers_post_meta()` | raise | rebuild non-checkpointed buffers after `to_empty` |
| `from_config`, `_check_and_adjust_attn_implementation` (FA3 default, bypasses HF checks), `get_correct_experts_implementation` → `"eager"` | | HF-API shims (`:50-78`) |

**Forward contract** (`model.forward`, `model.py:1081-1138`): kwargs `input_ids [1,S]`, `labels [1,S]`, `temperature [1,S]`, `position_ids` (omitted when `image_grid_thw` present → MRoPE computed in-model), optional `sampling_mask [1,S,K]`, `mm_kwargs` (verbatim per-model dict) + `mm_token_type_ids`, and for custom models `seq_lens [num_docs]` + `seq_lens_are_pre_shard`, plus `routed_experts [1,S,L,topk]`. Because `inject_prime_lm_head` rebinds `model.forward` to call `self.model(input_ids, position_ids, **kwargs)` then `self.lm_head(hidden, labels, temperature, sampling_mask)` (`lm_head.py:319-354`), **the family's own `*ForCausalLM.forward` is dead code in training**; the inner `*Model.forward` must accept `seq_lens`, `seq_lens_are_pre_shard`, `routed_experts` and mm kwargs. Output is a `PrimeLmOutput` TypedDict (`logits` or `logprobs`+`entropy`), cast to fp32 contiguous (`lm_head.py:14-34`).

Packed-row contract: batch dim is 1 and the row length varies per micro-batch (padded upstream only to `pad_to_multiple_of`, C); `seq_lens` are document lengths whose sum must equal the row length (or the pre-shard length under CP), converted to varlen `cu_seqlens` + `max_seqlen` (`utils/sequence.py:38-52`; e.g. `qwen3/modeling_qwen3.py:176-180`). **HF-impl models never receive `seq_lens`** (`model.py:1125-1127`): document boundaries reach them only through the per-document-restarting `position_ids`, i.e. transformers' flash-attention packed-sequence detection — which is why `attn` is restricted to the flash-attention literals (`configs/trainer.py:24`).

### 3.4 FSDP2 wrapping policy (`model.setup_fsdp`, `model.py:502-654`)

Common config: `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=reduce_dtype)` (default fp32 reductions), `CPUOffloadPolicy(pin_memory=True)` iff `fsdp_cpu_offload`, `reshard_after_forward` (default `True`), optional `shard_placement_fn` (fused-dim1, §3.14) (`:503-512`). Units, in call order:

1. Vision encoder (VLM only) on `hsdp` (`:525-532`).
2. Per decoder block: if EP and the block's `mlp` is a `MoE` → `fully_shard(block.mlp.experts, mesh=dp_mod_ep)` and `set_gradient_divide_factor(fsdp_gradient_divide_factor)` so expert grads are averaged over the same count as dense grads (`:537-542`); if fp32 router → its own unit with an fp32/fp32 policy and `set_reduce_scatter_max_input_buffers(2)` (`:544-556`); then the block itself on `hsdp` (`:558-562`).
3. If embeddings are **not tied**: embeddings as a unit, and `[lm_head, final_norm]` as one unit with `reshard_after_forward=False` (kept unsharded between forward and backward) (`:564-582`). Tied models skip this with a warning.
4. Root `fully_shard(model)` (`:586-593`).
5. With EP only: manual forward/backward prefetch chains (embed→block0, blockᵢ→[blockᵢ₊₁, router, experts], last→[norm, lm_head]; reverse for backward), because D2H syncs in dispatch/combine defeat FSDP's implicit prefetch (`:595-654`).

Gradient math across ranks: `compute_loss` already divides by the **global** token count (dp_cp all-reduce, `rl/train.py:306-322`), FSDP then *averages* over the DP×CP mesh, so the trainer multiplies grads back by `fsdp_gradient_divide_factor` after the micro-batch loop (`rl/train.py:561-564`; `utils.py:77-84`), or folds that factor into the offload manager's `gradient_scale` (`rl/train.py:323-327`).

### 3.5 Training-loop skeleton (RL, `rl/train.py:253-719`)

```mermaid
sequenceDiagram
    participant T as Trainer (all ranks)
    participant B as Batch transport (E/C)
    participant W as WeightSender (master-driven)
    participant O as Consumer (orchestrator)
    Note over T: first iteration only
    T->>W: broadcast(model, v=start_step-1)
    W->>O: .sender_ready  (master)
    O-->>W: .receiver_ready
    W->>O: .started, transfer, .finished
    loop step = start_step..max_steps
        T->>B: wait_for_batch()  (blocks)
        B-->>T: micro_batches (per DP rank)
        T->>T: all_reduce(token counts, dp_cp)
        T->>T: for mb: forward → loss → backward (FSDP AG/RS, EP a2a)
        T->>T: scale grads ×fsdp_divide, clip, optimizer.step, zero_grad, scheduler.step
        T->>T: cuda.synchronize(); empty_cache()
        T->>W: broadcast(model, v=step)  (blocks on consumer ack + transfer)
        opt every ckpt.interval (not last)
            T->>T: DCP save step_N/trainer (blocking), maybe_clean
        end
        T->>T: Tensors.compute_stats (world all_gather), monitors.log ×5 (asyncio.run each)
    end
    T->>T: final DCP save (if ckpt)
```

| Phase | Blocking? | Overlap |
|---|---|---|
| Startup broadcast v{start-1} (`:267-276`) | yes, master polls `.receiver_ready` every 0.1 s (`base.py:83-92`) | none — deliberately fail-fast |
| `wait_for_batch` (`:281`) | yes (receiver `.wait()`) | none; warns when wait ≥ active step time (`:636-641`) |
| token-count all-reduce (`:317-322`) | collective | — |
| fwd/bwd per micro-batch (`:336-557`) | GPU | FSDP prefetch; DeepEP chunk pipeline; full-offload grad D2H + CPU optimizer chunks during the **last** backward (`begin_backward(final_backward=…)`, `:499-501`) |
| clip (`:568-569`) | collective (disabled under full offload / DeepEP) | — |
| `optimizer.step` (`:572`) | state-offload: H2D states, step, D2H + `cuda.synchronize` (`state_offload.py:57-74`); full-offload: waits for CPU chunks to finish (`offload.py:1329-1344`) | full-offload overlapped with backward |
| weight broadcast (`:583-597`) | yes; `cuda.synchronize()` + `empty_cache()` first to give the gather headroom (per-step only — the startup broadcast at `:267-276` does neither) | none |
| checkpoint (`:600-613`) | yes (synchronous `dcp_save`) | none (no async DCP) |
| metrics (`:620-711`) | `all_gather_object` + per-key all-gathers over the **world** group (`utils.py:247-283`) | none |

Timing metrics: `time/forward_backward` spans `:297`→`:579`, i.e. token-count all-reduce, all micro-batches, grad rescale, clip, `optimizer.step`, `zero_grad` **and** `scheduler.step`; it excludes batch wait/load, broadcast and checkpoint (separate `time/*` keys, `:683-691`). Throughput/MFU use `seq_len = micro_batches[0]["input_ids"].shape[1]` × number of micro-batches × dp size (`:298, 623-624`) — rank-local first-micro-batch length, not the true token count — and `get_perf_counter` is a process singleton built from the first step's `seq_len` (`perf.py:240-245`), whose attention FLOPs use `hidden_size // num_attention_heads` rather than `head_dim` (`perf.py:182-194`).

`Progress` (`step=1, total_tokens, total_samples`, `ckpt.py:33-37`) is the loop counter; `progress.step` equals the policy version being *produced* by this iteration; the broadcast after the optimizer publishes `v{step}` (`:596`) — S9 accounting is C's.

### 3.6 Micro-batch forward details relevant to the engine (C owns the loss)

Per micro-batch the trainer moves tensors to CUDA, shifts labels left, optionally CP-shards inputs/labels/temperatures/routed-experts/sampling-mask, sets LoRA per-adapter token counts, and calls `forward` under `maybe_activation_offloading` (`rl/train.py:336-452`). If the LM head returned raw logits (vanilla head) it computes `log_softmax` with per-token temperature externally (`:454-464`). Under CP, logprobs are all-gathered with an autograd-aware collective (backward = reduce-scatter) and entropy without grad (`:467-469`; `utils/cp.py:104-111`; `collectives.py:155-219`). **CP normalization cancels exactly**: the token counts (`:306-322`) are taken from the *unsharded* micro-batch on every CP rank, so the dp_cp all-reduce yields $cp\cdot N$; every CP rank then computes the loss over the full gathered sequence, i.e. $L/cp$; the gather's reduce-scatter-sum backward multiplies each shard's gradient by $cp$, and the ×`fsdp_gradient_divide_factor` undo of FSDP averaging (`:561-564`) restores the sum — net gradient $=\nabla L$ with $L$ normalized by the true $N$. MoE routing stats are reduced EP→DP×CP per micro-step (`:545-547`; `model.py:243-304`).

### 3.7 Optimizers and LR schedulers

`setup_optimizer` (`optim/__init__.py:50-105`) → `_create_optimizer` hands **only `requires_grad` params** to the optimizer (frozen params would break strict DCP resume, `:119-123`):

| `optim.type` | Class | Notes |
|---|---|---|
| `adamw` (default) | `torch.optim.AdamW(betas=(betas1, betas2), weight_decay, fused=…)` | `fused=True` only when `optim_cpu_offload` is **off** (`:86`) — with the default `optim_cpu_offload=True`, AdamW is non-fused |
| `sgd` | `torch.optim.SGD(momentum=0.9, nesterov=True)` | |
| `muon` | `dion.Muon` | param groups: ≥2-D non-embed/non-lm_head → `muon`; experts (`mlp.experts`) own group with `distributed_mesh_name="dp_shard_mod_ep"` under EP; routers own group; the rest → `adamw` sub-algorithm (`:151-214`). `distributed_mesh = dp_shard_cp` (or world mesh), `fsdp_mesh_dim = 1 if dp_replicate>1 else 0`. Fused (packed) params are passed as `matrix_partitions` so each logical matrix is orthogonalized separately (`:221-230`). **Mandatory NCCL warm-up** bulk all-to-all on each Muon mesh before the first step (multi-node deadlock workaround, `:23-47, 244-248`). Incompatible with `fsdp_cpu_offload` (`configs/trainer.py:788-792`). |
| `sign_sgd` | `SignSGD` (`sign_sgd.py:7-53`): `p ← p − lr·wd·p − lr·sign(g)` | stateless; supported by full offload |

Gradient clipping: `optim.max_norm` (default 1.0) via torchtitan `clip_grad_norm_(…, ep_enabled)` or the offload manager's own norm (`utils.py:87-97`); auto-disabled with DeepEP dispatch (`configs/trainer.py:735-744`) and with full offload (`:752-761`; also `optim/__init__.py:63-65`).

Schedulers (`scheduler.py`), always stepped once per trainer step (`rl/train.py:576`) on the *base* optimizer (`scheduler.py:10-14, 98`):
- `constant` → `ConstantLR(factor=1)`.
- `linear` (WSD) → `LinearLR` warmup from `min_lr/lr` (or 1e-8) to 1 over `warmup_steps`, constant, then linear decay to `min_lr/lr` over `decay_steps-1` iterations starting at `max_steps − decay_steps`, glued with `SequentialLR` (`:22-58`).
- `cosine` → optional linear warmup then `CosineAnnealingLR(T_max=max_steps−warmup, eta_min=min_lr)` (`:61-85`).
Config-time validation in `validate_scheduler` (`configs/trainer.py:471-490`).

### 3.8 Optimizer / parameter offloading (three mutually exclusive modes)

Exclusivity enforced in `ModelConfig.cpu_offload_mutual_exclusion` (`configs/trainer.py:388-397`).

**(a) FSDP CPU offload** (`fsdp_cpu_offload`): FSDP2 `CPUOffloadPolicy(pin_memory=True)` (`model.py:504`) — params, grads, optimizer state on CPU; buffers are moved to CUDA manually (`model.py:911-917`); requires Gloo (step 3 above).

**(b) State-only offload** (`optim_cpu_offload=True`, **default**) — `CPUOffloadOptimizer` (`optim/state_offload.py`): first `step()` runs on GPU then moves state to pinned CPU buffers (reused per `(param, key)`, `:24-33`); every later step moves all state to GPU (`non_blocking`), steps, moves back to CPU and `cuda.synchronize()`s (`:57-74`). DTensor state is handled by swapping `_local_tensor` (`:40-48`). Checkpoint save temporarily moves state to GPU (`:113-119`). Tested for exact parity with plain AdamW (`tests/unit/train/test_state_offload.py:79-99`).

**(c) Full offload with optimizer-in-backward** (`full_offload`, `optim/offload.py`, "fully AI-generated" per its header `:1`):
- `_create_cpu_master_weights` copies each trainable DTensor's local shard into one **fp32 CPU slab** (pinned unless the native backend is used), creates CPU `nn.Parameter` masters, and **casts the GPU model to bf16 in place** except params whose dtype policy says fp32 (fp32 MoE routers, from `get_full_offload_dtype_policy`, `model.py:172-189`) (`offload.py:1568-1616, 1523-1565`). The optimizer is built over the CPU masters.
- `FullCPUOffloadOptimizer` groups params into ~16 M-element chunks in param-group order (`:1131-1147`). Gradients leave the GPU via `register_post_accumulate_grad_hook` on every FSDP sharded param (`:178, 585`).
- Two managers: `GradientOffloadManager` (torch CPU optimizer backend; per-dtype pinned accumulator + staging slabs pre-allocated at init to avoid a lazy `cudaHostAlloc` deadlock inside FSDP's reduce hook, `:130-184`; one copy thread `grad-offload`) and `BoundedGradientOffloadManager` (native backend, default; pageable fp32 accumulator slab, **bounded pinned transfer rings** of 4 slots per dtype + one oversized slot for D2H and for H2D, up to 16 in-flight backward generations, 120 s timeouts; three threads `grad-transfer`, `grad-offload`, `weight-transfer-reclaimer`; `:474-616`).
- On the **final** micro-batch backward, when the last gradient of a chunk lands, the copy worker thread runs `_step_cpu_chunk`: native multi-tensor AVX2/AVX-512 AdamW or SignSGD kernel (JIT-built C++ extension `cpu_adam/kernel.cpp` via `torch.utils.cpp_extension.load`, `cpu_adam/__init__.py:32-41`) on the fp32 masters, then writes the bf16 compute weights back into the GPU shard on a dedicated H2D stream (`offload.py:1282-1312, 1160-1185`). `optimizer.step()` on the main thread only drains (`wait_for_optimizer`) and makes the current stream wait on the H2D stream (`:1329-1344`). A synchronous path exists for steps without overlap (SFT validation steps, `:1345-1374`; `sft/train.py:464-468`).
- Numerics: gradients are reduced in fp32 but materialized in the bf16 param dtype, so each gradient is rounded to bf16 once before the fp32 CPU update (documented in `configs/trainer.py:51-62`). The DP `fsdp_gradient_divide_factor` is carried as a scalar `gradient_scale` into the kernel (`offload.py:1236`).
- Checkpoint integration: `checkpoint_optimizer()` exposes the CPU state under the *model* params' FQNs as DTensors sharing the model param's `_spec`, and adds the fp32 master as extra state key **`prime_rl_master_weight`** (`:1071, 1452-1488`); after load, masters are pushed back into the bf16 GPU weights (`finish_checkpoint_load`, `:1490-1509`); a model-only load (`skip_optimizer`) instead copies GPU→CPU masters (`:1511-1520`).
- Host-side: NUMA binding + thread count (§3.1 step 4); gradient clipping forced off.

### 3.9 Activation checkpointing and activation offloading

**AC** (`activation_checkpointing.py`): whole-block `checkpoint_wrapper(CheckpointImpl.NO_REENTRANT)` with a selective-checkpoint context (`:113-124`), applied to every `ac.freq`-th block (`model.py:848-862`). Both modes use an op-level policy:
- **Mandatory MUST_SAVE** in *both* modes: everything in the `deepep` namespace, `aten::topk`, `prime_rl::fp8_indexer`, `prime_rl::record_moe_routing_statistics`, and CUDA→CPU `_to_copy` — ops that are non-replayable/stateful (`:31-38, 81-95`). This is why MoE routing stats are recorded exactly once under AC (`tests/unit/train/models/test_checkpointing.py:72-99`).
- `mode="full"` (default): everything else recomputed. `mode="selective"`: additionally save matmuls/grouped GEMMs/attention kernels/collectives (namespaces `_c10d_functional`, `flash_attn`, `flash_attn_3`, `prime_rl_collectives`, `prime_rl_ring`, plus listed aten/prime_rl ops; `:40-76`); `ac.targets` replaces the default set but never the mandatory set.
- Profiler record-function ops are added to `SAC_IGNORED_OPS` (`:19-26`).

**Activation offloading** (`utils/act_offloading.py`, adapted from torchtune): a `saved_tensors_hooks` context entered around each micro-batch forward (`rl/train.py:439`). Saved CUDA tensors ≥1024 B that are not params/buffers are copied to (pinned) CPU during forward and brought back in backward; **streams are hard-disabled** (`use_streams=False`, `:62-64`) so copies are on the compute stream; warns at >90 % host RAM (`:95-100`). Enabled by default (`ac_offloading = ActivationOffloadingConfig()`, `configs/trainer.py:291`) and it force-enables AC (`:381-386`). `max_inflight_activations` is passed but only matters for the (disabled) stream path.

### 3.10 MoE runtime and expert parallelism

**Layer anatomy** (`layers/moe.py`): `MoE(router, experts, shared_expert, score_before_experts, load_balance_coeff)` (`:325-395`).
- `TokenChoiceTopKRouter`: `gate` Linear (fp32 GEMM when `fp32_gate`), score func `sigmoid|softmax` computed in fp32 (`topk_softmax` uses the raw logits — fp32 only with `fp32_gate` — and softmaxes the selected k, `moe.py:265-266, 294-295`), top-k on `scores + selection_bias` (selection-only bias), gather of un-biased scores, optional `route_norm` and `route_scale`, `histc` token counts (`:182-317`). **Router replay**: when `routed_experts` is given, the selected experts are taken verbatim and only scores are gathered (`:274-276`); requires `enable_router_replay` + vLLM returning routed experts (`rl/train.py:352-359`). `force_balanced` gives round-robin assignment (`:277-281`).
- `selection_bias` is a persistent fp32 buffer loaded from HF (`e_score_correction_bias`/`expert_bias`); **no code updates it during training** — the comment about an optimizer pre-hook (`moe.py:382-384`) has no implementation: no `register_step_pre_hook`/`register_optimizer_step_pre_hook` anywhere in `src/`, `load_balance_coeff` is only stored and asserted `>0` (`moe.py:385-387`), the only writers of `selection_bias` are init/zeroing paths (`moe.py:479-480`, `laguna`/`qwen3_5` `init_buffers_post_meta`), and the `tokens_per_expert` counts a bias update would need are zeroed after every micro-step's stats read (`model.py:279`). Net effect: the pretrained bias is frozen (arguably right for RL fine-tuning, but undocumented).
- `GroupedExperts`: params `gate_proj [E, H, D]`, `up_proj [E, H, D]`, `down_proj [E, D, H]` (+ optional biases), or fused `gate_up_proj [E, 2H, D]`; compute via the pluggable `GroupedGemm` on local shards (`to_local()` on DTensors) in bf16 (`:85-164`). Activations `silu | relu2 | clamped_swiglu` (`activations.py:37-43`).
- Stats: `tokens_per_expert` and `routing_confidence_sum` non-persistent fp32 buffers accumulated through the custom op `prime_rl::record_moe_routing_statistics` (`:24-45, 436-443`); read and zeroed per micro-step by `get_load_balance_stats` → `max_vio` (after dropping the `top_k` busiest experts) and routing confidence (`model.py:243-284`).

**Compute backends** (`grouped_gemm.py`, selected in `moe_runtime._resolve_grouped_gemm`, `moe_runtime.py:34-56`): `BF16GroupedGemm` (`torch._grouped_mm`, alignment 8), `DeepGemmFP8GroupedGemm` (DeepGEMM `m_grouped_fp8_gemm_nt_contiguous` fwd + K-grouped wgrad; SM90+, `fp8_grouped_gemm.py`), `MXFP8GroupedGemm` (`prime_kernels.load("mxfp8_moe")`, SM100, alignment from the kernel). `moe.compute.apply_to` selects a layer subset (`"all"`, `"85%"`, or indices); other layers use BF16 (`configs/trainer.py:187-211`; `moe_runtime.py:66-91`).

**Dispatchers** (`distributed/token_dispatcher.py`, `deepep.py`), chosen per MoE layer (`moe_runtime.py:92-125`):

| Dispatcher | When | Mechanism |
|---|---|---|
| `LocalTokenDispatcher` | `ep == 1` | local argsort by expert, `permute_for_grouped_gemm` (torchtitan `generate_permute_indices`, pads groups to alignment), scatter-add combine (`token_dispatcher.py:168-216`) |
| `TorchTokenDispatcher` | EP, `dispatch.type="torch"`, `transport="bf16"` | exchanges per-expert counts with an equal all-to-all, moves splits to CPU once, then variable-size `all_to_all_single` (autograd-aware custom ops) dispatch and combine (`:219-305`; `collectives.py:9-61`) |
| `MXFP8TorchTokenDispatcher` | EP, torch dispatch, `transport="mxfp8"`, MXFP8 compute | quantized all-to-all from `prime_kernels` (`:308-332`; `collectives.py:95-152`) |
| `DeepEPTokenDispatcher` | EP, `dispatch.type="deepep"` | DeepEP `Buffer` dispatch/combine registered as `deepep::dispatch/combine` custom ops with autograd, async events; optional `token_chunk_size` pipelines dispatch of chunk i+1 with expert compute of chunk i (lazy generator, `deepep.py:415-459`); `num_sms` via `Buffer.set_num_sms` (`:189-195`). Grad clipping unsupported. |

**EP weight sharding**: `parallelize_module(moe.experts, ep_mesh, ExpertWeightParallel())` replaces each expert param with a `DTensor` `Shard(0)` on the `ep` mesh (`distributed/expert_parallel.py:6-14`; `moe_runtime.py:127-128`); FSDP then shards those on `dp_shard_mod_ep` (§3.4). Expert count must be divisible by `ep` (`moe_runtime.py:87-90`). Constraints: MXFP8 compute ✗ DeepEP; `transport="mxfp8"` requires MXFP8 compute (`configs/trainer.py:412-425`).

### 3.11 Context parallelism

Two styles, selected by `model.cp_style` (default `"ring"`), installed by **monkeypatching class methods** (`utils/cp.py:40-70`):
- `ring`: `ring_flash_attn.substitute_hf_flash_attn` (HF models) + `substitute_ring_attn` which rebinds `FlashAttention._compute_attention` (and `AfmoeFlashAttention`, GPT-OSS attention) to `ring_varlen_attention` (`attn.py:167-208`). Despite the name it is an **all-gather of K/V** (per `heads_k_stride` KV-head group, double-buffered, `AllGatherComm`) followed by a local varlen flash call with `local_k_slice`; backward all-gathers again and reduce-scatters dK/dV (`ring_attn.py:188-374`). Backends FA2/3/4 (`:11-185`).
- `ulysses`: all-to-all seq→head before attention and head→seq after; any kernel works; GQA with fewer KV heads than `cp` replicates KV heads; learnable sinks sliced per rank (`ulysses_attn.py:53-163`). Also patches HF `_flash_attention_forward` / `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` (`:231-334`).

Per micro-batch, `setup_cp_params` computes **global** `cu_seqlens` from pre-shard `seq_lens`, publishes them to the patched attention (`DATA_PARAMS` for ring, `ULYSSES_PARAMS` for ulysses), then shards `input_ids`/`position_ids` into `cp` **contiguous** chunks (`torch.chunk`, `utils/cp.py:73-139`). Models that need CP-awareness beyond attention (DSA sparse MLA, DeepSeek V4, Mamba, Gated DeltaNet) read `self.cp_context` (`glm_moe_dsa/sparse_mla_attention.py:148-223`, etc.). Per-family support: DeepSeek V4 ring only; NemotronH and Qwen3.5 with linear-attention layers ulysses only; others both (`deepseek_v4/modeling_deepseek_v4.py:111-116`, `nemotron_h/modeling_nemotron_h.py:135-140`, `qwen3_5/modeling_qwen3_5.py:123-131`). VLM + CP requires ulysses (`configs/trainer.py:364-368`); CP requires the custom impl and flash attention (`:370-379`).

### 3.12 LM head: fused chunked logprobs/entropy

`inject_prime_lm_head` replaces `model.lm_head` (must be a bias-free `nn.Linear`) with (`lm_head.py:269-316`):
- `FusedOutputLinear(chunk_size)` when `fused_lm_head_token_chunk_size` is an int (default **8192**, `configs/trainer.py:347`): a custom `autograd.Function` that loops over token chunks × vocab chunks of 8192, keeps an online log-sum-exp and entropy accumulator in fp32, extracts the target logit branchlessly (no host sync), and returns per-token `logprobs` and `entropy` without materializing `[N, V]` logits (`:114-208`). Backward recomputes chunk logits and produces grads for hidden and weight; **no gradient through entropy** (asserted, `:210-266`). Per-token temperature is applied as `logits · (1/T)` **before** the online log-sum-exp and target gather (`:176-178`); the chunk logits come from a GEMM in the compute dtype (bf16 under FSDP mixed precision) and are upcast to fp32 afterwards, so only the reductions are fp32.
- Sampling-mask replay: when `sampling_mask [N, K]` (vocab ids the sampler allowed, `-1` padded) is supplied and contains the label, logprob is renormalized over the mask; entropy stays full-vocab (`:97-101, 167-199, 243-251`).
- `VanillaOutputLinear` (`"disabled"`) returns logits; temperature/log-softmax are done in `rl/train.py:454-464`.
- Gemma-style `final_logit_softcapping` dispatches to `lm_head_gemma.py` (softcap inside the chunk loop; sampling masks unsupported, `lm_head_gemma.py:30`).
- Applies to HF-impl models too (injection is unconditional in `setup_model`).

### 3.13 Low-precision training (compute only; exported weights stay bf16/fp32)

- **Dense FP8** (`quantization.type="fp8"`): `Float8BlockwiseLinear` → custom op `prime_rl::fp8_blockwise_mm` quantizing activations per-token and weights per-128×128-block on the fly (Triton casts) and calling DeepGEMM `fp8_gemm_nt`; backward re-quantizes for dX and uses the `(1,1,128)` recipe for dW into an fp32 buffer (`fp8_linear.py:18-137`). Skips layers matching `ignore_patterns` (default: `lm_head`, `router`, `mlp\.gate\.`, `shared_expert\.output_gate`, `eh_proj`, `weights_proj`, `in_proj_a`, `in_proj_b`; `configs/trainer.py:152-166`) and any dim not divisible by 128 (`fp8_linear.py:153-200`).
- **Dense MXFP8** (`type="mxfp8"`, SM100+): torchao `_to_mxfp8_then_scaled_mm`, recipe `mxfp8_rceil` or `mxfp8_rceil_wgrad_with_hp`; dims must be multiples of 32 (`mxfp8_linear.py`).
- **Routed experts**: separately via `moe.compute` (§3.10).
- The master weights remain in `optimization_dtype`; nothing in the trainer emits FP8 weights for inference. `models/fp8.py:quantize_to_fp8_blockwise` is used only by the offline tool `tools/convert_bf16_to_fp8.py` (grep-confirmed).

### 3.14 Runtime fusions (`models/fusions.py`)

Default `fusions.enabled = ["gate_up", "qkv"]` (`configs/trainer.py:92-100`), applied to every module advertising `supported_fusions` (`GroupedExperts` gated → `gate_up`; `FlashAttention`/GPT-OSS attention → `qkv`) (`fusions.py:121-141`; `moe.py:86`; `attn.py:53`). Skipped when LoRA is on (`model.py:1000-1001`).
- `PackedParameterSpec(name, logical_names, sizes, dim)`: one physical parameter (`gate_up_proj [E,2H,D]` packed on dim 1; `qkv_proj.weight [q+k+v, D]` + bias packed on dim 0) (`:16-44, 70-118`).
- **Checkpoints and exports stay canonical**: a `state_dict` post-hook splits the packed entry into logical views (`q_proj.weight`, `gate_proj`, …) and a load pre-hook re-packs (`:47-67`; verified by `tests/unit/train/models/test_fusions.py:39-54`). Views alias the packed storage *unless* the DTensor is sharded along the packing dim, in which case the split redistributes and yields copies. Concretely (probed with DTensor on a 2-rank gloo mesh): `gate_up_proj [E,2H,D]` packs on dim 1 while FSDP/EP shard dim 0 → live `Shard(0)` views; `qkv_proj` packs on dim 0 = FSDP's `Shard(0)` → every `state_dict()` call returns **`Replicate` full copies** of `q/k/v_proj` (an all-gather per call, and the copies do not track later in-place updates); with a shard mesh of size 1 the split stays a view. Then `write_back_loaded_packed_parameters` (load) and `split_packed_optimizer_state_for_checkpoint` / `join_loaded_optimizer_state_for_runtime` / `write_back_loaded_packed_optimizer_state` (optimizer state) do the packing (`:194-312`; used in `ckpt.py:90-93, 128-146`).
- `shard_fused_on_dim1` (experimental): FSDP `Shard(1)` for params packed on dim 0 so the logical split is a local view (zero-copy loads/checkpoints); requires hidden size divisible by the shard mesh (`:174-191`; `model.py:506`).
- Muon: packed params are orthogonalized per logical partition (`muon_matrix_partitions`, `:42-44`).

### 3.15 LoRA

- `apply_lora_to_model` (must run **before** FSDP; raises otherwise, `lora.py:231-237`) creates the `LoRAState` singleton holding the shared `lora_num_tokens` (int32 `[n_adapters]`) and `scaling_factors` (`alpha/rank`) tensors that every LoRA layer captures by reference (`lora.py:26-94`), then wraps target modules: `nn.Linear` → `MultiLoRALinear`, `GroupedExperts` → `MultiLoRAGroupedExperts` / `MultiLoRANonGatedGroupedExperts` / `MultiLoRAGptOssGroupedExperts` (GPT-OSS combined gate/up adapter format) (`lora.py:248-283`). Matching: plain names match any FQN component; regex patterns use `re.search` (`lora.py:125-168`). Everything except `lora_A/lora_B` and `modules_to_save` is frozen (`lora.py:193-207`).
- **"Multi"-LoRA is vestigial**: `n_adapters=1` is hard-coded (`lora.py:256, 273`); layers support `n_adapters` via grouped GEMMs over `lora_num_tokens` offsets (`multi_linear.py:136-158`), MoE LoRA picks `argmax(lora_num_tokens)` as the single active adapter (`multi_moe.py:286`).
- State-dict hygiene: `MultiLoRAModule` strips/restores the `base_layer.` prefix so base weights keep HF names (`lora/base.py:85-177`); `strip_lora_from_state_dict` excludes adapter keys from HF loading (`lora.py:97-104`; `model.py:738`).
- Adapter init: kaiming-uniform A, zero B (the constructor's init runs on the meta model and is a no-op); the effective init is `reset_adapter_parameters()` → `reset_parameters(0)` on every registered module *after* FSDP, directly on the sharded DTensors (`rl/train.py:199-202`; `lora.py:75-78`); a resume overwrites it. The `dp_replicate` seed broadcast in `model.py:752-764` never runs (dead code, §3.3). Replica agreement under `dp_replicate>1` therefore rests on DTensor's offset-based RNG for random init on DTensors `[UNVERIFIED]`.
- Adapter export (filesystem only; §3.17.2): `LoRAState.adapter_state_dict()` → `{"<module FQN>.lora_A.weight", ".lora_B.weight"}` for linears, per-expert `"<experts FQN>.<e>.<proj>.lora_{A,B}.weight"` for MoE (`multi_moe.py:230-277`), GPT-OSS 3-D stacked format (`multi_moe.py:722-757`), then `model.convert_adapter_to_hf` (`lora.py:58-73`). `save_lora_config` writes a PEFT `adapter_config.json` (`lora.py:314-358`).
- Constraints: NCCL/NIXL transports rejected with LoRA (`configs/trainer.py:794-800`; under `rl` the default auto-falls back to filesystem, `configs/rl.py:447-457`); `modules_to_save` rejected for real runs because broadcasts ship only adapters (`configs/trainer.py:801-805`); VLM with trainable encoder ✗ LoRA (`:763-770`).

### 3.16 Checkpointing and resume (S8, trainer side)

**Layout** (`utils/pathing.py:257-298`; `ckpt.py:179-181`):
```
<ckpt.output_dir or output_dir>/checkpoints/
  step_<N>/
    trainer/            # DCP checkpoint_id  (this section)
      .metadata         # DCP global metadata (written by coordinator at end of save)
      __<rank>_0.distcp # per-rank shard files (DCP FileSystemWriter default naming)
      dataloader/rank_<r>.pt   # SFT only (RL passes no dataloader)
    orchestrator/       # B's — only when ckpt.output_dir is unset (else under <run_dir>/checkpoints)
    weights/, weights-FP8/     # optional offline exports (tools/)
```

**What is saved** — `dcp_save({"app": AppState(model, [optimizer], scheduler, progress)}, checkpoint_id=path)` (`ckpt.py:197-211`). `AppState.state_dict()` (`:79-110`):
- `model`: `torch.distributed.checkpoint.state_dict.get_state_dict` → FSDP **sharded** model state dict with canonical FQNs (fusions split to logical names; AC/compile wrappers stripped). Resharding-safe: can resume at a different world size / mesh (DCP reshards).
- `optimizers`: optimizer state dict keyed by param FQN, with packed params split into logical entries (`fusions.split_packed_optimizer_state_for_checkpoint`). With full offload, includes the CPU fp32 masters under state key `prime_rl_master_weight` (§3.8c).
- `scheduler`: `LRScheduler.state_dict()`; `progress`: `{step, total_tokens, total_samples}`.
- RL does **not** save a dataloader (batches are the orchestrator's; `rl/train.py:608, 732` pass none).

**When**: every `ckpt.interval` steps except the last, plus a final save after the loop (`rl/train.py:600-613, 730-733`); `ckpt=None` → no saves at all (a `resume` still loads). **Synchronous**; no async DCP, no staging.

**Cleaning** `maybe_clean` (`ckpt.py:290-322`): keep the last `keep_last` steps ∪ multiples of `keep_interval`; master `rmtree`s the **whole `step_<N>` directory** (i.e. also the orchestrator's sub-dir and any `weights/` export) (`:313-319`), with no barrier afterwards. The trainer is the **only** cleaner: the orchestrator's `CheckpointManager` has no cleanup and never reads its `keep_*` (`orchestrator/ckpt.py:22-72`), and `validate_shared_ckpt_config` only enforces that both sides have `ckpt`, equal `interval`, and equal `resume` (`packages/prime-rl-configs/src/prime_rl/utils/validation.py:194-213`) — so retention is governed solely by `trainer.ckpt.keep_*`. Caveat: `ckpt.output_dir` propagates to the trainer only (`utils/validation.py:89-91`), so when it is set the orchestrator's `step_N/orchestrator` lives under `<run_dir>/checkpoints` in a separate tree that nobody cleans (B, 03 §3.15).

**Atomicity / completeness: none.** No marker is written or checked anywhere in the trainer, pathing helpers or launcher (grep for `.metadata`/done/complete markers finds none). DCP's own `.metadata` (written last by the coordinator) would be a natural completeness marker but is never consulted; "latest" is purely the highest `step_*` dir name. By contrast the orchestrator's `progress.pt` *is* written atomically (tmp file + `os.replace`, `orchestrator/ckpt.py:30-45`).

**Resume semantics**:
1. Step resolution (`rl/train.py:137-143`): `resume.dir` (must be named `step_<N>`, mutually exclusive with `resume.step`, `configs/shared.py:54-78`) → N and path `resume.dir/"trainer"`; `resume.step`; or latest `step_*` directory name under the ckpt dir (`pathing.py:295-311`) — no completeness check.
2. Model built on `to_empty` without HF weights (§3.3 step 12).
3. `load_from_path` → `dcp_load({"app": AppState(...)})` then `AppState.load_state_dict` (`ckpt.py:215-251, 112-160`): non-offload path uses `set_state_dict(model, optimizers, …)` with packed state re-joined; CPU-offload path relies on DCP having written in place into the aliased CPU tensors and only applies the model side (explicit comment on why `set_state_dict` would allocate GPU copies, `:116-134`). `skip_optimizer` loads model+scheduler+progress only (`:229-235`).
4. `progress.step += 1` → training resumes at N+1 (`rl/train.py:217`), the data loader starts at step N+1, and the **startup broadcast publishes v{N}** (the checkpoint's weights) so inference (which is stateless) is re-synced before the first batch (`:267-276`); master first prunes broadcast dirs > N (`base.py:29-36`).
5. The RL trainer ignores `ckpt.skip_progress`, `skip_scheduler`, `skip_dataloader` (only `skip_optimizer` is read, `ckpt.py:168`); SFT honors all four (`sft/train.py:172-220`).
6. `CheckpointManager.ckpt_steps` is seeded from existing `step_*` dirs, filtered to `≤ resume.step` when given (`ckpt.py:173-177`), so the cleaning policy never touches future-step dirs from a longer crashed run. Nothing else deletes them either: A's `clean_future_steps` removes only `batches/` (> step) and `broadcasts/` (≥ step) (`pathing.py:379-397`). A later bare `--resume` (latest) therefore jumps back onto the **abandoned timeline's** higher-numbered checkpoint.
7. Resuming a run whose checkpoint step already equals `max_steps` runs one **extra** step: `progress.step = N+1 ≥ max_steps` makes it the last step, but the loop body (startup broadcast v{N}, `wait_for_batch` for step N+1, train, broadcast, final save of `step_{N+1}`) still executes once (`rl/train.py:258, 717-733`).

**Offline conversions** (`tools/`):
| Tool | Input → output | Mechanism |
|---|---|---|
| `convert_dcp_to_bf16.py` | `step_N[/trainer]` → `step_N/weights` (HF safetensors + config/generation-config/tokenizer/processor/remote-code `.py`) | reads the run's resolved `configs/latest/resolved/{trainer,sft}.json`, overrides `compile/ac=None, dp_replicate=cp=1, ep=auto, moe=default, attn=FA2`, builds the model with `loading_from_checkpoint_later=True`, `dcp_load(AppState(model, [], None, None))`, `gather_weights_parallel(bf16)`, drops tied keys, `convert_state_dict_to_hf`, `save_state_dict_parallel` (`:48-185`). Works under `python` (1 rank) or `torchrun`. Rejects LoRA ckpts. |
| `convert_dcp_to_fp8.py` | ckpt → `weights-FP8` | same gather, per-rank blockwise e4m3 quantization, writes `quantization_config` (`:28-61`) |
| `convert_bf16_to_fp8.py` | HF bf16 dir → `<dir>-FP8` | 2-D `.weight` tensors except norms/embeds/lm_head/routers/etc. → e4m3 + `<name>_scale_inv` (128×128) (`:30-87`) |
| `convert_fp8_to_bf16.py` | fp8 HF dir → `<dir>-BF16` | dequantize with `weight_scale_inv`; ue8m0 scales unsupported (`:24-113`) |

### 3.17 Weight export for inference sync (S7, trainer side)

Shared handshake (E owns the wire and receiver; summarized): `WeightSender.broadcast(model, step)` (`transports/weights/base.py:53-70`) — master resets `broadcasts/step_<N>/`, touches `.sender_ready`, **polls for `.receiver_ready`** (timeout `weight_broadcast.timeout`), touches `.started`; then *all ranks* run the transport's `_broadcast`; master touches `.finished` and deletes broadcast dirs older than N−1 (`:101-108`). The trainer therefore **cannot run ahead of its consumer by more than one version**. Transport choice: `weight_broadcast.type` (trainer default `filesystem`, `configs/trainer.py:691`; under `rl` the default becomes `nccl` unless LoRA or no inference, `configs/rl.py:447-451`).

#### 3.17.1 What leaves the trainer (all transports)

| Aspect | Rule | Cite |
|---|---|---|
| Source | `model.state_dict()` — FSDP DTensors (params) + persistent buffers (plain tensors); canonical FQNs (fused params split, LoRA `base_layer.` stripped, AC/compile wrappers stripped; `resolve_fqn` strips `_orig_mod`) | `utils/weights.py:115-125`; `fusions.py:55-67`; `lora/base.py:134-156` |
| Layout | prime layout converted to **HF layout**. Entry points differ per transport: **filesystem** → `convert_state_dict_to_hf` on the rank-partial dict, format decided once from the model's *full* key set (`model.convert_to_hf` if `is_prime_state_dict(full_keys)`, else transformers `revert_weight_conversion`); **NCCL** → `preprocess_layer_checkpoint` per group, decided per *group dict* (`convert_layer_to_hf` if `is_prime_state_dict(group)`, else `revert_weight_conversion`) — hence the single-layer/non-layer discriminability requirement; **NIXL** → none on the trainer (receiver replays `conversion_chain`, §3.17.4) | `utils/weights.py:94-112`; `nccl.py:103-114, 155-159` |
| dtype | DTensors cast to **bf16 before gathering** (halves traffic), except names where `keep_in_fp32_for_weight_transfer(name)` → fp32; non-DTensor buffers keep their own dtype | `utils/weights.py:154-159, 186-190`; `nccl.py:84-100` |
| fp32 keys | GLM4-MoE / GLM-MoE-DSA: `mlp.router.selection_bias`; NemotronH: `mamba.A_log`, `mamba.D`, `mlp.router.selection_bias`; Qwen3.5: `linear_attn.A_log`, `linear_attn.norm.weight`; DeepSeek V4: anything containing `attn_hc`, `ffn_hc`, `hc_head`, `sinks`, … (`KEEP_IN_FP32_MODULES`) | `glm4_moe/modeling_glm4_moe.py:137-138`; `nemotron_h/modeling_nemotron_h.py:142-143`; `qwen3_5/modeling_qwen3_5.py:134-135`; `deepseek_v4/modeling_deepseek_v4.py:79-83, 129-130`; `tests/unit/utils/test_weights.py:28-43` |
| Gather | `DTensor.full_tensor()` = all-gather across the param's mesh (FSDP dims, plus EP for experts) on **every rank** (collective) | `utils/weights.py:190`; `nccl.py:99` |

#### 3.17.2 Filesystem sender (`transports/weights/filesystem.py:20-58`)

- Full model: `dist.barrier()`; `gather_weights_parallel(model)` — every rank all-gathers every DTensor, but each keeps only the tensors of the **decoder layers it owns** (layer-granular, byte-balanced greedy assignment, `partition_weights`, so per-rank conversion sees complete layers) on CPU (`utils/weights.py:131-199`); `convert_state_dict_to_hf` on the partial dict (format detected from the *full* key set, `:94-112`); `save_state_dict_parallel` — each rank splits its slice into ≤5 GB shards under temp names, exchanges shard maps (`all_gather_object`), renames its own files to global `model-XXXXX-of-YYYYY.safetensors` (or `model.safetensors`), master writes `model.safetensors.index.json` after a barrier, **only when there is more than one shard** (`:202-252`, `:248`). No `config.json` is written into the broadcast dir, and none is needed: vLLM workers stream the safetensors into their already-built model (06).
- LoRA: all ranks `full_tensor()` the adapter DTensors, master saves `adapter_model.safetensors` (unsharded) + `adapter_config.json` into the step dir (`filesystem.py:35-52`).

#### 3.17.3 NCCL sender (`transports/weights/nccl.py:117-190`)

- Setup: master creates a vLLM `StatelessProcessGroup(host, port=29501 default, rank=0, world_size=inference_world_size+1)` and `PyNcclCommunicator` on its GPU (`:131-138`; `configs/trainer.py:645-649`); non-master ranks have no communicator. Constructed during startup, so it blocks until inference joins (E). Rank 0's store binds exactly `(host, port)` (vLLM `StatelessProcessGroup.create` binds a listen socket to `host` — read in vLLM 0.24; the lock pins 0.29, `uv.lock:5142-5143`), so the default `host="localhost"` is loopback-only; multi-node `rl` rewrites the trainer's host to `0.0.0.0` (`configs/rl.py:788-789`) while the orchestrator gets `$MASTER_ADDR` (`multi_node_rl.sbatch.j2:563`).
- `_broadcast`: barrier on all ranks first (non-master ranks' gather collectives must not start before the receiver paused inference, `:182-190`), then `send`: broadcast `num_layers+1`; iterate **non-layer group first, then layer 0..L-1** by prefix `get_layer_prefix` (`model.layers.` or `<language_model_attr>.layers.` for VLMs) (`:67-81, 143-159`); per group: `resolve_dtensors` (all ranks), `preprocess_layer_checkpoint` → HF names (all ranks), and master `broadcast_state_dict`: pickled metadata `{dtype: [(key, shape, numel)]}` (size then bytes) followed by **one concatenated flat tensor per dtype** (`:30-64`).
- Peak memory: one full layer (bf16) on every rank's GPU at a time plus conversion temporaries; `rl/train.py:587-595` synchronizes and empties the cache before broadcasting for headroom.

#### 3.17.4 NIXL sender (`transports/weights/nixl/nixl.py:72-447`) — receiver-driven one-sided pulls

- **No HF conversion on the trainer**: tensors are published in *trainer (prime) naming*; the receiver builds `LazyWeight` roots per trainer tensor, replays `model_cls.conversion_chain(hf_config)` backward to HF names *as view ops*, and traces vLLM's `load_weights` into a replay plan (`nixl/graph.py:312-346`, E's). Hence NIXL requires a registered custom model whose prime→HF chain only uses supported view ops.
- Serving ranks: `dp_replicate` local-rank-0 replicas (`:96-100`). `initialize_transfer` (once, `:326-359`): transfer groups `non_layer` + `layer.<i>` (`:102-119`); each rank lists its **local FSDP shard** of every floating tensor (`to_local()` + global offset from `compute_local_shape_and_global_offset`, assuming the shard is a contiguous row range along dim 0), replicated/non-DTensor values served only by master; wire dtype bf16 or fp32 (`:121-188`). Allocates 1 (or 2 with `overlap_transfer_and_replay`) **GPU staging arenas per wire dtype** sized to the largest group, registers them with NIXL, builds a `TrainerTensorTable` (msgpack: agents, `staging_buffer_count`, groups → tensors → shards `{agent, offset, numel, addr}`) per rank, gathers fragments to master, merges, and publishes via ModelExpress (`:190-359`; `trainer_tensor_table.py`).
- `_broadcast` per version (`:372-447`): at the first broadcast, master waits for orchestrator + `inference_world_size` peers in ModelExpress and all serving ranks connect (`:377-408`); master sends `policy_notification(step,"ready")` to the orchestrator; for each group: wait for the credit of the group that last used this staging buffer, `copy_to_staging` (cast to wire dtype on GPU), `synchronize`, notify every inference peer; drain the tail; master waits `policy_notification(step,"complete")`; barrier.

#### 3.17.5 HF↔prime conversion ops (`models/conversion_ops.py`)

A family's `conversion_chain(config)` is a flat list of `ConvOp`s with fully templated keys; `apply_hf_to_prime` plays them forward, `apply_prime_to_hf` plays each op's backward in reverse (`:47-66`). Ops: `Rename`, `PrefixRename`, `Drop` (symmetric), `Stack` (stack `{e}`-indexed per-element keys along a new dim; backward `select`s with an `index_offset` so a rank could unstack only local experts), `SplitConcat` (split along an existing dim; backward `torch.cat`), `MapValue` (value transform with explicit backward), `SqueezeLeading` (backward-only), `Conditional` (predicate evaluated per direction on the current dict), `Sequence` (`:74-290`). Every op is **present-guarded** (no-op if its inputs are absent), so the same full chain works on a single layer's dict (which is what NCCL does) and on a rank-partial dict (filesystem). Helper `routed_experts_op` builds Stack×3 (per-expert `[H,D]` ↔ stacked `[E,H,D]`) and, with `fused=True`, also accepts the transformers-v5 fused `gate_up_proj` on the way in (`:302-348`). Backward always emits per-expert HF keys.

### 3.18 Model families

| Family (`model_type`) | Kind | Conversion specifics (prime ← HF) | CP styles | Other |
|---|---|---|---|---|
| `llama` | dense | identity (`is_prime_state_dict=False` disables auto-convert) | both | HF `LlamaConfig`; no configuration file (`llama/__init__.py`) |
| `qwen3` | dense | identity | both | exemplar; `qwen3/modeling_qwen3.py` |
| `qwen3_moe` | MoE | `mlp.gate.weight`→`mlp.router.gate.weight`; per-expert/fused experts → stacked `mlp.experts.{gate,up,down}_proj` (`converting_qwen3_moe.py:8-14`) | both | softmax router, `norm_topk_prob`, `score_before_experts=False`, `mlp_only_layers`/`decoder_sparse_step`, no selection bias by default (`load_balance_coeff=None`) |
| `glm4_moe` | MoE | router gate + `e_score_correction_bias`→`router.selection_bias`; stacked experts (fused accepted); `shared_experts.*`→`shared_expert.*` with backward `SqueezeLeading`; drop `tokens_per_expert` (`converting_glm4_moe.py:36-66`) | both | selection bias fp32 on wire; `first_k_dense_replace` |
| `glm_moe_dsa` (GLM-5/5.2) | MoE + MLA + DSA sparse indexer | same MoE ops as GLM4; attention/indexer names unchanged (`converting_glm_moe_dsa.py`) | own CP via `cp_context` (all-gather compressed KV) | indexers frozen; IndexShare via `indexer_types`/`index_cache`; `head_dim` kwarg dropped (`configuration_glm_moe_dsa.py:218-221`) |
| `deepseek_v4` | MoE + mHC + compressed/sparse attention | on-disk names without `model.` prefix (`attn`, `ffn`, `wq_a`, `w1/w2/w3`, `hc_*`); drops `mtp.*`; `head.weight`→`lm_head.weight`; indexer moved under the CSA compressor (`converting_deepseek_v4.py:28-109`); `convert_to_prime` dequantizes FP8/MXFP4 first | ring only | `transformers_compat.allow_deepseek_v4_layer_types` shim; fp32 wire for hyper-connection params; hash-routed first layers |
| `gpt_oss` | MoE | `router.{weight,bias}`→`router.gate.*`; custom `GptOssExperts` op de-interleaves `gate_up_proj [..., 2I]` (even=gate, odd=up) with transposes; backward rebuilds interleaved tensors via `new_empty` + strided writes (`converting_gpt_oss.py:8-64`) | both (own ring/ulysses substitutes) | FA4 with sinks; BF16 checkpoints only (docs) |
| `laguna` (Poolside) | MoE | router gate; bias from `experts.` or `gate.` (conditional); per-expert/fused experts; `shared_expert`/`shared_experts` (conditional) (`converting_laguna.py:32-77`) | both | per-layer head counts, nested rope params by layer type (`configuration_laguna.py:124-156`) |
| `minimax_m2` | MoE | `block_sparse_moe.*` ↔ `mlp.*`, `w1/w2/w3` ↔ gate/down/up (`converting_minimax_m2.py:18-33`) | both | `rotary_dim`→`partial_rotary_factor` |
| `nemotron_h` | hybrid Mamba/attention/MoE | `backbone.`↔`model.`, drop `mtp.`, `mixer.`→`mamba.`/`self_attn.`, MoE non-gated experts stacked, latent projections (`converting_nemotron_h.py:17-52`) | ulysses only | fp32 wire for `A_log`, `D`; overrides `convert_adapter_to_hf` (`modeling_nemotron_h.py:158-165`); `num_hidden_layers` derived from pattern |
| `qwen3_5`, `qwen3_5_moe` (+ `_text`) | hybrid Gated-DeltaNet/attention, dense or MoE, VLM-capable | drop `mtp.`; MoE: router gate, `shared_expert_gate`→`shared_expert.output_gate`, fused `gate_up_proj` split on dim 1 (backward re-fuses with `torch.cat`); prefix `model.language_model` for VLM (`converting_qwen3_5.py:19-45`) | ulysses when linear-attention layers exist | only registered custom VLM (`models/__init__.py:106-109`); `supports_packed_multimodal_training` (`modeling_qwen3_5.py:322`) |
| `afmoe` (Trinity) | MoE | `expert_bias`→`router.selection_bias`, shared experts, per-expert Stack; drop `tokens_per_expert`, `reorderer.*` (`converting_afmoe.py:21-32`) | both (patched attention class) | `layer_type_validation`, sliding/global schedule |

Registration: configs are registered with `AutoConfig.register` (`models/__init__.py:35-48`) and `(ConfigClass, ModelClass)` pairs in `_CUSTOM_CAUSAL_LM_MODELS` (`:51-76`). Because `supports_custom_impl` checks the *config type* (`:100`), a family is only auto-selected if `AutoConfig` returns prime-rl's registered config class (or the HF class listed, for `llama`/`qwen3`).

### 3.19 SFT trainer: what is shared

`sft/train.py` reuses the entire engine — `setup_torch_distributed`, `resolve_ep`/`get_parallel_dims` (with `seq_len` validation), `setup_model`, `setup_context_parallel`, `setup_optimizer` (incl. full offload), `setup_scheduler`, `CheckpointManager` (**with** a `StatefulDataLoader` and all `skip_*` flags), `forward` + fused LM head (as CE via negative target logprob, `sft/train.py:283-299`), MoE stats, `GarbageCollection`, and `WeightSender` (only for online-eval steps, plus startup and final broadcasts, `:386-411, 550-555, 688-691`). Separate: data (`sft/data.py`: renderer-based `SFTDataset`, `CatDataset` packing, `StatefulDataLoader`), loss (token-sum CE normalized by global token count), gradient accumulation from `data.batch_size·cp / (world_size·micro_batch_size)` (`:116-122`), validation loop with all-rank "has data" consensus (`:321-352`).

## 4. Interfaces & contracts

### 4.1 Files and directories

| Path | Writer | Reader | When |
|---|---|---|---|
| `<ckpt_dir>/step_N/trainer/{.metadata,__*_0.distcp}` | all trainer ranks (DCP) | trainer on resume; `tools/convert_dcp_to_*` | every `ckpt.interval`, final |
| `<ckpt_dir>/step_N/trainer/dataloader/rank_<r>.pt` | SFT only | SFT resume (falls back to rank 0's) | `ckpt.py:199-208, 238-249` |
| `<output_dir>/broadcasts/step_N/{.sender_ready,.receiver_ready,.started,.finished}` | trainer master / consumer (`.receiver_ready`) | consumer / trainer master | every step + startup (`base.py:23-26`) |
| `<output_dir>/broadcasts/step_N/model*.safetensors`, `model.safetensors.index.json` (index only if >1 shard) | all ranks (filesystem transport) | inference (E) | every step |
| `<output_dir>/broadcasts/step_N/{adapter_model.safetensors,adapter_config.json}` | master (LoRA) | inference `load_lora_adapter` (E) | every step |
| `<snapshot or conversion_dir>/prime/` + `.prime-v1` (or `/hf/`) | master, once | all ranks at load | first load of an HF-format snapshot into a prime-format model (`model.py:697-724`) |
| `<run_dir>/configs/latest/resolved/trainer.json` | launcher (A) | `tools/convert_dcp_to_bf16.py:72-90` | offline export |
| `memory_profiler_path/step_N/rank_<r>.pickle`, `trace_path/trace_<rank>.json.gz` | all ranks | humans | optional (`utils.py:325-344`; `rl/train.py:721-727`) |

### 4.2 Network endpoints / env

| Item | Value | Cite |
|---|---|---|
| torchrun rendezvous | `localhost:<free port>` single-node | `entrypoints/rl.py:333` |
| NCCL weight-broadcast rendezvous | `weight_broadcast.host:port` (default `localhost:29501`; multi-node `rl` sets trainer host `0.0.0.0`), world `inference_world_size+1`, trainer = rank 0 | `configs/trainer.py:633-649`; `nccl.py:172-179`; `configs/rl.py:788-789` |
| NIXL ModelExpress | `host:port` (default 8001), `session_id="default"` | `configs/trainer.py:652-662` |
| Metrics server | `metrics_server.port` (default 8000), `host 0.0.0.0` | `configs/shared.py:226-231` |
| Env read | `RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE` | `world.py:8-11` |
| Env set | `USE_HUB_KERNELS=NO` (setdefault) | `model.py:9` |

### 4.3 Config fields (`packages/prime-rl-configs/src/prime_rl/configs/trainer.py`)

**`ModelConfig`** (extends `BaseModelConfig`: `name="Qwen/Qwen3-0.6B"`, `trust_remote_code=False`, `vlm: VLMConfig|None`, `configs/shared.py:154-162`):

| Field | Type / default | Effect |
|---|---|---|
| `conversion_dir` | `Path|None`=None | where HF→prime converted weights are cached (`prime/` subdir) |
| `seq_len` | 2048 | used by SFT/perf; RL batches come pre-packed |
| `attn` | `flash_attention_{2,3,4}|auto`="auto" | §3.3 step 1 |
| `compile` | `CompileConfig|None`=`CompileConfig()` (`fullgraph=False`, `mode=None`) | per-block `torch.compile` |
| `fusions` | `enabled=["gate_up","qkv"]`, `raise_on_fail=False`, `shard_fused_on_dim1=False` | §3.14 |
| `ac` | `ActivationCheckpointConfig|None`=on (`mode="full"`, `freq=1`, `targets=None`) | §3.9 |
| `ac_offloading` | `ActivationOffloadingConfig|None`=on (`pin_memory=True`, `max_inflight_activations=5`) | §3.9; forces AC on |
| `fsdp_cpu_offload` | False | §3.8a |
| `optim_cpu_offload` | **True** | §3.8b; also disables fused AdamW |
| `full_offload` | `OptimizerInBackwardOffloadConfig|None`=None (`true` → `{}`; `cpu_optimizer_backend="native"|"torch"`, `numa_bind=True`) | §3.8c |
| `reshard_after_forward` | True | FSDP |
| `dp_replicate` | 1 | HSDP replicas |
| `ep` | `int|"auto"`="auto" | §3.2/§3.10 |
| `moe` | `compute=BF16MoEComputeConfig()` (`bf16|deepgemm_fp8|mxfp8`, `apply_to="all"`, MXFP8 `recipe`), `dispatch=TorchMoEDispatchConfig()` (`torch` `transport=bf16|mxfp8`; `deepep` `num_sms=20`, `token_chunk_size=None`) | §3.10 |
| `cp` | 1 | CP degree |
| `cp_style` | `"ring"|"ulysses"`="ring" | §3.11 |
| `impl` | `"hf"|"custom"|"auto"`="auto" | §3.3 |
| `optimization_dtype` | `"float32"` | sharded master param dtype |
| `reduce_dtype` | `"float32"` | FSDP gradient reduction dtype |
| `moe_router_dtype` | `"float32"` | fp32 router units/gates |
| `quantization` | `FP8Config|MXFP8Config|None`=None | §3.13 |
| `index_cache` | `IndexCacheConfig|None` (`topk_freq=1`, `topk_pattern=None`) | DSA index reuse |
| `freeze_moe_router` | False | |
| `lora` | `LoRAConfig|None` (`rank=16`, `alpha=32.0`, `dropout=0.0`, `target_modules=[q,k,v,o,gate,up,down proj, experts, fc1/fc2_latent_proj]`, `modules_to_save=[]`) | §3.15 |
| `debug` | `num_layers=None`, `random_init=False`, `force_balanced_routing=False` | |
| `fused_lm_head_token_chunk_size` | `int|"disabled"`=**8192** | §3.12 |

Validators: trust_remote_code only with hf/auto; VLM ⇒ `impl="custom"`; VLM+CP ⇒ ulysses; CP ⇒ flash attn + custom/auto; offload exclusivity; FA4/auto ⇒ custom/auto; quantization ⇒ custom/auto; MoE runtime compat (`:350-425`).

**`TrainerConfig`** (`:671-828`): `model`, `tokenizer` (`name` defaults to model, `chat_template`), `data` (`fake: FakeDataLoaderConfig(batch_size=2, generate_samples=False)|None`), `loss` (`ipo` default | `icepop` | `custom`; C), `optim` (`adamw` default; all: `lr=1e-6`, `weight_decay=0.01`, `max_norm=1.0`; adamw `betas1=0.9, betas2=0.999`; muon `mu=0.95, betas2=0.95`; sgd `momentum=0.9, nesterov=True`), `scheduler` (`constant` default; `linear`/`cosine`: `warmup_steps=10`, `decay_steps=10`, `min_lr=0.0`), `ckpt: CheckpointConfig|None=None` (`output_dir`, `interval=None`, `keep_last`, `keep_interval`, `skip_{progress,scheduler,dataloader,optimizer}=False`), `resume: ResumeConfig|None` (`step`, `dir`), `weight_broadcast` (`filesystem` default | `nccl` | `nixl`; all `timeout=1200`), `rollout_transport` (ZMQ default; E), `log` (`ranks_filter=[0]`), `monitors`, `output_dir` (`$PRL_OUTPUT_DIR` or `outputs`), `matmul_precision="high"`, `max_steps=None`, `enable_router_replay=False`, `memory_profiler_path`, `gc=GCConfig(interval=50)`, `trace_path` (requires `max_steps<10`), `dist_timeout_seconds=3600`, `heartbeat`, `metrics_server`, `env_vars`. Cross-field validators: DeepEP/full-offload ⇒ `max_norm=None`; full offload ⇒ adamw/sign_sgd; muon ✗ fsdp_cpu_offload; LoRA ⇒ filesystem broadcast and no `modules_to_save`; EP/router replay ⇒ custom/auto (`:735-828`).

## 5. Invariants & assumptions

1. **Every rank executes the same collective sequence.** Every micro-batch forward is a collective (FSDP all-gathers, EP all-to-all, CP all-gathers); every DP rank must receive the same number of micro-batches per step (packing is upstream, C/B). Per-rank-variable control flow is avoided by design: the loss-denominator all-reduce is one batched op for all loss components (`rl/train.py:300-321`), stats use `all_gather_object` for the key union (`utils.py:250-253`), SFT validation uses a MIN all-reduce to stop together (`sft/train.py:327-339`).
2. **Mesh order**: `cp` is the innermost (fastest) mesh dim, then `dp_shard(_in_ep)`, then `dp_replicate`. `DataLoader` computes `dp_rank = rank // (world/dp)` (`rl/data.py:190-191`) — correct only under this layout. EP groups are contiguous ranks.
3. **EP ⊆ dp_shard × cp**, `ep % cp == 0`, `num_experts % ep == 0` (`parallel_dims.py:73-75`; `moe_runtime.py:87-90`). `pp == 1` always.
4. **Canonical names are the checkpoint/export ABI.** Fusions, LoRA wrappers, AC/compile wrappers must all be transparent in `state_dict()`; packed tensors are split by hooks. Anything that changes an FQN breaks DCP resume and HF loading.
5. **Optimizer holds exactly the `requires_grad` params, and that set must be identical at save and load** (indexers and LoRA-frozen params frozen *before* optimizer creation; EP's re-created params re-frozen) (`optim/__init__.py:119-123`; `model.py:192-214, 1031-1035`).
6. **HF snapshot keys must match the model's keys after conversion** — `dcp_load` with `HuggingFaceStorageReader` needs a 1:1 key/shape match (tied `lm_head` dropped); the converted `prime/` cache is trusted if the `.prime-v1` marker exists (no content validation).
7. **Buffers are not sharded**; `init_buffers_post_meta` must rebuild every non-persistent buffer, and must run before `dcp_load` so persistent ones are overwritten by the checkpoint (`model.py:662-666`).
8. **Conversion ops are present-guarded and layer-local** so they work on one layer (NCCL), a byte-balanced set of whole layers (filesystem), or the full dict. A chain op that needs keys from two layers, or from a layer and the non-layer group, would silently no-op on the NCCL/filesystem paths.
9. **Weight export always converts to HF naming/layout** (except NIXL, where the receiver does it) and exports bf16 except declared fp32 keys; vLLM must be able to `load_weights` those names.
10. **Loop ↔ consumer lockstep**: `broadcast(v{step})` blocks until acknowledged; `progress.step` after resume = checkpoint step + 1; the startup broadcast re-publishes the checkpoint version.
11. **seq_lens sum == row length** (or pre-shard length under CP) — validated with host syncs in every custom forward (`utils/sequence.py:43-46`); CP further requires the row to be divisible by `cp` (`utils/cp.py:86-91`).
12. **GPU memory headroom for broadcast**: the gather peaks at ≈ one full layer (NCCL) or one full tensor (filesystem) per rank in bf16; the loop synchronizes + empties the cache first.

## 6. Extension points

### 6.1 Add a new model family (custom impl) — end to end

1. **Package** `src/prime_rl/trainer/models/<arch>/` with `configuration_<arch>.py` (a `PretrainedConfig` subclass with `model_type`, only needed if transformers' own config is unsuitable or missing), `converting_<arch>.py` (`conversion_chain(config) -> list[ConvOp]`, plus `is_hf_state_dict`/`is_prime_state_dict` helpers), `modeling_<arch>.py`, and `__init__.py` exporting the config + `*ForCausalLM` (mirror `qwen3_moe/` or `glm4_moe/`; `docs/development.md:69-71`).
2. **Modeling contract**:
   - `*PreTrainedModel(PreTrainedModelPrimeRL)`: implement `is_hf_state_dict`, `is_prime_state_dict` (must be *discriminative on a single layer's keys and on the non-layer group*, see the DeepSeek V4 comment `modeling_deepseek_v4.py:139-146`), `conversion_chain`, `init_buffers_post_meta`, and where relevant `cp_support`, `keep_in_fp32_for_weight_transfer`, `convert_adapter_to_hf`. Dense models with HF-identical names return `is_prime_state_dict=False` to skip auto-conversion (`qwen3/modeling_qwen3.py:102-111`).
   - `*Model.forward(input_ids, position_ids, inputs_embeds, [routed_experts], *, seq_lens, seq_lens_are_pre_shard)` → `BaseModelOutputWithPast(last_hidden_state=…)`; build `cu_seqlens` via `get_cu_seqlens_from_seq_lens(seq_lens, total_tokens=None if pre_shard else S)` and pass `cu_seqlens/max_seqlen` to attention (`qwen3/modeling_qwen3.py:153-194`). Slice `routed_experts[:, :, layer_idx, :]` per layer for MoE (`qwen3_moe/modeling_qwen3_moe.py:212-220`).
   - Layout requirements for the engine to find things: `model.model.layers` (ModuleList of decoder blocks; AC/compile/FSDP iterate it), `model.model.embed_tokens` (or `.embeddings`), `model.model.norm` (or `.norm_f`), `model.lm_head` bias-free `nn.Linear`, MoE blocks under `layer.mlp` as `layers.moe.MoE` (FSDP/EP/stats/freeze helpers look at `transformer_block.mlp`, `model.py:127-128, 538-540`).
   - Reuse building blocks: `ATTN_IMPL2CLASS[config._attn_implementation]` (gets CP patching + `qkv` fusion for free), `MoE.from_args(MoEArgs(...))` (gets dispatchers, EP, grouped GEMM backends, `gate_up` fusion, router replay), `FeedForward`, `RMSNorm`, `RotaryEmbedding`.
3. **Register** in `trainer/models/__init__.py`: `AutoConfig.register("<type>", Config, exist_ok=True)` (`:35-48`), add `(Config, ForCausalLM)` to `_CUSTOM_CAUSAL_LM_MODELS` (`:51-70`); for a VLM also `_CUSTOM_VLM_MAPPING` (`:106-109`) and `VLM_REGISTRY` in `utils/vlm.py:27-33`.
4. **Inference parity (lockstep)**: vLLM must serve the same architecture and accept the exported HF names; if NIXL is to be supported, the prime→HF chain must use only `LazyWeight`-supported view ops (`nixl/graph.py:35-54`); fp32-sensitive tensors must be declared in `keep_in_fp32_for_weight_transfer`; FP8 ignore patterns must match between trainer and inference (`configs/trainer.py:155-166`).
5. **Mini preset + smoke test**: add an `ARCH_PRESETS` entry in `scripts/mini_moe.py` (`config_class`, small `config_kwargs`, HF class or `None` for remote code, prime class, `tokenizer_source`) (`scripts/mini_moe.py:78-195`); `uv run python scripts/mini_moe.py --arch <arch> --output-dir ./mini-<arch>` creates a random ~0.5B model, then `verify` checks HF-vs-prime logits (`max diff < 0.1`) and an exact HF→prime→HF weight roundtrip (`:252-310`); `--verify-only` re-checks. Then SFT warm-up and `uv run rl @ configs/ci/integration/reverse-text-moe/start.toml --model.name <mini> --trainer.model.impl custom` (`docs/development.md:90-131`). ⚠ At the pin the script itself fails to import (§7).
6. **Tests to add** (pattern): HF-vs-prime forward/backward parity with the prime LM head injected (`tests/unit/train/models/test_qwen3_moe.py:22-130`), conversion roundtrip (`test_moe_conversions.py:60-93`), fp32 wire keys (`tests/unit/utils/test_weights.py:28-43`).
7. **Merge bar**: KL-mismatch table over 20 steps on `math`, batch 64, all < 0.015 (`docs/development.md:133-140`).

### 6.2 Other seams

| Seam | Recipe |
|---|---|
| New optimizer | add a `*Config(BaseOptimizerConfig)` with a `type` literal to the `OptimizerConfig` union (`configs/trainer.py:541-543`), a `case` in `_create_optimizer` (`optim/__init__.py:124-148`); full offload additionally needs a native/torch CPU step (`offload.py:1200-1280`) and the validator list (`configs/trainer.py:746-750`) |
| New LR schedule | config class + union (`configs/trainer.py:439-468`), `case` in `setup_scheduler` (`scheduler.py:100-121`), `validate_scheduler` |
| New MoE compute backend | implement the `GroupedGemm` protocol (`token_group_alignment`, `__call__(x, weight_t, *, offs)`), config class in `MoEComputeConfig`, branch in `_resolve_grouped_gemm` (`moe_runtime.py:34-56`); add its ops to AC save targets if expensive (`activation_checkpointing.py:49-73`) |
| New token dispatcher | subclass `TokenDispatcherBase` (`dispatch`/`combine`/optional `run`/`synchronize`, `token_dispatcher.py:37-74`), config in `MoEDispatchConfig`, branch in `configure_moe_runtime` (`moe_runtime.py:92-124`); non-replayable ops must be in `MANDATORY_SAVE_*` |
| New runtime fusion | a function `(module) -> None` that builds the packed `nn.Parameter` and calls `register_packed_parameter_state_dict_hooks(module, PackedParameterSpec(...))`; advertise it in the module's `supported_fusions`; add the name to `FusionsConfig.enabled`'s literal (`fusions.py:47-118`; `configs/trainer.py:93`) |
| New CP style | a `substitute_*` that rebinds `FlashAttention._compute_attention` (+ AFMoE/GPT-OSS classes), a params publisher called from `setup_cp_attention_params` (`utils/cp.py:142-164`), add to `CPStyle`/`ALL_CP_STYLES` (`models/base.py:9-10`) and the config literal |
| New weight transport (sender side) | subclass `WeightSender` implementing `_broadcast(model, step, step_dir)` (rank sync is the transport's job), add a config to `WeightBroadcastConfig`, dispatch in `setup_weight_sender` (`transports/weights/__init__.py:22-35`) and the `rl` auto-setup (`configs/rl.py:440-493`); reuse `gather_weights_parallel`/`resolve_dtensors` + `convert_*_to_hf` + `resolve_wire_dtype`; the receiver side is E's |
| Extra checkpoint state | extend `AppState.state_dict/load_state_dict` (`ckpt.py:79-160`) — DCP handles DTensors; plain Python objects are pickled into the metadata |
| Custom loss | C (`loss.type="custom"`, `import_path`) |

## 7. Gotchas & limitations

**Bugs / broken paths at the pin (static evidence + CPU probes; no GPU run):**
- **LoRA on MoE experts under EP → `ImportError` at the first MoE forward**: all three LoRA expert wrappers, when expert weights are DTensors (EP), take a branch that imports `TOKEN_GROUP_ALIGN_SIZE_M` from `prime_rl.trainer.distributed.expert_parallel` (`multi_moe.py:316-331, 542-557, 788-803`). That module defines only `ExpertWeightParallel` (`expert_parallel.py:1-14`; no star-import, no module `__getattr__`, `distributed/__init__.py` re-exports only that class, and the name exists nowhere else in the repo); the import fails in the venv (`ImportError: cannot import name 'TOKEN_GROUP_ALIGN_SIZE_M'`). The branch is reached whenever EP is on: `configure_moe_runtime` EP-shards the wrapper's base and LoRA params (`moe_runtime.py:127-128`) and FSDP2 keeps them as EP DTensors in forward. It would also be wrong if it imported: it re-permutes tokens the dispatcher already permuted (`token_dispatcher.py:278-286`), and it keys on a non-existent `ep_comm_backend` attribute (default `"torch"`, so DeepEP takes it too). `ep="auto"` resolves to up to 8 for any MoE model on >1 GPU, and the shipped `examples/extra/vlm/sft-moe.toml` (LoRA incl. `experts`, `ep = 8`) hits it. Dense LoRA and LoRA+MoE on one GPU (`ep=1` → plain tensors) are unaffected; no CI config exercises LoRA+MoE (the LoRA CI uses dense `Qwen3-0.6B`).
- **LoRA experts ignore the configured grouped-GEMM backend**: `configure_moe_runtime` calls `moe.experts.set_grouped_gemm(...)`, which `MultiLoRAModule.__getattr__` forwards to the base layer, but the LoRA expert forward calls `torch._grouped_mm` directly for base *and* adapter GEMMs (`multi_moe.py:32-33, 342-355`; `lora/base.py:123-128`) — `deepgemm_fp8`/`mxfp8` expert compute is silently BF16 with LoRA (dense LoRA is fine: `MultiLoRALinear` calls `self.base_layer(x)`, `multi_linear.py:146`, so FP8/MXFP8 dense linears are honored).
- **`scripts/mini_moe.py` does not import**: it imports `prime_rl.trainer.models.qwen3_5_moe` (`scripts/mini_moe.py:31`); that package was folded into `qwen3_5/` (`models/__init__.py:46-47, 108` register `qwen3_5_moe` against `Qwen3_5ForCausalLM`) and no alias module exists. The documented new-model smoke workflow fails until that line is fixed.
- **MoE `selection_bias` is never updated** (aux-loss-free balancing is described but not implemented; `moe.py:381-384`; details in §3.10). It is loaded, used for top-k selection, checkpointed, and exported unchanged.
- **NIXL + default `qkv` fusion serves stale q/k/v** (verified statically + DTensor probe): `qkv` packs on dim 0 = FSDP `Shard(0)` (unless `shard_fused_on_dim1`), so `state_dict()` yields `Replicate` **copies** of `q/k/v_proj` (§3.14; probe: an in-place update of the packed shard is not visible in the split on a 2-rank mesh). NIXL calls `state_dict()` once, in `initialize_transfer` at the first (startup) broadcast, keeps those copies as `source_tensor` (served by master as "replicated", `nixl.py:159-171, 326-338`) and re-copies from them every version (`:423-424`) → inference gets the **v{start−1}** q/k/v (and q/k/v biases) forever while every other weight updates. Unaffected: filesystem/NCCL (re-call `state_dict()` per broadcast), FSDP shard mesh of size 1, `fusions.enabled=[]`, LoRA (fusions skipped). `shard_fused_on_dim1` avoids the staleness but NIXL then fails at plan build (`no trainer shard owns element`), because it assumes dim-0 shards (06). No validator couples NIXL with fusions, and no CI/example config uses NIXL. (Probe ran torch 2.11; pin is torch ≥2.13.)
- **NIXL + non-view conversion ops**: Qwen3.5-MoE backward uses `torch.cat` (`SplitConcat`, `conversion_ops.py:208-212`), GPT-OSS uses `new_empty` + strided writes (`converting_gpt_oss.py:25-42`); `LazyWeight` only supports view ops (`nixl/graph.py:35-54, 301-309`). NIXL also assumes FSDP shards are contiguous dim-0 row ranges (`nixl.py:174-187`), which `shard_fused_on_dim1`'s `Shard(1)` breaks (confirmed by E: plan build raises `no trainer shard owns element`). Ops outside the supported view set raise at trace time; whether the Qwen3.5-MoE / GPT-OSS vLLM loaders hit them could not be checked without vLLM 0.29 source.
- **NCCL transport CI status**: the MoE integration config pins filesystem because "NCCL weight broadcast currently crashes the trainer on CI GPU runners" (`configs/ci/integration/reverse-text-moe/start.toml`), and the NCCL unit test is skipped (`tests/unit/transports/test_nccl_broadcast.py:12`), yet NCCL is the `rl` default.

**Sharp edges:**
- Default AdamW is **not fused** (because `optim_cpu_offload=True` by default, `optim/__init__.py:86`), and state offload adds a full H2D + D2H + `cuda.synchronize()` of optimizer state every step.
- `maybe_clean` deletes the entire `step_N` dir, including the orchestrator's checkpoint (when co-located, i.e. `ckpt.output_dir` unset) and any `weights/` export (`ckpt.py:315-319`); it is the only retention mechanism in the system.
- No checkpoint completeness marker: "latest" = highest `step_*` dir name (`pathing.py:295-311`), resolved independently by trainer (`rl/train.py:143`), orchestrator (`orchestrator.py:246`) and launcher (`entrypoints/rl.py:679`). A crash mid-save, or a `step_N` holding only the orchestrator's `progress.pt` (the orchestrator runs ahead of the trainer and saves into the same `step_N`), makes `--resume` fail (`FileNotFoundError`, `ckpt.py:266-267`, or a DCP metadata error) instead of falling back to the previous step.
- Stale future checkpoints survive a `--resume.step K` rewind (neither `maybe_clean` nor `clean_future_steps` deletes `checkpoints/step_{>K}`), so a later plain `--resume` silently resumes the abandoned timeline (§3.16 item 6).
- Resuming a run that already reached `max_steps` trains one extra step and blocks on a batch for step `max_steps+1` (§3.16 item 7).
- Synchronous checkpointing and synchronous broadcast both sit on the critical path (§3.5).
- The RL trainer silently ignores `ckpt.skip_progress/skip_scheduler/skip_dataloader` (`ckpt.py:168`).
- CP shards are contiguous chunks (no zig-zag load balancing, `utils/cp.py:93`) → causal attention work is skewed toward later CP ranks; the `2·cp` seq-length divisor is not checked in RL.
- `ep="auto"` = `min(world/dp_replicate, 8)` with no divisibility search (`parallel_dims.py:310-313`). Two failure modes (probed on `ParallelDims`): world 12 → `ep=8`, `12 % 8 ≠ 0` → bare `AssertionError` (no message) at `parallel_dims.py:75` for *any* MoE model, including ones that fall back to the HF impl and never use EP; world 6 → `ep=6` builds, then an 8-expert model fails `num_experts % ep` in `configure_moe_runtime` (`moe_runtime.py:87-90`). (`docs/scaling.md:97` describes a divisor search that is not in the code.)
- HF→prime auto-conversion: master loads the **entire** checkpoint into host RAM and writes a full converted copy next to the HF snapshot (needs write access to the HF cache or `conversion_dir`); an interrupted conversion leaves a `prime/` dir without `.prime-v1` that must be deleted by hand.
- Non-whitelisted HF models (unknown buffers) are loaded fully on CPU by every rank (`model.py:996-998`).
- Host syncs inside forward: `seq_lens` validation (`sequence.py:43-52`), `MultiLoRALinear`'s `assert offsets[-1] == x.shape[0]` (`multi_linear.py:144`), LoRA-MoE `argmax().item()` (`multi_moe.py:286, 295`), torch EP splits `.cpu()` (`token_dispatcher.py:276`).
- Activation offloading runs copies on the compute stream (streams disabled, `act_offloading.py:62-64`); `max_inflight_activations` has no effect.
- `inject_prime_lm_head` replaces `model.forward` — code that relies on the family's `*ForCausalLM.forward` (e.g. extra kwargs) is bypassed in training.
- Metrics all-gathers run over the whole world (CP/EP replicas duplicate token-level stats) and each `monitors.log` does its own `asyncio.run` (`rl/train.py:662-697`).
- Muon's NCCL warm-up is a correctness workaround (multi-node deadlock), not optional (`optim/__init__.py:23-29`).
- The `train.py` comment attributing ~50 GiB broadcast peak to "per-layer gather + fp8 conversion" (`rl/train.py:587-593`) has no trainer-side FP8 conversion behind it (grep shows `quantize_to_fp8_blockwise` only in `tools/`); presumably refers to inference-side conversion `[UNVERIFIED]`.

**Docs vs code drift (code wins):** fused LM head default is 8192 not 1024 (`configs/trainer.py:347` vs `docs/scaling.md:152-157`); SFT *does* use the fused head (`sft/train.py:283-299`) although both `docs/scaling.md:160` and the field's own docstring (`configs/trainer.py:348`) say SFT silently disables it; fused head is injected for HF impl too (`model.py:1010`), contrary to "Only available with `model.impl = "custom"`" (`docs/scaling.md:160`); NemotronH supports ulysses CP (`modeling_nemotron_h.py:135-140` vs the ❌ in `docs/advanced.md:34`); `docs/development.md:137` names `convert_hf_layer_to_tt`/`convert_tt_layer_to_hf`, which no longer exist; the "VLM bfloat16 mandatory" validator exists in `configs/sft.py:548-550` but not in `TrainerConfig`.

## 8. For a custom framework

**Essential design worth keeping:**
- **One canonical, HF-compatible naming/layout as the checkpoint and export ABI, with an explicit invertible conversion layer** (`conversion_ops`). The present-guarded, per-key op list is the right abstraction: the same declaration drives HF load, checkpoint export, per-layer NCCL streaming, rank-partitioned filesystem export and (NIXL) receiver-side lazy replay. Keep the ops *view-only* wherever possible so zero-copy/RDMA paths can reuse them.
- **Training layout decoupled from storage layout** via state-dict hooks (fusions, LoRA `base_layer.`): optimize freely at runtime, never leak it into checkpoints.
- **Meta-init → shard → load-by-slice** (DCP `HuggingFaceStorageReader` into FSDP DTensors): no rank ever materializes the full model. Mandatory for 100B+ MoE.
- **Per-model `cp_support` / `keep_in_fp32_for_weight_transfer` declarations**: capability and precision contracts live with the model, validated at build time.
- **Wire dtype policy (bf16 by default, declared fp32 exceptions, cast before gather)**.
- **Global token-count normalization with an explicit undo of FSDP averaging** — correct loss scaling independent of how tokens land on ranks (C details the loss).
- **AC policy with a mandatory-save set for non-replayable ops** (routing stats, top-k, DeepEP, CPU copies): the correctness lesson generalizes to any stateful op inside checkpointed regions.

**Incidental / would simplify or replace:**
- The fully synchronous loop. A custom framework should at least overlap the weight export with the next batch wait (double-buffered export; the NIXL design with staging arenas and credits is the right direction), and make checkpointing async (DCP async save exists).
- Three optimizer-offload implementations (FSDP CPU offload, state-only, full offload) with different numerics and ~1.9k lines of threaded code. If host offload is needed, pick one (full offload with optimizer-in-backward is the most capable) and test it hard; the state-only default costs a synchronous state round-trip every step and disables fused AdamW.
- Monkeypatching attention classes for CP (class-level rebinds of `_compute_attention` + global `DATA_PARAMS`/`ULYSSES_PARAMS` dicts). Prefer an attention backend object passed through the model with the CP context explicit.
- `inject_prime_lm_head` replacing `forward` at runtime; better to make the fused log-prob head a first-class module of every model.
- The HF `PreTrainedModel` base and `AutoModel` registry machinery add coupling (transformers version shims, `@strict` validators, `_check_and_adjust_attn_implementation` overrides) for little benefit in training; a small internal registry keyed by `model_type` plus HF-compatible *config parsing* and *weight names* is enough.
- "Multi-LoRA" plumbing that always runs with one adapter; LoRA MoE code that predates the dispatcher refactor.
- The HF→prime on-disk conversion cache (full copy on master): convert on the fly per shard during DCP load instead.

**Coupling points to watch:** vLLM weight names/shapes (the prime→HF chain *is* the contract with inference, including fp32 exceptions and FP8 ignore lists); mesh-order assumption shared with the batch receiver's `dp_rank`; `layers.<i>` naming used by NCCL grouping, NIXL grouping and filesystem partitioning; broadcast-dir marker protocol shared with the orchestrator/receiver (E).

## 9. Open questions

1. NIXL: which families can it serve given `LazyWeight`'s op whitelist (Qwen3.5-MoE `torch.cat`, GPT-OSS `new_empty`)? (E.) The stale-q/k/v question is resolved on the trainer side (§7).
2. What else breaks in LoRA+MoE+EP once the missing import is fixed (e.g. the double permutation)? Needs a GPU run. (NCCL fp32 keys, the inference-side location of the fp8 conversion, and PEFT keys without the `base_model.model.` prefix are resolved by E, 06.)
3. Does the orchestrator ever write `step_N/orchestrator` *after* the trainer's `maybe_clean` already deleted `step_N` (recreating a trainer-less step dir that then becomes "latest")? Retention itself is resolved (§3.16: trainer is the sole cleaner); the interleaving is runtime/B.
4. HF-impl models loaded via transformers v5 with runtime weight conversions: does `dcp_load` with `HuggingFaceStorageReader` see matching keys, given the model's `state_dict()` may use converted names? (Not exercised by a test I read.)
5. Exact DCP shard file naming (`__<rank>_0.distcp`) is torch's `FileSystemWriter` default, not set in prime-rl code — confirm against the pinned torch.
6. Does a bare `uv run trainer` (no torchrun) work? `World` defaults to rank 0 / world 1 (`world.py:8-11`), but `init_process_group` (`utils.py:192`) uses `env://`, which needs `MASTER_ADDR`/`MASTER_PORT` `[UNVERIFIED]`.
