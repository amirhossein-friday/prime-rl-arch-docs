# prime-rl — the mental model (read this first)

> Pin: `PrimeIntellect-ai/prime-rl` @ `b944873` (2026-09-26) + verifiers `69cc0f9`, renderers `6b8da3f`, prime-envs `b677502`.
> This page is the picture you should hold in your head before opening any code. Every claim here is
> expanded, with `path:line` cites, in the section docs (`sections/NN-*.md`). IDs like `H-01` point into
> `22-gotchas-and-bugs.md`. The index is in `README.md`. vLLM-internal behaviour (pause semantics, prefix cache)
> was read in the vLLM 0.24 source; the pin is 0.29.0.

## 1. One sentence

prime-rl is **asynchronous, off-policy-bounded RL for LLM agents**, built from four kinds of process. Inference serves the policy. Env servers run agents inside sandboxes. The orchestrator turns their finished episodes into packed, credit-annotated training batches. The trainer consumes those batches and pushes new weights back to inference, one version at a time.

## 2. The four process kinds and where they sit

```
                        ┌──────────────────────────── one RL run ─────────────────────────────┐
                        │                                                                       │
   GPUs (inference)     │  vLLM engines ×k  ◄── router (vllm-router :8000 or llm-d) ◄─ data ──┐  │
                        │     ▲  admin (/pause /update_weights /resume /init_broadcaster)     │  │
                        │     │  direct per-engine, bypassing the router                      │  │
   CPU (orch host)      │  ORCHESTRATOR (1 asyncio proc) ──ZMQ DEALER/ROUTER──► ENV SERVERS  │  │
                        │     │   ▲  WeightWatcher polls broadcasts/step_N/ markers     (1 per source, loopback,
                        │     │   │                                                      elastic worker pool)
                        │     │   │                                                        │  interception server
                        │     │   │                                                        │  (per worker) ─────┘
                        │     │   │                                                        │  harness in runtime
                        │     │   │                                                        ▼  (subprocess/docker/
                        │     │   │                                                           Prime/Modal sandbox)
                        │     ▼   │
   GPUs (trainer)       │  TRAINER (torchrun, FSDP2 SPMD, rank 0 = master)
                        │     batches ◄── ZMQ PUB :5555 (+READY :5556) or batches/step_S/rank_R.bin
                        │     weights ──► NCCL (rank 0 + all inference GPUs) | filesystem HF shards | NIXL RDMA
                        └───────────────────────────────────────────────────────────────────────┘
```

| Process | Count | Hardware | Owns | Section |
|---|---|---|---|---|
| **Inference** | 1+ vLLM engines behind 1 router | inference GPUs | the live policy weights; token-in/token-out generation (`/inference/v1/generate`) | 06 |
| **Env server** | 1 per train/eval source (a broker plus an elastic pool of workers) | orchestrator's host (binds `127.0.0.1`) | running one episode per request: harness in a runtime/sandbox, the interception server, the renderer/tokenizer, scoring | 07, 08, 10 |
| **Orchestrator** | exactly 1 | CPU | tasksets and sampling, scheduling, the policy-version view, credit assignment, admission, packing, shipping batches, driving weight updates | 03, 04 |
| **Trainer** | torchrun ranks | trainer GPUs | the optimizer, checkpoints, and exporting weights to inference | 05 |

The `rl` launcher starts them in this order: inference → env servers → orchestrator → torchrun. It watches each child and tears the whole run down if any child exits **non-zero**; a child exiting 0 is ignored, and success means trainer and orchestrator both exited 0 (01). On multi-node SLURM, inference nodes come first; inference node 0 runs the router; the orchestrator and env servers run on trainer node 0 (by default). Kubernetes is a pod scaffold only: the chart runs no env servers, and its example is broken (01).

## 3. The five planes (how pieces talk)

| Plane | Endpoints | Transport | Discovery | Section |
|---|---|---|---|---|
| **Rollout data** | harness → interception → `TrainClient` → router → engine | HTTP (chat-completions from the harness; token-in `/inference/v1/generate` to vLLM) | `base_url` (router); harness gets `runtime.host_url(interception)` + per-rollout secret; remote sandboxes via Prime tunnel | 06, 07, 09 |
| **Env control** | orchestrator ↔ env server | ZMQ DEALER→ROUTER, msgpack: `health`, `run(RunRequest)`, `cancel`; streamed `delta` frames then a reply | `envs/<split>/<name>.address` file written by the server (the **only** dynamic discovery) | 03, 08 |
| **Admin** | orchestrator → every engine | HTTP, one client per engine (`admin_base_url` list; **order = GPU rank order**) | static URLs / CLI-injected `ADMIN_URLS` | 03, 06 |
| **Batches** | orchestrator → trainer ranks | ZMQ PUB (topic `data_rank\|r\|`) + READY PULL, or files; msgspec msgpack `list[MicroBatch]`, **positional, no step id** | static host/port (`ORCH_ADDR` multi-node) | 04, 06 |
| **Weights** | trainer → engines | NCCL broadcast (trainer rank 0 + all inference GPUs), filesystem HF safetensors, or NIXL RDMA pulls; always wrapped in the 4-marker handshake `broadcasts/step_N/{.sender_ready,.receiver_ready,.started,.finished}` | NCCL: `weight_broadcast.host` (multi-node `$MASTER_ADDR`) :29501; NIXL: ModelExpress at `$WEIGHT_BROADCAST_HOST`:8001; FS: the shared run dir | 05, 06 |

## 4. The loop (one training step, compressed)

1. **Sample.** `TrainSource` picks an env per group, weighted by `ratio` with seed 42, and then a task from that env's curriculum sampler. A group is `group_size` separate `run` requests for the same task, dispatched one at a time and back-to-back for prefix-cache locality. It is stamped with `policy_version_at_start` and (live-policy groups only) `cache_salt = str(that version)`.
2. **Roll out.** The env server rebuilds the `Task` from its JSON. It starts a fresh sandbox (the network is open during setup; it is cut before the agent runs only under a restricted policy, see §6) and launches the harness pointed at the interception server. Every model call is intercepted and rendered to tokens by the renderer. Where possible it is *bridged* onto the previous turn's exact tokens, then sent to vLLM, and recorded into a message **graph** (a `Trace`). The episode is then scored: metrics → rewards → judges. Deltas stream back and are assembled into a `WireEpisode`.
3. **Credit.** The orchestrator's per-env `Algorithm` annotates the graph nodes. `score_episode` runs on arrival (OPD/OPSD reference logprobs); `score_group` runs when the group completes (GRPO $A_i=r_i-\bar r$, MaxRL, RAE, …). The curriculum's gates then admit or reject the group (no gates are configured by default, so every group is admitted).
4. **Compile.** Each trainable branch of each admitted trace becomes a `TrainingSample` with per-token `token_ids`, `mask`, inference `logprobs`, `temperatures`, `advantages`, `rl/ce/ref_kl` weights, `ref_logprobs`, `routed_experts` and `sampling_mask`. Zero-advantage tokens are pruned. A batch is cut at `batch_size` **traces**.
5. **Gate and ship.** Batch $S$ ships only once inference has applied $v_{S-2}$ (`TARGET_LAG=1`, hard-coded); the same $V \ge S-2$ test gates *train* dispatch while $S$ is collecting (eval ignores it). It is packed first-fit-decreasing into `seq_len` bins (overlong samples are silently truncated) and balanced across DP ranks by FLOPs, with dummy padding. It then goes over ZMQ or to files.
6. **Train.** The trainer (synchronous) takes batch $S$ from $\theta_{S-1}$. It runs forward/backward over variable-length micro-batches with a loss $\mathcal{L}=\sum_c \frac{1}{N_c}\sum_t w^c_t\ell^c_t$ over components `rl` (IPO by default, a masked importance ratio $\pi_\theta/\mu$), `ce` and `ref_kl`, each normalized by its global token count. Then clip, **one** optimizer step, scheduler step, and a **blocking** handshaked broadcast of $v_S$, followed by an optional blocking checkpoint.
7. **Swap.** The orchestrator's `WeightWatcher` sees `.sender_ready`. It raises the dispatch barrier, drops stale groups, and pauses the engines (`mode="keep"`: freeze, don't abort). It applies the weights, resumes, and sets `policy.version = S`. Only then is the barrier lifted and the lag gate re-evaluated.

## 5. The invariants everything leans on

- **Tokens are the source of truth.** The trainer sees exactly the ids the sampler saw. Renderers never re-tokenize sampled tokens, and a bridge either proves the exact-prefix property or returns `None`, which forks a new branch/sample. The first token of every packed sample is never trained.
- **Steps are 1-indexed; versions are 0-indexed.** $v_0$ is the base model. Trainer step $s$ trains batch $s$ from $v_{s-1}$ and broadcasts $v_s$; at startup it re-broadcasts $v_{\text{start}-1}$. Staleness of a trace in batch $s$ is $(s-1)-\text{start}$, capped hard by `max_off_policy_steps` (8) in the sink's sweep.
- **Counters instead of ids on the batch wire.** The orchestrator's sender and the trainer's receiver both start at `progress.step` and advance once per batch. No message carries a step: ZMQ pairs batches with steps by arrival order, and the FS path's `step_S` comes from the same counter. So trainer and orchestrator must (re)start together, and orchestrator-only restart is unsupported.
- **Lockstep weight versions.** The trainer blocks on `.receiver_ready` for every version. Versions are applied strictly in order, and `policy.version` advances only after the engines have applied it.
- **Order and shape agreements that nobody checks at runtime:**
  - the admin URL list must be in GPU-rank order, each URL owning an equal share of the NCCL `inference_world_size`, which must equal the engines' GPU count (local single-node `rl` computes it before DP auto-fill and gets it wrong: B-01);
  - `num_train_workers` must equal the trainer's DP degree, and `pad_to_multiple_of` its CP degree (the `rl` launcher auto-fills both);
  - the renderer tokenizer, the engine tokenizer and the trainer tokenizer must be identical;
  - `cp` is the innermost mesh dim, which is what makes `dp_rank = rank // (world/dp)` valid.
- **Task provenance round-trips.** `(task.key, task.hash)` must survive the JSON trip through the env server; a mismatch crashes the orchestrator.

## 6. Mental traps (things that look true but aren't)

- **"The trainer runs the algorithm."** No. The trainer is algorithm-blind; algorithms are orchestrator-side annotators (04).
- **"Each group is on one policy version."** No. The salt and `start` are per group, but members dispatch one at a time. A group, a multi-turn episode, or even a single mid-decode request can straddle a weight swap and reuse KV computed under the old weights, because the prefix cache is never reset. $\mu$ is then a hybrid of old and new. The IS ratio still sees the true sampling probabilities, but $\mu \ne \pi_{\text{start}}$ (03 §7.1, 04 §3.11, 06 §3.6.1; H-01).
- **"Any harness can be trained."** No. The train client accepts only chat-completions with un-namespaced function tools. `codex` and `claude_code` are eval-only, and the failure shows up at runtime as a retryable 502. Trainable at their defaults: `null`, `bash`, `rlm`, `pi`, `kimi_code`, `prime_agent`, `browser_use`, `mini_swe_agent`, `terminus_2`, and (inferred) `hermes_agent`. `openclaw` depends on the model-name prefix (08 §3.7; O-05).
- **"`batch_size` is samples."** No, it counts traces. One trace yields one sample per trainable branch (H-19).
- **"`[tokenizer]` / `chat_template` overrides change what's trained."** No. The renderer is built in the env-server worker from `model.name`; `chat_template` reaches no renderer. A local `model.name` plus a mapped `[tokenizer] name` passes validation and silently runs `DefaultRenderer` (no bridge, no tools). Set `[orchestrator.renderer] name` (09 §7.1-7.2; H-04).
- **"Sandboxes are local by default."** No. The default runtime is a remote Prime sandbox (credentials, tunnels, billing), and a harness block without `id` silently becomes `bash`. Set both `env.agent.runtime.type` and `env.agent.harness.id` (08, 10; O-04, H-03).
- **"The sandbox cuts the network before the agent runs."** Only under a restricted policy. The default `allow=["*"]` makes the cut a no-op on every runtime. Most worker-side timeouts also default to none; only the agent phase has one (4 h) (08; H-18, O-06).
- **"Env servers are supervised."** Only the broker is. Elastic-pool workers are never restarted, the broker answers health checks itself, and runs have no timeout, so a dead worker's runs hang (08; B-05).
- **"Resume picks the latest complete checkpoint."** No. Trainer, orchestrator and launcher each take the max `step_*` name independently, with no completeness marker. A crash can leave an orchestrator-only `step_N` that the trainer can't load (03, 05; O-20).
- **"Eval in an RL run measures one policy."** No. Epochs span weight updates; `policy_version` is a lower bound (11; H-17).

## 7. Where the complexity actually lives

| Subsystem | Why it's hard | Size (prime-rl src / deps) |
|---|---|---|
| Renderers + bridge | exact multi-turn tokenization across ~30 chat formats, with thinking-strip policies | renderers ≈ 23k LOC |
| Verifiers interception + graph | third-party agent CLIs, three dialects, retries, forks and compaction captured as exact token branches | verifiers v1 ≈ 31k LOC |
| Trainer engine | FSDP2 plus EP (DeepEP), CP (ring/ulysses), FP8/MXFP8, optimizer offload, and the HF↔prime conversion chain that is also the weight-sync ABI | trainer ≈ 28k LOC |
| Weight sync | NCCL rank arithmetic across two process groups; NIXL static copy plans; LoRA | transports + inference ≈ 5.2k LOC |
| Orchestrator | async scheduling, the lag gate, the staleness sweep, adaptive concurrency, and swap barriers | ≈ 6.9k LOC |

Where to start depends on your goal:
- **Building a replacement framework:** read `23-custom-framework-notes.md`.
- **Changing something:** read `21-extension-points.md` first, then `22-gotchas-and-bugs.md`.
- **Running or debugging:** `22-gotchas-and-bugs.md` (top-of-list traps, §6 safe-first-run checklist), then `12-end-to-end-flows.md`.
