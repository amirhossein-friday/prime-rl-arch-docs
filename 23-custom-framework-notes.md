# Building our own RL framework: what to take from prime-rl

> Pin: prime-rl @ `b944873`, verifiers `69cc0f9`, renderers `6b8da3f`. This is a **design brief**, not
> a description. Every "prime-rl does X" below is established, with cites, in the linked section
> (`→ 03 §7.1`); the bug and hazard IDs are in `22-gotchas-and-bugs.md`. Verdicts are Atlas's
> opinion, argued from those facts. Read `00-mental-model.md` first.

> **See also** [`notes/miles-arch/23-custom-framework-notes.md`](https://github.com/amirhossein-friday/atlas/blob/main/notes/miles-arch/23-custom-framework-notes.md) (Atlas workspace repo): a head-to-head miles vs prime-rl decision table. It refines several verdicts below (per-token weight versions, the engine-side weight-update session, register-after-sync, cells and fault tolerance, correctness machinery).

## 0. The one-paragraph verdict

prime-rl's **ideas** are mostly right. Its **seams** are where the pain is: counters instead of ids, per-group version tags, static ports, the filesystem marker bus, order-dependent validators, and monkeypatched vLLM.

The four hardest-won components are the ones to reuse rather than rewrite:
1. token-in/token-out generation with engine-reported logprobs, sampling masks and routed experts;
2. renderers with a proof-carrying bridge;
3. interception-as-contract with a token-exact message graph;
4. the invertible HF↔trainer conversion chain.

The parts to redesign are the ones where prime-rl couples processes through implicit shared state:
- step and version accounting;
- resume;
- discovery;
- the orchestrator pipeline shape;
- supervision (timeouts, liveness).

**Recommended build strategy:** depend on `renderers` and `verifiers` as libraries, pinned. Write our own orchestrator, launcher and wire protocol. Start from prime-rl's trainer engine (a fork or a trimmed copy), and keep its conversion-chain ABI.

## 1. Decision table

| # | Decision | prime-rl's answer | Verdict | Our answer (spec) | Evidence |
|---|---|---|---|---|---|
| D1 | Process topology | inference / env servers / orchestrator / trainer, all separate processes | **Keep** | Same four roles. One orchestrator is fine up to ~10³ in-flight episodes; after that, shard env control, not scheduling. | → 00 §2, 03 §2; P-12 |
| D2 | Discovery | static well-known ports + CLI-injected URLs; address files only for env servers | **Replace** | Every listener binds `:0` and registers `(role, idx) → addr` in a run-scoped registry (a file dir or small KV) keyed by run id. Consumers wait on the registry. This removes port collisions and multi-tenancy clashes. | → 01 §7, §8; O-07, O-11 |
| D3 | Config | pydantic-config TOML `@` composition; ~20 order-dependent after-validators; resolved JSON per process | **Keep the format, replace the resolution** | Keep the typed schema, `@` composition and resolved-JSON handoff. Replace the validator chain with one explicit `resolve(composite) → per-process configs` pass. Topology values (world sizes, ranks, DP/CP, hosts) are computed once from the placement plan, never by config arithmetic. This fixes the NCCL world-size undercount by construction. | → 02 §8, 01 §7.5; B-01, B-10, O-09, O-10 |
| D4 | Generation interface | token-in `/inference/v1/generate`; renderer tokenizes client-side; engine returns logprobs, sampling mask, routed experts | **Keep (core)** | Same contract, written down as a versioned schema that covers stop ids, `skip_special_tokens`, the logprob format, the routed-expert offset, the sampling-mask shape and `cache_salt`. Assert at startup that renderer and engine agree on special-token ids. | → 06 §3.3, 09 §8; H-04, H-12 |
| D5 | Multi-turn token exactness | renderer `bridge_to_next_turn` proves the exact prefix or returns `None`; graph forks a branch | **Keep (reuse `renderers`)** | Reuse the library. Expose `thinking_retention` at the algorithm level and log the extension-break rate per env. | → 09 §3.6, §8; H-25, P-03 |
| D6 | Agent integration | every model call goes through a per-rollout interception server; three dialects; training is chat-completions only | **Keep (reuse `verifiers`)**, plus a config-time check | Make trainability a config-time check: the harness must speak chat-completions, not fail per call with a 502. If Responses/Anthropic scaffolds must be trainable, add per-dialect rendering. | → 07 §7.1, 08 §3.7; O-05, H-03 |
| D7 | Sandbox execution | runtimes: subprocess/docker/apptainer/Prime/Modal; default remote Prime; per-rollout harness install | **Keep the interface, change the defaults** | Default to a local runtime (Apptainer or Docker on the cluster). Pre-bake harnesses into images. Default the network policy to deny-after-setup; prime-rl's default `allow=["*"]` makes the cut a no-op. Give every phase a default timeout. | → 08 §7; O-04, O-06, H-18, P-08 |
| D8 | Env / task model | orchestrator loads and owns the taskset; env server rebuilds a `Task` from `TaskData` JSON per request | **Keep the split, replace the loading** | A streaming or indexed task source with a cursor keyed by `task.key`, not by position. Keep the env schema importable without the heavy env package. Run each env's executor in its own venv or container, so one lock file doesn't force dependency conflicts. | → 10 §8; O-03, O-08, H-28 |
| D9 | Credit assignment placement | algorithm-blind trainer; orchestrator-side annotators write per-token advantages, loss weights and ref logprobs | **Keep (core)** | Same: trainer loss = $\sum_c \frac{1}{N_c}\sum_t w^c_t \ell^c_t$ over named components. Keep `score_episode`/`score_group`. Checkpoint algorithm state; RAE baselines are lost today. | → 04 §1, §8; O-25, H-32 |
| D10 | Off-policy control | `TARGET_LAG=1` hard-coded; dispatch gate + ship gate; per-**group** `PolicySpan.start` and `cache_salt`; hard staleness sweep at the sink | **Keep the two-level control, fix the tagging** | Tag version and salt per **request**: record each model call's serving version in the trace. Make lag a config value. A swap either aborts or re-prefills in-flight requests (configurable), or the trace marks the tokens generated across it. | → 03 §7.1, 04 §3.11, 06 §3.6.1; H-01, H-16 |
| D11 | Batch wire | msgspec positional structs; ZMQ PUB/SUB (lossy past its HWM of 10; only the lag gate keeps it below) or files; **no step id** in any message | **Replace** | A versioned schema with an explicit `step`, `policy_versions` and a producer run id on every batch. A reliable channel with acks (ROUTER/DEALER or PUSH/PULL). Either side can restart and resynchronize. | → 04 §8, 06 §8; O-24, O-53 |
| D12 | Trainer engine | FSDP2 + EP/CP + FP8 + three optimizer-offload modes; custom modeling registry; synchronous loop | **Fork and trim** | Keep: meta-init → shard → load-by-slice; per-component global-count normalization; the conversion chain; per-model capability declarations. Trim: to one offload mode, and make CP an explicit attention backend (not a monkeypatch). Overlap export with the next batch wait, and make checkpoints async. | → 05 §8; B-11, B-17, B-18, P-04, P-05 |
| D13 | Weight sync | 4-marker FS handshake around NCCL / FS / NIXL; admin HTTP per engine; `/pause` keep; retrying POSTs | **Keep the lockstep shape, replace the bus** | Keep "consumer pauses → acks → receives → resumes" and strict version order. Move markers to an RPC or control channel. Make updates idempotent by version id (retry-safe). Recompute NCCL world size from engines' self-reported worker counts. Prefer RDMA-pull (NIXL-style) for elasticity, but fix view-vs-copy for fused params. | → 06 §8, 05 §7; B-01, B-02, B-04, B-07 |
| D14 | Checkpoint / resume | each process resolves "latest" by max `step_*` name; no completeness marker; the orchestrator saves ahead of the trainer | **Replace** | The launcher pins the resume step. "Complete" means a marker written after **both** sides saved. A rewind deletes or archives newer steps. Checkpoint the sink buffer and in-flight task keys so resume doesn't skip tasks. | → 03 §3.15, 05 §3.16; O-20, O-21, O-23, O-25, B-15 |
| D15 | Supervision | launcher monitors top-level processes; no rollout timeout; elastic env-pool workers unsupervised; ship-gate hold has no timeout | **Replace** | Every wait has a deadline and a health check. Env workers heartbeat. A run request carries a deadline (agent timeout + slack). Cancellation never goes through a bounded queue whose only consumer may be blocked on the same event (the ship-gate deadlock). | → 03 §7.2, 08 §7; B-03, B-05, B-14, O-06 |
| D16 | Observability | file monitor as system of record; episode-once + append-only annotations; `all` vs `effective` subsets | **Keep (copy)** | Same design. Namespace metrics by producer. Use one MFU definition over the actual token counts and the real `head_dim`. Make trainer annotations sampled. | → 11 §8; B-13, P-14 |
| D17 | Launch | `rl` Python launcher local; ~600-line bash multi-node SLURM template; Helm scaffold for k8s | **Replace** | One Python per-node agent used by both local and multi-node launches, reading the same placement plan. k8s comes later, as a real operator: env-server workload, trainer JobSet, registry. | → 01 §8; O-12, O-13 |

## 2. What "correct" requires — the invariants to design in from day one

1. **Token identity end to end.** The trainer trains exactly the ids the engine sampled, the logprobs are the engine's processed logprobs, and the tokenizer is a single config value shared by renderer, engine and trainer. (prime-rl lets the `[tokenizer]` override silently not reach the renderer → 09 §2.)
2. **Behaviour policy is known per token.** Store $\mu_t$ (inference logprob) and the *serving version* per model call, and treat any token generated across a swap as mixed. prime-rl records $\mu_t$ correctly but its version tag is per group, so $\mu \neq \pi_{\text{start}}$ goes unnoticed → 03 §7.1.
3. **Explicit step and version on every cross-process message** (batches, weights, checkpoints). No counter coupling.
4. **Idempotent control operations keyed by version:** weight update, checkpoint, resume. prime-rl's `/update_weights` retry can deadlock NCCL → 06 §3.6.1.
5. **Numerics guards.** Clamp or mask *before* exponentiating log-ratios: prime-rl's IPO and `ref_kl` can turn a whole step into NaN from one token with $\log r > 88$ → 04 §7. Reject temperature 0 for trained sampling. Truncation to `seq_len` is either an error or a counted metric, never silent.
6. **Every wait bounded** (see D15).

## 3. Reuse vs rewrite

| Component | Reuse? | Why |
|---|---|---|
| `renderers` | **Reuse as a pinned dependency** | ~23k LOC of per-family chat formats with a parity test matrix against reference encoders. Rewriting is pure cost. Contribute fixes upstream. |
| `verifiers` (interception, graph, harnesses, runtimes, tasksets) | **Reuse as a pinned dependency** | ~31k LOC; the interception + graph design is the key idea and it is well tested. Our code talks to it through the env-server protocol (`RunRequest` → `WireEpisode`), a narrow seam. Watch the version-coupled pieces: `TaskData` hashing, the runtime union on the wire, and the delta field lists. |
| prime-envs | **Reuse selectively** | Take the envs we need and treat them as data plus scoring code. Drop the one-lock monorepo coupling. |
| Orchestrator | **Rewrite** | 6.9k LOC hand-wired pipeline: no registry for sources, sinks or dispatcher, and algorithms, samplers and gates are in-tree edits with no out-of-tree hook (→ 21, "Things with NO seam"). The core decisions, and all the fixes, live here (D10, D11, D14, D15). Keep the ideas: provenance stamping, the two-level lag control, adaptive concurrency from KV metrics (worth porting nearly verbatim), and FLOP-aware packing. |
| Trainer | **Fork and trim** | 28k LOC, much of it model families and kernels we'd otherwise rewrite badly. Keep the engine and the conversion ABI. Remove the monkeypatching and the extra offload modes. Fix the confirmed bugs first (see 22). |
| Inference extension | **Rewrite thin** | ~2.9k LOC of vLLM extension (server overrides, worker extensions, Dynamo plane, ~0.9k LOC of patches, many of them upstream workarounds). Implement the admin routes (pause, update, init) as a vLLM plugin with explicit, versioned, idempotent semantics. |
| Launcher / config | **Rewrite** | D2, D3, D17. Keep pydantic-config itself if its parser fixes land (→ 02 §8). |

## 4. A minimal first slice (to validate the design before scaling)

1. **Single node:** 1 vLLM engine, 1 trainer GPU, 1 env server with a subprocess runtime and the `null` harness, a math taskset.
2. **Wire:** a versioned batch schema with `step` and per-token `policy_version`, over a reliable channel; filesystem weight sync, idempotent by version.
3. **Orchestrator:** sampler → dispatcher → sink with GRPO via `score_group`; the lag gate as config; per-request salt; the staleness sweep.
4. **Trainer:** the forked engine with the IPO loss guarded before exponentiating.
5. **Test harness:** fault injection — kill each process mid-step, resume, and assert the step and version invariants and no skipped tasks.
6. **Only then:** NCCL/NIXL sync, multi-node, sandboxes, agentic harnesses, MoE.

## 5. Open design questions for us

- **Abort, re-prefill or mark on swap.** In-flight requests across a weight swap can be aborted (wasting compute), re-prefilled under the new weights (costs prefill, keeps $\mu$ single-version), or kept but marked (prime-rl keeps them without marking). It is a throughput/purity trade-off; measure it.
- **Where credit assignment lives when it needs model calls.** OPSD reference scoring goes through the policy's own inference pool today (OPD through an external teacher endpoint), at arrival and before admission, so it contends with rollouts and is paid even for rejected groups (P-13). Should reference scoring get a separate pool, or run inside the trainer?
- **Per-dialect rendering.** Is making Responses and Anthropic harnesses trainable worth it, or do we standardize on chat-completions scaffolds?
- **Lag.** Should lag be a scalar (`TARGET_LAG`) or a budget in tokens or time?
