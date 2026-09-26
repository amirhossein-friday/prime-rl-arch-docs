# Extension points — prime-rl @ b944873

> Every seam for adding or replacing something in prime-rl, written as recipes. Consolidates §6 of sections 01–11 plus the load-bearing §3/§7 material, reconciled against `_brief/LEDGER.md` (later ledger entries win).
> Pin: prime-rl `b944873` + verifiers `69cc0f9`, renderers `6b8da3f`, prime-envs `b677502`, pydantic-config `65b15df`. **vLLM caveat:** every vLLM-internal claim was read in vLLM 0.24 source; the pin is exact v0.29.0 wheels (`pyproject.toml:59, 276-278`). Re-check vLLM behaviour before relying on it.
> Read `00-mental-model.md` first. After this doc, read `22-gotchas-and-bugs.md`.

## How to use this doc

1. Find your seam in the [index](#index). The **Mechanism** column tells you whether you can work out-of-tree (config import path, `__all__` export, entry point) or must edit prime-rl / a submodule (registry dict, union, if-chain).
2. Jump to the recipe. Every recipe has the same seven headings in the same order: **What you're changing**, **Files to touch**, **Registration mechanism**, **Config wiring**, **Lockstep changes**, **How to test/validate**, **Pitfalls**.
3. If the seam you want is not there, check [Things with NO seam](#no-seam) before designing: it lists what you can only change by forking or editing core.
4. Cites: `→ 04 §6.1` = section doc `sections/04-*.md` §6.1; code cites are `path:line` at the pin.

**Path shorthand** (all relative to the prime-rl repo root, `/tmp/atlas-prime/prime-rl`):

| Prefix in this doc | Real path |
|---|---|
| `orchestrator/…`, `trainer/…`, `transports/…`, `inference/…`, `entrypoints/…`, `monitors/…`, `utils/…`, `eval/…`, `templates/…`, `dashboard/…` | `src/prime_rl/…` |
| `cfg/…` | `packages/prime-rl-configs/src/prime_rl/configs/…` |
| `cfgutils/…` | `packages/prime-rl-configs/src/prime_rl/utils/…` |
| `vf/…` | `deps/verifiers/verifiers/v1/…` |
| `rnd/…` | `deps/renderers/renderers/…` |
| `pyproject.toml`, `scripts/`, `tests/`, `docs/`, `k8s/`, `examples/`, `configs/` (TOML runs), `deps/…` | as written, from the repo root |

**Registration vocabulary** used in the index: *union* = pydantic discriminated union on `type`/`name` (a config class must be added); *registry dict* = module-level dict keyed by the discriminator; *if-chain* = a factory function with `if/elif`/`match`/`isinstance` branches; *`__all__`* = the verifiers plugin loader imports a module by name and takes the one exported subclass; *import path* = a config field holding `pkg.mod.fn`; *entry point* = Python packaging entry point; *monkeypatch* = runtime rebinding of third-party symbols.

**Rules that apply to every seam**

- **Every process re-parses its own resolved JSON** (`configs/attempt_N/resolved/*.json`). New validators must be fixed points on their own output, and every class in a config union must be importable in every process that parses it (→ 02 §1, §5.1; `entrypoints/rl.py:89-113`).
- **Config parsing imports plugin packages.** Taskset/harness/env ids are resolved by import while validating, in the launcher (including the SLURM submit host), orchestrator, env server and eval (→ 10 §3.2, §7 G2; `vf/configs/agent.py:53-66`).
- **Positional wire structs are append-only** (`TrainingSample`, `MicroBatch`: `transports/batch/types.py:79-80, 118`; → 06 §3.11).
- **Almost nothing is a plugin.** The only out-of-tree hooks are: custom rl loss and Echo filter (import path), taskset/harness/env/judge packages (`__all__`), reward/metric/stop functions (config `fn = "file.py:func"`), the `vllm.general_plugins` entry point, and a custom SLURM template path. Everything else is an in-tree edit to prime-rl or a submodule (verifiers, renderers).
- **Validate config changes with `uv run rl @ x.toml --dry-run`** and diff `configs/attempt_*/resolved/`. `--dry-run` is **not** side-effect free: `--clean --dry-run` rmtrees the run dir, `--resume --dry-run` deletes future `broadcasts/`/`batches/`, and it repoints `configs/latest` (→ 01 §7.1; `entrypoints/rl.py:656-686`).

---

<a id="index"></a>
## Index

Difficulty: **S** = config or out-of-tree only; **M** = in-tree edits in one component; **L** = several files across processes; **XL** = cross-repo plus inference/trainer parity.

| # | Seam | Registration lives in | Mechanism | Diff. | Lockstep changes elsewhere |
|---|---|---|---|---|---|
| [A1](#a1) | RL algorithm (`score_episode`/`score_group`) | `orchestrator/algo/__init__.py:43-53`; `cfg/algorithm.py:394-405` | registry dict + union | M | none in trainer/packer; extra models → `FrozenModelConfig` + `connect()` |
| [A2](#a2) | Advantages & per-token loss-weight streams | `orchestrator/algo/routing.py:11-85` | direct node mutation | S–M | none (fields already on the wire) |
| [A3](#a3) | Custom rl loss | `cfg/trainer.py:603-613`; `trainer/rl/loss.py:279-302` | import path | S | module importable on every trainer rank |
| [A4](#a4) | Using the `ce` / `ref_kl` components | `orchestrator/algo/routing.py:68-85`; `trainer/rl/loss.py:305-414` | `action_loss_type` + node weights | M | a new component type has no seam |
| [A5](#a5) | Curriculum task sampler | `orchestrator/curriculum/base.py:35-40`; `cfg/orchestrator.py:237` | union + isinstance | M | orchestrator checkpoint (automatic via `state_dict`) |
| [A6](#a6) | Curriculum admission gate | `orchestrator/curriculum/base.py:42-48`; `cfg/orchestrator.py:256` | alias → make a union + isinstance | M | resume refuses a changed gate set |
| [A7](#a7) | Frozen generation source / teacher | `orchestrator/generation_source.py:32-37` | config | S | endpoint you run; shared tokenizer |
| [A8](#a8) | New `TrainingSample`/`MicroBatch` field | `transports/batch/types.py:30-125` | positional msgspec, append-only | L | packer, trainer tensorizer, annotations |
| [B1](#b1) | Single-turn verifiable taskset | env package `__all__` | `__all__` + uv workspace member | S | install in every validating process; commit `uv.lock` |
| [B2](#b2) | Multi-turn tool env | `Task.toolsets` / `Taskset.toolsets` | MCP `Toolset` | M | MCP-capable harness; tunnels for remote runtimes |
| [B3](#b3) | Sandboxed agentic env (SWE-style) | env package | `__all__` + `NEEDS_CONTAINER` | M | images in a pullable registry; Prime creds by default |
| [B4](#b4) | Harbor tasks | `vf/tasksets/harbor/` (built-in) | config (`id="harbor"`) or thin subclass | S | export `HarborEnv` for separate verifiers |
| [B5](#b5) | Isolated verifier / multi-agent / judged envs | exported `vf.Env` subclass | `__all__` | M | `algo.validate_env` for structure-aware credit |
| [B6](#b6) | Rewards, judges, stops, intercepts | `vf/configs/task.py:11-63` | config `fn=` / judge `__all__` | S | judge endpoint + key in env-server env |
| [C1](#c1) | Harness / agent scaffold | `vf/harnesses/<id>` or top-level package | `__all__` | M | trainable only if chat-completions |
| [C2](#c2) | Sandbox runtime provider | `vf/runtimes/__init__.py:41-70` | hard-coded unions + map | L | orchestrator's verifiers must know the type (wire) |
| [C3](#c3) | Dialect | `vf/dialects/__init__.py:11` | module-level tuple | L | TrainClient + renderer path to train with it |
| [C4](#c4) | Client | `vf/clients/client.py:73-80`; `vf/configs/client.py:88` | isinstance + union | L | prime-rl `setup_client` |
| [C5](#c5) | Renderer (chat format) | `rnd/base.py:1316-1383, 930-1047`; `rnd/configs.py:997-1077` | registry dict + union + exact-name map | XL | engine/trainer tokenizer parity; mm feature branch |
| [C6](#c6) | Tunnel | `vf/interception/tunnel/__init__.py:10-19` | union + isinstance | M | elastic pool hard-codes Prime tunnels |
| [D1](#d1) | Model family in the trainer | `trainer/models/__init__.py:35-109` | `AutoConfig.register` + registry tuples | XL | vLLM architecture/loader, renderer, fp32 keys, NIXL ops |
| [D2](#d2) | Optimizer / LR scheduler | `trainer/optim/__init__.py:124-148`; `trainer/scheduler.py:100-121` | union + `match` | M | full-offload kernels, validators |
| [D3](#d3) | MoE compute backend / token dispatcher | `trainer/moe_runtime.py:34-124` | union + isinstance | M | AC mandatory-save ops |
| [D4](#d4) | Runtime fusion / CP style | `trainer/models/fusions.py:47-118`; `utils/cp.py:40-70` | module flag + monkeypatch | M | checkpoint names; NIXL aliasing |
| [D5](#d5) | Extra checkpoint state | `trainer/ckpt.py:79-160` | edit `AppState` | M | orchestrator state has no seam |
| [E1](#e1) | Weight transport | `transports/weights/__init__.py:22-51`; `inference/vllm/server.py:59-63` | 4 unions + if-chains + string map | XL | trainer, orchestrator, SFT online-eval, vLLM worker, SLURM |
| [E2](#e2) | Batch (rollout) transport | `transports/batch/__init__.py:21-40`; `cfg/shared.py:255` | union + if-chain | L | trainer + orchestrator + SLURM host injection |
| [E3](#e3) | Admin plane backend (Dynamo pattern) | `orchestrator/clients.py:239-245` | if-branch | M | all weight receivers, eval runner |
| [E4](#e4) | Inference route / vLLM patch / generate response | `inference/vllm/server.py`; `inference/patches.py:4-25`; `pyproject.toml:50-51` | FastAPI router + entry point + monkeypatch | M–L | renderer client + TrainClient parse the response |
| [E5](#e5) | Router backend | `cfg/inference.py:368` | union + launch branches | L | SLURM router script; local launcher |
| [F1](#f1) | Concurrency policy | `orchestrator/orchestrator.py:335-356` | none (class hard-wired) | M | `EvalRunner` builds its own controller |
| [F2](#f2) | Weight-swap observers / hooks | `orchestrator/orchestrator.py:380-388` | callbacks wired in `setup` | M | blocking hook blocks the swap |
| [G1](#g1) | Monitor backend | `monitors/__init__.py:48-100` | hand-written if-chain | M | 5 `monitors.setup` call sites |
| [G2](#g2) | Metrics | `orchestrator/metrics.py`; `trainer/rl/train.py:619-697` | edit dicts / `gauges()` | S–M | W&B overview + dashboard constants |
| [G3](#g3) | Eval driver / annotation producer | `eval/runner.py`; `monitors/file/traces/update.py:17` | reuse class / extend tuple | M | dashboard folds only `STREAM_FIELDS` |
| [H1](#h1) | New config field end to end | `cfg/*.py`; `cfgutils/validation.py:67-179` | pydantic field (+ shared propagation) | S–L | validator order; dump exclusions; SLURM injection |
| [H2](#h2) | Launch target: SLURM template | `cfg/rl.py:857-867`; `entrypoints/rl.py:426-440` | `slurm.template_path` | M | copy bundled includes |
| [H3](#h3) | Launch target: k8s / external components / new process | `entrypoints/rl.py:53-61, 89-113`; `k8s/prime-rl/` | `serve.address`, omit `[inference]`, edit launcher | L | chart is a scaffold; many manual wirings |

---

## A. Credit assignment, loss, and the sample path

<a id="a1"></a>
### A1 · RL algorithm (`score_episode` / `score_group`)

**What you're changing** — the per-env, orchestrator-side annotator that writes per-token credit, reference logprobs and loss-weight streams onto the verifiers message graph. The trainer never sees the algorithm (→ 04 §1). The orchestrator calls the template methods `finalize_episode` (on arrival; skips episodes with no trainable trace) and `finalize_group` (on group completion); **you override `score_episode` / `score_group`, never `finalize_*`** (`orchestrator/algo/base.py:67-80`; `orchestrator/train_sink.py:224, 264-266`; → 04 §3.2, ledger "RESOLVED"). `score_group` is reached only if the group has ≥1 trainable survivor (the guard is in the sink, `train_sink.py:264-266`).

**Files to touch** (in order)
1. `cfg/algorithm.py`: `class XAlgoConfig(BaseAlgoConfig)` with `type: Literal["x"] = "x"` and `action_loss_type: ClassVar[ActionLossType]` (`:75, 161-174`); add it to the `AlgoConfig` union (`:394-405`). Override `validate_env` if your math encodes episode structure (`:193-198`; `hierarchical_grpo` requires a proposer-solver env, `:279-294`).
2. `orchestrator/algo/x.py`: `class XAlgorithm(Algorithm)`; same `action_loss_type`; override `score_episode` (rollout-local, model I/O) and/or `score_group` (cohort-relative). Worked RLOO example: → 04 §6.1.
3. `orchestrator/algo/__init__.py`: add `"x": XAlgorithm` to `ALGORITHM_CLASSES` (`:43-53`) and to `__all__`.

**Registration mechanism** — config discriminated union on `type` + registry dict keyed by the same string. `build_algorithm` asserts `cls.action_loss_type == config.action_loss_type` (`orchestrator/algo/__init__.py:56-58`). No import-path hook: this is an in-tree edit.

**Config wiring** — `[orchestrator.algo] type = "x"` (default for every train source) or per source `[[orchestrator.train.source]] algo.type = "x"`. `inherit_env_algorithms` deep-copies the top-level algo into sources without one, then `validate_env_algorithms` calls `algo.validate_env(env)` (`cfg/orchestrator.py:627-642`). `validate_sampling_source`: `rl`/`ref_kl` need `sampling.source == "policy"`; `sft` (`ce`) needs a frozen source (`cfg/algorithm.py:179-191, 354-365`). Truncating train sampling (top-p/top-k) rejects `opd`/`opsd` (`cfg/orchestrator.py:683-689`). Fields under the `algo` union get no typed CLI parsing: use `--orchestrator.algo.x 1` (space form), never `--…=v` (→ 02 §7.2).

**Lockstep changes** — none in the dispatcher, packer or trainer (→ 04 §6.1). An extra model (teacher, judge): add a `FrozenModelConfig` field and `await self.connect(cfg)` in `setup()`; only **one** pool is tracked in `self.connected` for shutdown, so close any others yourself (`orchestrator/algo/base.py:18-32, 62-65`). Reference scoring uses `PrefillScorer` → `POST /inference/v1/generate` with `prompt_logprobs=1`, so the teacher must serve that route and the orchestrator must import vLLM (`orchestrator/clients.py:523-556, 528`; → 04 §7.11). A teacher/student pair must share a tokenizer; nothing checks it (→ 04 §7.10).

**How to test/validate** — mirror `tests/unit/orchestrator/test_advantage.py` and `test_algorithms.py` (→ 04 header); `uv run rl @ x.toml --dry-run` to prove the union resolves; then a short real `rl` run (console `Reward x.xxxx` line is what CI regexes, → 11 §7.15).

**Pitfalls**
- Iterate `iter_trainable_traces(group)` (`orchestrator/algo/base.py:35-42`). The docs example iterates raw `episode.traces` and would put errored/untrainable traces in the baseline (`docs/algorithms.md:418-425`; → 04 §6.1).
- Errored traces shrink the group; overload-cancelled groups and groups with errored members still train with fewer members (→ 03 §7.7, 04 §7.8).
- Algorithm state is not checkpointed (RAE baselines, `orchestrator/algo/rae.py:29-37`; → 03 §7.15, 04 §7.12). `Algorithm` has no `state_dict` hook; persisting state means editing the orchestrator checkpoint (see [no-seam](#no-seam)).
- `score_episode` runs at arrival, **before admission**, with unbounded per-episode concurrency; rejected groups still pay for it (`orchestrator/algo/opd.py:43`, `opsd.py:76`; → 04 §7.9).
- `advantages=None` on a sample with `rl` member tokens raises in the packer (`trainer/batch.py:380-390`; → 04 §5.5).
- `GRPO` pools all trainable traces of all agents into one baseline in multi-agent envs (→ 04 §3.4); use RAE / hierarchical GRPO or your own keying.
- Staleness is keyed on the group's open version; within-group straddling of a weight swap is invisible to you (→ 04 §3.11, 03 §7.1).

<a id="a2"></a>
### A2 · Advantages and per-token loss-weight streams

**What you're changing** — the only channels by which credit reaches the trainer:
- `assign_advantages(trace, float | list[float])`: a list must have one value per **sampled token across nodes that sit in ≥1 trainable branch**, in `trace.nodes` order; otherwise `ValueError` (`orchestrator/algo/routing.py:11-35`).
- `assign_reference_logprobs(branch, full_len_values)`: compact per node, first writer wins for shared nodes (`routing.py:38-51`).
- `node.loss_weights["rl"|"ce"|"ref_kl"]`: **full-length** over `node.token_ids`, not compact (`orchestrator/trajectories.py:117-133`; → 04 §4.3).
- `stamp_loss_routing(sample, action_loss_type)` then routes action tokens (`routing.py:68-85`).

**Files to touch** — your algorithm's `score_*` (A1). For shaping without code, GRPO/Echo accept `algo.length_penalty = LinearLengthPenaltyConfig(...)` (`cfg/algorithm.py:96-108, 208`).

**Registration mechanism** — none; you mutate graph nodes in a hook.

**Config wiring** — none (or `length_penalty` on `grpo`/`echo` only; `max_rl` has no such field, → 04 §7.17).

**Lockstep changes** — none: `advantages`, `ref_logprobs`, `rl/ce/ref_kl_weights` already ride `TrainingSample` → `MicroBatch` (`transports/batch/types.py:30-125`).

**How to test/validate** — `tests/unit/orchestrator/test_algorithms.py:138-156` (routing), `:287-291` (nodes only in non-trainable branches stay `None`), `:395-477` (Echo streams).

**Pitfalls**
- Routing semantics differ per `action_loss_type`: `"rl"` writes nothing (absent `rl_weights` = 1.0 on `mask`), `"ce"` **merges** into an existing ce stream, `"ref_kl"` **overwrites** any ref_kl stream you wrote (`routing.py:68-85`; → 04 §3.3).
- A sampled node shared by several branches trains only in the first branch (`orchestrator/trajectories.py:91-114`); credit is per node, not per branch (→ 07 §3.8).
- With `constant_trainer_batch_size=True` (default), zero-advantage `rl` tokens are removed from $N_{rl}$ and zero-signal samples are dropped before counting (`orchestrator/train_sink.py:34-59`; → 04 §3.6, §7.7).
- **First-token invariant:** never put `mask=True` or a nonzero `ce`/`ref_kl` weight on a sample's first token; `ce`/`ref_kl` masks are `weight != 0`, not ANDed with `loss_mask` (`trainer/rl/loss.py:406, 410`; → 04 §5.2).
- Echo-style observation weights depend on the renderer populating `node.is_content`; without it the whole non-sampled span is used (`orchestrator/algo/echo.py:52-70`; → 09 §3.8).

<a id="a3"></a>
### A3 · Custom rl loss

**What you're changing** — the per-token $\ell^{rl}$ only. `ce` and `ref_kl` are fixed functions (`cfg/trainer.py:679`; → 04 §6.2).

**Files to touch** — one function in your own module: `def my_loss(inputs: LossInputs, **kwargs) -> LossOutputs` (`trainer/rl/loss.py:14-54`). No prime-rl edit.

**Registration mechanism** — config import path: `CustomLossConfig{type="custom", import_path, kwargs}` (`cfg/trainer.py:603-613`); `CustomLoss` does `import_object(import_path)` at trainer startup and calls `fn(inputs, **kwargs)` per packed sample (`trainer/rl/loss.py:279-302`).

**Config wiring** — `[trainer.loss] type = "custom"`, `import_path = "pkg.mod.fn"`, `kwargs = {...}`. `trainer.loss` is a union: under it, CLI values are raw strings (→ 02 §7.2).

**Lockstep changes** — the module must import on every torchrun rank (same venv). The dashboard's stable-mask overlay reads `loss.eps` only for `ipo` (`dashboard/server.py:903-908`; cosmetic).

**How to test/validate** — mirror `tests/unit/train/rl/test_loss.py` (e.g. the empty-micro-batch anchor `:256-285`, overlapping components `:288-317`); short `rl` run watching `Mismatch KL` and `loss/*`.

**Pitfalls**
- Return an **unnormalized sum** over member tokens and apply `inputs.loss_weights` yourself (IPO/IcePop do, `loss.py:168-169, 205-206`); normalization is only by the global component token count (→ 04 §3.10, §6.2).
- `inputs.loss_mask` is already the component's member set; `trainer_logprobs` are temperature-scaled; `inference_logprobs` are vLLM **processed** logprobs (post-temperature/top-p/top-k) (→ 06 §3.3).
- Exponentiate the log-ratio **after** masking (IcePop pattern). IPO and `ref_kl` exponentiate before masking, so a single token with $\log r > 88$ NaNs the whole step (→ 04 §7.1).
- Greedy train sampling (`temperature=0`) is allowed by config but the trainer divides by $T$ ⇒ inf/NaN (→ 04 §7.3).
- Metric tensors: 0-d are stacked, 1-d concatenated (`loss.py:32-37`); tensor stats are then trimmed for **all** monitors by `filter_rl_trainer_tensor_stats_for_wandb` (`trainer/utils.py:286-319`; → 11 §3.9.6).
- There is exactly one optimizer step per batch; the ratio numerator is the recompute at $v_{s-1}$, so a PPO-epoch loss has no "old logprob" to read (→ 04 §5.8).

<a id="a4"></a>
### A4 · Using the `ce` and `ref_kl` loss components

**What you're changing** — which tokens feed the two fixed components. $\mathcal L=\sum_c \frac1{N_c}\sum_t w^c_t\ell^c_t$ over exactly `rl`, `ce`, `ref_kl` (`trainer/rl/loss.py:305-414`; → 04 §1). Example: a KL-to-reference regularizer next to RL = keep `action_loss_type="rl"`, write `ref_kl` weights $=\beta$ on action tokens and attach `ref_logprobs` via `assign_reference_logprobs` (→ 04 §6.1).

**Files to touch** — your algorithm (A1/A2). `ref_logprobs` come from a prefill scorer (`orchestrator/clients.py:523-556`).

**Registration mechanism** — `action_loss_type` ClassVar + node `loss_weights` streams (`routing.py:68-85`).

**Config wiring** — none. The `ref_kl` trust region (0.2) and KL coefficient ($10^{-3}$) are hard-coded (`trainer/rl/loss.py:237, 245`; → 04 §7.2).

**Lockstep changes** — a **new** component (a fourth stream) has no seam: it needs `compute_loss`, the global normalizer all-reduce, `MicroBatch`, the packer back-fill and routing (see [no-seam](#no-seam)).

**How to test/validate** — `tests/unit/train/rl/test_loss.py:288-317` (overlapping components add gradients with their own normalizers).

**Pitfalls**
- `ref_kl` raises if `ref_logprobs is None` (`loss.py:216-260`).
- Reference logprobs are prefilled at temperature 1.0 while the trainer scales by the env temperature ⇒ tempered student vs untempered teacher when $T\ne1$ (→ 04 §7.3).
- OPSD's teacher is the live policy at scoring time (not version-pinned) (→ 04 §7.9).

<a id="a5"></a>
### A5 · Curriculum task sampler

**What you're changing** — which task a new train group gets, per env (`TrainSource.next_task` picks the env by `ratio` with `Random(42)`, then `next(curriculum.sampler)`; `orchestrator/train_source.py:20-40`; → 04 §3.13).

**Files to touch**
1. `orchestrator/curriculum/samplers/x.py`: subclass `TaskSampler` (`samplers/base.py:12-32`): `__next__`, `observe(group)`, `state_dict`/`load_state_dict`, `metrics`. Export it from `samplers/__init__.py`.
2. `cfg/orchestrator.py`: config class with `type: Literal["x"]`; add to the `TaskSamplerConfig` union (`:237`).
3. `orchestrator/curriculum/base.py:35-40`: add the `isinstance` branch in `Curriculum.__init__`.

**Registration mechanism** — union + isinstance dispatch. "User-authored" in the docstrings, but there is no import-path hook (→ 04 §6.3).

**Config wiring** — `[[orchestrator.train.source]] curriculum.sampler.type = "x"` (`cfg/orchestrator.py:259-265`).

**Lockstep changes** — checkpointing is automatic: `Curriculum.state_dict` → `TrainSource.state_dict` → `checkpoints/step_N/orchestrator/progress.pt` (`curriculum/base.py:68-72`; `train_source.py:67-85`; → 03 §3.15).

**How to test/validate** — `tests/unit/orchestrator/test_curriculum.py`.

**Pitfalls**
- `observe` is called on every finalized group, including groups with arrived episodes but no trainable survivors (all errored) (`orchestrator/train_sink.py:264-268`; → 04 §3.13). Stale-voided and failure/cancellation-only groups never reach it.
- Finite tasksets must have unique `task.key` (both shipped samplers enforce it, `samplers/standard.py:23-26`, `samplers/pool.py:36-39`).
- The standard sampler's cursor advances at group **open**, so resume skips tasks that were in flight or buffered (→ 03 §7.15). Resume assumes the reloaded taskset has the same order; unpinned HF datasets drift (→ 10 §5.2, §7 G12).
- Infinite tasksets arrive as an iterator; `DifficultyPoolSampler` refuses them (`samplers/pool.py:31-32`).

<a id="a6"></a>
### A6 · Curriculum admission gate

**What you're changing** — per-group admit/reject after scoring. Gates run after `finalize_group` (so they can read advantages) and after `sampler.observe`; all gates are AND-ed (`orchestrator/curriculum/base.py:50-66`; `train_sink.py:264-267`).

**Files to touch**
1. `orchestrator/curriculum/gates/x.py`: subclass `AdmissionGate` (`gates/base.py:10-26`): `admit(group) -> bool`, `state_dict`/`load_state_dict`, `metrics`.
2. `cfg/orchestrator.py:256`: `AdmissionGateConfig` is a plain alias of `AdvRangeGateConfig`; turn it into `Annotated[AdvRangeGateConfig | XGateConfig, Field(discriminator="type")]` (→ 04 §6.3).
3. `orchestrator/curriculum/base.py:42-48`: add the `isinstance` branch.

**Registration mechanism** — (new) union + isinstance dispatch.

**Config wiring** — `curriculum.gates.<name> = {type = "x", ...}` per train source. There is **no gate by default** (`gates = {}`, `cfg/orchestrator.py:263`), so zero-advantage groups train unless you add one (→ 10 §3.7).

**Lockstep changes** — the checkpoint is strict: `Curriculum.load_state_dict` raises if the saved gate names differ from the configured ones (`orchestrator/curriculum/base.py:74-83`). Adding or removing a gate breaks `--resume` of an existing run.

**How to test/validate** — `tests/unit/orchestrator/test_curriculum.py`.

**Pitfalls**
- `admit` must return a real `bool` or `on_result` raises `TypeError` (`curriculum/base.py:59-63`).
- Groups without an advantage stream (opd/opsd/sft) are admitted by the shipped `AdvRangeGate` (`gates/adv.py:15-36`); design yours for that case.
- For survivor-less groups the verdict is computed and ignored (`train_sink.py:264-268`).
- Rejected groups have already paid for reference scoring (`score_episode` runs at arrival; → 04 §7.9).

<a id="a7"></a>
### A7 · Frozen generation source / teacher endpoint

**What you're changing** — who generates a source's train rollouts (`algo.sampling.source = FrozenModelConfig(...)`) or which external model scores them (OPD `algo.teacher`).

**Files to touch** — config only. `GenerationSource.setup` connects a frozen renderer client (`orchestrator/generation_source.py:32-37`); teachers connect in `Algorithm.setup` (`orchestrator/algo/base.py:18-32`).

**Registration mechanism** — config (`FrozenModelConfig`, `cfg/algorithm.py:47-67`).

**Config wiring** — `algo.sampling.source = {name = "...", base_url = "..."}`; only `sft` (`ce`) may sample from a frozen source (`cfg/algorithm.py:179-191, 354-365`). OPD: `algo.teacher = {name, base_url}` (`:308-313`).

**Lockstep changes** — the launcher never starts frozen endpoints; it only logs them (`entrypoints/rl.py:244-257`; → 01 §2.1). A frozen generator must serve `/inference/v1/generate` with `token_id:N` logprobs: `generate()` re-forces `logprobs=1` even though `GenerationSource.sampling_args` pops it (`orchestrator/generation_source.py:43-49`; → 07 §3.5.3, 09 §3.8).

**How to test/validate** — the GPU integration matrix has `reverse_text_rl_opd` and `reverse_text_rl_sft` (→ 11 §3.19).

**Pitfalls**
- A frozen source **reuses the policy's `orchestrator.renderer`** and nothing validates it: an explicit renderer of another family is wrong; `auto` with an unmapped frozen name silently falls back to `DefaultRenderer` (→ 09 §7.1 #3).
- Frozen work has no cache salt and `policy=None` provenance: it never goes stale (`orchestrator/dispatcher.py:559-565, 717-721`).
- Tokenizer identity between frozen model and policy is unchecked (→ 04 §7.10).

<a id="a8"></a>
### A8 · New `TrainingSample` / `MicroBatch` wire field

**What you're changing** — a new per-token (or per-sample) array carried from orchestrator to trainer.

**Files to touch** (in order; → 04 §6.5, 06 §6)
1. `transports/batch/types.py`: **append** to both `TrainingSample` and `MicroBatch`, with a default (`:79-80, 118`).
2. `orchestrator/trajectories.py`: fill it in `trace_to_samples` (`:136-180`).
3. `trainer/batch.py` (runs in the orchestrator): `prepare_sample` incl. truncation (`:370-498`, `:415-441`), `_materialize_bin` back-fill (`STREAM_FILL`, `:14`, `:578-672`), `pad_micro_batch` (`:747-796`), `_make_dummy_batch` (`:847-871`), `_assert_token_arrays_aligned` (`:799-844`).
4. `trainer/rl/data.py`: `TensorMicroBatch` + `_micro_batch_to_tensor` (`:19-62, 202-267`) and `FakeDataLoader` (`:103-167`).
5. Consumer in `trainer/rl/train.py` (CP-shard it with the other per-token tensors if the model consumes it, `:392-412`).
6. Optional: a new per-token annotation stream name in `STREAM_FIELDS` (`monitors/file/traces/update.py:17`) so the dashboard folds it (→ 11 §6).

**Registration mechanism** — positional msgspec `array_like` structs: field order is the wire layout.

**Config wiring** — none.

**Lockstep changes** — orchestrator and trainer must run compatible code. Appending a defaulted field is compatible both ways (old decoders ignore extra trailing elements, new ones default missing ones — executed at msgspec 0.21.1); inserting or reordering is not (→ 06 §3.11). `AnnotationWriter` depends on `trace_ids`, `branch_indices`, `sequence_lengths`, `loss_mask`, `env_names` staying on the wire (→ 11 §5.6).

**How to test/validate** — `tests/unit/orchestrator/test_batch.py:318-393` guards the existing streams across pack boundaries; add your field there.

**Pitfalls**
- Missing one back-fill silently misaligns arrays across pack boundaries (→ 04 §6.5).
- Dummy micro-batches strip streams and identity (`trainer/batch.py:847-871`); padding folds into the last sample (`:769-774`).
- If your field feeds the loss under CP, keep the token counting on the **unsharded** micro-batch or the CP normalization no longer cancels (→ 04 §3.10).
- Samples longer than `seq_len` are silently truncated at ship time (`trainer/batch.py:415-441`; → 03 §7.5).

---

## B. Environments and tasks

<a id="b1"></a>
### B1 · Single-turn verifiable taskset

**What you're changing** — a new env package: `TaskData` row, `Task` behaviour (rewards), `TasksetConfig`, `Taskset.load()` (→ 10 §1, §6.1).

**Files to touch**
1. Scaffold from the repo root: `uv run vf-init my-task -p deps/prime-envs/environments/<group>` (the default `./environments` is **not** a workspace member path; `vf/cli/init.py:205-232`; → 10 §6.0).
2. `my_task/pyproject.toml` (hatchling, `packages = ["my_task"]`, deps `verifiers>=0.3.1`), `README.md`, `my_task/__init__.py` (`__all__ = ["MyTaskset"]`), `my_task/taskset.py`: `MyData(vf.TaskData)`, `MyTaskConfig(vf.TaskConfig)` (every field defaulted), `MyTask(vf.Task[MyData, vf.State, MyTaskConfig])` with async `@vf.reward`/`@vf.metric`/`@vf.stop`, `MyConfig(vf.TasksetConfig)`, `MyTaskset(vf.Taskset[MyTask, MyConfig]).load()` (full code: → 10 §6.1; pattern `deps/verifiers/environments/reverse_text/reverse_text/taskset.py:20-60`).
3. Install: `uv sync --all-extras --all-packages` or `uv sync --package prime-rl --package my-task`; commit the rewritten `uv.lock` (→ 10 §3.8).

**Registration mechanism** — `__all__` export + import name: id `owner/name@v` → module `name` with `-`→`_`, lowercased; built-ins in `verifiers.v1.tasksets` win; exactly one `Taskset` subclass in `__all__` (`vf/utils/loaders.py:70-126`). The package is found only if installed as a uv workspace member (`pyproject.toml:157-173`). There is no registry to edit; `deps/prime-envs/registry.json` is a Harbor dataset registry (→ 10 §1).

**Config wiring**
```toml
[[orchestrator.train.source]]
name = "my-task"
env.taskset.id = "my-task"
env.taskset.dataset_split = "train"      # taskset's own fields; unknown ones are extra_forbidden
env.agent.harness.id = "null"            # default would be bash
env.agent.runtime.type = "subprocess"    # default would be a remote Prime sandbox
```
Per-source knobs: `ratio`, `group_size`, `sampling`, `algo`, `curriculum`, `shuffle` (→ 10 §4.2).

**Lockstep changes** — install the package in **every process that validates the config**: `rl` launcher (on the SLURM submit host too), orchestrator (also `load()`s tasks), env server, `eval` (→ 10 §3.2, §7 G2; 01 §7.21). A plain `uv sync` removes workspace members unless `--inexact`; the Docker build syncs `--locked` (→ 10 §3.8, §7 G3).

**How to test/validate** (→ 10 §6.0)
1. `uv run python -c "import my_task"`; `uv run vf-eval my-task --help` shows typed `--env.taskset.*` fields.
2. `uv run vf-validate my-task --runtime.type subprocess` if you implement `validate()` (default runtime is `prime`).
3. `uv run vf-eval my-task -n 3 -r 1 --no-push --env.agent.harness.id null --env.agent.runtime.type subprocess`.
4. prime-rl `uv run eval my-task …` exercises taskset load, env-server spawn, `TaskData` round-trip and scoring but **not** the training wire: eval uses the chat-completions `EvalClient`, not the renderer/`TrainClient` path (`orchestrator/dispatcher.py:543-552`).
5. `uv run rl @ my.toml --dry-run`, then a short real `rl` run.

**Pitfalls**
- Unpinned seats run the `bash` harness in a **Prime sandbox** (`vf/configs/agent.py:30`; → 10 §7 G1). A harness block without `id` also becomes `bash`, replacing a bundled harness (`vf/configs/agent.py:53-66`; → 08 §7.0).
- `TaskData` must JSON round-trip: the dispatcher checks `(task.key, task.hash)` on every episode and a mismatch **crashes the orchestrator** (`orchestrator/dispatcher.py:116-120, 712`; → 10 §3.5, 03 §7.24). `exclude=True` fields are refused at server start (`vf/serve/server.py:40-51`).
- Reward hooks must be `async`; any exception in scoring becomes a `TaskError` and prime-rl **drops** the trace (not reward 0) (`vf/task.py:203-274`; → 07 §3.4).
- `math_env`'s judge fallback is on by default and calls Prime inference (→ 10 §3.11, §7 G5); disable with `env.taskset.task.judge = "None"`.
- Override `Task.key` with a durable id if resume identity matters (`vf/task.py:150-164`).
- The orchestrator materializes the whole finite taskset in memory, even for small eval `num_examples` (→ 10 §7 G4).

<a id="b2"></a>
### B2 · Multi-turn tool env

**What you're changing** — tools the model calls over MCP, or env-driven user turns.

**Files to touch** (→ 10 §6.2)
- **Per-rollout toolset:** `vf-init my-tool -T` generates `servers/tool.py` (`class MyToolset(vf.Toolset[vf.ToolsetConfig])`, `TOOL_PREFIX`, `@vf.tool` methods) plus `MyTaskConfig.tools` and `Task.toolsets(cls, config)` (`vf/cli/init.py:61-134`). `tools.colocated = true` runs the server inside the harness box.
- **Shared toolset** (one per env-server worker): `vf.SharedToolsetConfig` on the `TasksetConfig` + `Taskset.toolsets`; per-rollout state in a `vf.State` subclass (pattern: `browsecomp_plus`, → 10 §3.13).
- **Remote MCP service:** set `url` on the toolset config.
- **User in the loop:** not a tool — export an `Env` subclass whose `run()` drives `agents.agent.interaction(task)`, or use `--env.id user-sim` (→ 10 §6.2; `vf/agent.py:473-536`).

**Registration mechanism** — `Task.toolsets` / `Taskset.toolsets` classmethods (`vf/task.py:309-320`; `vf/taskset.py:101-111`); `Env` via `__all__`.

**Config wiring** — `env.taskset.task.tools.*` / `env.taskset.tools.*`; `env.agent.harness.id` must support MCP (`null`, `bash`; `SUPPORTS_MCP`, `vf/utils/compile.py:88-94`).

**Lockstep changes** — tool servers reach `/state` and `/task` on the interception server; a remote harness runtime or a remote tool server forces a tunnel (Prime tunnel by default) (`vf/interception/__init__.py:34-55`; → 07 §7.13, 08 §3.5). Heavy indexes must be built in both `load()` (orchestrator) and `setup()` (server) (→ 10 §6.2).

**How to test/validate** — `vf-eval` with the harness you will train; `TOOL_PLACEMENTS`/`SHARED_TOOL_PLACEMENTS` in `deps/verifiers/tests/v1/test_e2e.py:133-268` show the placement matrix.

**Pitfalls**
- `TrainClient` refuses namespaced or non-function tools (`vf/clients/train.py:40-44`) — this is why `codex` is eval-only (→ 08 §3.7).
- Prime sandboxes cannot expose ports; colocate tool servers there (→ 08 §7.12). Restricted-network tool runtimes must be colocated (→ 08 §3.8).
- Bridging needs the uncommitted tail to be `[tool*, user?]`; anything else forces a full re-render and can fork a new branch = a new training sample (→ 07 §7.4).
- Thinking renderers default to `thinking_retention="tool_cycle"`, which breaks extension at **every new user query**; multi-user-turn envs then yield one sample per user turn unless you set `thinking_retention = "all"` (→ 09 §3.6, §7.1 #4).
- Rewards within a phase run concurrently; two rewards mutating the same box race (→ 10 §3.6).

<a id="b3"></a>
### B3 · Sandboxed agentic env (SWE-style)

**What you're changing** — tasks that need a per-task container image, a workdir, resources, network policy and in-box grading (→ 10 §6.3; worked example `swesmith_env`, → 10 §3.12).

**Files to touch**
1. `TaskData`: `image` (pullable ref), `workdir`, `resources=vf.TaskResources(cpu, memory, disk)`, optional `timeout`, `network_allow`/`network_block`, `artifacts` (`vf/task.py:81-122`).
2. `Task`: `NEEDS_CONTAINER = True`; `setup(runtime)` prepares the checkout; `finalize` captures the patch (`capture_patch`); `@vf.reward` runs the tests in `runtime`; `validate(runtime)` applies the gold patch and checks reward 1.
3. Hide grading material from the live agent until scoring (→ 10 §6.3; `deps/prime-envs/environments/swe/README.md` "Grading-material visibility").

**Registration mechanism** — `__all__` (as B1).

**Config wiring** — `env.agent.harness.id = "bash"` (or another trainable harness, [C1](#c1)); runtime defaults to `prime` (`env.agent.runtime.labels = [...]`); bound live sandboxes with `orchestrator.tasks_per_minute` and `orchestrator.concurrency.max_inflight` (`cfg/orchestrator.py:496-497, 573-574`).

**Lockstep changes** — images must live in a registry the runtime can pull (Prime VMs: Docker Hub or the Prime registry; no OCI digests, → 10 §7 G18); Prime credentials in the env-server environment (the key never crosses the wire, only its variable name, → 08 §4.1).

**How to test/validate** — `vf-validate` at scale: no-op runs must fail, gold runs must pass, repeated passes separate flaky infra (→ 10 §6.3 step 4).

**Pitfalls**
- Grader infra failures should **raise** (errored rollout), not score 0 (swesmith test exit > 1, Lean) (→ 10 §7 G14).
- Task network policies are enforced only under a restricted runtime policy; the default `allow=["*"]` makes the cut a no-op on every runtime; subprocess/apptainer reject policy-bearing tasks (→ 08 §3.4, §7.11; ledger R-G).
- Only the agent phase has a default timeout (4 h); setup/finalize/scoring/episode have none, and provisioning steps sit outside every deadline; set `env.timeout.episode` as a watchdog (→ 08 §7.4). Harbor-style tasksets ignore task timeouts by default (→ 10 §7 G15).
- Every rollout gets a fresh box and re-runs `harness.setup` (npm/uv installs); bake the harness into the image (→ 08 §7.7).

<a id="b4"></a>
### B4 · Harbor tasks

**What you're changing** — tasks in Harbor format: `task.toml`, `instruction.md`, `environment/`, `tests/test.sh` writing `/logs/verifier/reward.{json,txt}` (→ 10 §3.10).

**Files to touch**
- **Zero code:** none — `env.taskset.id = "harbor"` with `env.taskset.dataset = "org/name@ref"`, or `registry_path = "deps/prime-envs/registry.json"` + `dataset = "tmax@2026-07-01"` (→ 10 §6.4).
- **Pinned package:** subclass `HarborConfig` with a `Literal` dataset and `class T(HarborTaskset, vf.Taskset[HarborTask, TConfig])` (pattern `terminal_bench_2`, `deps/prime-envs/environments/terminal/terminal_bench_2/terminal_bench_2/taskset.py:14-19`). **Also export `HarborEnv`** if any task uses `[verifier].environment_mode = "separate"`.

**Registration mechanism** — the built-in `vf/tasksets/harbor/__init__.py` exports `HarborTaskset` and `HarborEnv` (an `IsolatedVerifierEnv`), so the bare id gets separate-verifier support; thin wrappers get it only if they export `HarborEnv` too.

**Config wiring** — `HarborConfig` fields: `dataset`, `repo`/`registry_path`/`registry_url`, `ignore_dockerfile`, `require_image`, `ignore_timeouts` (default `True`), `resource_multiplier`, `ignore_separate_verifier` (`vf/tasksets/harbor/taskset.py:63-103`).

**Lockstep changes** — needs `verifiers[harbor]` and the Harbor CLI (→ 10 §3.10). `HarborData.task_dir` is an **orchestrator-host path** read by the env server, so an externally managed env server needs the same cache mounted at the same path (→ 08 §7.10, 10 §7 G10).

**How to test/validate** — `vf-validate` / `vf-eval` on a few tasks; confirm images pull.

**Pitfalls**
- Dockerfile-only tasks are rejected unless `ignore_dockerfile` (then they run unbuilt on the runtime image) (→ 10 §3.10).
- A separate-verifier task under a thin wrapper without `HarborEnv` raises `TaskError` at scoring (`vf/tasksets/harbor/taskset.py:321-328`; → 10 §7 G9).
- `test.sh` exit codes are ignored; a crashed script with no reward file scores 0.0, a **negative example**, not an error (→ 10 §7 G14).
- `tmax`/`general_agent` fetch `registry.json` from `prime-envs@main`, not the pinned submodule, and cache the first download forever (→ 10 §7 G11).

<a id="b5"></a>
### B5 · Isolated verifier, multi-agent and judged envs

**What you're changing** — episode control flow across agent seats, or grading in a separate box.

**Files to touch**
- **Reuse first:** `--env.id best-of-n` / `agentic-judge` / `user-sim` / `isolated-verifier` (→ 10 §6.5; `vf/envs/`).
- **Isolated verifier:** export a subclass of `IsolatedVerifierEnv` (`vf/envs/isolated_verifier/env.py:60-124`: `run`, `verifier_config`, `finalize`, `stage_verifier`, `verify`, `grade`) and implement `Task.stage_verifier` (pattern `LeanEnv`, `deps/prime-envs/packages/lean_common/lean_common/env.py:10-26`; → 10 §6.3 step 3).
- **Custom multi-agent:** `class MyEnv(vf.Env[MyEnvConfig])`; each seat is a `vf.AgentConfig` field **with a default instance**; implement `run(task, agents)`, optionally `setup(agents)` (e.g. `agents.judge.trainable = False`) and `finalize(task, episode)` for cross-trace rewards (`vf/env.py:147-184`; → 07 §6.6, 10 §6.5).

**Registration mechanism** — `__all__` export of one `Env` subclass from the taskset package, or `verifiers.v1.envs.<id>` for `--env.id` (`vf/utils/loaders.py:164-187`).

**Config wiring** — `env.id` (empty = the taskset's own Env, else `SingleAgentEnv`); seats under `env.<role>.*`; `env.max_concurrent_agents` (default 1 serializes agents); `env.verifier.runtime.*` (→ 10 §4.3).

**Lockstep changes** — an algorithm whose credit encodes episode structure must override `validate_env` (`cfg/algorithm.py:279-294`). prime-rl trains every trace whose agent is `trainable` and has tokens (→ 07 §6.6).

**How to test/validate** — `vf-eval` with the target env id; check `trace.agent.trainable` per seat on the file-monitor trace stream.

**Pitfalls**
- A typo'd `Env` export silently falls back to `SingleAgentEnv` (0 matches are swallowed) (→ 10 §7 G8).
- A bare annotation or `default_factory` for a seat raises at class definition (`vf/configs/env.py:125-146`).
- A failed multi-trace episode marks its clean siblings failed: "partial episodes never train" (`orchestrator/envs.py:146-153`).
- GRPO pools all trainable traces of the group across agents (→ 04 §3.4, 07 §9 Q4).

<a id="b6"></a>
### B6 · Rewards, judges, stops and interceptors without new packages

**What you're changing** — scoring or turn policing on an existing taskset.

**Files to touch** — none, or a single `file.py`.

**Registration mechanism** — config plug: `env.taskset.task.rewards.<name> = {fn = "file.py:func", weight = ...}`, likewise `.metrics`, `.stops`; fn-less entries override metadata only (`vf/configs/task.py:11-63`; `vf/utils/loaders.py:236-271`; → 07 §6.2). Judges: `env.taskset.task.judges = [{id = "reference" | "rubric" | <pkg>, ...}]`; a plugin judge is a package exporting one `Judge` subclass (`vf/configs/judge.py:26-61`; → 07 §6.3). Task-class hooks: `@vf.intercept` (typed `Request` or `Response`), `@vf.stop` on `Request/Response/Trace` (→ 07 §6.4).

**Config wiring** — as above; judge client fields `model` (default `openai/gpt-5.6-luna`), `base_url`, `api_key_var` (default `PRIME_API_KEY`) (`vf/configs/judge.py:14-24`).

**Lockstep changes** — judges run in the **env-server worker**, which must have the key in its environment (→ 10 §3.11).

**How to test/validate** — `vf-eval` and inspect `trace.rewards`/`metrics` in the trace stream.

**Pitfalls**
- Scoring runs in three sequential phases (metrics → rewards → judges), concurrent within a phase (→ 07 §3.4).
- A missing judge key resolves to `"EMPTY"` → 401 → the trace **fails** (dropped), not 0; judge scoring timeout defaults to none, so a hung judge hangs its rollout (→ 07 §3.4; ledger R-F).
- **A `Response`-typed `@vf.intercept` rewrite drops the turn's tokens**: the turn commits as a tokenless "sampled" node, its tokens/routing never train and the next turn cannot bridge (confirmed by execution; `vf/interception/server.py:818-828`; → 07 §7.14).
- Tool-call verdicts before execution need a harness with `SUPPORTS_TOOL_INTERCEPTION`; otherwise a rewrite ends the rollout (`vf/session.py:469-472`).

---

## C. Agent execution substrate (verifiers, renderers)

<a id="c1"></a>
### C1 · Harness / agent scaffold

**What you're changing** — the agent program run inside a runtime, pointed at the interception endpoint (→ 08 §3.7, §6.2; 07 §6.5).

**Files to touch**
1. `vf/harnesses/<id_with_underscores>/{__init__.py, harness.py}` — or any installed top-level package named after the id.
2. `class MyConfig(HarnessConfig)` (pin versions with `PinnedVersion`) and `class MyHarness(Harness[MyConfig])`; the generic parameter is how `--env.agent.harness.*` flags get narrowed (`vf/utils/loaders.py:282-284`).
3. Capability flags, conservatively: `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_MCP`, `SUPPORTS_TOOL_INTERCEPTION`, `SUPPORTS_RESUME`, `EXECUTES_CODE`, `SUPPORTS_SKILLS`, `NEEDS_CONTAINER` (default True) (`vf/harness.py:34-59`).
4. One style:
   - *Process:* `setup(runtime)` installs (`ensure_installed(...)` or `runtime.prepare_uv_script(PEP723_source)`); `launch(ctx, trace, runtime, endpoint, secret, mcp_urls, data)` → `await runtime.run_program(argv, env)`; `launch_chat_program` reuses the bundled loop (`vf/harnesses/utils/launch.py:35-86`).
   - *ACP:* subclass `ACPHarness`, call `super().setup(runtime)` last, implement `prepare_acp(...) -> ACPConfig`; optional `gate_tools`, `acp_turn_result`, `acp_close_result` (`vf/acp/__init__.py:64-153`). Requires a runtime with `open_process`.
   - *In-process loop:* allowed if every model call goes to `endpoint` + `secret`; return a synthetic `ProgramResult` (`vf/harness.py:302-306`).
5. `cleanup(trace, runtime)`: remove per-trace dirs keyed by `trace.id` (needed on borrowed/shared boxes).

**Registration mechanism** — `__all__` with **exactly one** `Harness` subclass; id → `verifiers.v1.harnesses.<module>` else top-level `<module>` (`vf/utils/loaders.py:70-126`). A taskset package exporting a `Harness` makes it that taskset's default (`loaders.py:153-161`).

**Config wiring** — `env.agent.harness.id = "<id>"` plus its fields; common fields `env`, `forward_env`, `mcp_header_env`, `tool_timeout=600`, `disabled_tools`, `skills` (`vf/configs/harness.py:26-67`).

**Lockstep changes** — the module must import in every process that parses the env config (launcher, orchestrator, env server), because `AgentConfig._resolve_harness` narrows it there (`vf/configs/agent.py:53-66`). Harness configs ride the wire as extra-allow `WireHarnessConfig`, so no orchestrator-side union edit is needed (→ 08 §6.1 step 3).

**Trainability constraint (training = chat-completions only).** `TrainClient` refuses any non-`ChatDialect` request and namespaced/non-function tools (`vf/clients/train.py:40-44, 342-350`); chat also requires `n=1` (`vf/dialects/chat.py:512-514`). Final table (ledger R-G; → 08 §3.7):

| Trainable | Eval-only | Undetermined |
|---|---|---|
| `null`, `bash`, `browser_use`, `mini_swe_agent`, `terminus_2`, `prime_agent`, `rlm` (12 shipped configs), `pi`/`kimi_code` at `transport="chat_completions"`, `hermes_agent` [UNVERIFIED: inferred; Hermes default outside the pin] | `codex` (namespaced MCP tools), `claude_code` (Anthropic Messages) | `openclaw` (depends on the model-name prefix) |

If your program can switch APIs, branch on `ctx.client.type == "train"` like `hermes_agent` (`vf/harnesses/hermes_agent/harness.py:88-89`).

**How to test/validate** — add rows to `CHAT_PLACEMENTS` / `AGENTIC_PLACEMENTS` / `ACP_RESUME_PLACEMENTS` (`deps/verifiers/tests/v1/test_e2e.py:133-227`); `pair()` applies `getattr(pytest.mark, <id>)` (`:17-19`), so **register the mark** in `deps/verifiers/pyproject.toml:221-243` or collection fails under `--strict-markers` (`:216`). Then a short real `rl` run — the only path that exercises the train dialect (→ 10 §6.0 step 7).

**Pitfalls**
- A dialect refusal is **not** a config error: each turn returns a retryable 502 and the cause survives only in `trace.calls[*].error` (`vf/interception/server.py:862-870`; → 07 §7.1).
- Install everything in `setup`; `launch` runs after the (policy-dependent) egress cut (→ 08 §3.4).
- Every model call must go through `endpoint`; disable telemetry, auto-update and utility/title models (→ 08 §5.5).
- Re-send assistant turns verbatim including `reasoning_content`, or the graph forks every turn (→ 07 §7.5).
- The program's own sampling fields are ignored in training (→ 07 §7.3).
- Don't pass secrets via env vars that model-driven tools inherit; never retry `run_program` (forks the trace) (→ 08 §6.2 step 8).
- Anthropic-style SDKs need `/v1` stripped from `endpoint` (→ 08 §3.7).

<a id="c2"></a>
### C2 · Sandbox runtime provider

**What you're changing** — "where commands run" for a rollout (→ 08 §3.6, §6.1).

**Files to touch** (in order)
1. `vf/runtimes/<name>.py`: `class XConfig(NetworkPolicyConfig)` (or `BaseConfig` if you can't enforce egress) with `type: Literal["x"]`, `image`, `workdir`, and resource fields named **exactly** `cpu`, `memory` (GB), `gpu`, `disk` (task resources map by name, `vf/utils/compile.py:57-72`); `class XRuntimeInfo(XConfig, BaseRuntimeInfo)`; a validator rejecting policies you can't enforce.
2. `class XRuntime(Runtime)`: `is_local` ClassVar; abstract `start()`, `run(argv, env) -> ProgramResult`, `_read(path, max_bytes)`, `write(path, data)`; sync idempotent `cleanup()`; optional `open_process` (required by all 8 ACP harnesses), `run_background`, `expose`/`published_port`, `host_url`, `prepare_setup`/`prepare_execution(routes)` (`vf/runtimes/base.py:134-410`).
3. Register in `vf/runtimes/__init__.py`: add to the `RuntimeConfig` and `RuntimeInfo` unions, to the `_runtime_cls` map, and to `__all__` (`:41-70, 104-134`); optionally re-export from `vf/__init__.py:75-86`.
4. Lockstep edits in verifiers: `cap_remote_agent_timeout` if the provider has a lifetime limit (`vf/utils/compile.py:115-134`); `vf/rollout.py:107-116` for a "no network" flag; `vf/mcp/launch.py:242-246, 322-345` (assumes `workdir`, branches on `runtime.type == "subprocess"`); `vf/agent.py:86-140` (borrowed-box placement checks).

**Registration mechanism** — hard-coded unions + a dict map; there is no entry-point or plugin loader for runtimes anywhere in verifiers (→ 08 §6.1 step 3). This is a fork of `deps/verifiers`.

**Config wiring** — `env.agent.runtime.type = "x"` plus its fields; precedence CLI/TOML non-default > task data > default; task network policy **intersects** the runtime's (`vf/configs/runtime.py:88-126`; → 08 §4.3).

**Lockstep changes** — **the runtime unions are on the wire.** `Rollout.open` stamps `trace.agent.runtime = runtime.info` (`vf/rollout.py:197`), and the orchestrator re-validates `AgentInfo.runtime: RuntimeInfo | None` (`vf/trace.py:112`) and `agent.config.runtime` when it parses each `WireEpisode`. An orchestrator whose verifiers lacks your type fails **every episode** (executed; → 08 §6.1, ledger R-G). The orchestrator and env servers must run the same verifiers.

**How to test/validate** — add placement rows (`CHAT_PLACEMENTS`, `AGENTIC_PLACEMENTS`, `USER_RUNTIMES`, `ACP_RESUME_PLACEMENTS`, `TOOL_PLACEMENTS`, `TOOL_STATE_PLACEMENTS`, `SHARED_TOOL_PLACEMENTS`, `deps/verifiers/tests/v1/test_e2e.py:133-268`) and register the mark in `deps/verifiers/pyproject.toml:221-243`; fixtures `tool_runtime`, `run_v1_server` (`deps/verifiers/tests/v1/conftest.py:65-72, 208-260`).

**Pitfalls**
- Keep the provider SDK import **lazy** inside methods (Modal does); a module-level import (Prime's) becomes a dependency of every process importing `verifiers.v1` (→ 08 §6.1 step 3).
- `start()` and `prepare_execution()` run under no rollout deadline — bound your own waits (→ 08 §6.1 step 6).
- Wrap the create call in `run_shielded` and capture the id inside it, or cancellation leaks paid sandboxes (`vf/runtimes/modal.py:201-209`, `prime.py:197-215`).
- `run()` returns non-zero exits as data; raise `SandboxError` only for infra failure; `_read` must raise on a missing file and cap at the source.
- Clear image ENTRYPOINTs; `/tmp` may be a tiny tmpfs on VMs; a live process you can't signal must fail the box closed (→ 08 §6.1 step 6).
- `is_local=False` forces the interception through a tunnel, and the elastic interception pool always mints **Prime** tunnels (`vf/interception/pool.py:123-125`); bring-your-own tunnels need `interception.type = server | static` (→ 08 §3.5; see [C6](#c6)).

<a id="c3"></a>
### C3 · Dialect (harness wire format)

**What you're changing** — a new model-API wire format the interception server accepts (today: chat, Responses, Anthropic Messages) (→ 07 §3.5.1, §6.7).

**Files to touch**
1. Implement `Dialect[RespT]` (`vf/dialects/base.py:266-387`): `routes`, `upstream_path`, `response_type`, `sampling_fields`, `parse_request`, `parse_sampling`, `parse_response`, `validate_response`, `apply_overrides`, `rewrite_request/response`, `stream_events`, `stream_parser`, `mediate_external_capabilities`; optional `auth_headers`, `secret`, `error_body`, `stream_keepalive`, `stream_error`, `is_terminal_event`.
2. Append an instance to `DIALECTS` (`vf/dialects/__init__.py:11`).
3. To **train** with it: a renderer path for the dialect in `TrainClient.get_response` (`vf/clients/train.py:342-350`) and in renderers.

**Registration mechanism** — a module-level tuple; the interception server registers one POST handler per `dialect.routes` and picks the dialect from the **route the SDK posted to** (`vf/interception/server.py:423-459`). Source edit in verifiers.

**Config wiring** — none; harnesses declare nothing.

**Lockstep changes** — works with the `EvalClient` immediately; training needs the TrainClient + renderer change above (→ 07 §6.7; 08 §8).

**How to test/validate** — no dialect-specific unit test is cited in the sections; the e2e harness placements (`deps/verifiers/tests/v1/test_e2e.py:133-268`) exercise dialects through real harnesses.

**Pitfalls**
- `apply_overrides` policy differs per dialect (chat keeps program sampling unless overridden; Responses/Anthropic drop some fields) and the base docstring misdescribes chat (→ 07 §3.5.2 step 1, §7.16).
- Streaming is framing only: training generates whole, eval buffers the provider stream; past 60 s a 200 SSE with keepalives is committed and late errors become SSE errors (→ 07 §7.6).
- The per-rollout secret carrier is dialect-specific (`Authorization: Bearer` vs Anthropic `x-api-key`) (→ 07 §3.5.1).

<a id="c4"></a>
### C4 · Client (how intercepted calls reach a model)

**What you're changing** — the object that turns an intercepted request into a `Response` (today: `EvalClient` relays; `TrainClient` renders to tokens and calls `/inference/v1/generate`) (→ 07 §3.5.3-3.5.4, §6.8).

**Files to touch**
1. Subclass `Client` (`vf/clients/client.py:30-71`): `get_response(dialect, body, sampling, session_id, turn, headers) -> Response`; optional `relay`, `relay_aux`, `close`.
2. Extend `resolve_client` (`vf/clients/client.py:73-80`) and the `ClientConfig` union (`vf/configs/client.py:88`).
3. prime-rl: `setup_client` (`orchestrator/clients.py:259-280`) and the `InferenceClient(train_client_type=..., eval_client_type=...)` construction (`orchestrator/orchestrator.py:199-205`) choose client configs by string.

**Registration mechanism** — isinstance dispatch + union. Source edits in verifiers and prime-rl.

**Config wiring** — no user-facing knob: prime-rl hard-selects `"renderer"` for train and chat-completions for eval (`orchestrator/orchestrator.py:199-205`).

**Lockstep changes** — the `ClientConfig` union crosses the env-server wire inside `RunRequest.client` (→ 07 §4.1, 08 §4.1), so orchestrator and env servers need the same verifiers. The interception server builds one client per distinct client-config JSON (`vf/interception/server.py:356-366`). To train, `Response.tokens` must carry `prompt_ids/completion_ids/completion_logprobs/message_spans` consistent with the graph invariants (→ 07 §3.7, §6.8).

**How to test/validate** — `deps/verifiers/tests/v1/test_graph.py` / `test_trace.py` for token-exactness invariants (→ 07 header).

**Pitfalls**
- No client-side retries anywhere (`MAX_RETRIES = 0`) (→ 07 §7.8).
- Error typing matters: a `RolloutError` is stashed on `session.error` and returned with its status; any other exception becomes an unstashed 502 that SDKs retry (→ 07 §3.5.2 step 11; ledger R-H).
- The API key never crosses the wire; the worker resolves `$<api_key_var>` from its own env (→ 08 §4.1).

<a id="c5"></a>
### C5 · Renderer (model-family chat format)

**What you're changing** — the token-exact conversation encoder for a model family: `render`, `parse_response`, `get_stop_token_ids`, `bridge_to_next_turn` (→ 09 §3.2, §6.1).

**Files to touch** (in `deps/renderers`)
1. Study the reference encoder (Jinja template / Python encoder / Harmony): role wrappers, atomic specials, system+tools block, tool-call format, tool-response envelope, history-reasoning rule, gen prompt per kwarg (→ 09 §6.1 step 1).
2. `rnd/configs.py`: `class FooRendererConfig(BaseRendererConfig)` with `name: Literal["foo"]`; classify every field in `_template_fields` or `_internal_fields` (unclassified ⇒ `TypeError` at import, `:95-110`); add a `_reject_thinking_retention_conflict` validator if a template knob sets retention (`:30-50`); add to the `RendererConfig` union (`:997-1031`), `_CONFIG_BY_NAME` (`:1046-1077`) and `__all__`.
3. `rnd/foo.py`: `class FooRenderer` (copy the `Qwen3Renderer` skeleton, `rnd/qwen3.py:126-462`); `parse_foo` in `rnd/parsing.py` if needed.
4. Register: `RENDERER_REGISTRY` in `_populate_registry` (`rnd/base.py:1316-1383`), `_LAZY_RENDERERS` (`rnd/__init__.py:87-117`), exact HF ids in `MODEL_RENDERER_MAP` (`rnd/base.py:930-1047`), `MULTIMODAL_MODELS` for VLMs (`:1059-1094`); `TRUSTED_REVISIONS` / `TOKENIZER_SOURCE_OVERRIDES` if needed (`:1171-1186`).
5. Multimodal: `mm_token_type_id_map`, `previous_multi_modal_data`/`process_multimodal` kwargs, and a branch in `_build_mm_features` (`rnd/client.py:488-505`).
6. Bump the `deps/renderers` submodule in prime-rl.

**Registration mechanism** — registry dict + `RendererConfig` discriminated union on `name` + exact-name auto map.

**Config wiring** — `[orchestrator.renderer] name = "foo"` (+ template fields, `thinking_retention`) (`cfg/orchestrator.py:535`); OPSD hint block `algo.renderer` (`cfg/algorithm.py:338-342`); SFT `[renderer]` (`cfg/sft.py:201`). With `name = "auto"`, prime-rl's validator requires `tokenizer.name or model.name` ∈ `MODEL_RENDERER_MAP` (`cfg/orchestrator.py:698-730`).

**Lockstep changes** — the renderer runs in the **env-server worker**, built from `model.name` (`TrainClientConfig.renderer_model_name`, `orchestrator/clients.py:80-91`); the orchestrator's `[tokenizer]` and `chat_template` overrides never reach it (confirmed, ledger R-H; → 09 §7.1 #2). Renderer, engine and trainer tokenizers must be identical, unchecked (→ 09 §5.6). The engine must stop *on* the renderer's stop ids (the client forces them) and not return trailing scaffold (→ 09 §5.5). Multimodal features need `vllm`+`torch` importable in the client (env-server) process (→ 09 §3.9).

**How to test/validate** (→ 09 §6.1 steps 5-6) — add a `_model(...)` entry to `MODEL_CATALOG` (`deps/renderers/tests/parity.py:89-271`) and values for every template field in `KWARG_VALUES` (`:673-708`; enforced by `deps/renderers/tests/test_parity.py:108-135`); run the parity matrix (`test_parity.py:191-222`), `test_roundtrip.py`, `test_bridge.py`, `test_multimodal.py`. Non-Jinja references need an oracle in `REFERENCE_ORACLES` / `RENDERER_ORACLE_ROUTES` (`deps/renderers/tests/reference_rendering.py:605-622`).

**Pitfalls**
- The bridge contract: return `None` whenever the exact-prefix property can't be proven (any assistant in `new_messages`, retention demands a re-render); nothing the bridge emits is sampled (→ 09 §3.6, §5.1, §5.4).
- Use `trim_to_turn_close` with a synthesized close; GLM/Hy3/Laguna hand-roll it and have divergence bugs (→ 09 §7.1 #17).
- Exact-match auto-detection: local paths and fine-tunes fall back to `DefaultRenderer` (no masks, no bridge, no tools without a `tool_parser`); frozen sources and OPSD are not validated (→ 09 §7.1 #1, #3).
- Encoding is not thread-safe; the verifiers pool serializes per slot (→ 09 §5.11).
- Strict logprob validation: one `-9999.0` fails the turn as a plain `ValueError` → 502 (→ 09 §7.1 #8).
- Multimodal RL works only for Qwen3-VL, the Qwen3.5 family and Gemma 4, and verifiers never bridges prompts with images (→ 09 §3.9, §7.1 #12).

<a id="c6"></a>
### C6 · Tunnel

**What you're changing** — how a remote sandbox reaches the interception server (default Prime `frpc` tunnel) (→ 08 §6.3).

**Files to touch** — subclass `Tunnel[Config]` (`bind_host`, `bind_port`, async-context `expose(port) -> url`, `vf/interception/tunnel/base.py:24-47`); add the config to `TunnelConfig` and a branch in `make_tunnel` (`vf/interception/tunnel/__init__.py:10-19`).

**Registration mechanism** — union + isinstance; `make_tunnel` returns `PrimeTunnel` for anything that is not `CustomTunnelConfig` (`tunnel/__init__.py:15-19`), so an unhandled new type silently becomes a Prime tunnel.

**Config wiring** — `env.interception = {type = "server", tunnel = {type = "x", ...}}` or `static`.

**Lockstep changes** — the default elastic pool hard-codes `PrimeTunnelConfig()` (`vf/interception/pool.py:123-125`) and host-local tool servers for remote harnesses are bridged with `PrimeTunnel()` (`vf/mcp/launch.py:381-384`); a custom tunnel only works with `interception.type = server | static` and colocated tools (→ 08 §3.5).

**How to test/validate** — run a remote-runtime placement from `deps/verifiers/tests/v1/test_e2e.py`.

**Pitfalls** — `CustomTunnel` binds `0.0.0.0:<port>` protected only by the per-rollout secret (→ 07 §3.6); Prime tunnel starts are capped at 512/min per `VF_RUN_ID` scope (→ 07 §7.2).

---

## D. Trainer engine

<a id="d1"></a>
### D1 · Model family in the trainer

**What you're changing** — a custom ("prime") implementation of an architecture: modeling, HF↔prime conversion, CP support, fp32 wire keys, and weight-sync compatibility (→ 05 §6.1).

**Files to touch** (in order)
1. `trainer/models/<arch>/`: `configuration_<arch>.py` (only if transformers' config is unsuitable), `converting_<arch>.py` (`conversion_chain(config) -> list[ConvOp]` + `is_hf_state_dict`/`is_prime_state_dict` helpers), `modeling_<arch>.py`, `__init__.py` exporting config + `*ForCausalLM` (mirror `qwen3_moe/` or `glm4_moe/`).
2. **Modeling contract** on `*PreTrainedModel(PreTrainedModelPrimeRL)` (`trainer/models/base.py:21-156`): `is_hf_state_dict`, `is_prime_state_dict`, `conversion_chain`, `init_buffers_post_meta`; and where relevant `cp_support` (`:33-40`), `keep_in_fp32_for_weight_transfer` (`:43`), `convert_adapter_to_hf` (`:132`).
   - `*Model.forward(input_ids, position_ids, inputs_embeds, [routed_experts], *, seq_lens, seq_lens_are_pre_shard)`; build `cu_seqlens` with `get_cu_seqlens_from_seq_lens` (`trainer/models/qwen3/modeling_qwen3.py:153-194`); slice `routed_experts[:, :, layer_idx, :]` per MoE layer (`qwen3_moe/modeling_qwen3_moe.py:212-220`).
   - Layout the engine walks: `model.model.layers`, `model.model.embed_tokens`, `model.model.norm`, bias-free `model.lm_head` `nn.Linear`, MoE blocks at `layer.mlp` as `layers.moe.MoE` (`trainer/model.py:127-128, 538-540`).
   - Reuse blocks: `ATTN_IMPL2CLASS[...]` (CP patching + `qkv` fusion for free), `MoE.from_args(MoEArgs(...))` (dispatchers, EP, grouped GEMM, `gate_up`, router replay), `FeedForward`, `RMSNorm`, `RotaryEmbedding`.
3. **Conversion ops** (`trainer/models/conversion_ops.py:47-348`): `Rename`, `PrefixRename`, `Drop`, `Stack`, `SplitConcat`, `MapValue`, `SqueezeLeading`, `Conditional`, `Sequence`; `routed_experts_op` for per-expert ↔ stacked `[E,H,D]`. Every op must be present-guarded and layer-local (→ 05 §3.17.5, §5.8).
4. **CP support:** declare `cp_support(config)` (validated at build, `trainer/model.py:415-423`); models needing CP-awareness beyond attention read `self.cp_context` (→ 05 §3.11).
5. **fp32 wire keys:** return True from `keep_in_fp32_for_weight_transfer(name)` for precision-sensitive tensors (examples: GLM `mlp.router.selection_bias`, NemotronH `mamba.A_log`/`mamba.D`, Qwen3.5 `linear_attn.A_log`; → 05 §3.17.1).
6. **Register** in `trainer/models/__init__.py`: `AutoConfig.register("<type>", Config, exist_ok=True)` (`:35-48`), add `(Config, ForCausalLM)` to `_CUSTOM_CAUSAL_LM_MODELS` (`:51-70`); VLMs also `_CUSTOM_VLM_MAPPING` (`:106-109`) and `VLM_REGISTRY` (`utils/vlm.py:27-33`).
7. **Mini preset:** an `ARCH_PRESETS` entry in `scripts/mini_moe.py:78-195`.

**Registration mechanism** — `AutoConfig.register` + a registry tuple of `(ConfigClass, ModelClass)`; `impl="auto"` picks custom iff `type(model_config)` is in the mapping (`trainer/models/__init__.py:91-100`), so the config class `AutoConfig` returns must be the registered one.

**Config wiring** — `trainer.model.name`, `trainer.model.impl = "auto" | "custom"`; `cp`, `cp_style`, `ep`, `moe.*`, `fusions.*` (`cfg/trainer.py:350-425`). Validators: CP ⇒ flash attention + custom/auto; VLM ⇒ custom; VLM+CP ⇒ ulysses (→ 05 §4.3).

**Lockstep changes**
- **Inference parity:** vLLM must serve the architecture and `load_weights` the exported HF names (FS/NCCL export always converts to HF layout, bf16 except declared fp32 keys) (→ 05 §5.9, §6.1 step 4). Add a vLLM patch if the loader needs one (pattern: `inference/vllm/gpt_oss_weight_loading.py`; [E4](#e4)).
- **NIXL** (only if you want it): custom model required (`transports/weights/nixl/nixl.py:329`); the prime→HF chain runs **inside the vLLM worker** as `LazyWeight` view ops, so it may use only `SUPPORTED_OPS` (narrow/select/view/reshape/getitem/unsqueeze/squeeze/transpose/t/permute/flatten/contiguous/chunk/split/unbind/to/float/bfloat16) on bf16/fp32 (`transports/weights/nixl/graph.py:35-55`); `torch.cat` (Qwen3.5-MoE `SplitConcat` backward) and `new_empty` (GPT-OSS) are not supported; shards must be contiguous dim-0 and every served state-dict tensor must **alias** its live parameter (→ 05 §7, 06 §3.9).
- **FP8 ignore patterns** must match between trainer and inference (`cfg/trainer.py:155-166`).
- A renderer for the family ([C5](#c5)) and router replay needs vLLM returning routed experts.

**How to test/validate**
- `uv run python scripts/mini_moe.py --arch <arch> --output-dir ./mini-<arch>` builds a random ~0.5B model and checks HF-vs-prime logits (`max diff < 0.1`) and an exact HF→prime→HF round-trip (`scripts/mini_moe.py:252-310`) — **but the script does not import at the pin**: it imports `prime_rl.trainer.models.qwen3_5_moe` (`scripts/mini_moe.py:31`), which no longer exists (verified; ledger R-D). Fix that import first.
- Unit tests to mirror: HF-vs-prime forward/backward parity with the prime LM head (`tests/unit/train/models/test_qwen3_moe.py:22-130`), conversion round-trip (`test_moe_conversions.py:60-93`), fp32 wire keys (`tests/unit/utils/test_weights.py:28-43`).
- Integration: `uv run rl @ configs/ci/integration/reverse-text-moe/start.toml --model.name <mini> --trainer.model.impl custom` (→ 05 §6.1 step 5).
- **Merge bar:** a KL-mismatch table over 20 steps on `math`, `batch_size=64`, every entry < 0.015 (`docs/development.md:133-140`; the same doc names stale methods `convert_hf_layer_to_tt`, → 05 §7).

**Pitfalls**
- `is_*_state_dict` must be discriminative on **one layer's keys and on the non-layer group**, because NCCL converts per group (`trainer/models/deepseek_v4/modeling_deepseek_v4.py:139-146`; → 05 §3.17.1). An op needing keys from two layers silently no-ops on NCCL/FS paths (→ 05 §5.8).
- `inject_prime_lm_head` **replaces `model.forward`**; your `*ForCausalLM.forward` is dead in training and the inner `*Model.forward` must accept `seq_lens`, `seq_lens_are_pre_shard`, `routed_experts` and mm kwargs (`trainer/models/layers/lm_head.py:316, 319-354`; → 05 §3.3).
- `init_buffers_post_meta` must rebuild every non-persistent buffer and run before `dcp_load` (→ 05 §5.7). Freeze params **before** optimizer creation; strict DCP resume requires identical `requires_grad` sets (→ 05 §5.5).
- HF-impl models never get `seq_lens`; packing boundaries reach them only via `position_ids` (→ 05 §3.3).
- LoRA on MoE experts under EP raises `ImportError` (`TOKEN_GROUP_ALIGN_SIZE_M`, `trainer/models/layers/lora/multi_moe.py:316-331`) and LoRA experts bypass FP8/MXFP8 (confirmed, ledger R-D; → 05 §7).
- `ep="auto"` = `min(world/dp_replicate, 8)` with no divisor search: world 12 → bare `AssertionError` (`trainer/parallel_dims.py:75, 310-313`; → 05 §7).
- MoE `selection_bias` is never updated during training (→ 05 §3.10).
- NIXL + the default `qkv` fusion serves **stale q/k/v forever** on multi-GPU trainers; disable fusions for NIXL (confirmed; → 05 §7, 06 §3.9).

<a id="d2"></a>
### D2 · Optimizer and LR scheduler

**What you're changing** — the optimizer or schedule (→ 05 §3.7, §6.2).

**Files to touch**
- Optimizer: a `*Config(BaseOptimizerConfig)` with a `type` literal in the `OptimizerConfig` union (`cfg/trainer.py:541-543`); a `case` in `_create_optimizer` (`trainer/optim/__init__.py:124-148`). Full offload additionally needs a native/torch CPU step (`trainer/optim/offload.py:1200-1280`) and the validator list (`cfg/trainer.py:746-750`).
- Scheduler: config class + union (`cfg/trainer.py:439-468`), `case` in `setup_scheduler` (`trainer/scheduler.py:100-121`), and `validate_scheduler` (`cfg/trainer.py:471-490`).

**Registration mechanism** — union + `match`.

**Config wiring** — `[trainer.optim] type = "x"`; `[trainer.scheduler] type = "x"`. Both are unions: raw CLI values below them (→ 02 §7.2).

**Lockstep changes** — the optimizer receives only `requires_grad` params (`trainer/optim/__init__.py:119-123`); optimizer state must survive DCP save/load including packed-parameter splitting (`trainer/models/fusions.py` `split_packed_optimizer_state_for_checkpoint`, → 05 §3.14, §3.16).

**How to test/validate** — `tests/unit/train/test_state_offload.py:79-99` (exact parity pattern); `benchmarks.yaml` matrix (→ 11 §3.19).

**Pitfalls**
- Default `optim_cpu_offload=True` disables fused AdamW and round-trips state every step (→ 05 §7).
- DeepEP and full offload force `max_norm=None` (→ 05 §3.7).
- Muon needs its NCCL warm-up (multi-node deadlock workaround) and is incompatible with `fsdp_cpu_offload` (→ 05 §3.7).
- The scheduler is stepped once per trainer step on the base optimizer (`trainer/scheduler.py:10-14, 98`).

<a id="d3"></a>
### D3 · MoE compute backend and token dispatcher

**What you're changing** — grouped-GEMM kernels for experts, or how tokens move across EP ranks (→ 05 §3.10, §6.2).

**Files to touch**
- Compute: implement the `GroupedGemm` protocol (`token_group_alignment`, `__call__(x, weight_t, *, offs)`), a config class in `MoEComputeConfig`, a branch in `_resolve_grouped_gemm` (`trainer/moe_runtime.py:34-56`).
- Dispatcher: subclass `TokenDispatcherBase` (`dispatch`/`combine`/optional `run`/`synchronize`, `trainer/distributed/token_dispatcher.py:37-74`), config in `MoEDispatchConfig`, a branch in `configure_moe_runtime` (`trainer/moe_runtime.py:92-124`).

**Registration mechanism** — union + isinstance chain.

**Config wiring** — `[trainer.model.moe.compute] type = ...`, `apply_to`; `[trainer.model.moe.dispatch] type = ...` (`cfg/trainer.py:187-211, 412-425`).

**Lockstep changes** — non-replayable or stateful ops must be added to the activation-checkpoint **mandatory save** set, and expensive ones to the selective-save targets (`trainer/activation_checkpointing.py:31-38, 49-95`).

**How to test/validate** — `tests/unit/train/models/test_qwen3_moe.py`, `tests/integration/test_reverse_text_moe.py`; benchmark matrix.

**Pitfalls** — LoRA expert wrappers call `torch._grouped_mm` directly and ignore the configured backend (→ 05 §7); MXFP8 compute ✗ DeepEP; `transport="mxfp8"` needs MXFP8 compute (→ 05 §3.10).

<a id="d4"></a>
### D4 · Runtime fusion and CP style

**What you're changing** — packing logical parameters into one physical parameter, or a new context-parallel attention scheme (→ 05 §3.11, §3.14, §6.2).

**Files to touch**
- Fusion: a function `(module) -> None` that builds the packed `nn.Parameter` and calls `register_packed_parameter_state_dict_hooks(module, PackedParameterSpec(...))`; advertise it in the module's `supported_fusions`; add the name to `FusionsConfig.enabled`'s literal (`trainer/models/fusions.py:47-118`; `cfg/trainer.py:93`).
- CP style: a `substitute_*` that rebinds `FlashAttention._compute_attention` (and AFMoE/GPT-OSS classes), a params publisher called from `setup_cp_attention_params` (`utils/cp.py:142-164`), add to `CPStyle`/`ALL_CP_STYLES` (`trainer/models/base.py:9-10`) and the config literal.

**Registration mechanism** — fusion: module attribute + config literal; CP: **class-level monkeypatch** of attention (`utils/cp.py:40-70`).

**Config wiring** — `trainer.model.fusions.enabled = [...]`; `trainer.model.cp`, `cp_style`.

**Lockstep changes** — checkpoints and exports must stay canonical (state-dict hooks split packed tensors; → 05 §5.4). The orchestrator pads micro-batches to `pad_to_multiple_of = cp` (auto-filled only by the `rl` launcher, `cfg/rl.py:689`; → 04 §3.7).

**How to test/validate** — `tests/unit/train/models/test_fusions.py:39-54` (canonical round-trip).

**Pitfalls**
- A fused split along the FSDP shard dim yields **copies**, not views; any consumer that captures `state_dict()` once (NIXL) serves stale weights (→ 05 §3.14, §7).
- Fusions are skipped under LoRA (`trainer/model.py:1000-1001`).
- CP token counting must stay pre-shard (→ 04 §3.10); CP chunks are contiguous, unbalanced (→ 05 §7).

<a id="d5"></a>
### D5 · Extra checkpoint state

**What you're changing** — what a trainer checkpoint saves and restores.

**Files to touch** — extend `AppState.state_dict/load_state_dict` (`trainer/ckpt.py:79-160`). DCP handles DTensors; plain Python objects are pickled into metadata (→ 05 §6.2).

**Registration mechanism** — direct edit.

**Config wiring** — none; RL honours only `ckpt.skip_optimizer` (`trainer/ckpt.py:168`; → 05 §3.16).

**Lockstep changes** — orchestrator-side state has no general seam: only `Progress` and curriculum `state_dict` are saved (`orchestrator/ckpt.py:1-47`; → 03 §3.15).

**How to test/validate** — `tests/integration/test_reverse_text.py` resumes a run and converts the final DCP checkpoint (→ 11 §3.19).

**Pitfalls** — no completeness marker; "latest" = highest `step_*` name, possibly orchestrator-only (→ 05 §3.16, 01 §3.10); `maybe_clean` deletes whole `step_N` dirs including the orchestrator's (→ 05 §7).

---

## E. Transports and inference

<a id="e1"></a>
### E1 · Weight transport

**What you're changing** — how trainer weights reach the vLLM engines. The 4-marker handshake (`.sender_ready` → `.receiver_ready` → `.started` → `.finished` in `broadcasts/step_N/`) stays; only the bytes change (→ 06 §3.6, §6; 05 §3.17; 03 §6).

**Files to touch** (in order)
1. **Config (four places):** trainer union (`cfg/trainer.py:665-668`), orchestrator union (`cfg/orchestrator.py:479-482`), shared `rl` union + `auto_setup_weight_broadcast` (`cfg/rl.py:169-172, 439-495`), and the inference `WeightBroadcastConfig.type` literal (`cfg/inference.py:217-221`).
2. **Trainer sender:** subclass `WeightSender`, implement `_broadcast(model, step, step_dir)`; it runs on **all** ranks and must hold non-masters until the master finished the handshake (`transports/weights/base.py:53-99`); reuse `gather_weights_parallel` / `resolve_dtensors` + `convert_*_to_hf` + the fp32-key wire-dtype rule (`utils/weights.py:94-199`). Dispatch in `setup_weight_sender` (`transports/weights/__init__.py:22-35`).
3. **Consumer receiver:** subclass `WeightReceiver` with `initialize()` and `receive(step)`; ack exactly once; drive engines via `admin_plane.update_weights(..., on_paused=...)`. Dispatch in `setup_weight_receiver` (`transports/weights/__init__.py:38-51`).
4. **Inference worker:** a worker-extension class with `init_broadcaster(...)`, `update_weights_from_path(arg)`, `liveness_probe()`; add it to `WORKER_EXTENSION_CLS` (`inference/vllm/server.py:59-63`). Reuse `load_weights_checkpoint_layerwise` for any iterator of `(hf_name, tensor)` (`inference/vllm/worker/weight_transfer.py:12-23`).
5. **Launch wiring:** SLURM host/port injection (`templates/multi_node_rl.sbatch.j2`) and template vars in `entrypoints/rl.py:479-578`.

**Registration mechanism** — discriminated unions (config) + if-chain factories + a string→class-path dict for the vLLM worker extension (`args.worker_extension_cls`, `inference/vllm/server.py:232`).

**Config wiring** — `[weight_broadcast] type = "x"` (shared; `type` must be spelled, `default_factory`/`None` unions get no injected discriminator — → 02 §7.3). The `rl` default is NCCL unless LoRA or no `[inference]` (then filesystem) (`cfg/rl.py:447-451`). Disable form: `--weight-broadcast None`, not `--no-weight-broadcast` (→ 02 §3.3).

**Lockstep changes** — trainer, orchestrator and inference must agree on the type (`cfgutils/validation.py:306-318`). The SFT online-eval process is also a consumer (`eval/online.py:58-69`). LoRA forces filesystem (`cfg/rl.py:452-457`). Multi-node rendezvous hosts come from the sbatch template (→ 06 §4.8).

**How to test/validate** — `tests/unit/transports/test_nccl_broadcast.py` (skipped at the pin, → 05 §7); integration `reverse_text` (includes resume). Probe restart behaviour explicitly.

**Pitfalls**
- **Retries:** `_admin_post` retries `/update_weights` on timeout/5xx per engine (720 s/attempt, 1440 s total). For any collective transport a retry queues behind the running RPC and waits for a broadcast that never comes (NCCL/NIXL unsafe; filesystem safe). Make your apply idempotent or follow Dynamo's single-POST, terminal-on-failure pattern (`inference/dynamo.py:309-327, 352-384`; → 06 §3.6.1d; ledger R-E).
- The trainer blocks on every `.receiver_ready`; skipping a version strands it (→ 06 §5.4). `receive` must return only after **all** engines applied it (→ 06 §3.6).
- The prefix cache is never reset on update; `/pause` is hard-wired to `mode="keep", clear_cache=False` (freeze) (`inference/vllm/server.py:66-70`; → 06 §3.6.1a-c).
- `inference_world_size` is config arithmetic, and the single-node value is computed **before** DP auto-fill (bug; → 06 §3.8, 01 §7.5); rank layout `1 + i·W/n + device.index` assumes uniform GPUs per admin URL (→ 06 §5.1).
- `AdminPlane.initialize_nccl` swallows non-404 errors and has no timeout (`orchestrator/clients.py:192-196`).
- The FS receiver waits for `.finished` with no timeout, and dispatch is blocked for the whole export (→ 03 §7.23, 06 §7).
- NCCL group is one-shot: an inference restart needs a trainer restart (→ 06 §7).

<a id="e2"></a>
### E2 · Batch (rollout) transport

**What you're changing** — the orchestrator → trainer pipe for `list[list[MicroBatch]]` grids (today ZMQ PUB/SUB or files) (→ 06 §3.11, §6; 03 §6).

**Files to touch**
1. Implement `BatchSender.send(grid)` and `BatchReceiver.{wait, can_receive, receive}` (`transports/batch/base.py`).
2. Add a config variant to `TransportConfig` (`cfg/shared.py:255`) and branches in `setup_batch_sender` / `setup_batch_receiver` (`transports/batch/__init__.py:21-40`).
3. Multi-node host injection gated in `entrypoints/rl.py` (`use_zmq_transport`, `:531, 569`) and the sbatch template.

**Registration mechanism** — union + if-chain.

**Config wiring** — shared `[rollout_transport] type = "x"`; propagated verbatim to trainer and orchestrator, types must match (`cfg/rl.py:497-517`).

**Lockstep changes** — the sender asserts `len(grid) == num_train_workers` and equal row lengths (`transports/batch/filesystem.py:20-22`, `zmq.py:68-70`). Every trainer rank (CP peers too) builds a receiver for `dp_rank = rank // (world/dp)` (`trainer/rl/data.py:186-193`). Both sides start counting at their own `progress.step` (→ 04 §3.8).

**How to test/validate** — `tests/unit/orchestrator/test_batch.py`; an integration run with a resume.

**Pitfalls**
- Nothing on the wire says which step a batch is; orchestrator-only restart is unsupported (→ 03 §3.12, 00 §5). A new transport should carry step/version (→ 03 §8).
- ZMQ PUB drops past HWM; it is safe only because `TARGET_LAG=1` bounds unconsumed batches to 2 (→ 06 §3.11).
- READY is sent once and deduped by `dp_rank`, so it doesn't prove every CP peer subscribed (masked by startup order) (→ 06 §3.11).
- `num_train_workers` too large ⇒ ZMQ blocks forever / FS files nobody reads; too small ⇒ ranks starve (→ 04 §5.4).
- `--rollout-transport None` fails (→ 02 §3.3).

<a id="e3"></a>
### E3 · Admin plane backend (the Dynamo pattern)

**What you're changing** — how the orchestrator discovers engines and drives `/pause`, `/update_weights`, `/resume`, `/init_broadcaster`, `/load_lora_adapter` (→ 06 §3.4, §3.10).

**Files to touch** — subclass `AdminPlane` (`orchestrator/clients.py:132-236`) like `DynamoAdminPlane` (`inference/dynamo.py:154`); select it in `setup_admin_plane` (`orchestrator/clients.py:239-245`); add a config block on `ClientConfig` like `dynamo: DynamoConfig | None` (`cfg/shared.py:165-196`).

**Registration mechanism** — an if-branch in the factory keyed on a config block.

**Config wiring** — `orchestrator.model.client.<your_block>`; Dynamo: `client.dynamo = {discovery_url?}` (defaults to `base_url` with port+1).

**Lockstep changes** — every weight receiver calls `update_weights` / `initialize_nccl` / `load_lora_adapter` on it (→ 03 §6); `EvalRunner` builds an admin plane too (`eval/runner.py:77-166`); the metrics collector scrapes `/metrics` on the admin clients (→ 11 §3.9.5).

**How to test/validate** — `tests/unit/inference/test_dynamo.py`, `tests/unit/orchestrator/test_clients.py`.

**Pitfalls**
- **Client order is load-bearing**: it must equal GPU rank order for NCCL/NIXL `rank_offset` and the metrics roles (`orchestrator/clients.py:137-140`).
- Dynamo pins topology only after two identical snapshots, re-verifies before every mutation, and marks the plane terminal on any failed update; it requires `inference_world_size == 1` (→ 06 §3.10).
- `finish_sessions` (sticky-routing release) is a no-op unless `admin_base_url` is set (→ 03 §7.26).

<a id="e4"></a>
### E4 · Inference route, vLLM patch, or generate-response change

**What you're changing** — the vLLM server as prime-rl extends it in-process (→ 06 §3.1-3.3, §6).

**Files to touch**
- **Route:** append to `router` in `inference/vllm/server.py`; `custom_build_app` mounts it and `run_api_server_worker_proc` is wrapped so every API-server child re-imports the module (`inference/vllm/server.py:180-206`). Worker-side logic = a new worker-extension method called via `engine_client.collective_rpc(...)`.
- **Patch, pick the process scope:** every vLLM process incl. spawned workers → `apply_shared_vllm_patches` (`inference/patches.py:4-25`, entry point `pyproject.toml:50-51`); API server only → at `inference/vllm/server.py` import (`:28-43`); worker extension import → `inference/vllm/worker/__init__.py:11-18`. Make it idempotent with a `_prime_rl_*` marker (`inference/patches.py:66-67, 115-116`).
- **Generate response:** `PrimeRlServingTokens.serve_tokens_full_generator` (`inference/vllm/serving_tokens.py:58-87`), then the parser in `rnd/client.py` and verifiers `TrainClient` (→ 06 §6).

**Registration mechanism** — FastAPI router + `vllm.general_plugins` entry point + monkeypatches of vLLM app factories.

**Config wiring** — vLLM args need nothing: `[inference.vllm] foo = ...` passes through (`VllmConfig` is `extra="allow"`); type a field only if prime-rl reads it, and add it to `_OMIT_IF_NONE` if vLLM rejects `None` (`cfg/inference.py:607-611`; → 02 §6 step 8).

**Lockstep changes** — any response-shape change must be parsed by `rnd/client.py` and the verifiers TrainClient; routed experts must stay in the `{data, shape, start, dtype}` object form because the router merges prefill/decode objects (→ 06 §3.12).

**How to test/validate** — `tests/unit/inference/test_serving_tokens.py`.

**Pitfalls**
- **Plugin failures are fatal, not silent.** Only the import of `prime_rl.inference.patches` is guarded by vLLM; each patch's lazy vLLM import runs unguarded, so a renamed vLLM symbol aborts startup of every vLLM process. The docstring at `inference/patches.py:8-10` overstates the silent case (ledger R-E; → 06 §3.2, §7). `VLLM_PLUGINS` can exclude the plugin.
- Patches target private vLLM symbols and copies of 0.24 code; the pin is v0.29.0 (`pyproject.toml:59, 276-278`; → 06 §7).
- Under `--inference.vllm.*`, the `--x=v` CLI form is silently stored as an extra key `"x=v": True`; use the space form (→ 02 §7.2).
- Sampling-mask capture is engine-wide: eval requests without `top_k > 0` or with `temperature 0` are rejected (→ 06 §3.3).

<a id="e5"></a>
### E5 · Router backend

**What you're changing** — the single data-plane URL in front of engines (vllm-router fork or llm-d) (→ 06 §3.12, §6).

**Files to touch** — a new `RouterConfig` variant (`cfg/inference.py:367-368`), a launch branch in `templates/_launch_router.sh.j2`, and `start_router` for local runs (`entrypoints/inference.py:165-191`; it asserts `vllm-router` at `:167`).

**Registration mechanism** — union + template branch + local launcher branch.

**Config wiring** — `[inference.router] type = "x"` (must spell `type`; → 02 §7.3); `--inference.router None` disables.

**Lockstep changes** — admin traffic must keep bypassing the router (→ 06 §1); sticky sessions rely on `X-Session-ID` and the fork's `/finish_session` (→ 06 §3.5).

**How to test/validate** — a single-node `rl` run and a SLURM multi-node dry-run render (`launcher/rl.sbatch`).

**Pitfalls** — llm-d is rejected for single-node and for routed-expert return (→ 01 §7.13); the router's Prometheus port is `server.port + 21000` (→ 06 §4.5).

---

## F. Orchestrator policies

<a id="f1"></a>
### F1 · Concurrency policy

**What you're changing** — the in-flight episode cap and overload shedding (today an AIMD controller fed by vLLM `/metrics`) (→ 03 §3.7).

**Files to touch** — replace `ConcurrencyController` (`orchestrator/concurrency.py:116-338`). Interface: `max_inflight` attribute, `bind(set_limit, get_inflight, on_overload)` (`:148`), `record_episode(env, kind, tokens, duration)` (`:166`), `observe(list[EngineLoadSample])` (`:189`), `gauges()` (`:331`). Construction sites: `orchestrator/orchestrator.py:335` (bound at `:352-356`) and `EvalRunner` (`eval/runner.py:77-166`).

**Registration mechanism** — none: the class is instantiated directly. This is a core edit.

**Config wiring** — `[orchestrator.concurrency] initial_inflight, min_inflight=1, max_inflight=1024` (`cfg/orchestrator.py:485-506`).

**Lockstep changes** — `InferenceMetricsCollector(on_load=concurrency.observe)` feeds it (`orchestrator/orchestrator.py:359-368`); the dispatcher provides `set_limit` and `cancel_inflight`.

**How to test/validate** — there are no unit tests for `ConcurrencyController` (→ 03 §7.21); watch `concurrency/*` wall-time gauges.

**Pitfalls**
- No `/metrics` ⇒ the cap stays at `min_inflight` (1) forever in RL; `EvalRunner` fails fast instead (→ 03 §7.6).
- `max_inflight` also sizes `out_q` (`max(8, max_inflight)`), which is one condition of the ship-gate **deadlock** (→ 03 §7.2).
- Overload cancellation is train-only and cancelled groups still train with fewer members (→ 03 §7.7).
- Usage-derived token counts are lower bounds (→ 07 §7.17).

<a id="f2"></a>
### F2 · Weight-swap observers and pipeline hooks

**What you're changing** — code that runs around each policy update or episode completion.

**Files to touch** — `VersionObserver` (`on_version_pending` before engines pause, `on_new_version` after `Policy` is mutated; `orchestrator/types.py:163-172`) added to `WeightWatcher(observers=[...])`; `watcher.on_update(hook)`; `Dispatcher(on_episode_complete=...)` — all wired in `Orchestrator.setup` (`orchestrator/orchestrator.py:337-343, 380-388`).

**Registration mechanism** — callbacks passed in `setup` (core edit).

**Config wiring** — none.

**Lockstep changes** — none outside the orchestrator.

**How to test/validate** — no unit tests exist for the watcher/dispatcher (→ 03 §7.21); a probe against the real classes with a fake receiver is how the deadlock was reproduced (ledger R-B).

**Pitfalls**
- Observer exceptions are swallowed; hook exceptions and `receive` failures kill the watcher (→ 03 §3.8).
- An observer that **blocks** blocks the whole update; `drop_group` → bounded `out_q.put` inside `on_version_pending` is the deadlock mechanism (→ 03 §7.2).
- `policy.version` is set only after `receiver.receive` returns (`orchestrator/watcher.py:113-116`).

---

## G. Observability

<a id="g1"></a>
### G1 · Monitor backend

**What you're changing** — a new sink for metrics/episodes (e.g. MLflow) (→ 11 §6).

**Files to touch**
1. Subclass `Monitor` (`monitors/base.py:16-75`): `log_metrics(metrics, step)` (handle `step=None` wall-time rows), `log_episodes(episodes, step, kind, subset)`; optional `init(**kwargs)`, `log_annotations`, `log_live`, `log_eval_plan`, `log_eval_epoch`, `finalize`.
2. Config class in `cfg/monitors.py` and a field on `MonitorsConfig` (`:56`); for `rl` propagation also `SharedMonitorsConfig` (`cfg/rl.py:94-104`) and `propagate`/`presence_targets` lines (`cfgutils/validation.py:86-112, 166-179`).
3. Extend the `monitors.setup` signature and its construction block (`monitors/__init__.py:48-100`), and update **every** call site: orchestrator (`orchestrator/orchestrator.py:208`), RL trainer (`trainer/rl/train.py:86`), SFT trainer (`trainer/sft/train.py:81`), eval (`eval/eval.py:33`), online eval (`eval/online.py:47`).

**Registration mechanism** — a hand-written if-chain; there is no registry (→ 11 §6).

**Config wiring** — `[monitors.x]` (shared) or `[orchestrator.monitors.x]` / `[trainer.monitors.x]`.

**Lockstep changes** — only rank 0 registers monitors (`RANK`/`DP_RANK`, `monitors/__init__.py:67-69`).

**How to test/validate** — `tests/unit/orchestrator/test_metrics.py` for the dict shapes you will receive.

**Pitfalls**
- A configured monitor whose `init` raises crashes the process; later logging failures only warn (→ 11 §1).
- Fan-out is sequential: a slow backend stalls the orchestrator loop; push blocking I/O to a thread (→ 11 §3.1).
- Train episodes arrive twice by subset (`all` at arrival, `effective` at ship) (→ 11 §3.1).
- `finalize` runs only on a clean exit (→ 11 §5.9). NaN/inf are sanitized by file/prime but not W&B (→ 11 §7.5).
- Shared `[monitors.file]` propagates only `path` (→ 11 §4.5, ledger R-J).

<a id="g2"></a>
### G2 · Metrics

**What you're changing** — a new scalar, gauge or episode statistic (→ 11 §6, 03 §6).

**Files to touch**
- Step-keyed orchestrator scalars: the `finalize_train_batch` dict (`orchestrator/orchestrator.py:643-710`).
- Episode statistics: `TraceMetrics` / `EpisodeMetrics.to_wandb` (`orchestrator/metrics.py:191-349`). Env-specific values need nothing: `trace.metrics`/`trace.rewards` flow to `.../<agent>/metrics/<name>` and `rewards/<name>` automatically.
- Wall-time gauges: a `gauges()` method consumed by `collect_pipeline_view` (`orchestrator/orchestrator.py:801-862`).
- Trainer rows: dicts at `trainer/rl/train.py:619-697`; tensor stats go through `filter_rl_trainer_tensor_stats_for_wandb` (`trainer/utils.py:286-319`).
- Prometheus: add a `Gauge` in `MetricsServer.__init__`, a parameter in `update`, and the call (`utils/metrics_server.py:88-105, 151-176`; `trainer/rl/train.py:700-711`).
- Curated views: `monitors/wandb/overview.py` constants **and** `dashboard/static/app.js` (they mirror each other, `overview.py:3-5`).

**Registration mechanism** — edit dicts / methods.

**Config wiring** — none; `orchestrator.collect_inference_metrics` gates `inference/*` rows only.

**Lockstep changes** — dashboard/W&B view constants if you want it on curated panels.

**How to test/validate** — `tests/unit/orchestrator/test_metrics.py`; `metrics.jsonl` rows (merge by `(step, producer)`).

**Pitfalls**
- A W&B top-level prefix is either step-keyed or time-keyed: the first `step=None` row re-axes the whole prefix (→ 11 §5.7).
- Orchestrator and trainer share keys like `time/step` in one W&B run and one `metrics.jsonl` (→ 11 §7.1); namespace new keys by producer.
- Quality stats belong on `effective`; `has_error`/`error/<type>` exist only on `all`; empty sets read 0 (→ 11 §7.2, §7.5).
- Console `SUCCESS` lines are a CI contract; don't reformat them (→ 11 §7.15).

<a id="g3"></a>
### G3 · Eval driver and annotation producer

**What you're changing** — a new evaluation entrypoint, or new post-hoc per-trace facts in the trace stream (→ 11 §6).

**Files to touch**
- Eval driver: reuse `EvalRunner(config, run_dir)` → `setup()` → `start()` → `eval_source.trigger(step)` → `run_epoch(fired, step, superseding_step=...)` → `drain()`/`stop()` (`eval/runner.py`); the config must satisfy `ServedEvalConfig` plus `model`, `log.interval` (→ 11 §6).
- Annotation producer: `make_update(trace_id, info=..., branches={idx: {...}})` + `monitors.log_annotations`; a new per-token stream name must be added to `STREAM_FIELDS` (`monitors/file/traces/update.py:17`).

**Registration mechanism** — reuse of classes; tuple extension for stream names.

**Config wiring** — eval: `EvalConfig` / `SFTOnlineEvalConfig` fields (`cfg/eval.py:13-167`).

**Lockstep changes** — the dashboard folds only `STREAM_FIELDS`; streams must be full-length over the branch token prefix with nulls (→ 11 §5.6).

**How to test/validate** — `tests/unit/eval/test_resume.py`, `test_cli.py`; `tests/integration/test_gsm8k_eval.py` (needs `PRIME_API_KEY`).

**Pitfalls**
- Eval always uses the chat-completions `EvalClient`, never the renderer path (`orchestrator/dispatcher.py:543-552`; → 07 §7.11).
- RL in-orchestrator eval epochs span weight updates; `policy_version` is a lower bound (→ 11 §3.9.3).
- Eval adaptivity needs vLLM `/metrics`; against external APIs pin the concurrency band (→ 11 §7.18).

---

## H. Config and launch

<a id="h1"></a>
### H1 · New config field, end to end

**What you're changing** — a typed knob and its path to every process that needs it (→ 02 §6).

**Files to touch** (in order)
1. **Declare** on the smallest class that owns it in `cfg/<component>.py`, with a type, default and a PEP-224 docstring right below (it becomes `--help`). `X | None = None` for optional sub-blocks; a discriminated union with `type: Literal[...]` for variants; `Field(ge=...)` for bounds. Incompatibility checks go in a `model_validator(mode="after")` on the smallest class that sees all inputs.
2. **Component-local field:** done; read `config.<field>` in the consumer — it is dumped into that component's JSON automatically.
3. **Field that must agree across components:** add it to the shared block on `RLConfig` (`cfg/rl.py:57-307`), one `propagate("shared.path", "trainer.…", "orchestrator.…")` line (`cfgutils/validation.py:67-141`), a `validate_shared_*` check if sub-configs can still diverge (`cfgutils/validation.py:194-318`), and `presence_targets` if it is a bare enable block (`:166-179`).
4. **Topology-derived field** (GPU counts, hosts, world sizes): an after-validator on `RLConfig` placed after every validator whose output it reads (order table → 02 §3.6), honouring explicit values via `"field" not in obj.model_fields_set`, idempotent on re-parse.
5. **Launcher-only field:** put it on `RLConfig`; if it must live on a sub-config a child parses, add it to the dump exclusion set (`entrypoints/rl.py:101`, `entrypoints/inference.py:141`, `entrypoints/sft.py:121`).
6. **Known only inside the SLURM allocation:** a template variable in `write_slurm_script` (`entrypoints/rl.py:479-578`) injected as a CLI override on the child's command line in `templates/multi_node_rl.sbatch.j2` (pattern `--weight_broadcast.host $MASTER_ADDR`, `:563-566`).
7. **Env-var knob:** prefer `[<component>.env_vars]`; never a `PROTECTED_ENV_VARS` name (`cfg/shared.py:13-24`).
8. **vLLM argument:** nothing to add ([E4](#e4)).

**Registration mechanism** — pydantic field in the vendored `pydantic-config` `cli()` (defaults ⊂ `@` files ⊂ CLI, validated once on the merged dict; `deps/pydantic-config/src/pydantic_config/cli.py:1592-1730`).

**Config wiring** — shared values are **fill-if-absent defaults, never stompers**; a value written in two places must agree or all conflicts are raised together (`cfgutils/validation.py:10-65`; → 02 §3.5). Only values the user actually wrote propagate; shared-block defaults do not (→ 02 §7.5).

**Lockstep changes** — the resolved JSON is the inter-process contract: every validator must be a fixed point on its own output, because `model_fields_set` is larger in the child (→ 02 §5.1). After-validator order is semantic (→ 02 §5.3).

**How to test/validate** — extend `tests/unit/test_configs.py` (round-trip, propagation, conflict); every checked-in TOML under `configs/`, `examples/`, `k8s/` must parse with some config class (`tests/unit/test_configs.py:52-70`); `rl --dry-run` and diff `configs/attempt_*/resolved/`; the slim `prime-rl-configs` wheel must import without torch/vllm/etc. (CPU CI, → 11 §3.19).

**Pitfalls**
- Below an `Optional[BaseModel]` or union node the CLI is untyped: lists must be JSON, bare flags become `True`, `--x=v` is not split (→ 02 §7.2).
- `default_factory`- or `None`-defaulted unions never get their `type` injected; users must spell `type` (→ 02 §7.3).
- `"None"` becomes `None` for every field, strings included (→ 02 §7.4).
- The launcher mutates sub-configs after validation without `validate_assignment`; the child's view can differ (e.g. `max_lora_rank` 20 → 32) (→ 02 §5.2).
- Nested `--x @ file` overrides root `@` files regardless of order (→ 02 §7.1).
- Renames get **no aliases** by policy; a transitional remap is a `mode="before"` validator on the raw key (→ 02 §6 step 10).

<a id="h2"></a>
### H2 · Launch target: custom SLURM template

**What you're changing** — the sbatch script for single- or multi-node runs (→ 01 §3.3-3.4, §6).

**Files to touch** — your `x.sbatch.j2`; copy the bundled includes (`_launch_rank.sh.j2`, `_launch_router.sh.j2`, `_mooncake_store.sh.j2`, `llmd/*`) next to it; add template variables in `write_slurm_script` if needed (`entrypoints/rl.py:479-578`).

**Registration mechanism** — config path `slurm.template_path` (default auto-filled to the bundled template, `cfg/rl.py:857-867`); Jinja2 `FileSystemLoader` rooted at the template's own directory (`entrypoints/rl.py:433-434`).

**Config wiring** — `--slurm.template-path path/to/x.sbatch.j2` (`cfg/shared.py:88-89`); `[slurm]` fields `partition`, `account`, `time`, `pre_run_command`, `cleanup_grace_period=3600`, `shared_fs`, `launch_modelexpress` (→ 01 §4.4).

**Lockstep changes** — multi-node children receive topology only as CLI overrides you inject (`--model.client.base-url $INFER_URLS`, `--model.client.admin-base-url $ADMIN_URLS`, `--weight_broadcast.host`, `--rollout_transport.host $ORCH_ADDR`); env servers must run on the orchestrator's node (loopback bind) (→ 01 §3.4.6, §3.7). `ADMIN_URLS` order must be GPU rank order (→ 00 §5).

**How to test/validate** — `rl … --dry-run` renders `launcher/rl.sbatch` without submitting (but see the dry-run side effects in [How to use](#how-to-use-this-doc)).

**Pitfalls**
- Custom `rl` templates don't see bundled includes (the `sft` launcher adds the bundled dir; `rl` does not) (→ 01 §7.11).
- The template assumes `SLURM_PROCID == HOSTNAMES[i]` (→ 01 §5.3).
- Any non-zero exit holds the allocation for `cleanup_grace_period` (3600 s) (→ 01 §7.3).
- Single-node SLURM re-parses `rl.json` in the allocation, so validator results can differ from local launches (it heals the NCCL world-size bug) (→ 01 §3.3).
- Re-submitting the same `launcher/rl.sbatch` reuses the attempt dir and can read stale env-server `.address` files (→ 01 §7.18).

<a id="h3"></a>
### H3 · Launch target: k8s, externally managed components, or a new process

**What you're changing** — running components outside the `rl` launcher, or adding a component the launcher spawns (→ 01 §6, §7.16, §8).

**Files to touch**
- **Externally managed env servers:** set `serve.address` per source; the launcher stops spawning it and the orchestrator connects directly (`entrypoints/rl.py:53-61`; `cfg/orchestrator.py:163-164`).
- **External inference:** omit `[inference]` and point `orchestrator.model.client.base_url` / `admin_base_url` (or `dynamo`) at it; set LoRA / router-replay / sampling-mask flags on that server yourself (`cfg/rl.py:563-567, 582-585, 627-634`; → 01 §6).
- **New component process (local):** its config on `RLConfig`, a writer in `write_subconfigs` (`entrypoints/rl.py:89-113`), a `Popen` + `monitor_process` thread in `rl_local` (pattern `:259-292`), entries in `rl_config_components`/`format_log_message`; for multi-node a role block in `multi_node_rl.sbatch.j2` passing its address as CLI overrides. Keep it out of the completion condition unless it must finish (→ 01 §6).
- **k8s:** `k8s/prime-rl/` Helm chart (3 StatefulSets + RWX PVC); image override via `PRIME_RL_REF`, `PRIME_RL_REPO`, `VERIFIERS_VERSION` (`scripts/docker-entrypoint.sh:16-73`).

**Registration mechanism** — `serve.address` / omitted `[inference]` (config); launcher code edits for a new process; the chart is a pod scaffold only.

**Config wiring** — per-source `serve.address = "tcp://host:port"`; `orchestrator.model.client.*`; `rollout_transport.host` / `weight_broadcast.host` for cross-pod links.

**Lockstep changes** — outside the `rl` launcher nothing auto-fills `num_train_workers` / `pad_to_multiple_of`, partitions GPUs, spawns env servers or supervises (→ 01 §7.16). An external env server still requires the env package in the orchestrator (it validates and `load()`s) (→ 10 §7 G2) and the same host paths for Harbor tasks (→ 08 §7.10).

**How to test/validate** — parse each TOML with its component class (`tests/unit/test_configs.py:52-70` only checks that *some* class parses it); run each component by hand with `<cmd> @ resolved.json`.

**Pitfalls** — the shipped k8s example is broken (`infer.toml [model]` rejected; `--client.base-url` invalid; ZMQ on `localhost` across pods; comma-joined admin URLs at the router port; no env-server workload) (→ 01 §7.16); `uv run trainer` is a single rank without torchrun (→ 01 §7.16); env servers have no liveness supervision and runs have no wire timeout (→ 08 §7.2, §7.4).

---

<a id="no-seam"></a>
## Things with NO seam (fork or edit core)

These are hard-coded. Changing them means editing prime-rl core or forking a submodule; plan for it.

| What | Where it is fixed | What you must edit | Cite |
|---|---|---|---|
| Orchestrator pipeline shape (sources → dispatcher → `out_q` → sinks → packer → sender) | hand-wired in `Orchestrator.setup` / `main_loop`; no registry for sources, sinks or dispatcher | `orchestrator/orchestrator.py:184-401, 495-762` | → 03 §6, §8 |
| Async depth `TARGET_LAG = 1` (ship gate and dispatch gate) | module constant; only `max_off_policy_steps` is configurable | `orchestrator/orchestrator.py:95, 607-625, 976-1000` | → 03 §3.9, 04 §7.15 |
| Other orchestrator constants: `STARTUP_WEIGHT_WAIT_TIMEOUT_S=1200`, `SHUTDOWN_TIMEOUT_S=300`, watcher poll 1 s, metrics poll 5 s, admission window 5 s, env startup 600 s, shuffle/mixing seed 42, all controller constants | constants | `orchestrator/orchestrator.py:90-100`; `orchestrator/concurrency.py:42-89`; `orchestrator/envs.py:42, 47` | → 03 §4.6 |
| Admin HTTP timeouts and retry policy (`ADMIN_TIMEOUT_S=300`, `UPDATE_WEIGHTS_TIMEOUT_S=720`, retry on 5xx/timeout) | constants in `_admin_post` | `orchestrator/clients.py:370-412` | → 06 §3.4 |
| `/pause` semantics (`mode="keep"`, `clear_cache=False`; query params ignored) and no prefix-cache reset on update | hard-wired route | `inference/vllm/server.py:66-70` | → 06 §3.6.1 |
| Cache salt granularity (per group open, not per dispatch/turn) | dispatcher | `orchestrator/dispatcher.py:532, 559-565` | → 03 §7.1 |
| Batch wire has no step/version; staleness is orchestrator-only | positional `MicroBatch` | `transports/batch/types.py:90-125` | → 03 §8, 04 §8 |
| Weight-version discovery via filesystem markers (for every transport) | `WeightSender`/`WeightReceiver` base | `transports/weights/base.py:17-169` | → 06 §8 |
| Loss component set (`rl`, `ce`, `ref_kl`), fixed `ce`/`ref_kl` functions, `ref_kl` constants 0.2 / $10^{-3}$ | `compute_loss` | `trainer/rl/loss.py:236-246, 305-414` | → 04 §6.2, §7.2 |
| One optimizer step per batch; synchronous loop (batch → step → blocking broadcast → blocking ckpt) | RL train loop | `trainer/rl/train.py:253-719` | → 05 §3.5, 04 §5.8 |
| Algorithm state checkpointing | `Algorithm` has no `state_dict`; orchestrator ckpt saves `progress` + `train_source` only | `orchestrator/ckpt.py:1-47`; `orchestrator/algo/base.py:45-80` | → 03 §7.15 |
| Out-of-tree plugins for algorithms, samplers, gates, transports, monitors, admin planes | registry dict / isinstance / if-chains, no import-path hook | `orchestrator/algo/__init__.py:43-53`; `orchestrator/curriculum/base.py:35-48`; `transports/*/__init__.py`; `monitors/__init__.py:48-100` | → 04 §6.3, 11 §6 |
| Concurrency controller class | instantiated directly | `orchestrator/orchestrator.py:335`; `eval/runner.py:77-166` | → 03 §6 |
| Runtime registration (verifiers) | hard-coded unions + map; wire-load-bearing | `vf/runtimes/__init__.py:41-70` | → 08 §6.1 |
| Dialects (verifiers) | module-level tuple | `vf/dialects/__init__.py:11` | → 07 §6.7 |
| Training through non-chat dialects | `TrainClient` refuses | `vf/clients/train.py:342-350` | → 07 §7.1 |
| Client types (verifiers) | isinstance + wire union | `vf/clients/client.py:73-80`; `vf/configs/client.py:88` | → 07 §6.8 |
| Interception elastic pool tunnel type | hard-codes `PrimeTunnelConfig()` | `vf/interception/pool.py:123-125` | → 08 §3.5 |
| Env-server pool supervision; run timeout on the wire | none exists | `vf/serve/pool.py:18-20`; `vf/serve/client.py:100-102` | → 08 §7.2, §7.4 |
| Eval through the renderer/token path | eval always uses chat-completions `EvalClient` | `orchestrator/dispatcher.py:543-552` | → 10 §6.0, 07 §7.11 |
| Renderer auto-detection (exact `name_or_path` match) and multimodal feature encoding (class dispatch) | renderers package | `rnd/base.py:930-1047, 1490-1560`; `rnd/client.py:488-505` | → 09 §3.3, §3.9 |
| Where the renderer is built (env-server worker, from `model.name`; `[tokenizer]` ignored) | prime-rl client construction | `orchestrator/orchestrator.py:199-205`; `orchestrator/clients.py:80-91` | → 09 §2, §7.1 |
| NIXL conversion replay (view-op whitelist; custom models; dim-0 shards; static plan per process lifetime) | NIXL transport | `transports/weights/nixl/graph.py:35-55`; `nixl.py:326-338` | → 06 §3.9, §5.7 |
| NCCL rank arithmetic and world size (config values, uniform GPUs per admin URL, one-shot group) | admin plane + worker | `orchestrator/clients.py:165-203`; `inference/vllm/worker/nccl.py:91-126` | → 06 §3.8 |
| Pipeline parallelism (`pp` hard-wired to 1) | `ParallelDims` | `trainer/parallel_dims.py:329` | → 05 §1 |
| Multi-LoRA (`n_adapters=1`); LoRA ⇒ filesystem transport, one API server | trainer + config | `trainer/lora.py:256, 273`; `cfg/rl.py:452-457` | → 05 §3.15, 06 §5.12 |
| `model.forward` replacement by the fused LM head | build pipeline | `trainer/models/layers/lm_head.py:316` | → 05 §3.3 |
| Monitor rank gating (`RANK`/`DP_RANK`) and sequential fan-out | `monitors/__init__.py` | `monitors/__init__.py:67-69, 126-130` | → 11 §3.1 |
| Prometheus gauge list | fixed in `MetricsServer` | `utils/metrics_server.py:88-105` | → 11 §6 |
| Checkpoint completeness / "latest" resolution (max `step_*` name, resolved independently by launcher, trainer, orchestrator) | pathing helpers | `utils/pathing.py:295-311` | → 01 §3.10, 05 §3.16 |
| Multi-node topology (static ports, bash role logic, `SLURM_PROCID` ↔ host index) | sbatch template | `templates/multi_node_rl.sbatch.j2` | → 01 §7.4, §8 |
| Batch packing (FFD, silent truncation at `seq_len`, dummy padding) — lives in `trainer/batch.py` but runs in the orchestrator | packer | `trainer/batch.py:370-913`; `orchestrator/packing.py:28-35` | → 04 §3.7 |

---

## Source notes

- Sections are the primary source; code cites were taken from them and ~55 of the load-bearing ones were re-opened at the pin (algorithm hooks and registry, routing, loss plug-in, curriculum dispatch and checkpoint strictness, `TARGET_LAG`, plugin loader, runtime unions, TrainClient refusal, `DIALECTS`, `resolve_client`, renderer registry/config classification, model registration, `mini_moe.py:31`, NIXL `SUPPORTED_OPS`, weight/batch factories, `WORKER_EXTENSION_CLS` and `/pause`, admin-plane factory, the plugin entry point, monitor setup, shared-field propagation, template loader, weight-broadcast defaults, wire append-only comments, verifiers test markers, Harbor separate-verifier guard, harness default, and others). None were wrong.
- Two facts in this doc were read directly from code rather than from a section: the curriculum checkpoint's strict gate-name match (`orchestrator/curriculum/base.py:74-83`, [A6](#a6)) and `make_tunnel`'s silent fallback to `PrimeTunnel` for unknown configs (`vf/interception/tunnel/__init__.py:15-19`, [C6](#c6)).
- `[UNVERIFIED]` items carried from the sections: `hermes_agent` trainability (inferred), `openclaw` (model-dependent), and every vLLM-internal behaviour (read in vLLM 0.24, pin 0.29.0).
