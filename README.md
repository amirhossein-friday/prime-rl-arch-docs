# prime-rl architecture: agent docs

> Standalone copy of `notes/prime-rl-arch/` from the Atlas workspace repo ([`amirhossein-friday/atlas`](https://github.com/amirhossein-friday/atlas/tree/main/notes/prime-rl-arch)), which remains the canonical, maintained version.

Mechanical and mental map of **prime-rl** (async RL for LLM agents), written for agents who will build a custom RL framework on its ideas or work on its low-level code.

| Pin | Commit |
|---|---|
| prime-rl | `PrimeIntellect-ai/prime-rl` @ `b944873` (2026-09-26) |
| verifiers | `69cc0f9` |
| renderers | `6b8da3f` |
| prime-envs | `b677502` |
| pydantic-config | `65b15df` |
| prime-kernels | `3fb83eb` |
| vLLM | 0.29.0 (exact wheel pin) |

Checkout used: `/tmp/atlas-prime/prime-rl`. If it is gone, re-clone with `git clone --recurse-submodules https://github.com/PrimeIntellect-ai/prime-rl && git checkout b944873 && git submodule update --init --recursive`. Every `path:line` cite is relative to the prime-rl repo root; submodule paths are prefixed `deps/<name>/`.

A sibling doc set for **miles** (RadixArk's SGLang + Megatron/FSDP + Ray/k8s framework) lives in the Atlas workspace repo at [`notes/miles-arch/`](https://github.com/amirhossein-friday/atlas/tree/main/notes/miles-arch). Its `23-custom-framework-notes.md` compares miles and prime-rl decision by decision.

## Start here

1. **`00-mental-model.md`**: one page covering the four processes, five communication planes, the loop, invariants and traps. Read it first, always.
2. Then pick a path:

| If you are about to… | Read |
|---|---|
| design our own framework | `23-custom-framework-notes.md` → `20-contracts.md` → `22-gotchas-and-bugs.md` |
| change or add something (algorithm, env, harness, runtime, renderer, model, transport) | `21-extension-points.md` (the seam's recipe) → the owning section → `22-gotchas-and-bugs.md` |
| debug a run (hang, crash, bad reward, NaN, resume failure) | `12-end-to-end-flows.md` (the "timing and blocking" boxes and the global timeout and hang registry) → `22-gotchas-and-bugs.md` → the owning section §7 |
| deploy (single-node, SLURM, k8s) | `sections/01-deployment-topology-and-launch.md` → `sections/02-config-system.md` → the `22` "safe first run" checklist |
| understand one component deeply | its section below |

## The docs

**Cross-cutting**

| File | What it is |
|---|---|
| `00-mental-model.md` | the picture to hold in your head |
| `12-end-to-end-flows.md` | cross-process sequence traces: cold start, one rollout (token-level), one training step, weight update per transport, resume, shutdown, online eval |
| `20-contracts.md` | every cross-process and cross-repo contract, field-level (wire schemas, routes, ports, env vars, run-dir layout, step/version accounting), plus a desync detector |
| `21-extension-points.md` | every seam as a recipe (files, registration, config wiring, lockstep changes, tests, pitfalls), plus where there is no seam |
| `22-gotchas-and-bugs.md` | every bug and hazard, deduplicated with stable IDs (B- bugs, H- silent correctness hazards, O- operational, P- perf, D- upstream docs drift), plus the safe-first-run checklist |
| `23-custom-framework-notes.md` | design brief: decision table (keep / fix / replace), invariants to build in, reuse vs rewrite, minimal first slice |

**Component sections (`sections/`)**, each following the same template: 1 mental model · 2 where it runs · 3 mechanics · 4 interfaces & contracts · 5 invariants · 6 extension points · 7 gotchas · 8 for a custom framework · 9 open questions.

| # | Section | Owns |
|---|---|---|
| 01 | `deployment-topology-and-launch` | process inventory, spawn order, GPU placement, single-node / SLURM / k8s, discovery, ports, run dir, resume at the launcher |
| 02 | `config-system` | pydantic-config composition and precedence, the RLConfig graph, shared-field propagation, validators, CLI pitfalls |
| 03 | `orchestrator` | dispatcher, groups, lag and ship gates, weight watcher, adaptive concurrency, sinks, packing and shipping, orchestrator checkpoint |
| 04 | `algorithms-loss-data-path` | Algorithm hooks, every shipped algorithm's math, the loss (IPO/IcePop/custom, ce, ref_kl), normalization, MicroBatch, step/version accounting |
| 05 | `trainer` | torchrun/mesh, FSDP2/EP/CP, model build and registry, conversion chain, optimizers and offload, checkpoints, weight export |
| 06 | `inference-and-transports` | vLLM extension (routes, patches), `/generate`, router, weight sync (NCCL/FS/NIXL), batch transport, P/D, discovery |
| 07 | `verifiers-core` | env/taskset/harness object model, rollout lifecycle, interception, dialects, TrainClient, message graph, scoring, harness trainability |
| 08 | `harnesses-runtimes-serve` | env-server protocol and pools, harness interface (ACP and others), sandbox runtimes, network policy, adding a runtime or harness |
| 09 | `renderers` | renderer API, bridge/extension property, per-token attribution, parsing, generate client, family table, adding a renderer |
| 10 | `envs-and-tasks` | env resolution by module name, install (uv workspace), env anatomy, mixing, Harbor, the add-an-env recipe |
| 11 | `observability-eval-ops` | monitors, file-monitor record format, metric matrix, MFU, eval runner, dashboard, tests/CI |

## How these were built, and how far to trust them

- **Readers:** 10 study agents (one per section; one agent wrote both 01 and 02) read every in-scope file in full (~50k LOC in prime-rl `src`, ~31k verifiers, ~23k renderers, plus the config package, templates and k8s). prime-envs was covered through representative envs read whole plus a mechanical scan of the catalog. Each cross-process seam had a designated owner in `_brief/COMMON.md`; some are split by side (e.g. weight sync: trainer export vs. transport and receiver).
- **Reviewers:** 10 adversarial reviewers, one per study agent, each tried to break 38–53 load-bearing claims. Together they checked about 420 claims and corrected or sharpened about 100. Where possible they probed by execution, on CPU only: running config validators and `cli()`, msgspec round-trips, loss functions and packing, the renderer on a cached tokenizer, `graph.py` on synthetic traces, 2-rank gloo DTensor tests, the real Dispatcher and WeightWatcher with a fake receiver, and pyzmq HWM and bind tests. The probe venv does not match the pin (torch 2.11, vLLM 0.24; a few missing symbols were stubbed), so a probe proves the pinned code's logic, not its behaviour on the pinned dependency versions.
- **Reconciliation:** the coordinator logged every cross-seam claim in `_brief/LEDGER.md`, reviewers ruled on them, and the coordinator applied the verdicts across sections (a later ledger entry wins; review records are in `_brief/reviews/`).
- **Synthesis:** `12`, `20`, `21` and `22` were compiled from the reviewed sections and the ledger under `_brief/SYNTH.md`, which requires re-opening at least 25 load-bearing cites at the pin per doc. `00`, `23` and this README were written by the coordinator and then given a final adversarial review against the code, sections, catalogs and ledger.
- **Trust levels:** claims marked `[UNVERIFIED]` could not be settled statically. **vLLM-internal behaviour** (e.g. that `pause(mode="keep")` freezes requests, and prefix-cache semantics) was read in vLLM **0.24** source; the pin is **0.29.0**. Re-check those before relying on them. Bugs are labelled static, probe-executed or needs-runtime in `22`.
- **Staleness:** these docs describe `b944873`. prime-rl moves fast: 348 upstream commits separate our fork point (`84b413fd7`) from this pin. Before trusting a detail for new work, check `git log b944873..origin/main -- <file>`.

## Maintenance

- Docs describe the pinned code only. To refresh, re-pin and re-run the study with `_brief/COMMON.md` (template and seam ownership), `_brief/REVIEW.md` (reviewer procedure) and `_brief/SYNTH.md` (cross-cutting docs).
- Older docs (July 2026, prime-rl `6bfbd64a`–`9b47b77`) are superseded for upstream and carry a banner saying so: [`prime-rl-architecture-reference.md`](https://github.com/amirhossein-friday/atlas/blob/main/notes/prime-rl-architecture-reference.md), [`prime-rl-framework-walkthrough.md`](https://github.com/amirhossein-friday/atlas/blob/main/notes/prime-rl-framework-walkthrough.md), [`verifiers-v1-architecture-reference.md`](https://github.com/amirhossein-friday/atlas/blob/main/notes/verifiers-v1-architecture-reference.md) (Atlas workspace repo). They remain roughly right for our `atlas-rl-scaling` fork (base `84b413fd7`), which has not merged upstream since (merge-base with `upstream/main` is still `84b413fd7`; only upstream PR #3004 was backported).
