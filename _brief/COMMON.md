# Common brief — prime-rl architecture study (all study agents read this first, fully)

## Why this exists
We are building a **custom RL framework for LLMs** and will use prime-rl as the reference
implementation. The output is a set of **agent docs**: the docs a future agent (Atlas) reads to
build a complete mental AND mechanical image of how prime-rl works *before* touching low-level code,
doing migrations, or re-implementing pieces. The reader is a senior RL/systems engineer who has
never seen this codebase. They need: what runs where, how pieces find and talk to each other,
exact data shapes on every wire, the invariants the code silently assumes, every extension seam,
and every trap.

## Pin (everything you cite is at these commits)
- Checkout: `/tmp/atlas-prime/prime-rl` — upstream `PrimeIntellect-ai/prime-rl` @ `b944873` (2026-09-26)
- Submodules (under `deps/`): verifiers `69cc0f9`, renderers `6b8da3f`, prime-envs `b677502`,
  pydantic-config `65b15df`, prime-kernels `3fb83eb`.
- Config package lives in `packages/prime-rl-configs/src/prime_rl/configs/` (separate from `src/`).

## Rules (non-negotiable)
1. **Full-file reads.** Read every file in your scope end to end with the Read tool (chunk long files
   with offset/limit until done). Never derive an architectural claim from grep hits or file names —
   grep is only for *locating* things (e.g. "who calls X"), after which you READ the caller.
   When a claim depends on code outside your scope (a caller, a config class), read that code too.
2. **Cite everything.** Every mechanical claim gets a `path:line` cite, path relative to the
   prime-rl repo root (submodules as `deps/verifiers/verifiers/v1/...`). Names must be exact
   (classes, functions, config fields with defaults, env vars, HTTP routes, ports, file names).
3. **Static study only.** Do not install, `uv sync`, run training, or touch the network.
   Reading `tests/` to confirm behavior is encouraged (tests are executable specs).
4. **Current state only.** Describe the code as it is at the pin. No "this used to be" history.
   Prior Atlas docs exist at `~/cc-agents/atlas/notes/prime-rl-architecture-reference.md`,
   `prime-rl-framework-walkthrough.md`, `verifiers-v1-architecture-reference.md` — they are
   **STALE (@9b47b77, ~350 commits ago)**. You may skim them for orientation on *what questions
   to ask*, but never copy a fact from them without re-verifying it in the current code.
5. **Distinguish certainty.** Mark anything you inferred but could not confirm as `[UNVERIFIED]`,
   and put it in the Open Questions section. Wrong-but-confident is the worst outcome.
6. Don't write anywhere except your own output file(s) under
   `~/cc-agents/atlas/notes/prime-rl-arch/sections/`. Don't modify the checkout.
7. Use LaTeX (`$...$`) for math (losses, advantages, ratios). Use ASCII/mermaid diagrams where a
   picture beats prose (process layout, sequence of a call, state machine).

## Output template (one markdown file per assignment; use these sections, in this order)
```
# <Title> — prime-rl @ b944873
> Scope: <one line>. Files read in full: <list with LOC>. Related docs: <other section files>.

## 1. Mental model            — 1–3 paragraphs: what it is, why it exists, how to think about it.
## 2. Where it runs            — process, host/GPU placement, who launches it, lifecycle (start → steady → shutdown/crash).
## 3. Mechanics                — the main code paths walked step by step, with file:line. Sub-sections per path.
                                  This is the bulk. Low-level: data structures, queues, threads/asyncio tasks,
                                  locks, retries, timeouts, ordering.
## 4. Interfaces & contracts   — everything crossing this component's boundary: wire schemas (field-level),
                                  HTTP routes + payloads, files/dirs written & read (who writes, who reads, when),
                                  env vars, ports, config fields (name, type, default, effect).
## 5. Invariants & assumptions — what must hold for correctness; what the code silently assumes.
## 6. Extension points         — every seam to add/replace something: exact recipe (files, registration hook,
                                  config wiring), and what else must change in lockstep.
## 7. Gotchas & limitations    — traps, sharp edges, perf cliffs, known-unsupported combos, TODO/FIXMEs of note.
## 8. For a custom framework   — what is essential design vs incidental; coupling points; what we'd keep,
                                  simplify, or replace, and why. Opinionated but grounded.
## 9. Open questions           — what you could not verify, and where the answer probably lives.
```
Length: as long as needed to be complete — typically 600–1500 lines for a big area. Density over
padding; one idea per paragraph; tables for config fields/routes/schemas.

## Seam ownership (who documents each cross-component boundary in full; others summarize + link)
| Seam | Owner (full detail) | Also touches |
|---|---|---|
| S1 launcher → process spawn, placement, discovery (single-node, SLURM, k8s) | A-topology | all |
| S2 orchestrator ↔ inference (client pools, routes, router, weight-update control plane) | E-inference (server side + wire); B-orchestrator (client side) | |
| S3 orchestrator ↔ env workers (env server process, pools, protocol) | G-harness-runtime-serve (protocol/pools); I-envs (env resolution/loading); B (orchestrator call site) | |
| S4 env/harness → model calls (interception, dialects, token-in client, renderer use) | F-verifiers-core | H-renderers, E |
| S5 harness ↔ sandbox runtime | G | |
| S6 orchestrator → trainer rollout/batch transport | E (wire formats); B (producer); C (consumer data path) | |
| S7 trainer → inference weight sync | E (transport + receiver); D (trainer-side export/conversion) | |
| S8 checkpoint / resume across processes | D (trainer), B (orchestrator), A (launcher resume logic) | |
| S9 step / policy-version / off-policy accounting | C | B, D |

## Section files (so you can cross-link)
- 01-deployment-topology-and-launch.md (A) · 02-config-system.md (A)
- 03-orchestrator.md (B) · 04-algorithms-loss-data-path.md (C) · 05-trainer.md (D)
- 06-inference-and-transports.md (E) · 07-verifiers-core.md (F)
- 08-harnesses-runtimes-serve.md (G) · 09-renderers.md (H) · 10-envs-and-tasks.md (I)
- 11-observability-eval-ops.md (J)

## Final message to the coordinator
When done, reply with: (1) output path(s) + line counts; (2) the 10 most important facts a framework
builder must know from your area; (3) every cross-seam claim you made that another agent should
confirm from their side (so the coordinator can reconcile); (4) your open questions.
