# Review brief — adversarial verification of one section (read fully before starting)

You are a **reviewer**, not an author. A study agent wrote your section from full-file reads of
prime-rl @ `b944873` (checkout `/tmp/atlas-prime/prime-rl`, submodules under `deps/`). Your job is
to **try to break it**: find wrong claims, wrong cites, missing load-bearing mechanics, and resolve
the cross-seam questions other agents raised about your area. Read `_brief/COMMON.md` (pin,
template, rules) and `_brief/LEDGER.md` (the coordinator's cross-seam ledger) first, fully.

## Procedure
1. Read your section file(s) end to end.
2. **Pick the ~30–40 most load-bearing claims** — the ones a framework builder would act on:
   protocol sequences, wire schemas, defaults, ordering/locking, invariants, extension recipes,
   and every "bug"/"gotcha" claim. Verify each by READING the cited code (full function, plus
   callers/callees as needed — never trust a grep hit or the cite alone). Probe, don't confirm:
   for each claim, ask "what would the code look like if this were false?" and look for that —
   alternate branches, config flags that change the path, other call sites, error handlers,
   overrides/monkeypatches elsewhere.
3. **Resolve the ledger items addressed to your area** (lines `Confirm: <LETTER> —` / `<X>/<LETTER> —`
   targeting your owner letter, plus any open questions / suspected bugs in your area). Give each a
   verdict: CONFIRMED / REFUTED / PARTIAL / UNRESOLVABLE-STATICALLY, with cites.
4. **Probing by execution (optional, CPU-only, no installs):** you may run small pure-Python
   snippets against the checkout with
   `PYTHONPATH=/tmp/atlas-prime/prime-rl/src:/tmp/atlas-prime/prime-rl/packages/prime-rl-configs/src:/tmp/atlas-prime/prime-rl/deps/pydantic-config/src:/tmp/atlas-prime/prime-rl/deps/verifiers:/tmp/atlas-prime/prime-rl/deps/renderers $SCRATCH/atlas/rl-scaling/.venv/bin/python -c '...'`
   (that venv has pydantic 2.13, torch 2.11 CPU usable, msgspec, pyzmq; versions may not match the
   pin — if an import fails, fall back to static reasoning; NEVER pip/uv install, never touch GPUs,
   never hit the network, never write into the checkout). Good targets: config parsing/precedence,
   validators, msgspec encode/decode of wire structs, packing functions, pure math helpers.
   Work in your scratch dir `/tmp/claude-2122217957/-home-mila-a-amirhossein-kazemnejad-cc-agents-atlas/e4528407-4cf6-40ff-93b4-9487261cb920/scratchpad/`.
5. **Fix the section in place** (Edit tool): correct wrong claims, fix cites, add missing
   load-bearing mechanics you found, convert `[UNVERIFIED]` to verified (or keep and sharpen).
   Integrate ledger resolutions into the relevant section of the doc (not a separate appendix).
   Keep the template and density; do NOT add a changelog or "reviewer notes" section to the doc —
   the doc must read as current truth. Don't balloon it: net growth ≤ ~15%.
6. Write your review record to `~/cc-agents/atlas/notes/prime-rl-arch/_brief/reviews/<NN>.md`:
   claims checked (count + list, one line each with verdict), errors found → fix applied (before → after,
   cite), ledger items → verdict + cite, remaining open questions.
7. Don't git commit. Don't edit other sections (if you find an error in another section, report it in
   your review record under "Errors in other sections" with cite).

## Final message to the coordinator
(1) counts: claims checked / confirmed / corrected; (2) the corrections that matter most;
(3) ledger verdicts (one line each); (4) errors spotted in OTHER sections; (5) remaining opens.
