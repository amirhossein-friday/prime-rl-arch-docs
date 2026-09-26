# Synthesis brief — cross-cutting docs over the reviewed sections

You are writing one **cross-cutting** doc in an agent-doc set about prime-rl @ `b944873`
(checkout `/tmp/atlas-prime/prime-rl`, submodules under `deps/`). Eleven section docs
(`~/cc-agents/atlas/notes/prime-rl-arch/sections/01…11-*.md`) were written from full-file reads and
then adversarially reviewed and corrected. They are your primary source. `_brief/LEDGER.md` records
the cross-section reconciliation and every review verdict. Read it; where a section and the ledger
disagree, the **ledger's later entry wins**. `00-mental-model.md` is the one-page overview; stay
consistent with it.

## Rules
1. **Read every section fully** (they're ~40–90 KB each; chunk with offset/limit). Also read `LEDGER.md`
   and `00-mental-model.md` fully. Don't skim — your doc's value is completeness across sections.
2. **No new claims without code.** Everything you write must come from a section (cite it as
   `→ 03 §3.8`) *and* carry the code cite (`path:line`) the section gives. If you add anything the
   sections don't have, read the code and cite it. **Spot-check at least 25 of the most load-bearing
   code cites against the checkout** (open the file, confirm the line says what's claimed). Fix a wrong
   cite in YOUR doc and report it in your final message (don't edit the sections).
3. Current state only; no history; no "reviewer said". Mark anything still unverified `[UNVERIFIED]`.
   Carry the vLLM caveat: vLLM-internal claims were read in vLLM 0.24; the pin is 0.29.0.
4. Ergonomics first. This is a **reference an agent opens mid-task**. Use tables with stable
   anchors, one row per item, and consistent columns. Put a short "how to use this doc" at the top.
   Don't pad.
5. Write only your output file. Don't commit.

## Final message
Output path + line count; the cites you spot-checked (count, and any wrong ones); items you found
missing or contradictory across sections; anything you had to leave `[UNVERIFIED]`.
