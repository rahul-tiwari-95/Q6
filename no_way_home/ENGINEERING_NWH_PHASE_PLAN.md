# No Way Home — Engineering Phase Plan

*This is the handoff document. If you're a fresh Claude session with no prior context on
this project, read this file top to bottom before touching code — it's written so you can
pick up work cold. Keep it current: whoever completes a TODO item updates this file in the
same commit that completes the work, not as a separate cleanup pass later.*

Branch: `no-way-home`. Worktree: `/Users/rahul/Q6-nwh`. Last verified against the repo at
commit `af4c313` (2026-08-08): `no_way_home/` is 9 modules, 17 tests passing in
`tests/test_no_way_home_smoke.py`.

---

## 1. What this project is (short version)

A deterministic multi-agent gridworld testing whether a system built from simple, checkable
parts (noisy voting, raw message counting, tabular Q-learning) can make good collective
decisions under a hidden regime shift — and precisely diagnosing what breaks when it
doesn't. Plain-language walkthrough with examples: [`README_FOR_DUMMIES.md`](README_FOR_DUMMIES.md).
Formal spec and invariants (I-1 through I-12): [`PREREGISTRATION.md`](PREREGISTRATION.md).
Full bibliography with why each source was used: [`CITATIONS.md`](CITATIONS.md).

## 2. Where we are

Five increments built and verified, each with a full write-up in `results/`:

- **v1-v2**: single-locality physical economy (food/medicine/wealth with spoilage). Real
  separation between policies once the economy was non-degenerate.
- **v3**: 4 localities + message/lineage layer (`messages.py`). Headline result: a
  unique-origin-counting policy resists a forward storm 30/30 seeds; a raw-count policy
  using the identical threshold is fooled 24/30 seeds.
- **v4**: ballot + noisy election + delegated executor (`institutions.py`). Two real bugs
  found and fixed (redundant thresholds; a shared-RNG-stream bug that silently broke
  same-seed comparisons across policies). Headline result: the institution's ~4.4% cost
  vs. a hard-coded best executor is entirely the mandate's term-commitment length, not
  voting noise — isolated via a mechanism-kill ablation, 0 mismatched ticks out of 2000.
- **v5**: a learning citizen (`learning.py`, tabular Q-learning). Found and fixed a
  learning-rate methodology bug (floored decay: `alpha = max(0.05, 1/visits)`). Headline
  result: the learner discovers the lineage-aware distinction from raw features alone,
  with a named residual gap (no wealth/budget feature yet).

Full narrative history, including what didn't work at each stage: [`results/README.md`](results/README.md).

**Current decision point**: a 6-thread research sweep + two independent designs +
adversarial critique + synthesis produced [`ENVIRONMENT_REDESIGN.md`](ENVIRONMENT_REDESIGN.md),
approved by Rahul (2026-08-08). It reframes the world's data as distinct **channels**
(autogenic/allogenic emitter, cue/index/signal manipulability, Initiates/Happens-only
lineage role, decay function) instead of one undifferentiated state bucket, and proposes
one genuinely new mechanism (a `CORROBORATION` channel) plus a precise, falsifiable
experiment (does partial lineage-weighting via a β-dial recover most of full
lineage-awareness's forward-storm resistance, or is dedup all-or-nothing). We are now
executing its 4-increment migration path (§4 of that doc). **This phase plan tracks that
execution.**

## 3. Testing philosophy — read this before writing any test

Simplicity is the actual engineering strategy here, not a nicety: the simpler the concept,
the more precisely it can be tested, and precise tests are what let this project trust its
own results enough to write down uncomfortable findings instead of quietly smoothing them
over (see `results/README.md` — every increment has at least one). Before writing a test,
answer these explicitly, in the test's own docstring or the commit message if not obvious
from the name:

- **What single, atomic thing does this test assert?** One behavior per test. If you need
  "and" to describe what it checks, split it.
- **Why does that assertion matter?** What real design invariant does it protect — not
  "this function returns X" but "this is what breaks if lineage_role and is_forward ever
  disagree."
- **What concrete bug would this catch if it failed?** If you can't name one, the test is
  decorative. Name it.
- **Is it truthful?** A test that passes vacuously (e.g. asserting something about an empty
  list because the fixture never actually triggers the condition) is worse than no test —
  it's confidently wrong. Check that the test can actually fail given a real bug, not just
  that it currently passes.

This is the same standard the existing 16 tests were held to (e.g.
`test_world_physics_identical_across_policies_that_consume_different_amounts_of_randomness`
exists because that exact bug was found and silently true of every prior comparison in the
repo until it was caught).

## 4. ACTIVE TODO (max 3 items)

*Rule: never let this list exceed 3 items. When one is completed, delete it and pull the
next atomic task off the Backlog (§5) below — don't let the backlog's larger "Increment"
groupings sit here unbroken; break the next increment into atomic tasks the same way
Increment 1 is broken out below before adding it here. If mid-work you discover the next
atomic step isn't obvious, that's a signal to stop and ask Rahul rather than guessing.*

**Recently completed:** `lineage_role` derived property on `messages.py::Message`
(2026-08-08) — `@property` returning `"Happens-only"` iff `is_forward` else `"Initiates"`,
zero new state, locked in by `test_lineage_role_matches_is_forward_exactly`. Full narrative
in `results/README.md` once Increment 1 finishes; see git log on `no-way-home` for the exact
commit in the meantime.

1. **Tag the four live emitters with a `manipulability` class.** Blight state = `index`
   (zero agent write access — nothing in the current codebase writes to `state.blight_high`
   except `step()`'s own regime-shift logic, verify this with a grep before tagging).
   Resource ledger (`wealth`/`food_stock`/`medicine_stock`) = `cue`. `REPORT` = `signal`.
   `FORWARD` = `index-of-a-claim` (see `ENVIRONMENT_REDESIGN.md` §2 for why FORWARD isn't a
   full `signal`). This needs a small home — a plain dict or dataclass registry, not a new
   abstraction layer; resist the urge to build the full 8-tuple `Channel` type from §2, it's
   explicitly out of scope until there's a reason to need it (see §5's parked list).

2. **Add the manipulability invariant test.** Assert zero events in the log where an
   `allogenic` emitter (agent-driven: `REPORT`, `FORWARD`) write to an `index`-tagged
   channel (`blight_high`). Concretely: no code path lets a policy or message-generation
   function mutate `state.blight_high` — this should currently pass trivially, which is
   fine; the point is making it a checked invariant instead of an accident of how the code
   happens to be organized, so a future change can't silently violate it.

These two, plus the completed `lineage_role` item above, are Increment 1 from
`ENVIRONMENT_REDESIGN.md` §4 (est. 1-2 days), broken into atomic units. None of them change
simulation behavior — this increment is schema-only. Run the existing suite
(`python3 -m pytest tests/test_no_way_home_smoke.py -q`) after both land and confirm all
tests still passing before committing. Once both are done, pull Increment 2's first atomic
task from §5 to bring this list back to 3.

## 5. Backlog (not yet broken into atomic TODOs)

Full detail for all of these is in `ENVIRONMENT_REDESIGN.md` §4 — this is a pointer, not a
duplicate, so it can't drift out of sync with the source of truth.

- **Increment 2** (2-3 days): `instrument_mixed(β)` in `institutions.py`; a bounded-decay-EMA
  rate function in `messages.py` with zero lineage info (the first required control arm);
  `E_zero_intelligence_constrained` added to `CANDIDATES` (the second required control arm).
- **Increment 3** (3-5 days): lock the β grid, the Jonckheere-Terpstra test, and a
  power-calculated N in a committed pre-registration note *before* running anything — flagged
  in the redesign doc as the single biggest practical risk in the whole plan. Then run the
  sweep + both control arms. This produces the actual deliverable: a dose-response curve.
- **Increment 4** (1-2 weeks, contingent on Increment 3's result being interesting): build the
  `CORROBORATION` channel — a second, costly, independently-`Initiates`-ing evidence type.

**Explicitly parked, not rejected** (don't pull these forward without a specific reason —
see `ENVIRONMENT_REDESIGN.md` §5 for why each was cut): the full 8-tuple `Channel`
abstraction, an event-sourced/lamport-versioned log rewrite, Sims/Matejka-McKay portfolio
attention economics, Bayesian belief-manipulation instrumentation, reward-machine automata,
SMC/MORL reporting infrastructure. Also parked, a separate and older thread:
`.memory/no-way-home-phase0-v2/`'s seven-contract verifier system — rejected by independent
review twice, PARKED as of 2026-08-03. This is the direct local precedent for why this
project trims scope aggressively rather than building formal apparatus ahead of a working
result.

## 6. Key files map

| File | What it is |
|---|---|
| `world.py` | Core kernel: `WorldConfig`, `WorldState`, `step()`, `run()`. Two independent RNG streams (world physics vs. policy decisions) — do not collapse them back into one. |
| `messages.py` | `Message`, `MessageLog` — report/forward mechanics, raw vs. unique-origin rate counting. This is where Increment 1's `lineage_role` field goes. |
| `institutions.py` | Ballot + election + delegated executor. `CANDIDATES` registry, `EpistemicDelegationInstitution`. This is where Increment 2's `instrument_mixed(β)` and ZI-control candidate go. |
| `policies.py` | Scripted policies (ZI, heuristics, oracle). `zero_intelligence_constrained` already exists and is reused as-is for Increment 2's control arm. |
| `learning.py` | `TabularQMandateLearner` — the one learning agent so far. |
| `metrics.py` | Pure functions over the event log (`need_shortfall_per_10k`, `mitigation_rate`) — I-11 discipline: metrics are reproducible from raw events, never computed inline during the run. |
| `results/README.md` | Hand-maintained narrative index — the honest history, what broke and what fixed it, at every increment. Update this, not just the auto-generated per-run reports, when Increment 3 produces its result. |
| `tests/test_no_way_home_smoke.py` | 16 tests, growing. |

## 7. Standing constraints (carried from the rest of Q6, still in force)

- New commits, not amends. Stage specific files, not `git add -A`.
- Never force-push, never skip git hooks.
- Ask before destructive or credential-related operations.
- Keep negative/awkward findings in the write-up — the project's credibility comes from
  this, not from every increment looking clean.
- Determinism first: any new randomness must go through the existing
  `SeedSequence(seed).spawn(...)` pattern, not a fresh unseeded `np.random` call.
