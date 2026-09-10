# No Way Home — Engineering Phase Plan

> **2026-09-06 release update:** This document records the earlier project state. Read the [errata](../docs/experiments/provenance-errata.md) before using its research claims: beta monotonicity is structurally supplied, the old test ignored seed pairing, and the learner receives deduplication as a feature. The [new calibrated-control pilot](../docs/experiments/provenance-results.md) supersedes the old next-step plan. Original as-run result reports are preserved.

*This is the handoff document. If you're a fresh Claude session with no prior context on
this project, read this file top to bottom before touching code — it's written so you can
pick up work cold. Keep it current: whoever completes a TODO item updates this file in the
same commit that completes the work, not as a separate cleanup pass later.*

Branch: `no-way-home`. Worktree: `/Users/rahul/Q6-nwh`. Last verified against the repo
2026-08-09: `no_way_home/` is 15 modules, 32 tests passing across
`tests/test_no_way_home_smoke.py` (27) and `tests/test_stats.py` (5). Also reconfirmed against
the whole Q6 repo's suite (254/254) — this thread touches nothing outside `no_way_home/`.

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

**Increment 1 is complete** (2026-08-08, commits `cd33d0f`..`HEAD` on `no-way-home`):
`lineage_role` derived property on `Message`, `channels.py`'s manipulability registry for
the four live emitters, and the structural invariant test confirming `blight_exposure`'s
`index` tag actually holds at runtime across every policy. Schema-only, zero behavior
change. Full narrative: `results/README.md` (v6). Suite: 19/19 passing.

**Increment 2 is complete** (2026-08-08): `institutions.py::mixed_score` / `instrument_mixed(β)`
(verified both by direct arithmetic against manual interpolation at every pre-registered β, and
behaviorally against the two existing instruments at the β=0/β=1 endpoints across three distinct
log shapes); `messages.py::MessageLog.bounded_decay_rate` (verified to carry zero lineage
information — scores a forward storm and equal-volume independent reports identically, which is
the entire point of it as a control); `CANDIDATES["E_zero_intelligence_constrained"]` (verified to
calibrate cleanly, and verified — not assumed — that adding it doesn't change which candidate
qualification picks as best, which matters because the v4 mandate-ablation test's clean guarantee
silently depends on the winning candidate never touching `rng`). None of it touched
`EpistemicDelegationInstitution` or the election mechanism itself. Full narrative:
`results/README.md` (v7). Suite: 27/27 passing.

**Increment 3's core confirmatory result is in** (2026-08-09): `PREREGISTRATION_INCREMENT3.md`
locked N=50 (Rahul's call, above the initial N=30 suggestion, for knee-location precision) and
the grid/instrument/test *before* `run_beta_sweep_v1.py` ran against seeds 20000-20049. Result:
F(β) = 0.82, 0.82, 0.68, 0.00, 0.00 for β = 0, 0.25, 0.5, 0.75, 1.0 — Jonckheere-Terpstra
J=18650.0, p=0.00010. **A knee exists at β=0.75, not "no partial credit until β=1.0."** Confirms
the pilot's shape almost exactly (see `results/README.md` v9 for the full write-up and the
mechanistic explanation of why the knee sits there).

**One real gap left before this result is fully interpretable**: the bounded-decay-EMA control
arm has not run. Its threshold needs a held-out calibration pass (seeds disjoint from both the
pilot's 9000-9019 and the confirmatory 20000-20049 — e.g. 15000-15009, already named in
`PREREGISTRATION_INCREMENT3.md`) before it can be compared fairly. Until it runs, the result
above establishes the dose-response *shape* convincingly but not yet that it's specifically
about lineage-awareness rather than any sufficiently bounded/decayed counting rule.

Next atomic task, refilling the list per the rule above:

1. **Calibrate and run the bounded-decay-EMA control arm.** Pick `half_life`/`floor`/`ceiling`
   for `messages.py::bounded_decay_rate` and a threshold, using held-out seeds (e.g.
   15000-15009) by the same method `mandate_threshold=0.10` was picked (compare pre-shift vs.
   post-shift rate distributions, pick a cutoff that separates them). Then run it on the same
   20000-20049 confirmatory seeds and add its result to `results/beta_sweep_v1.md`. This closes
   the one named gap in Increment 3's confound-ruling-out story.

Once that lands, Increment 3 is fully complete, and the natural next conversation is whether
Increment 4 (the `CORROBORATION` channel) is warranted — the redesign doc's own gating condition
("only if Increment 3's curve is interesting enough to justify a new channel") looks satisfied
given a real, sharp, mechanistically-explained knee, but that's a call for Rahul, not an
automatic next step.

## 5. Backlog (not yet broken into atomic TODOs)

Full detail for all of these is in `ENVIRONMENT_REDESIGN.md` §4 — this is a pointer, not a
duplicate, so it can't drift out of sync with the source of truth.

- **Increment 4** (1-2 weeks, contingent on Increment 3's result being interesting — looking
  likely given the sharp knee found, pending Rahul's read): build the `CORROBORATION` channel —
  a second, costly, independently-`Initiates`-ing evidence type.

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
| `messages.py` | `Message` (has `.lineage_role`), `MessageLog` — report/forward mechanics, raw vs. unique-origin rate counting, plus `bounded_decay_rate` (Increment 2's first control arm). |
| `channels.py` | Manipulability registry (`CHANNELS`, `ChannelTag`) for the four live emitters — added in Increment 1. Not the full 8-tuple `Channel` type from `ENVIRONMENT_REDESIGN.md` §2; that's parked. |
| `institutions.py` | Ballot + election + delegated executor. `CANDIDATES` (now 5 entries), `EpistemicDelegationInstitution`, `mixed_score`/`instrument_mixed(β)` (Increment 2). This is where Increment 3's locked β grid gets wired into named candidates. |
| `policies.py` | Scripted policies (ZI, heuristics, oracle). `zero_intelligence_constrained` is now also `CANDIDATES["E_zero_intelligence_constrained"]`, reused as-is, zero adapter code. |
| `learning.py` | `TabularQMandateLearner` — the one learning agent so far. |
| `metrics.py` | Pure functions over the event log (`need_shortfall_per_10k`, `mitigation_rate`) — I-11 discipline: metrics are reproducible from raw events, never computed inline during the run. |
| `stats.py` | From-scratch, validated Jonckheere-Terpstra test (`jonckheere_terpstra_test`). No library provides this. Use this, don't re-derive or hand-roll another one. |
| `run_pilot_beta_sweep.py` | Non-confirmatory pilot (seeds 9000-9019). Do not reuse those seeds for the real Increment 3 run. |
| `run_power_calculation.py` | Monte Carlo power/precision calc feeding the N recommendation (currently N=30, unsigned-off). Re-run if the pilot is ever redone. |
| `results/README.md` | Hand-maintained narrative index — the honest history, what broke and what fixed it, at every increment. Update this, not just the auto-generated per-run reports, when Increment 3 produces its result. |
| `tests/test_no_way_home_smoke.py`, `tests/test_stats.py` | 27 + 5 tests, growing. |

## 7. Standing constraints (carried from the rest of Q6, still in force)

- New commits, not amends. Stage specific files, not `git add -A`.
- Never force-push, never skip git hooks.
- Ask before destructive or credential-related operations.
- Keep negative/awkward findings in the write-up — the project's credibility comes from
  this, not from every increment looking clean.
- Determinism first: any new randomness must go through the existing
  `SeedSequence(seed).spawn(...)` pattern, not a fresh unseeded `np.random` call.
