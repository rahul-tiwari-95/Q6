# No Way Home — Pre-registration: Increment 3 Beta-Sweep

**Status: LOCKED before any confirmatory seed is run.** Committed 2026-08-09, before the
confirmatory sweep below has been executed at this repo state. The point of a pre-registration
is that it's written down before looking at the result it constrains — if this file is ever
edited after `results/beta_sweep_v1.md` exists, that's a violation worth flagging loudly, not a
routine touch-up.

## The question

*Does partial lineage-weighting recover most of the forward-storm resistance of full
lineage-awareness, or is deduplication all-or-nothing?* (`ENVIRONMENT_REDESIGN.md` §3). Not "is
aware better than naive" — already proven, `provenance_test_v1.md`, 0/30 vs 24/30. This is the
*shape* of the curve between those two known endpoints.

## What's locked

- **Grid**: β ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.
- **Instrument**: `institutions.py::instrument_mixed(beta)`, run directly as a `policy_fn` (no
  mandate/election gating — matches how `C3b_lineage_naive_heuristic` /
  `C3c_lineage_aware_heuristic` were compared in `provenance_test_v1.md`). `window=30`,
  `threshold=0.30`, both reused verbatim from the existing instruments — not re-tuned.
- **Outcome, per seed**: binary "fooled" — `provenance_test_v1.md`'s exact definition, mitigated
  at least once during the pre-shift (`blight_high=False`) period.
- **N = 50 seeds/arm.** Rahul's decision (2026-08-09), above the power calculation's initial
  N=30 suggestion, specifically for tighter knee-location precision given how sharp the pilot's
  step-function shape looked (`results/pilot_beta_sweep_v1.md`: F(β) = 0.65, 0.60, 0.50, 0.00,
  0.00 — a step between β=0.5 and β=0.75, not a gradient). At N=50: Wilson 95% upper bound on
  the true rate given 0 observed successes is **7.1%** (`results/power_calculation_v1.md`,
  rerun with N=50 in `CANDIDATE_NS`).
- **Confirmatory seeds: 20000-20049** (50 per arm, reused across all 5 arms so each arm sees the
  same seed set — matching `election_test_v1.md`'s and `provenance_test_v1.md`'s convention of
  paired seeds across compared policies). Explicitly disjoint from the pilot's 9000-9019 —
  those seeds are never reused as confirmatory data, to avoid the pilot's own sampling noise
  leaking into the "confirmatory" result.
- **Primary test**: Jonckheere-Terpstra ordered-alternative trend test
  (`stats.py::jonckheere_terpstra_test`), `alternative="decreasing"` (β ascending, F(β)
  hypothesized descending), `n_permutations=9999`, `alpha=0.05`.
- **Falsifiable shape claim, stated before running**: a **knee** exists
  (`min{β : F(β) <= 0.10} < 1.0`, i.e. partial weighting suffices) versus **no partial credit**
  (F(β) stays near the naive rate until β=1). Both outcomes are real, useful, and currently
  unknown at the confirmatory scale — the pilot is suggestive, not proof, at N=20.

## Control arms — different readiness, both required for interpretation

- **`E_zero_intelligence_constrained`** (already in `institutions.py::CANDIDATES`, no further
  calibration needed — it has no threshold to tune): run at the same N=50, same seed range
  20000-20049, reporting its own "fooled" rate alongside the β grid. Ready now.
- **Bounded-decay-EMA control** (`messages.py::bounded_decay_rate`): **NOT yet calibrated.**
  Its `half_life`/`floor`/`ceiling`/threshold need a held-out calibration pass (same spirit as
  how `mandate_threshold=0.10` was picked by comparing pre-shift vs. post-shift rate
  distributions on calibration seeds, `institutions.py::_mandate_authorized`'s docstring) —
  using a seed range disjoint from both the pilot (9000-9019) and the confirmatory range
  (20000-20049), e.g. 15000-15009. This is the next atomic step, tracked in
  `ENGINEERING_NWH_PHASE_PLAN.md`, not skipped or silently deferred.

## Why the core sweep runs now, before the control's calibration is finished

The five-arm β grid and the ZI-constrained control are both fully specified with no remaining
free parameters — nothing about running them requires a decision that isn't already locked
above. The bounded-decay-EMA control genuinely does require one more calibration decision.
Rather than block the ready part on the unready part, the core sweep + ZI-constrained control
run under this pre-registration now; the bounded-decay-EMA control's result gets added to
`results/beta_sweep_v1.md` as a follow-up once calibrated, clearly labeled as landing after the
primary result rather than backdated into it.
