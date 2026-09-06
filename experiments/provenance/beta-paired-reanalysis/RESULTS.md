# No Way Home — Historical Beta-Sweep: paired reanalysis

Reuses the historical instrument, grid and seeds; corrects the permutation design: N=50 seeds/arm (20000-20049), same seed set reused across all arms, `instrument_mixed(beta)` run directly as a policy (window=30, threshold=0.30, both reused verbatim). "Fooled" = mitigated at least once during the pre-shift period.

| beta | Fooled / N | F(beta) |
|---:|---:|---:|
| 0.0 | 41 / 50 | 0.82 |
| 0.25 | 41 / 50 | 0.82 |
| 0.5 | 34 / 50 | 0.68 |
| 0.75 | 0 / 50 | 0.00 |
| 1.0 | 0 / 50 | 0.00 |

## Primary test

Within-seed Jonckheere-Terpstra permutation, alternative=decreasing, n_permutations=9999: **J = 18650.0, p = 0.00010**.

**Reject H0** (p < 0.05): a monotone decreasing trend across the beta grid is supported.

## Falsifiable shape claim

**Observed grid point**: the smallest tested beta with F(beta) <= 0.10 is beta=0.75 (F(0.75)=0.00), which is < 1.0 — partial lineage-weighting suffices in these tested worlds; full unique-origin dedup (beta=1.0) is not required for most of the forward-storm resistance.

## Control arm: E_zero_intelligence_constrained

Fooled 50/50 (1.00) on the same seeds. Has no beta-knob and no rate threshold, so it cannot itself produce a dose-response curve — reported for reference, not as a beta-grid point.

The decreasing trend is structurally supplied: raw >= unique, and pre-shift mitigation cannot change messages. The curve locates threshold crossings for one generator, not an emergent provenance mechanism. p=0.0001 is the Monte Carlo resolution floor, not an exact probability.

## Scope

The original equal-threshold comparison does not isolate the value of lineage information from action conservatism. A separate exploratory release pilot calibrates raw, unique and bounded-decay controllers with equal search budgets and reports whole-task outcomes; see docs/experiments/provenance-protocol.md. Historical as-run reports are unchanged.
