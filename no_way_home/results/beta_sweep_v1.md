# No Way Home — Beta-Sweep, Increment 3 (CONFIRMATORY)

Executes `PREREGISTRATION_INCREMENT3.md` exactly: N=50 seeds/arm (20000-20049), same seed set reused across all arms, `instrument_mixed(beta)` run directly as a policy (window=30, threshold=0.30, both reused verbatim). "Fooled" = mitigated at least once during the pre-shift period.

| beta | Fooled / N | F(beta) |
|---:|---:|---:|
| 0.0 | 41 / 50 | 0.82 |
| 0.25 | 41 / 50 | 0.82 |
| 0.5 | 34 / 50 | 0.68 |
| 0.75 | 0 / 50 | 0.00 |
| 1.0 | 0 / 50 | 0.00 |

## Primary test

Jonckheere-Terpstra, alternative=decreasing, n_permutations=9999: **J = 18650.0, p = 0.00010**.

**Reject H0** (p < 0.05): a monotone decreasing trend across the beta grid is supported.

## Falsifiable shape claim

**Knee exists**: the smallest beta with F(beta) <= 0.10 is beta=0.75 (F(0.75)=0.00), which is < 1.0 — partial lineage-weighting suffices, full unique-origin dedup (beta=1.0) is not required to get most of the forward-storm resistance.

## Control arm: E_zero_intelligence_constrained

Fooled 50/50 (1.00) on the same seeds. Has no beta-knob and no rate threshold, so it cannot itself produce a dose-response curve — reported for reference, not as a beta-grid point.

## Not yet in this result

The bounded-decay-EMA control arm (`messages.py::bounded_decay_rate`) still needs its threshold calibrated on held-out seeds before it can run as a fair comparison — see `PREREGISTRATION_INCREMENT3.md`. This is a real, named gap in the confound-ruling-out story, not a silent omission: the result above establishes the dose-response shape, not yet that it's specifically about lineage rather than any bounded/decayed counting.
