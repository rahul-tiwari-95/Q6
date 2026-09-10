# No Way Home — PILOT Beta-Sweep (non-confirmatory)

**Not a confirmatory result.** 20 seeds/arm (9000-9019), used only to estimate per-arm variance for a power calculation. These seeds must never be reused as confirmatory data once Increment 3's real N is locked.

`instrument_mixed(beta)` run directly as a policy (no mandate/election gating), matching provenance_test_v1.md's C3b/C3c comparison structure. "Fooled" = mitigated at least once during the pre-shift period (provenance_test_v1.md's exact definition).

| beta | Fooled / N | F(beta) |
|---:|---:|---:|
| 0.0 | 13 / 20 | 0.65 |
| 0.25 | 12 / 20 | 0.60 |
| 0.5 | 10 / 20 | 0.50 |
| 0.75 | 0 / 20 | 0.00 |
| 1.0 | 0 / 20 | 0.00 |

## Pilot JT trend test (informative only, NOT a confirmatory result)

J = 2760.0, one-sided permutation p-value (decreasing trend) = 0.0001, n_permutations=9999.

This is a pilot with only 20 seeds/arm — not powered to be decisive either way. Its purpose is the per-arm F(beta) values above, which feed the power calculation for picking a real N (see run_power_calculation.py).
