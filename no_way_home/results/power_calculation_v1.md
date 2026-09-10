# No Way Home — Power Calculation for Increment 3's Confirmatory N

Smoothed per-arm probabilities used as the working effect size, from run_pilot_beta_sweep.py's actual output (Jeffreys smoothing, (successes + 0.5) / (n + 1) -- NOT the raw pilot MLEs, since two arms hit exactly 0/20 and plugging in a literal 0.0 would make this calculation overconfident by assuming certainty a 20-seed pilot cannot establish):

| beta | pilot F(beta) | smoothed p |
|---:|---:|---:|
| 0.0 | 13/20 | 0.643 |
| 0.25 | 12/20 | 0.595 |
| 0.5 | 10/20 | 0.500 |
| 0.75 | 0/20 | 0.024 |
| 1.0 | 0/20 | 0.024 |

## Estimated power vs. N (seeds/arm)

500 Monte Carlo replicates per N, 999 permutations per replicate's JT test, alpha=0.05.

| N/arm | Estimated power |
|---:|---:|
| 5 | 0.87 **(first to reach 80%)** |
| 8 | 0.98 |
| 10 | 1.00 |
| 12 | 1.00 |
| 15 | 1.00 |
| 20 | 1.00 |
| 25 | 1.00 |
| 30 | 1.00 |
| 40 | 1.00 |
| 50 | 1.00 |

## Why 'smallest N for 80% power' is the wrong number to act on

The mechanical answer above (N=5) only asks 'will the JT test detect THAT a trend exists.' It says nothing about the actual deliverable (ENVIRONMENT_REDESIGN.md §3): locating the knee, `min(beta such that F(beta) <= 0.10)`. That's an ESTIMATION question, not a detection question, and small N is badly underpowered for it even when it clears the detection bar. Wilson 95% upper confidence bound on the true rate, given 0 observed successes at each N:

| N/arm | Wilson upper bound on true F(beta), given 0 observed |
|---:|---:|
| 5 | 43.4% |
| 8 | 32.4% |
| 10 | 27.8% |
| 12 | 24.2% |
| 15 | 20.4% |
| 20 | 16.1% |
| 25 | 13.3% |
| 30 | 11.4% |
| 40 | 8.8% |
| 50 | 7.1% |

At N=5, observing 0/5 'fooled' is still consistent with a true rate as high as 43% -- essentially uninformative about whether beta=0.75 is actually safe. At N=20, that tightens to 16%; at N=30, 11%. The detection-power table above and this precision table are answering two different questions, and only the second one is close to what Increment 3 actually needs to report.

## Decision

**Not N=5.** Rahul signed off on **N=50 seeds/arm**, past the initial N=30 suggestion, specifically for tighter precision on the knee-location given how sharp the pilot's step-function shape looked (a hunch worth respecting: a genuinely sharp transition is exactly the case where you want enough resolution to trust *where* it sits, not just that it exists). N=50 clears detection power with overwhelming margin against the pilot's smoothed effect size, and gets the Wilson upper bound on a zero-observed arm down to the value in the table above for N=50. Locked in PREREGISTRATION_INCREMENT3.md -- this script's own recommendation logic above is left as-is (mechanical smallest-N and the precision table) since it's useful context, but the actual decision is N=50, not whatever this function would output on its own.

**Honest caveats**: (1) this is a power/precision calculation against the PILOT's own observed effect size, not a hypothesis-free calculation -- if the true effect is smaller than what a 20-seed pilot happened to show (regression to the mean is a real risk, especially for the two 0/20 arms), even N=30 could be optimistic. (2) The pilot's shape (a near step-function between beta=0.5 and beta=0.75, not a smooth gradient) is itself only a 20-seed observation and could look different with more data -- it's a reason to be interested in the real result, not a preview of it.
