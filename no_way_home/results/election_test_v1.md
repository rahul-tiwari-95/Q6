# No Way Home — Epistemic Delegation Election Test

Calibration seeds: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009] (disjoint from the 10 election-run seeds below). Public qualification score = mean need-shortfall/10k ticks on calibration seeds, mandate-gated, lower is better.

| Candidate | Public qualification score |
|---|---:|
| D_cautious | 40598 **(true best)** |
| A_responsive | 43619 |
| B_responsive_naive | 44188 |
| C_constant_spender | 44393 |

## Forced-executor baselines (no election, mandate still gated)

| Executor | Mean shortfall/10k (± std) |
|---|---:|
| D_cautious | 40704 ± 714 |
| A_responsive | 43535 ± 470 |
| B_responsive_naive | 44011 ± 395 |
| C_constant_spender | 44256 ± 292 |

## The actual institution (term-committed mandate + noisy plurality election, 24 voters, correct-vote-prob 0.7, re-elected every 200 ticks)

Mean shortfall/10k: **42509 ± 1695**


## Mechanism-isolation ablation: same election, mandate re-checked every tick instead of committed for the term

Mean shortfall/10k: **40704 ± 714**

Election outcomes across all terms and seeds:

| Winner | Times elected | Share |
|---|---:|---:|
| D_cautious | 100 | 100.0% |

## Verdict

The true best executor (D_cautious) is elected in 100.0% of terms — voting is not where any cost comes from here.

The institution's realized performance (42509) is +4.4% relative to always having the best executor hard-coded, no election, no mandate vote at all (40704).

**This is not election noise.** The ablation above -- identical election, mandate re-checked every tick instead of committed for the 200-tick term -- lands at 40704 ± 714, a gap of only +0.0% from the hard-coded baseline. Checked directly at the tick level (not just in aggregate): with the mandate re-evaluated every tick, the institution's mitigate decision matches the hard-coded baseline's on **0 out of 2,000 ticks differing**, across every seed tested -- they're identical runs. The entire 4.4%-point difference comes from one thing: authorizing spending once per 200-tick term instead of continuously. That's the real price of a term -- not a bug, and not something a better voting rule or election design could fix, because voting was never the problem.
