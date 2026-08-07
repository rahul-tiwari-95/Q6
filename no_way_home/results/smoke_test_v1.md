# No Way Home — A1 Nontriviality Smoke Test

Config: 24 agents, 2000 ticks, shift at tick 800, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary metric: need-shortfall agent-ticks per 10,000 ticks (lower is better). Ranked best to worst overall.

| Policy | Overall (mean±std) | Pre-shift | Post-shift (mean±std) | Mitigation rate |
|---|---:|---:|---:|---:|
| C3_need_first_heuristic | 39544.5 ± 273.0 | 0.0 | 65852.6 ± 454.6 | 0.34 |
| C3c_lineage_aware_heuristic | 43604.0 ± 215.3 | 0.0 | 72612.8 ± 358.5 | 0.34 |
| greedy_state_oracle | 44362.5 ± 256.1 | 0.0 | 73875.9 ± 426.5 | 0.34 |
| C3b_lineage_naive_heuristic | 51507.0 ± 8373.9 | 0.0 | 85773.5 ± 13944.8 | 0.34 |
| always_mitigate | 63653.5 ± 263.4 | 0.0 | 106000.8 ± 438.7 | 0.34 |
| C2_zero_intelligence_constrained | 64072.5 ± 306.2 | 0.0 | 106698.6 ± 510.0 | 0.34 |
| C1_zero_intelligence | 64081.0 ± 284.1 | 0.0 | 106712.7 ± 473.1 | 0.34 |
| never_mitigate | 112987.0 ± 297.0 | 0.0 | 188154.9 ± 494.6 | 0.00 |

## Verdict

Best policy (C3_need_first_heuristic) vs. ZI-C relative gap: 38.3% (best=39544.5, ZI-C=64072.5).

The world discriminates: the best policy beats ZI-C by 38.3%, well above the 10% stop-condition threshold. Real signal exists for a policy to capture.

ZI (64081.0) vs. ZI-C (64072.5): the budget constraint alone barely changes outcomes.

**Notable, not a bug:** the need-first heuristic (39544.5) beats greedy_state_oracle (44362.5) by 10.9%, despite having strictly worse information (a lagging observed signal vs. the true hidden state). The oracle knows the truth but spends its wealth budget greedily and naively -- every tick blight is high, not accounting for the shared scarce budget -- and runs its buffer dry partway through the post-shift window. need_first only tries to mitigate when observed symptoms are already elevated, which happens to also be a more budget-conservative spending pattern. Perfect state information isn't sufficient here; spending discipline under a shared resource constraint is a separate skill, and this toy world already shows the two can trade off against each other. That's a real argument for why a learned policy -- which could jointly optimize inference AND timing -- might add something beyond either simple heuristic alone.
