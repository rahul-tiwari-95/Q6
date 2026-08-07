# No Way Home — A1 Nontriviality Smoke Test

Config: 24 agents, 2000 ticks, shift at tick 800, seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

Primary metric: need-shortfall agent-ticks per 10,000 ticks (lower is better). Ranked best to worst overall.

| Policy | Overall (mean±std) | Pre-shift | Post-shift (mean±std) | Mitigation rate |
|---|---:|---:|---:|---:|
| C3_need_first_heuristic | 39317.0 ± 581.5 | 0.0 | 65473.8 ± 968.4 | 0.34 |
| C3c_lineage_aware_heuristic | 43535.0 ± 470.1 | 0.0 | 72497.9 ± 782.9 | 0.34 |
| greedy_state_oracle | 44310.0 ± 298.2 | 0.0 | 73788.5 ± 496.6 | 0.34 |
| C3b_lineage_naive_heuristic | 56909.0 ± 8622.1 | 0.0 | 94769.4 ± 14358.2 | 0.34 |
| always_mitigate | 63528.0 ± 263.0 | 0.0 | 105791.8 ± 438.0 | 0.34 |
| C1_zero_intelligence | 63928.0 ± 347.7 | 0.0 | 106458.0 ± 579.0 | 0.34 |
| C2_zero_intelligence_constrained | 63941.5 ± 387.7 | 0.0 | 106480.4 ± 645.6 | 0.34 |
| never_mitigate | 112920.0 ± 399.3 | 0.0 | 188043.3 ± 664.9 | 0.00 |

## Verdict

Best policy (C3_need_first_heuristic) vs. ZI-C relative gap: 38.5% (best=39317.0, ZI-C=63941.5).

The world discriminates: the best policy beats ZI-C by 38.5%, well above the 10% stop-condition threshold. Real signal exists for a policy to capture.

ZI (63928.0) vs. ZI-C (63941.5): the budget constraint alone barely changes outcomes.

**Notable, not a bug:** the need-first heuristic (39317.0) beats greedy_state_oracle (44310.0) by 11.3%, despite having strictly worse information (a lagging observed signal vs. the true hidden state). The oracle knows the truth but spends its wealth budget greedily and naively -- every tick blight is high, not accounting for the shared scarce budget -- and runs its buffer dry partway through the post-shift window. need_first only tries to mitigate when observed symptoms are already elevated, which happens to also be a more budget-conservative spending pattern. Perfect state information isn't sufficient here; spending discipline under a shared resource constraint is a separate skill, and this toy world already shows the two can trade off against each other. That's a real argument for why a learned policy -- which could jointly optimize inference AND timing -- might add something beyond either simple heuristic alone.
