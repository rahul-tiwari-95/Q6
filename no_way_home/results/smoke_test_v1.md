# No Way Home — A1 Nontriviality Smoke Test

*This is the auto-generated table below `run_smoke_test.py` produces; this note is hand-written
context above it, in Q6's usual practice of documenting what didn't work first.*

**Iteration 1 was degenerate**, and worth recording rather than quietly overwriting. The first
version of the world had no spoilage on food/medicine stocks and food yield that exceeded need
even under high blight without mitigation. Result: every policy except `never_mitigate` scored a
perfect 0.0 — the pre-shift low-blight period banked a food surplus that peaked above 11,000
units (starting stock was 20), large enough to absorb almost any post-shift deficit regardless of
policy. That's not signal, it's an unbounded bank account standing in for a physical economy. The
table below is iteration 2: spoilage added to both stocks (matching invariant I-3's own "artifact
decay" principle), and mitigation's cost/benefit consolidated into one blight-severity lever
instead of two independently-tuned channels fighting each other's parameterization.

Config: 8 agents, 2000 ticks, shift at tick 800, seeds [0, 1, 2, 3, 4].

Primary metric: need-shortfall agent-ticks per 10,000 ticks (lower is better). Ranked best to worst overall.

| Policy | Overall (mean±std) | Pre-shift | Post-shift (mean±std) | Mitigation rate |
|---|---:|---:|---:|---:|
| C3_need_first_heuristic | 8308.0 ± 306.9 | 0.0 | 13835.1 ± 511.1 | 0.34 |
| greedy_state_oracle | 10588.0 ± 170.3 | 0.0 | 17632.0 ± 283.6 | 0.34 |
| always_mitigate | 16037.0 ± 214.8 | 0.0 | 26706.1 ± 357.7 | 0.34 |
| C2_zero_intelligence_constrained | 16113.0 ± 281.9 | 0.0 | 26832.6 ± 469.4 | 0.34 |
| C1_zero_intelligence | 16205.0 ± 225.4 | 0.0 | 26985.8 ± 375.4 | 0.34 |
| never_mitigate | 31755.0 ± 281.4 | 0.0 | 52880.9 ± 468.7 | 0.00 |

## Verdict

Best policy (C3_need_first_heuristic) vs. ZI-C relative gap: 48.4% (best=8308.0, ZI-C=16113.0).

The world discriminates: the best policy beats ZI-C by 48.4%, well above the 10% stop-condition threshold. Real signal exists for a policy to capture.

ZI (16205.0) vs. ZI-C (16113.0): the budget constraint alone barely changes outcomes.

**Notable, not a bug:** the need-first heuristic (8308.0) beats greedy_state_oracle (10588.0) by 21.5%, despite having strictly worse information (a lagging observed signal vs. the true hidden state). The oracle knows the truth but spends its wealth budget greedily and naively -- every tick blight is high, not accounting for the shared scarce budget -- and runs its buffer dry partway through the post-shift window. need_first only tries to mitigate when observed symptoms are already elevated, which happens to also be a more budget-conservative spending pattern. Perfect state information isn't sufficient here; spending discipline under a shared resource constraint is a separate skill, and this toy world already shows the two can trade off against each other. That's a real argument for why a learned policy -- which could jointly optimize inference AND timing -- might add something beyond either simple heuristic alone.
