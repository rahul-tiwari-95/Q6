# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: familiar-start success hides a coverage gap

The [state-coverage comparison](experiments/coverage_results_v1.md) holds DDQN, four-action loss, network, initialization and update budget fixed. Exhaustive training reaches **80.21% fresh greedy success**, versus **68.23%** from 59,626 unique states visited by a fixed random collector. All exhaustive seeds pass the fresh gate; collected seeds 0 and 2 miss it. Collected training achieves **97.14% familiar-start success**, but only **41.15% efficient fresh success**, versus exhaustive's 65.63%. Both miss the every-seed fit and efficiency diagnostics.

The collector visited 36.39% of the full state bank, with unequal map, clock and goal-near coverage. This makes current-state support and its induced sampling distribution a consequential part of this offline setup. It does not isolate support size from composition, or show that replay weighting or online exploration is the sole cause. Both arms retain privileged four-action transitions and full-bank detached successor queries.

The exhaustive weights and sampling reproduce the preceding [fixed-target DDQN study](experiments/fixed_targets_results_v1.md) exactly. The new fresh panel differs, so its 80.21% score is not a regression from the earlier 86.46% on another panel. That earlier experiment showed exact and bootstrapped targets both work under broad fixed coverage. The [supervised study](experiments/supervised_results_v1.md) first established useful exact-target capacity. The [A-only RL comparison](experiments/competence_results_v1.md) and both earlier adaptation pilots failed fresh or initial competence. Their artifacts remain unchanged, and later A → B → A outcomes do not establish forgetting or recovery.

## 1. Next bounded milestone: equal-size state-support control

Compare the collected bank with a **uniformly selected fixed subset of 59,626 exhaustive states**, keeping DDQN, all-four-action targets, paired initialization and update budget unchanged. Declare subset RNG, sampling, successor access and a new evaluation panel before execution. Record map, clock, goal-near and successor coverage. One fixed subset shared across learner seeds is one bank, not three independent subset replications.

Why this comes next: the completed comparison changed both unique-state count and state composition. A better equal-size uniform subset would show that unique-state count alone is insufficient to explain this observed gap. Composition includes transition-graph structure: randomly scattered states can require more outside-support successor queries. If both subsets struggle, that does not prove state count is the cause; support size, repeated exposure and target fitting remain candidates for a subsequent declared control. Familiar-start success alone should not decide the next step.

This follow-up has not run. Do not add memory or continuous online feedback in the same experiment. **Decision to resume A → B → A:** first reproduce an online neural learner meeting fresh-world competence across independent seeds, then compare retained, reset and simple memory baselines under stated interaction and compute budgets. Continue to show the active rule initially; memory should address a demonstrated need.

## 2. Measure the cost of more worlds

Measure world steps per second, wall-clock learning time, and peak memory for a stated number of worlds, observation shape, device, and thread count. Compare sequential execution with batching while checking seeded trajectories and evaluation outcomes. Report both equal-interaction and equal-wall-clock comparisons when throughput changes.

**Decision to proceed:** scaling produces a measured bottleneck or a useful gain without silently changing task dynamics. Parallel processes alone are not evidence of a more efficient simulator.

## 3. Test memory, then nested learning

Start with ordinary baselines: reset versus retained networks, replay across task phases, and a recurrent policy where the observation design warrants it. State what information and computational budget each method receives.

Nested learning or multiple learning timescales becomes a concrete follow-up only after a repeatable limitation survives these controls. Specify which parameters update at which timescale, what loss each update follows, and the extra memory/compute cost. A gate between two heads alone does not establish nested learning or learned specialization.

## Companion release: provenance

The [bounded provenance comparison](experiments/provenance-results.md) is implemented and preserved as a research preview: independently calibrated raw, deduplicated, and decayed signals; held-out seeds/scenarios; paired outcomes; and an explicit account of calibration access. Resource shortfall is the primary task outcome. False alarms alone can reward a policy that never acts. Decay has the lowest mean shortfall in the default scenario, while unique-origin counting has the lowest under the two selected shifts; there is no universal winner. The license remains undecided at the owner's request.

A useful release can demonstrate a result within this synthetic generator and include its limitations. It need not become a general theory of institutions, a large population of learning citizens, or a new message-passing framework.

## Preserve evidence and stop expanding by default

Use the [experiment template](../CONTRIBUTING.md#experiment-template) before a new comparison. Give each study one primary question, a fixed pilot budget, a decision criterion, and a preserved result even when it is negative. Treat historical reports as records of what was believed at the time; add dated corrections rather than silently rewriting the evidence.
