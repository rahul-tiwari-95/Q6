# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: bootstrapping works under broad fixed coverage

The [paired fixed-data comparison](experiments/fixed_targets_results_v1.md) holds training states, all four sampled actions, initial weights, network, loss, optimizer and update budget fixed. On a new 64-layout panel, **exact targets reach 84.38% greedy success and Double DQN targets reach 86.46%**. Every seed in both conditions passes the declared offline fresh-success gate. All three exact-arm final weights reproduce the preceding supervised study exactly, and both arms have identical sampled batches within each seed.

DDQN produces more efficient successes (**66.15% versus 39.06%**, using at most twice each map's shortest-path length), but both miss the 80%-in-every-seed efficiency diagnostic. DDQN's training fit passes in two seeds and narrowly misses action agreement in seed 0; the every-seed gate remains unmet. Its absolute Q error is higher while action agreement is better. The small paired success gap does not establish superiority or equivalence. Both arms have privileged exhaustive offline coverage, so this is not online-RL competence.

The [preceding supervised study](experiments/supervised_results_v1.md) first established useful fresh-layout capacity with exact targets; the [A-only RL comparison](experiments/competence_results_v1.md) learned selected repeated tasks but failed fresh-world competence. Earlier [adaptation v2](experiments/adaptation_pilot_v2.md) and [v1](experiments/adaptation_pilot_v1.md) also failed initial competence, so their later boundary changes do not establish forgetting or recovery. Those artifacts remain unchanged. Each milestone ends at its declared budget without an extra tuning run.

## 1. Next bounded milestone: bridge exhaustive coverage and collected experience

Collect a fixed state bank with a declared exploratory trajectory policy, then compare that bank with exhaustive state coverage using the **same DDQN target/update procedure and all-four-action loss**. Hold network, initialization and optimization budget fixed. Supplying all four transitions at each visited state keeps four-action coverage per sampled state fixed; that counterfactual access remains privileged. Declare collection budget, replay weighting, successor handling, a new evaluation panel and per-seed success/efficiency gates before execution.

Why this comes next: the specified bootstrap procedure works with broad coverage, weakening an explanation based on bootstrapped targets alone. A fixed collected-state comparison tests the effect of state coverage before introducing feedback between a changing policy and its data. If collected-state replay works offline, test online collection and updating next. If it fails while exhaustive coverage works, inspect missing states, clocks and replay weighting. Avoid changing target mechanics or adding memory at the same time as coverage.

This follow-up has not run. **Decision to resume A → B → A:** first reproduce an online neural learner meeting the fresh-world competence criterion across independent seeds, then compare retained, reset, and simple memory baselines under stated interaction and compute budgets. Continue to show the active rule initially; memory should address a demonstrated need.

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
