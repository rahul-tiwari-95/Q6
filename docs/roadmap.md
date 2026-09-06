# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: route-efficiency advantage survives panel variation

The [frozen-policy evaluation](experiments/panel_evaluation_results_v1.md) tested nine archived networks on **eight new, disjoint 64-layout panels**. Collected, uniform-subset and exhaustive policies reach **78.19%, 80.73% and 84.90% greedy success**. Uniform improves efficient success over equal-size collected support on **all eight panel means**: **61.72% versus 46.42%**, a pooled **+15.30-point** difference. Mean episode length is lower throughout. Success favors uniform on seven panels and reverses on one; the behavioral distinction is strongest in route efficiency.

The same policies still vary considerably across panels: collected success spans 69.27–84.90%. Uniform efficiency improves in 22/24 panel-by-learner cells, not every cell. All panels share the same three saved learner seeds and the same support banks. This result establishes repeatability across these prospective panels, not independent bank replication, a general sampler ranking or a particular composition cause. Historical threshold crossings are descriptive; old gates remain unchanged and no new competence gate was introduced.

The preceding [equal-size comparison](experiments/equal_support_results_v1.md) fixed bank size at 59,626 states and first exposed the route difference. Map, clock, goal-near and successor-graph composition changed together. The [coverage study](experiments/coverage_results_v1.md) compared unequal bank sizes. The [fixed-target control](experiments/fixed_targets_results_v1.md) and [supervised diagnostic](experiments/supervised_results_v1.md) established useful behavior with broad offline coverage. The earlier [A-only RL comparison](experiments/competence_results_v1.md) and adaptation pilots failed fresh or initial competence; later A → B → A behavior cannot yet establish forgetting or recovery. Their evidence remains preserved.

## 1. Next bounded milestone: replicate the support banks

Create **three new fixed-budget exploratory banks**, each paired with one uniform subset containing the same realized number of unique states. Preserve network, DDQN, four-action loss, learner initializations and update budget. Declare panel selection and aggregation before training. Report each independent bank pair and its composition before pooling; a matched state count within a pair does not remove collector size/composition variation between pairs.

Why this comes next: the efficiency distinction survived panel choice, but all current evidence still relies on one collector history and one uniform-subset draw. Independent bank pairs test whether the result is particular to those inputs before choosing a collection/replay intervention. They do not by themselves isolate goal-near coverage, successor structure or another single cause. This follow-up has not run; its protocol and resource budget remain to be declared.

**Decision to resume A → B → A:** first establish an online neural learner meeting a declared fresh-world competence criterion across independent seeds and a robust evaluation design, then compare retained, reset and simple memory baselines under stated interaction and compute budgets. All current offline conditions retain privileged four-action transitions and full-bank successor access. Continue to show the active rule initially. Memory and continuous online feedback remain deferred during these controls.

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
