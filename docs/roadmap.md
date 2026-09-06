# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: useful fresh behavior with exact supervision

The [exact-target comparison](experiments/supervised_results_v1.md) trains the same 20,420-parameter network on exact four-action Q* values over 163,840 states from 256 layouts. After 30,000 updates per seed, fresh-layout greedy success is **82.81%, 85.94% and 87.50%**. All three seeds pass the declared supervised fresh gate on 64 layouts with no training-layout overlap. This establishes useful capacity under the supplied targets and coverage. The stricter training-fit gate remains unmet, and even successful fresh episodes average 11.29 steps with 52.67% blocked actions. Exact fitting and efficient navigation remain unresolved.

Historical frozen stream RL policies score **21.35%** on the same new panel; random actions score 44.27% and the planner 100%. These are contextual references with different training coverage, targets and computation, not a controlled target-only ablation. The prior RL policies had seen 3/3/4 of these layouts in their own training histories. Three initializations share one bank and panel, so these results are not independent task-bank replications.

The preceding [A-only RL comparison](experiments/competence_results_v1.md) learned the selected one-task and sixteen-task training sets perfectly but failed fresh-world competence. Earlier [adaptation v2](experiments/adaptation_pilot_v2.md) and [v1](experiments/adaptation_pilot_v1.md) also failed initial competence, so their later boundary changes do not establish forgetting or recovery. Those artifacts remain unchanged. The supervised milestone ends with its declared comparison and no extra tuning run.

## 1. Next bounded milestone: exact versus bootstrapped targets on fixed data

Hold the transition dataset, sampled state-action pairs, network, initializations, loss reduction and update budget constant. Compare **exact Q* targets against bootstrapped one-step targets**, with the same all-four-action loss in both conditions. Declare a separate protocol, a fresh evaluation panel, target-network schedule and per-seed success/efficiency criteria before execution. Preserve the fixed-data experiment's scope: even a successful bootstrap condition would not establish online collection and learning.

Why this comes next: the current representation can express useful fresh-layout behavior, so replacing it or adding memory is not the first explanation to test. Exact labels and broad state coverage changed together in this milestone. Keeping coverage fixed makes the target comparison more informative. If bootstrapped fitting also works, investigate online collection and replay next. If it fails while exact fitting succeeds, focus on target construction and optimization under that dataset. If both disappoint on a new bank, first assess reproducibility and coverage before escalating architecture.

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
