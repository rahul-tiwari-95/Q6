# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: fixed tasks learned, fresh-world competence missing

The [A-only comparison](experiments/competence_results_v1.md) completed 1,080,000 training transitions under its [separate protocol](experiments/competence_protocol_v1.md). The existing Double DQN reached **100% greedy success on the one-task and sixteen-task training sets for every learner seed**. All seeds also followed shortest paths on those probes at the final checkpoint.

Fresh-map success was **9.90%**, **22.40%**, and **19.79%** after one-task, sixteen-task, and stream training. The common random reference scored 42.45%; the shortest-path reference solved all maps. No neural condition met the gate of 70% fresh-map success in every seed and above random. These are descriptive outcomes on one shared task bank, not a causal diagnosis or a general random-versus-neural effect.

The earlier [adaptation v2](experiments/adaptation_pilot_v2.md) and [v1](experiments/adaptation_pilot_v1.md) remain unchanged. Both failed initial competence, so later boundary changes do not establish forgetting or recovery. The new diagnosis establishes that selected repeated tasks are learnable; it does not yet supply a competent cross-world baseline. This milestone ends with the declared comparison and no extra tuning run.

## 1. Next bounded milestone: separate target learning from online RL

Use the exact finite-horizon reference already built for evaluation to ask whether the **existing network and observations can learn navigation across tasks with reliable supervised targets**. Declare a new fixed task/state dataset, training budget, untouched evaluation panel, and per-seed decision criterion before running it. Keep demonstrations explicitly separate from online RL evidence.

Why this comes next: repeated-task learning works, while stream training produces inaccurate action values and frequent blocked actions on fresh maps. Those observations do not distinguish limitations of the representation from difficulties in exploration or bootstrapped learning. Supervised targets remove those two online complications for a focused diagnostic. Good fresh-map performance would direct attention toward the RL procedure; weak performance would motivate a controlled spatial-representation comparison, with optimization and coverage still possible explanations.

This follow-up has not run. Do not add memory or nested networks based on the current scores. **Decision to resume A → B → A:** first reproduce a neural learner meeting the fresh-world competence criterion across independent seeds, then compare retained, reset, and simple memory baselines under stated interaction and compute budgets.

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
