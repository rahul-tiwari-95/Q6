# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: equal-map replay helps inconsistently

The [replay control](experiments/map_replay_results_v1.md) reused three collected supports and nine archived controls, training nine treatments with equal map-selection probabilities. Efficient-success effects are **−3.91, +7.16 and +4.56 points**, with an equal-bank mean **42.17% → 44.77%**. Map exposure CV fell from about 0.25 to 0.01, confirming the intended intervention, while sparse-map states repeated far more often. Bank 1 worsened on six of eight panel means. Keep the original replay as the baseline; a pooled gain does not establish a reliable improvement.

The [earlier bank study](experiments/bank_replication_results_v1.md) replicated uniform-support efficiency gains across three draws. This study changes replay on the same supports and produces mixed effects; it does not isolate missing states, clocks, goal proximity or repetition as the cause. The previous uniform policies were not evaluated on these panels, so old scores cannot measure gap closure here. Historical evidence and gates remain preserved.

## 1. Next bounded milestone: within-map support composition

For each existing bank and training map, draw a uniform subset of exhaustive valid states with exactly that map's collected unique-state count. Fix each draw before predictions and share it across learner seeds. Preserve sorted map blocks, the original uniform-over-unique-state sampler and exact local draws, network, initializations, DDQN, four-action loss and 30,000-update budget.

Equal map quotas make every update's ordered map identities and batch diversity match the archived control; verify the ordered map digest, not only totals. Within-map current states change, including their combined position/clock/successor composition. This targets state composition while preserving map exposure. Nine new fits can be compared with the archived original controls on prospectively declared common panels; report each bank separately. This proposal has not run and needs its own protocol.

A repeatable benefit would motivate broader within-map coverage in a subsequent collector; mixed results would leave interactions unresolved. This remains privileged support access, not a practical collector demonstration. After this single control, prioritize the bridge to learning only from actually recorded actions/transitions over an open-ended series of balancing variants.

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
