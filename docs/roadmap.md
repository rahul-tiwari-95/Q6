# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: better recorded routes are insufficient

The [guided-collection comparison](experiments/guided_collection_results_v1.md) retains constrained DDQN and replaces eight of sixteen random episodes per map with greedy episodes from one frozen collector. Logged efficient-route availability rises **60.94% → 82.29%**, but fresh efficient success falls **49.13% → 43.32%**, with negative effects in all three bank averages. Fresh success also declines, **69.53% → 63.82%**.

The mixture uses 32.7% fewer collection interactions while losing 35.7% of the bank-indexed unique states and 43.5% of recorded state/action pairs. Eight guided repetitions duplicate one route per map; they provide no extra deduplicated loss weight. These changes make reduced exploration a plausible explanation, but coverage, action masks, replay composition and target counts changed together. Do not adopt this fixed mixture or conclude that guidance generally fails.

**Retain constrained DDQN trained on the original 16-random-episode banks.** Its bootstrap uses recorded successor actions; fresh action selection stays unrestricted. The earlier [familiar-start diagnostic](experiments/familiar_starts_results_v1.md) showed useful choices beyond the logs and an exact-label ranking regression that masking did not fix. All earlier [exact-label](experiments/logged_graph_results_v1.md), [bootstrap](experiments/constrained_bootstrap_results_v1.md), [recorded-action](experiments/recorded_actions_results_v1.md) and [composition](experiments/within_map_results_v1.md) evidence remains intact. Different evaluation panels cannot directly rank policies against one another.

## 1. Next bounded milestone: recover the missing comparison

Derive **eight-random-only** banks from exactly the existing retained episode slots 0–7, without recollection. Fit nine policies with the unchanged constrained-DDQN network, objective, replay algorithm and update budget. Evaluate them alongside the frozen 16-random and mixed policies on one prospectively declared common family of fresh panels.

The primary contrast, mixed minus eight-random-only, measures the effect of adding these guided records to the retained experience. Eight-random-only minus 16-random measures removing the other eight random episodes. Together they equal the full mixture effect on the same evaluation data. This can show whether guidance helps, hurts, or fails to recover lost performance; it is not a mediation proof or a control fixing realized replay and action-target counts.

The collector's historical knowledge and cost remain explicit. Three banks share its guided routes and the training maps, so they are conditional draws, not independent collector replications. A later full-random-plus-guidance augmentation could test benefit with exploration retained, but first fill the missing comparison. No new mixture sweep, collector fitting, continuous online learning or memory is included in the next milestone, which has not run.

**Decision to resume A → B → A:** first establish an online neural learner meeting a declared fresh-world competence criterion across independent seeds and a robust evaluation design, then compare retained, reset and simple memory baselines under stated interaction and compute budgets. Earlier offline conditions use privileged four-action outcomes; the latest treatment uses recorded outcomes but still learns from offline deduplicated state replay. Continue to show the active rule initially. Memory and continuous online feedback remain deferred during these controls.

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
