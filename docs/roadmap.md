# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: useful choices extend beyond the logs

The [familiar-start diagnostic](experiments/familiar_starts_results_v1.md) freezes both existing models and evaluates identical original starts with unrestricted versus logged-only actions. DDQN efficient success changes **68.49% → 40.76%**, and exact-label efficiency **25.82% → 24.05%**. The positive interaction reflects less harm to the exact model, not an efficiency rescue. Exact-label success improves under masking, but it still trails masked DDQN in both behavior metrics.

The regression survives familiar maps and selection restricted to supervised actions. Original-start ranking agreement falls **89.28% → 79.43%** despite much better numerical value fit; most of the full-support MSE gain corrects per-state value offsets. Independent source, recorded-graph and frozen-weight checks pass. This narrows the explanation without proving a unique cause of the training dynamics.

The recorded graphs permit **96.88% successful starts**, but only **60.94% efficient starts**. Unrestricted DDQN exceeds that efficiency ceiling, demonstrating useful routes beyond the logs. Hard deployment masks therefore discard opportunities as well as mistakes. Keep constrained DDQN as the baseline, with unrestricted fresh-world evaluation; familiar masks remain privileged diagnostics.

Earlier [exact-label](experiments/logged_graph_results_v1.md), [constrained-bootstrap](experiments/constrained_bootstrap_results_v1.md), [recorded-action](experiments/recorded_actions_results_v1.md) and [composition](experiments/within_map_results_v1.md) evidence stays intact. Scores from different evaluation populations cannot rank policies against one another.

## 1. Next bounded milestone: change the collection policy

Compare uniform-random complete episodes with one declared mixture of random and frozen-DDQN-guided complete episodes, across three collection draws on the same training maps. Fix the collector and mixture before outcomes. Match complete-episode allocations per map and report actual transitions and the collector's historical training cost separately; this is not an equal-interaction claim. Completing admitted episodes preserves the logged graph's successor closure.

Keep the DDQN network, learning procedure and update budget fixed. Measure logged shortest successful paths, success/efficiency ceilings and state/action composition, then test learned policies on common new fresh panels. This estimates the total collection-policy effect; route length, coverage, realized interactions and resulting replay composition may change together. A better graph ceiling without better learned fresh behavior is not sufficient progress.

This prioritizes better experience for the stronger baseline over another branch tuning the weaker exact-label loss. Ranking headroom remains an explicit diagnostic. The next collection experiment has not run and needs its own protocol. Continuous online feedback and memory remain deferred.

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
