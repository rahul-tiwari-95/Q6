# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: exact recorded values improve fit but harm behavior

The [exact logged-graph study](experiments/logged_graph_results_v1.md) computes targets backward using only recorded transitions and allowed actions. State-balanced value MAE improves **0.2473 → 0.1134**, but fresh efficient success falls **48.70% → 18.64%** and success **68.84% → 44.36%**. Every bank-panel mean and all 72 learner-panel cells decline in both behavior metrics. Data, action masks, exact replay, loss, network and target presentations stay fixed; optimizer neural successor queries become zero, with preparation/inference costs reported separately.

Restricted action agreement slips **97.51% → 97.05%** despite better value fit. About **69% of supported states have one logged action**, making their restricted agreement automatic. Both models' unrestricted argmax chooses outside the logged set on about **65% of supported states**. Recorded-graph fit therefore describes only part of the deployed decision. The result does not establish a single cause, prove implicit regularization or imply exact labels generally harm learning.

Keep [constrained recorded-action DDQN](experiments/constrained_bootstrap_results_v1.md) as the stronger behavioral baseline, and the exact graph as a useful diagnostic reference. Earlier [recorded-action](experiments/recorded_actions_results_v1.md), [within-map composition](experiments/within_map_results_v1.md), [map-replay](experiments/map_replay_results_v1.md) and [bank-replication](experiments/bank_replication_results_v1.md) findings remain preserved. Policies absent from the new panels cannot be ranked using older scores.

## 1. Next bounded milestone: familiar-start frozen-policy diagnosis

Before another training comparison, evaluate both existing frozen arms from every original training map/spawn at clock 32, shared within each bank/seed comparison and chosen without outcome filtering. Compare unrestricted all-four greedy choices with logged-mask greedy choices. Masked trajectories must remain in the closed recorded graph; track off-mask choices while supported and first support exit separately for unrestricted trajectories.

Add an exact logged-graph policy reference to show what its available actions permit. It is optimal for the graph's declared return, not automatically a universal success ceiling without a reachability check. Keep this familiar-state experiment separate from the existing fresh-map results. Logged masks are privileged diagnostic information on familiar support and are not an assumed deployable mask for fresh worlds.

A disproportionate rescue of exact-label policies would support a supervised-action/deployment-selection mismatch. Weak masked behavior would focus attention on graph policy/ranking limitations before blaming fresh-map transfer. Either result can localize the failure without another fit or collection budget, but cannot uniquely explain every fresh-world failure. This proposal has not run and needs its own protocol. Memory and continuous online collection remain deferred.

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
