# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: the bootstrap action set changes route efficiency

The [constrained-bootstrap study](experiments/constrained_bootstrap_results_v1.md) keeps original recorded outcomes, exact replay, loss and target counts fixed. Restricting training-time successor argmax to logged actions improves efficient success **22.66% → 49.52%**, with bank effects **+27.80, +25.72 and +27.08 points**. All 24 bank-panel means and 72 learner-panel cells improve in efficiency. Success improves more modestly, **66.17% → 69.49%**, with 20/72 learner-panel cells declining. Fresh evaluation remains unrestricted over all four actions.

The restriction changes the chosen action on **60.11% of actual training queries** under treatment weights. Signed target changes average **−0.06680**, but 3.44% of all queries increase. These are same-weight treatment diagnostics, not a historical-control curve. The result supports the importance of the bootstrap action set under sparse recorded experience; it does not prove overestimation or unsupported-action extrapolation as the sole cause. Restriction can exclude useful actions, and unobserved current-action outputs remain unsupervised.

The [recorded-action study](experiments/recorded_actions_results_v1.md) previously exposed the decline after removing four-action outcome access. The [within-map study](experiments/within_map_results_v1.md) improved efficiency with map exposure fixed, following mixed [equal-map replay](experiments/map_replay_results_v1.md) effects and replicated [uniform-support gains](experiments/bank_replication_results_v1.md). Those privileged policies were not evaluated on the latest panels; their old scores do not establish a matched ranking. Earlier evidence and gates remain preserved.

## 1. Next bounded milestone: exact targets on the recorded graph

Use constrained recorded-action DDQN as the next offline baseline. Compare its moving approximate bootstrap targets against exact targets computed backward on the same logged transition graph. Remaining time decreases on each recorded nonterminal edge, every successor has a nonempty logged action set, and terminal rewards are recorded. This permits exact dynamic programming for the restricted graph without querying unrecorded outcomes or full-world Q*.

Keep original supports and recorded tables, current-action masks, exact 64-state replay, per-state loss, network, initialization, action-target presentations and 30,000-update budget fixed. Reevaluate frozen constrained controls against new fixed-target fits on common prospective panels; fresh policies still consider all four actions. The target preparation and neural successor-query counts will differ, so report those costs separately instead of claiming an equal query budget. Compare fit and action ranking against recorded-graph values, separate from full-world exact-value diagnostics.

This holds the restricted Bellman operator fixed while testing the difficulty of learning from moving approximate labels. The graph's exact values are not the full world's optimal values; remaining fresh-world error cannot automatically be assigned to one cause. This proposal has not run and needs its own protocol. Memory and continuous online collection remain deferred.

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
