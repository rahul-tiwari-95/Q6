# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: the efficiency advantage repeats across bank draws

The [independent bank study](experiments/bank_replication_results_v1.md) trained three newly collected supports, each paired with a uniform subset of equal size. Uniform efficient success improves by **17.12, 27.67 and 23.31 points**; all 24 bank-panel means favor uniform. Across eight new 64-map panels, equal-bank success is **83.57% versus 78.54%**, efficient success **65.47% versus 42.77%**, and mean steps **11.51 versus 15.90**. One learner-panel efficiency cell reverses. The three bank pairs share training layouts, evaluation panels and learner initializations; they are not independent world-family draws.

The [preceding frozen-policy study](experiments/panel_evaluation_results_v1.md) established robustness across panel selection. Together these support a composition effect for the tested bank-construction procedures, without identifying one causal component. All historical evidence and gates remain preserved. Initial online competence remains unresolved.

## 1. Next bounded milestone: balance replay across maps

Reuse these three archived collected supports. Compare current uniform-over-unique-state replay with **64 distinct uniformly selected maps per batch, one uniformly selected supported state per map**. Keep the network, DDQN, learner initializations, four-action loss and 30,000-update budget fixed. Declare fresh evaluation panels before fitting; evaluate archived baseline policies on the same panels. Report each bank and actual exposure before pooling. No new collection or support augmentation is needed.

Why this comes next: collected supports contain as few as **7–9 states on some maps versus more than 300 on others**. Sampling uniformly over unique states therefore gives unequal map exposure. This intervention tests whether existing experience can support better routes when replay changes. It also changes within-batch map diversity, so it does not isolate map weighting alone. A gain would show that missing states are not the whole explanation; a failure would leave within-map composition and missing support unresolved. Clock and goal-near balancing are separate hypotheses. This recommendation has not run; its protocol and resource budget remain to be declared.

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
