# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: within-map composition improves efficiency

The [within-map control](experiments/within_map_results_v1.md) replaced states while preserving every map's exact collected quota, original replay schedule and ordered map identities in each batch. Efficient-success effects are **+19.47, +28.65 and +20.70 points**, with an equal-bank mean **45.75% → 68.68%**. All 24 bank-panel means and all 72 bank/panel/learner cells improve. Those cells share banks, seeds and panels; they are not independent replications. Exact map exposure and batch diversity checks passed.

The [earlier bank study](experiments/bank_replication_results_v1.md) replicated uniform-support efficiency gains; [equal-map replay](experiments/map_replay_results_v1.md) subsequently helped two banks and harmed one. The new result supports within-map composition as a useful collection target while retaining the original replay baseline. Position, clock, goal proximity and successor structure change together, so no individual cause is isolated. Uniform and map-balanced policies were not evaluated on these new panels; old scores cannot measure gap closure here. Historical evidence and gates remain preserved.

## 1. Next bounded milestone: recorded-action supervision

Use the original collected supports and archived collection logs to compare four-action supervision with supervision restricted to actually recorded actions. Verify repeated state/action records have identical rewards, successors and end flags, then deduplicate them into an observed-action mask. The new synthetic within-map supports cannot supply recorded actions for every row and are not this comparison's inputs. No new collection is needed.

Keep the exact original 64-state replay draws, network, initialization, DDQN optimizer and 30,000-update budget. Average loss over the recorded actions per state, then over the batch's states. Supervised rewards and successors must come from logs; exhaustive rows may identify observed successor observations, not supply counterfactual transitions. DDQN can still rank all predicted actions at those observed successors, retaining the possibility of unsupported-action extrapolation. Report action counts and target presentations explicitly: equal state/update budgets do not mean equal action-target counts.

Compare nine new masked-action fits with nine archived four-action controls on prospective common panels, reporting each bank separately. A drop would quantify the cost of removing counterfactual supervision and motivate action-coverage or offline-learning work. Comparable behavior would support moving toward raw trajectory replay and online collection. This remains offline, deduplicated state replay. The proposal has not run and needs its own protocol; avoid an open-ended sequence of further balancing variants.

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
