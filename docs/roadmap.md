# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: recorded outcomes expose a learning limitation

The [recorded-action control](experiments/recorded_actions_results_v1.md) keeps original collected states and exact local/global/map replay, but supervises only actions actually logged. Efficient success falls **43.03% → 21.29%**, with bank effects **−25.20, −19.01 and −21.03 points**. Success falls **78.43% → 64.00%**. Every bank-panel mean declines; 71/72 learner-panel cells decline and one ties for each metric. These cells share banks, seeds and panels and are not independent replications.

The logs cover about **35% of the four possible actions per collected state**, averaged across those states, and treatment uses **24.27 million action targets versus 69.12 million** historically, with equal state/update budgets. All recorded nonterminal successor queries stay within collected support; counterfactual control queries do not. Removing action-outcome privilege jointly changes target count, effective action weighting, action coverage and successor access. The result does not prove overestimation or isolate one cause.

The [within-map study](experiments/within_map_results_v1.md) previously improved efficiency with map exposure fixed, following mixed [equal-map replay](experiments/map_replay_results_v1.md) effects and replicated [uniform-support gains](experiments/bank_replication_results_v1.md). Those policies were not evaluated on the new recorded-action panels; older scores do not supply a matched ranking. Their evidence and gates remain preserved.

## 1. Next bounded milestone: recorded-successor action constraints

Compare the new recorded-action baseline against a treatment changing only the nonterminal bootstrap action set. At each recorded successor, restrict online-network argmax to its nonempty logged-action set, then evaluate the chosen action with the target network. Keep original collected supports, recorded outcomes, exact 64-state replay, current-state masks, per-state loss normalization, network, initialization, action-target count and 30,000-update budget fixed. No new data or oracle targets are needed.

Fresh-map evaluation must remain greedy over all four predicted actions, because those states have no logged masks. Verify every queried successor has a recorded action; never silently fall back to unrestricted selection. Record how often unrestricted successor argmax chooses outside the logged set, and the signed target change under the same weights. Online selection and lagged target evaluation may disagree, so targets need not always decrease.

A benefit would show sensitivity to restricting the backup operator, not prove unsupported-action extrapolation was the sole cause. Sparse masks can exclude useful actions, and current unobserved-action outputs remain unconstrained; a negative result would not vindicate unrestricted backups generally. The proposal has not run and needs its own protocol. Keep recorded-action learning as the next baseline, with the four-action control retained as a privileged diagnostic.

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
