# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: initial competence is still missing

[Pilot v2](experiments/adaptation_pilot_v2.md) completed **1,080,000 training transitions** with the same compact Double DQN learner and task as [archived v1](experiments/adaptation_pilot_v1.md), at ten times the interaction budget. Mean success on A after initial A training was **31.25%**, below the locally predeclared **70% competence gate**. The study therefore remains `baseline_underlearned`: subsequent boundary changes do not support a forgetting, adaptation, or recovery claim.

[Post-hoc diagnostics](../experiments/adaptation/pilot_v2/diagnostics.json) on the same A evaluation panel gave **41.67%** success for seeded random actions and **100%** for the shortest-path controller. These are descriptive references: the planner demonstrates that the tested maps are solvable within the horizon, while the neural learner has not established useful initial competence. They are not predeclared outcomes or evidence of a statistically established random-versus-neural effect.

Training success is not fixed-policy evaluation success. For the first A phase:

| Training seed | Last 100 completed training episodes | Greedy evaluation on 32 fixed A maps |
| --- | ---: | ---: |
| 0 | 49% | 31.25% |
| 1 | 47% | 34.375% |
| 2 | 29% | 28.125% |

The training and evaluation columns differ in map samples and exploration, so their gap does not isolate generalization. Raw episode results and settings are preserved with the pilot.

The [v2 protocol](experiments/adaptation_protocol_v2.md) changed the evaluation panel because v1's maps had informed diagnosis; the larger training budget also stretched the existing epsilon schedule. The v1→v2 comparison therefore does not isolate the causal effect of extra gradient steps. **This milestone ends with these two runs; there is no third tuning run.**

## 1. Next bounded milestone: establish competence on A

Write a separate A-only diagnostic protocol before further training. Check observations, rewards, learner targets, and greedy trajectories on small, inspectable cases; compare with the existing random and shortest-path references under the same horizon. State one diagnosis to test, a fixed interaction/time budget, fresh evaluation seeds, and a stopping rule. If task simplification is necessary, record it as a new condition instead of comparing its score directly with these pilots.

**Decision to proceed:** a learner reliably meets the initial-task competence criterion on evaluation maps across independent seeds, with preserved artifacts and reproducible execution. Only then resume A → B → A and compare retained, reset, and simple memory baselines under stated data and compute budgets. Increasing architecture complexity before that point would make the current failure harder to diagnose.

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
