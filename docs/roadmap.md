# Research direction

Q6's primary question is: **when a task changes from A to B and back to A, what does a small neural learner retain, and what must it relearn?** Efficient simulation serves this experiment. The companion provenance study has a smaller, separate release boundary.

## Current result: similar success, different route quality, sensitive panels

The [equal-size comparison](experiments/equal_support_results_v1.md) fixes bank size at **59,626 states**, along with DDQN, four-action loss, initializations and update budget. Collected training reaches **83.85% fresh greedy success**, versus **82.81%** for one uniform subset. The two-episode primary gap is mixed across seeds and does not establish equivalence. Uniform support produces **67.71% efficient fresh success versus 44.79%**, with a route-efficiency advantage in every seed and shorter episodes. Both fresh gates pass; both stricter fit and efficiency diagnostics remain unmet.

Equal row counts cannot explain this observed efficiency difference, but composition includes map, clock, goal-near and successor-graph coverage. No single component is isolated, and one subset draw shared by three learner seeds is one bank. The uniform condition actually requires more detached successor queries outside its support. All four transitions and the full observation bank remain privileged offline access.

The collected control's weights and sampling reproduce the preceding [coverage study](experiments/coverage_results_v1.md) exactly. Its score changes from **68.23% on panel 950000 to 83.85% on panel 960000** without a learning change. Random/planner references also differ. Earlier within-panel comparisons remain valid, but one small panel's gate outcome should not become a general competence label.

The [fixed-target study](experiments/fixed_targets_results_v1.md) showed both exact and bootstrapped targets working under broad coverage. The [supervised study](experiments/supervised_results_v1.md) first established useful exact-target capacity. The [A-only RL comparison](experiments/competence_results_v1.md) and earlier adaptation pilots failed fresh or initial competence; their later A → B → A outcomes do not establish forgetting or recovery. Historical artifacts remain unchanged.

## 1. Next bounded milestone: evaluate frozen policies across fresh panels

Evaluate the **frozen collected, uniform and archived exhaustive policies on eight new, fixed, disjoint 64-layout panels**. Declare panel selection, exclusions, metrics and aggregation before execution. Reuse final checkpoints; perform no training, support redraw or tuning. Report panel-wise paired success, efficient success and mean steps, alongside pooled per-learner results and variation across panels.

Why this comes next: the same network's score moved by 15.63 percentage points between fresh panels, while the equal-size primary gap is only 1.04 points. A broader prospective evaluation can test how reliably the route-efficiency difference repeats and how sensitive the competence readout is to panel choice. It cannot retrospectively replace the old gates or replicate independent training/support banks. If the behavioral distinction persists, independent bank replications can follow before choosing a collection/replay intervention.

This follow-up has not run. **Decision to resume A → B → A:** first establish an online neural learner meeting a declared fresh-world competence criterion across independent seeds and a robust evaluation design, then compare retained, reset and simple memory baselines under stated interaction and compute budgets. Continue to show the active rule initially. Memory and continuous online feedback remain deferred during these controls.

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
