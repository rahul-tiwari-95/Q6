# Better recorded journeys — protocol v1

Status: locally declared before main execution, not externally registered.
Keep constrained DDQN after the familiar-start investigation. Test whether a
fixed collection-policy mixture improves fresh-map behavior under the same
learner and complete-episode allocation. Memory remains deferred.

## Collection intervention and historical control

Control `constrained_bootstrap` (random collection) reuses the original banks **1, 2, 3** from
`experiments/bank_replication/pilot_v1` and the nine final constrained-DDQN
snapshots from `experiments/constrained_bootstrap/pilot_v1`, learner seeds
**0, 1, 2**, **30,000 updates** each. Verify all source logs, support, recorded
tables, sampling evidence and model identities. No baseline recollection,
retraining or checkpoint selection. Original bank interactions are
**100,878 / 100,326 / 100,381**, totaling **301,585**, from **4,096 complete
episodes per bank**, 16 on each of the same **256 maps, 300000–300255**.

Treatment `guided_collection` (mixed collection) collects 16 complete episodes per map:

- Repetitions **0–7** use uniform random actions with the original independent
  stream `SeedSequence([map_seed,repetition,77301,bank_id])`. Verify these slots
  reproduce the corresponding archived random episodes exactly.
- Repetitions **8–15** use greedy, lowest-label argmax over all four outputs of
  one frozen collector: **bank 1 / learner seed 0 / update 30000 constrained
  DDQN**. This is a deterministic index choice, not a best-score selection.
  No epsilon within guided episodes, action mask, oracle action, fitted update
  or collector refresh. The same collector is used across all three draws.

Reset every episode to the original spawn and clock 32. Finish every admitted
episode through actual termination or truncation; do not stop it to equalize
interaction counts. Resource interruption disqualifies the run rather than
turning an unfinished transition into an artificial terminal. Record every
action, reward, successor, clock, end flag and collection-policy label.

Eight deterministic guided repetitions on the same map follow the same route.
Count all their interactions, and report distinct routes separately. They do
not create eight independent demonstrations or extra deduplicated loss weight.
The three treatment banks vary through their eight random episodes and share
guided experience; training maps, learner seeds and evaluation panels are also
shared. Do not treat crossed cells as independent replications.

The collector already consumed bank 1's full historical 16-episode experience,
including the random slots replaced here: **100,878 interactions**, **30,000
optimizer updates / 1,920,000 sampled-state presentations**. Report its archived
action-target count and model provenance too. This knowledge and prior compute
are additional inputs, not free guidance or an equal-total-history comparison.
New collection has **12,288 episodes** and at most **393,216 interactions**;
actual interactions, successful episodes, lengths and blocked moves are outcomes.

## Unchanged learner; changed experience

Preserve fully visible World A and show the movement rule initially: 5×5,
four walls, one pellet, horizon 32, the existing rewards/potential shaping,
92 observations and **20,420-parameter 92 → 128 → 64 → 4 ReLU network**.
Use matching seed-specific initial online/target weights, Adam **0.001**,
gradient clipping **5**, gamma **0.97**, soft target **tau 0.01** and the
unchanged `constrained_bootstrap.constrained_update` procedure.

Reconstruct deterministic distinct recorded `(state,action)` outcomes from
actual logs. Preserve occurrence counts separately. Deduplicate current states
and actions; repeated collection visits have no extra replay/loss weight.
Every live logged successor must be a supported state with a nonempty logged
action mask, in the same map with one fewer remaining move. Missing outcomes
stay absent; reject inconsistent duplicates or missing successor masks.

Sample 64 distinct support-local positions per update using the existing
owned `SeedSequence([learner_seed,66301])` stream. Average SmoothL1 beta 1 over
each sampled state's distinct recorded actions, then average across states.
For each nonterminal recorded edge, select the online argmax among actions
logged at its successor and evaluate it with the lagged target network.
Terminal edges use their recorded reward without successor queries.
Exhaustive observation rows may resolve recorded states; full-world outcomes
and exact values are integrity/reference diagnostics, never training targets,
collection choices or sources of additional logged actions.

Nine new fits × **30,000 updates** = **270,000 updates / 17,280,000 sampled
states**, matching the historical controls' learning budget. Save checkpoints
at **0 / 1,000 / 3,000 / 10,000 / 30,000**, evaluate only the final policies.
There is no requirement for identical state/map minibatches across arms:
different support sizes change the realized replay. Reconstruct historical
control sampling and compare with its archive; record each new stream's local,
global and map digests, counts and coverage. Count actual supervised action
targets, terminal targets and nonterminal successor queries separately.
The control's **24,273,701 action targets** are historical; treatment target
counts need not match. Report zero direct samples and queried successors
outside each arm's own support. Preserve initial/final parameter identities.

## Recorded routes and fresh behavior

Before fitting, compute success reachability and shortest successful path
through each recorded graph by backward induction over remaining clocks 1
through 32; every live successor has one fewer move. Only
logged successful terminal edges and logged nonterminal successors contribute.
Preserve arrays and per-original-start success/efficient-success ceilings;
full-world shortest paths define efficiency (success within twice the shortest
route, failures included), but never supply missing graph edges. Report unique
states/actions, action-mask sizes, per-map counts, remaining-clock composition,
and collection success/steps/blocked moves, by bank and policy component.

Fresh evaluation uses **eight new panels × 64 maps**, starts
**1140000 + 1000 × panel index**, accepting ascending seeds. Exclude training
layouts, layouts in all prior fresh studies through `logged_graph`, and earlier
accepted panels using the established layout identity. The familiar-start
study adds no fresh maps. Both arms receive identical fresh starts and the
same greedy and epsilon-0.1 evaluation schedule: one greedy and two epsilon
episodes per map/learner; epsilon stream
`SeedSequence([learner_seed,map_seed,repetition,55219])`. Fresh action selection
always considers all four actions, with no logged mask.

This gives **27,648 learner evaluation episodes** (2 arms × 3 banks × 3 seeds
× 512 maps × 3 rollouts). Shared random and shortest-path references follow
the existing fresh-panel schedule: **3,584 reference episodes / 31,232 total
evaluation episodes**, separate from learner counts.
Primary comparison: **mixed minus random greedy efficient success**, per bank,
then equal-bank mean/range and panel signs. Preserve paired map/seed/panel
reductions, success, all-episode steps and returns. Successful-only steps use
different successful subsets and are labeled separately. No new competence
gate, significance claim or checkpoint tuning.

Better recorded routes without better learned fresh-map behavior do not meet
the experiment's purpose. An efficiency gain repeating across all three bank
averages supports this fixed mixture as a candidate collection baseline; mixed
or negative effects remain evidence and motivate a narrower control before
adoption. This estimates the total collection-policy effect, not route length
alone: coverage, action masks, unique-state count, actual interactions and
resulting replay composition can all change. Collector provenance overlaps
bank 1, and guided routes are shared; no independent general method ranking.
This is frozen-policy collection followed by offline learning, without
continuous online feedback, online-RL competence claims or memory.

## Evidence, dashboard and bounded execution

Output `experiments/guided_collection/pilot_v1` and
`dashboard/data/guided_collection.json`. Capture protocol/source/git/runtime,
commands, archive hashes, collector and learner snapshots, raw collection
logs and complete episodes, recorded graphs/support, route ceilings, replay
counts/digests, training losses, target counts, fresh episodes/reductions,
resources and a SHA-256 manifest. Preserve all previous evidence byte for byte.
Independent audit checks the actual logged transitions, shared random slots,
greedy collector decisions, frozen identities, graph recurrences, training
budget/exposure and paired result reductions. Portable audit uses archived
predictions; full audit regenerates each unique guided-collector decision once
per training map, plus selected fresh replay predictions, without another fit,
new collection or complete newly trained-policy evaluation. Use float64 for
new reductions.

Add **Guided collection** to the existing coverage dashboard, preserving all
**2,938 prior recordings**. Keep collection route quality, interaction costs
and learned fresh behavior distinct. Retain the established **304** fresh
recordings: first accepted map of each panel, every bank/seed/arm/mode with
epsilon repetition zero, plus the two shared references. Also predetermine
training maps **300000 + 32 × i**, i=0…7, and collection slots **0 and 8** for
both arms and every bank: **96 collection recordings**, **400 new recordings
total**. No favorable substitutions; show policy component and movement rule.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU with
one Torch thread and `nice -n 10`. Admission cap **1,200 seconds**, including
collection, preparation, fitting, references and evaluation. Sampled process
RSS guard **4 GiB** is not a hard OS or machine-wide cap. Preserve partial
evidence on guard/identity/nonfinite failure and mark it ineligible. Review and
push this protocol, then freeze clean pushed source before **one main run**.
PR #1 stays open; license remains undecided.

Smoke retains all 256 collection maps and 16 episodes per map, but uses bank
1, learner seed 0, **24 updates**, checkpoints **0 and 24**, and two panels of
two maps starting at **1200000 / 1201000**. It checks collection closure and the whole
pipeline, including unequal support and budget accounting; historical control
still has 30,000 updates, so smoke is not a fair research comparison. Smoke
and synthetic unit tests write only separate temporary outputs. No main rerun
or mixture sweep after inspecting results.
