# Recorded-action supervision on fixed collected states — protocol v1

Status: locally declared before main execution, not externally registered.
Within-map replacements improved efficiency with map exposure and replay
fixed, but all preceding offline arms received outcomes for all four actions.
This comparison removes counterfactual action outcomes from direct training.

## Question and fixed comparison

Primary: **recorded_actions minus collected_unique greedy efficient-success
rate**, per bank and then the equal-bank mean. How does behavior change when
only actions actually recorded at a sampled state receive supervision, keeping
the original collected states, replay schedule, network and update budget?

Reuse original collected banks **1, 2, 3** from
`experiments/bank_replication/pilot_v1`: **59,839 / 59,984 / 59,838 states**
on maps **300000–300255**. Both arms have exactly the same support. The recent
synthetic within-map replacements are not training inputs to this comparison.
Control `collected_unique` uses its nine archived final 30,000-update models,
seeds **0, 1, 2** per bank, with no retraining or checkpoint selection.
Treatment `recorded_actions` trains nine new policies from identical archived
seed-specific online/target initializations.

Banks share training maps, learner seeds and new evaluation panels. Report
each bank before equal-weight means, ranges and signs. Crossed cells are not
independent replications. No significance, equivalence or population-wide
algorithm ranking is declared. Memory and online collection remain deferred.

## Recorded outcomes and access boundary

Before fitting, verify each bank's `collection_steps.csv` against its archived
manifest and preserve the input bytes and hash. Use logged current row, action,
reward, next row, terminated and truncated flags to build a deterministic
observed state/action table. Verify valid row/action ranges, finite rewards,
map and clock identities, end flags and successor semantics. Ended transitions
have successor −1; nonterminal successors must be valid recorded row IDs on
the same map with one fewer remaining step.

Repeated `(current_row, action)` records must agree on reward, end flags and
successor. Deduplicate after checking; **do not frequency-weight** repeated
visits. Every original supported current state must have at least one observed
action, and no outside-support current row may be added. Reject inconsistent
or missing evidence; never synthesize an unobserved action or silently drop a
supported state. Freeze all recorded tables and both-arm supports before fits.
There are **zero new collection steps, episodes or support draws**.

The unchanged exhaustive observation bank resolves logged current and
successor row IDs. Full four-action transitions and exact Q* may be copied and
validated for archived-control integrity, descriptive coverage and rollout
diagnostics. **They must not supply treatment rewards, ends, successor choices,
targets or loss masks.** Treatment optimization takes observations, the
record-derived table and replay indices, without an exact-target or exhaustive
transition input. Preserve recorded-table hashes and the provenance boundary.

## Learning objective

Unchanged fully visible World A: 5×5, four walls, one pellet, horizon 32,
ordinary movement mapping visible from frame zero, 92 observations, existing
rewards/potential shaping and gamma **0.97**. Preserve the **20,420-parameter**
92 → 128 → 64 → 4 ReLU network, Adam **0.001**, gradient clipping **5**, soft
target update **tau 0.01**, deterministic CPU and 30,000 optimizer updates.

For every sampled state `s`, let `A_log(s)` contain its distinct recorded
actions. For each such action, use its logged reward as the target when ended.
Otherwise add gamma times the target-network value at the online-network
argmax over all four predicted actions at that **logged successor**. Detach
all targets. Terminal transitions never index successor −1 or query a network
at a manufactured next observation.

The loss is:

`mean_over_64_states(mean_over_A_log(s)(SmoothL1_beta1(Q(s,a), target(s,a))))`.

This preserves equal state weighting. A global mean over observed state/action
pairs would overweight states with more recorded actions and is not this
protocol. Unobserved outcomes never enter arithmetic; predicting all four
outputs at current/recorded-successor observations is allowed. DDQN argmax
still includes actions without recorded outcomes there, retaining offline
extrapolation difficulty. No action restriction, conservative penalty,
frequency weighting or additional algorithm modification is introduced.

The loss family, optimizer, state presentations and update budget are fixed;
the supervised action set, effective per-action weighting and action-target
count necessarily change. Do not claim equal action-target budgets. The
archived control used four targets per state, **69,120,000 targets** over nine
fits. Report treatment target count exactly from actual sampled rows and
their mask cardinalities, along with terminal/nonterminal target counts and
recorded-successor query counts. Exact Q* never trains or selects this mask.

## Exact original replay and work order

Use unchanged `SupportSampler` / `BatchSampler`, owned
**SeedSequence([learner_seed,66301])**, selecting 64 distinct support-local
positions per update in returned order. Repeats between updates are allowed.
Both arms have identical sorted supports, so local, global current-state and
ordered map streams must all match within each bank/seed.

Reconstruct each original control's full **30,000-update** replay without
training, checking archived local/global digests and count vectors. Verify all
treatment local/global/map digests and count vectors against the corresponding
same-budget control stream. No cross-bank identity is assumed. Assert
updates × 64 presentations and zero direct off-support sampling. Preserve
map/clock/category exposure separately from observed-action availability and
actual successor queries; four-action structural diagnostics are not treatment
query counts. Complete logged episodes imply that each nonterminal successor
appears as a current state on the next step. Verify the expected zero recorded
successor queries outside current-state support rather than assuming it.
The control's counterfactual successors can be outside support, so successor
query composition also changes with removal of unrecorded action outcomes.

Verify archive data, logs, frozen controls and initial identities; freeze all
supports/tables and select all panels before any fit or learned prediction.
Execute sequentially by bank then seed: **nine new fits / 270,000 updates /
17,280,000 state presentations**. Baselines receive zero new updates; their
270,000 historical updates and collection costs are labelled historical.

Save treatment inference snapshots at **0, 1,000, 3,000, 10,000, 30,000**:
**45 new snapshots plus nine copied control finals**. Log 100-update loss
windows, **2,700 raw rows**, without inventing new control losses. All nine
fits finish before any learned-policy evaluation. Evaluate final policies
only, with no gradients, optimizer or replay, verifying file/online/target
identities around each panel. No full-state fitting sweep or gate is added.

## Prospective evaluation and reporting

Select **eight disjoint 64-map panels** by ascending scan from
**1080000 + 1000 × panel index**. Exclude training, the named four previous
fresh panels (supervised/fixed-target/coverage/equal-support), and all 512-map
panels from each of frozen-panel evaluation, bank replication, map replay and
within-map composition. Also exclude every already accepted new layout.
Identity is walls + pellet + rules, ignoring seed/spawn/clock. Preserve
original spawns and all rejected candidates; do not balance difficulty or
consult outcomes. This does not claim exclusion of every historical Q6 task.

Each model/map receives one greedy and two epsilon-0.1 episodes, with paired
external draws **[learner_seed,map_seed,repetition,55219]** across arms/banks
and lowest-label greedy ties. Total **27,648 learner episodes**: 9,216 greedy
and 18,432 epsilon. Shared references comprise **3,072 random episodes**
(three seeds × two repetitions × 512) and **512 planner episodes**, giving
**31,232 evaluation episodes**. Retain rollout Q* diagnostics separately from
training; no intermediate evaluation or additional tuning run.

Efficient success is reaching the goal within twice the map's planner length;
**failures remain in the denominator**. Each bank/condition has 1,536 greedy
and 3,072 epsilon episodes. Report success, efficient success, all-episode
steps and blocked steps separately. Keep **4,608 greedy layout pairs**, **81
seed-pair rows** (72 panel plus nine all-panel), and **27 aggregate pairs**
(24 panel plus three all-panel). Do not manufacture epsilon paired rows.
Equal-bank success/efficiency/step means equal pooling because counts match;
pooled blocked-step ratios, mean bank ratios and mean learner ratios remain
distinct. Successful-only route means compare different successful subsets.

Historical 70% success/above-random and 80% efficient-success reference
crossings remain descriptive, with no new competence gate or rewritten old
fit result. Eligibility indicates integrity/completeness, not good performance.
Previous synthetic within-map/uniform/map-balanced policies are not arms on
these new panels; their old scores cannot establish matched-panel rankings.

A repeatable decline would show the cost of removing counterfactual action
supervision under these fixed supports, state schedules and optimizer budgets.
It would motivate recorded-action coverage or offline-learning work, without
identifying target count, action imbalance or extrapolation as the sole cause.
Comparable or better behavior would support moving toward actual trajectory
replay and then online collection. This is still offline, deduplicated state
replay with observed action masks, not raw frequency-weighted trajectory replay
or evidence of online-RL competence. No new experiment follows automatically
within this fixed milestone; memory remains deferred.

## Dashboard, resources and reproducibility

Add **Recorded actions** to the existing dashboard, preserving all earlier
studies and initially visible rules. Show support/replay identity, observed
actions per state (counts 1–4), distinct state/action coverage, actual action-
target presentations, terminal/nonterminal target and recorded-successor query
counts, and per-bank/final-panel outcomes. Distinguish the control's historical
four-action exposure from new treatment exposure and ordinary predicted
all-action argmax from access to unrecorded outcomes.

Preselect first accepted map per panel, repetition zero, all bank/arm/learner
combinations in both modes: **288 learner plus 16 shared reference recordings
= 304**. No favorable trace substitution or intermediate checkpoint selector.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, one CPU thread,
deterministic operations, sequential execution at `nice -n 10`. Admission
cap **1,200 seconds** covers preparation, training, evaluation and aggregation;
sampled process-peak-RSS guard **4 GiB** is not a hard OS allocation or total-
machine limit. Nonfinite values, mismatched identities, missing work or guard
failures preserve partial artifacts and disqualify evidence. Run one main
comparison from clean, pushed source after protocol review and publication.

Capture source/protocol/git/environment/command, input manifests and logs,
recorded tables/masks/hashes, observations and diagnostic transitions, exact
supports, sampling digests/counts/reconstruction, action exposure, models/losses,
panels/rejections, raw episodes/pairs, summaries/replays/resources and SHA-256
manifest. Output `experiments/recorded_actions/pilot_v1`; dashboard export
`dashboard/data/recorded_actions.json`. License remains undecided.

Smoke uses all three shipped banks and logged action tables, seed 0 only,
**24 updates per bank = 72**, six new snapshots (0/final) and three archived
30,000-update finals. Two two-map panels start **1160000 / 1161000**, producing
**72 learner + 12 reference episodes** and **28 recordings**. Verify treatment
replay against the first 24 historical updates while retaining baseline
30,000-update weights/exposure labels. Unequal smoke budgets and alternate
panels are explicitly ineligible scientific evidence. Tests may use smaller
fixtures. Audit saved evidence and selected replays without retraining or
repeating the complete learned-policy evaluation.
