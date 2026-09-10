# Within-map state composition with exact map quotas — protocol v1

Status: locally declared before main execution, not externally registered.
Equal-map replay balanced map exposure successfully but produced mixed behavior
across three collected banks. This comparison changes selected states within
each map while keeping its exact quota and original replay schedule fixed.

## Question and comparison

Does **within_map_uniform minus collected_unique greedy efficient success**
improve when state identity changes within maps, with per-map support size,
ordered replay map schedule, batch diversity, network and update budget fixed?

Reuse collected bank IDs **1, 2, 3** from
`experiments/bank_replication/pilot_v1`, containing **59,839 / 59,984 / 59,838**
unique current states on maps **300000–300255**. Baseline `collected_unique`
uses its **nine archived final 30,000-update policies**, learner seeds **0, 1,
2** per bank. Reevaluate those controls on the new panels without retraining,
resuming or selecting checkpoints. Treatment `within_map_uniform` trains nine
new policies from the same initial online and target identities.

These are previously drawn banks sharing training layouts, learner seeds and
new evaluation panels. Report every bank before the equal-bank mean. They are
not new collector or world-family replications; crossed panel/learner cells do
not increase the number of independent bank draws. No significance,
equivalence or population-wide sampler ranking is declared.

## Fixed task and learning

Unchanged fully visible World A: 5×5, four walls, one pellet, horizon 32,
ordinary action mapping visible initially, 92 observations, gamma 0.97 and
existing rewards/potential shaping. Reuse the exact **163,840 archived training
states**, original row order, metadata and **655,360 four-action transitions**.
Validate these against completed coverage and bank-replication manifests.

Preserve the **20,420-parameter** 92 → 128 → 64 → 4 ReLU network and unchanged
`fixed_targets.fixed_update(..., "double_dqn")`: Adam 0.001, batch 64,
all-four-action SmoothL1 beta 1 averaged over states/actions, gradient clipping
at 5, detached Double DQN successor targets, terminal reward targets and soft
target tau 0.01. Each fit has **30,000 updates**, with no memory, architecture,
loss or online-feedback change. Initial online/target hashes must match the
archived same-seed controls.

All four transition targets and full-bank detached successor observations
remain available, including observations outside direct-training support.
Exact Q* is an integrity/rollout diagnostic; labels, distance, outcomes and
learner predictions must not select replacement states.

## Select and freeze replacement supports

Before any optimizer update or learned-policy prediction, process bank IDs and
ascending map seeds. For each map, define **K_map** as its original collected
unique-state count; define candidate rows as that map's exhaustive valid
current-state rows in ascending global order. Current states include position
and remaining clock, not merely positions.

Create one owned generator with **SeedSequence([bank_id, map_seed, 105301])**.
Call `choice(number_of_candidates, size=K_map, replace=False)` exactly once;
map returned indices to global rows and sort ascending. Concatenate ascending
map blocks and save sorted int32 global IDs as `within_map_uniform`. One draw
per bank/map, shared across learner seeds; no balancing within maps, redraw,
rejection by labels or outcome-based selection. Preserve natural overlaps with
the original support. For a zero quota, retain an empty block without adding
states; the shipped banks have positive quotas on all 256 maps.

This gives **768 per-map subset draws**, forming **three new treatment supports**,
with **zero new collection episodes or steps**. Preserve exact total and map
quotas, candidate identities, RNG declaration, selected IDs, hashes and
within-pair overlap/Jaccard. Freeze all six baseline/treatment support arrays
before any fit. The two arms need not have the same clock, spatial, winnable,
goal-near or successor-graph composition within a map; these change jointly.

Check that exhaustive row order groups maps contiguously and that sorted
support blocks have identical lengths and ordered map IDs in both arms. Stop
as a configuration failure if a quota exceeds its candidate count or support
cannot fit the distinct 64-state batch. Never silently sample with replacement.

## Preserve the original replay schedule

Treatment uses the unchanged `SupportSampler` / `BatchSampler` procedure:
**SeedSequence([learner_seed,66301])**, 64 distinct local support positions per
update, repeats allowed between updates. Preserve returned batch order.
Because each map block keeps its exact quota, identical local draws imply
identical ordered map IDs and within-batch map diversity at every update.
Global current states intentionally differ.

Reconstruct the original control stream from its collected support and RNG,
without training or collecting. On the main run, verify its **full 30,000-update
local/global digests and count vectors** against the archived record. Verify
treatment local digests/counts and its ordered map-ID digest/counts against
that reconstructed control, including the per-update order. No cross-bank map
schedule equality is expected because different banks have different quotas.

Save treatment ordered local/global/map digests and all count vectors. Assert
updates × 64 total presentations and zero direct off-support samples. Record
matching per-map exposure alongside changed clock/category/state and
successor-query exposure. Historical collection costs are inherited provenance,
not this run's cost. Distinguish unique support membership from actual training
presentation frequency and query-weighted successor statistics.

Smoke treatment has only 24 updates. Match its local/map stream to the
reconstructed **first 24 updates** of the control schedule; retain and clearly
label the control's actual 30,000-update weights and exposure. Never compare
its prefix digest with the full archived digest or claim equal smoke budgets.

## Training, snapshots and work order

Verify all supports, archive inputs, control models and initial identities;
select every evaluation panel before training or predictions. Execute
sequentially by bank ID then learner seed: **nine new fits / 270,000 updates /
17,280,000 sampled-state presentations / 69,120,000 action targets**. Baselines
consumed the same update budget historically and receive zero new updates.

Save treatment online/target snapshots at **0, 1,000, 3,000, 10,000 and 30,000**:
**45 treatment snapshots**, plus copies of **nine archived baseline finals**.
These are inference snapshots, not optimizer-resume checkpoints. Log loss
windows every 100 updates: **2,700 finite raw windows**. Do not manufacture new
control losses or select intermediate checkpoints based on outcomes.

All treatment fits finish before any learned-policy evaluation. Evaluate only
final treatment and final archived control policies, with no optimizer/replay
or gradients. Preserve source/copy file and online/target identities before,
after and around each model-panel evaluation. Nonfinite values, changed
identities, missing work or deviations disqualify evidence and preserve partial
artifacts with the reason. Do not tune or run a second main comparison.

## Prospective evaluation and outcomes

Select **eight new disjoint 64-map panels**, scanning upward in panel order
from **1060000 + 1000 × panel index**. Exclude training, all four earlier fresh
panels (supervised, fixed-target, coverage, equal-size), all **512 frozen-panel
layouts**, all **512 bank-replication layouts**, all **512 map-replay layouts**,
and every already accepted new layout. Identity is walls + pellet + rules,
ignoring seed/spawn/clock. Preserve original spawns and every rejection;
no difficulty balancing or outcome-based replacement.

Each model/map receives one greedy and two epsilon-0.1 episodes, using paired
external draws **[learner_seed,map_seed,repetition,55219]** across arms/banks.
Greedy ties choose the lowest label. **27,648 learner episodes** comprise 9,216
greedy and 18,432 epsilon episodes. Shared references run once: three random
seeds × two repetitions × 512 maps = **3,072**, plus **512 planner episodes**.
Total **31,232 evaluation episodes**. Retain rollout Q* diagnostics; no new
full-state regression sweep or fitting gate.

Primary: **within_map_uniform minus collected_unique greedy efficient-success
rate**, per bank across 512 maps and three learners, then equal-weight mean,
range and signs across the three effects. Efficient success means success
within twice the map's planner length; **all episodes including failures**
enter the denominator. Each bank/condition has 1,536 greedy and 3,072 epsilon
episodes. Equal episode counts make success/efficient-success/mean-step
equal-bank means numerically equal to complete episode pooling.

Keep success and efficiency separate. Save **4,608 greedy layout pairs**,
**72 per-panel/seed plus nine all-panel/seed pairs**, and **24 per-panel plus
three all-panel aggregate effects**. Retain complete keys and additive counts.
Do not invent epsilon paired rows. Report pooled blocked-step/total-step ratios
separately from means of within-bank pooled and learner ratios. Successful-only
lengths compare different successful subsets. Historical 70% success/above-
panel-random and 80% efficiency crossings remain descriptive; no new competence
gate, old-gate rewrite or significance claim. Eligibility means integrity and
completeness, not a performance pass.

A repeatable gain would support within-map state composition as a useful
collection target under fixed map exposure. Mixed or negative effects leave
interactions unresolved; they do not prove a particular clock/goal mechanism
or justify tuning within this run. Previous globally uniform and map-balanced
policies are not new arms on these panels; do not claim matched-panel gap
closure from their older scores. After this bounded control, prioritize the
bridge to learning from actually recorded actions/transitions over an
open-ended series of balancing variants. Memory remains deferred; privileged
offline support/action access does not establish online-RL competence or
interpretable A → B → A forgetting.

## Dashboard, resources and reproducibility

Add **Within-map states** to the existing dashboard. Show exact per-map quotas,
paired replay checks, changed support membership and clock/category exposure,
per-bank effects, final panels and clearly sourced controls. Preserve historical
studies and active rules at frame zero. Preselect the first accepted map per
panel, repetition zero, every bank × arm × learner in both modes: **288 learner
recordings**, plus random seed 0 and planner per panel = **304 recordings**.
No favorable trace selection or intermediate-evaluation selector.

Use Python **3.12**, Torch **2.8.0**, NumPy **2.0.2**, deterministic CPU, one
Torch thread, sequential execution at `nice -n 10`. Admission cap **1,200
seconds** includes preparation, training, evaluation and aggregation; sampled
peak-process-RSS cap **4 GiB**. This is not a hard OS limit or total-machine
memory. Preserve partial work and stop reasons when a guard fires.

Review and push this protocol before freezing clean implementation source and
executing one main run. Capture source/protocol/git/environment/command, input
manifests/identities, data/transitions, original and replacement supports,
quota/draw records, reconstructed control sampling, actual exposure/counts/
digests, archived controls, snapshots/losses, selected panels/skips, raw episodes/
pairs, summaries/replays/resources and SHA-256 manifest. Output
`experiments/within_map/pilot_v1`, export `dashboard/data/within_map.json`.
Keep license undecided.

Smoke uses the same shipped training bank and full three collected supports,
with the same quota-draw rule and original sampler. Train seed 0 only for
**24 updates per bank = 72 new updates**, saving six treatment snapshots
(0/final) and three archived 30,000-update finals. Evaluate two two-map panels
starting **1140000 / 1141000**: **72 learner + 12 shared reference episodes**,
**28 recordings**. Smoke's unequal training budgets check execution only and
are ineligible research evidence. Tests may use explicit smaller fixtures.
Audit saved evidence and preselected recording observations without retraining
or repeating the complete learned-policy evaluation.
